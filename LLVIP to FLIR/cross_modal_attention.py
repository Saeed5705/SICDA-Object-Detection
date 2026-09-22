import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# ============================================================
# Window Partition
# ============================================================

def window_partition(x, window_size):
    """
    Input:
        x : (B, C, H, W)

    Output:
        windows :
        (num_windows * B,
         window_size,
         window_size,
         C)
    """

    B, C, H, W = x.shape

    x = x.view(
        B,
        C,
        H // window_size,
        window_size,
        W // window_size,
        window_size
    )

    x = x.permute(
        0,
        2,
        4,
        3,
        5,
        1
    ).contiguous()

    windows = x.view(
        -1,
        window_size,
        window_size,
        C
    )

    return windows


# ============================================================
# Window Reverse
# ============================================================

def window_reverse(
        windows,
        window_size,
        H,
        W):

    """
    Reverse Window Partition
    """

    B = int(
        windows.shape[0] //
        ((H // window_size) * (W // window_size))
    )

    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1
    )

    x = x.permute(
        0,
        5,
        1,
        3,
        2,
        4
    ).contiguous()

    x = x.view(
        B,
        -1,
        H,
        W
    )

    return x


# ============================================================
# Feed Forward Network
# ============================================================

class FeedForward(nn.Module):

    def __init__(
            self,
            dim,
            expansion=4,
            dropout=0.1):

        super().__init__()

        hidden_dim = dim * expansion

        self.net = nn.Sequential(

            nn.Linear(dim, hidden_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, dim),

            nn.Dropout(dropout)

        )

    def forward(self, x):

        return self.net(x)
# ============================================================
# Window Cross Attention (Bidirectional)
# ============================================================

class WindowCrossAttention(nn.Module):
    """
    Bidirectional Cross Attention

    RGB  ---> Query
    Thermal ---> Key, Value

    Thermal ---> Query
    RGB  ---> Key, Value

    Returns
    -------
    Updated RGB Feature
    Updated Thermal Feature
    """

    def __init__(
            self,
            dim=Config.FEATURE_DIM,
            num_heads=Config.ATTENTION_HEADS,
            window_size=7,
            dropout=0.1):

        super().__init__()

        self.dim = dim
        self.window_size = window_size

        # RGB -> Thermal
        self.rgb_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Thermal -> RGB
        self.thermal_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.rgb_norm1 = nn.LayerNorm(dim)
        self.rgb_norm2 = nn.LayerNorm(dim)

        self.thermal_norm1 = nn.LayerNorm(dim)
        self.thermal_norm2 = nn.LayerNorm(dim)

        self.rgb_ffn = FeedForward(dim, dropout=dropout)
        self.thermal_ffn = FeedForward(dim, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    # =======================================================
    # Forward
    # =======================================================

    def forward(
            self,
            rgb_feature,
            thermal_feature):

        assert rgb_feature.shape == thermal_feature.shape, \
            "RGB and Thermal features must have identical shape."

        B, C, H, W = rgb_feature.shape

        ws = self.window_size

        # ---------------- Padding ----------------

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_h or pad_w:

            rgb_feature = F.pad(
                rgb_feature,
                (0, pad_w, 0, pad_h)
            )

            thermal_feature = F.pad(
                thermal_feature,
                (0, pad_w, 0, pad_h)
            )

        Hp = H + pad_h
        Wp = W + pad_w

        # ---------------- Partition ----------------

        rgb_windows = window_partition(
            rgb_feature,
            ws
        )

        thermal_windows = window_partition(
            thermal_feature,
            ws
        )

        rgb_windows = rgb_windows.view(
            -1,
            ws * ws,
            C
        )

        thermal_windows = thermal_windows.view(
            -1,
            ws * ws,
            C
        )

        # ==================================================
        # RGB attends to Thermal
        # ==================================================

        rgb_out, _ = self.rgb_attention(

            query=rgb_windows,

            key=thermal_windows,

            value=thermal_windows,

            need_weights=False

        )

        rgb_out = self.rgb_norm1(

            rgb_windows +

            self.dropout(rgb_out)

        )

        rgb_ffn = self.rgb_ffn(rgb_out)

        rgb_out = self.rgb_norm2(

            rgb_out +

            self.dropout(rgb_ffn)

        )

        # ==================================================
        # Thermal attends to RGB
        # ==================================================

        thermal_out, _ = self.thermal_attention(

            query=thermal_windows,

            key=rgb_windows,

            value=rgb_windows,

            need_weights=False

        )

        thermal_out = self.thermal_norm1(

            thermal_windows +

            self.dropout(thermal_out)

        )

        thermal_ffn = self.thermal_ffn(thermal_out)

        thermal_out = self.thermal_norm2(

            thermal_out +

            self.dropout(thermal_ffn)

        )

        # ---------------- Restore windows ----------------

        rgb_out = rgb_out.view(
            -1,
            ws,
            ws,
            C
        )

        thermal_out = thermal_out.view(
            -1,
            ws,
            ws,
            C
        )

        rgb_out = window_reverse(
            rgb_out,
            ws,
            Hp,
            Wp
        )

        thermal_out = window_reverse(
            thermal_out,
            ws,
            Hp,
            Wp
        )

        # ---------------- Remove padding ----------------

        rgb_out = rgb_out[:, :, :H, :W]

        thermal_out = thermal_out[:, :, :H, :W]

        return rgb_out, thermal_out
# ============================================================
# Multi-Scale Cross Modal Attention
# ============================================================

class MultiScaleCrossAttention(nn.Module):
    """
    Apply Bidirectional Cross Attention on

        P2
        P3
        P4
        P5

    Returns

        updated_rgb_features

        updated_thermal_features
    """

    def __init__(
            self,
            channels=Config.FEATURE_DIM,
            num_heads=Config.ATTENTION_HEADS,
            window_size=7):

        super().__init__()

        self.levels = [
            "p2",
            "p3",
            "p4",
            "p5"
        ]

        self.attention_modules = nn.ModuleDict({

            level: WindowCrossAttention(

                dim=channels,

                num_heads=num_heads,

                window_size=window_size

            )

            for level in self.levels

        })

    # ==========================================================
    # Forward
    # ==========================================================

    def forward(
            self,
            rgb_features,
            thermal_features):

        updated_rgb_features = {}

        updated_thermal_features = {}

        for level in self.levels:

            if level not in rgb_features:
                raise KeyError(f"{level} missing from RGB features.")

            if level not in thermal_features:
                raise KeyError(f"{level} missing from Thermal features.")

            rgb_out, thermal_out = self.attention_modules[level](

                rgb_features[level],

                thermal_features[level]

            )

            updated_rgb_features[level] = rgb_out

            updated_thermal_features[level] = thermal_out

        return updated_rgb_features, updated_thermal_features
# ============================================================
# Factory Function
# ============================================================

def build_cross_modal_attention():
    """
    Build SICDA Bidirectional Cross Modal Attention
    """

    return MultiScaleCrossAttention(

        channels=Config.FEATURE_DIM,

        num_heads=Config.ATTENTION_HEADS,

        window_size=7

    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    print("=" * 70)
    print("Testing SICDA Bidirectional Cross Modal Attention")
    print("=" * 70)

    model = build_cross_modal_attention().to(device)

    model.eval()

    rgb_features = {

        "p2": torch.randn(
            2,
            Config.FEATURE_DIM,
            200,
            200,
            device=device
        ),

        "p3": torch.randn(
            2,
            Config.FEATURE_DIM,
            100,
            100,
            device=device
        ),

        "p4": torch.randn(
            2,
            Config.FEATURE_DIM,
            50,
            50,
            device=device
        ),

        "p5": torch.randn(
            2,
            Config.FEATURE_DIM,
            25,
            25,
            device=device
        )

    }

    thermal_features = {

        "p2": torch.randn(
            2,
            Config.FEATURE_DIM,
            200,
            200,
            device=device
        ),

        "p3": torch.randn(
            2,
            Config.FEATURE_DIM,
            100,
            100,
            device=device
        ),

        "p4": torch.randn(
            2,
            Config.FEATURE_DIM,
            50,
            50,
            device=device
        ),

        "p5": torch.randn(
            2,
            Config.FEATURE_DIM,
            25,
            25,
            device=device
        )

    }

    with torch.no_grad():

        updated_rgb, updated_thermal = model(

            rgb_features,

            thermal_features

        )

    print()

    print("=" * 70)
    print("RGB Feature Shapes")
    print("=" * 70)

    for level in updated_rgb:

        print(

            f"{level} : {updated_rgb[level].shape}"

        )

    print()

    print("=" * 70)
    print("Thermal Feature Shapes")
    print("=" * 70)

    for level in updated_thermal:

        print(

            f"{level} : {updated_thermal[level].shape}"

        )

    print()

    print("=" * 70)
    print("Shape Verification")
    print("=" * 70)

    for level in updated_rgb:

        print(

            level,

            updated_rgb[level].shape == updated_thermal[level].shape

        )

    print()

    print("=" * 70)
    print("Cross Modal Attention Test Passed Successfully")
    print("=" * 70)