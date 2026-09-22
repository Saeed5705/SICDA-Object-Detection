"""
==============================================================
File : cross_model.py

SICDA Cross Modal Attention Module

Memory-Efficient Spatial Reduction Cross Attention
for RGB-Thermal Feature Alignment

Input:
    RGB FPN Features
    Thermal FPN Features

Output:
    Updated RGB Features
    Updated Thermal Features

Levels:
    p2, p3, p4, p5

Designed for:
    NVIDIA GPU with 8 GB VRAM

Author:
    Muhammad Saeed
==============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# ============================================================
# Feed Forward Network
# ============================================================

class FeedForward(nn.Module):

    def __init__(
        self,
        dim=256,
        expansion=2,
        dropout=0.1
    ):

        super().__init__()

        hidden = dim * expansion

        self.net = nn.Sequential(

            nn.Linear(
                dim,
                hidden
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden,
                dim
            ),

            nn.Dropout(dropout)
        )

    def forward(self, x):

        return self.net(x)


# ============================================================
# Efficient Cross Attention Block
# ============================================================

class CrossAttentionBlock(nn.Module):

    def __init__(
        self,
        dim=256,
        heads=4,
        reduction=8,
        max_tokens=256,
        dropout=0.1
    ):

        super().__init__()

        self.dim = dim
        self.reduction = reduction
        self.max_tokens = max_tokens

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.rgb_attention = nn.MultiheadAttention(

            embed_dim=dim,

            num_heads=heads,

            dropout=dropout,

            batch_first=True
        )

        self.thermal_attention = nn.MultiheadAttention(

            embed_dim=dim,

            num_heads=heads,

            dropout=dropout,

            batch_first=True
        )

        # ----------------------------------------------------
        # Normalization
        # ----------------------------------------------------

        self.rgb_norm1 = nn.LayerNorm(dim)

        self.rgb_norm2 = nn.LayerNorm(dim)

        self.thermal_norm1 = nn.LayerNorm(dim)

        self.thermal_norm2 = nn.LayerNorm(dim)

        # ----------------------------------------------------
        # FFN
        # ----------------------------------------------------

        self.rgb_ffn = FeedForward(

            dim=dim,

            expansion=2,

            dropout=dropout
        )

        self.thermal_ffn = FeedForward(

            dim=dim,

            expansion=2,

            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    # ========================================================
    # Calculate Reduction
    # ========================================================

    def _get_reduction(
        self,
        H,
        W
    ):

        reduction = self.reduction

        tokens = (

            (H // reduction)
            *
            (W // reduction)

        )

        while tokens > self.max_tokens:

            reduction *= 2

            tokens = (

                max(1, H // reduction)
                *
                max(1, W // reduction)

            )

        return max(1, reduction)

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        rgb,
        thermal
    ):

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if rgb.dim() != 4:

            raise ValueError(

                f"RGB feature must be 4D [B,C,H,W], "
                f"got {rgb.shape}"

            )

        if thermal.dim() != 4:

            raise ValueError(

                f"Thermal feature must be 4D [B,C,H,W], "
                f"got {thermal.shape}"

            )

        B, C, H, W = rgb.shape

        # ----------------------------------------------------
        # Channel check
        # ----------------------------------------------------

        if C != self.dim:

            raise ValueError(

                f"Expected {self.dim} channels, "
                f"got {C}"

            )

        # ----------------------------------------------------
        # Spatial alignment
        # ----------------------------------------------------

        if thermal.shape[-2:] != (H, W):

            thermal = F.interpolate(

                thermal,

                size=(H, W),

                mode="bilinear",

                align_corners=False
            )

        orig_H = H
        orig_W = W

        # ----------------------------------------------------
        # Adaptive reduction
        # ----------------------------------------------------

        reduction = self._get_reduction(

            H,
            W
        )

        # ----------------------------------------------------
        # Spatial reduction
        # ----------------------------------------------------

        if reduction > 1:

            rgb_small = F.avg_pool2d(

                rgb,

                kernel_size=reduction,

                stride=reduction
            )

            thermal_small = F.avg_pool2d(

                thermal,

                kernel_size=reduction,

                stride=reduction
            )

        else:

            rgb_small = rgb

            thermal_small = thermal

        # ----------------------------------------------------
        # Make sure spatial dimensions are valid
        # ----------------------------------------------------

        h = rgb_small.shape[2]

        w = rgb_small.shape[3]

        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        rgb_tokens = (

            rgb_small

            .flatten(2)

            .transpose(1, 2)

            .contiguous()

        )

        thermal_tokens = (

            thermal_small

            .flatten(2)

            .transpose(1, 2)

            .contiguous()

        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if rgb_tokens.shape[1] > self.max_tokens:

            raise RuntimeError(

                f"RGB attention tokens={rgb_tokens.shape[1]} "
                f"> max_tokens={self.max_tokens}"

            )

        if thermal_tokens.shape[1] > self.max_tokens:

            raise RuntimeError(

                f"Thermal attention tokens="
                f"{thermal_tokens.shape[1]} "
                f"> max_tokens={self.max_tokens}"

            )

        # ====================================================
        # RGB attends Thermal
        # ====================================================

        rgb_attn, _ = self.rgb_attention(

            query=rgb_tokens,

            key=thermal_tokens,

            value=thermal_tokens,

            need_weights=False
        )

        rgb_tokens = self.rgb_norm1(

            rgb_tokens
            +
            self.dropout(rgb_attn)

        )

        rgb_tokens = self.rgb_norm2(

            rgb_tokens
            +
            self.rgb_ffn(rgb_tokens)

        )

        # ====================================================
        # Thermal attends RGB
        # ====================================================

        thermal_attn, _ = self.thermal_attention(

            query=thermal_tokens,

            key=rgb_tokens,

            value=rgb_tokens,

            need_weights=False
        )

        thermal_tokens = self.thermal_norm1(

            thermal_tokens
            +
            self.dropout(thermal_attn)

        )

        thermal_tokens = self.thermal_norm2(

            thermal_tokens
            +
            self.thermal_ffn(thermal_tokens)

        )

        # ====================================================
        # Restore feature maps
        # ====================================================

        rgb_out = (

            rgb_tokens

            .transpose(1, 2)

            .reshape(

                B,
                C,
                h,
                w
            )

        )

        thermal_out = (

            thermal_tokens

            .transpose(1, 2)

            .reshape(

                B,
                C,
                h,
                w
            )

        )

        # ====================================================
        # Restore original spatial resolution
        # ====================================================

        if (h, w) != (orig_H, orig_W):

            rgb_out = F.interpolate(

                rgb_out,

                size=(orig_H, orig_W),

                mode="bilinear",

                align_corners=False
            )

            thermal_out = F.interpolate(

                thermal_out,

                size=(orig_H, orig_W),

                mode="bilinear",

                align_corners=False
            )

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        rgb_out = rgb_out + rgb

        thermal_out = thermal_out + thermal

        return rgb_out, thermal_out


# ============================================================
# Multi Scale Cross Attention
# ============================================================

class MultiScaleCrossAttention(nn.Module):

    def __init__(
        self,
        channels=256,
        heads=4,
        max_tokens=256
    ):

        super().__init__()

        self.levels = [

            "p2",
            "p3",
            "p4",
            "p5"

        ]

        # ----------------------------------------------------
        # Different reduction for each FPN level
        # ----------------------------------------------------

        reductions = {

            "p2": 16,

            "p3": 8,

            "p4": 4,

            "p5": 2

        }

        self.blocks = nn.ModuleDict({

            level:

            CrossAttentionBlock(

                dim=channels,

                heads=heads,

                reduction=reductions[level],

                max_tokens=max_tokens,

                dropout=0.1

            )

            for level in self.levels

        })

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        rgb_features,
        thermal_features
    ):

        rgb_out = {}

        thermal_out = {}

        for level in self.levels:

            if level not in rgb_features:

                raise KeyError(

                    f"Missing RGB feature level: {level}"

                )

            if level not in thermal_features:

                raise KeyError(

                    f"Missing Thermal feature level: {level}"

                )

            rgb = rgb_features[level]

            thermal = thermal_features[level]

            # ------------------------------------------------
            # Ensure tensor input
            # ------------------------------------------------

            if not torch.is_tensor(rgb):

                raise TypeError(

                    f"RGB {level} is not Tensor: "
                    f"{type(rgb)}"

                )

            if not torch.is_tensor(thermal):

                raise TypeError(

                    f"Thermal {level} is not Tensor: "
                    f"{type(thermal)}"

                )

            # ------------------------------------------------
            # Cross attention
            # ------------------------------------------------

            (

                rgb_out[level],

                thermal_out[level]

            ) = self.blocks[level](

                rgb,

                thermal

            )

        return rgb_out, thermal_out


# ============================================================
# Factory
# ============================================================

def build_cross_modal_attention():

    heads = (

        getattr(

            Config,

            "ATTENTION_HEADS",

            4

        )

    )

    return MultiScaleCrossAttention(

        channels=256,

        heads=heads,

        max_tokens=256

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

    print("SICDA Cross Modal Attention Test")

    print("=" * 70)

    print(

        "Device:",

        device

    )

    model = build_cross_modal_attention().to(device)

    model.eval()

    rgb = {

        "p2": torch.randn(

            1,
            256,
            128,
            128,

            device=device
        ),

        "p3": torch.randn(

            1,
            256,
            64,
            64,

            device=device
        ),

        "p4": torch.randn(

            1,
            256,
            32,
            32,

            device=device
        ),

        "p5": torch.randn(

            1,
            256,
            16,
            16,

            device=device
        )

    }

    thermal = {

        "p2": torch.randn(

            1,
            256,
            100,
            128,

            device=device
        ),

        "p3": torch.randn(

            1,
            256,
            60,
            64,

            device=device
        ),

        "p4": torch.randn(

            1,
            256,
            32,
            32,

            device=device
        ),

        "p5": torch.randn(

            1,
            256,
            16,
            16,

            device=device
        )

    }

    with torch.no_grad():

        rgb_out, thermal_out = model(

            rgb,

            thermal

        )

    print()

    for level in rgb_out:

        print(

            f"{level}:",

            "RGB",

            tuple(rgb_out[level].shape),

            "| Thermal",

            tuple(thermal_out[level].shape)

        )

    print()

    print("=" * 70)

    print("Cross Modal Attention Test PASSED")

    print("=" * 70)