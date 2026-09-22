import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# ============================================================
# Configuration Helpers
# ============================================================

def get_temperature():
    """
    Use SSL_TEMPERATURE from Config.

    Fallback:
        TEMPERATURE
        0.07
    """

    if hasattr(Config, "SSL_TEMPERATURE"):
        return float(Config.SSL_TEMPERATURE)

    if hasattr(Config, "TEMPERATURE"):
        return float(Config.TEMPERATURE)

    return 0.07


def get_embedding_dim():
    """
    Get embedding dimension from Config.

    Priority:
        1. EMBEDDING_DIM
        2. FEATURE_DIM
        3. 256
    """

    if hasattr(Config, "EMBEDDING_DIM"):
        return int(Config.EMBEDDING_DIM)

    if hasattr(Config, "FEATURE_DIM"):
        return int(Config.FEATURE_DIM)

    return 256


# ============================================================
# Contrastive Loss
# ============================================================

class ContrastiveLoss(nn.Module):
    """
    Symmetric InfoNCE Contrastive Loss.

    Positive pair:
        RGB <-> Thermal

    Negative pairs:
        Other samples within the mini-batch.

    Input:
        z_rgb      : [B, D]
        z_thermal  : [B, D]

    Output:
        Scalar loss
    """

    def __init__(
        self,
        temperature=None
    ):

        super().__init__()

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        if temperature is None:
            temperature = get_temperature()

        temperature = float(temperature)

        if temperature <= 0:
            raise ValueError(
                f"Temperature must be > 0. "
                f"Got {temperature}"
            )

        self.temperature = temperature

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        z_rgb,
        z_thermal
    ):

        # ----------------------------------------------------
        # Type validation
        # ----------------------------------------------------

        if not isinstance(z_rgb, torch.Tensor):

            raise TypeError(
                "z_rgb must be torch.Tensor."
            )

        if not isinstance(z_thermal, torch.Tensor):

            raise TypeError(
                "z_thermal must be torch.Tensor."
            )

        # ----------------------------------------------------
        # Shape validation
        # ----------------------------------------------------

        if z_rgb.shape != z_thermal.shape:

            raise ValueError(
                "RGB and Thermal embedding shapes "
                "must match.\n"
                f"RGB     : {z_rgb.shape}\n"
                f"Thermal : {z_thermal.shape}"
            )

        if z_rgb.ndim != 2:

            raise ValueError(
                "Embeddings must have shape [B, D]. "
                f"Got {z_rgb.shape}"
            )

        batch_size = z_rgb.size(0)

        if batch_size == 0:

            raise ValueError(
                "Batch size cannot be zero."
            )

        # ----------------------------------------------------
        # L2 normalization
        # ----------------------------------------------------

        z_rgb = F.normalize(
            z_rgb,
            p=2,
            dim=1
        )

        z_thermal = F.normalize(
            z_thermal,
            p=2,
            dim=1
        )

        # ====================================================
        # Batch Size = 1
        # ====================================================

        if batch_size == 1:

            cosine_similarity = F.cosine_similarity(
                z_rgb,
                z_thermal,
                dim=1
            )

            loss = (
                1.0 - cosine_similarity
            ).mean()

            return loss

        # ====================================================
        # Similarity Matrix
        # ====================================================

        similarity = torch.matmul(
            z_rgb,
            z_thermal.transpose(0, 1)
        )

        similarity = (
            similarity /
            self.temperature
        )

        # ----------------------------------------------------
        # Positive pair labels
        # ----------------------------------------------------

        labels = torch.arange(
            batch_size,
            device=z_rgb.device,
            dtype=torch.long
        )

        # ====================================================
        # RGB -> Thermal
        # ====================================================

        loss_rgb = F.cross_entropy(
            similarity,
            labels
        )

        # ====================================================
        # Thermal -> RGB
        # ====================================================

        loss_thermal = F.cross_entropy(
            similarity.transpose(0, 1),
            labels
        )

        # ====================================================
        # Symmetric InfoNCE
        # ====================================================

        loss = (
            loss_rgb +
            loss_thermal
        ) * 0.5

        return loss


# ============================================================
# Multi-Scale Contrastive Loss
# ============================================================

class MultiScaleContrastiveLoss(nn.Module):
    """
    Multi-scale contrastive loss.

    Computes RGB-Thermal contrastive alignment
    independently at:

        P2
        P3
        P4
        P5
    """

    LEVELS = (
        "p2",
        "p3",
        "p4",
        "p5"
    )

    def __init__(
        self,
        temperature=None
    ):

        super().__init__()

        self.loss_fn = ContrastiveLoss(
            temperature=temperature
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        rgb_embeddings,
        thermal_embeddings
    ):

        # ----------------------------------------------------
        # Dictionary validation
        # ----------------------------------------------------

        if not isinstance(
            rgb_embeddings,
            dict
        ):

            raise TypeError(
                "rgb_embeddings must be a dictionary."
            )

        if not isinstance(
            thermal_embeddings,
            dict
        ):

            raise TypeError(
                "thermal_embeddings must be a dictionary."
            )

        losses = []

        # ====================================================
        # P2-P5
        # ====================================================

        for level in self.LEVELS:

            if level not in rgb_embeddings:

                raise KeyError(
                    f"{level} missing in RGB embeddings."
                )

            if level not in thermal_embeddings:

                raise KeyError(
                    f"{level} missing in Thermal embeddings."
                )

            rgb_embedding = (
                rgb_embeddings[level]
            )

            thermal_embedding = (
                thermal_embeddings[level]
            )

            # ------------------------------------------------
            # Calculate level loss
            # ------------------------------------------------

            level_loss = self.loss_fn(
                rgb_embedding,
                thermal_embedding
            )

            losses.append(level_loss)

        # ====================================================
        # Multi-scale average
        # ====================================================

        return torch.stack(
            losses
        ).mean()


# ============================================================
# Factory
# ============================================================

def build_contrastive_loss(
    temperature=None
):

    return MultiScaleContrastiveLoss(
        temperature=temperature
    )


# ============================================================
# Direct Test
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("CONTRASTIVE LOSS TEST")
    print("=" * 70)

    print(
        f"Device      : {device}"
    )

    # ========================================================
    # Configuration
    # ========================================================

    temperature = get_temperature()

    embedding_dim = get_embedding_dim()

    batch_size = 4

    print(
        f"Temperature : {temperature}"
    )

    print(
        f"Embedding D : {embedding_dim}"
    )

    print(
        f"Batch Size  : {batch_size}"
    )

    # ========================================================
    # Build criterion
    # ========================================================

    criterion = build_contrastive_loss(
        temperature=temperature
    ).to(device)

    # ========================================================
    # Create embeddings
    # ========================================================

    rgb_embeddings = {

        level:
            torch.randn(
                batch_size,
                embedding_dim,
                device=device
            )

        for level in
        MultiScaleContrastiveLoss.LEVELS

    }

    thermal_embeddings = {

        level:
            torch.randn(
                batch_size,
                embedding_dim,
                device=device
            )

        for level in
        MultiScaleContrastiveLoss.LEVELS

    }

    # ========================================================
    # Calculate loss
    # ========================================================

    loss = criterion(
        rgb_embeddings,
        thermal_embeddings
    )

    # ========================================================
    # Output
    # ========================================================

    print("-" * 70)

    print(
        "RGB Levels     :",
        list(rgb_embeddings.keys())
    )

    print(
        "Thermal Levels :",
        list(thermal_embeddings.keys())
    )

    print(
        "P2 Shape       :",
        rgb_embeddings["p2"].shape
    )

    print(
        "P3 Shape       :",
        rgb_embeddings["p3"].shape
    )

    print(
        "P4 Shape       :",
        rgb_embeddings["p4"].shape
    )

    print(
        "P5 Shape       :",
        rgb_embeddings["p5"].shape
    )

    print("-" * 70)

    print(
        f"Contrastive Loss : {loss.item():.6f}"
    )

    print("=" * 70)
    print(
        "Contrastive Loss Test Passed Successfully."
    )
    print("=" * 70)