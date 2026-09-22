import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# ============================================================
# Contrastive Loss (InfoNCE)
# ============================================================

class ContrastiveLoss(nn.Module):
    """
    InfoNCE Loss

    Positive Pair:
        RGB <-> Thermal

    Negative Pair:
        Other samples within the mini-batch
    """

    def __init__(
        self,
        temperature=Config.TEMPERATURE
    ):

        super().__init__()

        self.temperature = temperature

    # --------------------------------------------------------

    def forward(
        self,
        z_rgb,
        z_thermal
    ):

        """
        Inputs

            z_rgb      : [B, D]

            z_thermal  : [B, D]

        Returns

            Scalar Loss
        """

        if z_rgb.shape != z_thermal.shape:

            raise ValueError(
                f"Shape mismatch:\n"
                f"RGB: {z_rgb.shape}\n"
                f"Thermal: {z_thermal.shape}"
            )

        if z_rgb.ndim != 2:

            raise ValueError(
                "Embeddings must have shape [Batch, Embedding]"
            )

        batch_size = z_rgb.size(0)

        # -----------------------------------------
        # L2 Normalize
        # -----------------------------------------

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

        # -----------------------------------------
        # Similarity Matrix
        # -----------------------------------------

        similarity = torch.matmul(
            z_rgb,
            z_thermal.t()
        )

        similarity = similarity / self.temperature

        # -----------------------------------------
        # Positive Labels
        # -----------------------------------------

        labels = torch.arange(
            batch_size,
            device=z_rgb.device,
            dtype=torch.long
        )

        # -----------------------------------------
        # RGB -> Thermal
        # -----------------------------------------

        loss_rgb = F.cross_entropy(
            similarity,
            labels
        )

        # -----------------------------------------
        # Thermal -> RGB
        # -----------------------------------------

        loss_thermal = F.cross_entropy(
            similarity.t(),
            labels
        )

        # -----------------------------------------
        # Final Loss
        # -----------------------------------------

        loss = (
            loss_rgb +
            loss_thermal
        ) / 2.0

        return loss


# ============================================================
# Multi-Level Contrastive Loss
# ============================================================

class MultiScaleContrastiveLoss(nn.Module):
    """
    Computes contrastive loss for

        P2

        P3

        P4

        P5
    """

    def __init__(
        self,
        temperature=Config.TEMPERATURE
    ):

        super().__init__()

        self.loss_fn = ContrastiveLoss(
            temperature
        )

    # --------------------------------------------------------

    def forward(
        self,
        rgb_embeddings,
        thermal_embeddings
    ):

        total_loss = 0.0

        levels = ["p2", "p3", "p4", "p5"]

        for level in levels:

            if level not in rgb_embeddings:

                raise KeyError(
                    f"{level} missing in RGB embeddings."
                )

            if level not in thermal_embeddings:

                raise KeyError(
                    f"{level} missing in Thermal embeddings."
                )

            total_loss += self.loss_fn(

                rgb_embeddings[level],

                thermal_embeddings[level]

            )

        total_loss /= len(levels)

        return total_loss


# ============================================================
# Factory
# ============================================================

def build_contrastive_loss():

    return MultiScaleContrastiveLoss()


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    criterion = build_contrastive_loss().to(device)

    rgb_embeddings = {

        "p2": torch.randn(
            8,
            Config.EMBEDDING_DIM,
            device=device
        ),

        "p3": torch.randn(
            8,
            Config.EMBEDDING_DIM,
            device=device
        ),

        "p4": torch.randn(
            8,
            Config.EMBEDDING_DIM,
            device=device
        ),

        "p5": torch.randn(
            8,
            Config.EMBEDDING_DIM,
            device=device
        )

    }

    thermal_embeddings = {

        "p2": torch.randn(
            8,
            Config.EMBEDDING_DIM,
            device=device
        ),

        "p3": torch.randn(
            8,
            Config.EMBEDDING_DIM,
            device=device
        ),

        "p4": torch.randn(
            8,
            Config.EMBEDDING_DIM,
            device=device
        ),

        "p5": torch.randn(
            8,
            Config.EMBEDDING_DIM,
            device=device
        )

    }

    loss = criterion(

        rgb_embeddings,

        thermal_embeddings

    )

    print("=" * 60)
    print("Contrastive Loss Test")
    print("=" * 60)

    print("RGB Levels:")
    print(rgb_embeddings.keys())

    print()

    print("Thermal Levels:")
    print(thermal_embeddings.keys())

    print()

    print("Contrastive Loss :", loss.item())

    print("=" * 60)
    print("Contrastive Loss Test Passed Successfully.")
    print("=" * 60)