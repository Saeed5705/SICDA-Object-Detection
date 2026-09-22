import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

# ============================================================
# Feature Consistency Loss
# ============================================================

class ConsistencyLoss(nn.Module):
    """
    Mean Squared Error consistency loss
    between RGB and Thermal embeddings.
    """

    def __init__(self):

        super().__init__()

    def forward(self, z_rgb, z_thermal):

        if z_rgb.shape != z_thermal.shape:

            raise ValueError(
                f"Shape mismatch:\n"
                f"RGB: {z_rgb.shape}\n"
                f"Thermal: {z_thermal.shape}"
            )

        loss = F.mse_loss(

            z_rgb,

            z_thermal,

            reduction="mean"

        )

        return loss


# ============================================================
# Multi-Level Consistency Loss
# ============================================================

class MultiScaleConsistencyLoss(nn.Module):
    """
    Compute consistency loss for

        P2
        P3
        P4
        P5
    """

    def __init__(self):

        super().__init__()

        self.loss_fn = ConsistencyLoss()

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

def build_consistency_loss():

    return MultiScaleConsistencyLoss()


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    criterion = build_consistency_loss().to(device)

    rgb_embeddings = {

        "p2": torch.randn(8,128).to(device),

        "p3": torch.randn(8,128).to(device),

        "p4": torch.randn(8,128).to(device),

        "p5": torch.randn(8,128).to(device)

    }

    thermal_embeddings = {

        "p2": torch.randn(8,128).to(device),

        "p3": torch.randn(8,128).to(device),

        "p4": torch.randn(8,128).to(device),

        "p5": torch.randn(8,128).to(device)

    }

    loss = criterion(

        rgb_embeddings,

        thermal_embeddings

    )

    print("=" * 60)
    print("Consistency Loss")
    print("=" * 60)

    print("RGB Keys:", rgb_embeddings.keys())
    print("Thermal Keys:", thermal_embeddings.keys())

    print()

    print("Loss :", loss.item())

    print("=" * 60)
    print("Consistency Loss Test Passed Successfully.")
    print("=" * 60)