

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

# ============================================================
# Projection Head
# ============================================================

class ProjectionHead(nn.Module):
    """
    MLP Projection Head

    F -> z

    Used for contrastive learning.
    """

    def __init__(
            self,
            in_channels=Config.FEATURE_DIM,
            hidden_dim=256,
            embedding_dim=128):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projector = nn.Sequential(

            nn.Linear(
                in_channels,
                hidden_dim
            ),

            nn.LayerNorm(hidden_dim),

            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
                embedding_dim
            )

        )

    ###########################################################

    def forward(self, feature):

        x = self.pool(feature)

        x = x.flatten(1)

        z = self.projector(x)

        z = F.normalize(

            z,

            p=2,

            dim=1

        )

        return z


# ============================================================
# Multi-Level Projection
# ============================================================

class MultiScaleProjectionHead(nn.Module):

    """
    Projection for

    P2

    P3

    P4

    P5
    """

    def __init__(
            self,
            channels=Config.FEATURE_DIM,
            embedding_dim=Config.PROJECTION_DIM):

        super().__init__()

        self.projectors = nn.ModuleDict({

            "p2": ProjectionHead(

                channels,

                embedding_dim=embedding_dim

            ),

            "p3": ProjectionHead(

                channels,

                embedding_dim=embedding_dim

            ),

            "p4": ProjectionHead(

                channels,

                embedding_dim=embedding_dim

            ),

            "p5": ProjectionHead(

                channels,

                embedding_dim=embedding_dim

            )

        })

    ###########################################################

    def forward(self, features):

        embeddings = {}
        required_levels = ["p2", "p3", "p4", "p5"]
        for level in required_levels:
            if level not in features:
                raise KeyError(f"{level} missing in feature  dictionary")

        for level in ["p2", "p3", "p4", "p5"]:
            
            embeddings[level] = self.projectors[level](

                features[level]

            )

        return embeddings


# ============================================================
# Shared Projection Module
# ============================================================

class SharedProjection(nn.Module):
    """
    Produce embeddings for

    RGB

    Thermal
    """

    def __init__(
            self,
            channels=256,
            embedding_dim=128):

        super().__init__()

        self.rgb_projection = MultiScaleProjectionHead(

            channels,

            embedding_dim

        )

        self.thermal_projection = MultiScaleProjectionHead(

            channels,

            embedding_dim

        )

    ###########################################################

    def forward(
            self,
            rgb_features,
            thermal_features):

        rgb_embeddings = self.rgb_projection(

            rgb_features

        )

        thermal_embeddings = self.thermal_projection(

            thermal_features

        )
        return (
            rgb_embeddings, 
            thermal_embeddings
        )
        #return rgb_embeddings


# ============================================================
# Factory
# ============================================================

def build_projection_head():

    return SharedProjection()


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    model = build_projection_head().to(device)

    rgb = {

        "p2": torch.randn(2,Config.FEATURE_DIM,200,200).to(device),

        "p3": torch.randn(2,Config.FEATURE_DIM,100,100).to(device),

        "p4": torch.randn(2,Config.FEATURE_DIM,50,50).to(device),

        "p5": torch.randn(2,Config.FEATURE_DIM,25,25).to(device)

    }

    thermal = {

        "p2": torch.randn(2,Config.FEATURE_DIM,200,200).to(device),

        "p3": torch.randn(2,Config.FEATURE_DIM,100,100).to(device),

        "p4": torch.randn(2,Config.FEATURE_DIM,50,50).to(device),

        "p5": torch.randn(2,Config.FEATURE_DIM,25,25).to(device)

    }

    rgb_embed, thermal_embed = model(

        rgb,

        thermal

    )
    print("RGB Keys:", rgb_embed.keys())
    print("Thermal Keys: ", thermal_embed.keys())
    print("\n")

    print("=" * 60)

    print("Projection Head")

    print("=" * 60)

    for level in rgb_embed:

        print(

            level,

            rgb_embed[level].shape,

            thermal_embed[level].shape

        )