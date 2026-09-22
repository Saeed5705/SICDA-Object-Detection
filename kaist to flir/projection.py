"""
==============================================================
File : projection.py

SICDA Projection Head

Purpose:
    Convert FPN feature maps into embedding vectors
    for Cross-Modal Contrastive Learning.

Input:
    P2, P3, P4, P5 feature maps

Output:
    RGB embeddings
    Thermal embeddings

Author:
Muhammad Saeed
==============================================================
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config



# ============================================================
# Single Scale Projection Head
# ============================================================


class ProjectionHead(nn.Module):
    """
    Projection:

        Feature Map
            |
        Global Pool
            |
        MLP
            |
        Embedding Vector


    Input:
        [B,C,H,W]

    Output:
        [B,D]

    """


    def __init__(
            self,
            in_channels=Config.FEATURE_DIM,
            hidden_dim=512,
            embedding_dim=Config.PROJECTION_DIM
    ):


        super().__init__()



        self.pool = nn.AdaptiveAvgPool2d(1)



        self.projector = nn.Sequential(

            nn.Linear(
                in_channels,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.GELU(),


            nn.Linear(
                hidden_dim,
                embedding_dim
            )

        )



    # --------------------------------------------------------

    def forward(self, x):


        if x.ndim != 4:

            raise ValueError(
                "Projection input must be [B,C,H,W]"
            )


        x = self.pool(x)


        x = torch.flatten(
            x,
            start_dim=1
        )


        z = self.projector(x)


        # Normalize embedding
        z = F.normalize(
            z,
            p=2,
            dim=1
        )


        return z





# ============================================================
# Multi Scale Projection
# ============================================================


class MultiScaleProjectionHead(nn.Module):
    """

    Projection for:

        P2
        P3
        P4
        P5


    """


    def __init__(
            self,
            channels=Config.FEATURE_DIM,
            embedding_dim=Config.PROJECTION_DIM
    ):


        super().__init__()



        self.levels = [

            "p2",
            "p3",
            "p4",
            "p5"

        ]



        self.projectors = nn.ModuleDict({


            level:

            ProjectionHead(

                in_channels=channels,

                embedding_dim=embedding_dim

            )

            for level in self.levels


        })




    # --------------------------------------------------------

    def forward(self, features):


        embeddings = {}



        for level in self.levels:


            if level not in features:

                raise KeyError(

                    f"{level} missing from feature dictionary"

                )



            embeddings[level] = self.projectors[level](

                features[level]

            )



        return embeddings





# ============================================================
# Shared RGB Thermal Projection
# ============================================================


class SharedProjection(nn.Module):


    """

    RGB Feature

          |
     Projection

          |

    RGB Embedding



    Thermal Feature

          |
     Projection

          |

    Thermal Embedding


    """


    def __init__(self):


        super().__init__()



        self.rgb_projection = MultiScaleProjectionHead()



        self.thermal_projection = MultiScaleProjectionHead()



    # --------------------------------------------------------

    def forward(

            self,

            rgb_features,

            thermal_features

    ):



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



    features = {


        "p2":torch.randn(
            2,
            Config.FEATURE_DIM,
            200,
            200,
            device=device
        ),


        "p3":torch.randn(
            2,
            Config.FEATURE_DIM,
            100,
            100,
            device=device
        ),


        "p4":torch.randn(
            2,
            Config.FEATURE_DIM,
            50,
            50,
            device=device
        ),


        "p5":torch.randn(
            2,
            Config.FEATURE_DIM,
            25,
            25,
            device=device
        )

    }



    rgb_embed, thermal_embed = model(

        features,

        features

    )



    print("="*60)

    print("Projection Test")

    print("="*60)



    for level in rgb_embed:


        print(

            level,

            rgb_embed[level].shape,

            thermal_embed[level].shape

        )


    print("="*60)

    print("Projection Test Passed")
