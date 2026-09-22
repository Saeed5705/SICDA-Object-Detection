"""
==============================================================
File : losses/contrastive_loss.py

SICDA Cross Modal Contrastive Loss

Loss:
    L_cm = InfoNCE(RGB, Thermal)

Positive:
    Same image RGB-Thermal pair

Negative:
    Other samples in batch


Author:
Muhammad Saeed
==============================================================
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config



# ============================================================
# InfoNCE Contrastive Loss
# ============================================================


class ContrastiveLoss(nn.Module):
    """

    Bidirectional Contrastive Loss


    RGB -----> Thermal

    Thermal -> RGB


    Input:

        rgb_embedding:
            [B,D]


        thermal_embedding:
            [B,D]


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

            rgb_embedding,

            thermal_embedding

    ):


        if rgb_embedding.shape != thermal_embedding.shape:


            raise ValueError(

                f"Embedding mismatch:"
                f"{rgb_embedding.shape} vs "
                f"{thermal_embedding.shape}"

            )



        if rgb_embedding.ndim != 2:


            raise ValueError(

                "Embedding must be [Batch, Dimension]"

            )



        batch_size = rgb_embedding.size(0)



        if batch_size < 2:


            raise ValueError(

                "Contrastive learning requires batch size >= 2"

            )



        # ----------------------------------------------------
        # Normalize embeddings
        # ----------------------------------------------------


        rgb_embedding = F.normalize(

            rgb_embedding,

            dim=1

        )


        thermal_embedding = F.normalize(

            thermal_embedding,

            dim=1

        )



        # ----------------------------------------------------
        # Similarity matrix
        # ----------------------------------------------------


        logits = torch.matmul(

            rgb_embedding,

            thermal_embedding.T

        )



        logits = logits / self.temperature



        labels = torch.arange(

            batch_size,

            device=logits.device

        )



        # ----------------------------------------------------
        # RGB -> Thermal
        # ----------------------------------------------------


        loss_rgb = F.cross_entropy(

            logits,

            labels

        )



        # ----------------------------------------------------
        # Thermal -> RGB
        # ----------------------------------------------------


        loss_thermal = F.cross_entropy(

            logits.T,

            labels

        )



        loss = (

            loss_rgb +

            loss_thermal

        ) / 2.0



        return loss





# ============================================================
# Multi Scale Contrastive Loss
# ============================================================


class MultiScaleContrastiveLoss(nn.Module):

    """

    Contrastive loss on:

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



        self.levels = [

            "p2",

            "p3",

            "p4",

            "p5"

        ]




    # --------------------------------------------------------

    def forward(

            self,

            rgb_embeddings,

            thermal_embeddings

    ):


        total_loss = 0.0


        loss_dict = {}



        for level in self.levels:



            if level not in rgb_embeddings:


                raise KeyError(

                    f"{level} missing in RGB embeddings"

                )



            if level not in thermal_embeddings:


                raise KeyError(

                    f"{level} missing in Thermal embeddings"

                )




            loss = self.loss_fn(

                rgb_embeddings[level],

                thermal_embeddings[level]

            )



            loss_dict[level] = loss



            total_loss += loss



        total_loss = total_loss / len(self.levels)



        return total_loss, loss_dict





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



    rgb = {


        "p2": torch.randn(

            4,

            Config.PROJECTION_DIM,

            device=device

        ),


        "p3": torch.randn(

            4,

            Config.PROJECTION_DIM,

            device=device

        ),


        "p4": torch.randn(

            4,

            Config.PROJECTION_DIM,

            device=device

        ),


        "p5": torch.randn(

            4,

            Config.PROJECTION_DIM,

            device=device

        )

    }



    thermal = {


        "p2": torch.randn(

            4,

            Config.PROJECTION_DIM,

            device=device

        ),


        "p3": torch.randn(

            4,

            Config.PROJECTION_DIM,

            device=device

        ),


        "p4": torch.randn(

            4,

            Config.PROJECTION_DIM,

            device=device

        ),


        "p5": torch.randn(

            4,

            Config.PROJECTION_DIM,

            device=device

        )

    }



    loss, loss_dict = criterion(

        rgb,

        thermal

    )



    print("="*60)

    print("SICDA Contrastive Loss Test")

    print("="*60)



    print(

        "Total Contrastive Loss:",

        loss.item()

    )



    for k,v in loss_dict.items():

        print(

            k,

            v.item()

        )


    print("="*60)

    print("Contrastive Loss Test Passed")
