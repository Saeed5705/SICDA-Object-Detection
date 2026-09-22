"""
==============================================================
File : losses/consistency.py

SICDA Consistency Loss

Purpose:
    Enforce consistency between:

        RGB branch
        Thermal branch


Loss:

    L_cons =
        Feature Consistency
        +
        Prediction Consistency


Author:
Muhammad Saeed
==============================================================
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config



# ============================================================
# Feature Consistency Loss
# ============================================================


class FeatureConsistencyLoss(nn.Module):
    """
    Compare RGB and Thermal feature representations.


    Loss:

        || RGB - Thermal ||2


    Input:

        feature_rgb:
            [B,C,H,W]

        feature_thermal:
            [B,C,H,W]

    """


    def __init__(self):

        super().__init__()



    def forward(

            self,

            rgb_feature,

            thermal_feature

    ):


        if rgb_feature.shape != thermal_feature.shape:

            raise ValueError(

                f"Feature mismatch:"
                f"{rgb_feature.shape} vs "
                f"{thermal_feature.shape}"

            )



        loss = F.mse_loss(

            rgb_feature,

            thermal_feature

        )


        return loss





# ============================================================
# Prediction Consistency Loss
# ============================================================


class PredictionConsistencyLoss(nn.Module):
    """

    KL divergence between RGB and Thermal predictions


    Input:

        rgb_logits

        thermal_logits


    """


    def __init__(self):

        super().__init__()



    def forward(

            self,

            rgb_logits,

            thermal_logits

    ):


        if rgb_logits.shape != thermal_logits.shape:


            raise ValueError(

                "Prediction shape mismatch"

            )



        rgb_prob = F.softmax(

            rgb_logits,

            dim=1

        )



        thermal_log_prob = F.log_softmax(

            thermal_logits,

            dim=1

        )



        loss = F.kl_div(

            thermal_log_prob,

            rgb_prob,

            reduction="batchmean"

        )



        return loss





# ============================================================
# Multi Scale Consistency Loss
# ============================================================


class MultiScaleConsistencyLoss(nn.Module):
    """

    Consistency on:

        P2

        P3

        P4

        P5


    """


    def __init__(self):

        super().__init__()


        self.feature_loss = FeatureConsistencyLoss()


        self.levels = [

            "p2",

            "p3",

            "p4",

            "p5"

        ]



    # --------------------------------------------------------

    def forward(

            self,

            rgb_features,

            thermal_features

    ):


        total_loss = 0.0


        loss_dict = {}



        for level in self.levels:


            if level not in rgb_features:

                raise KeyError(

                    f"{level} missing in RGB features"

                )


            if level not in thermal_features:

                raise KeyError(

                    f"{level} missing in Thermal features"

                )



            loss = self.feature_loss(

                rgb_features[level],

                thermal_features[level]

            )



            loss_dict[level] = loss



            total_loss += loss



        total_loss /= len(self.levels)



        return total_loss, loss_dict





# ============================================================
# Complete Consistency Loss
# ============================================================


class ConsistencyLoss(nn.Module):

    """

    Complete SICDA Consistency Module


    Supports:

        1. Feature consistency

        2. Prediction consistency



    """


    def __init__(

            self,

            lambda_feature=1.0,

            lambda_prediction=1.0

    ):


        super().__init__()



        self.lambda_feature = lambda_feature

        self.lambda_prediction = lambda_prediction



        self.feature_consistency = MultiScaleConsistencyLoss()


        self.prediction_consistency = PredictionConsistencyLoss()



    # --------------------------------------------------------

    def forward(

            self,

            rgb_features,

            thermal_features,

            rgb_logits=None,

            thermal_logits=None

    ):


        feature_loss, feature_dict = self.feature_consistency(

            rgb_features,

            thermal_features

        )



        total_loss = (

            self.lambda_feature *

            feature_loss

        )



        loss_dict = {


            "feature_consistency":

            feature_loss

        }



        if (

            rgb_logits is not None

            and

            thermal_logits is not None

        ):


            prediction_loss = self.prediction_consistency(

                rgb_logits,

                thermal_logits

            )


            total_loss += (

                self.lambda_prediction *

                prediction_loss

            )


            loss_dict[

                "prediction_consistency"

            ] = prediction_loss



        return total_loss, loss_dict





# ============================================================
# Factory
# ============================================================


def build_consistency_loss():


    return ConsistencyLoss()





# ============================================================
# Test
# ============================================================


if __name__ == "__main__":


    device=torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )



    criterion = build_consistency_loss().to(device)



    rgb = {


        "p2":torch.randn(

            2,

            Config.FEATURE_DIM,

            64,

            64,

            device=device

        ),


        "p3":torch.randn(

            2,

            Config.FEATURE_DIM,

            32,

            32,

            device=device

        ),


        "p4":torch.randn(

            2,

            Config.FEATURE_DIM,

            16,

            16,

            device=device

        ),


        "p5":torch.randn(

            2,

            Config.FEATURE_DIM,

            8,

            8,

            device=device

        )

    }



    thermal = {


        "p2":torch.randn(

            2,

            Config.FEATURE_DIM,

            64,

            64,

            device=device

        ),


        "p3":torch.randn(

            2,

            Config.FEATURE_DIM,

            32,

            32,

            device=device

        ),


        "p4":torch.randn(

            2,

            Config.FEATURE_DIM,

            16,

            16,

            device=device

        ),


        "p5":torch.randn(

            2,

            Config.FEATURE_DIM,

            8,

            8,

            device=device

        )

    }



    loss, losses = criterion(

        rgb,

        thermal

    )



    print("="*60)

    print("SICDA Consistency Loss")

    print("="*60)


    print(

        "Total Loss:",

        loss.item()

    )


    for k,v in losses.items():

        print(

            k,

            v.item()

        )


    print("="*60)

    print("Consistency Loss Test Passed")