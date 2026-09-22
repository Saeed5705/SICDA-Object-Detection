"""
==============================================================
File : models/resnet.py

SICDA ResNet50 Dual Encoder

Branches:
    1. RGB Encoder
    2. Thermal Encoder

Output:
    C2, C3, C4, C5 features

Compatible with:
    - FPN
    - Cross Modal Attention
    - MMD
    - Contrastive Learning
    - Faster R-CNN

Author:
Muhammad Saeed
==============================================================
"""


import torch
import torch.nn as nn


from torchvision.models import (
    resnet50,
    ResNet50_Weights
)



# ============================================================
# ResNet50 Feature Encoder
# ============================================================


class ResNetEncoder(nn.Module):


    def __init__(
            self,
            pretrained=True,
            in_channels=3
    ):


        super().__init__()



        # ----------------------------------------------------
        # Load ResNet50
        # ----------------------------------------------------

        if pretrained:

            weights = ResNet50_Weights.DEFAULT

        else:

            weights = None



        backbone = resnet50(

            weights=weights

        )



        # ----------------------------------------------------
        # Modify First Layer for Thermal
        # ----------------------------------------------------

        if in_channels == 1:


            old_conv = backbone.conv1



            backbone.conv1 = nn.Conv2d(

                in_channels=1,

                out_channels=64,

                kernel_size=7,

                stride=2,

                padding=3,

                bias=False

            )



            # Copy RGB pretrained weights

            with torch.no_grad():


                backbone.conv1.weight.copy_(

                    old_conv.weight.mean(

                        dim=1,

                        keepdim=True

                    )

                )



        # ----------------------------------------------------
        # Backbone Layers
        # ----------------------------------------------------


        self.conv1 = backbone.conv1

        self.bn1 = backbone.bn1

        self.relu = backbone.relu

        self.maxpool = backbone.maxpool



        self.layer1 = backbone.layer1

        self.layer2 = backbone.layer2

        self.layer3 = backbone.layer3

        self.layer4 = backbone.layer4



        # ----------------------------------------------------
        # Output Channels for FPN
        # ----------------------------------------------------

        self.out_channels_list = [

            256,

            512,

            1024,

            2048

        ]


        self.out_channels = 2048



    # ========================================================
    # Forward
    # ========================================================


    def forward(self, x):


        # Stem

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)



        # ResNet Stages


        c2 = self.layer1(x)

        c3 = self.layer2(c2)

        c4 = self.layer3(c3)

        c5 = self.layer4(c4)



        return {


            "0": c2,

            "1": c3,

            "2": c4,

            "3": c5


        }




# ============================================================
# RGB Encoder
# ============================================================


class RGBEncoder(ResNetEncoder):


    def __init__(

            self,

            pretrained=True

    ):


        super().__init__(

            pretrained=pretrained,

            in_channels=3

        )




# ============================================================
# Thermal Encoder
# ============================================================


class ThermalEncoder(ResNetEncoder):


    def __init__(

            self,

            pretrained=True

    ):


        super().__init__(

            pretrained=pretrained,

            in_channels=1

        )




# ============================================================
# Freeze Utilities
# ============================================================


def freeze_model(model):


    for param in model.parameters():

        param.requires_grad = False




def unfreeze_model(model):


    for param in model.parameters():

        param.requires_grad = True




# ============================================================
# Test
# ============================================================


if __name__ == "__main__":


    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )



    print("="*70)

    print("Testing SICDA ResNet50 Encoder")

    print("="*70)



    rgb_encoder = RGBEncoder().to(device)

    thermal_encoder = ThermalEncoder().to(device)



    rgb = torch.randn(

        2,

        3,

        512,

        512,

        device=device

    )



    thermal = torch.randn(

        2,

        1,

        512,

        512,

        device=device

    )



    rgb_features = rgb_encoder(rgb)


    thermal_features = thermal_encoder(thermal)



    print("\nRGB Features")


    for k,v in rgb_features.items():

        print(

            k,

            v.shape

        )



    print("\nThermal Features")


    for k,v in thermal_features.items():

        print(

            k,

            v.shape

        )



    print("="*70)

    print("ResNet50 Test Passed Successfully")

    print("="*70)