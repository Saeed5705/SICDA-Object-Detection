
import torch

import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)



# ===============================================================
# ResNet50 Feature Encoder
# ===============================================================


class ResNetEncoder(nn.Module):


    def __init__(
            self,
            pretrained=True,
            in_channels=3
    ):

        super().__init__()



        # -------------------------------------------------------
        # Load pretrained ResNet50
        # -------------------------------------------------------

        if pretrained:

            weights = ResNet50_Weights.DEFAULT

        else:

            weights = None



        backbone = resnet50(
            weights=weights
        )



        # -------------------------------------------------------
        # Thermal Branch
        #
        # RGB:
        #       3 channels
        #
        # Thermal:
        #       1 channel
        #
        # -------------------------------------------------------


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



            # Initialize thermal weights
            # using RGB pretrained weights


            with torch.no_grad():


                backbone.conv1.weight.copy_(

                    old_conv.weight.mean(

                        dim=1,

                        keepdim=True

                    )

                )



        # -------------------------------------------------------
        # ResNet Layers
        # -------------------------------------------------------


        self.conv1 = backbone.conv1

        self.bn1 = backbone.bn1

        self.relu = backbone.relu

        self.maxpool = backbone.maxpool


        self.layer1 = backbone.layer1

        self.layer2 = backbone.layer2

        self.layer3 = backbone.layer3

        self.layer4 = backbone.layer4



        # -------------------------------------------------------
        # Channels for FPN
        # -------------------------------------------------------


        self.out_channels_list = [

            256,

            512,

            1024,

            2048

        ]


        self.out_channels = 2048



    # ===========================================================
    # Forward
    # ===========================================================


    def forward(self, x):


        # Stem

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)



        # ResNet stages


        c2 = self.layer1(x)

        c3 = self.layer2(c2)

        c4 = self.layer3(c3)

        c5 = self.layer4(c4)



        # Output for FPN


        return {


            "0": c2,

            "1": c3,

            "2": c4,

            "3": c5


        }





# ===============================================================
# RGB Encoder
# ===============================================================


class RGBEncoder(ResNetEncoder):


    def __init__(
            self,
            pretrained=True
    ):


        super().__init__(

            pretrained=pretrained,

            in_channels=3

        )





# ===============================================================
# Thermal Encoder
# ===============================================================


class ThermalEncoder(ResNetEncoder):


    def __init__(
            self,
            pretrained=True
    ):


        super().__init__(

            pretrained=pretrained,

            in_channels=1

        )





# ===============================================================
# Test
# ===============================================================


if __name__ == "__main__":


    print("="*70)

    print("Testing SICDA ResNet50 Encoder")

    print("="*70)



    rgb_encoder = RGBEncoder()



    thermal_encoder = ThermalEncoder()



    rgb = torch.randn(

        2,

        3,

        800,

        800

    )



    thermal = torch.randn(

        2,

        1,

        800,

        800

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

    print("Test completed successfully")

    print("="*70)