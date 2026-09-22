

import torch

import torch.nn as nn


from resnet import (
    RGBEncoder,
    ThermalEncoder
)


from fpn import FeaturePyramidNetwork




class DualBackbone(nn.Module):


    def __init__(
            self,
            pretrained=True
    ):


        super().__init__()



        # --------------------------------------------------
        # RGB Branch
        # --------------------------------------------------


        self.rgb_encoder = RGBEncoder(

            pretrained=pretrained

        )


        self.rgb_fpn = FeaturePyramidNetwork(

            in_channels=[

                256,

                512,

                1024,

                2048

            ],

            out_channels=256

        )



        # --------------------------------------------------
        # Thermal Branch
        # --------------------------------------------------


        self.thermal_encoder = ThermalEncoder(

            pretrained=pretrained

        )


        self.thermal_fpn = FeaturePyramidNetwork(

            in_channels=[

                256,

                512,

                1024,

                2048

            ],

            out_channels=256

        )



        # output dimension

        self.out_channels = 256



    # ======================================================
    # Forward
    # ======================================================


    def forward(
            self,
            rgb,
            thermal=None
    ):



        # --------------------------------------------------
        # RGB Feature Extraction
        # --------------------------------------------------


        rgb_features = self.rgb_encoder(

            rgb

        )


        rgb_features = self.rgb_fpn(

            rgb_features

        )



        # --------------------------------------------------
        # Thermal Feature Extraction
        # --------------------------------------------------


        thermal_features = None


        if thermal is not None:


            thermal_features = self.thermal_encoder(

                thermal

            )


            thermal_features = self.thermal_fpn(

                thermal_features

            )



        return {


            "rgb":

            rgb_features,


            "thermal":

            thermal_features


        }



    # ======================================================
    # Freeze
    # ======================================================


    def freeze_backbone(self):


        for param in self.rgb_encoder.parameters():

            param.requires_grad = False



        for param in self.thermal_encoder.parameters():

            param.requires_grad = False



    # ======================================================
    # Unfreeze
    # ======================================================


    def unfreeze_backbone(self):


        for param in self.rgb_encoder.parameters():

            param.requires_grad = True



        for param in self.thermal_encoder.parameters():

            param.requires_grad = True





# ===========================================================
# Test
# ===========================================================


if __name__ == "__main__":


    print("="*70)

    print("Testing SICDA Dual Backbone")

    print("="*70)



    model = DualBackbone()



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



    outputs = model(

        rgb,

        thermal

    )



    print("\nRGB FPN Features")

    for k,v in outputs["rgb"].items():

        print(

            k,

            v.shape

        )



    print("\nThermal FPN Features")


    for k,v in outputs["thermal"].items():

        print(

            k,

            v.shape

        )



    print("="*70)

    print("Backbone test completed")