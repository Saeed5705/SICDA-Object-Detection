"""
==============================================================
File : models/fpn.py

SICDA Feature Pyramid Network

Input:
    ResNet50 C2,C3,C4,C5 features

Output:
    P2,P3,P4,P5 features

Used by:
    - Cross Modal Attention
    - MMD Loss
    - Contrastive Loss
    - Detector

Author:
Muhammad Saeed
==============================================================
"""


import torch
import torch.nn as nn
import torch.nn.functional as F



# ============================================================
# Feature Pyramid Network
# ============================================================


class FeaturePyramidNetwork(nn.Module):


    def __init__(

            self,

            in_channels=None,

            out_channels=256

    ):


        super().__init__()



        if in_channels is None:

            in_channels = [

                256,

                512,

                1024,

                2048

            ]



        self.out_channels = out_channels



        # ----------------------------------------------------
        # Lateral 1x1 Convolution
        # ----------------------------------------------------


        self.lateral2 = nn.Conv2d(

            in_channels[0],

            out_channels,

            kernel_size=1

        )


        self.lateral3 = nn.Conv2d(

            in_channels[1],

            out_channels,

            kernel_size=1

        )


        self.lateral4 = nn.Conv2d(

            in_channels[2],

            out_channels,

            kernel_size=1

        )


        self.lateral5 = nn.Conv2d(

            in_channels[3],

            out_channels,

            kernel_size=1

        )



        # ----------------------------------------------------
        # Smooth 3x3 Convolution
        # ----------------------------------------------------


        self.smooth2 = nn.Conv2d(

            out_channels,

            out_channels,

            kernel_size=3,

            padding=1

        )


        self.smooth3 = nn.Conv2d(

            out_channels,

            out_channels,

            kernel_size=3,

            padding=1

        )


        self.smooth4 = nn.Conv2d(

            out_channels,

            out_channels,

            kernel_size=3,

            padding=1

        )


        self.smooth5 = nn.Conv2d(

            out_channels,

            out_channels,

            kernel_size=3,

            padding=1

        )



    # ========================================================
    # Forward
    # ========================================================


    def forward(self, features):


        """

        Input:

            features = {

                "0": C2,

                "1": C3,

                "2": C4,

                "3": C5

            }


        Output:

            {

                p2,

                p3,

                p4,

                p5

            }

        """



        if not isinstance(features, dict):

            raise TypeError(

                "FPN input must be dictionary"

            )



        required = [

            "0",

            "1",

            "2",

            "3"

        ]



        for key in required:

            if key not in features:

                raise KeyError(

                    f"Missing feature level {key}"

                )



        # ----------------------------------------------------
        # Backbone Features
        # ----------------------------------------------------


        c2 = features["0"]

        c3 = features["1"]

        c4 = features["2"]

        c5 = features["3"]



        # ----------------------------------------------------
        # Lateral Features
        # ----------------------------------------------------


        p5 = self.lateral5(c5)

        p4 = self.lateral4(c4)

        p3 = self.lateral3(c3)

        p2 = self.lateral2(c2)



        # ----------------------------------------------------
        # Top Down Pathway
        # ----------------------------------------------------


        p4 = p4 + F.interpolate(

            p5,

            size=p4.shape[-2:],

            mode="nearest"

        )



        p3 = p3 + F.interpolate(

            p4,

            size=p3.shape[-2:],

            mode="nearest"

        )



        p2 = p2 + F.interpolate(

            p3,

            size=p2.shape[-2:],

            mode="nearest"

        )



        # ----------------------------------------------------
        # Smooth
        # ----------------------------------------------------


        p5 = self.smooth5(p5)

        p4 = self.smooth4(p4)

        p3 = self.smooth3(p3)

        p2 = self.smooth2(p2)



        return {


            "p2": p2,

            "p3": p3,

            "p4": p4,

            "p5": p5


        }




# ============================================================
# Test
# ============================================================


if __name__ == "__main__":


    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )



    print("="*60)

    print("Testing SICDA FPN")

    print("="*60)



    fpn = FeaturePyramidNetwork().to(device)



    features = {


        "0": torch.randn(

            2,

            256,

            128,

            128,

            device=device

        ),


        "1": torch.randn(

            2,

            512,

            64,

            64,

            device=device

        ),


        "2": torch.randn(

            2,

            1024,

            32,

            32,

            device=device

        ),


        "3": torch.randn(

            2,

            2048,

            16,

            16,

            device=device

        )


    }



    outputs = fpn(features)



    print("\nFPN Outputs")


    for k,v in outputs.items():

        print(

            k,

            v.shape

        )


    print("="*60)

    print("FPN Test Passed Successfully")

    print("="*60)