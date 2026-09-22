
import torch

import torch.nn as nn

import torch.nn.functional as F




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



        # --------------------------------------------------
        # Lateral 1x1 convolutions
        # --------------------------------------------------


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



        # --------------------------------------------------
        # Smoothing layers
        # --------------------------------------------------


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



    # ==================================================
    # Forward
    # ==================================================


    def forward(self, features):


        # Compatible with updated resnet.py


        c2 = features["0"]

        c3 = features["1"]

        c4 = features["2"]

        c5 = features["3"]




        # --------------------------------------------------
        # Lateral
        # --------------------------------------------------


        p5 = self.lateral5(c5)

        p4 = self.lateral4(c4)

        p3 = self.lateral3(c3)

        p2 = self.lateral2(c2)




        # --------------------------------------------------
        # Top-down pathway
        # --------------------------------------------------


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




        # --------------------------------------------------
        # Smooth
        # --------------------------------------------------


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





# ===============================================================
# Test
# ===============================================================


if __name__ == "__main__":


    print("="*60)

    print("Testing SICDA FPN")

    print("="*60)



    fpn = FeaturePyramidNetwork()



    features = {


        "0":
        torch.randn(
            2,
            256,
            200,
            200
        ),


        "1":
        torch.randn(
            2,
            512,
            100,
            100
        ),


        "2":
        torch.randn(
            2,
            1024,
            50,
            50
        ),


        "3":
        torch.randn(
            2,
            2048,
            25,
            25
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