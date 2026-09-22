"""
==============================================================
File : models/illumination_module.py

SICDA Illumination Aware Adaptive Fusion

Formula:

F_fused =
        α * F_RGB
        +
        (1-α) * F_Thermal


α:
    RGB illumination reliability score


Used for:
    - Low Light Adaptation
    - RGB/Thermal Fusion
    - Domain Adaptation


Author:
Muhammad Saeed
==============================================================
"""


import torch
import torch.nn as nn



from config import Config




# ============================================================
# Illumination Estimator
# ============================================================


class IlluminationEstimator(nn.Module):


    """
    Estimate RGB reliability weight α

    Output:

        α ∈ [0,1]

    """


    def __init__(

            self,

            channels=Config.FEATURE_DIM

    ):


        super().__init__()



        hidden = max(

            channels // 4,

            32

        )



        self.pool = nn.AdaptiveAvgPool2d(1)



        self.fc = nn.Sequential(

            nn.Linear(

                channels,

                hidden

            ),

            nn.ReLU(inplace=True),


            nn.Linear(

                hidden,

                1

            ),

            nn.Sigmoid()

        )



    def forward(self, feature):


        B,C,_,_ = feature.shape



        x = self.pool(feature)


        x = x.view(

            B,

            C

        )


        alpha = self.fc(x)



        return alpha





# ============================================================
# Adaptive Fusion Block
# ============================================================


class AdaptiveFusion(nn.Module):


    def __init__(self):

        super().__init__()



    def forward(

            self,

            rgb,

            thermal,

            alpha

    ):


        alpha = alpha.view(

            -1,

            1,

            1,

            1

        )



        fused = (

            alpha * rgb

            +

            (1-alpha) * thermal

        )



        return fused





# ============================================================
# Multi Scale Illumination Fusion
# ============================================================


class IlluminationAwareFusion(nn.Module):


    """

    Fusion on:

        P2

        P3

        P4

        P5


    Input:

        RGB features dictionary

        Thermal features dictionary



    Output:

        fused features dictionary

        illumination scores


    """


    def __init__(

            self,

            channels=256

    ):


        super().__init__()



        self.levels = [

            "p2",

            "p3",

            "p4",

            "p5"

        ]



        self.estimators = nn.ModuleDict({


            level:

            IlluminationEstimator(

                channels

            )


            for level in self.levels

        })



        self.fusion = AdaptiveFusion()




    # ========================================================
    # Forward
    # ========================================================


    def forward(

            self,

            rgb_features,

            thermal_features

    ):


        fused_features = {}

        illumination_scores = {}



        for level in self.levels:


            if level not in rgb_features:

                raise KeyError(

                    f"{level} missing in RGB features"

                )



            if level not in thermal_features:

                raise KeyError(

                    f"{level} missing in Thermal features"

                )



            rgb = rgb_features[level]


            thermal = thermal_features[level]



            if rgb.shape != thermal.shape:

                raise ValueError(

                    f"Shape mismatch at {level}: "

                    f"{rgb.shape} vs {thermal.shape}"

                )



            # Estimate RGB illumination reliability

            alpha = self.estimators[level](

                rgb

            )



            fused = self.fusion(

                rgb,

                thermal,

                alpha

            )



            fused_features[level] = fused



            illumination_scores[level] = alpha



        return (

            fused_features,

            illumination_scores

        )





# ============================================================
# Factory
# ============================================================


def build_illumination_module():


    return IlluminationAwareFusion(

        channels=256

    )





# ============================================================
# Test
# ============================================================


if __name__ == "__main__":


    device=torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )



    model = build_illumination_module().to(device)



    rgb={


        "p2":torch.randn(

            2,256,128,128,

            device=device

        ),


        "p3":torch.randn(

            2,256,64,64,

            device=device

        ),


        "p4":torch.randn(

            2,256,32,32,

            device=device

        ),


        "p5":torch.randn(

            2,256,16,16,

            device=device

        )

    }




    thermal={


        "p2":torch.randn(

            2,256,128,128,

            device=device

        ),


        "p3":torch.randn(

            2,256,64,64,

            device=device

        ),


        "p4":torch.randn(

            2,256,32,32,

            device=device

        ),


        "p5":torch.randn(

            2,256,16,16,

            device=device

        )

    }



    fused, alpha = model(

        rgb,

        thermal

    )



    print("="*60)

    print("Illumination Module Test")

    print("="*60)



    for level in fused:


        print(

            level,

            "Fused:",

            fused[level].shape,

            "Alpha:",

            alpha[level].shape

        )



    print("="*60)

    print("Test Passed Successfully")