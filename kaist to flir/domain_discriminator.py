"""
==============================================================
File : domain_discriminator.py

SICDA Domain Discriminator

Source Domain:
    KAIST (RGB + Thermal)

Target Domain:
    FLIR-ADAS

Purpose:
    Adversarial Domain Adaptation using DANN

Author:
    Muhammad Saeed
==============================================================
"""


import torch
import torch.nn as nn

from config import Config


# ============================================================
# Single Scale Domain Classifier
# ============================================================

class DomainClassifier(nn.Module):
    """
    Binary Domain Classifier

    Domain Labels:

        0 -> Source Domain (KAIST)

        1 -> Target Domain (FLIR-ADAS)

    Input:
        Feature Map [B,C,H,W]

    Output:
        Domain logits [B,2]

    """

    def __init__(
            self,
            in_channels=Config.FEATURE_DIM,
            hidden_dim=Config.DOMAIN_HIDDEN
    ):

        super().__init__()


        self.pool = nn.AdaptiveAvgPool2d(1)


        self.classifier = nn.Sequential(

            nn.Linear(
                in_channels,
                hidden_dim
            ),
            #nn.BatchNorm1d(256)
            nn.LayerNorm(256),

            nn.ReLU(inplace=True),


            nn.Dropout(
                0.5
            ),


            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),


            nn.LayerNorm(
                hidden_dim // 2
            ),


            nn.ReLU(inplace=True),


            nn.Dropout(
                0.5
            ),


            nn.Linear(
                hidden_dim // 2,
                2
            )

        )


    # --------------------------------------------------------

    def forward(self, feature):

        """

        feature:

            [B,C,H,W]


        return:

            [B,2]

        """


        x = self.pool(feature)

        x = torch.flatten(
            x,
            start_dim=1
        )


        logits = self.classifier(x)


        return logits



# ============================================================
# Multi Scale Domain Discriminator
# ============================================================


class MultiScaleDomainDiscriminator(nn.Module):
    """

    Domain classifier on FPN levels:

        P2
        P3
        P4
        P5


    """

    def __init__(
            self,
            channels=Config.FEATURE_DIM
    ):

        super().__init__()



        self.levels = [

            "p2",
            "p3",
            "p4",
            "p5"

        ]


        self.discriminators = nn.ModuleDict({

            level:

            DomainClassifier(
                channels
            )

            for level in self.levels

        })



    # --------------------------------------------------------

    def forward(
            self,
            features
    ):

        outputs = {}


        for level in self.levels:


            if level not in features:

                raise KeyError(
                    f"{level} missing from FPN features"
                )


            outputs[level] = self.discriminators[level](

                features[level]

            )


        return outputs



# ============================================================
# Domain Classification Loss
# ============================================================


class DomainLoss(nn.Module):
    """

    Cross Entropy Domain Loss


    Source:
        KAIST -> label 0


    Target:
        FLIR -> label 1


    """


    def __init__(self):

        super().__init__()


        self.criterion = nn.CrossEntropyLoss()



    # --------------------------------------------------------

    def forward(
            self,
            predictions,
            labels
    ):


        if len(predictions)==0:

            raise ValueError(
                "No domain predictions found"
            )


        total_loss = 0.0


        for level in predictions:


            total_loss += self.criterion(

                predictions[level],

                labels

            )


        total_loss /= len(predictions)


        return total_loss




# ============================================================
# Factory
# ============================================================


def build_domain_discriminator():


    return MultiScaleDomainDiscriminator()




# ============================================================
# Testing
# ============================================================


if __name__ == "__main__":


    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    model = build_domain_discriminator().to(device)



    features = {


        "p2":
        torch.randn(
            4,
            Config.FEATURE_DIM,
            200,
            200,
            device=device
        ),


        "p3":
        torch.randn(
            4,
            Config.FEATURE_DIM,
            100,
            100,
            device=device
        ),


        "p4":
        torch.randn(
            4,
            Config.FEATURE_DIM,
            50,
            50,
            device=device
        ),


        "p5":
        torch.randn(
            4,
            Config.FEATURE_DIM,
            25,
            25,
            device=device
        )

    }



    outputs = model(features)



    # Example:

    # first two images -> KAIST source

    # last two images -> FLIR target


    labels = torch.tensor(

        [

            0,

            0,

            1,

            1

        ],

        device=device

    )



    criterion = DomainLoss()


    loss = criterion(

        outputs,

        labels

    )


    print("="*60)

    print("SICDA Domain Discriminator Test")

    print("="*60)



    for k,v in outputs.items():

        print(
            k,
            v.shape
        )


    print()

    print(
        "Domain Loss:",
        loss.item()
    )


    print("="*60)

    print(
        "Test Passed Successfully"
    )

    print("="*60)