"""
==============================================================
File : grl.py

Gradient Reversal Layer (GRL)

Used in SICDA Domain Adaptation

Source:
    KAIST

Target:
    FLIR-ADAS


DANN:

Feature Extractor
        |
        |
        v
 Gradient Reversal Layer
        |
        |
        v
 Domain Discriminator


Forward:
    Identity mapping


Backward:
    Reverse gradient


Author:
    Muhammad Saeed
==============================================================
"""


import torch
import torch.nn as nn

from torch.autograd import Function

import math

from config import Config



# ============================================================
# Gradient Reverse Function
# ============================================================


class GradientReverseFunction(Function):
    """
    Forward:
        x


    Backward:

        -lambda * gradient

    """


    @staticmethod
    def forward(
            ctx,
            x,
            lambda_grl
    ):


        ctx.lambda_grl = lambda_grl


        return x.view_as(x)



    @staticmethod
    def backward(
            ctx,
            grad_output
    ):


        grad_input = (

            -ctx.lambda_grl *

            grad_output

        )


        return grad_input, None




# ============================================================
# Gradient Reversal Layer
# ============================================================


class GradientReversalLayer(nn.Module):


    def __init__(
            self,
            lambda_grl=1.0
    ):


        super().__init__()


        self.lambda_grl = lambda_grl



    # --------------------------------------------------------

    def forward(
            self,
            x
    ):


        return GradientReverseFunction.apply(

            x,

            self.lambda_grl

        )



    # --------------------------------------------------------

    def update_lambda(
            self,
            lambda_value
    ):


        self.lambda_grl = float(lambda_value)




# ============================================================
# DANN Lambda Scheduler
# ============================================================


class GRLScheduler:
    """
    Original DANN Schedule:


        λ(p)=2/(1+exp(-10p))-1


    p:

        training progress

        0 -> start

        1 -> end



    """



    def __init__(
            self,
            gamma=10.0
    ):


        self.gamma = gamma




    def get_lambda(
            self,
            progress
    ):


        progress = max(

            0.0,

            min(

                progress,

                1.0

            )

        )


        lambda_value = (

            2.0 /

            (

                1.0 +

                math.exp(

                    -self.gamma *

                    progress

                )

            )

            -

            1.0

        )


        return lambda_value




# ============================================================
# Build GRL
# ============================================================


def build_grl(
        lambda_grl=Config.LAMBDA_ADV
):


    return GradientReversalLayer(

        lambda_grl

    )




# ============================================================
# Utility Function
# ============================================================


def update_grl_lambda(
        grl_layer,
        epoch,
        total_epochs
):

    """
    Update GRL lambda during training.


    Example:

        Epoch 0:
            lambda close to 0


        Final Epoch:
            lambda close to 1


    """


    progress = epoch / max(

        total_epochs - 1,

        1

    )


    scheduler = GRLScheduler()


    lambda_value = scheduler.get_lambda(

        progress

    )


    grl_layer.update_lambda(

        lambda_value

    )


    return lambda_value




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

    print("Testing GRL")

    print("="*60)



    grl = build_grl().to(device)



    x = torch.randn(

        4,

        Config.FEATURE_DIM,

        requires_grad=True,

        device=device

    )



    output = grl(x)


    loss = output.mean()


    loss.backward()



    print(

        "Input Shape:",

        x.shape

    )


    print(

        "Gradient Mean:",

        x.grad.mean().item()

    )



    print(

        "Initial Lambda:",

        grl.lambda_grl

    )



    print("\nDANN Lambda Schedule")



    scheduler = GRLScheduler()



    for p in [

        0.0,

        0.25,

        0.50,

        0.75,

        1.0

    ]:


        print(

            f"Progress {p:.2f}",

            "Lambda",

            scheduler.get_lambda(p)

        )



    print("\nEpoch Update Test")



    for epoch in range(5):


        value = update_grl_lambda(

            grl,

            epoch,

            5

        )


        print(

            "Epoch:",

            epoch,

            "Lambda:",

            value

        )



    print("="*60)

    print(
        "GRL Test Completed Successfully"
    )

    print("="*60)