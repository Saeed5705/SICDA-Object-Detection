

import torch
import torch.nn as nn
from torch.autograd import Function
from config import Config
import math
# ============================================================
# Gradient Reversal Function
# ============================================================

class GradientReverseFunction(Function):
    """
    Forward:
        Identity

    Backward:
        Multiply gradient by -lambda
    """

    @staticmethod
    def forward(ctx, x, lambda_grl):

        ctx.lambda_grl = lambda_grl

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):

        grad_input = grad_output.neg() * ctx.lambda_grl

        return grad_input, None


# ============================================================
# GRL Layer
# ============================================================

class GradientReversalLayer(nn.Module):

    def __init__(self, lambda_grl=Config.LAMBDA_ADV):

        super().__init__()

        self.lambda_grl = lambda_grl

    def forward(self, x):

        return GradientReverseFunction.apply(

            x,

            self.lambda_grl

        )

    def set_lambda(self, value):

        """
        Update lambda during training.
        """

        self.lambda_grl = float(value)


# ============================================================
# Lambda Scheduler
# ============================================================

class GRLScheduler:
    """
    DANN Schedule

    λ = 2/(1+exp(-10p))-1

    p ∈ [0,1]
    """

    def __init__(self):

        pass

    @staticmethod
    def get_lambda(progress):

        progress = max(0.0, min(progress, 1.0))

        return 2.0/(1.0 + math.exp(-10.0 * progress)) - 1.0

        #return lambda_value.item()


# ============================================================
# Factory
# ============================================================

def build_grl(lambda_grl=1.0):

    return GradientReversalLayer(

        lambda_grl=lambda_grl

    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    grl = build_grl().to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device =torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    x = torch.randn(

        4,

        Config.FEATURE_DIM,

        requires_grad=True,
        device=device

    )

    y = grl(x)

    loss = y.mean()

    loss.backward()
    print("Gradient Mean:", x.grad.mean().item())
    print("Gradient Std:", x.grad.std().item())

    print("=" * 60)

    print("GRL Test")

    print("=" * 60)

    print("Input Shape :", x.shape)

    print("Gradient Shape :", x.grad.shape)

    print("Lambda :", grl.lambda_grl)

    scheduler = GRLScheduler()

    print("\nLambda Schedule")

    for p in [0.0,0.25,0.5,0.75,1.0]:

        print(

            f"Progress={p:.2f}",

            " Lambda=",

            scheduler.get_lambda(p)

        )

    print("=" * 60)