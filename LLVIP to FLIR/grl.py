"""
SICDA Gradient Reversal Layer (GRL)
===================================

Purpose:
    Gradient Reversal Layer for adversarial domain adaptation.

Forward:
    Identity mapping.

Backward:
    Gradient is multiplied by -lambda.

DANN Schedule:
    lambda(p) = 2 / (1 + exp(-10p)) - 1

Classes:
    GradientReverseFunction
    GradientReversalLayer
    GRLScheduler

Factory:
    build_grl()

Supports:
    CPU
    CUDA
    Dynamic lambda scheduling
    Config-based lambda
"""


import math

import torch
import torch.nn as nn
from torch.autograd import Function

from config import Config


# ======================================================================
# GRADIENT REVERSAL FUNCTION
# ======================================================================

class GradientReverseFunction(Function):
    """
    Autograd implementation of Gradient Reversal.

    Forward:
        Returns input unchanged.

    Backward:
        Reverses the gradient and scales it by lambda.

        grad_input = -lambda * grad_output
    """

    @staticmethod
    def forward(
        ctx,
        x,
        lambda_grl
    ):

        # --------------------------------------------------------------
        # Store lambda for backward
        # --------------------------------------------------------------

        ctx.lambda_grl = float(
            lambda_grl
        )

        # --------------------------------------------------------------
        # Identity operation
        # --------------------------------------------------------------

        return x.view_as(x)

    @staticmethod
    def backward(
        ctx,
        grad_output
    ):

        # --------------------------------------------------------------
        # Reverse gradient
        # --------------------------------------------------------------

        grad_input = (
            -ctx.lambda_grl
            *
            grad_output
        )

        # --------------------------------------------------------------
        # x gradient + lambda gradient
        #
        # Lambda is not a learnable tensor.
        # Therefore return None for lambda.
        # --------------------------------------------------------------

        return grad_input, None


# ======================================================================
# GRADIENT REVERSAL LAYER
# ======================================================================

class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer.

    Parameters
    ----------
    lambda_grl : float
        Gradient reversal strength.

    Example
    -------
        grl = GradientReversalLayer(
            lambda_grl=1.0
        )

        reversed_features = grl(features)
    """

    def __init__(
        self,
        lambda_grl=None
    ):

        super().__init__()

        # --------------------------------------------------------------
        # Get default lambda from Config
        # --------------------------------------------------------------

        if lambda_grl is None:

            lambda_grl = getattr(
                Config,
                "LAMBDA_ADV",
                1.0
            )

        # --------------------------------------------------------------
        # Validate lambda
        # --------------------------------------------------------------

        try:

            lambda_grl = float(
                lambda_grl
            )

        except (
            TypeError,
            ValueError
        ):

            raise ValueError(
                "lambda_grl must be a numeric value."
            )

        if not math.isfinite(
            lambda_grl
        ):

            raise ValueError(
                "lambda_grl must be finite."
            )

        if lambda_grl < 0.0:

            raise ValueError(
                "lambda_grl must be >= 0."
            )

        # --------------------------------------------------------------
        # Store lambda
        # --------------------------------------------------------------

        self.lambda_grl = lambda_grl

    # ==================================================================
    # FORWARD
    # ==================================================================

    def forward(
        self,
        x
    ):

        if not torch.is_tensor(x):

            raise TypeError(
                "GradientReversalLayer expects "
                "a torch.Tensor."
            )

        return GradientReverseFunction.apply(
            x,
            self.lambda_grl
        )

    # ==================================================================
    # SET LAMBDA
    # ==================================================================

    def set_lambda(
        self,
        value
    ):

        """
        Dynamically update lambda.

        Useful during DANN training.

        Example:
            grl.set_lambda(0.5)
        """

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            raise ValueError(
                "GRL lambda must be numeric."
            )

        if not math.isfinite(
            value
        ):

            raise ValueError(
                "GRL lambda must be finite."
            )

        if value < 0.0:

            raise ValueError(
                "GRL lambda must be >= 0."
            )

        self.lambda_grl = value

    # ==================================================================
    # GET LAMBDA
    # ==================================================================

    def get_lambda(self):

        return float(
            self.lambda_grl
        )

    # ==================================================================
    # EXTRA REPRESENTATION
    # ==================================================================

    def extra_repr(self):

        return (
            f"lambda_grl={self.lambda_grl:.6f}"
        )


# ======================================================================
# DANN LAMBDA SCHEDULER
# ======================================================================

class GRLScheduler:
    """
    DANN Gradient Reversal Lambda Scheduler.

    Formula:

        λ(p) = 2 / (1 + exp(-γp)) - 1

    where:

        p ∈ [0, 1]

    Default:
        γ = 10

    Therefore:

        p = 0.00  -> λ ≈ 0.000
        p = 0.25  -> λ ≈ 0.848
        p = 0.50  -> λ ≈ 0.987
        p = 0.75  -> λ ≈ 0.999
        p = 1.00  -> λ ≈ 1.000
    """

    def __init__(
        self,
        gamma=10.0,
        max_lambda=1.0,
        min_lambda=0.0
    ):

        self.gamma = float(
            gamma
        )

        self.max_lambda = float(
            max_lambda
        )

        self.min_lambda = float(
            min_lambda
        )

        if self.gamma <= 0:

            raise ValueError(
                "gamma must be > 0."
            )

        if self.max_lambda < self.min_lambda:

            raise ValueError(
                "max_lambda must be >= min_lambda."
            )

    # ==================================================================
    # GET LAMBDA
    # ==================================================================

    def get_lambda(
        self,
        progress
    ):

        """
        Calculate GRL lambda.

        Parameters
        ----------
        progress : float
            Training progress in [0, 1].

        Returns
        -------
        float
            Scheduled lambda.
        """

        progress = float(
            progress
        )

        # --------------------------------------------------------------
        # Clamp progress
        # --------------------------------------------------------------

        progress = max(
            0.0,
            min(
                progress,
                1.0
            )
        )

        # --------------------------------------------------------------
        # DANN equation
        # --------------------------------------------------------------

        value = (
            2.0
            /
            (
                1.0
                +
                math.exp(
                    -self.gamma
                    *
                    progress
                )
            )
            -
            1.0
        )

        # --------------------------------------------------------------
        # Scale to configured range
        # --------------------------------------------------------------

        lambda_value = (
            self.min_lambda
            +
            (
                self.max_lambda
                -
                self.min_lambda
            )
            *
            value
        )

        return float(
            lambda_value
        )

    # ==================================================================
    # UPDATE GRL
    # ==================================================================

    def step(
        self,
        grl,
        progress
    ):

        """
        Calculate lambda and directly update GRL.

        Example:

            scheduler.step(
                grl,
                epoch / total_epochs
            )
        """

        if not isinstance(
            grl,
            GradientReversalLayer
        ):

            raise TypeError(
                "scheduler.step() expects "
                "GradientReversalLayer."
            )

        lambda_value = self.get_lambda(
            progress
        )

        grl.set_lambda(
            lambda_value
        )

        return lambda_value


# ======================================================================
# FACTORY
# ======================================================================

def build_grl(
    lambda_grl=None
):

    """
    Build a Gradient Reversal Layer.

    If lambda_grl is not supplied,
    Config.LAMBDA_ADV is used.
    """

    return GradientReversalLayer(
        lambda_grl=lambda_grl
    )


# ======================================================================
# HELPER
# ======================================================================

def get_grl_lambda(
    progress,
    gamma=10.0
):

    """
    Standalone helper for obtaining
    DANN lambda.

    Example:

        lambda_value = get_grl_lambda(
            progress=0.5
        )
    """

    scheduler = GRLScheduler(
        gamma=gamma
    )

    return scheduler.get_lambda(
        progress
    )


# ======================================================================
# TEST GRADIENT REVERSAL
# ======================================================================

def test_gradient_reversal():

    print()
    print("=" * 78)
    print("TESTING GRADIENT REVERSAL LAYER")
    print("=" * 78)

    # --------------------------------------------------------------
    # Device
    # --------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device : {device}"
    )

    # --------------------------------------------------------------
    # Lambda
    # --------------------------------------------------------------

    lambda_value = 1.0

    # --------------------------------------------------------------
    # Build GRL
    # --------------------------------------------------------------

    grl = build_grl(
        lambda_grl=lambda_value
    ).to(
        device
    )

    # --------------------------------------------------------------
    # Input
    # --------------------------------------------------------------

    feature_dim = int(
        getattr(
            Config,
            "FEATURE_DIM",
            256
        )
    )

    x = torch.randn(
        4,
        feature_dim,
        device=device,
        requires_grad=True
    )

    # --------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------

    y = grl(
        x
    )

    # --------------------------------------------------------------
    # Check forward identity
    # --------------------------------------------------------------

    forward_difference = (
        y - x
    ).abs().max().item()

    print()
    print(
        "Forward Test"
    )

    print(
        "-" * 78
    )

    print(
        f"Input Shape       : {tuple(x.shape)}"
    )

    print(
        f"Output Shape      : {tuple(y.shape)}"
    )

    print(
        f"Maximum Difference: {forward_difference:.10f}"
    )

    assert torch.allclose(
        x,
        y
    ), "GRL forward pass is not identity."

    # --------------------------------------------------------------
    # Backward
    # --------------------------------------------------------------

    loss = y.mean()

    loss.backward()

    # --------------------------------------------------------------
    # Gradient
    # --------------------------------------------------------------

    gradient = x.grad

    if gradient is None:

        raise RuntimeError(
            "GRL gradient is None."
        )

    print()
    print(
        "Backward Test"
    )

    print(
        "-" * 78
    )

    print(
        f"Gradient Shape : {tuple(gradient.shape)}"
    )

    print(
        f"Gradient Mean  : {gradient.mean().item():.8f}"
    )

    print(
        f"Gradient Std   : {gradient.std().item():.8f}"
    )

    print(
        f"Lambda         : {grl.lambda_grl:.6f}"
    )

    # --------------------------------------------------------------
    # Expected gradient
    #
    # loss = mean(y)
    #
    # Normal gradient = +1/N
    #
    # GRL gradient = -lambda/N
    # --------------------------------------------------------------

    expected_gradient = (
        -lambda_value
        /
        x.numel()
    )

    actual_gradient = (
        gradient.mean().item()
    )

    print(
        f"Expected Grad  : {expected_gradient:.8f}"
    )

    print(
        f"Actual Grad    : {actual_gradient:.8f}"
    )

    assert actual_gradient < 0.0, (
        "Gradient was not reversed."
    )

    print()
    print(
        "[PASS] Gradient reversal test."
    )

    print("=" * 78)


# ======================================================================
# TEST LAMBDA SCHEDULER
# ======================================================================

def test_scheduler():

    print()
    print("=" * 78)
    print("TESTING GRL LAMBDA SCHEDULER")
    print("=" * 78)

    scheduler = GRLScheduler(
        gamma=10.0,
        min_lambda=0.0,
        max_lambda=1.0
    )

    progress_values = [
        0.00,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        1.00
    ]

    print()
    print(
        f"{'Progress':<15}"
        f"{'Lambda':<15}"
    )

    print(
        "-" * 30
    )

    previous = -1.0

    for progress in progress_values:

        lambda_value = scheduler.get_lambda(
            progress
        )

        print(
            f"{progress:<15.2f}"
            f"{lambda_value:<15.6f}"
        )

        # ----------------------------------------------------------
        # Lambda should be monotonically increasing
        # ----------------------------------------------------------

        assert lambda_value >= previous

        previous = lambda_value

        # ----------------------------------------------------------
        # Lambda should stay within range
        # ----------------------------------------------------------

        assert (
            0.0
            <=
            lambda_value
            <=
            1.0
        )

    print()
    print(
        "[PASS] Lambda scheduler test."
    )

    print("=" * 78)


# ======================================================================
# TEST DYNAMIC LAMBDA UPDATE
# ======================================================================

def test_dynamic_lambda():

    print()
    print("=" * 78)
    print("TESTING DYNAMIC GRL UPDATE")
    print("=" * 78)

    grl = build_grl(
        lambda_grl=0.0
    )

    scheduler = GRLScheduler()

    progress = 0.50

    lambda_value = scheduler.step(
        grl,
        progress
    )

    print(
        f"Progress       : {progress:.2f}"
    )

    print(
        f"Scheduled λ    : {lambda_value:.6f}"
    )

    print(
        f"GRL λ          : {grl.get_lambda():.6f}"
    )

    assert abs(
        grl.get_lambda()
        -
        lambda_value
    ) < 1e-8

    print()
    print(
        "[PASS] Dynamic lambda update test."
    )

    print("=" * 78)


# ======================================================================
# MAIN TEST
# ======================================================================

if __name__ == "__main__":

    print()
    print("=" * 78)
    print("              SICDA GRL MODULE TEST")
    print("=" * 78)

    # --------------------------------------------------------------
    # Run gradient test
    # --------------------------------------------------------------

    test_gradient_reversal()

    # --------------------------------------------------------------
    # Run scheduler test
    # --------------------------------------------------------------

    test_scheduler()

    # --------------------------------------------------------------
    # Run dynamic update test
    # --------------------------------------------------------------

    test_dynamic_lambda()

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    print()
    print("=" * 78)
    print("              ALL GRL TESTS PASSED")
    print("=" * 78)