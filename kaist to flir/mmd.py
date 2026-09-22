"""
==============================================================
File : mmd.py

SICDA Multi-Scale Maximum Mean Discrepancy Loss

Purpose:
    Align source and target feature distributions.

Input:
    FPN features from P2, P3, P4, P5

Supported feature formats:
    [B, C, H, W]
    [B, C]

Memory Optimization:
    Global Average Pooling is applied before
    Gaussian-kernel MMD.

Batch Size:
    Supports BATCH_SIZE = 1.

==============================================================
"""

import torch
import torch.nn as nn


# ============================================================
# Gaussian Kernel
# ============================================================

def gaussian_kernel(
        source,
        target,
        kernel_mul=2.0,
        kernel_num=5,
        fix_sigma=None
):
    """
    Multi-kernel Gaussian RBF.

    Input:
        source: [B, C]
        target: [B, C]

    Output:
        Kernel matrix: [2B, 2B]
    """

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if source.ndim != 2:
        raise ValueError(
            f"Source must be [B,C], "
            f"got {tuple(source.shape)}"
        )

    if target.ndim != 2:
        raise ValueError(
            f"Target must be [B,C], "
            f"got {tuple(target.shape)}"
        )

    if source.size(1) != target.size(1):
        raise ValueError(
            f"Feature dimension mismatch: "
            f"{source.shape} vs {target.shape}"
        )

    # --------------------------------------------------------
    # Concatenate source and target
    # --------------------------------------------------------

    total = torch.cat(
        [
            source,
            target
        ],
        dim=0
    )

    total_size = total.size(0)

    # --------------------------------------------------------
    # Pairwise squared Euclidean distance
    #
    # [N,C]
    #     ->
    # [N,N,C]
    # --------------------------------------------------------

    total0 = total.unsqueeze(0)

    total1 = total.unsqueeze(1)

    L2_distance = (
        (total0 - total1) ** 2
    ).sum(dim=2)

    # --------------------------------------------------------
    # Bandwidth
    # --------------------------------------------------------

    if fix_sigma is not None:

        bandwidth = torch.as_tensor(
            fix_sigma,
            dtype=total.dtype,
            device=total.device
        )

    else:

        if total_size <= 1:

            bandwidth = torch.tensor(
                1.0,
                dtype=total.dtype,
                device=total.device
            )

        else:

            bandwidth = (
                torch.sum(
                    L2_distance.detach()
                )
                /
                (
                    total_size ** 2
                    - total_size
                )
            )

    # --------------------------------------------------------
    # Numerical stability
    # --------------------------------------------------------

    bandwidth = torch.clamp(
        bandwidth,
        min=1e-6
    )

    bandwidth = (
        bandwidth
        /
        (
            kernel_mul
            **
            (kernel_num // 2)
        )
    )

    # --------------------------------------------------------
    # Multi-kernel Gaussian RBF
    # --------------------------------------------------------

    kernel_list = []

    for i in range(kernel_num):

        bandwidth_i = (
            bandwidth
            *
            (
                kernel_mul ** i
            )
        )

        bandwidth_i = torch.clamp(
            bandwidth_i,
            min=1e-6
        )

        kernel = torch.exp(
            -L2_distance
            /
            bandwidth_i
        )

        kernel_list.append(
            kernel
        )

    return sum(kernel_list)


# ============================================================
# Single Scale MMD
# ============================================================

class MMDLoss(nn.Module):

    def __init__(
            self,
            kernel_mul=2.0,
            kernel_num=5,
            fix_sigma=None
    ):

        super().__init__()

        self.kernel_mul = kernel_mul

        self.kernel_num = kernel_num

        self.fix_sigma = fix_sigma

    # --------------------------------------------------------

    def forward(
            self,
            source,
            target
    ):

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if source.ndim != 2:

            raise ValueError(
                f"MMD source must be [B,C], "
                f"got {tuple(source.shape)}"
            )

        if target.ndim != 2:

            raise ValueError(
                f"MMD target must be [B,C], "
                f"got {tuple(target.shape)}"
            )

        if source.size(0) != target.size(0):

            raise ValueError(
                f"Source and target batch sizes "
                f"must match for MMD: "
                f"{source.shape} vs {target.shape}"
            )

        if source.size(1) != target.size(1):

            raise ValueError(
                f"Source and target feature dimensions "
                f"must match: "
                f"{source.shape} vs {target.shape}"
            )

        # ----------------------------------------------------
        # Gaussian kernel
        # ----------------------------------------------------

        kernels = gaussian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma
        )

        batch_size = source.size(0)

        # ----------------------------------------------------
        # Source-source
        # ----------------------------------------------------

        XX = kernels[
            :batch_size,
            :batch_size
        ]

        # ----------------------------------------------------
        # Target-target
        # ----------------------------------------------------

        YY = kernels[
            batch_size:,
            batch_size:
        ]

        # ----------------------------------------------------
        # Source-target
        # ----------------------------------------------------

        XY = kernels[
            :batch_size,
            batch_size:
        ]

        # ----------------------------------------------------
        # Target-source
        # ----------------------------------------------------

        YX = kernels[
            batch_size:,
            :batch_size
        ]

        # ----------------------------------------------------
        # MMD
        # ----------------------------------------------------

        loss = torch.mean(
            XX
            + YY
            - XY
            - YX
        )

        # ----------------------------------------------------
        # Numerical protection
        # ----------------------------------------------------

        loss = torch.nan_to_num(
            loss,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4
        )

        return loss


# ============================================================
# Feature Pooling
# ============================================================

class FeaturePooling(nn.Module):

    """
    Convert FPN features into compact representations.

    Supported input:

        [B,C,H,W]

    Output:

        [B,C]

    Also accepts:

        [B,C]

    and returns it unchanged.
    """

    def __init__(self):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(
            output_size=1
        )

    # --------------------------------------------------------

    def forward(self, x):

        # ----------------------------------------------------
        # Case 1:
        #
        # Standard FPN feature
        #
        # [B,C,H,W]
        # ----------------------------------------------------

        if x.ndim == 4:

            x = self.pool(x)

            x = torch.flatten(
                x,
                start_dim=1
            )

            return x

        # ----------------------------------------------------
        # Case 2:
        #
        # Already pooled feature
        #
        # [B,C]
        # ----------------------------------------------------

        if x.ndim == 2:

            return x

        # ----------------------------------------------------
        # Invalid dimension
        # ----------------------------------------------------

        raise ValueError(
            "Feature must be either "
            "[B,C,H,W] or [B,C], "
            f"got {tuple(x.shape)}"
        )


# ============================================================
# Multi Scale MMD
# ============================================================

class MultiScaleMMD(nn.Module):

    """
    Multi-scale MMD applied to:

        P2
        P3
        P4
        P5

    Supports:

        Source feature:
            [B,C,H,W]

        Target feature:
            [B,C,H,W]

    OR:

        Source feature:
            [B,C]

        Target feature:
            [B,C]

    This makes the module compatible with the
    memory-efficient trainer.
    """

    def __init__(
            self,
            levels=None,
            kernel_mul=2.0,
            kernel_num=5,
            fix_sigma=None
    ):

        super().__init__()

        # ----------------------------------------------------
        # Feature pooling
        # ----------------------------------------------------

        self.pool = FeaturePooling()

        # ----------------------------------------------------
        # Single-scale MMD
        # ----------------------------------------------------

        self.mmd = MMDLoss(
            kernel_mul=kernel_mul,
            kernel_num=kernel_num,
            fix_sigma=fix_sigma
        )

        # ----------------------------------------------------
        # FPN levels
        # ----------------------------------------------------

        if levels is None:

            self.levels = [
                "p2",
                "p3",
                "p4",
                "p5"
            ]

        else:

            self.levels = levels

    # --------------------------------------------------------

    def forward(
            self,
            source_features,
            target_features
    ):

        # ----------------------------------------------------
        # Validate dictionaries
        # ----------------------------------------------------

        if not isinstance(
                source_features,
                dict
        ):

            raise TypeError(
                "source_features must be a dictionary"
            )

        if not isinstance(
                target_features,
                dict
        ):

            raise TypeError(
                "target_features must be a dictionary"
            )

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        total_loss = torch.zeros(
            (),
            device=next(
                iter(
                    source_features.values()
                )
            ).device
        )

        loss_dict = {}

        valid_levels = 0

        # ====================================================
        # Process each FPN level
        # ====================================================

        for level in self.levels:

            # ------------------------------------------------
            # Check source level
            # ------------------------------------------------

            if level not in source_features:

                raise KeyError(
                    f"{level} missing in source features. "
                    f"Available: "
                    f"{list(source_features.keys())}"
                )

            # ------------------------------------------------
            # Check target level
            # ------------------------------------------------

            if level not in target_features:

                raise KeyError(
                    f"{level} missing in target features. "
                    f"Available: "
                    f"{list(target_features.keys())}"
                )

            source_feature = (
                source_features[level]
            )

            target_feature = (
                target_features[level]
            )

            # ------------------------------------------------
            # Ignore None
            # ------------------------------------------------

            if source_feature is None:

                continue

            if target_feature is None:

                continue

            # ------------------------------------------------
            # Pool source
            #
            # [B,C,H,W] -> [B,C]
            # [B,C]     -> [B,C]
            # ------------------------------------------------

            source = self.pool(
                source_feature
            )

            # ------------------------------------------------
            # Pool target
            # ------------------------------------------------

            target = self.pool(
                target_feature
            )

            # ------------------------------------------------
            # Check batch size
            # ------------------------------------------------

            if source.size(0) != target.size(0):

                raise ValueError(
                    f"{level}: source and target "
                    f"batch sizes differ: "
                    f"{source.shape} vs "
                    f"{target.shape}"
                )

            # ------------------------------------------------
            # Check feature dimension
            # ------------------------------------------------

            if source.size(1) != target.size(1):

                raise ValueError(
                    f"{level}: source and target "
                    f"feature dimensions differ: "
                    f"{source.shape} vs "
                    f"{target.shape}"
                )

            # ------------------------------------------------
            # MMD
            # ------------------------------------------------

            level_loss = self.mmd(
                source,
                target
            )

            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            loss_dict[level] = level_loss

            total_loss = (
                total_loss
                + level_loss
            )

            valid_levels += 1

        # ====================================================
        # No valid levels
        # ====================================================

        if valid_levels == 0:

            zero = torch.zeros(
                (),
                device=total_loss.device,
                requires_grad=True
            )

            return zero, loss_dict

        # ====================================================
        # Average over FPN levels
        # ====================================================

        total_loss = (
            total_loss
            /
            valid_levels
        )

        return total_loss, loss_dict


# ============================================================
# Factory
# ============================================================

def build_mmd():

    return MultiScaleMMD()


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)

    print(
        "SICDA Multi-Scale MMD Test"
    )

    print("=" * 70)

    print(
        "Device:",
        device
    )

    # ========================================================
    # Test 1
    #
    # Original FPN features
    # [B,C,H,W]
    # ========================================================

    print()
    print("-" * 70)
    print("Test 1: FPN [B,C,H,W]")
    print("-" * 70)

    model = build_mmd().to(device)

    source = {

        "p2": torch.randn(
            1,
            256,
            100,
            100,
            device=device
        ),

        "p3": torch.randn(
            1,
            256,
            50,
            50,
            device=device
        ),

        "p4": torch.randn(
            1,
            256,
            25,
            25,
            device=device
        ),

        "p5": torch.randn(
            1,
            256,
            13,
            13,
            device=device
        )
    }

    target = {

        "p2": torch.randn(
            1,
            256,
            100,
            100,
            device=device
        ),

        "p3": torch.randn(
            1,
            256,
            50,
            50,
            device=device
        ),

        "p4": torch.randn(
            1,
            256,
            25,
            25,
            device=device
        ),

        "p5": torch.randn(
            1,
            256,
            13,
            13,
            device=device
        )
    }

    loss, losses = model(
        source,
        target
    )

    print(
        "Total MMD:",
        loss.item()
    )

    for key, value in losses.items():

        print(
            f"{key}: {value.item():.8f}"
        )

    print(
        "Test 1 Passed"
    )

    # ========================================================
    # Test 2
    #
    # Already pooled features
    # [B,C]
    #
    # This is the format currently produced by
    # trainer.py.
    # ========================================================

    print()
    print("-" * 70)
    print("Test 2: Already pooled [B,C]")
    print("-" * 70)

    source_pooled = {

        "p2": torch.randn(
            1,
            256,
            device=device
        ),

        "p3": torch.randn(
            1,
            256,
            device=device
        ),

        "p4": torch.randn(
            1,
            256,
            device=device
        ),

        "p5": torch.randn(
            1,
            256,
            device=device
        )
    }

    target_pooled = {

        "p2": torch.randn(
            1,
            256,
            device=device
        ),

        "p3": torch.randn(
            1,
            256,
            device=device
        ),

        "p4": torch.randn(
            1,
            256,
            device=device
        ),

        "p5": torch.randn(
            1,
            256,
            device=device
        )
    }

    loss_pooled, losses_pooled = model(
        source_pooled,
        target_pooled
    )

    print(
        "Total MMD:",
        loss_pooled.item()
    )

    for key, value in losses_pooled.items():

        print(
            f"{key}: {value.item():.8f}"
        )

    print(
        "Test 2 Passed"
    )

    # ========================================================
    # Gradient Test
    # ========================================================

    print()
    print("-" * 70)
    print("Test 3: Gradient Flow")
    print("-" * 70)

    source_grad = {

        "p2": torch.randn(
            1,
            256,
            requires_grad=True,
            device=device
        ),

        "p3": torch.randn(
            1,
            256,
            requires_grad=True,
            device=device
        ),

        "p4": torch.randn(
            1,
            256,
            requires_grad=True,
            device=device
        ),

        "p5": torch.randn(
            1,
            256,
            requires_grad=True,
            device=device
        )
    }

    target_grad = {

        "p2": torch.randn(
            1,
            256,
            requires_grad=True,
            device=device
        ),

        "p3": torch.randn(
            1,
            256,
            requires_grad=True,
            device=device
        ),

        "p4": torch.randn(
            1,
            256,
            requires_grad=True,
            device=device
        ),

        "p5": torch.randn(
            1,
            256,
            requires_grad=True,
            device=device
        )
    }

    grad_loss, _ = model(
        source_grad,
        target_grad
    )

    grad_loss.backward()

    print(
        "Gradient test passed"
    )

    print(
        "P2 source gradient:",
        source_grad["p2"].grad is not None
    )

    print(
        "P2 target gradient:",
        target_grad["p2"].grad is not None
    )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 70)

    print(
        "SICDA MMD TEST PASSED"
    )

    print("=" * 70)