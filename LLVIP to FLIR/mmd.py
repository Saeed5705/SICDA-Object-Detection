

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Gaussian Kernel
# ============================================================

def gaussian_kernel(
        source,
        target,
        kernel_mul=2.0,
        kernel_num=5,
        fix_sigma=None):

    """
    Compute Gaussian Kernel Matrix
    """

    n_samples = source.size(0) + target.size(0)

    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(

        total.size(0),

        total.size(0),

        total.size(1)

    )

    total1 = total.unsqueeze(1).expand(

        total.size(0),

        total.size(0),

        total.size(1)

    )

    l2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:

        bandwidth = fix_sigma

    else:

        bandwidth = torch.sum(l2_distance.detach()) / (

            n_samples ** 2 - n_samples

        )
        bandwidth=torch.clamp(
            bandwidth,
            min=1e-6
        )

    bandwidth /= kernel_mul ** (kernel_num // 2)

    bandwidth_list = [

        bandwidth * (kernel_mul ** i)

        for i in range(kernel_num)

    ]

    kernel_val = [

        torch.exp(-l2_distance / bandwidth_temp)

        for bandwidth_temp in bandwidth_list

    ]

    return sum(kernel_val)


# ============================================================
# Single Scale MMD
# ============================================================

class MMDLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(
            self,
            source,
            target):

        kernels = gaussian_kernel(

            source,

            target

        )

        batch_size = source.size(0)

        XX = kernels[:batch_size, :batch_size]

        YY = kernels[batch_size:, batch_size:]

        XY = kernels[:batch_size, batch_size:]

        YX = kernels[batch_size:, :batch_size]

        loss = torch.mean(

            XX +

            YY -

            XY -

            YX

        )

        return loss


# ============================================================
# Feature Pooling
# ============================================================

class FeaturePooling(nn.Module):

    def __init__(self):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feature):

        x = self.pool(feature)

        x = x.flatten(1)

        return x


# ============================================================
# Multi Scale MMD
# ============================================================

class MultiScaleMMD(nn.Module):

    """
    MMD for

    P2

    P3

    P4

    P5
    """

    def __init__(self):

        super().__init__()

        self.pool = FeaturePooling()

        self.loss = MMDLoss()

    ###########################################################

    def forward(

            self,

            source_features,

            target_features

    ):

        total_loss = 0.0

        losses = {}
        levels=["p2", "p3", "p4","p5"]
        for level in levels:
            if level not in source_features:
                raise KeyError(f"{level} missing in source features.")
            if level not in target_features:
                raise KeyError(f"{level} missing in target features.")
                
            src = self.pool(

                source_features[level]

            )

            tgt = self.pool(

                target_features[level]

            )

            loss = self.loss(

                src,

                tgt

            )

            losses[level] = loss

            total_loss += loss

        total_loss /= 4

        return total_loss, losses


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

    model = MultiScaleMMD().to(device)

    source = {

        "p2": torch.randn(4,256,200,200).to(device),

        "p3": torch.randn(4,256,100,100).to(device),

        "p4": torch.randn(4,256,50,50).to(device),

        "p5": torch.randn(4,256,25,25).to(device)

    }

    target = {

        "p2": torch.randn(4,256,200,200).to(device),

        "p3": torch.randn(4,256,100,100).to(device),

        "p4": torch.randn(4,256,50,50).to(device),

        "p5": torch.randn(4,256,25,25).to(device)

    }

    loss, losses = model(

        source,

        target

    )

    print("=" * 60)

    print("Multi Scale MMD")

    print("=" * 60)

    print("Total Loss :", loss.item())

    print()

    for k, v in losses.items():

        print(

            k,

            v.item()

        )

    print("=" * 60)