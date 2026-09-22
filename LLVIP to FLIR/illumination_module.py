
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

# ============================================================
# Illumination Estimator
# ============================================================

class IlluminationEstimator(nn.Module):
    """
    Estimate illumination reliability α.

    Output:
        alpha ∈ [0,1]
    """

    def __init__(self, channels=Config.FEATURE_DIM):

        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(channels, channels // 4),

            nn.ReLU(inplace=True),

            nn.Linear(channels // 4, 1),

            nn.Sigmoid()

        )

    def forward(self, rgb_feature):

        B, C, _, _ = rgb_feature.shape

        x = self.global_pool(rgb_feature)

        x = x.view(B, C)

        alpha = self.fc(x)

        return alpha


# ============================================================
# Adaptive Fusion
# ============================================================

class AdaptiveFusion(nn.Module):
    """
    F_adaptive =
        α * RGB
        +
        (1-α) * Thermal
    """

    def forward(
            self,
            rgb_feature,
            thermal_feature,
            alpha):

        alpha = alpha.view(-1, 1, 1, 1)

        fused = (

            alpha * rgb_feature +

            (1.0 - alpha) * thermal_feature

        )

        return fused


# ============================================================
# Illumination Module
# ============================================================

class IlluminationAwareFusion(nn.Module):

    """
    Performs

    1. Illumination estimation

    2. Adaptive fusion
    """

    def __init__(self, channels=Config.FEATURE_DIM):

        super().__init__()

        self.estimator = IlluminationEstimator(

            channels

        )

        self.fusion = AdaptiveFusion()

    ###########################################################

    def forward(
            self,
            rgb_features,
            thermal_features):

        fused_features = {}

        alpha_scores = {}

        for level in rgb_features.keys():
            if level not in thermal_features:
                raise KeyError(f"{level} not found in thermal features")
            alpha = self.estimator(

                rgb_features[level]

            )

            fused = self.fusion(

                rgb_features[level],

                thermal_features[level],

                alpha

            )

            fused_features[level] = fused

            alpha_scores[level] = alpha

        
        return fused_features, alpha_scores
        


# ============================================================
# Factory
# ============================================================

def build_illumination_module():

    return IlluminationAwareFusion()


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    model = build_illumination_module().to(device)

    rgb = {

        "p2": torch.randn(2, Config.FEATURE_DIM,200,200).to(device),

        "p3": torch.randn(2,Config.FEATURE_DIM,100,100).to(device),

        "p4": torch.randn(2,Config.FEATURE_DIM,50,50).to(device),

        "p5": torch.randn(2,Config.FEATURE_DIM,25,25).to(device)

    }

    thermal = {

        "p2": torch.randn(2,Config.FEATURE_DIM,200,200).to(device),

        "p3": torch.randn(2,Config.FEATURE_DIM,100,100).to(device),

        "p4": torch.randn(2,Config.FEATURE_DIM,50,50).to(device),

        "p5": torch.randn(2,Config.FEATURE_DIM,25,25).to(device)

    }

    fused, alpha = model(

        rgb,

        thermal

    )

    print()

    print("=" * 60)

    print("Illumination Module")

    print("=" * 60)

    for level in fused:

        print(

            level,

            fused[level].shape,

            alpha[level].shape

        )