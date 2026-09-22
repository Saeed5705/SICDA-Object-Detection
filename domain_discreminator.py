import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

# ============================================================
# Domain Classifier
# ============================================================

class DomainClassifier(nn.Module):
    """
    Binary Domain Classifier

    0 -> Source (COCO)

    1 -> Target (FLIR)
    """

    def __init__(
            self,
            in_channels=Config.FEATURE_DIM,
            hidden_dim=Config.DOMAIN_HIDDEN):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(

            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, feature):
        x = self.pool(feature)
        x = x.flatten(1)
        logits = self.classifier(x)
        return logits


# ============================================================
# Multi-Level Domain Discriminator
# ============================================================

class MultiScaleDomainDiscriminator(nn.Module):
    """
    Domain Classification for P2, P3, P4, P5
    """

    def __init__(self, channels=Config.FEATURE_DIM):
        super().__init__()

        self.discriminators = nn.ModuleDict({
            "p2": DomainClassifier(channels),
            "p3": DomainClassifier(channels),
            "p4": DomainClassifier(channels),
            "p5": DomainClassifier(channels)
        })

    def forward(self, features):
        outputs = {}
        required_levels = ["p2", "p3", "p4", "p5"]

        for level in required_levels:
            if level not in features:
                raise KeyError(f"{level} missing in feature dictionary")
            outputs[level] = self.discriminators[level](features[level])

        return outputs


# ============================================================
# Domain Loss
# ============================================================

class DomainLoss(nn.Module):
    """
    Cross Entropy Domain Loss
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, predictions, labels):
        total_loss = 0.0
        for level in predictions:
            total_loss += self.loss(predictions[level], labels)

        if len(predictions) == 0:
            raise ValueError("No Domain predictions found.")

        total_loss /= len(predictions)
        return total_loss


def build_domain_discriminator():
    return MultiScaleDomainDiscriminator()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_domain_discriminator().to(device)

    features = {
        "p2": torch.randn(4, Config.FEATURE_DIM, 200, 200).to(device),
        "p3": torch.randn(4, Config.FEATURE_DIM, 100, 100).to(device),
        "p4": torch.randn(4, Config.FEATURE_DIM, 50, 50).to(device),
        "p5": torch.randn(4, Config.FEATURE_DIM, 25, 25).to(device)
    }

    outputs = model(features)
    labels = torch.tensor([0, 0, 1, 1], device=device)
    criterion = DomainLoss()
    loss = criterion(outputs, labels)

    print("=" * 60)
    print("Domain Discriminator")
    print("=" * 60)
    for level in outputs:
        print(level, outputs[level].shape)
    print("Loss :", loss.item())
    print("=" * 60)