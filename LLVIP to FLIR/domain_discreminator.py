"""
SICDA Domain Discriminator
===========================

Purpose:
    Multi-scale adversarial domain classification for SICDA.

Domains:
    0 -> Source Domain (LLVIP)
    1 -> Target Domain (FLIR)

Feature Levels:
    P2
    P3
    P4
    P5

Pipeline:

    Source / Target Features
            |
            v
        GRL Layer
            |
            v
    Multi-Scale Domain Discriminator
            |
            v
      Domain Logits
            |
            v
      Cross Entropy Loss

Important:
    Gradient Reversal Layer is NOT permanently included inside
    this discriminator.

    Recommended usage:

        reversed_features = grl(features)
        domain_logits = discriminator(reversed_features)

    This keeps the discriminator reusable and makes lambda scheduling
    easier during SICDA training.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# ======================================================================
# CONFIGURATION HELPERS
# ======================================================================

def _get_config(
    name,
    default
):
    """
    Safely obtain a value from Config.

    This prevents the discriminator from crashing if an optional
    configuration value does not exist.
    """

    value = getattr(
        Config,
        name,
        default
    )

    return value


FEATURE_DIM = int(
    _get_config(
        "FEATURE_DIM",
        256
    )
)

DOMAIN_HIDDEN = int(
    _get_config(
        "DOMAIN_HIDDEN",
        256
    )
)

DOMAIN_DROPOUT = float(
    _get_config(
        "DOMAIN_DROPOUT",
        0.5
    )
)

NUM_DOMAIN_CLASSES = int(
    _get_config(
        "NUM_DOMAIN_CLASSES",
        2
    )
)


# ======================================================================
# DOMAIN CLASSIFIER
# ======================================================================

class DomainClassifier(nn.Module):
    """
    Binary Domain Classifier.

    Domain labels:

        0 -> Source / LLVIP
        1 -> Target / FLIR

    Input:
        [B, C, H, W]

    Output:
        [B, 2]

    The spatial dimensions can be different for P2, P3, P4 and P5
    because AdaptiveAvgPool2d(1) is used.
    """

    def __init__(
        self,
        in_channels=FEATURE_DIM,
        hidden_dim=DOMAIN_HIDDEN,
        num_domains=NUM_DOMAIN_CLASSES,
        dropout=DOMAIN_DROPOUT
    ):

        super().__init__()

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if in_channels <= 0:

            raise ValueError(
                "in_channels must be > 0."
            )

        if hidden_dim <= 0:

            raise ValueError(
                "hidden_dim must be > 0."
            )

        if num_domains < 2:

            raise ValueError(
                "num_domains must be >= 2."
            )

        if not 0.0 <= dropout < 1.0:

            raise ValueError(
                "dropout must be in [0, 1)."
            )

        # --------------------------------------------------------------
        # Store configuration
        # --------------------------------------------------------------

        self.in_channels = int(
            in_channels
        )

        self.hidden_dim = int(
            hidden_dim
        )

        self.num_domains = int(
            num_domains
        )

        self.dropout_rate = float(
            dropout
        )

        # --------------------------------------------------------------
        # Global Average Pooling
        # --------------------------------------------------------------

        self.pool = nn.AdaptiveAvgPool2d(
            output_size=1
        )

        # --------------------------------------------------------------
        # Hidden dimensions
        # --------------------------------------------------------------

        hidden_dim_2 = max(
            hidden_dim // 2,
            1
        )

        # --------------------------------------------------------------
        # Domain classifier
        # --------------------------------------------------------------

        self.classifier = nn.Sequential(

            # ----------------------------------------------------------
            # Block 1
            # ----------------------------------------------------------

            nn.Linear(
                in_channels,
                hidden_dim
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                dropout
            ),

            # ----------------------------------------------------------
            # Block 2
            # ----------------------------------------------------------

            nn.Linear(
                hidden_dim,
                hidden_dim_2
            ),

            nn.LayerNorm(
                hidden_dim_2
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                dropout
            ),

            # ----------------------------------------------------------
            # Output
            # ----------------------------------------------------------

            nn.Linear(
                hidden_dim_2,
                num_domains
            )
        )

    # ==================================================================
    # FORWARD
    # ==================================================================

    def forward(
        self,
        feature
    ):

        # --------------------------------------------------------------
        # Input validation
        # --------------------------------------------------------------

        if not torch.is_tensor(
            feature
        ):

            raise TypeError(
                "DomainClassifier expects a torch.Tensor."
            )

        if feature.ndim != 4:

            raise ValueError(
                "DomainClassifier expects feature shape "
                "[B, C, H, W], got "
                f"{tuple(feature.shape)}"
            )

        if feature.shape[1] != self.in_channels:

            raise ValueError(
                "Feature channel mismatch. "
                f"Expected {self.in_channels}, "
                f"got {feature.shape[1]}."
            )

        # --------------------------------------------------------------
        # Global Average Pooling
        # --------------------------------------------------------------

        x = self.pool(
            feature
        )

        # [B, C, 1, 1]
        #       ->
        # [B, C]

        x = x.flatten(
            start_dim=1
        )

        # --------------------------------------------------------------
        # Domain classification
        # --------------------------------------------------------------

        logits = self.classifier(
            x
        )

        return logits

    # ==================================================================
    # PREDICT DOMAIN
    # ==================================================================

    @staticmethod
    def predict_domain(
        logits
    ):

        return torch.argmax(
            logits,
            dim=1
        )


# ======================================================================
# MULTI-SCALE DOMAIN DISCRIMINATOR
# ======================================================================

class MultiScaleDomainDiscriminator(nn.Module):
    """
    Multi-level domain discriminator.

    Inputs:

        features = {
            "p2": [B,C,H2,W2],
            "p3": [B,C,H3,W3],
            "p4": [B,C,H4,W4],
            "p5": [B,C,H5,W5]
        }

    Outputs:

        {
            "p2": [B,2],
            "p3": [B,2],
            "p4": [B,2],
            "p5": [B,2]
        }

    Each FPN level has an independent domain classifier.
    """

    REQUIRED_LEVELS = (
        "p2",
        "p3",
        "p4",
        "p5"
    )

    def __init__(
        self,
        channels=FEATURE_DIM,
        hidden_dim=DOMAIN_HIDDEN,
        num_domains=NUM_DOMAIN_CLASSES,
        dropout=DOMAIN_DROPOUT
    ):

        super().__init__()

        self.channels = int(
            channels
        )

        self.hidden_dim = int(
            hidden_dim
        )

        self.num_domains = int(
            num_domains
        )

        # --------------------------------------------------------------
        # One discriminator per FPN level
        # --------------------------------------------------------------

        self.discriminators = nn.ModuleDict({

            level: DomainClassifier(

                in_channels=channels,

                hidden_dim=hidden_dim,

                num_domains=num_domains,

                dropout=dropout

            )

            for level in self.REQUIRED_LEVELS

        })

    # ==================================================================
    # FORWARD
    # ==================================================================

    def forward(
        self,
        features
    ):

        if not isinstance(
            features,
            dict
        ):

            raise TypeError(
                "MultiScaleDomainDiscriminator expects "
                "a feature dictionary."
            )

        outputs = {}

        # --------------------------------------------------------------
        # Process P2-P5
        # --------------------------------------------------------------

        for level in self.REQUIRED_LEVELS:

            if level not in features:

                raise KeyError(
                    f"Required feature level '{level}' "
                    "is missing."
                )

            feature = features[
                level
            ]

            outputs[
                level
            ] = self.discriminators[
                level
            ](
                feature
            )

        return outputs

    # ==================================================================
    # PREDICT
    # ==================================================================

    @torch.no_grad()
    def predict(
        self,
        features
    ):

        logits = self.forward(
            features
        )

        predictions = {}

        for level, value in logits.items():

            predictions[
                level
            ] = torch.argmax(
                value,
                dim=1
            )

        return predictions


# ======================================================================
# DOMAIN LOSS
# ======================================================================

class DomainLoss(nn.Module):
    """
    Multi-scale domain classification loss.

    Uses Cross Entropy Loss independently on P2-P5 and averages
    the losses.

    Source:
        label = 0

    Target:
        label = 1
    """

    def __init__(
        self,
        reduction="mean"
    ):

        super().__init__()

        self.loss = nn.CrossEntropyLoss(
            reduction=reduction
        )

    # ==================================================================
    # FORWARD
    # ==================================================================

    def forward(
        self,
        predictions,
        labels
    ):

        if not isinstance(
            predictions,
            dict
        ):

            raise TypeError(
                "predictions must be a dictionary."
            )

        if len(predictions) == 0:

            raise ValueError(
                "No domain predictions found."
            )

        if not torch.is_tensor(
            labels
        ):

            labels = torch.as_tensor(
                labels,
                dtype=torch.long
            )

        labels = labels.long()

        total_loss = None

        level_losses = {}

        # --------------------------------------------------------------
        # Each FPN level
        # --------------------------------------------------------------

        for level, logits in predictions.items():

            if logits.ndim != 2:

                raise ValueError(
                    f"{level} logits must have shape "
                    f"[B, num_domains], got "
                    f"{tuple(logits.shape)}"
                )

            if logits.shape[0] != labels.shape[0]:

                raise ValueError(
                    f"Batch size mismatch at {level}: "
                    f"logits={logits.shape[0]}, "
                    f"labels={labels.shape[0]}"
                )

            # ----------------------------------------------------------
            # Cross entropy
            # ----------------------------------------------------------

            level_loss = self.loss(
                logits,
                labels.to(
                    logits.device
                )
            )

            level_losses[
                level
            ] = level_loss

            if total_loss is None:

                total_loss = level_loss

            else:

                total_loss = (
                    total_loss
                    +
                    level_loss
                )

        # --------------------------------------------------------------
        # Average P2-P5 losses
        # --------------------------------------------------------------

        total_loss = (
            total_loss
            /
            len(level_losses)
        )

        return total_loss

    # ==================================================================
    # PER LEVEL LOSS
    # ==================================================================

    def compute_per_level(
        self,
        predictions,
        labels
    ):

        if not isinstance(
            predictions,
            dict
        ):

            raise TypeError(
                "predictions must be a dictionary."
            )

        labels = labels.long()

        losses = {}

        for level, logits in predictions.items():

            losses[
                level
            ] = self.loss(
                logits,
                labels.to(
                    logits.device
                )
            )

        return losses


# ======================================================================
# DOMAIN ACCURACY
# ======================================================================

@torch.no_grad()
def domain_accuracy(
    predictions,
    labels
):
    """
    Calculate average domain classification accuracy across P2-P5.
    """

    if not isinstance(
        predictions,
        dict
    ):

        raise TypeError(
            "predictions must be a dictionary."
        )

    labels = labels.long()

    accuracies = {}

    for level, logits in predictions.items():

        predicted = torch.argmax(
            logits,
            dim=1
        )

        correct = (
            predicted
            ==
            labels.to(
                predicted.device
            )
        )

        accuracy = (
            correct.float().mean()
        )

        accuracies[
            level
        ] = float(
            accuracy.item()
        )

    if not accuracies:

        return 0.0, {}

    mean_accuracy = float(
        sum(
            accuracies.values()
        )
        /
        len(accuracies)
    )

    return mean_accuracy, accuracies


# ======================================================================
# DOMAIN PREDICTION SUMMARY
# ======================================================================

@torch.no_grad()
def domain_prediction_summary(
    predictions
):
    """
    Return predicted Source/Target counts for every FPN level.
    """

    summary = {}

    for level, logits in predictions.items():

        predicted = torch.argmax(
            logits,
            dim=1
        )

        source_count = int(
            (
                predicted == 0
            ).sum().item()
        )

        target_count = int(
            (
                predicted == 1
            ).sum().item()
        )

        summary[
            level
        ] = {

            "source": source_count,

            "target": target_count,

            "total": len(
                predicted
            )

        }

    return summary


# ======================================================================
# FACTORY
# ======================================================================

def build_domain_discriminator(
    channels=None,
    hidden_dim=None,
    num_domains=2,
    dropout=None
):
    """
    Build SICDA multi-scale domain discriminator.
    """

    if channels is None:

        channels = FEATURE_DIM

    if hidden_dim is None:

        hidden_dim = DOMAIN_HIDDEN

    if dropout is None:

        dropout = DOMAIN_DROPOUT

    return MultiScaleDomainDiscriminator(

        channels=channels,

        hidden_dim=hidden_dim,

        num_domains=num_domains,

        dropout=dropout

    )


# ======================================================================
# TEST
# ======================================================================

def test_domain_discriminator():

    print()
    print("=" * 78)
    print("TESTING SICDA DOMAIN DISCRIMINATOR")
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
    # Configuration
    # --------------------------------------------------------------

    batch_size = 4

    channels = FEATURE_DIM

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model = build_domain_discriminator(
        channels=channels
    ).to(
        device
    )

    model.train()

    # --------------------------------------------------------------
    # FPN features
    # --------------------------------------------------------------

    features = {

        "p2": torch.randn(
            batch_size,
            channels,
            64,
            64,
            device=device,
            requires_grad=True
        ),

        "p3": torch.randn(
            batch_size,
            channels,
            32,
            32,
            device=device,
            requires_grad=True
        ),

        "p4": torch.randn(
            batch_size,
            channels,
            16,
            16,
            device=device,
            requires_grad=True
        ),

        "p5": torch.randn(
            batch_size,
            channels,
            8,
            8,
            device=device,
            requires_grad=True
        )

    }

    # --------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------

    outputs = model(
        features
    )

    print()
    print(
        "OUTPUT SHAPES"
    )

    print(
        "-" * 78
    )

    for level in (
        "p2",
        "p3",
        "p4",
        "p5"
    ):

        print(
            f"{level:<10}: "
            f"{tuple(outputs[level].shape)}"
        )

        assert outputs[level].shape == (
            batch_size,
            2
        )

    # --------------------------------------------------------------
    # Domain labels
    #
    # First two = Source
    # Last two  = Target
    # --------------------------------------------------------------

    labels = torch.tensor(
        [
            0,
            0,
            1,
            1
        ],
        dtype=torch.long,
        device=device
    )

    # --------------------------------------------------------------
    # Loss
    # --------------------------------------------------------------

    criterion = DomainLoss()

    loss = criterion(
        outputs,
        labels
    )

    print()
    print(
        "DOMAIN LOSS"
    )

    print(
        "-" * 78
    )

    print(
        f"Loss : {loss.item():.6f}"
    )

    assert torch.isfinite(
        loss
    )

    assert loss.item() >= 0.0

    # --------------------------------------------------------------
    # Per-level loss
    # --------------------------------------------------------------

    level_losses = criterion.compute_per_level(
        outputs,
        labels
    )

    print()
    print(
        "PER-LEVEL LOSSES"
    )

    print(
        "-" * 78
    )

    for level, level_loss in level_losses.items():

        print(
            f"{level:<10}: "
            f"{level_loss.item():.6f}"
        )

    # --------------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------------

    mean_accuracy, accuracies = domain_accuracy(
        outputs,
        labels
    )

    print()
    print(
        "DOMAIN ACCURACY"
    )

    print(
        "-" * 78
    )

    for level, accuracy in accuracies.items():

        print(
            f"{level:<10}: "
            f"{accuracy:.4f}"
        )

    print(
        f"Mean       : "
        f"{mean_accuracy:.4f}"
    )

    # --------------------------------------------------------------
    # Prediction summary
    # --------------------------------------------------------------

    summary = domain_prediction_summary(
        outputs
    )

    print()
    print(
        "DOMAIN PREDICTION SUMMARY"
    )

    print(
        "-" * 78
    )

    for level, values in summary.items():

        print(
            f"{level:<10}: "
            f"Source={values['source']} "
            f"Target={values['target']} "
            f"Total={values['total']}"
        )

    # --------------------------------------------------------------
    # Backward test
    # --------------------------------------------------------------

    model.zero_grad()

    loss.backward()

    gradients_found = False

    for parameter in model.parameters():

        if parameter.grad is not None:

            if torch.isfinite(
                parameter.grad
            ).all():

                gradients_found = True

                break

    assert gradients_found, (
        "No valid gradients found."
    )

    print()
    print(
        "[PASS] Forward test."
    )

    print(
        "[PASS] Loss test."
    )

    print(
        "[PASS] Accuracy test."
    )

    print(
        "[PASS] Backward test."
    )

    print()
    print("=" * 78)
    print("DOMAIN DISCRIMINATOR TEST PASSED")
    print("=" * 78)


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":

    test_domain_discriminator()