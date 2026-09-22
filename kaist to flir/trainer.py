"""
==============================================================
File : trainer.py

SICDA Memory-Efficient Training Pipeline

Source:
    KAIST RGB-Thermal

Target:
    FLIR RGB-Thermal

Framework:
    Self-Supervised Illumination-aware
    Cross-Modal Domain Adaptation

Loss:

    L_total =
        L_det
        + lambda_ssl * L_ssl
        + lambda_adv * L_adv
        + lambda_cm  * L_cm
        + lambda_mmd * L_mmd

RTX 3070 Optimizations:

    1. Mixed Precision Training
    2. Sequential source/target processing
    3. Batch size = 1 compatible
    4. Contrastive SSL safely disabled for B=1
    5. Detached source MMD features
    6. Global average pooling for MMD
    7. No retain_graph=True
    8. Gradient clipping
    9. CUDA memory diagnostics
   10. Periodic CUDA cache cleanup
   11. Validation under inference mode

Author:
    Muhammad Saeed
==============================================================
"""

import gc
import time

import torch
import torch.nn as nn

from tqdm import tqdm


# ============================================================
# Losses
# ============================================================

from total_loss import build_total_loss

from contrastive import (
    build_contrastive_loss
)

from domain_discriminator import (
    DomainLoss
)

from mmd import (
    build_mmd
)

from evaluator import (
    Evaluator
)


# ============================================================
# Trainer
# ============================================================

class Trainer:

    def __init__(
        self,
        model,
        detector,
        train_loader,
        target_loader,
        val_loader,
        optimizer,
        device
    ):

        self.model = model

        self.detector = detector

        self.source_loader = train_loader

        self.target_loader = target_loader

        self.val_loader = val_loader

        self.optimizer = optimizer

        self.device = device

        # ----------------------------------------------------
        # Losses
        # ----------------------------------------------------

        self.total_loss = (
            build_total_loss().to(device)
        )

        self.ssl_loss = (
            build_contrastive_loss().to(device)
        )

        self.mmd_loss = (
            build_mmd().to(device)
        )

        self.domain_loss = (
            DomainLoss().to(device)
        )

        # ----------------------------------------------------
        # Evaluator
        # ----------------------------------------------------

        self.evaluator = Evaluator(
            num_classes=4
        )

        # ----------------------------------------------------
        # AMP
        # ----------------------------------------------------

        self.use_amp = (
            self.device.type == "cuda"
        )

        if self.use_amp:

            self.scaler = torch.amp.GradScaler(
                "cuda",
                enabled=True
            )

        else:

            self.scaler = None

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        self.max_grad_norm = 5.0

        # ----------------------------------------------------
        # MMD pooling
        #
        # B x C x H x W
        #
        #       ↓
        #
        # B x C
        # ----------------------------------------------------

        self.mmd_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        # ----------------------------------------------------
        # Batch-size configuration
        # ----------------------------------------------------

        self.batch_size_warning_printed = False

        print()
        print("=" * 70)
        print("Trainer Initialized")
        print("=" * 70)

        print(
            "AMP:",
            self.use_amp
        )

        print(
            "Gradient clipping:",
            self.max_grad_norm
        )

        print(
            "MMD:",
            "Global Average Pooling"
        )

        print(
            "Batch Size:",
            "1 compatible"
        )

        print("=" * 70)


    # ========================================================
    # AMP Context
    # ========================================================

    def autocast_context(self):

        if self.use_amp:

            return torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True
            )

        return torch.autocast(
            device_type="cpu",
            enabled=False
        )


    # ========================================================
    # CUDA Memory
    # ========================================================

    def print_memory(
        self,
        tag=""
    ):

        if self.device.type != "cuda":
            return

        allocated = (
            torch.cuda.memory_allocated(
                self.device
            )
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved(
                self.device
            )
            / 1024**3
        )

        peak = (
            torch.cuda.max_memory_allocated(
                self.device
            )
            / 1024**3
        )

        print(
            f"[CUDA {tag}] "
            f"Allocated={allocated:.2f} GB | "
            f"Reserved={reserved:.2f} GB | "
            f"Peak={peak:.2f} GB"
        )


    # ========================================================
    # Move Targets To Device
    # ========================================================

    def move_targets_to_device(
        self,
        targets
    ):

        if targets is None:

            return None

        output = []

        for target in targets:

            if target is None:

                output.append(
                    None
                )

                continue

            new_target = {}

            for key, value in target.items():

                if torch.is_tensor(value):

                    new_target[key] = value.to(
                        self.device,
                        non_blocking=True
                    )

                else:

                    new_target[key] = value

            output.append(
                new_target
            )

        return output


    # ========================================================
    # Pool FPN Features For MMD
    # ========================================================

    def pool_features_for_mmd(
        self,
        features
    ):

        """
        Convert FPN features:

            B x C x H x W

        into:

            B x C

        This significantly reduces MMD memory consumption.
        """

        pooled = {}

        if features is None:

            return pooled

        for level, feature in features.items():

            if feature is None:

                continue

            # ------------------------------------------------
            # Standard FPN tensor
            # ------------------------------------------------

            if feature.dim() == 4:

                pooled[level] = (
                    self.mmd_pool(
                        feature
                    )
                    .flatten(1)
                )

            # ------------------------------------------------
            # Already pooled
            # ------------------------------------------------

            elif feature.dim() == 2:

                pooled[level] = feature

            # ------------------------------------------------
            # Unexpected shape
            # ------------------------------------------------

            else:

                raise ValueError(
                    f"MMD feature '{level}' has "
                    f"invalid shape: {feature.shape}. "
                    f"Expected [B,C,H,W] or [B,C]."
                )

        return pooled


    # ========================================================
    # Domain Loss Helper
    # ========================================================

    def compute_domain_loss(
        self,
        domain_output,
        labels
    ):

        if domain_output is None:

            return torch.zeros(
                (),
                device=self.device
            )

        return self.domain_loss(
            domain_output,
            labels
        )


    # ========================================================
    # Safe SSL Loss
    # ========================================================

    def compute_ssl_loss(
        self,
        rgb_embedding,
        thermal_embedding,
        batch_size
    ):

        """
        Contrastive learning requires at least
        two samples in a batch.

        Therefore:

            B >= 2
                -> calculate SSL

            B = 1
                -> return zero

        This allows memory-constrained training
        with BATCH_SIZE=1.
        """

        # ----------------------------------------------------
        # SSL disabled through lambda
        # ----------------------------------------------------

        if self.total_loss.lambda_ssl <= 0:

            return torch.zeros(
                (),
                device=self.device
            )

        # ----------------------------------------------------
        # Batch size 1
        # ----------------------------------------------------

        if batch_size < 2:

            if not self.batch_size_warning_printed:

                print()
                print(
                    "[INFO] BATCH_SIZE=1 detected."
                )

                print(
                    "[INFO] Contrastive SSL skipped "
                    "because batch size must be >= 2."
                )

                print(
                    "[INFO] Set lambda_ssl=0 for "
                    "BATCH_SIZE=1 training."
                )

                self.batch_size_warning_printed = True

            return torch.zeros(
                (),
                device=self.device
            )

        # ----------------------------------------------------
        # Normal SSL
        # ----------------------------------------------------

        return self.ssl_loss(
            rgb_embedding,
            thermal_embedding
        )


    # ========================================================
    # Train Epoch
    # ========================================================

    def train_epoch(
        self,
        epoch
    ):

        self.model.train()

        self.detector.train()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_epoch_loss = 0.0

        total_det_loss = 0.0

        total_ssl_loss = 0.0

        total_adv_loss = 0.0

        total_mmd_loss = 0.0

        total_cm_loss = 0.0

        # ----------------------------------------------------
        # Reset CUDA peak memory
        # ----------------------------------------------------

        if self.device.type == "cuda":

            torch.cuda.reset_peak_memory_stats(
                self.device
            )

        # ----------------------------------------------------
        # Target iterator
        # ----------------------------------------------------

        target_iter = iter(
            self.target_loader
        )

        # ----------------------------------------------------
        # Progress bar
        # ----------------------------------------------------

        loop = tqdm(
            self.source_loader,
            desc=f"Epoch {epoch}",
            dynamic_ncols=True
        )

        # ====================================================
        # Epoch Loop
        # ====================================================

        for iteration, source_batch in enumerate(
            loop,
            start=1
        ):

            iteration_start = time.time()

            # =================================================
            # Get Target Batch
            # =================================================

            try:

                target_batch = next(
                    target_iter
                )

            except StopIteration:

                target_iter = iter(
                    self.target_loader
                )

                target_batch = next(
                    target_iter
                )

            # =================================================
            # Move Source Data
            # =================================================

            src_rgb = source_batch[
                "rgb"
            ].to(
                self.device,
                non_blocking=True
            )

            src_thermal = source_batch[
                "thermal"
            ].to(
                self.device,
                non_blocking=True
            )

            src_images = source_batch[
                "images"
            ]

            src_targets = (
                self.move_targets_to_device(
                    source_batch[
                        "targets"
                    ]
                )
            )

            # =================================================
            # Move Target Data
            # =================================================

            tgt_rgb = target_batch[
                "rgb"
            ].to(
                self.device,
                non_blocking=True
            )

            tgt_thermal = target_batch[
                "thermal"
            ].to(
                self.device,
                non_blocking=True
            )

            # =================================================
            # Clear Gradients
            # =================================================

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # =================================================
            # SOURCE FORWARD
            # =================================================

            with self.autocast_context():

                src_output = self.model(

                    src_rgb,

                    src_thermal,

                    domain_adaptation=True

                )

                # ---------------------------------------------
                # Save SOURCE MMD features
                #
                # IMPORTANT:
                #
                # We detach these features BEFORE source
                # backward so that the source computation graph
                # is not required again by MMD.
                #
                # We also pool them immediately to B x C.
                # ---------------------------------------------

                src_mmd_features = (
                    self.pool_features_for_mmd(
                        src_output["features"]
                    )
                )

                src_mmd_features = self.pool_features_for_mmd(
                    src_output["features"]
                )

                src_mmd_features = {
                    level: feature.detach()
                    for level, feature in src_mmd_features.items()
                }

                # ---------------------------------------------
                # Detection
                # ---------------------------------------------

                detection_features = (
                    src_output[
                        "features"
                    ]
                )

                det_loss_dict = self.detector(

                    detection_features,

                    src_images,

                    src_targets

                )

                det_loss = sum(
                    det_loss_dict.values()
                )

                # ---------------------------------------------
                # SSL
                # ---------------------------------------------

                ssl_loss = (
                    self.compute_ssl_loss(

                        src_output[
                            "embeddings"
                        ][
                            "rgb"
                        ],

                        src_output[
                            "embeddings"
                        ][
                            "thermal"
                        ],

                        src_rgb.size(0)

                    )
                )

                # ---------------------------------------------
                # Source Domain Labels
                # ---------------------------------------------

                src_domain_labels = torch.zeros(

                    src_rgb.size(0),

                    dtype=torch.long,

                    device=self.device

                )

                # ---------------------------------------------
                # Source Domain Loss
                # ---------------------------------------------

                src_adv_loss = (
                    self.compute_domain_loss(

                        src_output[
                            "domain"
                        ],

                        src_domain_labels

                    )
                )

                # ---------------------------------------------
                # Cross-modal loss
                # ---------------------------------------------

                cm_loss = torch.zeros(

                    (),

                    device=self.device

                )

                # ---------------------------------------------
                # SOURCE LOSS
                #
                # MMD is intentionally NOT included here.
                # MMD requires target features.
                # ---------------------------------------------

                source_loss = (

                    det_loss

                    + self.total_loss.lambda_ssl
                    * ssl_loss

                    + self.total_loss.lambda_adv
                    * src_adv_loss

                    + self.total_loss.lambda_cm
                    * cm_loss

                )

            # =================================================
            # SOURCE BACKWARD
            # =================================================

            if self.use_amp:

                self.scaler.scale(
                    source_loss
                ).backward()

            else:

                source_loss.backward()

            # =================================================
            # Save Logging Values BEFORE Cleanup
            # =================================================

            det_value = (
                det_loss.detach().item()
            )

            ssl_value = (
                ssl_loss.detach().item()
            )

            src_adv_value = (
                src_adv_loss.detach().item()
            )

            cm_value = (
                cm_loss.detach().item()
            )

            # =================================================
            # SOURCE GRAPH CLEANUP
            # =================================================

            del det_loss_dict

            del detection_features

            del source_loss

            del src_output

            del det_loss

            del ssl_loss

            del src_adv_loss

            del cm_loss

            # ------------------------------------------------
            # Source image tensors are no longer needed.
            #
            # Keep only pooled + detached MMD features.
            # ------------------------------------------------

            del src_rgb

            del src_thermal

            del src_targets

            # =================================================
            # TARGET FORWARD
            # =================================================

            with self.autocast_context():

                tgt_output = self.model(

                    tgt_rgb,

                    tgt_thermal,

                    domain_adaptation=True

                )

                # ---------------------------------------------
                # Target Domain Labels
                # ---------------------------------------------

                tgt_domain_labels = torch.ones(

                    tgt_rgb.size(0),

                    dtype=torch.long,

                    device=self.device

                )

                # ---------------------------------------------
                # Target Domain Loss
                # ---------------------------------------------

                tgt_adv_loss = (
                    self.compute_domain_loss(

                        tgt_output[
                            "domain"
                        ],

                        tgt_domain_labels

                    )
                )

                # ---------------------------------------------
                # Combined ADV Loss
                #
                # Source gradient has already been calculated.
                # Therefore source ADV is detached here.
                # ---------------------------------------------

                adv_loss = (

                    torch.tensor(
                        src_adv_value,
                        device=self.device
                    )

                    + tgt_adv_loss

                ) / 2.0

                # ---------------------------------------------
                # Target MMD Features
                #
                # IMPORTANT:
                #
                # Pool target features before passing to MMD.
                # ---------------------------------------------

                tgt_mmd_features = self.pool_features_for_mmd(
                    tgt_output["features"]
                )

                # ---------------------------------------------
                # Multi-scale MMD
                # ---------------------------------------------

                mmd_loss, mmd_loss_dict = (
                    self.mmd_loss(

                        src_mmd_features,

                        tgt_mmd_features

                    )
                )

                # ---------------------------------------------
                # Target Adaptation Loss
                # ---------------------------------------------

                target_loss = (

                    self.total_loss.lambda_adv
                    * tgt_adv_loss

                    + self.total_loss.lambda_mmd
                    * mmd_loss

                )

            # =================================================
            # TARGET BACKWARD
            # =================================================

            if self.use_amp:

                self.scaler.scale(
                    target_loss
                ).backward()

            else:

                target_loss.backward()

            # =================================================
            # Gradient Clipping
            # =================================================

            if self.use_amp:

                self.scaler.unscale_(
                    self.optimizer
                )

            torch.nn.utils.clip_grad_norm_(

                list(
                    self.model.parameters()
                )
                +
                list(
                    self.detector.parameters()
                ),

                max_norm=self.max_grad_norm

            )

            # =================================================
            # Optimizer Step
            # =================================================

            if self.use_amp:

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

            else:

                self.optimizer.step()

            # =================================================
            # Logging Values
            # =================================================

            tgt_adv_value = (
                tgt_adv_loss.detach().item()
            )

            mmd_value = (
                mmd_loss.detach().item()
            )

            total_adv_value = (
                src_adv_value
                + tgt_adv_value
            ) / 2.0

            total_loss_value = (

                det_value

                + self.total_loss.lambda_ssl
                * ssl_value

                + self.total_loss.lambda_adv
                * total_adv_value

                + self.total_loss.lambda_cm
                * cm_value

                + self.total_loss.lambda_mmd
                * mmd_value

            )

            # =================================================
            # Statistics
            # =================================================

            total_epoch_loss += (
                total_loss_value
            )

            total_det_loss += (
                det_value
            )

            total_ssl_loss += (
                ssl_value
            )

            total_adv_loss += (
                total_adv_value
            )

            total_mmd_loss += (
                mmd_value
            )

            total_cm_loss += (
                cm_value
            )

            # =================================================
            # Cleanup
            # =================================================

            del tgt_output

            del tgt_mmd_features

            del src_mmd_features

            del target_loss

            del tgt_adv_loss

            del mmd_loss

            del mmd_loss_dict

            del adv_loss

            del tgt_rgb

            del tgt_thermal

            # ------------------------------------------------
            # Garbage Collection
            # ------------------------------------------------

            gc.collect()

            # ------------------------------------------------
            # Periodic CUDA Cache Cleanup
            # ------------------------------------------------

            if (

                self.device.type == "cuda"

                and iteration % 50 == 0

            ):

                torch.cuda.empty_cache()

            # =================================================
            # Timing
            # =================================================

            iteration_time = (
                time.time()
                - iteration_start
            )

            # =================================================
            # Progress
            # =================================================

            loop.set_postfix(

                loss=f"{total_loss_value:.4f}",

                det=f"{total_det_loss / iteration:.4f}",

                ssl=f"{total_ssl_loss / iteration:.4f}",

                adv=f"{total_adv_loss / iteration:.4f}",

                mmd=f"{total_mmd_loss / iteration:.4f}",

                time=f"{iteration_time:.1f}s"

            )

            # =================================================
            # Memory Debugging
            # =================================================

            if (

                iteration % 50 == 0

                and self.device.type == "cuda"

            ):

                self.print_memory(

                    f"Epoch {epoch} "
                    f"Iter {iteration}"

                )

        # ====================================================
        # Epoch Statistics
        # ====================================================

        num_iterations = max(
            len(self.source_loader),
            1
        )

        avg_loss = (
            total_epoch_loss
            / num_iterations
        )

        avg_det = (
            total_det_loss
            / num_iterations
        )

        avg_ssl = (
            total_ssl_loss
            / num_iterations
        )

        avg_adv = (
            total_adv_loss
            / num_iterations
        )

        avg_mmd = (
            total_mmd_loss
            / num_iterations
        )

        avg_cm = (
            total_cm_loss
            / num_iterations
        )

        # ====================================================
        # Epoch Summary
        # ====================================================

        print()

        print("=" * 70)

        print(
            f"Epoch {epoch} Training Summary"
        )

        print("=" * 70)

        print(
            f"Total Loss : {avg_loss:.6f}"
        )

        print(
            f"Detection  : {avg_det:.6f}"
        )

        print(
            f"SSL        : {avg_ssl:.6f}"
        )

        print(
            f"ADV        : {avg_adv:.6f}"
        )

        print(
            f"CM         : {avg_cm:.6f}"
        )

        print(
            f"MMD        : {avg_mmd:.6f}"
        )

        # ----------------------------------------------------
        # GPU Memory
        # ----------------------------------------------------

        self.print_memory(
            f"Epoch {epoch} END"
        )

        print("=" * 70)

        return avg_loss


    # ========================================================
    # Validation
    # ========================================================

    @torch.inference_mode()
    def validate(self):

        self.model.eval()

        self.detector.eval()

        self.evaluator.reset()

        # ----------------------------------------------------
        # Validation Loop
        # ----------------------------------------------------

        loop = tqdm(

            self.val_loader,

            desc="Validation",

            dynamic_ncols=True

        )

        for batch in loop:

            images = batch[
                "images"
            ]

            rgb = batch[
                "rgb"
            ].to(
                self.device,
                non_blocking=True
            )

            thermal = batch[
                "thermal"
            ].to(
                self.device,
                non_blocking=True
            )

            targets = batch[
                "targets"
            ]

            # ------------------------------------------------
            # Model
            # ------------------------------------------------

            with self.autocast_context():

                output = self.model(

                    rgb,

                    thermal,

                    domain_adaptation=False

                )
                

                # ---------------------------------------------
                # Detector
                # ---------------------------------------------

                detections = self.detector(

                    output[
                        "features"
                    ],

                    images

                )

            # ------------------------------------------------
            # Evaluator
            # ------------------------------------------------

            self.evaluator.update(

                detections,

                {
                    "targets": targets
                }

            )

            # ------------------------------------------------
            # Cleanup
            # ------------------------------------------------

            del output

            del detections

            del rgb

            del thermal

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        results = (
            self.evaluator.evaluate()
        )

        print()

        print("=" * 70)

        print(
            "Validation Results"
        )

        print("=" * 70)

        for key, value in results.items():

            print(
                f"{key}: {value}"
            )

        print("=" * 70)

        return results


    # ========================================================
    # Fit
    # ========================================================

    def fit(
        self,
        epochs
    ):

        best_map = 0.0

        for epoch in range(
            epochs
        ):

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            loss = self.train_epoch(

                epoch + 1

            )

            print()

            print(
                f"Epoch {epoch + 1}"
            )

            print(
                "Loss:",
                loss
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            results = (
                self.validate()
            )

            current_map = results.get(

                "mAP50",

                results.get(
                    "mAP",
                    0.0
                )

            )

            # ------------------------------------------------
            # Best Model
            # ------------------------------------------------

            if current_map > best_map:

                best_map = current_map

                torch.save(

                    {

                        "epoch":
                            epoch + 1,

                        "model":
                            self.model.state_dict(),

                        "detector":
                            self.detector.state_dict(),

                        "optimizer":
                            self.optimizer.state_dict(),

                        "best_map50":
                            best_map

                    },

                    "best_sicda_model.pth"

                )

                print()

                print(
                    "Best model saved"
                )

        return best_map


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "SICDA Trainer Module"
    )

    print("=" * 70)

    print(
        "AMP support:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "VRAM:",
            round(

                torch.cuda.get_device_properties(
                    0
                ).total_memory
                / 1024**3,

                2

            ),

            "GB"
        )

    print("=" * 70)