import os
import math
import time

import torch
from tqdm import tqdm


# =====================================================================
# SICDA TRAINER
# =====================================================================

class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        total_loss_fn,
        source_loader=None,
        target_loader=None,
        train_loader=None,
        val_loader=None,
        evaluator=None,
        device="cuda",
        cfg=None,
    ):

        # -------------------------------------------------------------
        # Main objects
        # -------------------------------------------------------------

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_loss_fn = total_loss_fn

        self.source_loader = source_loader
        self.target_loader = target_loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator

        self.device = torch.device(device)
        self.cfg = cfg

        # -------------------------------------------------------------
        # Configuration
        # -------------------------------------------------------------

        self.epochs = int(
            getattr(cfg, "EPOCHS", 1)
        )

        self.grad_accum_steps = max(
            1,
            int(
                getattr(
                    cfg,
                    "GRAD_ACCUM_STEPS",
                    1
                )
            )
        )

        self.max_grad_norm = float(
            getattr(
                cfg,
                "MAX_GRAD_NORM",
                0.0
            )
        )

        # Numerical-stability safeguards
        self.skip_nonfinite_batches = bool(getattr(cfg, "SKIP_NONFINITE_BATCHES", True))
        self.nonfinite_batches = 0
        self.nan_check_every = max(1, int(getattr(cfg, "NAN_CHECK_EVERY", 1)))
        self.max_feature_abs = float(getattr(cfg, "MAX_FEATURE_ABS", 1e4))
        self.max_loss_value = float(getattr(cfg, "MAX_LOSS_VALUE", 1e4))

        self.validate_freq = max(
            1,
            int(
                getattr(
                    cfg,
                    "VALIDATE_FREQ",
                    1
                )
            )
        )

        self.print_freq = max(
            1,
            int(
                getattr(
                    cfg,
                    "PRINT_FREQ",
                    20
                )
            )
        )

        # -------------------------------------------------------------
        # AMP
        # -------------------------------------------------------------

        self.use_amp = bool(
            getattr(
                cfg,
                "USE_AMP",
                getattr(cfg, "AMP", True)
            )
            and self.device.type == "cuda"
        )

        self.amp_device = (
            "cuda"
            if self.device.type == "cuda"
            else "cpu"
        )

        self.scaler = self._create_grad_scaler()

        # -------------------------------------------------------------
        # Target adaptation
        # -------------------------------------------------------------

        self.use_target_adaptation = any(
            bool(
                getattr(
                    cfg,
                    key,
                    False
                )
            )
            for key in [
                "USE_SSL",
                "USE_GRL",
                "USE_MMD",
                "USE_CM_LOSS",
            ]
        )

        # -------------------------------------------------------------
        # Checkpoint directory
        # -------------------------------------------------------------

        self.checkpoint_dir = getattr(
            cfg,
            "CHECKPOINT_DIR",
            "./checkpoints"
        )

        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True
        )

        # -------------------------------------------------------------
        # Best model
        # -------------------------------------------------------------

        self.best_map = -math.inf
        self.best_epoch = 0
        self.best_metrics = None

        # -------------------------------------------------------------
        # Counters
        # -------------------------------------------------------------

        self.global_step = 0
        self.optimizer_steps = 0
        self.oom_batches = 0

        # Actual accumulation counter
        self.accumulation_counter = 0

        # -------------------------------------------------------------
        # Print configuration
        # -------------------------------------------------------------

        self._print_configuration()


    # =================================================================
    # PRINT CONFIGURATION
    # =================================================================

    def _print_configuration(self):

        print()
        print("=" * 78)
        print("                         SICDA TRAINER")
        print("=" * 78)

        print(
            f"Device                : {self.device}"
        )

        if self.device.type == "cuda":

            try:

                print(
                    f"GPU                   : "
                    f"{torch.cuda.get_device_name(self.device)}"
                )

            except Exception:

                print(
                    "GPU                   : CUDA"
                )

        print(
            f"Epochs                : "
            f"{self.epochs}"
        )

        print(
            f"Micro batch           : "
            f"{getattr(self.cfg, 'BATCH_SIZE', 'N/A')}"
        )

        print(
            f"Gradient accumulation : "
            f"{self.grad_accum_steps}"
        )

        print(
            f"Effective batch       : "
            f"{getattr(self.cfg, 'BATCH_SIZE', 1) * self.grad_accum_steps}"
        )

        print(
            f"Mixed precision       : "
            f"{self.use_amp}"
        )

        print(
            f"Target adaptation     : "
            f"{self.use_target_adaptation}"
        )

        print(
            f"Max grad norm         : {self.max_grad_norm}"
        )

        print(
            f"Non-finite protection : {self.skip_nonfinite_batches}"
        )

        print(
            "AMP API               : torch.amp"
        )

        print(
            "Detector              : Faster R-CNN"
        )

        print(
            "Feature levels        : p2, p3, p4, p5"
        )

        print(
            "Classes               : "
            "background, person, car, bike"
        )

        print(
            "Memory strategy       : "
            "Source -> cleanup -> Target -> cleanup"
        )

        print("=" * 78)


    # =================================================================
    # AMP GRAD SCALER
    # =================================================================

    def _create_grad_scaler(self):

        if self.device.type != "cuda":

            try:

                return torch.amp.GradScaler(
                    "cpu",
                    enabled=False
                )

            except Exception:

                return torch.cuda.amp.GradScaler(
                    enabled=False
                )

        try:

            return torch.amp.GradScaler(
                "cuda",
                enabled=self.use_amp
            )

        except TypeError:

            return torch.cuda.amp.GradScaler(
                enabled=self.use_amp
            )


    # =================================================================
    # AMP CONTEXT
    # =================================================================

    def autocast_context(self):

        return torch.amp.autocast(
            device_type=self.amp_device,
            enabled=self.use_amp
        )


    # =================================================================
    # MOVE TARGETS
    # =================================================================

    def move_targets_to_device(
        self,
        targets
    ):

        if targets is None:

            return None

        if not isinstance(
            targets,
            (list, tuple)
        ):

            raise TypeError(
                "Targets must be list/tuple of dictionaries. "
                f"Got {type(targets)}"
            )

        output = []

        for target in targets:

            if not isinstance(
                target,
                dict
            ):

                raise TypeError(
                    "Each target must be a dictionary."
                )

            moved = {}

            for key, value in target.items():

                if torch.is_tensor(value):

                    moved[key] = value.to(
                        self.device,
                        non_blocking=True
                    )

                else:

                    moved[key] = value

            output.append(
                moved
            )

        return output


    # =================================================================
    # PREPARE PAIRED BATCH
    # =================================================================

    def prepare_paired_batch(
        self,
        batch,
        require_targets=False
    ):

        if not isinstance(
            batch,
            dict
        ):

            raise TypeError(
                "Expected dictionary batch, "
                f"got {type(batch)}"
            )

        # =============================================================
        # RGB
        # =============================================================

        rgb_data = batch.get(
            "rgb"
        )

        if rgb_data is None:

            raise KeyError(
                "Batch does not contain 'rgb'."
            )

        if torch.is_tensor(rgb_data):

            if rgb_data.ndim == 3:

                rgb_list = [
                    rgb_data
                ]

            elif rgb_data.ndim == 4:

                rgb_list = [
                    x for x in rgb_data
                ]

            else:

                raise ValueError(
                    f"Invalid RGB tensor shape: "
                    f"{tuple(rgb_data.shape)}"
                )

        else:

            rgb_list = list(
                rgb_data
            )

        rgb_list = [

            x.to(
                self.device,
                non_blocking=True
            )

            for x in rgb_list

        ]

        # =============================================================
        # THERMAL
        # =============================================================

        thermal_data = batch.get(
            "thermal"
        )

        if thermal_data is None:

            raise KeyError(
                "Batch does not contain 'thermal'."
            )

        if torch.is_tensor(thermal_data):

            if thermal_data.ndim == 3:

                thermal_list = [
                    thermal_data
                ]

            elif thermal_data.ndim == 4:

                thermal_list = [
                    x for x in thermal_data
                ]

            else:

                raise ValueError(
                    f"Invalid thermal tensor shape: "
                    f"{tuple(thermal_data.shape)}"
                )

        else:

            thermal_list = list(
                thermal_data
            )

        thermal_list = [

            x.to(
                self.device,
                non_blocking=True
            )

            for x in thermal_list

        ]

        # =============================================================
        # CHECK PAIRING
        # =============================================================

        if len(rgb_list) != len(
            thermal_list
        ):

            raise ValueError(
                "RGB/Thermal batch mismatch: "
                f"{len(rgb_list)} vs "
                f"{len(thermal_list)}"
            )

        if len(rgb_list) == 0:

            raise ValueError(
                "Empty RGB/Thermal batch."
            )

        # =============================================================
        # STACK FOR SICDA
        # =============================================================

        rgb = torch.stack(
            rgb_list,
            dim=0
        )

        thermal = torch.stack(
            thermal_list,
            dim=0
        )

        # =============================================================
        # DETECTOR INPUT
        # =============================================================

        detector_images = rgb_list

        # =============================================================
        # TARGETS
        # =============================================================

        targets = None

        if "targets" in batch:

            targets = self.move_targets_to_device(
                batch["targets"]
            )

        elif "target" in batch:

            targets = self.move_targets_to_device(
                batch["target"]
            )

        if require_targets:

            if targets is None:

                raise ValueError(
                    "Detection targets are required "
                    "but were not found in batch."
                )

            if len(targets) != len(rgb_list):

                raise ValueError(
                    "Target count does not match "
                    "image count: "
                    f"{len(targets)} vs "
                    f"{len(rgb_list)}"
                )

        # =============================================================
        # DOMAIN LABELS
        # =============================================================

        domains = batch.get(
            "domain",
            None
        )

        if domains is None:

            domains = [
                0
            ] * len(rgb_list)

        elif torch.is_tensor(domains):

            if domains.ndim == 0:

                domains = [
                    int(domains.item())
                ] * len(rgb_list)

            else:

                domains = (
                    domains
                    .detach()
                    .cpu()
                    .tolist()
                )

        elif isinstance(
            domains,
            int
        ):

            domains = [
                domains
            ] * len(rgb_list)

        if len(domains) != len(rgb_list):

            raise ValueError(
                "Domain label count does not match "
                "batch size."
            )

        domain_labels = torch.tensor(
            domains,
            dtype=torch.long,
            device=self.device
        )

        return (
            rgb,
            thermal,
            detector_images,
            targets,
            domain_labels
        )


    # =================================================================
    # NUMERICAL STABILITY
    # =================================================================

    @staticmethod
    def _is_finite_tensor(value):
        return (not torch.is_tensor(value)) or bool(torch.isfinite(value).all().item())

    def _check_tensor_finite(self, value, name, allow_empty=True):
        if value is None:
            return
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a tensor, got {type(value)}")
        if allow_empty and value.numel() == 0:
            return
        if not torch.isfinite(value).all():
            bad = (~torch.isfinite(value)).sum().item()
            raise FloatingPointError(f"Non-finite values detected in {name}: {bad}")
        if self.max_feature_abs > 0 and value.numel() > 0:
            max_abs = float(value.detach().abs().max().item())
            if max_abs > self.max_feature_abs:
                raise FloatingPointError(
                    f"Exploding values detected in {name}: max_abs={max_abs:.6e}"
                )

    def _check_paired_inputs(self, rgb, thermal, prefix="batch"):
        self._check_tensor_finite(rgb, f"{prefix}.rgb")
        self._check_tensor_finite(thermal, f"{prefix}.thermal")

    def _check_model_output_finite(self, outputs, prefix="model"):
        if not isinstance(outputs, dict):
            return
        for key, value in outputs.items():
            if key == "detections" or key == "losses":
                continue
            if torch.is_tensor(value):
                self._check_tensor_finite(value, f"{prefix}.{key}")
        losses = outputs.get("losses", {})
        if isinstance(losses, dict):
            for key, value in losses.items():
                if torch.is_tensor(value):
                    self._check_tensor_finite(value, f"{prefix}.losses.{key}")
                    if value.numel() == 1 and self.max_loss_value > 0:
                        scalar = float(value.detach().abs().item())
                        if scalar > self.max_loss_value:
                            raise FloatingPointError(
                                f"Loss explosion in {prefix}.losses.{key}: {scalar:.6e}"
                            )

    def _check_gradients_finite(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise FloatingPointError(f"Non-finite gradient detected in {name}")

    def _recover_nonfinite_batch(self, batch_index, message):
        """
        Safely recover from NaN/Inf without leaving GradScaler in an
        invalid state. This is important because AMP requires update()
        after an unscale_()/step attempt before the optimizer can be
        unscaled again.
        """

        self.nonfinite_batches += 1

        # Clear accumulated gradients first.
        self.zero_grad()
        self.accumulation_counter = 0

        # Refresh GradScaler state when the failed batch reached the AMP
        # optimizer stage. If the batch failed before unscale_(), update()
        # can legitimately raise; in that case there is no scaler state
        # to refresh, so it is safe to ignore the exception.
        if self.use_amp:
            try:
                self.scaler.update()
            except Exception:
                pass

        self._clear_detector_feature_cache()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        print()
        print("[WARNING] NON-FINITE NUMERICAL STATE")
        print(f"Batch skipped: {batch_index + 1}")
        print(f"Reason       : {message}")
        print("Action       : gradients cleared, AMP state refreshed and CUDA cache released")


    # =================================================================
    # ZERO GRAD
    # =================================================================

    def zero_grad(self):

        self.optimizer.zero_grad(
            set_to_none=True
        )


    # =================================================================
    # UNWRAP MODEL
    # =================================================================

    def _unwrap_model(self):

        if hasattr(
            self.model,
            "module"
        ):

            return self.model.module

        return self.model


    # =================================================================
    # CLEAR DETECTOR CACHE
    # =================================================================

    def _clear_detector_feature_cache(self):

        try:

            base = self._unwrap_model()

            detector = getattr(
                base,
                "detector",
                None
            )

            if detector is not None:

                fpn_wrapper = getattr(
                    detector,
                    "fpn_wrapper",
                    None
                )

                if fpn_wrapper is not None:

                    if hasattr(
                        fpn_wrapper,
                        "features"
                    ):

                        fpn_wrapper.features = None

        except Exception:

            pass


    # =================================================================
    # MEMORY CLEANUP
    # =================================================================

    def _memory_cleanup(
        self,
        force=False
    ):

        self._clear_detector_feature_cache()

        if self.device.type == "cuda":

            if force:

                torch.cuda.empty_cache()


    # =================================================================
    # OPTIMIZER STEP
    # =================================================================

    def _step_optimizer(self):
        """
        Perform one safe optimizer update.

        The key AMP rule is that unscale_() is called at most once for an
        optimizer between scaler.update() calls. If a gradient check fails
        after unscale_(), scaler.update() is still attempted so the next
        batch starts with a clean GradScaler state.
        """

        try:
            # --------------------------------------------------------
            # AMP: unscale exactly once before checking/clipping grads
            # --------------------------------------------------------
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)

            # --------------------------------------------------------
            # Finite-gradient check
            # --------------------------------------------------------
            self._check_gradients_finite()

            # --------------------------------------------------------
            # Gradient clipping
            # --------------------------------------------------------
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

            # --------------------------------------------------------
            # Optimizer step
            # --------------------------------------------------------
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer_steps += 1

        except FloatingPointError:
            # If unscale_() was already called, GradScaler must be
            # advanced/reset before the next optimizer update.
            if self.use_amp:
                try:
                    self.scaler.update()
                except Exception:
                    pass
            raise

        finally:
            self.zero_grad()
            self.accumulation_counter = 0
            self._clear_detector_feature_cache()


    # =================================================================
    # SAFE LOSS
    # =================================================================

    @staticmethod
    def _safe_loss(
        losses,
        key,
        device
    ):

        if losses is None:

            return torch.zeros(
                (),
                device=device
            )

        value = losses.get(
            key,
            None
        )

        if value is None:

            return torch.zeros(
                (),
                device=device
            )

        if not torch.is_tensor(value):

            return torch.tensor(
                float(value),
                device=device
            )

        return value


    # =================================================================
    # NORMALIZE DETECTIONS
    # =================================================================

    @staticmethod
    def normalize_detections(
        detections
    ):

        """
        Normalize SICDA/Faster R-CNN detection output.

        Expected final format:

            [
                {
                    "boxes": Tensor[N,4],
                    "labels": Tensor[N],
                    "scores": Tensor[N]
                },
                ...
            ]
        """

        if detections is None:

            return []

        if isinstance(
            detections,
            tuple
        ):

            detections = list(
                detections
            )

        if not isinstance(
            detections,
            list
        ):

            raise TypeError(
                "Detector detections must be "
                "list/tuple of dictionaries. "
                f"Got {type(detections)}"
            )

        normalized = []

        for det in detections:

            if det is None:

                normalized.append(
                    {
                        "boxes": torch.empty(
                            (0, 4),
                            dtype=torch.float32
                        ),
                        "labels": torch.empty(
                            (0,),
                            dtype=torch.long
                        ),
                        "scores": torch.empty(
                            (0,),
                            dtype=torch.float32
                        ),
                    }
                )

                continue

            if not isinstance(
                det,
                dict
            ):

                raise TypeError(
                    "Each detection must be dictionary."
                )

            boxes = det.get(
                "boxes",
                None
            )

            labels = det.get(
                "labels",
                None
            )

            scores = det.get(
                "scores",
                None
            )

            if boxes is None:

                boxes = torch.empty(
                    (0, 4),
                    dtype=torch.float32
                )

            if labels is None:

                labels = torch.empty(
                    (0,),
                    dtype=torch.long
                )

            if scores is None:

                scores = torch.empty(
                    (0,),
                    dtype=torch.float32
                )

            # ---------------------------------------------------------
            # Tensor conversion
            # ---------------------------------------------------------

            if not torch.is_tensor(boxes):

                boxes = torch.as_tensor(
                    boxes,
                    dtype=torch.float32
                )

            else:

                boxes = boxes.detach()

            if not torch.is_tensor(labels):

                labels = torch.as_tensor(
                    labels,
                    dtype=torch.long
                )

            else:

                labels = labels.detach().long()

            if not torch.is_tensor(scores):

                scores = torch.as_tensor(
                    scores,
                    dtype=torch.float32
                )

            else:

                scores = scores.detach().float()

            # ---------------------------------------------------------
            # Shape safety
            # ---------------------------------------------------------

            if boxes.numel() == 0:

                boxes = torch.empty(
                    (0, 4),
                    dtype=torch.float32
                )

            elif boxes.ndim != 2 or boxes.shape[-1] != 4:

                raise ValueError(
                    "Invalid detector boxes shape: "
                    f"{tuple(boxes.shape)}"
                )

            if labels.ndim != 1:

                labels = labels.reshape(-1)

            if scores.ndim != 1:

                scores = scores.reshape(-1)

            # ---------------------------------------------------------
            # Count consistency
            # ---------------------------------------------------------

            n = min(
                len(boxes),
                len(labels),
                len(scores)
            )

            boxes = boxes[:n]
            labels = labels[:n]
            scores = scores[:n]

            normalized.append(
                {
                    "boxes": boxes,
                    "labels": labels,
                    "scores": scores,
                }
            )

        return normalized


    # =================================================================
    # CHECK MODEL OUTPUT
    # =================================================================

    @staticmethod
    def validate_model_output(
        outputs
    ):

        if not isinstance(
            outputs,
            dict
        ):

            raise TypeError(
                "SICDA model output must be dictionary. "
                f"Got {type(outputs)}"
            )

        if "losses" not in outputs:

            raise KeyError(
                "SICDA model output does not contain "
                "'losses'."
            )

        losses = outputs["losses"]

        if not isinstance(
            losses,
            dict
        ):

            raise TypeError(
                "outputs['losses'] must be dictionary."
            )

        return losses


    # =================================================================
    # CHECKPOINT PAYLOAD
    # =================================================================

    def _checkpoint_payload(
        self,
        epoch,
        map_score=0.0,
        metrics=None
    ):

        base = self._unwrap_model()

        return {

            "epoch":
                int(epoch),

            "model":
                self.model.state_dict(),

            "optimizer":
                self.optimizer.state_dict(),

            "scheduler":
                (
                    self.scheduler.state_dict()
                    if self.scheduler is not None
                    else None
                ),

            "map":
                float(map_score),

            "metrics":
                metrics or {},

            "source":
                "LLVIP",

            "target":
                "FLIR",

            "experiment":
                getattr(
                    self.cfg,
                    "EXPERIMENT_NAME",
                    "SICDA"
                ),

            "global_step":
                int(self.global_step),

            "optimizer_steps":
                int(self.optimizer_steps),

            "ablation":
                (
                    base.ablation_state()
                    if hasattr(
                        base,
                        "ablation_state"
                    )
                    else {}
                ),

        }


    # =================================================================
    # SAVE CHECKPOINT
    # =================================================================

    def save_checkpoint(
        self,
        epoch,
        map_score=0.0,
        metrics=None
    ):

        path = os.path.join(
            self.checkpoint_dir,
            "latest.pth"
        )

        torch.save(
            self._checkpoint_payload(
                epoch,
                map_score,
                metrics
            ),
            path
        )

        print(
            f"Checkpoint saved: {path}"
        )


    # =================================================================
    # SAVE BEST
    # =================================================================

    def save_best(
        self,
        epoch,
        map_score,
        metrics
    ):

        if map_score <= self.best_map:

            return False

        self.best_map = float(
            map_score
        )

        self.best_epoch = int(
            epoch
        )

        self.best_metrics = dict(
            metrics
        )

        path = os.path.join(
            self.checkpoint_dir,
            "best_model.pth"
        )

        torch.save(
            self._checkpoint_payload(
                epoch,
                map_score,
                metrics
            ),
            path
        )

        print(
            f"Best model saved: "
            f"mAP50:95={map_score:.6f}"
        )

        return True


    # =================================================================
    # TRAIN ONE EPOCH
    # =================================================================

    def train_one_epoch(self):

        self.model.train()

        self.zero_grad()

        self.accumulation_counter = 0

        totals = {

            "loss": 0.0,

            "det_loss": 0.0,

            "source_ssl": 0.0,

            "source_adv": 0.0,

            "source_cm": 0.0,

            "source_mmd": 0.0,

            "target_aux": 0.0,

        }

        successful = 0

        self.oom_batches = 0
        self.nonfinite_batches = 0

        if self.source_loader is None:

            raise RuntimeError(
                "source_loader is None."
            )

        if (
            self.use_target_adaptation
            and
            self.target_loader is None
        ):

            raise RuntimeError(
                "Target adaptation is enabled, "
                "but target_loader is None."
            )

        target_iterator = None

        if self.use_target_adaptation:

            target_iterator = iter(
                self.target_loader
            )

        progress = tqdm(
            self.source_loader,
            desc="Training LLVIP -> FLIR",
            dynamic_ncols=True
        )

        epoch_start = time.time()

        for batch_index, source_batch in enumerate(
            progress
        ):

            self.global_step += 1

            source_total_value = 0.0
            target_total_value = 0.0

            try:

                # =====================================================
                # SOURCE
                # =====================================================

                (
                    rgb_source,
                    thermal_source,
                    detector_images_source,
                    source_targets,
                    source_domain_labels
                ) = self.prepare_paired_batch(
                    source_batch,
                    require_targets=True
                )

                self._check_paired_inputs(
                    rgb_source, thermal_source, prefix="source_input"
                )

                # -----------------------------------------------------
                # Source forward
                # -----------------------------------------------------

                with self.autocast_context():

                    source_outputs = self.model(

                        rgb_images=rgb_source,

                        thermal_images=thermal_source,

                        detector_images=detector_images_source,

                        domain_labels=source_domain_labels,

                        targets=source_targets,

                    )

                    self._check_model_output_finite(
                        source_outputs, prefix="source_model"
                    )

                    source_losses = (
                        self.validate_model_output(
                            source_outputs
                        )
                    )

                    source_total_raw = (
                        source_losses.get(
                            "total_loss",
                            None
                        )
                    )

                    if source_total_raw is None:

                        raise RuntimeError(
                            "Model output does not contain "
                            "'losses[total_loss]'."
                        )

                    self._check_tensor_finite(
                        source_total_raw, "source_total_raw", allow_empty=False
                    )

                    source_total = (
                        source_total_raw
                        /
                        self.grad_accum_steps
                    )

                    self._check_tensor_finite(
                        source_total, "source_total", allow_empty=False
                    )

                # -----------------------------------------------------
                # Source backward
                # -----------------------------------------------------

                self.scaler.scale(
                    source_total
                ).backward()

                self.accumulation_counter += 1

                # -----------------------------------------------------
                # Statistics
                # -----------------------------------------------------

                det_loss = self._safe_loss(
                    source_losses,
                    "det_loss",
                    self.device
                )

                ssl_loss = self._safe_loss(
                    source_losses,
                    "ssl_loss",
                    self.device
                )

                adv_loss = self._safe_loss(
                    source_losses,
                    "adv_loss",
                    self.device
                )

                consistency_loss = self._safe_loss(
                    source_losses,
                    "consistency_loss",
                    self.device
                )

                # Support possible alternate naming
                if consistency_loss.item() == 0:

                    consistency_loss = self._safe_loss(
                        source_losses,
                        "cm_loss",
                        self.device
                    )

                mmd_loss = self._safe_loss(
                    source_losses,
                    "mmd_loss",
                    self.device
                )

                totals["det_loss"] += float(
                    det_loss.detach()
                )

                totals["source_ssl"] += float(
                    ssl_loss.detach()
                )

                totals["source_adv"] += float(
                    adv_loss.detach()
                )

                totals["source_cm"] += float(
                    consistency_loss.detach()
                )

                totals["source_mmd"] += float(
                    mmd_loss.detach()
                )

                source_total_value = float(
                    source_total_raw.detach()
                )

                # -----------------------------------------------------
                # Source cleanup
                # -----------------------------------------------------

                self._clear_detector_feature_cache()

                del source_outputs
                del source_losses
                del source_total
                del source_total_raw

                del rgb_source
                del thermal_source
                del detector_images_source
                del source_targets
                del source_domain_labels

                # =====================================================
                # TARGET
                # =====================================================

                if self.use_target_adaptation:

                    try:

                        target_batch = next(
                            target_iterator
                        )

                    except StopIteration:

                        target_iterator = iter(
                            self.target_loader
                        )

                        target_batch = next(
                            target_iterator
                        )

                    (
                        rgb_target,
                        thermal_target,
                        detector_images_target,
                        _,
                        target_domain_labels
                    ) = self.prepare_paired_batch(
                        target_batch,
                        require_targets=False
                    )

                    self._check_paired_inputs(
                        rgb_target, thermal_target, prefix="target_input"
                    )

                    # -------------------------------------------------
                    # Target forward
                    # -------------------------------------------------

                    with self.autocast_context():

                        target_outputs = self.model(

                            rgb_images=rgb_target,

                            thermal_images=thermal_target,

                            detector_images=detector_images_target,

                            domain_labels=target_domain_labels,

                            targets=None,

                        )

                        self._check_model_output_finite(
                            target_outputs, prefix="target_model"
                        )

                        target_losses = (
                            self.validate_model_output(
                                target_outputs
                            )
                        )

                        target_total_raw = (
                            target_losses.get(
                                "total_loss",
                                None
                            )
                        )

                        if target_total_raw is None:

                            raise RuntimeError(
                                "Target model output does not contain "
                                "'losses[total_loss]'."
                            )

                        self._check_tensor_finite(
                            target_total_raw, "target_total_raw", allow_empty=False
                        )

                        target_total = (
                            target_total_raw
                            /
                            self.grad_accum_steps
                        )

                        self._check_tensor_finite(
                            target_total, "target_total", allow_empty=False
                        )

                    # -------------------------------------------------
                    # Target backward
                    # -------------------------------------------------

                    self.scaler.scale(
                        target_total
                    ).backward()

                    target_total_value = float(
                        target_total_raw.detach()
                    )

                    totals["target_aux"] += (
                        target_total_value
                    )

                    # -------------------------------------------------
                    # Target cleanup
                    # -------------------------------------------------

                    self._clear_detector_feature_cache()

                    del target_outputs
                    del target_losses
                    del target_total
                    del target_total_raw

                    del rgb_target
                    del thermal_target
                    del detector_images_target
                    del target_domain_labels

                # =====================================================
                # TOTAL
                # =====================================================

                totals["loss"] += (
                    source_total_value
                    +
                    target_total_value
                )

                successful += 1

                # =====================================================
                # OPTIMIZER STEP
                # =====================================================

                if (
                    self.accumulation_counter
                    >=
                    self.grad_accum_steps
                ):

                    self._step_optimizer()

                # =====================================================
                # PROGRESS
                # =====================================================

                progress.set_postfix({

                    "src":
                        f"{source_total_value:.3f}",

                    "tgt":
                        f"{target_total_value:.3f}",

                    "det":
                        f"{float(det_loss.detach()):.3f}",

                    "step":
                        self.optimizer_steps,

                })

                # =====================================================
                # MEMORY CLEANUP
                # =====================================================

                if (
                    batch_index + 1
                ) % 50 == 0:

                    self._memory_cleanup(
                        force=True
                    )

            # =========================================================
            # ERROR
            # =========================================================

            except FloatingPointError as exc:

                if self.skip_nonfinite_batches:
                    self._recover_nonfinite_batch(batch_index, str(exc))
                    continue

                raise

            except RuntimeError as exc:

                message = str(
                    exc
                ).lower()

                self.zero_grad()

                self.accumulation_counter = 0

                self._clear_detector_feature_cache()

                if self.device.type == "cuda":

                    torch.cuda.empty_cache()

                if "unscale_() has already been called" in message:

                    if self.use_amp:
                        try:
                            self.scaler.update()
                        except Exception:
                            pass

                    self._recover_nonfinite_batch(
                        batch_index,
                        "GradScaler state conflict; AMP state refreshed."
                    )
                    continue

                if "out of memory" in message:

                    self.oom_batches += 1

                    print()
                    print(
                        "[WARNING] CUDA OUT OF MEMORY"
                    )

                    print(
                        f"Batch skipped: "
                        f"{batch_index + 1}"
                    )

                    print(
                        "Recommended:"
                    )

                    print(
                        "  BATCH_SIZE = 1"
                    )

                    print(
                        "  IMAGE_SIZE = 256 or 320"
                    )

                    print(
                        "  GRAD_ACCUM_STEPS = 2"
                    )

                    continue

                raise

        # =============================================================
        # LAST PARTIAL ACCUMULATION
        # =============================================================

        if self.accumulation_counter > 0:

            self._step_optimizer()

        # =============================================================
        # AVERAGE
        # =============================================================

        denominator = max(
            successful,
            1
        )

        stats = {

            key:
                value / denominator

            for key, value in totals.items()

        }

        stats["epoch_time"] = (
            time.time()
            -
            epoch_start
        )

        stats["successful_batches"] = (
            successful
        )

        stats["oom_batches"] = (
            self.oom_batches
        )

        stats["nonfinite_batches"] = (
            self.nonfinite_batches
        )

        stats["optimizer_steps"] = (
            self.optimizer_steps
        )

        return stats


    # =================================================================
    # VALIDATION
    # =================================================================

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        if self.val_loader is None:

            print(
                "[WARNING] Validation loader is None."
            )

            return {

                "mAP": 0.0,
                "mAP50": 0.0,
                "mAP75": 0.0,
                "mAP95": 0.0,
                "mAP50_95": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1-score": 0.0,

            }

        if self.evaluator is None:

            raise RuntimeError(
                "Evaluator is required "
                "for validation."
            )

        # -------------------------------------------------------------
        # Reset evaluator
        # -------------------------------------------------------------

        self.evaluator.reset()

        print()
        print("=" * 70)
        print("VALIDATION STARTED")
        print("=" * 70)

        validation_images = 0
        validation_predictions = 0
        validation_gt = 0
        validation_batches = 0

        # =============================================================
        # VALIDATION LOOP
        # =============================================================

        for batch_index, batch in enumerate(
            tqdm(
                self.val_loader,
                desc="Validation",
                dynamic_ncols=True
            )
        ):

            try:

                (
                    rgb,
                    thermal,
                    detector_images,
                    targets,
                    domain_labels
                ) = self.prepare_paired_batch(
                    batch,
                    require_targets=True
                )

                self._check_paired_inputs(
                    rgb, thermal, prefix="validation_input"
                )

                validation_batches += 1

                validation_images += len(
                    detector_images
                )

                # -----------------------------------------------------
                # GT count
                # -----------------------------------------------------

                for target in targets:

                    if isinstance(
                        target,
                        dict
                    ):

                        boxes = target.get(
                            "boxes",
                            None
                        )

                        if boxes is not None:

                            validation_gt += int(
                                len(boxes)
                            )

                # =====================================================
                # INFERENCE
                # =====================================================

                with self.autocast_context():

                    outputs = self.model(

                        rgb_images=rgb,

                        thermal_images=thermal,

                        detector_images=detector_images,

                        domain_labels=domain_labels,

                        targets=None,

                    )

                self._check_model_output_finite(
                    outputs, prefix="validation_model"
                )

                # =====================================================
                # CHECK OUTPUT
                # =====================================================

                if not isinstance(
                    outputs,
                    dict
                ):

                    raise TypeError(
                        "Validation model output must "
                        "be dictionary."
                    )

                detections = outputs.get(
                    "detections",
                    []
                )

                # -----------------------------------------------------
                # Normalize detector output
                # -----------------------------------------------------

                detections = self.normalize_detections(
                    detections
                )

                # -----------------------------------------------------
                # Critical batch check
                # -----------------------------------------------------

                if len(detections) != len(
                    detector_images
                ):

                    print(
                        "[WARNING] Detector returned "
                        f"{len(detections)} detections for "
                        f"{len(detector_images)} images."
                    )

                    # Pad empty detections if required
                    while len(detections) < len(
                        detector_images
                    ):

                        detections.append({

                            "boxes": torch.empty(
                                (0, 4)
                            ),

                            "labels": torch.empty(
                                (0,),
                                dtype=torch.long
                            ),

                            "scores": torch.empty(
                                (0,)
                            ),

                        })

                    # Never silently use extra detections
                    detections = detections[
                        :len(detector_images)
                    ]

                # -----------------------------------------------------
                # Prediction count
                # -----------------------------------------------------

                for det in detections:

                    validation_predictions += int(
                        len(
                            det["boxes"]
                        )
                    )

                # =====================================================
                # EVALUATOR
                # =====================================================

                self.evaluator.update(

                    detections,

                    {
                        "targets":
                            targets
                    }

                )

                # =====================================================
                # CLEANUP
                # =====================================================

                self._clear_detector_feature_cache()

                del outputs
                del detections

                del rgb
                del thermal
                del detector_images
                del targets
                del domain_labels

                if (
                    batch_index + 1
                ) % 50 == 0:

                    self._memory_cleanup(
                        force=True
                    )

            except RuntimeError as exc:

                message = str(
                    exc
                ).lower()

                self._clear_detector_feature_cache()

                if self.device.type == "cuda":

                    torch.cuda.empty_cache()

                if "out of memory" in message:

                    print(
                        "[WARNING] Validation CUDA OOM "
                        f"at batch {batch_index + 1}. "
                        "Skipping batch."
                    )

                    continue

                raise

        # =============================================================
        # DATA SUMMARY
        # =============================================================

        print()
        print("=" * 70)
        print("VALIDATION DATA SUMMARY")
        print("=" * 70)

        print(
            f"Validation batches     : "
            f"{validation_batches}"
        )

        print(
            f"Validation images      : "
            f"{validation_images}"
        )

        print(
            f"Ground-truth boxes     : "
            f"{validation_gt}"
        )

        print(
            f"Predicted boxes        : "
            f"{validation_predictions}"
        )

        print("=" * 70)

        # =============================================================
        # EVALUATE
        # =============================================================

        results = self.evaluator.evaluate()

        # =============================================================
        # PRINT
        # =============================================================

        print()
        print("=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)

        for key, value in results.items():

            if isinstance(
                value,
                (int, float)
            ):

                print(
                    f"{key:<20}: "
                    f"{value:.6f}"
                )

        print("=" * 70)

        return results


    # =================================================================
    # EVALUATE
    # =================================================================

    def evaluate(self):

        return self.validate()


    # =================================================================
    # FIT
    # =================================================================

    def fit(
        self,
        evaluator=None,
        scheduler=None
    ):

        if evaluator is not None:

            self.evaluator = evaluator

        active_scheduler = (

            scheduler

            if scheduler is not None

            else self.scheduler

        )

        print()
        print("=" * 78)
        print("                       STARTING SICDA TRAINING")
        print("=" * 78)

        print(
            f"Experiment : "
            f"{getattr(self.cfg, 'EXPERIMENT_NAME', 'SICDA')}"
        )

        print(
            "Source     : LLVIP"
        )

        print(
            "Target     : FLIR ADAS v2"
        )

        print(
            f"Epochs     : {self.epochs}"
        )

        print(
            f"AMP        : {self.use_amp}"
        )

        print("=" * 78)

        last_metrics = None

        # =============================================================
        # EPOCH LOOP
        # =============================================================

        for epoch in range(
            self.epochs
        ):

            print()
            print("=" * 78)

            print(
                f"EPOCH [{epoch + 1}/{self.epochs}]"
            )

            print("=" * 78)

            # ---------------------------------------------------------
            # Reset peak memory
            # ---------------------------------------------------------

            if self.device.type == "cuda":

                torch.cuda.reset_peak_memory_stats(
                    self.device
                )

            # ---------------------------------------------------------
            # Train
            # ---------------------------------------------------------

            train_stats = (
                self.train_one_epoch()
            )

            print()
            print(
                "TRAINING STATISTICS"
            )

            print("-" * 70)

            for key, value in train_stats.items():

                if isinstance(
                    value,
                    (int, float)
                ):

                    if key == "epoch_time":

                        print(
                            f"{key:<25}: "
                            f"{value:.2f} sec"
                        )

                    else:

                        print(
                            f"{key:<25}: "
                            f"{value:.6f}"
                        )

            # ---------------------------------------------------------
            # Scheduler
            # ---------------------------------------------------------

            if active_scheduler is not None:

                active_scheduler.step()

            current_lr = (
                self.optimizer.param_groups[0]["lr"]
            )

            print(
                f"Learning Rate           : "
                f"{current_lr:.8e}"
            )

            # =========================================================
            # VALIDATION
            # =========================================================

            should_validate = (

                (
                    epoch + 1
                )
                %
                self.validate_freq
                ==
                0

                or

                (
                    epoch + 1
                )
                ==
                self.epochs

            )

            if should_validate:

                last_metrics = (
                    self.validate()
                )

                # -----------------------------------------------------
                # BEST MODEL METRIC
                #
                # Use mAP50:95
                # -----------------------------------------------------

                current_map = float(
                    last_metrics.get(
                        "mAP50_95",
                        last_metrics.get(
                            "mAP",
                            0.0
                        )
                    )
                )

                print()
                print("=" * 70)
                print(
                    "EPOCH VALIDATION SUMMARY"
                )

                print(
                    f"mAP50      : "
                    f"{last_metrics.get('mAP50', 0.0):.6f}"
                )

                print(
                    f"mAP75      : "
                    f"{last_metrics.get('mAP75', 0.0):.6f}"
                )

                print(
                    f"mAP95      : "
                    f"{last_metrics.get('mAP95', 0.0):.6f}"
                )

                print(
                    f"mAP50:95   : "
                    f"{current_map:.6f}"
                )

                print(
                    f"Precision  : "
                    f"{last_metrics.get('Precision', 0.0):.6f}"
                )

                print(
                    f"Recall     : "
                    f"{last_metrics.get('Recall', 0.0):.6f}"
                )

                print(
                    f"F1-score   : "
                    f"{last_metrics.get('F1-score', 0.0):.6f}"
                )

                print("=" * 70)

                # -----------------------------------------------------
                # Latest
                # -----------------------------------------------------

                self.save_checkpoint(

                    epoch + 1,

                    current_map,

                    last_metrics

                )

                # -----------------------------------------------------
                # Best
                # -----------------------------------------------------

                self.save_best(

                    epoch + 1,

                    current_map,

                    last_metrics

                )

            else:

                self.save_checkpoint(

                    epoch + 1,

                    0.0,

                    None

                )

            # =========================================================
            # GPU MEMORY
            # =========================================================

            if self.device.type == "cuda":

                allocated = (
                    torch.cuda.memory_allocated(
                        self.device
                    )
                    /
                    1024**3
                )

                reserved = (
                    torch.cuda.memory_reserved(
                        self.device
                    )
                    /
                    1024**3
                )

                peak = (
                    torch.cuda.max_memory_allocated(
                        self.device
                    )
                    /
                    1024**3
                )

                print()
                print(
                    "GPU MEMORY"
                )

                print(
                    f"Allocated : "
                    f"{allocated:.3f} GB"
                )

                print(
                    f"Reserved  : "
                    f"{reserved:.3f} GB"
                )

                print(
                    f"Peak      : "
                    f"{peak:.3f} GB"
                )

                torch.cuda.reset_peak_memory_stats(
                    self.device
                )

        # =============================================================
        # FINAL VALIDATION
        # =============================================================

        if last_metrics is None:

            last_metrics = (
                self.validate()
            )

            current_map = float(
                last_metrics.get(
                    "mAP50_95",
                    last_metrics.get(
                        "mAP",
                        0.0
                    )
                )
            )

            self.save_best(

                self.epochs,

                current_map,

                last_metrics

            )

        # =============================================================
        # FINAL SUMMARY
        # =============================================================

        print()
        print("=" * 78)
        print("                     TRAINING FINISHED")
        print("=" * 78)

        print(
            f"Best Epoch       : "
            f"{self.best_epoch}"
        )

        print(
            f"Best mAP50:95    : "
            f"{self.best_map:.6f}"
        )

        print(
            f"Optimizer Steps  : "
            f"{self.optimizer_steps}"
        )

        print(
            f"OOM Batches      : "
            f"{self.oom_batches}"
        )

        print("=" * 78)

        return {

            "best_epoch":
                self.best_epoch,

            "best_metrics":
                self.best_metrics
                or
                last_metrics,

            "last_metrics":
                last_metrics,

        }