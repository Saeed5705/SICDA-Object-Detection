"""
==============================================================
File : trainer.py

Trainer for SICDA Framework

Compatible with:
- SICDA.py
- COCO Dataset   (source – labeled RGB)
- FLIR Dataset   (target – RGB + Thermal)
- FasterRCNN Detector
- SSL + DANN + MMD losses

Author:
Muhammad Saeed
==============================================================
"""

import os
import torch

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


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
            cfg=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_loss_fn = total_loss_fn

        self.source_loader = source_loader
        self.target_loader = target_loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator

        self.device = device
        self.cfg = cfg

        self.epochs = cfg.EPOCHS

        # AMP
        self.use_amp = getattr(cfg, "USE_AMP", True)
        self.scaler = GradScaler(enabled=self.use_amp)

        # Checkpoint
        self.checkpoint_dir = getattr(cfg, "CHECKPOINT_DIR", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_map = 0.0

        print("\n" + "=" * 70)
        print("SICDA Trainer Initialized")
        print("=" * 70)
        print(f"Device           : {device}")
        print(f"Epochs           : {self.epochs}")
        print(f"Batch Size       : {cfg.BATCH_SIZE}")
        print(f"Learning Rate    : {cfg.LEARNING_RATE}")
        print(f"Mixed Precision  : {self.use_amp}")
        print(f"Checkpoint Dir   : {self.checkpoint_dir}")
        print("=" * 70)

    # -----------------------------------------------------------
    # Move Targets To Device
    # -----------------------------------------------------------
    def move_targets_to_device(self, targets):
        output = []
        for t in targets:
            output.append({
                "boxes": t["boxes"].to(self.device),
                "labels": t["labels"].to(self.device),
                "image_id": t["image_id"].to(self.device),
                "area": t["area"].to(self.device),
                "iscrowd": t["iscrowd"].to(self.device)
            })
        return output

    # -----------------------------------------------------------
    # Prepare Source Batch (COCO – RGB + labels)
    # Fake thermal = mean of RGB channels
    # -----------------------------------------------------------
    def prepare_source_batch(self, images, targets):
        rgb_images = []
        thermal_images = []
        detector_images = []

        for img in images:
            img = img.to(self.device)
            rgb_images.append(img)
            detector_images.append(img)
            thermal_images.append(
                torch.mean(img, dim=0, keepdim=True)
            )

        rgb_images = torch.stack(rgb_images)
        thermal_images = torch.stack(thermal_images)
        targets = self.move_targets_to_device(targets)

        return rgb_images, thermal_images, detector_images, targets

    # -----------------------------------------------------------
    # Prepare Target Batch (FLIR RGB + Thermal)
    # -----------------------------------------------------------
    def prepare_target_batch(self, batch):
        rgb_images = []
        thermal_images = []
        detector_images = []

        if isinstance(batch, dict):
            rgb_batch = batch["rgb"]
            thermal_batch = batch["thermal"]

            for rgb, thermal in zip(rgb_batch, thermal_batch):
                rgb = rgb.to(self.device)
                thermal = thermal.to(self.device)
                rgb_images.append(rgb)
                thermal_images.append(thermal)
                detector_images.append(rgb)

        elif isinstance(batch, list):
            for sample in batch:
                rgb = sample["rgb"].to(self.device)
                thermal = sample["thermal"].to(self.device)
                rgb_images.append(rgb)
                thermal_images.append(thermal)
                detector_images.append(rgb)

        else:
            raise TypeError(f"Unsupported target batch type {type(batch)}")

        rgb_images = torch.stack(rgb_images)
        thermal_images = torch.stack(thermal_images)

        return rgb_images, thermal_images, detector_images

    # -----------------------------------------------------------
    # Domain Labels (DANN)
    # 0 = Source (COCO), 1 = Target (FLIR)
    # -----------------------------------------------------------
    def build_domain_labels(self, batch_size, source=True):
        if source:
            labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        else:
            labels = torch.ones(batch_size, dtype=torch.long, device=self.device)
        return labels

    # -----------------------------------------------------------
    # Compute Loss Helper
    # -----------------------------------------------------------
    def compute_loss(self, outputs):
        losses = outputs["losses"]
        total_loss = losses["total_loss"]

        loss_items = {
            "total_loss": total_loss.item(),
            "det_loss": losses["det_loss"].item(),
            "ssl_loss": losses["ssl_loss"].item(),
            "adv_loss": losses["adv_loss"].item(),
            "cm_loss": losses["consistency_loss"].item(),
            "mmd_loss": losses["mmd_loss"].item()
        }
        return total_loss, loss_items

    # -----------------------------------------------------------
    # Zero Grad / Optimizer Step
    # -----------------------------------------------------------
    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self, loss):
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    # -----------------------------------------------------------
    # Save Checkpoint
    # -----------------------------------------------------------
    def save_checkpoint(self, epoch, map_score=0.0):
        path = os.path.join(self.checkpoint_dir, "latest.pth")
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "map": map_score
        }, path)
        print(f"Checkpoint saved: {path}")

    def save_best(self, epoch, map_score):
        if map_score <= self.best_map:
            return

        self.best_map = map_score
        path = os.path.join(self.checkpoint_dir, "best_model.pth")
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "map": map_score
        }, path)
        print(f"Best model saved : mAP50:95={map_score:.4f}")

    # -----------------------------------------------------------
    # Train One Epoch
    # -----------------------------------------------------------
    def train_one_epoch(self):
        self.model.train()

        total_losses = 0.0
        total_det = 0.0
        total_ssl = 0.0
        total_adv = 0.0
        total_mmd = 0.0
        total_cm = 0.0

        target_iterator = iter(self.target_loader)

        progress = tqdm(self.source_loader, desc="Training")

        for batch_idx, batch in enumerate(progress):
            try:
                # Source (COCO)
                images, targets = batch
                (
                    rgb_source,
                    thermal_source,
                    detector_images,
                    targets
                ) = self.prepare_source_batch(images, targets)

                # Target (FLIR)
                try:
                    target_batch = next(target_iterator)
                except StopIteration:
                    target_iterator = iter(self.target_loader)
                    target_batch = next(target_iterator)

                (
                    rgb_target,
                    thermal_target,
                    _
                ) = self.prepare_target_batch(target_batch)

                # Domain labels
                source_labels = self.build_domain_labels(
                    rgb_source.size(0), source=True
                )
                target_labels = self.build_domain_labels(
                    rgb_target.size(0), source=False
                )

                self.zero_grad()

                with autocast(enabled=self.use_amp):
                    # Source forward (with detection)
                    source_outputs = self.model(
                        rgb_images=rgb_source,
                        thermal_images=thermal_source,
                        detector_images=detector_images,
                        domain_labels=source_labels,
                        targets=targets
                    )

                    # Target forward (auxiliary losses only)
                    target_outputs = self.model(
                        rgb_images=rgb_target,
                        thermal_images=thermal_target,
                        detector_images=detector_images,
                        domain_labels=target_labels,
                        targets=None
                    )

                    source_loss = source_outputs["losses"]
                    total_loss = source_loss["total_loss"]

                    target_aux = target_outputs["auxiliary_losses"]
                    total_loss = (
                        total_loss
                        + self.cfg.LAMBDA_SSL * target_aux["ssl_loss"]
                        + self.cfg.LAMBDA_ADV * target_aux["adv_loss"]
                        + self.cfg.LAMBDA_CM * target_aux["consistency_loss"]
                        + self.cfg.LAMBDA_MMD * target_aux["mmd_loss"]
                    )

                self.optimizer_step(total_loss)

                total_losses += total_loss.item()
                total_det += source_loss["det_loss"].item()
                total_ssl += source_loss["ssl_loss"].item()
                total_adv += source_loss["adv_loss"].item()
                total_cm += source_loss["consistency_loss"].item()
                total_mmd += source_loss["mmd_loss"].item()

                progress.set_postfix({"Loss": f"{total_loss.item():.4f}"})

            except RuntimeError as e:
                print("\nTraining Error:")
                print(e)
                torch.cuda.empty_cache()
                continue

        num_batches = max(len(self.source_loader), 1)
        stats = {
            "loss": total_losses / num_batches,
            "det_loss": total_det / num_batches,
            "ssl_loss": total_ssl / num_batches,
            "adv_loss": total_adv / num_batches,
            "cm_loss": total_cm / num_batches,
            "mmd_loss": total_mmd / num_batches
        }
        return stats

    # -----------------------------------------------------------
    # Validation
    # -----------------------------------------------------------
    @torch.no_grad()
    def validate(self):
        self.model.eval()

        if self.val_loader is None:
            return {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0,
                    "mAP95": 0.0, "mAP50_95": 0.0}

        if self.evaluator is not None:
            self.evaluator.reset()

        print("\n" + "=" * 70)
        print("Validation Started")
        print("=" * 70)

        for batch in tqdm(self.val_loader, desc="Validation"):
            images, targets = batch

            (
                rgb_images,
                thermal_images,
                detector_images,
                targets
            ) = self.prepare_source_batch(images, targets)

            domain_labels = self.build_domain_labels(
                rgb_images.size(0), source=True
            )

            outputs = self.model(
                rgb_images=rgb_images,
                thermal_images=thermal_images,
                detector_images=detector_images,
                domain_labels=domain_labels,
                targets=None          # eval mode → detections
            )

            detections = outputs.get("detections", [])

            if self.evaluator is not None and detections:
                self.evaluator.update(
                    detections,
                    {"targets": targets}
                )

        if self.evaluator is not None:
            results = self.evaluator.evaluate()
        else:
            results = {
                "mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0,
                "mAP95": 0.0, "mAP50_95": 0.0
            }

        print("\nValidation Results")
        print(results)
        return results

    def evaluate(self):
        return self.validate()

    # -----------------------------------------------------------
    # Full Training Loop
    # -----------------------------------------------------------
    def fit(self, evaluator=None, scheduler=None):
        print("\n" + "=" * 70)
        print("Starting SICDA Training")
        print("=" * 70)

        for epoch in range(self.epochs):
            print("\n" + "=" * 70)
            print(f"Epoch [{epoch + 1}/{self.epochs}]")
            print("=" * 70)

            # Train
            train_stats = self.train_one_epoch()
            print("\nTraining Statistics")
            for k, v in train_stats.items():
                print(f"{k:20s}: {v:.6f}")

            # Scheduler
            if scheduler is not None:
                scheduler.step()
            elif self.scheduler is not None:
                self.scheduler.step()

            # Validation
            val_results = self.validate()

            # Prefer COCO-style mAP@0.50:0.95
            current_map = val_results.get(
                "mAP50_95",
                val_results.get("mAP", 0.0)
            )
            print(
                f"Val mAP50={val_results.get('mAP50', 0):.4f}  "
                f"mAP75={val_results.get('mAP75', 0):.4f}  "
                f"mAP95={val_results.get('mAP95', 0):.4f}  "
                f"mAP50:95={current_map:.4f}"
            )

            # Checkpoints
            self.save_checkpoint(epoch + 1, current_map)
            self.save_best(epoch + 1, current_map)

        print("\n" + "=" * 70)
        print("SICDA Training Finished")
        print("=" * 70)