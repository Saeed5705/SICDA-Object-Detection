import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter
from config import Config


def _resize_target(target, old_w, old_h, new_size):
    if target is None:
        return None
    target = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in target.items()}
    boxes = target.get("boxes")
    if boxes is not None and boxes.numel() > 0:
        boxes[:, [0, 2]] *= float(new_size) / float(old_w)
        boxes[:, [1, 3]] *= float(new_size) / float(old_h)
        target["boxes"] = boxes
        target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return target


def _flip_target(target, width):
    if target is None:
        return None
    boxes = target.get("boxes")
    if boxes is not None and boxes.numel() > 0:
        boxes = boxes.clone()
        xmin = boxes[:, 0].clone()
        xmax = boxes[:, 2].clone()
        boxes[:, 0] = width - xmax
        boxes[:, 2] = width - xmin
        target["boxes"] = boxes
    return target


class PairedTrainTransform:
    """Apply identical geometry to aligned RGB/thermal pairs.

    Color jitter is intentionally applied only to RGB.
    Supports an optional detection target (LLVIP source).
    """

    def __init__(self, size=Config.IMAGE_SIZE, flip_prob=0.5):
        self.size = size
        self.flip_prob = flip_prob
        self.rgb_jitter = ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        )

    def __call__(self, rgb, thermal=None, target=None):
        # Legacy single-RGB adapter (keeps coco.py usable if needed).
        if isinstance(thermal, dict) and target is None:
            target = thermal
            thermal = rgb.convert("L")
        if thermal is None:
            thermal = rgb.convert("L")

        old_w, old_h = rgb.size
        rgb = F.resize(rgb, (self.size, self.size))
        thermal = F.resize(thermal, (self.size, self.size))
        target = _resize_target(target, old_w, old_h, self.size)

        if random.random() < self.flip_prob:
            rgb = F.hflip(rgb)
            thermal = F.hflip(thermal)
            target = _flip_target(target, self.size)

        rgb = self.rgb_jitter(rgb)
        rgb = F.to_tensor(rgb)
        thermal = F.to_tensor(thermal)

        rgb = F.normalize(rgb, Config.PIXEL_MEAN, Config.PIXEL_STD)
        thermal = F.normalize(thermal, mean=[0.5], std=[0.5])

        if target is None:
            return rgb, thermal
        return rgb, thermal, target


class PairedValidationTransform:
    """Deterministic resize/normalization for aligned RGB/thermal pairs."""

    def __init__(self, size=Config.IMAGE_SIZE):
        self.size = size

    def __call__(self, rgb, thermal=None, target=None):
        if isinstance(thermal, dict) and target is None:
            target = thermal
            thermal = rgb.convert("L")
        if thermal is None:
            thermal = rgb.convert("L")

        old_w, old_h = rgb.size
        rgb = F.resize(rgb, (self.size, self.size))
        thermal = F.resize(thermal, (self.size, self.size))
        target = _resize_target(target, old_w, old_h, self.size)

        rgb = F.to_tensor(rgb)
        thermal = F.to_tensor(thermal)
        rgb = F.normalize(rgb, Config.PIXEL_MEAN, Config.PIXEL_STD)
        thermal = F.normalize(thermal, mean=[0.5], std=[0.5])

        if target is None:
            return rgb, thermal
        return rgb, thermal, target


# Factory functions retained with the original names.
def build_source_train_transform():
    return PairedTrainTransform()


def build_source_validation_transform():
    return PairedValidationTransform()


def build_target_train_transform():
    return PairedTrainTransform()


def build_target_validation_transform():
    return PairedValidationTransform()
