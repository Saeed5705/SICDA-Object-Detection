import torch


def source_collate_fn(batch):
    """LLVIP source: paired RGB/IR + detection targets."""
    return {
        "rgb": [sample["rgb"] for sample in batch],
        "thermal": [sample["thermal"] for sample in batch],
        "targets": [sample["target"] for sample in batch],
        "rgb_path": [sample["rgb_path"] for sample in batch],
        "thermal_path": [sample["thermal_path"] for sample in batch],
        "domain": [sample.get("domain", 0) for sample in batch],
    }


def target_collate_fn(batch):
    """FLIR target training batch; labels are intentionally not used."""
    out = {
        "rgb": [sample["rgb"] for sample in batch],
        "thermal": [sample["thermal"] for sample in batch],
        "rgb_path": [sample["rgb_path"] for sample in batch],
        "thermal_path": [sample["thermal_path"] for sample in batch],
        "domain": [sample.get("domain", 1) for sample in batch],
    }
    if batch and "target" in batch[0]:
        out["targets"] = [sample["target"] for sample in batch]
    return out


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, list):
        return [move_to_device(x, device) for x in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    return batch
