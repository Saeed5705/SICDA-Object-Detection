import torch


# ===============================================================
# COCO Source Collate
# ===============================================================

def source_collate_fn(batch):

    images = []
    targets = []

    for sample in batch:
        image, target = sample
        images.append(image)
        targets.append(target)

    return images, targets


# ===============================================================
# FLIR Target Collate (RGB + Thermal)
# ===============================================================

def target_collate_fn(batch):

    rgb_images = []
    thermal_images = []
    rgb_paths = []
    thermal_paths = []

    for sample in batch:
        rgb_images.append(sample["rgb"])
        thermal_images.append(sample["thermal"])
        rgb_paths.append(sample["rgb_path"])
        thermal_paths.append(sample["thermal_path"])

    return {
        "rgb": rgb_images,
        "thermal": thermal_images,
        "rgb_path": rgb_paths,
        "thermal_path": thermal_paths
    }


# ===============================================================
# SICDA Joint Collate
# Source: COCO RGB + labels
# Target: FLIR RGB + Thermal
# ===============================================================

def sicda_collate_fn(source_batch, target_batch):

    source_images = []
    source_targets = []

    for image, target in source_batch:
        source_images.append(image)
        source_targets.append(target)

    target_rgb = []
    target_thermal = []

    for sample in target_batch:
        target_rgb.append(sample["rgb"])
        target_thermal.append(sample["thermal"])

    return {
        "source_images": source_images,
        "source_targets": source_targets,
        "target_rgb": target_rgb,
        "target_thermal": target_thermal
    }


# ===============================================================
# Move Batch To GPU
# ===============================================================

def move_to_device(batch, device):

    if isinstance(batch, torch.Tensor):
        return batch.to(device)

    elif isinstance(batch, list):
        return [move_to_device(item, device) for item in batch]

    elif isinstance(batch, tuple):
        return tuple(move_to_device(item, device) for item in batch)

    elif isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}

    else:
        return batch


if __name__ == "__main__":
    print("SICDA collate functions loaded successfully.")