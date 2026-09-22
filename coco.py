"""
===============================================================
COCO Source Dataset for SICDA
(with missing-image filtering)
===============================================================
"""

import os
import json
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

from config import Config


class CocoDataset(Dataset):

    def __init__(
            self,
            image_dir,
            annotation_file,
            transforms=None,
            training=True
    ):
        super().__init__()

        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.transforms = transforms
        self.training = training

        print("=" * 70)
        print("Loading COCO Source Dataset...")
        print("=" * 70)
        print(f"Image dir       : {image_dir}")
        print(f"Annotation file : {annotation_file}")

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(
                f"SOURCE_IMAGE_DIR does not exist:\n{image_dir}\n"
                f"Check Config.SOURCE_IMAGE_DIR in config.py"
            )

        with open(annotation_file, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]
        self.categories = coco["categories"]

        self.category_name = {}
        for cat in self.categories:
            self.category_name[cat["id"]] = cat["name"].lower()

        wanted = {"person", "car", "bicycle", "bike"}

        self.valid_categories = []
        for cat in self.categories:
            if cat["name"].lower() in wanted:
                self.valid_categories.append(cat["id"])

        # Background=0, Person=1, Car=2, Bike=3
        self.label_map = {}
        for cat in self.categories:
            name = cat["name"].lower()
            if name == "person":
                self.label_map[cat["id"]] = 1
            elif name == "car":
                self.label_map[cat["id"]] = 2
            elif name in ("bicycle", "bike"):
                self.label_map[cat["id"]] = 3

        self.image_annotations = defaultdict(list)
        for ann in self.annotations:
            if ann["category_id"] not in self.valid_categories:
                continue
            if ann.get("iscrowd", 0) == 1:
                continue
            self.image_annotations[ann["image_id"]].append(ann)

        # Keep only images that have valid boxes AND exist on disk
        valid_images = []
        missing = 0

        for img in self.images:
            if training and img["id"] not in self.image_annotations:
                continue

            path = self._resolve_image_path(img["file_name"])
            if path is None:
                missing += 1
                continue

            valid_images.append(img)

        self.images = valid_images

        print(f"Total Images      : {len(self.images)}")
        print(f"Missing on disk   : {missing}")
        print(f"Valid Categories  : {len(self.valid_categories)}")
        print(f"Training Mode     : {training}")
        print(f"Label Map         : {self.label_map}")
        print("=" * 70)

        if len(self.images) == 0:
            raise RuntimeError(
                "No COCO images found on disk.\n"
                f"Checked folder: {self.image_dir}\n"
                "Download train2017.zip and extract it there."
            )

    def _resolve_image_path(self, file_name):
        """Try common COCO path layouts. Return path or None."""
        file_name = file_name.replace("\\", "/")
        candidates = [
            os.path.join(self.image_dir, file_name),
            os.path.join(self.image_dir, os.path.basename(file_name)),
            os.path.join(self.image_dir, "train2017", os.path.basename(file_name)),
            os.path.join(self.image_dir, "val2017", os.path.basename(file_name)),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = self.images[index]
        image_id = image_info["id"]
        file_name = image_info["file_name"]

        image_path = self._resolve_image_path(file_name)
        if image_path is None:
            raise FileNotFoundError(
                f"Image not found for id={image_id}, file={file_name}\n"
                f"Searched under: {self.image_dir}"
            )

        image = Image.open(image_path).convert("RGB")
        annotations = self.image_annotations[image_id]

        boxes, labels, area, iscrowd = [], [], [], []

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            x1, y1 = float(x), float(y)
            x2, y2 = float(x + w), float(y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.label_map[ann["category_id"]])
            area.append(float(ann.get("area", (x2 - x1) * (y2 - y1))))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            area = torch.tensor(area, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            image = F.to_tensor(image)

        return image, target

    def get_num_classes(self):
        return Config.NUM_CLASSES

    def get_class_names(self):
        return Config.CLASS_NAMES

    def get_category_mapping(self):
        return self.label_map

    def print_statistics(self):
        print("\n" + "=" * 60)
        print("COCO Source Dataset Statistics")
        print("=" * 60)
        print(f"Images           : {len(self.images)}")
        total_boxes = sum(len(a) for a in self.image_annotations.values())
        print(f"Bounding Boxes   : {total_boxes}")
        print(f"Selected Classes : {Config.CLASS_NAMES}")
        print(f"Category Mapping : {self.label_map}")
        print("=" * 60)


if __name__ == "__main__":
    dataset = CocoDataset(
        image_dir=Config.SOURCE_IMAGE_DIR,
        annotation_file=Config.SOURCE_ANNOTATION,
        transforms=None,
        training=True
    )
    print("Length:", len(dataset))
    if len(dataset) > 0:
        image, target = dataset[0]
        print("Image shape:", image.shape)
        print("Labels:", target["labels"])
        dataset.print_statistics()