import os
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import torchvision.transforms.functional as F

from config import Config


# ======================================================================
# LLVIP SOURCE DATASET
# ======================================================================

class LLVIPSourceDataset(Dataset):

    """
    SICDA LLVIP RGB-Thermal Source Dataset

    Classes:

        0 -> background
        1 -> person
        2 -> car
        3 -> bike

    Expected structure:

        LLVIP/
        ├── visible/
        │   ├── train/
        │   └── test/
        │
        ├── infrared/
        │   ├── train/
        │   └── test/
        │
        └── Generated_Annotations/
            ├── train/
            │   └── labels/
            └── test/
                └── labels/

    Annotation:

        class_id xmin ymin xmax ymax

    OR

        class_id xc yc width height

    If all coordinates are between 0 and 1,
    they are treated as YOLO normalized coordinates.
    """

    # ==================================================================
    # IMAGE EXTENSIONS
    # ==================================================================

    IMAGE_EXTS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    # ==================================================================
    # CLASS MAPPING
    # ==================================================================

    CLASS_MAP = {

        "person": 1,
        "people": 1,
        "pedestrian": 1,
        "pedestrians": 1,
        "human": 1,
        "humans": 1,

        "car": 2,
        "cars": 2,
        "vehicle": 2,
        "vehicles": 2,
        "automobile": 2,

        "bike": 3,
        "bikes": 3,
        "bicycle": 3,
        "bicycles": 3,
        "cycle": 3,
        "cycles": 3,
    }

    CLASS_NAMES_LOCAL = {
        0: "background",
        1: "person",
        2: "car",
        3: "bike",
    }

    VALID_CLASSES = {1, 2, 3}

    # ==================================================================
    # CONSTRUCTOR
    # ==================================================================

    def __init__(
        self,
        visible_dir,
        infrared_dir,
        annotation_dir,
        transforms=None,
        training=True,
        filter_corrupt=True,
    ):

        super().__init__()

        self.visible_dir = Path(visible_dir)
        self.infrared_dir = Path(infrared_dir)
        self.annotation_dir = Path(annotation_dir)

        self.transforms = transforms
        self.training = training
        self.filter_corrupt = filter_corrupt

        self.samples = []

        # ==============================================================
        # STATISTICS
        # ==============================================================

        self.total_boxes = 0

        self.person_boxes = 0
        self.car_boxes = 0
        self.bike_boxes = 0

        self.images_with_person = 0
        self.images_with_car = 0
        self.images_with_bike = 0

        self.unknown_objects = Counter()

        self.invalid_annotation_files = 0
        self.empty_annotations = 0

        self.visible_image_count = 0
        self.infrared_image_count = 0
        self.annotation_file_count = 0

        self.missing_ir_count = 0
        self.missing_visible_count = 0
        self.missing_annotation_count = 0

        self.corrupt_pairs_skipped = 0

        # ==============================================================
        # DATASET
        # ==============================================================

        self._validate_paths()

        self._count_raw_files()

        self._build_samples()

        if self.filter_corrupt:
            self._filter_readable()

        self._calculate_statistics()

        self._print_dataset_information()

        # ==============================================================
        # SAFETY
        # ==============================================================

        if not self.samples:

            raise RuntimeError(
                "\n"
                "No valid LLVIP RGB-IR samples were found.\n\n"
                f"Visible:\n{self.visible_dir}\n\n"
                f"Infrared:\n{self.infrared_dir}\n\n"
                f"Annotations:\n{self.annotation_dir}\n"
            )

    # ==================================================================
    # VALIDATE PATHS
    # ==================================================================

    def _validate_paths(self):

        paths = [
            (self.visible_dir, "Visible"),
            (self.infrared_dir, "Infrared"),
            (self.annotation_dir, "Annotation"),
        ]

        for path, name in paths:

            if not path.exists():

                raise FileNotFoundError(
                    f"\nLLVIP {name} path does not exist:\n{path}"
                )

            if not path.is_dir():

                raise NotADirectoryError(
                    f"\nLLVIP {name} path is not a directory:\n{path}"
                )

    # ==================================================================
    # FIND IMAGE FILES
    # ==================================================================

    def _image_files(self, folder):

        return sorted(
            p
            for p in folder.rglob("*")
            if (
                p.is_file()
                and p.suffix.lower() in self.IMAGE_EXTS
            )
        )

    # ==================================================================
    # FIND TXT ANNOTATIONS
    # ==================================================================

    def _annotation_files(self):

        return sorted(
            p
            for p in self.annotation_dir.rglob("*.txt")
            if p.is_file()
        )

    # ==================================================================
    # COUNT RAW FILES
    # ==================================================================

    def _count_raw_files(self):

        visible_files = self._image_files(
            self.visible_dir
        )

        infrared_files = self._image_files(
            self.infrared_dir
        )

        annotation_files = self._annotation_files()

        self.visible_image_count = len(
            visible_files
        )

        self.infrared_image_count = len(
            infrared_files
        )

        self.annotation_file_count = len(
            annotation_files
        )

    # ==================================================================
    # BUILD SAMPLES
    # ==================================================================

    def _build_samples(self):

        visible_files = {
            p.stem: p
            for p in self._image_files(
                self.visible_dir
            )
        }

        infrared_files = {
            p.stem: p
            for p in self._image_files(
                self.infrared_dir
            )
        }

        annotation_files = {
            p.stem: p
            for p in self._annotation_files()
        }

        visible_stems = set(
            visible_files.keys()
        )

        infrared_stems = set(
            infrared_files.keys()
        )

        annotation_stems = set(
            annotation_files.keys()
        )

        # ==============================================================
        # COMMON
        # ==============================================================

        common = sorted(
            visible_stems
            &
            infrared_stems
            &
            annotation_stems
        )

        # ==============================================================
        # MISSING
        # ==============================================================

        self.missing_ir_count = len(
            visible_stems - infrared_stems
        )

        self.missing_visible_count = len(
            infrared_stems - visible_stems
        )

        self.missing_annotation_count = len(
            (
                visible_stems
                &
                infrared_stems
            )
            -
            annotation_stems
        )

        # ==============================================================
        # WARNINGS
        # ==============================================================

        if self.missing_ir_count > 0:

            print(
                f"[WARNING] {self.missing_ir_count} "
                f"visible images have no infrared pair."
            )

        if self.missing_visible_count > 0:

            print(
                f"[WARNING] {self.missing_visible_count} "
                f"infrared images have no visible pair."
            )

        if self.missing_annotation_count > 0:

            print(
                f"[WARNING] {self.missing_annotation_count} "
                f"RGB-IR pairs have no annotation."
            )

        # ==============================================================
        # CREATE SAMPLES
        # ==============================================================

        for stem in common:

            annotation_path = annotation_files[stem]

            target = self._read_txt(
                annotation_path,
                visible_files[stem]
            )

            # ----------------------------------------------------------
            # Training:
            # skip samples without valid objects
            # ----------------------------------------------------------

            if self.training:

                if target["boxes"].numel() == 0:

                    self.empty_annotations += 1

                    continue

            self.samples.append(
                {
                    "visible": visible_files[stem],
                    "infrared": infrared_files[stem],
                    "annotation": annotation_path,
                    "stem": stem,
                }
            )

    # ==================================================================
    # EMPTY TARGET
    # ==================================================================

    @staticmethod
    def _empty_target():

        return {
            "boxes": torch.zeros(
                (0, 4),
                dtype=torch.float32
            ),

            "labels": torch.zeros(
                (0,),
                dtype=torch.int64
            ),

            "area": torch.zeros(
                (0,),
                dtype=torch.float32
            ),

            "iscrowd": torch.zeros(
                (0,),
                dtype=torch.int64
            ),

            "class_names": [],

            "unknown_classes": [],
        }

    # ==================================================================
    # READ TXT ANNOTATION
    # ==================================================================

    def _read_txt(
        self,
        annotation_path,
        image_path
    ):

        boxes = []
        labels = []
        class_names = []
        unknown_classes = []

        # ==============================================================
        # IMAGE SIZE
        # ==============================================================

        try:

            with Image.open(image_path) as image:

                image_width, image_height = image.size

        except Exception:

            return self._empty_target()

        # ==============================================================
        # READ TXT
        # ==============================================================

        try:

            with open(
                annotation_path,
                "r",
                encoding="utf-8"
            ) as file:

                lines = file.readlines()

        except Exception:

            self.invalid_annotation_files += 1

            return self._empty_target()

        # ==============================================================
        # PROCESS LINES
        # ==============================================================

        for line in lines:

            line = line.strip()

            if not line:
                continue

            line = line.replace(",", " ")

            parts = line.split()

            if len(parts) < 5:
                continue

            # ==========================================================
            # CLASS
            # ==========================================================

            class_token = parts[0]

            try:

                if class_token.replace(
                    ".",
                    "",
                    1
                ).isdigit():

                    raw_class_id = int(
                        float(class_token)
                    )

                    # --------------------------------------------------
                    # IMPORTANT:
                    #
                    # SICDA LLVIP mapping:
                    #
                    # 1 = person
                    # 2 = car
                    # 3 = bike
                    #
                    # Background 0 is ignored.
                    # --------------------------------------------------

                    class_id = raw_class_id

                else:

                    class_name = (
                        class_token
                        .strip()
                        .lower()
                    )

                    class_id = self.CLASS_MAP.get(
                        class_name,
                        -1
                    )

            except (
                ValueError,
                TypeError
            ):

                continue

            # ==========================================================
            # CLASS VALIDATION
            # ==========================================================

            if class_id not in self.VALID_CLASSES:

                unknown_classes.append(
                    class_token
                )

                continue

            # ==========================================================
            # COORDINATES
            # ==========================================================

            try:

                x1 = float(parts[1])
                y1 = float(parts[2])
                x2 = float(parts[3])
                y2 = float(parts[4])

            except (
                ValueError,
                TypeError
            ):

                continue

            # ==========================================================
            # YOLO OR ABSOLUTE
            # ==========================================================

            all_normalized = (
                0.0 <= x1 <= 1.0
                and
                0.0 <= y1 <= 1.0
                and
                0.0 <= x2 <= 1.0
                and
                0.0 <= y2 <= 1.0
            )

            if all_normalized:

                # ------------------------------------------------------
                # YOLO
                # xc, yc, width, height
                # ------------------------------------------------------

                xc = x1 * image_width
                yc = y1 * image_height

                box_width = x2 * image_width
                box_height = y2 * image_height

                xmin = (
                    xc
                    -
                    box_width / 2.0
                )

                ymin = (
                    yc
                    -
                    box_height / 2.0
                )

                xmax = (
                    xc
                    +
                    box_width / 2.0
                )

                ymax = (
                    yc
                    +
                    box_height / 2.0
                )

            else:

                # ------------------------------------------------------
                # Absolute
                # xmin, ymin, xmax, ymax
                # ------------------------------------------------------

                xmin = x1
                ymin = y1
                xmax = x2
                ymax = y2

            # ==========================================================
            # CLIP
            # ==========================================================

            xmin = max(
                0.0,
                min(
                    xmin,
                    image_width - 1
                )
            )

            ymin = max(
                0.0,
                min(
                    ymin,
                    image_height - 1
                )
            )

            xmax = max(
                0.0,
                min(
                    xmax,
                    image_width - 1
                )
            )

            ymax = max(
                0.0,
                min(
                    ymax,
                    image_height - 1
                )
            )

            # ==========================================================
            # VALID BOX
            # ==========================================================

            if xmax <= xmin:
                continue

            if ymax <= ymin:
                continue

            # ==========================================================
            # STORE
            # ==========================================================

            boxes.append(
                [
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ]
            )

            labels.append(
                class_id
            )

            class_names.append(
                self.CLASS_NAMES_LOCAL[
                    class_id
                ]
            )

        # ==============================================================
        # CONVERT
        # ==============================================================

        if boxes:

            boxes = torch.tensor(
                boxes,
                dtype=torch.float32
            )

            labels = torch.tensor(
                labels,
                dtype=torch.int64
            )

            area = (
                boxes[:, 2]
                -
                boxes[:, 0]
            ) * (
                boxes[:, 3]
                -
                boxes[:, 1]
            )

        else:

            boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

            labels = torch.zeros(
                (0,),
                dtype=torch.int64
            )

            area = torch.zeros(
                (0,),
                dtype=torch.float32
            )

        # ==============================================================
        # RETURN
        # ==============================================================

        return {

            "boxes": boxes,

            "labels": labels,

            "area": area,

            "iscrowd": torch.zeros(
                len(labels),
                dtype=torch.int64
            ),

            "class_names": class_names,

            "unknown_classes": unknown_classes,
        }

    # ==================================================================
    # READABLE IMAGE
    # ==================================================================

    @staticmethod
    def _readable(path):

        try:

            if not os.path.exists(path):
                return False

            if os.path.getsize(path) < 100:
                return False

            with Image.open(path) as image:
                image.verify()

            return True

        except Exception:

            return False

    # ==================================================================
    # FILTER CORRUPT
    # ==================================================================

    def _filter_readable(self):

        good = []
        skipped = 0

        for sample in self.samples:

            visible_ok = self._readable(
                sample["visible"]
            )

            infrared_ok = self._readable(
                sample["infrared"]
            )

            if visible_ok and infrared_ok:

                good.append(sample)

            else:

                skipped += 1

        self.samples = good

        self.corrupt_pairs_skipped = skipped

        if skipped > 0:

            print(
                f"[WARNING] Corrupt/unreadable "
                f"RGB-IR pairs skipped: {skipped}"
            )

    # ==================================================================
    # CALCULATE STATISTICS
    # ==================================================================

    def _calculate_statistics(self):

        self.total_boxes = 0

        self.person_boxes = 0
        self.car_boxes = 0
        self.bike_boxes = 0

        self.images_with_person = 0
        self.images_with_car = 0
        self.images_with_bike = 0

        self.unknown_objects = Counter()

        for sample in self.samples:

            target = self._read_txt(
                sample["annotation"],
                sample["visible"]
            )

            labels = target["labels"]

            self.total_boxes += len(labels)

            # ----------------------------------------------------------
            # PERSON
            # ----------------------------------------------------------

            person_count = int(
                (labels == 1).sum().item()
            )

            self.person_boxes += person_count

            if person_count > 0:
                self.images_with_person += 1

            # ----------------------------------------------------------
            # CAR
            # ----------------------------------------------------------

            car_count = int(
                (labels == 2).sum().item()
            )

            self.car_boxes += car_count

            if car_count > 0:
                self.images_with_car += 1

            # ----------------------------------------------------------
            # BIKE
            # ----------------------------------------------------------

            bike_count = int(
                (labels == 3).sum().item()
            )

            self.bike_boxes += bike_count

            if bike_count > 0:
                self.images_with_bike += 1

            # ----------------------------------------------------------
            # UNKNOWN
            # ----------------------------------------------------------

            for unknown in target[
                "unknown_classes"
            ]:

                self.unknown_objects[
                    unknown
                ] += 1

    # ==================================================================
    # PRINT INFORMATION
    # ==================================================================

    def _print_dataset_information(self):

        print()

        print("=" * 78)

        print(
            "                    LLVIP SOURCE DATASET"
        )

        print("=" * 78)

        print()

        print(
            f"Visible directory      : "
            f"{self.visible_dir}"
        )

        print(
            f"Infrared directory     : "
            f"{self.infrared_dir}"
        )

        print(
            f"Annotation directory   : "
            f"{self.annotation_dir}"
        )

        print()

        print(
            "[RAW DATASET FILES]"
        )

        print(
            f"Visible images         : "
            f"{self.visible_image_count}"
        )

        print(
            f"Infrared images        : "
            f"{self.infrared_image_count}"
        )

        print(
            f"TXT annotation files   : "
            f"{self.annotation_file_count}"
        )

        print()

        print(
            "[DATASET SAMPLES]"
        )

        print(
            f"RGB-IR pairs with TXT  : "
            f"{len(self.samples)}"
        )

        print(
            f"Training mode          : "
            f"{self.training}"
        )

        print(
            f"Corrupt filtering      : "
            f"{self.filter_corrupt}"
        )

        print(
            f"Corrupt pairs skipped  : "
            f"{self.corrupt_pairs_skipped}"
        )

        print(
            f"Empty annotations     : "
            f"{self.empty_annotations}"
        )

        print()

        print(
            "[DATASET MATCHING]"
        )

        print(
            f"Missing infrared      : "
            f"{self.missing_ir_count}"
        )

        print(
            f"Missing visible       : "
            f"{self.missing_visible_count}"
        )

        print(
            f"Missing annotation    : "
            f"{self.missing_annotation_count}"
        )

        print()

        print(
            "[CLASS MAPPING]"
        )

        for class_id, class_name in self.CLASS_NAMES_LOCAL.items():

            print(
                f"{class_id} -> {class_name}"
            )

        print()

        print(
            "[ANNOTATION STATISTICS]"
        )

        print(
            f"Total bounding boxes     : "
            f"{self.total_boxes}"
        )

        print(
            f"Person boxes             : "
            f"{self.person_boxes}"
        )

        print(
            f"Car boxes                : "
            f"{self.car_boxes}"
        )

        print(
            f"Bike boxes               : "
            f"{self.bike_boxes}"
        )

        print()

        print(
            "[IMAGE-LEVEL CLASS PRESENCE]"
        )

        print(
            f"Images with person       : "
            f"{self.images_with_person}"
        )

        print(
            f"Images with car          : "
            f"{self.images_with_car}"
        )

        print(
            f"Images with bike         : "
            f"{self.images_with_bike}"
        )

        print()

        print(
            "[UNKNOWN CLASSES]"
        )

        if self.unknown_objects:

            for name, count in sorted(
                self.unknown_objects.items()
            ):

                print(
                    f"{name:<25}: {count}"
                )

        else:

            print("None")

        print()

        print("=" * 78)

    # ==================================================================
    # LENGTH
    # ==================================================================

    def __len__(self):

        return len(self.samples)

    # ==================================================================
    # GET ITEM
    # ==================================================================

    def __getitem__(self, index):

        if len(self.samples) == 0:

            raise RuntimeError(
                "LLVIP dataset is empty."
            )

        max_tries = min(
            20,
            len(self.samples)
        )

        last_error = None

        for attempt in range(max_tries):

            idx = (
                index + attempt
            ) % len(self.samples)

            sample = self.samples[idx]

            try:

                # ======================================================
                # RGB
                # ======================================================

                visible = Image.open(
                    sample["visible"]
                ).convert("RGB")

                # ======================================================
                # THERMAL
                # ======================================================

                infrared = Image.open(
                    sample["infrared"]
                ).convert("L")

                # ======================================================
                # TARGET
                # ======================================================

                target = self._read_txt(
                    sample["annotation"],
                    sample["visible"]
                )

            except (
                OSError,
                UnidentifiedImageError
            ) as exc:

                last_error = exc

                continue

            # ==========================================================
            # IMAGE ID
            # ==========================================================

            target["image_id"] = torch.tensor(
                [idx],
                dtype=torch.int64
            )

            # ==========================================================
            # TRANSFORMS
            # ==========================================================

            if self.transforms is not None:

                visible, infrared, target = (
                    self.transforms(
                        visible,
                        infrared,
                        target
                    )
                )

            else:

                visible = F.to_tensor(
                    visible
                )

                infrared = F.to_tensor(
                    infrared
                )

            # ==========================================================
            # RETURN
            # ==========================================================

            return {

                "rgb": visible,

                "thermal": infrared,

                "target": target,

                "rgb_path": str(
                    sample["visible"]
                ),

                "thermal_path": str(
                    sample["infrared"]
                ),

                "annotation_path": str(
                    sample["annotation"]
                ),

                "domain": 0,
            }

        raise RuntimeError(
            f"Failed to load LLVIP sample "
            f"after {max_tries} attempts. "
            f"Last error: {last_error}"
        )

    # ==================================================================
    # PUBLIC STATISTICS
    # ==================================================================

    def print_statistics(self):

        print()

        print("=" * 78)

        print(
            "                  LLVIP FINAL STATISTICS"
        )

        print("=" * 78)

        print()

        print(
            f"LLVIP RGB-IR pairs    : "
            f"{len(self.samples)}"
        )

        print(
            f"Visible images        : "
            f"{self.visible_image_count}"
        )

        print(
            f"Infrared images       : "
            f"{self.infrared_image_count}"
        )

        print(
            f"TXT annotations       : "
            f"{self.annotation_file_count}"
        )

        print()

        print(
            f"Total bounding boxes  : "
            f"{self.total_boxes}"
        )

        print(
            f"Person boxes          : "
            f"{self.person_boxes}"
        )

        print(
            f"Car boxes             : "
            f"{self.car_boxes}"
        )

        print(
            f"Bike boxes            : "
            f"{self.bike_boxes}"
        )

        print()

        print(
            f"Images with person    : "
            f"{self.images_with_person}"
        )

        print(
            f"Images with car       : "
            f"{self.images_with_car}"
        )

        print(
            f"Images with bike      : "
            f"{self.images_with_bike}"
        )

        print()

        print(
            f"Classes               : "
            f"{Config.CLASS_NAMES}"
        )

        print(
            f"NUM_CLASSES           : "
            f"{Config.NUM_CLASSES}"
        )

        print()

        print("=" * 78)


# ======================================================================
# CONFIG PATH HELPER
# ======================================================================

def get_llvip_train_paths():

    """
    Use the exact LLVIP training paths from Config.

    Preferred names:

        SOURCE_VISIBLE_TRAIN_DIR
        SOURCE_INFRARED_TRAIN_DIR
        SOURCE_ANNOTATION_TRAIN_DIR

    Backward-compatible IR alias:

        SOURCE_IR_TRAIN_DIR
    """

    visible_dir = getattr(
        Config,
        "SOURCE_VISIBLE_TRAIN_DIR"
    )

    if hasattr(
        Config,
        "SOURCE_INFRARED_TRAIN_DIR"
    ):

        infrared_dir = (
            Config.SOURCE_INFRARED_TRAIN_DIR
        )

    elif hasattr(
        Config,
        "SOURCE_IR_TRAIN_DIR"
    ):

        infrared_dir = (
            Config.SOURCE_IR_TRAIN_DIR
        )

    else:

        raise AttributeError(
            "Config must contain either "
            "SOURCE_INFRARED_TRAIN_DIR or "
            "SOURCE_IR_TRAIN_DIR."
        )

    if hasattr(
        Config,
        "SOURCE_ANNOTATION_TRAIN_DIR"
    ):

        annotation_dir = (
            Config.SOURCE_ANNOTATION_TRAIN_DIR
        )

    elif hasattr(
        Config,
        "SOURCE_ANNOTATION_DIR"
    ):

        annotation_dir = (
            Config.SOURCE_ANNOTATION_DIR
        )

    else:

        raise AttributeError(
            "Config must contain "
            "SOURCE_ANNOTATION_TRAIN_DIR."
        )

    return (
        visible_dir,
        infrared_dir,
        annotation_dir
    )


# ======================================================================
# MAIN TEST
# ======================================================================

if __name__ == "__main__":

    print()

    print("=" * 78)

    print(
        "                  TESTING LLVIP DATASET"
    )

    print("=" * 78)

    print()

    print(
        f"Configured classes : "
        f"{Config.CLASS_NAMES}"
    )

    print(
        f"Number of classes  : "
        f"{Config.NUM_CLASSES}"
    )

    print()

    # ==============================================================
    # GET CONFIG PATHS
    # ==============================================================

    (
        visible_dir,
        infrared_dir,
        annotation_dir
    ) = get_llvip_train_paths()

    print(
        "LLVIP CONFIGURED PATHS"
    )

    print("-" * 78)

    print(
        f"Visible train      : {visible_dir}"
    )

    print(
        f"Infrared train     : {infrared_dir}"
    )

    print(
        f"Annotation train   : {annotation_dir}"
    )

    print()

    # ==============================================================
    # CREATE DATASET
    # ==============================================================

    dataset = LLVIPSourceDataset(

        visible_dir=visible_dir,

        infrared_dir=infrared_dir,

        annotation_dir=annotation_dir,

        transforms=None,

        training=True,

        filter_corrupt=True,
    )

    # ==============================================================
    # FINAL STATISTICS
    # ==============================================================

    dataset.print_statistics()

    # ==============================================================
    # SAMPLE TEST
    # ==============================================================

    print()

    print("=" * 78)

    print(
        "                  LLVIP SAMPLE TEST"
    )

    print("=" * 78)

    sample = dataset[0]

    print()

    print(
        "RGB shape       :",
        tuple(
            sample["rgb"].shape
        )
    )

    print(
        "Thermal shape   :",
        tuple(
            sample["thermal"].shape
        )
    )

    print(
        "Boxes           :",
        len(
            sample["target"]["boxes"]
        )
    )

    print(
        "Labels          :",
        sample["target"]["labels"].tolist()
    )

    print(
        "Class names     :",
        sample["target"]["class_names"]
    )

    print(
        "RGB path        :",
        sample["rgb_path"]
    )

    print(
        "Thermal path    :",
        sample["thermal_path"]
    )

    print(
        "Annotation path :",
        sample["annotation_path"]
    )

    print(
        "Domain          :",
        sample["domain"]
    )

    print()

    print("=" * 78)

    print(
        "LLVIP DATASET TEST COMPLETED"
    )

    print("=" * 78)