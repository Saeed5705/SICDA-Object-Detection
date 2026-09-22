import os
import re
import json
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import torchvision.transforms.functional as F

from config import Config


# ======================================================================
# FLIR TARGET DATASET
# ======================================================================

class FLIRTargetDataset(Dataset):

    """
    FLIR ADAS v2 RGB-Thermal Target Dataset.

    Domain:
        FLIR = Target Domain = 1

    Classes:
        0 = Background
        1 = Person
        2 = Car
        3 = Bicycle/Bike

    Output:
        {
            "rgb": Tensor[C,H,W],
            "thermal": Tensor[1,H,W],
            "rgb_path": str,
            "thermal_path": str,
            "domain": 1,

            "target": {
                "boxes": Tensor[N,4],
                "labels": Tensor[N],
                "image_id": Tensor[1],
                "area": Tensor[N],
                "iscrowd": Tensor[N]
            }
        }
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

    CLASS_NAME_TO_ID = {

        # --------------------------------------------------------------
        # PERSON
        # --------------------------------------------------------------

        "person": 1,
        "pedestrian": 1,
        "pedestrians": 1,
        "people": 1,
        "human": 1,
        "humans": 1,

        # --------------------------------------------------------------
        # CAR
        # --------------------------------------------------------------

        "car": 2,
        "cars": 2,
        "vehicle": 2,
        "vehicles": 2,
        "automobile": 2,

        # --------------------------------------------------------------
        # BICYCLE / BIKE
        # --------------------------------------------------------------

        "bicycle": 3,
        "bicycles": 3,
        "bike": 3,
        "bikes": 3,
        "cycle": 3,
        "cycles": 3,
        "motorcycle": 3,
        "motorbike": 3,
    }

    CLASS_NAMES_LOCAL = {

        0: "background",
        1: "person",
        2: "car",
        3: "bike",

    }

    # ==================================================================
    # INIT
    # ==================================================================

    def __init__(
        self,
        rgb_dir,
        thermal_dir,
        transforms=None,
        training=True,
        annotation_file=None,
        return_targets=False,
        filter_corrupt=True,
    ):

        super().__init__()

        # --------------------------------------------------------------
        # PATHS
        # --------------------------------------------------------------

        self.rgb_dir = Path(rgb_dir)

        self.thermal_dir = Path(thermal_dir)

        self.transforms = transforms

        self.training = bool(training)

        self.annotation_file = (
            Path(annotation_file)
            if annotation_file is not None
            else None
        )

        self.return_targets = bool(
            return_targets
        )

        self.filter_corrupt = bool(
            filter_corrupt
        )

        # --------------------------------------------------------------
        # DATA SAMPLES
        # --------------------------------------------------------------

        self.samples = []

        # --------------------------------------------------------------
        # COCO DATA
        # --------------------------------------------------------------

        self.coco_images = {}

        self.coco_anns = defaultdict(list)

        self.image_lookup = {}

        self.category_to_class = {}

        # --------------------------------------------------------------
        # STATISTICS
        # --------------------------------------------------------------

        self.class_statistics = {

            1: 0,
            2: 0,
            3: 0,

        }

        self.images_with_annotations = set()

        self.corrupt_pairs_skipped = 0

        self.annotation_images_matched = 0

        self.annotation_images_unmatched = 0

        # ==============================================================
        # VALIDATE DIRECTORIES
        # ==============================================================

        self._validate_directories()

        # ==============================================================
        # RESOLVE ACTUAL IMAGE ROOTS
        # ==============================================================

        self.rgb_dir = self._resolve_image_root(
            self.rgb_dir
        )

        self.thermal_dir = self._resolve_image_root(
            self.thermal_dir
        )

        # ==============================================================
        # BUILD RGB-THERMAL PAIRS
        # ==============================================================

        self._build_pairs()

        # ==============================================================
        # FILTER CORRUPT PAIRS
        # ==============================================================

        if self.filter_corrupt:

            self._filter_readable_pairs()

        # ==============================================================
        # LOAD ANNOTATIONS
        # ==============================================================

        if self.return_targets:

            self._load_annotations()

        # ==============================================================
        # PRINT INFORMATION
        # ==============================================================

        self._print_information()

        # ==============================================================
        # SAFETY CHECK
        # ==============================================================

        if not self.samples:

            raise RuntimeError(
                "\n"
                "No valid FLIR RGB-Thermal pairs found.\n\n"
                f"RGB directory    : {self.rgb_dir}\n"
                f"Thermal directory: {self.thermal_dir}\n"
            )

    # ==================================================================
    # VALIDATE DIRECTORIES
    # ==================================================================

    def _validate_directories(self):

        if not self.rgb_dir.exists():

            raise FileNotFoundError(
                "\nFLIR RGB directory not found:\n"
                f"{self.rgb_dir}"
            )

        if not self.rgb_dir.is_dir():

            raise NotADirectoryError(
                "\nFLIR RGB path is not a directory:\n"
                f"{self.rgb_dir}"
            )

        if not self.thermal_dir.exists():

            raise FileNotFoundError(
                "\nFLIR Thermal directory not found:\n"
                f"{self.thermal_dir}"
            )

        if not self.thermal_dir.is_dir():

            raise NotADirectoryError(
                "\nFLIR Thermal path is not a directory:\n"
                f"{self.thermal_dir}"
            )

    # ==================================================================
    # LIST IMAGES
    # ==================================================================

    def _list_images(
        self,
        folder,
        recursive=True,
    ):

        folder = Path(folder)

        if not folder.exists():

            return []

        if recursive:

            iterator = folder.rglob("*")

        else:

            iterator = folder.glob("*")

        images = []

        for path in iterator:

            if not path.is_file():

                continue

            if path.suffix.lower() not in self.IMAGE_EXTS:

                continue

            images.append(path)

        return sorted(images)

    # ==================================================================
    # RESOLVE IMAGE ROOT
    # ==================================================================

    def _resolve_image_root(
        self,
        folder,
    ):

        folder = Path(folder)

        # --------------------------------------------------------------
        # Direct images
        # --------------------------------------------------------------

        direct_images = self._list_images(
            folder,
            recursive=False,
        )

        if direct_images:

            return folder

        # --------------------------------------------------------------
        # Common FLIR folders
        # --------------------------------------------------------------

        common_names = (

            "data",
            "images",
            "rgb",
            "visible",
            "thermal",
            "infrared",
            "ir",

        )

        for name in common_names:

            candidate = folder / name

            if not candidate.is_dir():

                continue

            if self._list_images(
                candidate,
                recursive=True,
            ):

                return candidate

        # --------------------------------------------------------------
        # Recursive search
        # --------------------------------------------------------------

        for sub in sorted(folder.rglob("*")):

            if not sub.is_dir():

                continue

            if self._list_images(
                sub,
                recursive=False,
            ):

                return sub

        return folder

    # ==================================================================
    # NORMALIZE NAME
    # ==================================================================

    @staticmethod
    def _normalize_name(path):

        stem = Path(path).stem.lower()

        # Remove common modality identifiers

        replacements = [

            "_rgb",
            "-rgb",
            "rgb_",
            "rgb-",

            "_thermal",
            "-thermal",
            "thermal_",
            "thermal-",

            "_infrared",
            "-infrared",
            "infrared_",
            "infrared-",

            "_ir",
            "-ir",
            "ir_",
            "ir-",

        ]

        name = stem

        for item in replacements:

            name = name.replace(
                item,
                "",
            )

        # Keep alphanumeric characters

        name = re.sub(
            r"[^a-z0-9]+",
            "",
            name,
        )

        return name

    # ==================================================================
    # FRAME KEY
    # ==================================================================

    @staticmethod
    def _frame_key(path):

        stem = Path(path).stem

        # --------------------------------------------------------------
        # frame-000123
        # frame_000123
        # --------------------------------------------------------------

        match = re.search(
            r"frame[-_]?(\d+)",
            stem,
            flags=re.IGNORECASE,
        )

        if match:

            number = match.group(1)

            return (
                number.lstrip("0")
                or "0"
            )

        # --------------------------------------------------------------
        # I000123
        # image000123
        # img000123
        # --------------------------------------------------------------

        match = re.search(
            r"(?:i|img|image)[-_]?(\d+)$",
            stem,
            flags=re.IGNORECASE,
        )

        if match:

            number = match.group(1)

            return (
                number.lstrip("0")
                or "0"
            )

        # --------------------------------------------------------------
        # Any trailing number
        # --------------------------------------------------------------

        match = re.search(
            r"(\d+)$",
            stem,
        )

        if match:

            number = match.group(1)

            return (
                number.lstrip("0")
                or "0"
            )

        return stem.lower()

    # ==================================================================
    # BUILD RGB THERMAL PAIRS
    # ==================================================================

    def _build_pairs(self):

        rgb_files = self._list_images(
            self.rgb_dir,
            recursive=True,
        )

        thermal_files = self._list_images(
            self.thermal_dir,
            recursive=True,
        )

        print()
        print("=" * 78)
        print("FLIR IMAGE DISCOVERY")
        print("=" * 78)

        print(
            f"RGB images found     : {len(rgb_files)}"
        )

        print(
            f"Thermal images found : {len(thermal_files)}"
        )

        # ==============================================================
        # METHOD 1
        # EXACT STEM
        # ==============================================================

        thermal_by_stem = {}

        for thermal in thermal_files:

            key = thermal.stem.lower()

            if key not in thermal_by_stem:

                thermal_by_stem[key] = thermal

        matched_rgb = set()

        for rgb in rgb_files:

            key = rgb.stem.lower()

            thermal = thermal_by_stem.get(
                key
            )

            if thermal is None:

                continue

            self.samples.append({

                "rgb": rgb,

                "thermal": thermal,

            })

            matched_rgb.add(
                rgb
            )

        # ==============================================================
        # METHOD 2
        # NORMALIZED NAME
        # ==============================================================

        if len(self.samples) < len(rgb_files):

            thermal_by_normalized = {}

            for thermal in thermal_files:

                key = self._normalize_name(
                    thermal
                )

                if key:

                    if key not in thermal_by_normalized:

                        thermal_by_normalized[key] = thermal

            for rgb in rgb_files:

                if rgb in matched_rgb:

                    continue

                key = self._normalize_name(
                    rgb
                )

                thermal = thermal_by_normalized.get(
                    key
                )

                if thermal is None:

                    continue

                self.samples.append({

                    "rgb": rgb,

                    "thermal": thermal,

                })

                matched_rgb.add(
                    rgb
                )

        # ==============================================================
        # METHOD 3
        # FRAME KEY
        # ==============================================================

        if len(self.samples) < len(rgb_files):

            thermal_by_key = {}

            for thermal in thermal_files:

                key = self._frame_key(
                    thermal
                )

                if key not in thermal_by_key:

                    thermal_by_key[key] = thermal

            for rgb in rgb_files:

                if rgb in matched_rgb:

                    continue

                key = self._frame_key(
                    rgb
                )

                thermal = thermal_by_key.get(
                    key
                )

                if thermal is None:

                    continue

                self.samples.append({

                    "rgb": rgb,

                    "thermal": thermal,

                })

                matched_rgb.add(
                    rgb
                )

        # ==============================================================
        # REMOVE DUPLICATES
        # ==============================================================

        unique_pairs = []

        seen = set()

        for sample in self.samples:

            key = (

                str(sample["rgb"]).lower(),

                str(sample["thermal"]).lower(),

            )

            if key in seen:

                continue

            seen.add(key)

            unique_pairs.append(
                sample
            )

        self.samples = unique_pairs

        # ==============================================================
        # PRINT RESULT
        # ==============================================================

        print(
            f"RGB-Thermal pairs   : {len(self.samples)}"
        )

        unmatched = len(rgb_files) - len(
            matched_rgb
        )

        print(
            f"Unmatched RGB       : {max(unmatched, 0)}"
        )

        print("=" * 78)

    # ==================================================================
    # CHECK IMAGE READABILITY
    # ==================================================================

    @staticmethod
    def _readable(path):

        try:

            path = Path(path)

            if not path.exists():

                return False

            if path.stat().st_size < 100:

                return False

            with Image.open(path) as image:

                image.verify()

            return True

        except Exception:

            return False

    # ==================================================================
    # FILTER CORRUPT
    # ==================================================================

    def _filter_readable_pairs(self):

        good_pairs = []

        skipped = 0

        for sample in self.samples:

            rgb_ok = self._readable(
                sample["rgb"]
            )

            thermal_ok = self._readable(
                sample["thermal"]
            )

            if rgb_ok and thermal_ok:

                good_pairs.append(
                    sample
                )

            else:

                skipped += 1

        self.samples = good_pairs

        self.corrupt_pairs_skipped = skipped

        if skipped > 0:

            print(
                "[WARNING] "
                f"{skipped} corrupt/unreadable "
                "RGB-Thermal pairs skipped."
            )

    # ==================================================================
    # LOAD COCO ANNOTATIONS
    # ==================================================================

    def _load_annotations(self):

        if self.annotation_file is None:

            raise FileNotFoundError(
                "\n"
                "annotation_file is required "
                "when return_targets=True."
            )

        if not self.annotation_file.exists():

            raise FileNotFoundError(
                "\n"
                "FLIR annotation file not found:\n"
                f"{self.annotation_file}"
            )

        print()
        print("=" * 78)
        print("LOADING FLIR COCO ANNOTATIONS")
        print("=" * 78)

        print(
            f"Annotation JSON : {self.annotation_file}"
        )

        # ==============================================================
        # LOAD JSON
        # ==============================================================

        try:

            with open(
                self.annotation_file,
                "r",
                encoding="utf-8",
            ) as file:

                coco = json.load(file)

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "\nInvalid COCO JSON annotation file:\n"
                f"{self.annotation_file}\n\n"
                f"JSON error: {exc}"
            )

        # ==============================================================
        # CATEGORIES
        # ==============================================================

        categories = coco.get(
            "categories",
            []
        )

        print(
            f"\nCOCO categories : {len(categories)}"
        )

        print("-" * 78)

        for category in categories:

            category_id = category.get(
                "id"
            )

            category_name = str(
                category.get(
                    "name",
                    ""
                )
            ).strip().lower()

            if category_id is None:

                continue

            try:

                category_id = int(
                    category_id
                )

            except (
                ValueError,
                TypeError,
            ):

                continue

            class_id = self.CLASS_NAME_TO_ID.get(
                category_name
            )

            if class_id is None:

                print(
                    f"{category_id:>5} -> "
                    f"{category_name:<20} -> IGNORED"
                )

                continue

            self.category_to_class[
                category_id
            ] = class_id

            class_name = self.CLASS_NAMES_LOCAL[
                class_id
            ]

            print(
                f"{category_id:>5} -> "
                f"{category_name:<20} -> "
                f"{class_id} ({class_name})"
            )

        if not self.category_to_class:

            raise RuntimeError(
                "\n"
                "No Person/Car/Bicycle categories "
                "were found in FLIR COCO annotations.\n\n"
                f"Annotation file: {self.annotation_file}"
            )

        # ==============================================================
        # IMAGES
        # ==============================================================

        images = coco.get(
            "images",
            []
        )

        for image_info in images:

            image_id = image_info.get(
                "id"
            )

            if image_id is None:

                continue

            try:

                image_id = int(
                    image_id
                )

            except (
                ValueError,
                TypeError,
            ):

                continue

            self.coco_images[
                image_id
            ] = image_info

            filename = str(
                image_info.get(
                    "file_name",
                    ""
                )
            ).replace(
                "\\",
                "/"
            )

            if not filename:

                continue

            base = os.path.basename(
                filename
            ).lower()

            stem = Path(
                base
            ).stem.lower()

            normalized = self._normalize_name(
                filename
            )

            frame_key = self._frame_key(
                filename
            )

            # ----------------------------------------------------------
            # Exact filename
            # ----------------------------------------------------------

            self.image_lookup[
                ("base", base)
            ] = image_id

            # ----------------------------------------------------------
            # Stem
            # ----------------------------------------------------------

            self.image_lookup[
                ("stem", stem)
            ] = image_id

            # ----------------------------------------------------------
            # Normalized name
            # ----------------------------------------------------------

            if normalized:

                self.image_lookup[
                    ("normalized", normalized)
                ] = image_id

            # ----------------------------------------------------------
            # Frame key
            # ----------------------------------------------------------

            self.image_lookup[
                ("key", frame_key)
            ] = image_id

        # ==============================================================
        # ANNOTATIONS
        # ==============================================================

        annotations = coco.get(
            "annotations",
            []
        )

        total_used = 0

        total_ignored = 0

        for ann in annotations:

            category_id = ann.get(
                "category_id"
            )

            if category_id is None:

                total_ignored += 1

                continue

            try:

                category_id = int(
                    category_id
                )

            except (
                ValueError,
                TypeError,
            ):

                total_ignored += 1

                continue

            if category_id not in self.category_to_class:

                total_ignored += 1

                continue

            # ----------------------------------------------------------
            # Ignore crowd
            # ----------------------------------------------------------

            if int(
                ann.get(
                    "iscrowd",
                    0
                )
            ) != 0:

                total_ignored += 1

                continue

            image_id = ann.get(
                "image_id"
            )

            if image_id is None:

                total_ignored += 1

                continue

            try:

                image_id = int(
                    image_id
                )

            except (
                ValueError,
                TypeError,
            ):

                total_ignored += 1

                continue

            # ----------------------------------------------------------
            # Save annotation
            # ----------------------------------------------------------

            self.coco_anns[
                image_id
            ].append(
                ann
            )

            self.images_with_annotations.add(
                image_id
            )

            class_id = self.category_to_class[
                category_id
            ]

            self.class_statistics[
                class_id
            ] += 1

            total_used += 1

        # ==============================================================
        # STATISTICS
        # ==============================================================

        print()
        print("COCO DATASET STATISTICS")
        print("-" * 78)

        print(
            f"COCO images         : {len(self.coco_images)}"
        )

        print(
            f"Person annotations  : "
            f"{self.class_statistics[1]}"
        )

        print(
            f"Car annotations     : "
            f"{self.class_statistics[2]}"
        )

        print(
            f"Bicycle annotations : "
            f"{self.class_statistics[3]}"
        )

        print(
            f"Total used boxes    : {total_used}"
        )

        print(
            f"Ignored annotations : {total_ignored}"
        )

        print(
            f"Images with GT      : "
            f"{len(self.images_with_annotations)}"
        )

        print("=" * 78)

    # ==================================================================
    # FIND COCO IMAGE ID
    # ==================================================================

    def _find_image_id(
        self,
        rgb_path,
    ):

        path = Path(rgb_path)

        candidates = [

            (
                "base",
                path.name.lower()
            ),

            (
                "stem",
                path.stem.lower()
            ),

            (
                "normalized",
                self._normalize_name(path)
            ),

            (
                "key",
                self._frame_key(path)
            ),

        ]

        for lookup_key in candidates:

            value = lookup_key[1]

            if not value:

                continue

            image_id = self.image_lookup.get(
                lookup_key
            )

            if image_id is not None:

                return image_id

        return None

    # ==================================================================
    # CREATE TARGET
    # ==================================================================

    def _target_for_rgb(
        self,
        rgb_path,
        fallback_index,
    ):

        image_id = self._find_image_id(
            rgb_path
        )

        if image_id is not None:

            annotations = self.coco_anns.get(
                image_id,
                []
            )

        else:

            annotations = []

        boxes = []

        labels = []

        # ==============================================================
        # COCO BBOX
        #
        # [x, y, width, height]
        #
        # converted to:
        #
        # [xmin, ymin, xmax, ymax]
        # ==============================================================

        for ann in annotations:

            bbox = ann.get(
                "bbox"
            )

            if bbox is None:

                continue

            if len(bbox) != 4:

                continue

            try:

                x = float(
                    bbox[0]
                )

                y = float(
                    bbox[1]
                )

                width = float(
                    bbox[2]
                )

                height = float(
                    bbox[3]
                )

            except (
                ValueError,
                TypeError,
            ):

                continue

            if width <= 0:

                continue

            if height <= 0:

                continue

            x1 = x

            y1 = y

            x2 = x + width

            y2 = y + height

            if x2 <= x1:

                continue

            if y2 <= y1:

                continue

            category_id = ann.get(
                "category_id"
            )

            if category_id is None:

                continue

            try:

                category_id = int(
                    category_id
                )

            except (
                ValueError,
                TypeError,
            ):

                continue

            class_id = self.category_to_class.get(
                category_id
            )

            if class_id is None:

                continue

            boxes.append([

                x1,
                y1,
                x2,
                y2,

            ])

            labels.append(
                class_id
            )

        # ==============================================================
        # TENSORS
        # ==============================================================

        if boxes:

            boxes = torch.tensor(
                boxes,
                dtype=torch.float32,
            )

            labels = torch.tensor(
                labels,
                dtype=torch.int64,
            )

            area = (

                (
                    boxes[:, 2]
                    -
                    boxes[:, 0]
                )

                *

                (
                    boxes[:, 3]
                    -
                    boxes[:, 1]
                )

            )

        else:

            boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32,
            )

            labels = torch.zeros(
                (0,),
                dtype=torch.int64,
            )

            area = torch.zeros(
                (0,),
                dtype=torch.float32,
            )

        # ==============================================================
        # IMAGE ID
        # ==============================================================

        if image_id is not None:

            numeric_id = int(
                image_id
            )

        else:

            numeric_id = int(
                fallback_index
            )

        # ==============================================================
        # TARGET
        # ==============================================================

        target = {

            "boxes":
                boxes,

            "labels":
                labels,

            "image_id":
                torch.tensor(
                    [numeric_id],
                    dtype=torch.int64,
                ),

            "area":
                area,

            "iscrowd":
                torch.zeros(
                    (len(labels),),
                    dtype=torch.int64,
                ),

        }

        return target

    # ==================================================================
    # DATASET INFORMATION
    # ==================================================================

    def _print_information(self):

        print()
        print("=" * 78)
        print("FLIR DATASET INFORMATION")
        print("=" * 78)

        print(
            f"RGB directory        : {self.rgb_dir}"
        )

        print(
            f"Thermal directory    : {self.thermal_dir}"
        )

        print(
            f"RGB-Thermal pairs    : {len(self.samples)}"
        )

        print(
            f"Training mode        : {self.training}"
        )

        print(
            f"Return targets       : {self.return_targets}"
        )

        print(
            f"Corrupt pairs skipped: "
            f"{self.corrupt_pairs_skipped}"
        )

        print(
            f"Annotation file      : "
            f"{self.annotation_file}"
        )

        if self.return_targets:

            print()

            print(
                "ANNOTATION SUMMARY"
            )

            print("-" * 78)

            print(
                f"COCO images          : "
                f"{len(self.coco_images)}"
            )

            print(
                f"Images with GT       : "
                f"{len(self.images_with_annotations)}"
            )

            print(
                f"Person boxes         : "
                f"{self.class_statistics[1]}"
            )

            print(
                f"Car boxes            : "
                f"{self.class_statistics[2]}"
            )

            print(
                f"Bike boxes           : "
                f"{self.class_statistics[3]}"
            )

            total_boxes = sum(
                self.class_statistics.values()
            )

            print(
                f"Total boxes          : "
                f"{total_boxes}"
            )

        print()

        print(
            "CLASS MAPPING"
        )

        print("-" * 78)

        for class_id in range(4):

            print(
                f"{class_id} -> "
                f"{self.CLASS_NAMES_LOCAL[class_id]}"
            )

        print("=" * 78)

    # ==================================================================
    # LENGTH
    # ==================================================================

    def __len__(self):

        return len(
            self.samples
        )

    # ==================================================================
    # GET ITEM
    # ==================================================================

    def __getitem__(
        self,
        index,
    ):

        if not self.samples:

            raise RuntimeError(
                "FLIR dataset contains no samples."
            )

        max_tries = min(
            20,
            len(self.samples)
        )

        last_error = None

        for attempt in range(
            max_tries
        ):

            idx = (

                index
                +
                attempt

            ) % len(
                self.samples
            )

            sample = self.samples[
                idx
            ]

            rgb_path = sample[
                "rgb"
            ]

            thermal_path = sample[
                "thermal"
            ]

            try:

                # ------------------------------------------------------
                # RGB
                # ------------------------------------------------------

                rgb = Image.open(
                    rgb_path
                ).convert(
                    "RGB"
                )

                # ------------------------------------------------------
                # THERMAL
                # ------------------------------------------------------

                thermal = Image.open(
                    thermal_path
                ).convert(
                    "L"
                )

            except (
                OSError,
                UnidentifiedImageError,
                FileNotFoundError,
            ) as exc:

                last_error = exc

                continue

            # ==========================================================
            # TARGET
            # ==========================================================

            if self.return_targets:

                target = self._target_for_rgb(
                    rgb_path,
                    idx,
                )

            else:

                target = None

            # ==========================================================
            # TRANSFORMS
            # ==========================================================

            if self.transforms is not None:

                try:

                    if target is None:

                        result = self.transforms(
                            rgb,
                            thermal,
                        )

                        if isinstance(
                            result,
                            (tuple, list)
                        ) and len(result) == 2:

                            rgb, thermal = result

                        else:

                            raise ValueError(
                                "Transforms must return "
                                "(rgb, thermal)."
                            )

                    else:

                        result = self.transforms(
                            rgb,
                            thermal,
                            target,
                        )

                        if (
                            isinstance(
                                result,
                                (tuple, list)
                            )
                            and
                            len(result) == 3
                        ):

                            rgb, thermal, target = result

                        else:

                            raise ValueError(
                                "Transforms with targets "
                                "must return "
                                "(rgb, thermal, target)."
                            )

                except TypeError:

                    # --------------------------------------------------
                    # Fallback:
                    # Standard torchvision transform
                    # --------------------------------------------------

                    rgb = self.transforms(
                        rgb
                    )

                    thermal = self.transforms(
                        thermal
                    )

            else:

                # ======================================================
                # DEFAULT TENSOR CONVERSION
                # ======================================================

                rgb = F.to_tensor(
                    rgb
                )

                thermal = F.to_tensor(
                    thermal
                )

            # ==========================================================
            # ENSURE TENSOR
            # ==========================================================

            if not torch.is_tensor(rgb):

                rgb = F.to_tensor(
                    rgb
                )

            if not torch.is_tensor(thermal):

                thermal = F.to_tensor(
                    thermal
                )

            # ==========================================================
            # ENSURE THERMAL = [1,H,W]
            # ==========================================================

            if thermal.ndim == 2:

                thermal = thermal.unsqueeze(
                    0
                )

            # ==========================================================
            # OUTPUT
            # ==========================================================

            output = {

                "rgb":
                    rgb,

                "thermal":
                    thermal,

                "rgb_path":
                    str(rgb_path),

                "thermal_path":
                    str(thermal_path),

                # FLIR = target domain

                "domain":
                    1,

            }

            if target is not None:

                output[
                    "target"
                ] = target

            return output

        raise RuntimeError(

            "Failed to load FLIR sample "
            f"after {max_tries} attempts.\n"
            f"Last error: {last_error}"

        )

    # ==================================================================
    # PRINT FINAL STATISTICS
    # ==================================================================

    def print_statistics(self):

        print()
        print("=" * 78)
        print("FLIR FINAL DATASET STATISTICS")
        print("=" * 78)

        print()

        print(
            f"RGB-Thermal pairs : "
            f"{len(self.samples)}"
        )

        print(
            f"Training mode     : "
            f"{self.training}"
        )

        print(
            f"Return targets    : "
            f"{self.return_targets}"
        )

        print(
            f"Domain            : "
            f"1 (Target)"
        )

        print(
            f"Corrupt skipped   : "
            f"{self.corrupt_pairs_skipped}"
        )

        if self.return_targets:

            total_boxes = sum(
                self.class_statistics.values()
            )

            print()

            print(
                "GROUND TRUTH"
            )

            print("-" * 78)

            print(
                f"COCO images       : "
                f"{len(self.coco_images)}"
            )

            print(
                f"Images with GT    : "
                f"{len(self.images_with_annotations)}"
            )

            print(
                f"Person boxes      : "
                f"{self.class_statistics[1]}"
            )

            print(
                f"Car boxes         : "
                f"{self.class_statistics[2]}"
            )

            print(
                f"Bike boxes        : "
                f"{self.class_statistics[3]}"
            )

            print(
                f"Total boxes       : "
                f"{total_boxes}"
            )

        print()

        print(
            "CLASS MAPPING"
        )

        print("-" * 78)

        for class_id in range(4):

            print(
                f"{class_id} -> "
                f"{self.CLASS_NAMES_LOCAL[class_id]}"
            )

        print()

        print("=" * 78)


# ======================================================================
# CONFIG PATH HELPER
# ======================================================================

def _config_path(
    *names,
):

    """
    Returns the first valid Config attribute.

    Supports different naming conventions used
    in config.py.
    """

    for name in names:

        if not hasattr(
            Config,
            name
        ):

            continue

        value = getattr(
            Config,
            name
        )

        if value is None:

            continue

        value = str(
            value
        ).strip()

        if not value:

            continue

        return value

    raise AttributeError(

        "\n"
        "None of the following Config attributes "
        "were found or they are empty:\n"
        +
        "\n".join(
            f"  - {name}"
            for name in names
        )

    )


# ======================================================================
# FIND CONFIG PATH
# ======================================================================

def _safe_config_path(
    names,
    required=True,
):

    try:

        return _config_path(
            *names
        )

    except AttributeError:

        if required:

            raise

        return None


# ======================================================================
# TEST
# ======================================================================

if __name__ == "__main__":

    print()
    print("=" * 78)
    print("TESTING FLIR TARGET DATASET")
    print("=" * 78)

    print()

    # ==============================================================
    # RGB PATH
    # ==============================================================

    rgb_dir = _safe_config_path(

        (

            "TARGET_VAL_RGB_DIR",

            "TARGET_RGB_VAL_DIR",

            "TARGET_RGB_DIR",

            "FLIR_TARGET_VAL_RGB_DIR",

            "FLIR_VAL_RGB_DIR",

            "FLIR_RGB_VAL_DIR",

            "FLIR_RGB_DIR",

        ),

        required=True,

    )

    # ==============================================================
    # THERMAL PATH
    # ==============================================================

    thermal_dir = _safe_config_path(

        (

            "TARGET_VAL_INFRARED_DIR",

            "TARGET_VAL_IR_DIR",

            "TARGET_INFRARED_VAL_DIR",

            "TARGET_IR_VAL_DIR",

            "TARGET_VAL_THERMAL_DIR",

            "TARGET_THERMAL_VAL_DIR",

            "TARGET_THERMAL_DIR",

            "FLIR_TARGET_VAL_INFRARED_DIR",

            "FLIR_TARGET_VAL_IR_DIR",

            "FLIR_TARGET_VAL_THERMAL_DIR",

            "FLIR_INFRARED_VAL_DIR",

            "FLIR_IR_VAL_DIR",

            "FLIR_THERMAL_VAL_DIR",

            "FLIR_THERMAL_DIR",

        ),

        required=True,

    )

    # ==============================================================
    # ANNOTATION PATH
    # ==============================================================

    annotation_file = _safe_config_path(

        (

            "TARGET_VAL_ANNOTATION",

            "TARGET_VAL_ANNOTATION_FILE",

            "TARGET_ANNOTATION",

            "TARGET_ANNOTATION_FILE",

            "FLIR_TARGET_VAL_ANNOTATION",

            "FLIR_TARGET_VAL_ANNOTATION_FILE",

            "FLIR_VAL_ANNOTATION",

            "FLIR_VAL_ANNOTATION_FILE",

            "FLIR_ANNOTATION",

            "FLIR_ANNOTATION_FILE",

        ),

        required=True,

    )

    # ==============================================================
    # PRINT RESOLVED PATHS
    # ==============================================================

    print(
        "Resolved FLIR paths:"
    )

    print()

    print(
        f"RGB        : {rgb_dir}"
    )

    print(
        f"Thermal    : {thermal_dir}"
    )

    print(
        f"Annotation : {annotation_file}"
    )

    # ==============================================================
    # VALIDATION DATASET
    # ==============================================================

    dataset = FLIRTargetDataset(

        rgb_dir=rgb_dir,

        thermal_dir=thermal_dir,

        transforms=None,

        training=False,

        annotation_file=annotation_file,

        return_targets=True,

        filter_corrupt=True,

    )

    # ==============================================================
    # STATISTICS
    # ==============================================================

    dataset.print_statistics()

    # ==============================================================
    # SAMPLE TEST
    # ==============================================================

    if len(dataset) > 0:

        sample = dataset[0]

        print()
        print("=" * 78)
        print("FIRST FLIR SAMPLE")
        print("=" * 78)

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
            "Domain          :",
            sample["domain"]
        )

        print(
            "RGB path        :",
            sample["rgb_path"]
        )

        print(
            "Thermal path    :",
            sample["thermal_path"]
        )

        # ==========================================================
        # TARGET
        # ==========================================================

        if "target" in sample:

            target = sample[
                "target"
            ]

            print()

            print(
                "Boxes           :",
                tuple(
                    target["boxes"].shape
                )
            )

            print(
                "Labels          :",
                target[
                    "labels"
                ].tolist()
            )

            print(
                "Image ID        :",
                target[
                    "image_id"
                ].tolist()
            )

            print(
                "Areas           :",
                target[
                    "area"
                ].tolist()
            )

            # ======================================================
            # CLASS COUNTS
            # ======================================================

            print()

            print(
                "FIRST IMAGE CLASS COUNTS"
            )

            print("-" * 78)

            for class_id in (

                1,
                2,
                3,

            ):

                count = int(

                    (
                        target[
                            "labels"
                        ]
                        ==
                        class_id
                    ).sum()

                )

                class_name = (
                    FLIRTargetDataset
                    .CLASS_NAMES_LOCAL[
                        class_id
                    ]
                )

                print(
                    f"{class_name:<12}: "
                    f"{count}"
                )

    # ==============================================================
    # FINAL
    # ==============================================================

    print()
    print("=" * 78)
    print("FLIR DATASET TEST COMPLETED")
    print("=" * 78)