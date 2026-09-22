"""
===============================================================
FLIR Target Dataset for SICDA

Target domain (unlabeled for detection training):
    - RGB image
    - Thermal / LWIR image (paired)

FLIR annotation file is optional (validation only).

Output:
    {
        "rgb"          : Tensor [3, H, W],
        "thermal"      : Tensor [1, H, W],
        "rgb_path"     : str,
        "thermal_path" : str
    }
===============================================================
"""

import os
import re
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import torchvision.transforms.functional as F

from config import Config


class FLIRTargetDataset(Dataset):

    IMAGE_EXTS = {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".tif", ".tiff",
        ".JPG", ".JPEG", ".PNG", ".BMP",
        ".TIF", ".TIFF",
    }

    def __init__(
            self,
            rgb_dir,
            thermal_dir,
            transforms=None,
            training=True,
            annotation_file=None,
            filter_corrupt=True
    ):
        super().__init__()

        self.rgb_dir = Path(rgb_dir)
        self.thermal_dir = Path(thermal_dir)
        self.transforms = transforms
        self.training = training
        self.annotation_file = annotation_file
        self.samples = []

        print("=" * 70)
        print("Loading FLIR Target Dataset (RGB + Thermal)...")
        print("=" * 70)
        print(f"Config RGB dir     : {self.rgb_dir}")
        print(f"Config Thermal dir : {self.thermal_dir}")

        if not self.rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found:\n{self.rgb_dir}")
        if not self.thermal_dir.exists():
            raise FileNotFoundError(f"Thermal directory not found:\n{self.thermal_dir}")

        # Agar images 'data/' ke andar hain to auto enter
        self.rgb_dir = self._resolve_image_root(self.rgb_dir)
        self.thermal_dir = self._resolve_image_root(self.thermal_dir)

        print(f"Using RGB dir      : {self.rgb_dir}")
        print(f"Using Thermal dir  : {self.thermal_dir}")

        self._build_pairs()

        if filter_corrupt and len(self.samples) > 0:
            self._filter_readable_pairs()

        print(f"Total Target Pairs : {len(self.samples)}")
        print(f"Training Mode      : {training}")
        print("=" * 70)

        if len(self.samples) == 0:
            self._print_debug()
            raise RuntimeError(
                "No RGB–Thermal pairs found.\n"
                "Thermal folder empty ho sakti hai, ya file names match nahi kar rahe.\n"
                "Upar DEBUG section dekho."
            )

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def _resolve_image_root(self, folder: Path) -> Path:
        """
        Prefer folder that actually contains images.
        Tries: folder itself, folder/data, first subdir with images.
        """
        if self._list_images(folder, recursive=False):
            return folder

        data = folder / "data"
        if data.is_dir() and self._list_images(data, recursive=False):
            return data

        if folder.is_dir():
            for sub in sorted(folder.iterdir()):
                if sub.is_dir() and self._list_images(sub, recursive=True):
                    return sub

        return folder

    def _list_images(self, folder: Path, recursive=True):
        if not folder.exists():
            return []
        files = []
        iterator = folder.rglob("*") if recursive else folder.glob("*")
        for p in iterator:
            if p.is_file() and p.suffix in self.IMAGE_EXTS:
                files.append(p)
        return files

    def _frame_key(self, path: Path) -> str:
        """
        Extract stable key from names like:
          video-xxx-frame-000510-yyy.jpg -> 000510
          FLIR_00001.jpg                 -> 00001
          12345.jpg                      -> 12345
        """
        stem = path.stem

        m = re.search(r"frame[-_]?(\d+)", stem, flags=re.IGNORECASE)
        if m:
            return m.group(1)

        m = re.search(r"(\d+)$", stem)
        if m:
            return m.group(1)

        return stem.lower()

    def _is_readable(self, path: str) -> bool:
        try:
            if not os.path.exists(path):
                return False
            if os.path.getsize(path) < 100:
                return False
            with Image.open(path) as im:
                im.verify()
            return True
        except Exception:
            return False

    def _filter_readable_pairs(self):
        """Remove pairs where RGB or thermal cannot be opened by PIL."""
        good = []
        bad = 0

        for s in self.samples:
            if self._is_readable(s["rgb"]) and self._is_readable(s["thermal"]):
                good.append(s)
            else:
                bad += 1

        self.samples = good
        print(f"Readable pairs     : {len(self.samples)}")
        print(f"Skipped corrupt    : {bad}")

    # ----------------------------------------------------------
    # Pair building
    # ----------------------------------------------------------
    def _build_pairs(self):
        rgb_files = self._list_images(self.rgb_dir, recursive=True)
        thermal_files = self._list_images(self.thermal_dir, recursive=True)

        print(f"RGB files found     : {len(rgb_files)}")
        print(f"Thermal files found : {len(thermal_files)}")

        if rgb_files:
            print(f"  RGB sample        : {rgb_files[0].name}")
        if thermal_files:
            print(f"  Thermal sample    : {thermal_files[0].name}")
        else:
            print("  Thermal sample    : <NONE>")
            if self.thermal_dir.exists():
                children = list(self.thermal_dir.iterdir())[:10]
                print(
                    f"  Thermal dir entries "
                    f"({len(list(self.thermal_dir.iterdir()))} total), first:"
                )
                for c in children:
                    kind = "DIR" if c.is_dir() else "FILE"
                    print(f"    [{kind}] {c.name}")

        if not rgb_files:
            print("ERROR: No RGB images found.")
            return
        if not thermal_files:
            print("ERROR: No Thermal images found in thermal dir.")
            return

        # ---- Strategy 1: exact stem ----
        th_stem = {p.stem: p for p in thermal_files}
        for rgb in rgb_files:
            if rgb.stem in th_stem:
                self.samples.append({
                    "rgb": str(rgb),
                    "thermal": str(th_stem[rgb.stem])
                })
        if self.samples:
            print("Pairing strategy   : exact stem")
            return

        # ---- Strategy 2: lower-case stem ----
        th_stem_l = {p.stem.lower(): p for p in thermal_files}
        for rgb in rgb_files:
            key = rgb.stem.lower()
            if key in th_stem_l:
                self.samples.append({
                    "rgb": str(rgb),
                    "thermal": str(th_stem_l[key])
                })
        if self.samples:
            print("Pairing strategy   : case-insensitive stem")
            return

        # ---- Strategy 3: frame number / trailing digits ----
        th_frame = {}
        for p in thermal_files:
            th_frame[self._frame_key(p)] = p

        for rgb in rgb_files:
            key = self._frame_key(rgb)
            if key in th_frame:
                self.samples.append({
                    "rgb": str(rgb),
                    "thermal": str(th_frame[key])
                })
        if self.samples:
            print("Pairing strategy   : frame/number key")
            return

        print("Pairing strategy   : FAILED")

    def _print_debug(self):
        print("\n" + "!" * 70)
        print("DEBUG: Could not pair RGB and Thermal")
        print("!" * 70)

        rgb_files = self._list_images(self.rgb_dir, recursive=True)
        th_files = self._list_images(self.thermal_dir, recursive=True)

        print(f"RGB count     : {len(rgb_files)}")
        print(f"Thermal count : {len(th_files)}")

        if th_files:
            ext = Counter(p.suffix.lower() for p in th_files)
            print("Thermal extensions:", dict(ext))

        print("First RGB files:")
        for p in rgb_files[:5]:
            print(f"  {p.name}  key={self._frame_key(p)}")

        print("First Thermal files:")
        for p in th_files[:5]:
            print(f"  {p.name}  key={self._frame_key(p)}")

        if not th_files and self.thermal_dir.exists():
            print("Thermal directory listing (top level):")
            for c in list(self.thermal_dir.iterdir())[:20]:
                print(f"  {'[DIR]' if c.is_dir() else '[FILE]'} {c.name}")

        print("!" * 70 + "\n")

    # ----------------------------------------------------------
    # Dataset API
    # ----------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Load sample. Agar file corrupt ho to next readable pair try karo.
        """
        max_tries = min(20, len(self.samples))
        last_error = None

        for attempt in range(max_tries):
            idx = (index + attempt) % len(self.samples)
            sample = self.samples[idx]
            rgb_path = sample["rgb"]
            thermal_path = sample["thermal"]

            try:
                if not os.path.exists(rgb_path):
                    raise FileNotFoundError(f"RGB not found: {rgb_path}")
                if not os.path.exists(thermal_path):
                    raise FileNotFoundError(f"Thermal not found: {thermal_path}")

                rgb_image = Image.open(rgb_path).convert("RGB")
                thermal_image = Image.open(thermal_path).convert("L")

            except (UnidentifiedImageError, OSError, FileNotFoundError) as e:
                last_error = e
                continue

            if self.transforms is not None:
                rgb_image, thermal_image = self.transforms(
                    rgb_image, thermal_image
                )
            else:
                rgb_image = F.to_tensor(rgb_image)
                thermal_image = F.to_tensor(thermal_image)

            return {
                "rgb": rgb_image,
                "thermal": thermal_image,
                "rgb_path": rgb_path,
                "thermal_path": thermal_path
            }

        raise RuntimeError(
            f"Failed to load FLIR sample after {max_tries} tries. "
            f"Last error: {last_error}"
        )

    def get_rgb_path(self, index):
        return self.samples[index]["rgb"]

    def get_thermal_path(self, index):
        return self.samples[index]["thermal"]

    def print_statistics(self):
        print("\n" + "=" * 60)
        print("FLIR Target Dataset Statistics")
        print("=" * 60)
        print(f"Total Pairs     : {len(self.samples)}")
        print(f"Training Mode   : {self.training}")
        if self.samples:
            print(f"Example RGB     : {self.samples[0]['rgb']}")
            print(f"Example Thermal : {self.samples[0]['thermal']}")
        print("=" * 60)

    def verify_dataset(self):
        missing = 0
        for s in self.samples:
            if not os.path.exists(s["rgb"]):
                missing += 1
            if not os.path.exists(s["thermal"]):
                missing += 1
        if missing == 0:
            print("FLIR Target Dataset Verification Passed.")
        else:
            print(f"Missing files : {missing}")

    def __repr__(self):
        return (
            f"FLIRTargetDataset(samples={len(self.samples)}, "
            f"training={self.training})"
        )


if __name__ == "__main__":
    dataset = FLIRTargetDataset(
        rgb_dir=Config.TARGET_RGB_DIR,
        thermal_dir=Config.TARGET_IR_DIR,
        transforms=None,
        training=True,
        annotation_file=getattr(Config, "TARGET_ANNOTATION", None),
        filter_corrupt=True
    )
    print(dataset)
    dataset.print_statistics()
    if len(dataset) > 0:
        sample = dataset[0]
        print("RGB shape    :", sample["rgb"].shape)
        print("Thermal shape:", sample["thermal"].shape)