"""Quick dataset/config path check before training."""
from pathlib import Path
from config import Config


def show(name, path):
    p = Path(path)
    print(f"{name:26s}: {p}")
    print(f"{'':26s}  exists={p.exists()}")
    return p.exists()


checks = [
    ("LLVIP visible train", Config.SOURCE_VISIBLE_TRAIN_DIR),
    ("LLVIP infrared train", Config.SOURCE_IR_TRAIN_DIR),
    ("LLVIP visible test", Config.SOURCE_VISIBLE_VAL_DIR),
    ("LLVIP infrared test", Config.SOURCE_IR_VAL_DIR),
    ("LLVIP annotations", Config.SOURCE_ANNOTATION_DIR),
    ("FLIR RGB train", Config.TARGET_RGB_DIR),
    ("FLIR thermal train", Config.TARGET_IR_DIR),
]

ok = True
for name, path in checks:
    ok &= show(name, path)

print("\nOptional target validation:")
show("FLIR RGB val", Config.TARGET_VAL_RGB_DIR)
show("FLIR thermal val", Config.TARGET_VAL_IR_DIR)
show("FLIR val COCO JSON", Config.TARGET_VAL_ANNOTATION)

print("\nRESULT:", "READY" if ok else "EDIT DATASET PATHS IN config.py")
