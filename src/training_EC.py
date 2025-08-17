#!/usr/bin/env python
"""
Train YOLOv8-s on the CAD-component dataset.

Folder layout (relative to *this* script):
├─ src/
│  └─ train_cad_yolov8s.py   ← you are here
├─ images/
│  └─ annotated_data/
│     ├─ data.yaml
│     ├─ train/…
│     ├─ val/…   (optional)
│     └─ test/…  (optional)

Run:
$ conda activate yolo_env
$ python src/train_cad_yolov8s.py --img 640 --epochs 100 --batch 8
"""

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image
import yaml
from ultralytics import YOLO

# ------------------------------------------------------------------ #
#  Paths                                                             #
# ------------------------------------------------------------------ #
SRC_DIR   = Path(__file__).resolve().parent          # …/src
ROOT_DIR  = SRC_DIR.parent                           # project root
DATA_DIR  = ROOT_DIR / "images" / "annotated_data"   # dataset root
YAML_PATH = DATA_DIR / "data.yaml"                   # data.yaml


# ------------------------------------------------------------------ #
#  Helpers                                                           #
# ------------------------------------------------------------------ #
def load_yaml(path: Path):
    with path.open() as f:
        return yaml.safe_load(f)


def save_yaml(obj, path: Path):
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def resize_folder(folder: Path, size: int):
    """Force every image in *folder* to size×size."""
    for p in folder.rglob("*.[jp][pn]g"):
        with Image.open(p) as im:
            if im.size != (size, size):
                im.convert("RGB").resize((size, size), Image.LANCZOS).save(p, quality=95)


def ensure_splits(img_size: int):
    """
    1. Resize train/val/test images to img_size.
    2. If val/ or test/ missing → sample 15 % each from train/.
    3. Write absolute paths back into data.yaml.
    """
    train_img = DATA_DIR / "train" / "images"
    resize_folder(train_img, img_size)

    for split in ("val", "test"):
        img_dir = DATA_DIR / split / "images"
        lbl_dir = DATA_DIR / split / "labels"

        if img_dir.exists():                # split already present
            resize_folder(img_dir, img_size)
            continue

        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        imgs = list(train_img.glob("*.[jp][pn]g"))
        k = max(1, int(0.15 * len(imgs)))   # 15 %
        for img in random.sample(imgs, k):
            lbl = (img.parent.parent / "labels" / img.with_suffix(".txt").name)
            shutil.move(img, img_dir / img.name)
            if lbl.exists():
                shutil.move(lbl, lbl_dir / lbl.name)

    # update YAML with absolute paths
    cfg = load_yaml(YAML_PATH)
    for split in ("train", "val", "test"):
        cfg[split] = str((DATA_DIR / split / "images").resolve())
    save_yaml(cfg, YAML_PATH)


# ------------------------------------------------------------------ #
#  Main                                                              #
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img",    type=int, default=640, help="square image size")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch",  type=int, default=8)
    ap.add_argument("--device", default=0, help="GPU id or 'cpu'")
    args = ap.parse_args()

    print(f"[•] Preparing dataset at {DATA_DIR} …")
    ensure_splits(args.img)

    print("[•] Training YOLOv8-s …")
    model = YOLO("yolov8s.pt")
    model.train(
        data=str(YAML_PATH),
        imgsz=args.img,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(ROOT_DIR / "runs"),
        name="cad_yolov8s",
        exist_ok=True,
        workers=4,
    )

    print("[•] Final evaluation on test split …")
    model.val(data=str(YAML_PATH), split="test", device=args.device)


if __name__ == "__main__":
    main()
