#!/usr/bin/env python
"""
visualize.py
------------
Shows random images from images/images_pdf_png/, runs the most-recent
YOLOv8 *.pt weights it can find under runs/, draws colour-coded boxes,
then waits for you to press ENTER for the next image (q or ESC to quit).

Project tree assumed:

computer_vision_project/
├─ images/images_pdf_png/*.png|jpg
├─ runs/**/weights/*.pt        (your trained weights)
└─ src/visualize.py            (this file)
"""

import cv2, random, glob, time
from pathlib import Path
from ultralytics import YOLO

# ------------------------------------------------------------------ #
#  Class colours (BGR)                                               #
# ------------------------------------------------------------------ #
COLORS = {
    "Control pannel"   : (255, 230, 160),  # light-blue
    "EC fan array"     : (204,   0, 204),  # purple
    "Vertical array"   : (  0,   0, 255),  # red
    "backdraft dumper" : (255, 102,   0),  # blue-ish
    "wirless pannel"   : (  0, 128, 255),  # orange
}
CLASS_NAMES = list(COLORS.keys())

# ------------------------------------------------------------------ #
#  Auto-locate things                                                #
# ------------------------------------------------------------------ #
ROOT = Path(__file__).resolve().parents[1]

IMG_DIR = ROOT / "images" / "images_pdf_png"
if not IMG_DIR.is_dir():
    raise FileNotFoundError(f"No image folder at {IMG_DIR}")

# newest *.pt under runs/**/weights/
pt_files = sorted(
    glob.glob(str(ROOT / "runs" / "**" / "weights" / "*.pt"), recursive=True),
    key=lambda p: Path(p).stat().st_mtime,
    reverse=True,
)
if not pt_files:
    raise FileNotFoundError("No .pt weights found under runs/**/weights/")
WEIGHTS = pt_files[0]
print(f"[INFO] Using weights: {WEIGHTS}")

# ------------------------------------------------------------------ #
#  Helper: draw boxes                                                #
# ------------------------------------------------------------------ #
def annotate(img, results):
    for xyxy, cls, conf in zip(
        results.boxes.xyxy.cpu().numpy(),
        results.boxes.cls.cpu().numpy().astype(int),
        results.boxes.conf.cpu().numpy(),
    ):
        x1, y1, x2, y2 = map(int, xyxy)
        name  = CLASS_NAMES[cls]
        color = COLORS[name]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return img

# ------------------------------------------------------------------ #
#  Main loop                                                         #
# ------------------------------------------------------------------ #
def main():
    model = YOLO(str(WEIGHTS))
    images = [p for p in IMG_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not images:
        raise RuntimeError(f"No PNG/JPG images inside {IMG_DIR}")

    cv2.namedWindow("YOLOv8 predictions", cv2.WINDOW_NORMAL)

    while True:
        img_path = random.choice(images)
        frame    = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Cannot read {img_path}")
            continue

        result   = model.predict(frame, imgsz=640, conf=0.5, verbose=False)[0]
        annotated = annotate(frame.copy(), result)

        cv2.imshow("YOLOv8 predictions", annotated)
        key = cv2.waitKey(0) & 0xFF    # wait indefinitely for key

        if key in (ord('q'), 27):      # q or ESC to quit
            break                      # exit loop → close window
        # any other key → next image (ENTER, space, etc.)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
