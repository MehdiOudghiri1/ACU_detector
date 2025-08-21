#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

from pdf2image import convert_from_path
import numpy as np
import cv2
import tkinter as tk  # stdlib: only to get screen size

WINDOW_TITLE = "PDF Preview → type folder number, Enter to copy (q=quit)"


def get_screen_size():
    root = tk.Tk()
    root.withdraw()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return w, h


def center_window(win_title: str, win_w: int, win_h: int):
    screen_w, screen_h = get_screen_size()
    x = max((screen_w - win_w) // 2, 0)
    y = max((screen_h - win_h) // 2, 0)
    cv2.resizeWindow(win_title, win_w, win_h)
    cv2.moveWindow(win_title, x, y)


def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV BGR array (no temp files)."""
    rgb = np.array(pil_image)
    if rgb.ndim == 2:  # grayscale
        return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def fit_image_to_target(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_overlay(img, text):
    """Semi-transparent bottom bar with instructions / typed digits."""
    overlay = img.copy()
    h, w = img.shape[:2]
    bar_h = max(40, h // 14)
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    alpha = 0.55
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    margin = 12
    cv2.putText(
        img, text.strip(), (margin, h - bar_h // 3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    return img


def parse_args():
    p = argparse.ArgumentParser(
        description="Stable, centered PDF sorter: copies PDFs into outputs/<folder> (live on Enter)."
    )
    p.add_argument("-f", type=str, required=True, help="Folder containing PDFs")
    p.add_argument("-n", type=int, required=True, help="Number of PDFs to process")
    p.add_argument(
        "-s", "--start", type=int, default=1,
        help="1-based index in the sorted PDF list to start from (default: 1)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    pdf_folder = Path(args.f).expanduser().resolve()
    if not pdf_folder.exists():
        print(f"Folder not found: {pdf_folder}")
        return

    all_pdfs = sorted(pdf_folder.glob("*.pdf"))
    if not all_pdfs:
        print("No PDFs found.")
        return

    # Range slice using 1-based start
    start_idx_1based = max(1, args.start)
    start_idx = start_idx_1based - 1  # 0-based
    end_idx = min(len(all_pdfs), start_idx + max(0, args.n))
    if start_idx >= len(all_pdfs):
        print(f"Start index {start_idx_1based} is beyond the number of PDFs ({len(all_pdfs)}).")
        return
    pdf_files = all_pdfs[start_idx:end_idx]

    # OUTPUT ROOT: <input>/outputs
    output_root = (pdf_folder / "outputs").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Pre-create 1..5 as you requested
    for k in range(1, 6):
        (output_root / str(k)).mkdir(parents=True, exist_ok=True)

    # One persistent window, centered, ~60% of screen area
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    screen_w, screen_h = get_screen_size()
    area_scale = 0.6 ** 0.5  # scale both dims → ~60% area
    target_win_w = max(400, int(screen_w * area_scale))
    target_win_h = max(300, int(screen_h * area_scale))
    center_window(WINDOW_TITLE, target_win_w, target_win_h)
    try:
        cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass  # not supported everywhere

    for i, pdf_path in enumerate(pdf_files, 1):
        # Convert only the first page for preview
        images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=1)
        img_bgr = pil_to_cv2(images[0])
        preview = fit_image_to_target(img_bgr, target_win_w, target_win_h)

        typed = ""
        while True:
            # Render the stable view with instructions
            display = preview.copy()
            instr = (
                f"[{start_idx_1based + i - 1}/{len(all_pdfs)}] {pdf_path.name}  |  "
                f"type folder number → Enter  |  q=quit"
            )
            if typed:
                instr += f"  |  Chosen: outputs/{typed}"
            display = draw_overlay(display, instr)
            cv2.imshow(WINDOW_TITLE, display)

            key = cv2.waitKey(10) & 0xFF  # responsive polling

            if key == 255:
                continue

            if key in (ord('q'), ord('Q')):
                cv2.destroyWindow(WINDOW_TITLE)
                print("Quit.")
                return

            # Enter = live copy & advance
            if key in (10, 13):
                folder = typed.strip()
                if not folder:
                    continue
                target_dir = (output_root / folder).resolve()
                target_dir.mkdir(parents=True, exist_ok=True)  # create if missing
                target_path = target_dir / pdf_path.name
                shutil.copy2(pdf_path, target_path)
                # Quick on-screen confirmation
                confirm = preview.copy()
                confirm = draw_overlay(confirm, f"Copied → {target_path}")
                cv2.imshow(WINDOW_TITLE, confirm)
                cv2.waitKey(250)  # brief ack
                break

            # Backspace / delete
            if key in (8, 127):
                typed = typed[:-1]
                continue

            # Accept digits only (folders are numbers)
            ch = chr(key)
            if ch.isdigit():
                typed += ch
            # ignore everything else

    cv2.destroyWindow(WINDOW_TITLE)
    print(f"Done. Outputs in: {output_root}")


if __name__ == "__main__":
    main()
