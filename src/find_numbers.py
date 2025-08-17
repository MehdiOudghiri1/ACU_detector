#!/usr/bin/env python3
"""
find_numbers.py  –  three interactive modes

▶ classify   : white-band separation, numbers above = red, below = blue
▶ bottom     : highlights only the lowest dimension numbers (yellow)
▶ sections   : same as bottom + thin vertical section-split lines (green)

Navigation: zoomable TkAgg window, press Enter → next random page,
q / quit / ESC → exit.

Usage
-----
python find_numbers.py [classify|bottom|sections|dimensions] /path/to/pdfs
"""
import os, random, sys, argparse
xdg = os.environ.get("XDG_RUNTIME_DIR", "")
if not xdg or not os.access(xdg, os.W_OK):
    os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import pdfplumber

DPI = 150
SCALE = DPI / 72.0

def is_number(txt: str) -> bool:
    txt = txt.replace(",", "").strip()
    try:
        float(txt)
        return True
    except ValueError:
        return False

def number_bboxes(page):
    return [
        (w["x0"], w["top"], w["x1"], w["bottom"])
        for w in page.extract_words()
        if is_number(w["text"])
    ]

def find_first_white_band(pil: Image.Image,
                          lower_frac=0.2, upper_frac=0.8,
                          min_height=33):
    w, h = pil.size
    y_min, y_max = int(h * lower_frac), int(h * upper_frac)
    g = np.array(pil.convert("L"))
    blank = np.all(g == 255, axis=1)

    for y in range(y_max - 1, y_min - 1, -1):
        if blank[y]:
            b = y
            t = y
            while t - 1 >= y_min and blank[t - 1]:
                t -= 1
            if b - t + 1 >= min_height:
                return t, b + 1
    return None

def page_fig_classification(page):
    boxes = number_bboxes(page)
    img   = page.to_image(resolution=DPI)
    pil   = img.original
    band  = find_first_white_band(pil)

    top_boxes, bot_boxes = [], []
    if band:
        y_top, y_bot = band
        for x0, t, x1, b in boxes:
            cy_px = ((t + b) / 2) * SCALE
            if cy_px < y_top:
                top_boxes.append((x0, t, x1, b))
            elif cy_px > y_bot:
                bot_boxes.append((x0, t, x1, b))
    else:
        top_boxes = boxes

    for bx in top_boxes:
        img.draw_rect(bx, stroke="red",  stroke_width=2)
    for bx in bot_boxes:
        img.draw_rect(bx, stroke="blue", stroke_width=2)

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img.annotated)
    if band:
        ax.add_patch(Rectangle((0, y_top), pil.width, y_bot - y_top,
                               linewidth=3, edgecolor="lime", facecolor="none"))
    ax.axis("off")
    return fig

def find_bottom_numbers(boxes, page_height_pts, scale, tolerance_px=4):
    if not boxes:
        return []

    centers_px = [
        (page_height_pts - (t + b) / 2) * scale for _, t, _, b in boxes
    ]
    top_y = min(centers_px)
    return [bx for bx, cy in zip(boxes, centers_px) if cy - top_y <= tolerance_px]

def page_fig_bottom(page, tol_px=4):
    boxes  = number_bboxes(page)
    img    = page.to_image(resolution=DPI)
    bottom = find_bottom_numbers(boxes, page.height, tol_px)

    for bx in boxes:
        img.draw_rect(bx, stroke=(170, 170, 170), stroke_width=1)
    for bx in bottom:
        img.draw_rect(bx, stroke="yellow", stroke_width=2)

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img.annotated)
    ax.axis("off")
    return fig, bottom, img

def max_run_length(bool_arr):
    best = cur = 0
    for v in bool_arr:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best

def find_section_lines(pil: Image.Image,
                       bottom_boxes,
                       page_height_pts,
                       dark_thr=40,
                       max_line_length=150,
                       search_every=1,
                       pad_px=4):
    if not bottom_boxes:
        return [], None

    centres = [((t + b) / 2) * SCALE for _, t, _, b in bottom_boxes]
    avg_y = float(np.mean(centres))
    g = np.array(pil.convert("L"))
    H, W = g.shape
    all_lines = []

    for col in range(0, W, search_every):
        start = None
        for row in range(H):
            if g[row, col] < dark_thr:
                if start is None:
                    start = row
            else:
                if start is not None:
                    length = row - start
                    if length < max_line_length and start < avg_y < row:
                        all_lines.append((col, start, row))
                    start = None
        if start is not None:
            length = H - start
            if length < max_line_length and start < avg_y < H:
                all_lines.append((col, start, H))

    filtered = []
    for col, y0, y1 in all_lines:
        too_close = False
        for x0, _, x1, _ in bottom_boxes:
            x0_px = x0 * SCALE
            x1_px = x1 * SCALE
            if x0_px - pad_px <= col <= x1_px + pad_px:
                too_close = True
                break
        if not too_close:
            filtered.append((col, y0, y1))

    return filtered, avg_y

def page_fig_sections(page):
    fig, bottom, img = page_fig_bottom(page)
    pil = img.original
    splits, avg_y = find_section_lines(
        pil, bottom, page.height,
        dark_thr=40,
        max_line_length=150,
        search_every=1,
        pad_px=4
    )

    ax = fig.axes[0]
    if avg_y is not None:
        ax.axhline(avg_y, color="blue", linewidth=2)
    for x, y0, y1 in splits:
        ax.add_patch(Rectangle((x, y0), 1, y1 - y0,
                               linewidth=2, edgecolor="green",
                               facecolor="none"))
    return fig

def _interactive_loop(folder, renderer):
    pdfs = [p for p in os.listdir(folder) if p.lower().endswith(".pdf")]
    if not pdfs:
        sys.exit(1)
    while True:
        pdf_path = os.path.join(folder, random.choice(pdfs))
        with pdfplumber.open(pdf_path) as pdf:
            page = random.choice(pdf.pages)
            fig  = renderer(page)
            fig.suptitle(f"{os.path.basename(pdf_path)} – page {page.page_number}")
            plt.tight_layout(); plt.show()
        key = input()
        if key.lower() in {"q", "quit", "exit"}:
            plt.close("all"); break
        plt.close("all")

def find_dimensions(page, limit_px):
    tokens = [
        (w["x0"], w["top"], w["x1"], w["bottom"], w["text"])
        for w in page.extract_words()
        if is_number(w["text"])
    ]
    pil = page.to_image(resolution=DPI).original
    band = find_first_white_band(pil)
    if not band:
        return {}
    y_top, y_bot = band
    limit_pt = limit_px / SCALE

    top_toks, bot_toks = [], []
    for x0, t, x1, b, txt in tokens:
        cy = ((t + b) / 2) * SCALE
        if cy < y_top:
            top_toks.append((x0, t, x1, b, txt))
        elif cy > y_bot:
            bot_toks.append((x0, t, x1, b, txt))

    dims = {}
    left_mode = any(x0 < limit_pt for x0, *_ in top_toks)
    if top_toks:
        if left_mode:
            base_tok = min(top_toks, key=lambda tk: tk[0])
            rest     = [tk for tk in top_toks if tk[0] > base_tok[0]]
        else:
            base_tok = max(top_toks, key=lambda tk: tk[0])
            rest     = [tk for tk in top_toks if tk[0] < base_tok[0]]
        dims["base"] = base_tok

        if rest:
            rest_sorted = sorted(rest, key=lambda tk: tk[0], reverse=not left_mode)[:6]
            diffs = []
            base_x0 = base_tok[0]
            for tk in rest_sorted:
                gap = (tk[0] - base_x0) if left_mode else (base_x0 - tk[0])
                if gap > 0:
                    diffs.append((gap, tk))
            if diffs:
                min_gap = min(diffs, key=lambda x: x[0])[0]
                tol_pdf = 3.0 / SCALE
                group = [tk for gap, tk in diffs if gap <= min_gap + tol_pdf]
                if not group:
                    _, best_tok = min(diffs, key=lambda x: x[0])
                else:
                    bx_cx = ((base_tok[0] + base_tok[2]) / 2) * SCALE
                    bx_cy = ((base_tok[1] + base_tok[3]) / 2) * SCALE
                    def dist2(tok):
                        cx = ((tok[0] + tok[2]) / 2) * SCALE
                        cy = ((tok[1] + tok[3]) / 2) * SCALE
                        return (cx - bx_cx)**2 + (cy - bx_cy)**2
                    best_tok = min(group, key=dist2)
                dims["cabinet_width"] = best_tok

    bottom_left_mode = any(x0 < limit_pt for x0, *_ in bot_toks)
    if bot_toks:
        if bottom_left_mode:
            bot_sorted = sorted(bot_toks, key=lambda tk: tk[0])
        else:
            bot_sorted = sorted(bot_toks, key=lambda tk: tk[0], reverse=True)
        height_tok = bot_sorted[0]
        rest_bot   = bot_sorted[1:]
        dims["cabinet_height"] = height_tok
        hx0, ht, hx1, hb, _ = height_tok
        h_cx = ((hx0 + hx1) / 2) * SCALE
        for tk in rest_bot:
            cx = ((tk[0] + tk[2]) / 2) * SCALE
            if abs(cx - h_cx) <= 3:
                dims["base_height"] = tk
                break

    return dims

def page_fig_dimensions(page):
    fig, bottom, img = page_fig_bottom(page)
    page_dims = find_dimensions(page, 762.0)
    ax = fig.axes[0]
    ORANGE      = (1.0, 0.5, 0.0, 1.0)
    DARK_ORANGE = (0.8, 0.3, 0.0, 1.0)
    TEAL        = (0.0, 0.7, 0.7, 1.0)
    DARK_TEAL   = (0.0, 0.4, 0.4, 1.0)

    for name, color in [
        ("base", ORANGE),
        ("cabinet_width", DARK_ORANGE),
        ("cabinet_height", TEAL),
        ("base_height", DARK_TEAL),
    ]:
        if name in page_dims:
            x0, t, x1, b, text = page_dims[name]
            rect = Rectangle(
                (x0 * SCALE, (t) * SCALE),
                (x1 - x0) * SCALE,
                (b - t) * SCALE,
                linewidth=3, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x0 * SCALE, (t) * SCALE - 4,
                name, color=color, fontsize=10, weight="bold"
            )
    return fig

def main_classify(folder):  _interactive_loop(folder, page_fig_classification)
def main_bottom(folder):    _interactive_loop(folder, lambda p: page_fig_bottom(p)[0])
def main_sections(folder):  _interactive_loop(folder, page_fig_sections)
def main_dimensions(folder): _interactive_loop(folder, page_fig_dimensions)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PDF dimension viewer")
    ap.add_argument("mode", nargs="?", default="classify",
                    choices=["classify", "bottom", "sections", "dimensions"])
    ap.add_argument("folder", help="Folder containing PDFs")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(1)

    {
      "classify" : main_classify,
      "bottom"   : main_bottom,
      "sections" : main_sections,
      "dimensions": main_dimensions
    }[args.mode](args.folder)
