#!/usr/bin/env python3
"""
data_extractor.py

A clean, scalable framework for extracting and plotting PDF page data:

• WhiteBandExtractor     → finds the blank white band separating top/bottom
• BottomExtractor        → finds the bottom‐row numbers
• SectionLinesExtractor  → finds vertical section‐split lines
• DimensionExtractor     → extracts the four key dimensions

Each extractor implements:
    extract(page) → raw data
    plot(img, data) → draws overlays on a pdfplumber Image

The runner ties them together:
    python data_extractor.py [whiteband|bottom|sections|dimensions|full] /path/to/pdfs
"""
import os, sys, random
import numpy as np
from PIL import Image            # ← add this!
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse           # ← add this!

from abc import ABC, abstractmethod
from find_numbers import (
    find_first_white_band,
    number_bboxes,
    find_bottom_numbers,
    find_section_lines,
    find_dimensions,
    SCALE, DPI
)
import pdfplumber

# ─────────────────────────── Base Interface ──────────────────────────────
class DataExtractor(ABC):
    @abstractmethod
    def extract(self, page):
        """Compute and return raw data from the page."""
        pass

    @abstractmethod
    def plot(self, img, data):
        """Draw overlays on the pdfplumber Image according to data."""
        pass

class WhiteBandExtractor(DataExtractor):
    """
    Extracts the blank band separating top vs bottom by re‐using
    find_first_white_band() verbatim, then returns its four corners.
    """

    def __init__(self, dpi: int = 150):
        self.dpi = dpi

    def extract(self, page):
        # render to PIL so we can examine raw pixels
        pil = page.to_image(resolution=self.dpi).original

        # call your original function unchanged
        band = find_first_white_band(pil,
                                    lower_frac=0.2,
                                    upper_frac=0.8,
                                    min_height=33)
        if not band:
            return None

        y0, y1 = band
        w, _  = pil.size

        # build the four‐corner tuple (x0,y0, x1,y0, x0,y1, x1,y1)
        return (0,  y0,
                w,  y0,
                0,  y1,
                w,  y1)

    def plot(self, img, quad):
        """
        Draws exactly the same band rectangle as your old code:
            ax.add_patch(Rectangle((0,y0), width, y1-y0, …))
        but here using img.draw_rect for consistency with the rest.
        """
        if quad is None:
            return

        x0, y0, x1, _, _, y1, _, _ = quad

        # identical to old file's single-line Rectangle drawing:
        img.draw_rect(
            (x0, y0, x1, y1),
            stroke="lime",        # same colour as in page_fig_classification
            stroke_width=3        # same thickness
        )

# ──────────────────────── 2) Bottom Numbers ─────────────────────────────
class BottomExtractor(DataExtractor):
    def extract(self, page):
        boxes = number_bboxes(page)
        bottom_boxes = find_bottom_numbers(boxes, page.height, SCALE)
        texts = []
        for x0, t, x1, b in bottom_boxes:
            try:
                txt = page.within_bbox((x0, t, x1, b)).extract_text().strip()
            except:
                txt = ""
            texts.append(txt)
        return list(zip(bottom_boxes, texts))

    def plot(self, img, data):
        for (x0, t, x1, b), _ in data:
            img.draw_rect((x0, t, x1, b), stroke="yellow", stroke_width=2)

# ─────────────────── 3) Section Split Lines ──────────────────────────────
def _dedup_lines(lines, gap=3, y_tol=5):
    if not lines:
        return []
    lines = sorted(lines, key=lambda x: x[0])
    merged = [lines[0]]
    for x, y0, y1 in lines[1:]:
        lx, ly0, ly1 = merged[-1]
        if abs(x - lx) <= gap and abs(y0 - ly0) <= y_tol and abs(y1 - ly1) <= y_tol:
            continue
        merged.append((x, y0, y1))
    return merged

class SectionLinesExtractor(DataExtractor):
    def extract(self, page):
        # need bottom_boxes to exclude nearby lines
        boxes = number_bboxes(page)
        bottom_boxes = find_bottom_numbers(boxes, page.height, SCALE)
        pil = page.to_image(resolution=DPI).original
        raw, _ = find_section_lines(pil, bottom_boxes, page.height)
        return _dedup_lines(raw)

    def plot(self, img, lines):
        for x, y0, y1 in lines:
            img.draw_rect((x, y0, x+1, y1), stroke="green", stroke_width=2)

# ──────────────────────── 4) Dimensions ─────────────────────────────────
class DimensionExtractor(DataExtractor):
    def extract(self, page):
        """
        1) compute `limit` = smallest x0 among section‐split lines (PDF points),
           default to half‐page if none found.
        2) pass that `limit` into find_dimensions(page, limit).
        """
        # gather vertical splits to determine `limit`
        boxes = number_bboxes(page)
        bottom_boxes = find_bottom_numbers(boxes, page.height, SCALE)
        pil = page.to_image(resolution=DPI).original
        lines, _ = find_section_lines(pil, bottom_boxes, page.height)

        if lines:
            limit = min(x for x, _, _ in lines)
        else:
            # default to midpoint of page width (in PDF points)
            limit = page.width / 2.0

        # Now call the updated function
        dims = find_dimensions(page, limit - 30)
        return dims

    def plot(self, img, dims):
        colors = {
            "base":           "orange",
            "cabinet_width":  "darkorange",
            "cabinet_height": "teal",
            "base_height":    "magenta",
        }
        for name, box in dims.items():
            x0, t, x1, b, _ = box
            img.draw_rect((x0, t, x1, b),
                          stroke=colors.get(name, "white"),
                          stroke_width=3)

# ───────────────────────── Runner / CLI ──────────────────────────────────
MODES = {
    "whiteband":   [WhiteBandExtractor()],
    "bottom":      [BottomExtractor()],
    "sections":    [SectionLinesExtractor()],
    "dimensions":  [DimensionExtractor()],
    "full":        [
        WhiteBandExtractor(),
        BottomExtractor(),
        SectionLinesExtractor(),
        DimensionExtractor(),
    ]
}

def render_page(page, extractors):
    img = page.to_image(resolution=DPI)
    results = {}
    for ext in extractors:
        data = ext.extract(page)
        ext.plot(img, data)
        results[ext.__class__.__name__] = data
    return img, results

def main(mode, folder, specific_file=None):
    if mode not in MODES:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)
    extractors = MODES[mode]

    # List all PDFs in the folder
    all_pdfs = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    if not all_pdfs:
        print("No PDFs in", folder, file=sys.stderr)
        sys.exit(1)

    # If --file was provided, validate and use only that
    if specific_file:
        if specific_file not in all_pdfs:
            print(f"File '{specific_file}' not found in folder '{folder}'", file=sys.stderr)
            sys.exit(1)
        pdfs = [specific_file]
    else:
        pdfs = all_pdfs

    print("Press Enter for next page, q to quit…")
    while True:
        pdf_name = random.choice(pdfs)
        path = os.path.join(folder, pdf_name)
        with pdfplumber.open(path) as pdf:
            page = random.choice(pdf.pages)
            img, results = render_page(page, extractors)

        print(f"\n{pdf_name} – page {page.page_number}")
        for name, data in results.items():
            print(f"{name}: {data}")

        fig, ax = plt.subplots(figsize=(8, 10))
        ax.imshow(img.annotated)
        ax.axis("off")
        plt.show()

        if input().strip().lower() in {"q", "quit", "exit"}:
            break
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF dimension viewer – choose mode and optionally a single file"
    )
    parser.add_argument(
        "mode",
        choices=list(MODES.keys()),
        help="Extraction mode: whiteband, bottom, sections, dimensions, or full"
    )
    parser.add_argument(
        "folder",
        help="Folder containing PDF files"
    )
    parser.add_argument(
        "--file",
        dest="specific_file",
        metavar="FILE.pdf",
        help="If set, only this PDF (inside the folder) will be processed"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Folder not found: {args.folder}", file=sys.stderr)
        sys.exit(1)

    main(args.mode, args.folder, args.specific_file)
