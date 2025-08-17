#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
find_numbers_v4.py

Organized, version-agnostic PDF page analysis with four extractors:

  • WhiteBandExtractor       – find the blank horizontal band (top/bottom separator)
  • BottomNumbersExtractor   – find the bottom-row numeric boxes
  • SectionLinesExtractor    – find thin vertical section-split lines
  • DimensionExtractor       – extract base / cabinet_width / cabinet_height / base_height

Design:
  - ExtractionContext caches page -> image/PIL/gray/words/numeric boxes (no recompute)
  - Each extractor stores its result in self.result AND returns it from extract()
  - plot(self, img) signature everywhere; img is the pdfplumber image
  - Dependencies flow via a shared dict (no repeated white-band detection, etc.)
  - Type hints avoid pdfplumber internals so it works across versions

Usage:
  python find_numbers_v4.py [whiteband|bottom|sections|dimensions|full] /path/to/pdfs [--file FILE.pdf] [--dpi 150]

Navigation:
  - Tk window shows the annotated page
  - Press Enter for next random page; type 'q' or 'quit' to exit.
"""
import os, sys, random, argparse
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any

# XDG runtime fix for Tk on some Linux setups
xdg = os.environ.get("XDG_RUNTIME_DIR", "")
if not xdg or not os.access(xdg, os.W_OK):
    os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import numpy as np
from PIL import Image
import pdfplumber

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ────────────────────────────── Context & Helpers ──────────────────────────────

class ExtractionContext:
    """
    Shared, cached resources for a single pdfplumber Page at a given DPI.
    Handles conversions between pixel space (PIL, origin top-left) and PDF points
    (origin bottom-left).
    """
    def __init__(self, page: pdfplumber.page.Page, dpi: int = 150):
        self.page = page
        self.dpi = dpi
        self.scale = dpi / 72.0  # points → pixels

        # lazy caches
        self._img: Optional[Any] = None    # pdfplumber image object (type varies by version)
        self._pil: Optional[Image.Image] = None
        self._gray: Optional[np.ndarray] = None
        self._words: Optional[List[Dict]] = None
        self._num_boxes: Optional[List[Tuple[float, float, float, float]]] = None
        self._pil_w: Optional[int] = None
        self._pil_h: Optional[int] = None

    # ---- lazy resources ----
    def img(self) -> Any:
        if self._img is None:
            self._img = self.page.to_image(resolution=self.dpi)
        return self._img

    def pil(self) -> Image.Image:
        if self._pil is None:
            self._pil = self.img().original
            self._pil_w, self._pil_h = self._pil.size
        return self._pil

    def gray(self) -> np.ndarray:
        if self._gray is None:
            self._gray = np.array(self.pil().convert("L"))
        return self._gray

    def words(self) -> List[Dict]:
        if self._words is None:
            self._words = self.page.extract_words()
        return self._words

    # ---- numeric text utilities ----
    @staticmethod
    def _is_number(txt: str) -> bool:
        t = txt.replace(",", "").strip()
        try:
            float(t)
            return True
        except ValueError:
            return False

    def number_bboxes(self) -> List[Tuple[float, float, float, float]]:
        """Bounding boxes (x0, top, x1, bottom) in PDF coords (points)."""
        if self._num_boxes is None:
            self._num_boxes = [
                (w["x0"], w["top"], w["x1"], w["bottom"])
                for w in self.words()
                if self._is_number(w.get("text", ""))
            ]
        return self._num_boxes

    # ---- conversions: pixel <-> PDF (points) ----
    def px_to_pt_x(self, x_px: float) -> float:
        return x_px / self.scale

    def px_to_pt_y(self, y_px: float) -> float:
        # pixel origin: top-left; PDF origin: bottom-left
        return (self.pil_h() - y_px) / self.scale

    def pt_to_px_x(self, x_pt: float) -> float:
        return x_pt * self.scale

    def pt_to_px_y(self, y_pt: float) -> float:
        return self.pil_h() - (y_pt * self.scale)

    def pil_w(self) -> int:
        if self._pil_w is None: self.pil()
        return self._pil_w

    def pil_h(self) -> int:
        if self._pil_h is None: self.pil()
        return self._pil_h


# ───────────────────────────── Base Extractor API ──────────────────────────────

class DataExtractor(ABC):
    """Base class: store result as attribute; extract() also returns it."""
    def __init__(self, name: str):
        self.name = name
        self.result: Any = None
        self._ctx: Optional[ExtractionContext] = None

    @abstractmethod
    def extract(self, ctx: ExtractionContext, shared: Dict[str, object]):
        """Compute and store results in self.result, also return it."""
        ...

    @abstractmethod
    def plot(self, img: Any):
        """Draw overlays on a pdfplumber image object (PDF coords)."""
        ...

    # Utility (kept for parity with older code)
    @staticmethod
    def _max_run_length(bool_arr: np.ndarray) -> int:
        best = cur = 0
        for v in bool_arr:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    def _set_ctx(self, ctx: ExtractionContext):
        self._ctx = ctx


# ───────────────────────────── White Band Extractor ────────────────────────────

class WhiteBandExtractor(DataExtractor):
    """
    Finds a blank (white) horizontal band between lower_frac*H and upper_frac*H.
    Result: dict {'y_top_px': int, 'y_bot_px': int}
    """
    def __init__(self, lower_frac=0.2, upper_frac=0.8, min_height_px=33):
        super().__init__("WhiteBandExtractor")
        self.lower_frac = lower_frac
        self.upper_frac = upper_frac
        self.min_height_px = min_height_px

    def _find_white_band_px(self, ctx: ExtractionContext) -> Optional[Tuple[int, int]]:
        g = ctx.gray()
        H, _ = g.shape
        y_min, y_max = int(H * self.lower_frac), int(H * self.upper_frac)
        blank_row = np.all(g == 255, axis=1)

        # scan downward (from upper bound to lower bound)
        for y in range(y_max - 1, y_min - 1, -1):
            if blank_row[y]:
                b = y
                t = y
                while t - 1 >= y_min and blank_row[t - 1]:
                    t -= 1
                if b - t + 1 >= self.min_height_px:
                    return t, b + 1
        return None

    def extract(self, ctx: ExtractionContext, shared: Dict[str, object]):
        self._set_ctx(ctx)
        band = self._find_white_band_px(ctx)
        if band:
            self.result = {"y_top_px": band[0], "y_bot_px": band[1]}
        else:
            self.result = None
        return self.result

    def plot(self, img: Any):
        if not self.result or not self._ctx:
            return
        y_top_px = self.result["y_top_px"]
        y_bot_px = self.result["y_bot_px"]

        x0_pt = 0.0
        x1_pt = self._ctx.page.width

        # Convert pixel→PDF, then ORDER so y0 <= y1
        y0_pt = self._ctx.px_to_pt_y(y_top_px)
        y1_pt = self._ctx.px_to_pt_y(y_bot_px)
        top_pt, bot_pt = (y0_pt, y1_pt) if y0_pt <= y1_pt else (y1_pt, y0_pt)

        img.draw_rect((x0_pt, top_pt, x1_pt, bot_pt), stroke="lime", stroke_width=3)


# ─────────────────────────── Bottom Numbers Extractor ──────────────────────────

class BottomNumbersExtractor(DataExtractor):
    """
    Finds the lowest-row numeric boxes (tolerance in pixels).
    Result: list of dicts: [{ 'bbox': (x0, top, x1, bottom), 'text': '...' }, ...]
    """
    def __init__(self, tolerance_px: int = 4):
        super().__init__("BottomNumbersExtractor")
        self.tolerance_px = tolerance_px

    def _find_bottom_numbers(self, ctx: ExtractionContext) -> List[Tuple[float, float, float, float]]:
        boxes = ctx.number_bboxes()
        if not boxes:
            return []

        # Convert centers to image pixel Y (origin top-left)
        centers_y_px = [ (ctx.page.height - (t + b) / 2.0) * ctx.scale for _, t, _, b in boxes ]
        top_y = min(centers_y_px)
        return [bx for bx, cy in zip(boxes, centers_y_px) if (cy - top_y) <= self.tolerance_px]

    def extract(self, ctx: ExtractionContext, shared: Dict[str, object]):
        self._set_ctx(ctx)
        bottom_boxes = self._find_bottom_numbers(ctx)
        items = []
        for (x0, t, x1, b) in bottom_boxes:
            try:
                txt = ctx.page.within_bbox((x0, t, x1, b)).extract_text() or ""
                txt = txt.strip()
            except Exception:
                txt = ""
            items.append({"bbox": (x0, t, x1, b), "text": txt})
        self.result = items
        return self.result

    def plot(self, img: Any):
        if not self.result:
            return
        for item in self.result:
            img.draw_rect(item["bbox"], stroke="yellow", stroke_width=3)


# ─────────────────────────── Section Lines Extractor ───────────────────────────

class SectionLinesExtractor(DataExtractor):
    """
    Finds thin vertical dark runs that cross around the average Y of the bottom numbers,
    excluding columns too close to those bottom boxes.

    Result: dict with:
      - 'lines_px': [(x_px, y0_px, y1_px), ...]
      - 'avg_y_px': float
    """
    def __init__(self, dark_thr=40, max_line_length_px=150, search_every=1, pad_px=4):
        super().__init__("SectionLinesExtractor")
        self.dark_thr = dark_thr
        self.max_len = max_line_length_px
        self.search_every = search_every
        self.pad_px = pad_px

    @staticmethod
    def _dedup_lines(lines: List[Tuple[int, int, int]], gap=3, y_tol=5) -> List[Tuple[int, int, int]]:
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

    def _compute(self, ctx: ExtractionContext, bottom_boxes: List[Tuple[float, float, float, float]]):
        g = ctx.gray()
        H, W = g.shape

        # avg y (in pixels) across bottom boxes (use centers in pixel space)
        if bottom_boxes:
            centers_px = [ ((t + b) / 2.0) * ctx.scale for _, t, _, b in bottom_boxes ]
            avg_y_px = float(np.mean(centers_px))
        else:
            avg_y_px = H * 0.75  # fallback: lower part of page

        # collect all vertical dark runs that bracket avg_y_px
        all_lines = []
        for col in range(0, W, self.search_every):
            start = None
            for row in range(H):
                if g[row, col] < self.dark_thr:
                    if start is None:
                        start = row
                else:
                    if start is not None:
                        length = row - start
                        if length < self.max_len and start < avg_y_px < row:
                            all_lines.append((col, start, row))
                        start = None
            if start is not None:
                length = H - start
                if length < self.max_len and start < avg_y_px < H:
                    all_lines.append((col, start, H))

        # exclude columns too close to bottom boxes
        filtered = []
        for col, y0, y1 in all_lines:
            too_close = False
            for x0, t, x1, b in bottom_boxes:
                x0_px = x0 * ctx.scale
                x1_px = x1 * ctx.scale
                if (x0_px - self.pad_px) <= col <= (x1_px + self.pad_px):
                    too_close = True
                    break
            if not too_close:
                filtered.append((col, y0, y1))

        return self._dedup_lines(filtered), avg_y_px

    def extract(self, ctx: ExtractionContext, shared: Dict[str, object]):
        self._set_ctx(ctx)
        # Prefer bottom boxes from shared; else compute once here:
        bottom = shared.get("BottomNumbersExtractor")
        if bottom is None:
            tmp = BottomNumbersExtractor()
            bottom = tmp.extract(ctx, shared)
        bottom_boxes = [d["bbox"] for d in (bottom or [])]

        lines_px, avg_y_px = self._compute(ctx, bottom_boxes)
        self.result = {"lines_px": lines_px, "avg_y_px": avg_y_px}
        return self.result

    def plot(self, img: Any):
        if not self.result or not self._ctx:
            return
        for x_px, y0_px, y1_px in self.result["lines_px"]:
            x0_pt = self._ctx.px_to_pt_x(x_px)
            x1_pt = self._ctx.px_to_pt_x(x_px + 1)

            a_pt = self._ctx.px_to_pt_y(y0_px)
            b_pt = self._ctx.px_to_pt_y(y1_px)
            top_pt, bot_pt = (a_pt, b_pt) if a_pt <= b_pt else (b_pt, a_pt)

            img.draw_rect((x0_pt, top_pt, x1_pt, bot_pt), stroke="green", stroke_width=2)



# ───────────────────────────── Dimension Extractor ─────────────────────────────

class DimensionExtractor(DataExtractor):
    """
    Extracts four key dimensions by reusing the white band split and (optionally)
    the left-most section line as a column limit.

    Result: dict with optional keys:
      - 'base':           (x0, top, x1, bottom, text)
      - 'cabinet_width':  (x0, top, x1, bottom, text)
      - 'cabinet_height': (x0, top, x1, bottom, text)
      - 'base_height':    (x0, top, x1, bottom, text)
    """
    def __init__(self, left_margin_px: float = 30.0):
        super().__init__("DimensionExtractor")
        self.left_margin_px = left_margin_px

    def _tokens_by_band(self, ctx: ExtractionContext, band_px: Tuple[int, int]):
        y_top_px, y_bot_px = band_px
        tokens = [
            (w["x0"], w["top"], w["x1"], w["bottom"], w["text"])
            for w in ctx.words() if ctx._is_number(w.get("text", ""))
        ]

        top_toks, bot_toks = [], []
        for x0, t, x1, b, txt in tokens:
            cy_px = ((t + b) / 2.0) * ctx.scale
            if cy_px < y_top_px:
                top_toks.append((x0, t, x1, b, txt))
            elif cy_px > y_bot_px:
                bot_toks.append((x0, t, x1, b, txt))
        return top_toks, bot_toks

    @staticmethod
    def _center_pt(box) -> Tuple[float, float]:
        x0, t, x1, b, _ = box
        return ( (x0 + x1) / 2.0, (t + b) / 2.0 )

    def _extract(self, ctx: ExtractionContext, band_px: Tuple[int, int], limit_px: Optional[float]):
        top_toks, bot_toks = self._tokens_by_band(ctx, band_px)
        dims: Dict[str, Tuple[float, float, float, float, str]] = {}

        limit_pt = (limit_px / ctx.scale) if limit_px is not None else (ctx.page.width / 2.0)
        left_mode = any(x0 < limit_pt for x0, *_ in top_toks)

        # BASE + CABINET WIDTH (top band)
        if top_toks:
            if left_mode:
                base_tok = min(top_toks, key=lambda tk: tk[0])  # smallest x0
                rest = [tk for tk in top_toks if tk[0] > base_tok[0]]
            else:
                base_tok = max(top_toks, key=lambda tk: tk[0])  # largest x0
                rest = [tk for tk in top_toks if tk[0] < base_tok[0]]

            dims["base"] = base_tok

            if rest:
                base_x0 = base_tok[0]
                diffs = []
                for tk in sorted(rest, key=lambda tk: tk[0], reverse=not left_mode)[:6]:
                    gap = (tk[0] - base_x0) if left_mode else (base_x0 - tk[0])
                    if gap > 0:
                        diffs.append((gap, tk))
                if diffs:
                    min_gap = min(diffs, key=lambda x: x[0])[0]
                    tol_pt = 3.0 / ctx.scale  # ≈3 px
                    group = [tk for gap, tk in diffs if gap <= (min_gap + tol_pt)]
                    if group:
                        bx, by = self._center_pt(base_tok)
                        def dist2(tok):
                            tx, ty = self._center_pt(tok)
                            return (tx - bx) ** 2 + (ty - by) ** 2
                        best = min(group, key=dist2)
                    else:
                        best = min(diffs, key=lambda x: x[0])[1]
                    dims["cabinet_width"] = best

        # CABINET HEIGHT + BASE HEIGHT (bottom band)
        bottom_left_mode = any(x0 < limit_pt for x0, *_ in bot_toks)
        if bot_toks:
            bot_sorted = sorted(bot_toks, key=lambda tk: tk[0], reverse=not bottom_left_mode)
            height_tok = bot_sorted[0]
            dims["cabinet_height"] = height_tok

            hx0, ht, hx1, hb, _ = height_tok
            hcx = (hx0 + hx1) / 2.0
            for tk in bot_sorted[1:]:
                cx = (tk[0] + tk[2]) / 2.0
                if abs(cx - hcx) <= (3.0 / ctx.scale):  # ≈3 px tolerance
                    dims["base_height"] = tk
                    break

        return dims

    def extract(self, ctx: ExtractionContext, shared: Dict[str, object]):
        self._set_ctx(ctx)

        # Need the band; prefer shared WhiteBandExtractor; fallback compute locally
        wb = shared.get("WhiteBandExtractor")
        if wb and wb is not None:
            band_px = (wb["y_top_px"], wb["y_bot_px"])
        else:
            wb_local = WhiteBandExtractor().extract(ctx, shared)
            band_px = (wb_local["y_top_px"], wb_local["y_bot_px"]) if wb_local else None

        if not band_px:
            self.result = {}
            return self.result

        # Optional: use left-most section line as limit; else mid-page
        limit_px = None
        sl = shared.get("SectionLinesExtractor")
        if sl and sl.get("lines_px"):
            limit_px = min(x for x, _, _ in sl["lines_px"]) - self.left_margin_px
        else:
            limit_px = (ctx.pil_w() / 2.0) - self.left_margin_px

        dims = self._extract(ctx, band_px, limit_px)
        self.result = dims
        return self.result

    def plot(self, img: Any):
        if not self.result:
            return
        color_map = {
            "base": "orange",
            "cabinet_width": "darkorange",
            "cabinet_height": "teal",
            "base_height": "magenta",
        }
        for k, box in self.result.items():
            x0, t, x1, b, _ = box
            img.draw_rect((x0, t, x1, b), stroke=color_map.get(k, "white"), stroke_width=3)


# ─────────────────────────────── Pipeline / Runner ─────────────────────────────

MODES: Dict[str, List[DataExtractor]] = {
    "whiteband":   [WhiteBandExtractor()],
    "bottom":      [WhiteBandExtractor(), BottomNumbersExtractor()],
    "sections":    [WhiteBandExtractor(), BottomNumbersExtractor(), SectionLinesExtractor()],
    "dimensions":  [WhiteBandExtractor(), BottomNumbersExtractor(), SectionLinesExtractor(), DimensionExtractor()],
    "full":        [WhiteBandExtractor(), BottomNumbersExtractor(), SectionLinesExtractor(), DimensionExtractor()],
}

def render_page(page: pdfplumber.page.Page, extractors: List[DataExtractor], dpi: int):
    ctx = ExtractionContext(page, dpi=dpi)
    img = ctx.img()
    shared: Dict[str, object] = {}

    for ext in extractors:
        data = ext.extract(ctx, shared)
        shared[ext.name] = data
        ext.plot(img)

    return img, {ext.name: ext.result for ext in extractors}

def _print_results(results: Dict[str, object]):
    def _short_box(b):
        if not b: return b
        if isinstance(b, (list, tuple)) and len(b) >= 4 and all(isinstance(v, (int,float)) for v in b[:4]):
            x0, t, x1, btm = b[:4]
            return (round(x0,1), round(t,1), round(x1,1), round(btm,1))
        return b

    for name, data in results.items():
        print(f"\n{name}:")
        if data is None:
            print("  None")
        elif name == "BottomNumbersExtractor":
            print(f"  count={len(data)}")
            for it in data[:6]:
                print("   ", _short_box(it['bbox']), repr(it.get('text',''))[:40])
            if len(data) > 6:
                print("   ...")
        elif name == "SectionLinesExtractor":
            lines = data.get("lines_px", [])
            avg_y = data.get("avg_y_px", None)
            print(f"  lines={len(lines)}, avg_y_px={round(avg_y,1) if avg_y is not None else None}")
            for x,y0,y1 in lines[:8]:
                print(f"    (x={x}, y0={y0}, y1={y1})")
            if len(lines) > 8:
                print("    ...")
        elif name == "DimensionExtractor":
            if not data:
                print("  {}")
            else:
                for k, v in data.items():
                    print(f"  {k}: {_short_box(v)} text={repr(v[4])[:32]}")
        else:
            print(f"  {data}")

def interactive_loop(folder: str, extractors: List[DataExtractor], specific_file: Optional[str], dpi: int):
    pdfs = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    if not pdfs:
        print("No PDFs found in:", folder, file=sys.stderr)
        sys.exit(1)

    if specific_file:
        if specific_file not in pdfs:
            print(f"File '{specific_file}' not found in folder '{folder}'", file=sys.stderr)
            sys.exit(1)
        pdfs = [specific_file]

    print("Press Enter for next page, 'q' to quit…")
    while True:
        pdf_name = random.choice(pdfs)
        path = os.path.join(folder, pdf_name)
        with pdfplumber.open(path) as pdf:
            page = random.choice(pdf.pages)
            img, results = render_page(page, extractors, dpi=dpi)

        print(f"\n{pdf_name} — page {page.page_number}")
        _print_results(results)

        fig, ax = plt.subplots(figsize=(8, 10))
        ax.imshow(img.annotated)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

        key = input().strip().lower()
        if key in {"q", "quit", "exit"}:
            plt.close("all")
            break
        plt.close("all")


# ──────────────────────────────────── CLI ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PDF data extractor (organized, cached, fast, and version-agnostic).")
    parser.add_argument("mode", choices=list(MODES.keys()),
                        help="Extraction mode: whiteband | bottom | sections | dimensions | full")
    parser.add_argument("folder", help="Folder containing PDF files")
    parser.add_argument("--file", dest="specific_file", help="Restrict to a single PDF in the folder (FILE.pdf)")
    parser.add_argument("--dpi", type=int, default=150, help="Rendering DPI (affects detection precision)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Folder not found: {args.folder}", file=sys.stderr)
        sys.exit(1)

    extractors = MODES[args.mode]
    interactive_loop(args.folder, extractors, args.specific_file, dpi=args.dpi)


if __name__ == "__main__":
    main()
