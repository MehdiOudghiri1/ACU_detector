# h_extractor.py
# Extract horizontal lines near vertical lines that are close to binding words/numbers.

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional, Set
import logging
import os

import numpy as np
import pdfplumber
from PIL import Image, ImageDraw

from data_extractor import ExtractionContext
from band_extractor import BandeExtractor, Bande
from number_binding_extractor import NumberBindingExtractor


# ─────────────────────────── Helpers: band association ───────────────────────────

def find_vertical_bands_near_number(
    ctx: ExtractionContext,
    bbox_pdf: Tuple[float, float, float, float],
    *,
    tol_px: int = 10,          # vertical proximity to bbox top/bottom
    x_tol_px: int = 12,        # horizontal proximity to bbox center
    bde: "BandeExtractor",
) -> Dict[str, List["Bande"]]:
    """
    Given a number bbox in PDF coords (x0, top, x1, bottom), return vertical bands
    whose y1 is near the bbox TOP (near_top) or whose y0 is near the bbox BOTTOM
    (near_bottom), AND whose horizontal center is near the bbox center (±x_tol_px).

    IMPORTANT:
      pdfplumber words use TOP-LEFT origin for 'top'/'bottom'. Do NOT invert Y.
      Use ctx.pt_to_px_y_top(...) to convert those to pixels.
    """
    assert bde is not None and getattr(bde, "result", None) is not None, "Provide a BandeExtractor with .result"
    vertical_bands: List["Bande"] = bde.result.get("vertical", [])

    # PDF (points) → pixel (top-left origin) conversions
    x0_pt, top_pt, x1_pt, bottom_pt = bbox_pdf
    top_px    = ctx.pt_to_px_y_top(top_pt)
    bottom_px = ctx.pt_to_px_y_top(bottom_pt)
    cx_px     = ctx.pt_to_px_x((x0_pt + x1_pt) / 2.0)  # bbox horizontal center in pixels

    def x_close(b: "Bande") -> bool:
        b_cx = (b.x0 + b.x1) / 2.0  # band horizontal center (pixels)
        return abs(b_cx - cx_px) <= x_tol_px

    near_top: List["Bande"] = [
        b for b in vertical_bands
        if abs(b.y1 - top_px) <= tol_px and x_close(b)
    ]
    near_bottom: List["Bande"] = [
        b for b in vertical_bands
        if abs(b.y0 - bottom_px) <= tol_px and x_close(b)
    ]

    return {
        "near_top": near_top,
        "near_bottom": near_bottom,
        "all_vertical": vertical_bands,  # for debugging if needed
    }


def find_horizontal_lines_near_vbands(
    v_bands: List[Bande],
    *,
    bde: "BandeExtractor",
    min_x0_diff_px: int = 9,
    min_right_extension_px: int = 9,
) -> List[Tuple[Bande, List[Bande]]]:
    """
    For each vertical band v in v_bands, return the horizontal bands h that:
      1) start at least `min_x0_diff_px` to the right of the vertical band's x0
         (h.x0 - v.x0 >= min_x0_diff_px), and
      2) extend to the right of the vertical band's x1 by at least `min_right_extension_px`
         (h.x1 - v.x1 >= min_right_extension_px).

    No maximum threshold is enforced.
    """
    assert bde is not None and getattr(bde, "result", None) is not None, \
        "Provide a BandeExtractor with .result populated (call extract() first)."

    horizontals: List[Bande] = bde.result.get("horizontal", [])
    out: List[Tuple[Bande, List[Bande]]] = []

    for v in v_bands:
        if getattr(v, "orientation", "") != "vertical":
            continue
        vx0, vx1 = v.x0, v.x1
        matches = [
            h for h in horizontals
            if getattr(h, "orientation", "") == "horizontal"
            and abs(h.x0 - vx0) <= min_x0_diff_px
            or abs(h.x1 - vx1) <= min_right_extension_px
        ]
        out.append((v, matches))
    return out


# ───────────────────────────────────── Main ─────────────────────────────────────

def main():
    import argparse
    from pathlib import Path
    import logging
    import os
    import pdfplumber
    import matplotlib
    from PIL import Image, ImageDraw

    # Use a non-interactive backend if headless (avoids Qt/GLX errors)
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data_extractor import ExtractionContext
    from band_extractor import BandeExtractor
    from number_binding_extractor import NumberBindingExtractor
    # expects helpers in scope:
    #   - find_vertical_bands_near_number(ctx, bbox_pdf, tol_px, x_tol_px, bde)
    #   - find_horizontal_lines_near_vbands(v_bands, bde=..., min_x0_diff_px=5, min_right_extension_px=5)

    parser = argparse.ArgumentParser(
        description="Use binding/number bboxes: find vertical bands near them, then nearby horizontal bands; draw vertical (red) and horizontal (blue)."
    )
    parser.add_argument("pdf", type=str, help="Path to a single PDF file")
    parser.add_argument("--page", type=int, default=0, help="Zero-based page index")
    parser.add_argument("--dpi", type=int, default=150, help="Rasterization DPI")

    # Bande extraction params
    parser.add_argument("--thr", type=int, default=60, help="Black threshold 0..255")
    parser.add_argument("--min-thick", type=int, default=1, help="Min band thickness (px)")
    parser.add_argument("--max-thick", type=int, default=20, help="Max band thickness (px)")
    parser.add_argument("--v-min-len", type=int, default=20, help="Vertical: min length (px)")
    parser.add_argument("--h-min-len", type=int, default=23, help="Horizontal: min length (px)")
    parser.add_argument("--bridge", type=int, default=1, help="Bridge tiny gaps (px)")

    # Vertical-band ↔ bbox association tolerances
    parser.add_argument("--tol", type=int, default=17, help="Vertical proximity to bbox top/bottom (px)")
    parser.add_argument("--x-tol", type=int, default=12, help="Horizontal proximity to bbox center (px)")

    # NumberBindingExtractor tolerances
    parser.add_argument("--pair-radius", type=float, default=120.0, help="Max px distance to pair BASE with O.D.")
    parser.add_argument("--x-align", type=float, default=7.0, help="Horizontal px tolerance for vertical words")
    parser.add_argument("--y-align", type=float, default=7.0, help="Vertical px tolerance for horizontal words")

    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--out", type=str, default=None, help="Optional output PNG path")
    args = parser.parse_args()

    logging.basicConfig(level=(logging.DEBUG if args.verbose else logging.INFO),
                        format="%(levelname)s %(message)s")

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pdfplumber.open(str(pdf_path)) as pdf:
        if not (0 <= args.page < len(pdf.pages)):
            raise IndexError(f"Page index {args.page} out of range [0..{len(pdf.pages)-1}]")
        page = pdf.pages[args.page]

        # Build context
        ctx = ExtractionContext(page, dpi=args.dpi)

        # 1) Detect bands (both V and H)
        bde = BandeExtractor(
            thr=args.thr,
            min_thick=args.min_thick,
            max_thick=args.max_thick,
            v_min_len_px=args.v_min_len,
            v_max_len_px=None,
            h_min_len_px=args.h_min_len,
            h_max_len_px=None,
            bridge_px=args.bridge,
        )
        bde.extract(ctx, shared={})

        # 2) Build items & keep ONLY bindings and numbers (use THEIR bbox)
        nbe = NumberBindingExtractor(
            pair_radius_px=args.pair_radius,
            x_tol_px=args.x_align,
            y_tol_px=args.y_align,
        )
        items = nbe.extract(ctx, shared={})
        targets = [it for it in items if it["type"] in {"binding", "number"}]
        logging.info("Targets to analyze (bindings + numbers): %d", len(targets))

        # 3) For each target bbox, find nearby vertical bands, then horizontal bands near those verticals
        all_vertical_bands = set()    # tuples (x0, x1, y0, y1)
        all_horizontal_bands = set()  # tuples (x0, x1, y0, y1)

        for i, it in enumerate(targets):
            nb_bbox = it["bbox"]  # <-- use the item's bbox directly (PDF coords)

            v_found = find_vertical_bands_near_number(
                ctx,
                nb_bbox,
                tol_px=args.tol,
                x_tol_px=args.x_tol,
                bde=bde,
            )
            v_near = v_found["near_top"] + v_found["near_bottom"]

            v_to_h = find_horizontal_lines_near_vbands(
                v_near,
                bde=bde,
                min_x0_diff_px=5,
                min_right_extension_px=5,
            )

            for v_band, h_list in v_to_h:
                all_vertical_bands.add((int(v_band.x0), int(v_band.x1), int(v_band.y0), int(v_band.y1)))
                for h in h_list:
                    all_horizontal_bands.add((int(h.x0), int(h.x1), int(h.y0), int(h.y1)))

            logging.debug(
                f"[Target #{i:02d}] v_near={len(v_near)} matched_h={sum(len(hs) for _, hs in v_to_h)}"
            )

        logging.info("Unique vertical bands: %d | Unique horizontal bands: %d",
                     len(all_vertical_bands), len(all_horizontal_bands))

        # 4) Draw: vertical (red), horizontal (blue)
        base = ctx.pil().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        for (x0, x1, y0, y1) in all_vertical_bands:
            l, r = sorted([x0, x1])
            t, b = sorted([y0, y1])
            draw.rectangle([l, t, r, b], outline=(255, 0, 0, 255), width=3)

        for (x0, x1, y0, y1) in all_horizontal_bands:
            l, r = sorted([x0, x1])
            t, b = sorted([y0, y1])
            draw.rectangle([l, t, r, b], outline=(0, 102, 255, 255), width=3)

        composed = Image.alpha_composite(base, overlay).convert("RGB")

        # 5) Show / save
        raw_rgb = ctx.pil().convert("RGB")
        plt.figure(figsize=(12, 7))
        plt.subplot(1, 2, 1); plt.title(f"RAW — {pdf_path.name} p{args.page+1}")
        plt.imshow(raw_rgb); plt.axis("off")
        plt.subplot(1, 2, 2); plt.title("Vertical (red) & Horizontal (blue) near bindings/numbers (bbox-based)")
        plt.imshow(composed); plt.axis("off")
        plt.tight_layout(); plt.show()

        if args.out:
            out_path = Path(args.out).expanduser().resolve()
            composed.save(str(out_path))
            logging.info("Saved overlay → %s", out_path)


if __name__ == "__main__":
    main()
