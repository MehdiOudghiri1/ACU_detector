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
    tol_px: int = 17,          # vertical proximity to bbox top/bottom
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

# ─────────────────────────── Band association helpers ───────────────────────────


def find_horizontal_bands_near_number(
    ctx: ExtractionContext,
    bbox_pdf: Tuple[float, float, float, float],
    *,
    tol_px: int = 5,         # MAX distance BELOW bbox center (pixels)
    y_tol_px: int = 2,       # MAX distance ABOVE bbox center (pixels)
    x_edge_tol_px: int = 27,  # tolerance for x-edge proximity (pixels)
    bde: "BandeExtractor",
) -> Dict[str, List["Bande"]]:
    """
    Accept H bands that satisfy BOTH:
      1) Vertical gating (relative to bbox vertical center, pixel top-origin):
         - near_above: b_cy <= cy_px and (cy_px - b_cy) <= y_tol_px
         - near_below: b_cy >= cy_px and (b_cy - cy_px) <= tol_px
      2) Horizontal gating (any ONE of):
         - |h.x0 - bbox_x1| <= x_edge_tol_px   (band starts near RIGHT edge)
         - |h.x1 - bbox_x0| <= x_edge_tol_px   (band ends   near LEFT  edge)
         - h.x1 > bbox_x1 AND h.x0 < bbox_x0   (band spans across bbox)
    """
    assert bde is not None and getattr(bde, "result", None) is not None, "Provide a BandeExtractor with .result"
    horizontals: List["Bande"] = bde.result.get("horizontal", [])

    x0_pt, top_pt, x1_pt, bottom_pt = bbox_pdf
    # pixel (top-origin for Y)
    left_px   = ctx.pt_to_px_x(x0_pt)
    right_px  = ctx.pt_to_px_x(x1_pt)
    cy_px     = ctx.pt_to_px_y_top((top_pt + bottom_pt) / 2.0)

    near_above: List["Bande"] = []
    near_below: List["Bande"] = []

    for b in horizontals:
        if getattr(b, "orientation", "") != "horizontal":
            continue

        # Vertical gating wrt bbox center
        b_cy = (b.y0 + b.y1) / 2.0
        dy = b_cy - cy_px  # >0 below, <0 above

        # Horizontal gating: edge-touch or spanning across bbox
        touches_right_edge = abs(b.x0 - right_px) <= x_edge_tol_px
        touches_left_edge  = abs(b.x1 - left_px)  <= x_edge_tol_px
        spans_across       = (b.x1 > right_px) and (b.x0 < left_px)
        h_ok = touches_right_edge or touches_left_edge or spans_across
        if not h_ok:
            continue

        # Asymmetric vertical windows (keep your original ordering/behavior)
        if -y_tol_px <= dy <= 0:
            near_above.append(b)
        elif 0 <= dy <= tol_px:
            near_below.append(b)

    return {
        "near_above": near_above,
        "near_below": near_below,
        "all_horizontal": horizontals,
    }




def find_vertical_lines_near_hbands(
    h_bands: List["Bande"],
    *,
    bde: "BandeExtractor",
    min_y0_diff_px: int = 9,
    min_bottom_extension_px: int = 9,
) -> List[Tuple["Bande", List["Bande"]]]:
    """
    Inverse linkage: for each HORIZONTAL band h, return VERTICAL bands v that:
      1) start at least `min_y0_diff_px` BELOW h.y0  (v.y0 - h.y0 >= min_y0_diff_px), AND
      2) extend at least `min_bottom_extension_px` beyond h.y1 (v.y1 - h.y1 >= min_bottom_extension_px).
    """
    assert bde is not None and getattr(bde, "result", None) is not None, \
        "Provide a BandeExtractor with .result populated (call extract() first)."

    verticals: List["Bande"] = bde.result.get("vertical", [])
    out: List[Tuple["Bande", List["Bande"]]] = []

    for h in h_bands:
        if getattr(h, "orientation", "") != "horizontal":
            continue
        hy0, hy1 = h.y0, h.y1

        matches = [
            v for v in verticals
            if getattr(v, "orientation", "") == "vertical"
            and abs(v.y0 - hy0) <= min_y0_diff_px
            or abs(v.y1 - hy1) <= min_bottom_extension_px
        ]
        out.append((h, matches))
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

    # Headless backend if needed
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data_extractor import ExtractionContext
    from band_extractor import BandeExtractor
    from number_binding_extractor import NumberBindingExtractor
    # helpers expected in scope:
    #   - find_vertical_bands_near_number(...)
    #   - find_horizontal_bands_near_number(...)
    #   - find_horizontal_lines_near_vbands(...)
    #   - find_vertical_lines_near_hbands(...)

    parser = argparse.ArgumentParser(
        description=(
            "Toujours calculer les bandes VERTICALES près des bboxes (bindings/numbers) d'abord.\n"
            "Modes:\n"
            "  --v (défaut): bbox → V près du bbox → H près de ces V (on affiche V+H)\n"
            "  --h        : bbox → V près du bbox (blacklist si V trouvée)\n"
            "               puis uniquement pour les bboxes NON-blacklistées: H près du bbox\n"
            "               et V près de ces H (on n'affiche PAS les V du PASS A)"
        )
    )
    parser.add_argument("pdf", type=str, help="Path to a single PDF file")
    parser.add_argument("--page", type=int, default=0, help="Zero-based page index")
    parser.add_argument("--dpi", type=int, default=150, help="Rasterization DPI")

    # Bande extraction params
    parser.add_argument("--thr", type=int, default=60, help="Black threshold 0..255")
    parser.add_argument("--min-thick", type=int, default=1, help="Min band thickness (px)")
    parser.add_argument("--max-thick", type=int, default=20, help="Max band thickness (px)")
    parser.add_argument("--v-min-len", type=int, default=20, help="Vertical: min length (px)")
    parser.add_argument("--h-min-len", type=int, default=17, help="Horizontal: min length (px)")
    parser.add_argument("--bridge", type=int, default=1, help="Bridge tiny gaps (px)")

    # Tolerances for band↔bbox association
    parser.add_argument("--tol", type=int, default=17, help="V: proximité verticale au bbox; H: distance sous centre")
    parser.add_argument("--x-tol", type=int, default=12, help="V: proximité du centre X; H: distance au-dessus du centre")

    # NumberBindingExtractor tolerances
    parser.add_argument("--pair-radius", type=float, default=120.0)
    parser.add_argument("--x-align", type=float, default=7.0)
    parser.add_argument("--y-align", type=float, default=7.0)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--v", action="store_true", help="Vertical-first linkage to H (default)")
    mode.add_argument("--h", action="store_true", help="Horizontal linkage uses ONLY bboxes that yielded NO verticals")

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", type=str, default=None)
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

        # 2) Keep ONLY bindings and numbers (use THEIR bbox)
        nbe = NumberBindingExtractor(
            pair_radius_px=args.pair_radius,
            x_tol_px=args.x_align,
            y_tol_px=args.y_align,
        )
        items = nbe.extract(ctx, shared={})
        targets = [it for it in items if it["type"] in {"binding", "number"}]
        logging.info("Targets (bindings+numbers): %d", len(targets))

        # Freeze target bboxes (we never mutate this list)
        target_bboxes: List[Tuple[float, float, float, float]] = [it["bbox"] for it in targets]

        # 3) PASS A — compute V near bbox for ALL targets (always first)
        v_near_per_target: List[List[Bande]] = []
        blacklist: Set[Tuple[float, float, float, float]] = set()  # bboxes that yielded any vertical(s)

        for bbox_pdf in target_bboxes:
            v_found = find_vertical_bands_near_number(
                ctx, bbox_pdf, tol_px=args.tol, x_tol_px=args.x_tol, bde=bde
            )
            v_near = v_found["near_top"] + v_found["near_bottom"]
            v_near_per_target.append(v_near)
            if len(v_near) > 0:
                blacklist.add(bbox_pdf)

        logging.info("Blacklist size (bboxes that produced verticals in PASS A): %d", len(blacklist))

        # Accumulators for drawing (unique rectangles)
        all_vertical: Set[Tuple[int, int, int, int]] = set()
        all_horizontal: Set[Tuple[int, int, int, int]] = set()

        horizontal_mode = bool(args.h) and not bool(args.v)

        if horizontal_mode:
            # --h: DO NOT draw PASS-A verticals. Only:
            #  - H near non-blacklisted bboxes
            #  - V near those H
            used_count = 0
            for bbox_pdf in target_bboxes:
                if bbox_pdf in blacklist:
                    continue  # skip blacklisted bboxes for H detection

                # H near bbox (above<=x_tol, below<=tol)
                h_found = find_horizontal_bands_near_number(
                    ctx, bbox_pdf, tol_px=args.tol, y_tol_px=args.x_tol, bde=bde
                )
                h_near = h_found.get("near_above", []) + h_found.get("near_below", [])

                # Add H
                for h in h_near:
                    all_horizontal.add((int(h.x0), int(h.x1), int(h.y0), int(h.y1)))

                # V near those H
                h_to_v = find_vertical_lines_near_hbands(
                    h_near, bde=bde, min_y0_diff_px=9, min_bottom_extension_px=9
                )
                for _, v_list in h_to_v:
                    for v in v_list:
                        all_vertical.add((int(v.x0), int(v.x1), int(v.y0), int(v.y1)))

                used_count += 1

            logging.info("H detection used %d non-blacklisted bboxes; skipped %d blacklisted.",
                         used_count, len(blacklist))

        else:
            # --v (default): Draw PASS-A verticals and then H near those V
            for v_near in v_near_per_target:
                # Add PASS-A verticals
                for v in v_near:
                    all_vertical.add((int(v.x0), int(v.x1), int(v.y0), int(v.y1)))

                # H near these V
                v_to_h = find_horizontal_lines_near_vbands(
                    v_near, bde=bde, min_x0_diff_px=9, min_right_extension_px=9
                )
                for _, h_list in v_to_h:
                    for h in h_list:
                        all_horizontal.add((int(h.x0), int(h.x1), int(h.y0), int(h.y1)))

        logging.info("Unique vertical: %d | unique horizontal: %d", len(all_vertical), len(all_horizontal))

        # 4) Draw overlays
        base = ctx.pil().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        # Vertical bands (red)
        for (x0, x1, y0, y1) in all_vertical:
            l, r = sorted([x0, x1]); t, b = sorted([y0, y1])
            draw.rectangle([l, t, r, b], outline=(255, 0, 0, 255), width=3)

        # Horizontal bands (blue)
        for (x0, x1, y0, y1) in all_horizontal:
            l, r = sorted([x0, x1]); t, b = sorted([y0, y1])
            draw.rectangle([l, t, r, b], outline=(0, 102, 255, 255), width=3)

        composed = Image.alpha_composite(base, overlay).convert("RGB")

        # 5) Show / save
        raw_rgb = ctx.pil().convert("RGB")
        plt.figure(figsize=(12, 7))
        plt.subplot(1, 2, 1); plt.title(f"RAW — {pdf_path.name} p{args.page+1}")
        plt.imshow(raw_rgb); plt.axis("off")
        mode_title = "H mode (V first for blacklist only; draw H near bbox & V near H)" if horizontal_mode \
                     else "V mode (draw V near bbox & H near V)"
        plt.subplot(1, 2, 2); plt.title(mode_title)
        plt.imshow(composed); plt.axis("off")
        plt.tight_layout(); plt.show()

        if args.out:
            out_path = Path(args.out).expanduser().resolve()
            composed.save(str(out_path))
            logging.info("Saved overlay → %s", out_path)


if __name__ == "__main__":
    main()

