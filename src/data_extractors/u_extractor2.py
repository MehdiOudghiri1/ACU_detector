# u_extractor.py
# Extract vertical/horizontal lines around binding/number bboxes with two modes:
#   --v (default): bbox → V near bbox → H near those V (draw V+H)
#   --h          : bbox → V near bbox (blacklist those bboxes)
#                  then only for NON-blacklisted bboxes: H near bbox → V near those H (draw H+V)

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional, Set
import logging
import os

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
    v_bands: List["Bande"],
    *,
    bde: "BandeExtractor",
    min_x0_diff_px: int = 9,
    min_right_extension_px: int = 9,
    y_center_margin_px: int = 10,  # NEW: horizontal center must lie within v.y-span ± margin
) -> List[Tuple["Bande", List["Bande"]]]:
    """
    For each vertical band v in v_bands, return the horizontal bands h that:
      (keeps original X logic exactly)
        - (orientation==horizontal AND abs(h.x0 - v.x0) <= min_x0_diff_px) OR
          (abs(h.x1 - v.x1) <= min_right_extension_px)
      (NEW) and also whose vertical center lies within [v.y0, v.y1] expanded by ±y_center_margin_px.
    """
    assert bde is not None and getattr(bde, "result", None) is not None, \
        "Provide a BandeExtractor with .result populated (call extract() first)."

    horizontals: List["Bande"] = bde.result.get("horizontal", [])
    out: List[Tuple["Bande", List["Bande"]]] = []

    for v in v_bands:
        if getattr(v, "orientation", "") != "vertical":
            continue

        vx0, vx1 = v.x0, v.x1
        vy0, vy1 = (min(v.y0, v.y1), max(v.y0, v.y1))

        # --- original selection (unchanged) ---
        raw_matches = [
            h for h in horizontals
            if (getattr(h, "orientation", "") == "horizontal"
                and abs(h.x0 - vx0) <= min_x0_diff_px)
            or (abs(h.x1 - vx1) <= min_right_extension_px)
        ]

        # --- NEW: Y-center containment filter ---
        matches: List["Bande"] = []
        for h in raw_matches:
            h_cy = (h.y0 + h.y1) * 0.5
            if (vy0 - y_center_margin_px) <= h_cy <= (vy1 + y_center_margin_px):
                matches.append(h)

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
    min_y0_diff_px: int = 10,
    min_bottom_extension_px: int = 10,
    x_edge_align_tol_px: int = 7,   # NEW: vertical cx must be within this of h.x0 or h.x1
) -> List[Tuple["Bande", List["Bande"]]]:
    """
    Inverse linkage: for each HORIZONTAL band h, return VERTICAL bands v that:
      1) satisfy the existing Y logic (same as before), and
      2) have center-x within `x_edge_align_tol_px` of either h.x0 or h.x1.
    """
    assert bde is not None and getattr(bde, "result", None) is not None, \
        "Provide a BandeExtractor with .result populated (call extract() first)."

    verticals: List["Bande"] = bde.result.get("vertical", [])
    out: List[Tuple["Bande", List["Bande"]]] = []

    for h in h_bands:
        if getattr(h, "orientation", "") != "horizontal":
            continue

        hx0, hx1 = h.x0, h.x1
        hy0, hy1 = h.y0, h.y1

        matches: List["Bande"] = []
        for v in verticals:
            if getattr(v, "orientation", "") != "vertical":
                continue

            # --- keep the SAME Y logic as before ---
            y_ok = (abs(v.y0 - hy0) <= min_y0_diff_px) or (abs(v.y1 - hy1) <= min_bottom_extension_px)

            # --- NEW: x-edge gating (pas d'absolu) ---
            v_cx = (v.x0 + v.x1) / 2.0
            left_ok  = v_cx >= (hx0 - x_edge_align_tol_px)
            right_ok = v_cx <= (hx1 + x_edge_align_tol_px)
            x_align_ok = left_ok and right_ok

            if y_ok and x_align_ok:
                matches.append(v)


        out.append((h, matches))

    return out



# ───────────────────────────── UExtractor class ─────────────────────────────

from typing import List, Dict, Any, Tuple, Optional, Set
import logging

# assumes these are imported elsewhere in your module:
# from data_extractor import ExtractionContext
# from band_extractor import BandeExtractor, Bande
# from number_binding_extractor import NumberBindingExtractor
# from your_helpers import (
#     find_vertical_bands_near_number,
#     find_horizontal_bands_near_number,
#     find_horizontal_lines_near_vbands,
#     find_vertical_lines_near_hbands,
# )

class UExtractor:
    """
    One-shot pipeline wrapper that:
      • runs BandeExtractor (V & H)
      • runs NumberBindingExtractor
      • builds target bboxes = only {"binding","number"}
      • PASS-A: V near bbox (always computed first). Blacklist bboxes that produce any V.
      • Also compute H near those PASS-A verticals and blacklist those H.
      • In --v mode: draw/show V near bbox, then H near those V.
      • In --h mode: ignore PASS-A V in drawing; for NON-blacklisted bboxes, find H near bbox
                     (EXCLUDING any H that are blacklisted-from-V), then V near those H.

    extract(...) returns a dict with four sets of pixel rectangles:
      {
        "v_near_bbox": set[(x0,x1,y0,y1)],  # PASS-A verticals (not drawn in --h)
        "h_near_v":    set[(x0,x1,y0,y1)],  # horizontals near PASS-A verticals (only in --v)
        "h_near_bbox": set[(x0,x1,y0,y1)],  # horizontals near NON-blacklisted bboxes (only in --h)
        "v_near_h":    set[(x0,x1,y0,y1)],  # verticals near those horizontals (only in --h)
        "blacklist_bboxes": set[bbox_pdf],  # the bboxes that produced verticals in PASS-A
        "blacklist_h_from_v": set[(x0,x1,y0,y1)],  # H bands near PASS-A V (excluded in --h)
        "targets": [bbox_pdf],              # convenience
      }
    """

    def __init__(
        self,
        *,
        # BandeExtractor params
        thr: int = 60,
        min_thick: int = 1,
        max_thick: int = 18,
        v_min_len: int = 15,
        h_min_len: int = 15,
        bridge: int = 1,
        # Band↔bbox association tolerances
        tol: int = 12,         # for V: vertical proximity to bbox; for H: BELOW-center window
        x_tol: int = 12,       # for V: center-X proximity; for H: ABOVE-center window (and x-edge tol)
        # NumberBindingExtractor params
        pair_radius: float = 120.0,
        x_align: float = 7.0,
        y_align: float = 7.0,
        # Mode flags
        horizontal_mode: bool = False,  # False = --v (default), True = --h
    ):
        self.thr = thr
        self.min_thick = min_thick
        self.max_thick = max_thick
        self.v_min_len = v_min_len
        self.h_min_len = h_min_len
        self.bridge = bridge

        self.tol = tol
        self.x_tol = x_tol

        self.pair_radius = pair_radius
        self.x_align = x_align
        self.y_align = y_align

        self.horizontal_mode = horizontal_mode

        # set at runtime in extract()
        self.ctx: Optional[ExtractionContext] = None
        self.bde: Optional[BandeExtractor] = None
        self.nbe: Optional[NumberBindingExtractor] = None

    @staticmethod
    def _as_rect_px(b: "Bande") -> Tuple[int, int, int, int]:
        return (int(b.x0), int(b.x1), int(b.y0), int(b.y1))

    # ---- NEW: resolve or build BandeExtractor from `shared` ----
    def _resolve_bde(self, ctx: "ExtractionContext", shared: Optional[Dict[str, Any]]) -> "BandeExtractor":
        # Prefer a provided BandeExtractor (keyed by class name); if missing or uncomputed, build/compute.
        if shared and isinstance(shared.get("BandeExtractor"), BandeExtractor):
            bde: BandeExtractor = shared["BandeExtractor"]
            if getattr(bde, "result", None) is None:
                bde.extract(ctx, shared=shared)
            return bde
        # Fallback: create and compute
        bde = BandeExtractor(
            thr=self.thr,
            min_thick=self.min_thick,
            max_thick=self.max_thick,
            v_min_len_px=self.v_min_len,
            v_max_len_px=None,
            h_min_len_px=self.h_min_len,
            h_max_len_px=None,
            bridge_px=self.bridge,
        )
        bde.extract(ctx, shared=(shared or {}))
        return bde

    # ---- NEW: resolve or build NumberBindingExtractor from `shared` ----
    def _resolve_nbe(self, ctx: "ExtractionContext", shared: Optional[Dict[str, Any]]) -> "NumberBindingExtractor":
        if shared and isinstance(shared.get("NumberBindingExtractor"), NumberBindingExtractor):
            nbe: NumberBindingExtractor = shared["NumberBindingExtractor"]
            # If the caller passed a precomputed NBE, assume its result matches the same ctx.
            # If not computed, compute now.
            if getattr(nbe, "result", None) is None:
                nbe.extract(ctx, shared=shared)
            return nbe
        # Fallback: create a new one (with our configured tolerances).
        nbe = NumberBindingExtractor(
            pair_radius_px=self.pair_radius,
            x_tol_px=self.x_align,
            y_tol_px=self.y_align,
        )
        # We will compute when fetching targets (so that we can reuse items if needed)
        return nbe

    def _targets(self, ctx: "ExtractionContext", nbe: "NumberBindingExtractor", shared: Optional[Dict[str, Any]]) -> List[Tuple[float, float, float, float]]:
        # Use existing result if available; otherwise compute now.
        if getattr(nbe, "result", None) is None:
            items = nbe.extract(ctx, shared=(shared or {}))
        else:
            items = nbe.result
        targets = [it["bbox"] for it in items if it["type"] in {"binding", "number"}]
        logging.info("Targets (bindings+numbers): %d", len(targets))
        return targets

    def extract(self, ctx: "ExtractionContext", shared: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the full pipeline in either V or H mode and return sets of pixel rectangles."""
        self.ctx = ctx
        self.bde = self._resolve_bde(ctx, shared)
        self.nbe = self._resolve_nbe(ctx, shared)

        # 1) Targets (bindings + numbers)
        target_bboxes = self._targets(ctx, self.nbe, shared)

        # 2) PASS-A: compute V near bbox for ALL targets (always first)
        v_near_per_target: List[List["Bande"]] = []
        blacklist: Set[Tuple[float, float, float, float]] = set()

        for bbox_pdf in target_bboxes:
            found = find_vertical_bands_near_number(
                ctx,
                bbox_pdf,
                tol_px=self.tol,
                x_tol_px=self.x_tol,
                bde=self.bde,
            )
            v_near = found["near_top"] + found["near_bottom"]
            v_near_per_target.append(v_near)
            if v_near:
                blacklist.add(bbox_pdf)

        logging.info("Blacklist size (bboxes that produced verticals in PASS A): %d", len(blacklist))

        # Also compute horizontals near those PASS-A verticals and blacklist them
        pass_a_v_flat: List["Bande"] = [v for lst in v_near_per_target for v in lst]
        blacklist_h_from_v: Set[Tuple[int, int, int, int]] = set()
        if pass_a_v_flat:
            v_to_h_for_blacklist = find_horizontal_lines_near_vbands(
                pass_a_v_flat,
                bde=self.bde,
                min_x0_diff_px=5,
                min_right_extension_px=5,
            )
            for _, h_list in v_to_h_for_blacklist:
                for h in h_list:
                    blacklist_h_from_v.add(self._as_rect_px(h))
        logging.info("Blacklisted H (near PASS-A V): %d", len(blacklist_h_from_v))

        # 3) Accumulate results
        v_near_bbox: Set[Tuple[int, int, int, int]] = set()
        h_near_v:    Set[Tuple[int, int, int, int]] = set()
        h_near_bbox: Set[Tuple[int, int, int, int]] = set()
        v_near_h:    Set[Tuple[int, int, int, int]] = set()

        if not self.horizontal_mode:
            # --v: draw PASS-A verticals and then H near those V
            for v_list in v_near_per_target:
                for v in v_list:
                    v_near_bbox.add(self._as_rect_px(v))
                # H near these V (for display in --v)
                v_to_h = find_horizontal_lines_near_vbands(
                    v_list,
                    bde=self.bde,
                    min_x0_diff_px=5,
                    min_right_extension_px=5,
                )
                for _, h_list in v_to_h:
                    for h in h_list:
                        h_near_v.add(self._as_rect_px(h))
        else:
            # --h: only use NON-blacklisted bboxes, and EXCLUDE any H that are blacklisted-from-V
            used = 0
            for bbox_pdf in target_bboxes:
                if bbox_pdf in blacklist:
                    continue

                # H near bbox (above<=x_tol, below<=tol, plus x-edge/spanning)
                h_found = find_horizontal_bands_near_number(
                    ctx,
                    bbox_pdf,
                    tol_px=self.tol,            # BELOW-center
                    y_tol_px=self.x_tol,        # ABOVE-center
                    x_edge_tol_px=self.x_tol,   # x-edge proximity
                    bde=self.bde,
                )
                # keep only those not in blacklist_h_from_v
                h_near = [
                    h for h in (h_found["near_above"] + h_found["near_below"])
                    if self._as_rect_px(h) not in blacklist_h_from_v
                ]
                for h in h_near:
                    h_near_bbox.add(self._as_rect_px(h))

                # V near those remaining H
                h_to_v = find_vertical_lines_near_hbands(
                    h_near,
                    bde=self.bde,
                )
                for _, v_list in h_to_v:
                    for v in v_list:
                        v_near_h.add(self._as_rect_px(v))
                used += 1

            logging.info(
                "H detection used %d non-blacklisted bboxes; skipped %d. "
                "Filtered out %d H via blacklist-from-V.",
                used, len(blacklist), len(blacklist_h_from_v)
            )

        return {
            "v_near_bbox": v_near_bbox,
            "h_near_v": h_near_v,
            "h_near_bbox": h_near_bbox,
            "v_near_h": v_near_h,
            "blacklist_bboxes": blacklist,
            "blacklist_h_from_v": blacklist_h_from_v,
            "targets": target_bboxes,
        }


# ───────────────────────────────────── Main ─────────────────────────────────────

def main():
    import argparse
    from pathlib import Path
    import matplotlib

    # Headless backend if needed
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

        # Configure pipeline
        uex = UExtractor(
            horizontal_mode=bool(args.h) and not bool(args.v),
        )

        results = uex.extract(ctx)

        v_near_bbox = results["v_near_bbox"]
        h_near_v    = results["h_near_v"]
        h_near_bbox = results["h_near_bbox"]
        v_near_h    = results["v_near_h"]
        horizontal_mode = uex.horizontal_mode

        logging.info("Unique vertical (PASS-A): %d | H near V: %d | H near bbox: %d | V near H: %d",
                     len(v_near_bbox), len(h_near_v), len(h_near_bbox), len(v_near_h))

        # 4) Draw overlays
        base = ctx.pil().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        if horizontal_mode:
            # Draw ONLY: H near bbox (blue) + V near those H (red)
            for (x0, x1, y0, y1) in v_near_h:
                l, r = sorted([x0, x1]); t, b = sorted([y0, y1])
                draw.rectangle([l, t, r, b], outline=(255, 0, 0, 255), width=3)         # verticals
            for (x0, x1, y0, y1) in h_near_bbox:
                l, r = sorted([x0, x1]); t, b = sorted([y0, y1])
                draw.rectangle([l, t, r, b], outline=(0, 102, 255, 255), width=3)       # horizontals
        else:
            # Draw: V near bbox (red) + H near those V (blue)
            for (x0, x1, y0, y1) in v_near_bbox:
                l, r = sorted([x0, x1]); t, b = sorted([y0, y1])
                draw.rectangle([l, t, r, b], outline=(255, 0, 0, 255), width=3)         # verticals
            for (x0, x1, y0, y1) in h_near_v:
                l, r = sorted([x0, x1]); t, b = sorted([y0, y1])
                draw.rectangle([l, t, r, b], outline=(0, 102, 255, 255), width=3)       # horizontals

        composed = Image.alpha_composite(base, overlay).convert("RGB")

        # 5) Show / save
        import matplotlib.pyplot as plt
        raw_rgb = ctx.pil().convert("RGB")
        plt.figure(figsize=(12, 7))
        plt.subplot(1, 2, 1); plt.title(f"RAW — {pdf_path.name} p{args.page+1}")
        plt.imshow(raw_rgb); plt.axis("off")
        mode_title = "H mode (draw H near bbox & V near H; PASS-A verticals only for blacklist)" \
                     if horizontal_mode else \
                     "V mode (draw V near bbox & H near V)"
        plt.subplot(1, 2, 2); plt.title(mode_title)
        plt.imshow(composed); plt.axis("off")
        plt.tight_layout(); plt.show()

        if args.out:
            out_path = Path(args.out).expanduser().resolve()
            composed.save(str(out_path))
            logging.info("Saved overlay → %s", out_path)


if __name__ == "__main__":
    main()
