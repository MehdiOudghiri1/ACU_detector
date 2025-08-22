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

from .data_extractor import ExtractionContext
from .band_extractor import BandeExtractor, Bande
from .number_binding_extractor import NumberBindingExtractor





# ─────────────────────────── Helpers: band association ───────────────────────────
def find_vertical_bands_near_number(
    context: ExtractionContext,
    bbox_pts: Tuple[float, float, float, float],
    *,
    v_y_tol_px: int = 17,            # vertical proximity to bbox top/bottom
    v_cx_tol_px: int = 12,           # horizontal proximity to bbox center
    band_extractor: "BandeExtractor",
) -> Dict[str, List["Bande"]]:
    """
    Given a number bbox in PDF coords (x0, top, x1, bottom), return vertical bands
    whose y1 is near the bbox TOP (near_top) or whose y0 is near the bbox BOTTOM
    (near_bottom), AND whose horizontal center is near the bbox center (±v_cx_tol_px).

    IMPORTANT:
      pdfplumber words use TOP-LEFT origin for 'top'/'bottom'. Do NOT invert Y.
      Use context.pt_to_px_y_top(...) to convert those to pixels.
    """
    assert band_extractor is not None and getattr(band_extractor, "result", None) is not None, \
        "Provide a BandeExtractor with .result"
    vertical_bands_list: List["Bande"] = band_extractor.result.get("vertical", [])

    # PDF (points) → pixel (top-left origin) conversions
    x0_pt, top_pt, x1_pt, bottom_pt = bbox_pts
    top_px = context.pt_to_px_y_top(top_pt)
    bottom_px = context.pt_to_px_y_top(bottom_pt)
    bbox_center_x_px = context.pt_to_px_x((x0_pt + x1_pt) / 2.0)  # bbox horizontal center in pixels

    def is_x_center_close(v_band: "Bande") -> bool:
        v_center_x = (v_band.x0 + v_band.x1) / 2.0  # band horizontal center (pixels)
        return abs(v_center_x - bbox_center_x_px) <= v_cx_tol_px

    near_top_list: List["Bande"] = [
        v for v in vertical_bands_list
        if abs(v.y1 - top_px) <= v_y_tol_px and is_x_center_close(v)
    ]
    near_bottom_list: List["Bande"] = [
        v for v in vertical_bands_list
        if abs(v.y0 - bottom_px) <= v_y_tol_px and is_x_center_close(v)
    ]

    return {
        "near_top": near_top_list,
        "near_bottom": near_bottom_list,
        "all_vertical": vertical_bands_list,  # for debugging if needed
    }


def find_horizontal_lines_near_vbands(
    vertical_bands: List["Bande"],
    *,
    band_extractor: "BandeExtractor",
    h_start_x0_tol_px: int = 20,
    h_right_ext_tol_px: int = 20,
    h_center_y_margin_px: int = 10,  # horizontal center must lie within v.y-span ± margin
) -> List[Tuple["Bande", List["Bande"]]]:
    """
    For each vertical band v in vertical_bands, return the horizontal bands h that:
      (keeps original X logic exactly)
        - (orientation==horizontal AND abs(h.x0 - v.x0) <= h_start_x0_tol_px) OR
          (abs(h.x1 - v.x1) <= h_right_ext_tol_px)
      and also whose vertical center lies within [v.y0, v.y1] expanded by ±h_center_y_margin_px.
    """
    assert band_extractor is not None and getattr(band_extractor, "result", None) is not None, \
        "Provide a BandeExtractor with .result populated (call extract() first)."

    horizontal_bands_list: List["Bande"] = band_extractor.result.get("horizontal", [])
    pairs_out: List[Tuple["Bande", List["Bande"]]] = []

    for v_band in vertical_bands:
        if getattr(v_band, "orientation", "") != "vertical":
            continue

        v_x0, v_x1 = v_band.x0, v_band.x1
        v_y0, v_y1 = (min(v_band.y0, v_band.y1), max(v_band.y0, v_band.y1))

        # --- original selection (unchanged precedence) ---
        prefiltered_matches = [
            h for h in horizontal_bands_list
            if (getattr(h, "orientation", "") == "horizontal"
                and abs(h.x0 - v_x0) <= h_start_x0_tol_px)
            or (abs(h.x1 - v_x1) <= h_right_ext_tol_px)
        ]

        # --- Y-center containment post-filter ---
        final_matches: List["Bande"] = []
        for h_band in prefiltered_matches:
            h_center_y = (h_band.y0 + h_band.y1) * 0.5
            if (v_y0 - h_center_y_margin_px) <= h_center_y <= (v_y1 + h_center_y_margin_px):
                final_matches.append(h_band)

        pairs_out.append((v_band, final_matches))

    return pairs_out


# ─────────────────────────── Band association helpers ───────────────────────────

def find_horizontal_bands_near_number(
    context: ExtractionContext,
    bbox_pts: Tuple[float, float, float, float],
    *,
    h_below_tol_px: int = 5,          # MAX distance BELOW bbox center (pixels)
    h_above_tol_px: int = 2,          # MAX distance ABOVE bbox center (pixels)
    h_x_edge_tol_px: int = 27,        # tolerance for x-edge proximity (pixels)
    band_extractor: "BandeExtractor",
) -> Dict[str, List["Bande"]]:
    """
    Accept H bands that satisfy BOTH:
      1) Vertical gating (relative to bbox vertical center, pixel top-origin):
         - near_above: b_cy <= cy_px and (cy_px - b_cy) <= h_above_tol_px
         - near_below: b_cy >= cy_px and (b_cy - cy_px) <= h_below_tol_px
      2) Horizontal gating (any ONE of):
         - |h.x0 - bbox_x1| <= h_x_edge_tol_px   (band starts near RIGHT edge)
         - |h.x1 - bbox_x0| <= h_x_edge_tol_px   (band ends   near LEFT  edge)
         - h.x1 > bbox_x1 AND h.x0 < bbox_x0     (band spans across bbox)
    """
    assert band_extractor is not None and getattr(band_extractor, "result", None) is not None, \
        "Provide a BandeExtractor with .result"
    horizontal_bands_all: List["Bande"] = band_extractor.result.get("horizontal", [])

    x0_pt, top_pt, x1_pt, bottom_pt = bbox_pts
    # pixel (top-origin for Y)
    bbox_left_px = context.pt_to_px_x(x0_pt)
    bbox_right_px = context.pt_to_px_x(x1_pt)
    bbox_center_y_px = context.pt_to_px_y_top((top_pt + bottom_pt) / 2.0)

    near_above_list: List["Bande"] = []
    near_below_list: List["Bande"] = []

    for h_band in horizontal_bands_all:
        if getattr(h_band, "orientation", "") != "horizontal":
            continue

        # Vertical gating wrt bbox center
        h_center_y = (h_band.y0 + h_band.y1) / 2.0
        delta_y = h_center_y - bbox_center_y_px  # >0 below, <0 above

        # Horizontal gating: edge-touch or spanning across bbox
        starts_near_right_edge = abs(h_band.x0 - bbox_right_px) <= h_x_edge_tol_px
        ends_near_left_edge = abs(h_band.x1 - bbox_left_px) <= h_x_edge_tol_px
        spans_horizontally_across = (h_band.x1 > bbox_right_px) and (h_band.x0 < bbox_left_px)
        horizontal_gate_ok = starts_near_right_edge or ends_near_left_edge or spans_horizontally_across
        if not horizontal_gate_ok:
            continue

        # Asymmetric vertical windows (ordering preserved)
        if -h_above_tol_px <= delta_y <= 0:
            near_above_list.append(h_band)
        elif 0 <= delta_y <= h_below_tol_px:
            near_below_list.append(h_band)

    return {
        "near_above": near_above_list,
        "near_below": near_below_list,
        "all_horizontal": horizontal_bands_all,
    }


def find_vertical_lines_near_hbands(
    horizontal_bands: List["Bande"],
    *,
    band_extractor: "BandeExtractor",
    v_top_y0_tol_px: int = 10,
    v_bottom_y1_tol_px: int = 10,
    v_center_x_edge_tol_px: int = 7,   # vertical cx must be within this of h.x0 or h.x1 window
) -> List[Tuple["Bande", List["Bande"]]]:
    """
    Inverse linkage: for each HORIZONTAL band h, return VERTICAL bands v that:
      1) satisfy the existing Y logic (same as before), and
      2) have center-x within the [h.x0 - tol, h.x1 + tol] window.
    """
    assert band_extractor is not None and getattr(band_extractor, "result", None) is not None, \
        "Provide a BandeExtractor with .result populated (call extract() first)."

    vertical_bands_all: List["Bande"] = band_extractor.result.get("vertical", [])
    pairs_out: List[Tuple["Bande", List["Bande"]]] = []

    for h_band in horizontal_bands:
        if getattr(h_band, "orientation", "") != "horizontal":
            continue

        h_x0, h_x1 = h_band.x0, h_band.x1
        h_y0, h_y1 = h_band.y0, h_band.y1

        matched_verticals: List["Bande"] = []
        for v_band in vertical_bands_all:
            if getattr(v_band, "orientation", "") != "vertical":
                continue

            # --- keep the SAME Y logic as before ---
            y_ok = (abs(v_band.y0 - h_y0) <= v_top_y0_tol_px) or (abs(v_band.y1 - h_y1) <= v_bottom_y1_tol_px)

            # --- x-edge gating (no absolute across range) ---
            v_center_x = (v_band.x0 + v_band.x1) / 2.0
            left_ok = v_center_x >= (h_x0 - v_center_x_edge_tol_px)
            right_ok = v_center_x <= (h_x1 + v_center_x_edge_tol_px)
            x_in_window_ok = left_ok and right_ok

            if y_ok and x_in_window_ok:
                matched_verticals.append(v_band)

        pairs_out.append((h_band, matched_verticals))

    return pairs_out


# ───────────────────────────── UExtractor class ─────────────────────────────

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
        max_thick: int = 19,
        v_min_len: int = 22,
        h_min_len: int = 21,
        bridge: int = 1,

        # ---- exact mirrors of helper-function params ----
        # find_vertical_bands_near_number
        v_y_tol_px: int = 14,
        v_cx_tol_px: int = 10,

        # find_horizontal_lines_near_vbands
        h_start_x0_tol_px: int = 10,
        h_right_ext_tol_px: int = 10,
        h_center_y_margin_px: int = 10,

        # find_horizontal_bands_near_number
        h_below_tol_px: int = 10,
        h_above_tol_px: int = 10,
        h_x_edge_tol_px: int = 27,
        min_number_for_h: float = 10.0,

        # find_vertical_lines_near_hbands
        v_top_y0_tol_px: int = 10,
        v_bottom_y1_tol_px: int = 10,
        v_center_x_edge_tol_px: int = 10,

        # NumberBindingExtractor params
        pair_radius: float = 120.0,
        x_align: float = 11.0,
        y_align: float = 11.0,
        

        # Mode flags
        horizontal_mode: bool = False,  # False = --v (default), True = --h
    ):
        # BandeExtractor config
        self.thr = thr
        self.min_thick = min_thick
        self.max_thick = max_thick
        self.v_min_len = v_min_len
        self.h_min_len = h_min_len
        self.bridge = bridge

        # Mirrors of helper-function kwargs (identical names)
        self.v_y_tol_px = v_y_tol_px
        self.v_cx_tol_px = v_cx_tol_px

        self.h_start_x0_tol_px = h_start_x0_tol_px
        self.h_right_ext_tol_px = h_right_ext_tol_px
        self.h_center_y_margin_px = h_center_y_margin_px

        self.h_below_tol_px = h_below_tol_px
        self.h_above_tol_px = h_above_tol_px
        self.h_x_edge_tol_px = h_x_edge_tol_px
        self.min_number_for_h = min_number_for_h,


        self.v_top_y0_tol_px = v_top_y0_tol_px
        self.v_bottom_y1_tol_px = v_bottom_y1_tol_px
        self.v_center_x_edge_tol_px = v_center_x_edge_tol_px

        # NumberBindingExtractor config
        self.pair_radius = pair_radius
        self.x_align = x_align
        self.y_align = y_align

        # Mode
        self.horizontal_mode = horizontal_mode

        # set at runtime in extract()
        self.context: Optional[ExtractionContext] = None
        self.band_extractor: Optional[BandeExtractor] = None  # name mirrors helper param
        self.number_binding_extractor: Optional[NumberBindingExtractor] = None

    @staticmethod
    def _as_rect_px(band: "Bande") -> Tuple[int, int, int, int]:
        return (int(band.x0), int(band.x1), int(band.y0), int(band.y1))

    # resolve or build BandeExtractor
    def _resolve_bde(self, context: "ExtractionContext", shared_pool: Optional[Dict[str, Any]]) -> "BandeExtractor":
        if shared_pool and isinstance(shared_pool.get("BandeExtractor"), BandeExtractor):
            existing_bde: BandeExtractor = shared_pool["BandeExtractor"]
            if getattr(existing_bde, "result", None) is None:
                existing_bde.extract(context, shared=shared_pool)
            return existing_bde
        # Fallback: create and compute
        new_bde = BandeExtractor(
            thr=self.thr,
            min_thick=self.min_thick,
            max_thick=self.max_thick,
            v_min_len_px=self.v_min_len,
            v_max_len_px=None,
            h_min_len_px=self.h_min_len,
            h_max_len_px=None,
            bridge_px=self.bridge,
        )
        new_bde.extract(context, shared=(shared_pool or {}))
        return new_bde

    # resolve or build NumberBindingExtractor
    def _resolve_nbe(self, context: "ExtractionContext", shared_pool: Optional[Dict[str, Any]]) -> "NumberBindingExtractor":
        if shared_pool and isinstance(shared_pool.get("NumberBindingExtractor"), NumberBindingExtractor):
            existing_nbe: NumberBindingExtractor = shared_pool["NumberBindingExtractor"]
            if getattr(existing_nbe, "result", None) is None:
                existing_nbe.extract(context, shared=shared_pool)
            return existing_nbe
        return NumberBindingExtractor(
            pair_radius_px=self.pair_radius,
            x_tol_px=self.x_align,
            y_tol_px=self.y_align,
        )

    def _targets(self, context: "ExtractionContext", nbe_obj: "NumberBindingExtractor",
                 shared_pool: Optional[Dict[str, Any]]) -> List[Tuple[float, float, float, float]]:
        if getattr(nbe_obj, "result", None) is None:
            items_found = nbe_obj.extract(context, shared=(shared_pool or {}))
        else:
            items_found = nbe_obj.result
        bbox_targets = [it["bbox"] for it in items_found if it["type"] in {"binding", "number"}]
        logging.info("Targets (bindings+numbers): %d", len(bbox_targets))
        return bbox_targets

    def extract(self, context: "ExtractionContext", shared: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the full pipeline in either V or H mode and return sets of pixel rectangles."""
        import re

        # ---------------- helpers ----------------

        def _coerce_float(v: Any, default: float) -> float:
            if isinstance(v, (list, tuple)):
                v = v[0] if v else default
            try:
                return float(v)
            except Exception:
                return default

        def _parse_num_with_suffix(s: str) -> Optional[float]:
            """
            Parse a numeric token with optional k/m/b suffix, e.g. '25k', '12.5kV', '9.5M'.
            """
            if not s:
                return None
            m = re.search(r'[-+]?\d+(?:[.,]\d+)?\s*([kKmMbB])?', s)
            if not m:
                return None
            num_part = re.search(r'[-+]?\d+(?:[.,]\d+)?', m.group(0))
            if not num_part:
                return None
            val = float(num_part.group(0).replace(',', '.'))
            suf = m.group(1)
            if suf:
                if suf in ('k', 'K'):
                    val *= 1_000.0
                elif suf in ('m', 'M'):
                    val *= 1_000_000.0
                elif suf in ('b', 'B'):
                    val *= 1_000_000_000.0
            return val

        def _bbox_contains(outer, inner, pad: float = 0.0) -> bool:
            ox0, ot, ox1, ob = outer
            ix0, it, ix1, ib = inner
            return (ix0 >= ox0 - pad and ix1 <= ox1 + pad and it >= ot - pad and ib <= ob + pad)

        # ------------- setup (unchanged) -------------

        self.context = context
        self.band_extractor = self._resolve_bde(context, shared)
        self.number_binding_extractor = self._resolve_nbe(context, shared)

        # threshold for NUMBER bboxes used in H-mode
        min_num_thresh = _coerce_float(getattr(self, "min_number_for_h", 10.0), 10.0)

        # Get NBE items once
        if getattr(self.number_binding_extractor, "result", None) is None:
            nbe_items: List[Dict[str, Any]] = self.number_binding_extractor.extract(context, shared=(shared or {}))
        else:
            nbe_items = self.number_binding_extractor.result

        # Build a quick value map for NUMBER bboxes by reading words inside each number bbox
        # This is the key fix: your NumberBindingExtractor doesn't store the text for numbers.
        words_all = context.words()  # list of dicts with x0, top, x1, bottom, text
        number_value_by_bbox: Dict[Tuple[float, float, float, float], Optional[float]] = {}

        for it in nbe_items:
            if it.get("type") != "number":
                continue
            nb = it["bbox"]
            # gather tokens whose bbox is fully inside the number bbox (strict but robust)
            tokens_inside = []
            for w in words_all:
                wb = (w["x0"], w["top"], w["x1"], w["bottom"])
                if _bbox_contains(nb, wb, pad=0.0):
                    tokens_inside.append(str(w.get("text", "")))
            # choose the best token to parse (longest digit-bearing token first)
            parsed_val: Optional[float] = None
            if tokens_inside:
                # prefer tokens that contain a digit
                tokens_inside.sort(key=lambda t: (any(c.isdigit() for c in t), len(t)), reverse=True)
                for tok in tokens_inside:
                    parsed_val = _parse_num_with_suffix(tok)
                    if parsed_val is not None:
                        break
            number_value_by_bbox[nb] = parsed_val  # may be None if nothing parsed

        # 1) Targets (bindings + numbers) — for PASS-A verticals
        target_bboxes_pts = [it["bbox"] for it in nbe_items if it.get("type") in {"binding", "number"}]
        logging.info("Targets (bindings+numbers): %d", len(target_bboxes_pts))

        # 2) PASS-A: compute V near bbox for ALL targets (always first)
        verticals_per_target: List[List["Bande"]] = []
        blacklist_bboxes: Set[Tuple[float, float, float, float]] = set()

        for bbox_pts in target_bboxes_pts:
            v_found_map = find_vertical_bands_near_number(
                context,
                bbox_pts,
                v_y_tol_px=self.v_y_tol_px,
                v_cx_tol_px=self.v_cx_tol_px,
                band_extractor=self.band_extractor,
            )
            v_near_list = v_found_map["near_top"] + v_found_map["near_bottom"]
            verticals_per_target.append(v_near_list)
            if v_near_list:
                blacklist_bboxes.add(bbox_pts)

        logging.info("Blacklist size (bboxes that produced verticals in PASS A): %d", len(blacklist_bboxes))

        # Also compute horizontals near those PASS-A verticals and blacklist them
        pass_a_verticals_flat: List["Bande"] = [v for sub in verticals_per_target for v in sub]
        blacklist_h_from_v: Set[Tuple[int, int, int, int]] = set()
        if pass_a_verticals_flat:
            v_to_h_for_blacklist = find_horizontal_lines_near_vbands(
                pass_a_verticals_flat,
                band_extractor=self.band_extractor,
                h_start_x0_tol_px=self.h_start_x0_tol_px,
                h_right_ext_tol_px=self.h_right_ext_tol_px,
                h_center_y_margin_px=self.h_center_y_margin_px,
            )
            for _, h_list in v_to_h_for_blacklist:
                for h in h_list:
                    blacklist_h_from_v.add(self._as_rect_px(h))
        logging.info("Blacklisted H (near PASS-A V): %d", len(blacklist_h_from_v))

        # 3) Accumulate results
        out_v_near_bbox: Set[Tuple[int, int, int, int]] = set()
        out_h_near_v:    Set[Tuple[int, int, int, int]] = set()
        out_h_near_bbox: Set[Tuple[int, int, int, int]] = set()
        out_v_near_h:    Set[Tuple[int, int, int, int]] = set()

        if not self.horizontal_mode:
            # --v: draw PASS-A verticals and then H near those V
            for v_list in verticals_per_target:
                for v_band in v_list:
                    out_v_near_bbox.add(self._as_rect_px(v_band))
                v_to_h_pairs = find_horizontal_lines_near_vbands(
                    v_list,
                    band_extractor=self.band_extractor,
                    h_start_x0_tol_px=self.h_start_x0_tol_px,
                    h_right_ext_tol_px=self.h_right_ext_tol_px,
                    h_center_y_margin_px=self.h_center_y_margin_px,
                )
                for _, h_list in v_to_h_pairs:
                    for h_band in h_list:
                        out_h_near_v.add(self._as_rect_px(h_band))
        else:
            # --h: only use NON-blacklisted NUMBER bboxes with parsed value >= threshold
            num_total = sum(1 for it in nbe_items if it.get("type") == "number")
            num_non_blacklisted = 0
            num_passing_threshold = 0
            used_count = 0

            for it in nbe_items:
                if it.get("type") != "number":
                    continue
                bbox_pts = it["bbox"]
                if bbox_pts in blacklist_bboxes:
                    continue
                num_non_blacklisted += 1

                val = number_value_by_bbox.get(bbox_pts)
                if val is None or val < min_num_thresh:
                    continue
                num_passing_threshold += 1

                h_found_map = find_horizontal_bands_near_number(
                    context,
                    bbox_pts,
                    h_below_tol_px=self.h_below_tol_px,     # BELOW-center
                    h_above_tol_px=self.h_above_tol_px,     # ABOVE-center
                    h_x_edge_tol_px=self.h_x_edge_tol_px,   # x-edge proximity
                    band_extractor=self.band_extractor,
                )
                h_near_filtered = [
                    h for h in (h_found_map["near_above"] + h_found_map["near_below"])
                    if self._as_rect_px(h) not in blacklist_h_from_v
                ]
                for h_band in h_near_filtered:
                    out_h_near_bbox.add(self._as_rect_px(h_band))

                h_to_v_pairs = find_vertical_lines_near_hbands(
                    h_near_filtered,
                    band_extractor=self.band_extractor,
                    v_top_y0_tol_px=self.v_top_y0_tol_px,
                    v_bottom_y1_tol_px=self.v_bottom_y1_tol_px,
                    v_center_x_edge_tol_px=self.v_center_x_edge_tol_px,
                )
                for _, v_list in h_to_v_pairs:
                    for v_band in v_list:
                        out_v_near_h.add(self._as_rect_px(v_band))
                used_count += 1

            logging.info(
                "Numbers: total=%d, non-blacklisted=%d, >= threshold(%.2f)=%d",
                num_total, num_non_blacklisted, min_num_thresh, num_passing_threshold
            )
            logging.info(
                "H detection used %d non-blacklisted NUMBER bboxes (>= %.2f). "
                "Filtered out %d H via blacklist-from-V.",
                used_count, min_num_thresh, len(blacklist_h_from_v)
            )

        return {
            "v_near_bbox": out_v_near_bbox,
            "h_near_v": out_h_near_v,
            "h_near_bbox": out_h_near_bbox,
            "v_near_h": out_v_near_h,
            "blacklist_bboxes": blacklist_bboxes,
            "blacklist_h_from_v": blacklist_h_from_v,
            "targets": target_bboxes_pts,
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

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--v", action="store_true", help="Vertical-first linkage to H (default)")
    mode_group.add_argument("--h", action="store_true", help="Horizontal linkage uses ONLY bboxes that yielded NO verticals")

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
        context = ExtractionContext(page, dpi=args.dpi)

        # Configure pipeline (override any UExtractor attributes at instantiation if desired)
        u_extractor = UExtractor(
            horizontal_mode=bool(args.h) and not bool(args.v),
        )

        results_map = u_extractor.extract(context)

        out_v_near_bbox = results_map["v_near_bbox"]
        out_h_near_v    = results_map["h_near_v"]
        out_h_near_bbox = results_map["h_near_bbox"]
        out_v_near_h    = results_map["v_near_h"]
        is_horizontal_mode = u_extractor.horizontal_mode

        logging.info("Unique vertical (PASS-A): %d | H near V: %d | H near bbox: %d | V near H: %d",
                     len(out_v_near_bbox), len(out_h_near_v), len(out_h_near_bbox), len(out_v_near_h))

        # 4) Draw overlays
        base_img = context.pil().convert("RGBA")
        overlay_img = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_img, "RGBA")

        if is_horizontal_mode:
            # Draw ONLY: H near bbox (blue) + V near those H (red)
            for (x0, x1, y0, y1) in out_v_near_h:
                l, r = sorted([x0, x1]); t, b = sorted([y0, y1])
                draw.rectangle([l, t, r, b], outline=(255, 0, 0, 255), width=3)         # verticals
            for (x0, x1, y0, y1) in out_h_near_bbox:
                l, r = sorted([x0, x1]); t, b = sorted([y0, y1])
                draw.rectangle([l, t, r, b], outline=(0, 102, 255, 255), width=3)       # horizontals
        else:
            # Draw: V near bbox (red) + H near those V (blue)
            for (x0, x1, y0, y1) in out_v_near_bbox:
                l, r = sorted([x0, x1]); t, b = sorted([y0, y1])
                draw.rectangle([l, t, r, b], outline=(255, 0, 0, 255), width=3)         # verticals
            for (x0, x1, y0, y1) in out_h_near_v:
                l, r = sorted([x0, x1]); t, b = sorted([y0, y1])
                draw.rectangle([l, t, r, b], outline=(0, 102, 255, 255), width=3)       # horizontals

        composed_img = Image.alpha_composite(base_img, overlay_img).convert("RGB")

        # 5) Show / save
        import matplotlib.pyplot as plt
        raw_rgb = context.pil().convert("RGB")
        plt.figure(figsize=(12, 7))
        plt.subplot(1, 2, 1); plt.title(f"RAW — {pdf_path.name} p{args.page+1}")
        plt.imshow(raw_rgb); plt.axis("off")
        mode_title = "H mode (draw H near bbox & V near H; PASS-A verticals only for blacklist)" \
                     if is_horizontal_mode else \
                     "V mode (draw V near bbox & H near V)"
        plt.subplot(1, 2, 2); plt.title(mode_title)
        plt.imshow(composed_img); plt.axis("off")
        plt.tight_layout(); plt.show()

        if args.out:
            out_path = Path(args.out).expanduser().resolve()
            composed_img.save(str(out_path))
            logging.info("Saved overlay → %s", out_path)


import re

def _parse_number_value(it) -> Optional[float]:
    # Works with either it["text"] or it["word"]["text"] or it["value"]
    raw = it.get("value")
    if raw is None:
        raw = it.get("text")
    if raw is None and isinstance(it.get("word"), dict):
        raw = it["word"].get("text")

    if raw is None:
        return None

    s = re.sub(r"[^0-9.,\-]", "", str(raw)).replace(",", ".")
    try:
        return float(s)
    except:
        return None


if __name__ == "__main__":
    main()