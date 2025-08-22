from typing import Tuple, List, Optional, Dict, Any, Set
import math

# ============================ Basic text helpers ============================

def _letters(s: str) -> str:
    """Keep only A–Z letters, uppercase."""
    return "".join(ch for ch in s.upper().strip() if ch.isalpha())

def is_opn_like(txt: str) -> bool:
    """Opn / npO (case-insensitive)."""
    if not txt: return False
    t = txt.strip().lower()
    return t in {"opn", "npo"}

def is_od_like(txt: str) -> bool:
    """O.D.-like tokens regardless of dots/order (OD / DO / O.D. / .D.O / etc.)."""
    if not txt: return False
    t = _letters(txt)
    return set(t) == {"O", "D"}

def is_base_like(txt: str) -> bool:
    """'BASE' or reversed 'ESAB' (case-insensitive)."""
    if not txt: return False
    t = _letters(txt)
    return t == "BASE" or t == "ESAB"

# --- add this helper next to is_base_like() ---
def is_inlet_like(txt: str) -> bool:
    """'INLET' or reversed 'TELNI' (case-insensitive)."""
    if not txt: 
        return False
    t = _letters(txt)
    return t == "INLET" or t == "TELNI"


def is_target_word(txt: str) -> bool:
    """We consider Opn/npO, any O.D.-like, and BASE/ESAB."""
    return (
        is_opn_like(txt)
        or is_od_like(txt) 
        or is_base_like(txt)
        or is_inlet_like(txt)
    )


# ============================ Geometry helpers ==============================

def bbox_union_pdf(b1: Tuple[float, float, float, float],
                   b2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Union of two pdfplumber-style bboxes (x0, top, x1, bottom)."""
    x0 = min(b1[0], b2[0])
    x1 = max(b1[2], b2[2])
    top = min(b1[1], b2[1])
    bottom = max(b1[3], b2[3])
    return (x0, top, x1, bottom)

def bbox_center_px(ctx, box_pdf: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Center of a pdfplumber bbox, returned in PIXELS (top-left origin)."""
    x0, top, x1, bottom = box_pdf
    cx_pt = (x0 + x1) / 2.0
    cy_pt = (top + bottom) / 2.0
    cx_px = ctx.pt_to_px_x(cx_pt)
    cy_px = ctx.pt_to_px_y_top(cy_pt)  # top-origin conversion (no flip)
    return (cx_px, cy_px)


# ===================== Orientation + number pairing =========================

def classify_word_orientation(bbox_pdf: Tuple[float, float, float, float]) -> str:
    """
    Classify a word bbox as 'vertical' or 'horizontal'.
    Input bbox is (x0, top, x1, bottom) in pdfplumber coords (TOP-LEFT origin).
    """
    x0, top, x1, bottom = bbox_pdf
    w = abs(x1 - x0)
    h = abs(bottom - top)
    return "vertical" if h > w else "horizontal"


def closest_number_for_vertical_word(word_bbox, number_bboxes, x_tol: float = 7.0):
    """
    For a vertical word: find the closest number below it,
    but only if horizontally aligned within x_tol (same units as bboxes).
    """
    wx0, wtop, wx1, wbottom = word_bbox
    wcx = (wx0 + wx1) / 2.0  # center x of word

    best = None
    best_d = float("inf")
    for nb in number_bboxes:
        nx0, ntop, nx1, nbottom = nb
        ncx = (nx0 + nx1) / 2.0  # center x of number

        if abs(ncx - wcx) > x_tol:
            continue  # too far horizontally

        d = ntop - wbottom  # vertical gap (>=0 if number is below word)
        if d >= 0 and d < best_d:
            best_d = d
            best = nb

    return (best, best_d) if best is not None else None


def closest_number_for_horizontal_word(word_bbox, number_bboxes, y_tol: float = 7.0):
    """
    For a horizontal word: find the closest number to its left,
    but only if vertically aligned within y_tol (same units as bboxes).
    """
    wx0, wtop, wx1, wbottom = word_bbox
    wcy = (wtop + wbottom) / 2.0  # center y of word

    best = None
    best_d = float("inf")
    for nb in number_bboxes:
        nx0, ntop, nx1, nbottom = nb
        ncy = (ntop + nbottom) / 2.0  # center y of number

        if abs(ncy - wcy) > y_tol:
            continue  # too far vertically

        d = wx0 - nx1  # horizontal gap (>=0 if number is left of word)
        if d >= 0 and d < best_d:
            best_d = d
            best = nb

    return (best, best_d) if best is not None else None


# ============================ DataExtractor impl ============================

from .data_extractor import DataExtractor, ExtractionContext  # keep your existing imports

class NumberBindingExtractor(DataExtractor):
    """
    Extractor that:
      1) Finds target words (Opn/npO, O.D.-like, BASE/ESAB).
      2) Merges BASE↔O.D. nearest pairs into a single word bbox.
      3) Binds each (possibly merged) word to the closest number using orientation rules:
         - vertical word: closest number BELOW, within horizontal tolerance
         - horizontal word: closest number LEFT, within vertical tolerance
      4) Returns a list of items with unified, non-duplicated bboxes:
         - {'type':'binding', 'bbox': union(word, number), 'word': <worddict>, 'number_bbox': (x0,top,x1,bottom)}
         - {'type':'number',  'bbox': number_bbox} for numbers not bound
         - {'type':'word',    'bbox': word_bbox}   for target-words not bound
    """

    def __init__(
        self,
        name: str = "NumberBindingExtractor",
        *,
        pair_radius_px: float = 120.0,  # max pixel distance to pair BASE with O.D.
        x_tol_px: float = 7.0,          # horiz tolerance for vertical words (pixels)
        y_tol_px: float = 7.0           # vert tolerance for horizontal words (pixels)
    ):
        super().__init__(name)
        self.pair_radius_px = pair_radius_px
        self.x_tol_px = x_tol_px
        self.y_tol_px = y_tol_px

    # ---- internal helpers (ctx needed) ----
    @staticmethod
    def _find_nearest_counterpart_word(
        ctx: ExtractionContext,
        anchor_word: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        max_dist_px: float,
    ) -> Optional[Dict[str, Any]]:
        ax, ay = bbox_center_px(ctx, (anchor_word["x0"], anchor_word["top"], anchor_word["x1"], anchor_word["bottom"]))
        best = None
        best_d2 = float("inf")
        for c in candidates:
            cx, cy = bbox_center_px(ctx, (c["x0"], c["top"], c["x1"], c["bottom"]))
            d2 = (cx - ax)**2 + (cy - ay)**2
            if d2 < best_d2:
                best_d2 = d2
                best = c
        if best is None:
            return None
        return best if math.sqrt(best_d2) <= max_dist_px else None

    def _merge_base_od_pairs(self, ctx: ExtractionContext, word_objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        indexed = list(enumerate(word_objs))
        base_like = [(i, w) for i, w in indexed if is_base_like(w.get("text", ""))]
        od_like   = [(i, w) for i, w in indexed if is_od_like(w.get("text", ""))]

        used_ids: Set[int] = set()
        merged: List[Dict[str, Any]] = []

        def _merged_entry(w1: Dict[str, Any], w2: Dict[str, Any], label: str) -> Dict[str, Any]:
            b1 = (w1["x0"], w1["top"], w1["x1"], w1["bottom"])
            b2 = (w2["x0"], w2["top"], w2["x1"], w2["bottom"])
            ux0, utop, ux1, ubot = bbox_union_pdf(b1, b2)
            out = dict(w1)
            out["x0"], out["top"], out["x1"], out["bottom"] = ux0, utop, ux1, ubot
            out["text"] = label
            return out

        # Pass 1: BASE -> nearest OD
        for i, wb in base_like:
            if i in used_ids: continue
            od_candidates = [w for j, w in od_like if j not in used_ids]
            if not od_candidates:
                continue
            near = self._find_nearest_counterpart_word(ctx, wb, od_candidates, self.pair_radius_px)
            if near is not None:
                j = next(j for j, w in od_like if w is near)
                used_ids.add(i); used_ids.add(j)
                merged.append(_merged_entry(wb, near, "BASE+OD"))
            else:
                merged.append(wb); used_ids.add(i)

        # Pass 2: OD -> nearest BASE (for remaining)
        for j, wo in od_like:
            if j in used_ids: continue
            base_candidates = [w for i, w in base_like if i not in used_ids]
            if not base_candidates:
                merged.append(wo); used_ids.add(j)
                continue
            near = self._find_nearest_counterpart_word(ctx, wo, base_candidates, self.pair_radius_px)
            if near is not None:
                i = next(i for i, w in base_like if w is near)
                used_ids.add(j); used_ids.add(i)
                merged.append(_merged_entry(wo, near, "OD+BASE"))
            else:
                merged.append(wo); used_ids.add(j)

        # Any remaining (e.g., Opn/npO)
        for k, w in indexed:
            if k not in used_ids:
                merged.append(w)

        return merged

    def _bind_word_to_number(
        self,
        ctx: ExtractionContext,
        wbox: Tuple[float, float, float, float],
        number_bboxes: List[Tuple[float, float, float, float]],
    ) -> Optional[Tuple[Tuple[float, float, float, float], float, str]]:
        """Bind a single word bbox to closest number using orientation-specific rules, with px→pt tol conversion."""
        orient = classify_word_orientation(wbox)
        # convert pixel tolerances to PDF points so units match bboxes
        x_tol_pts = self.x_tol_px / ctx.scale
        y_tol_pts = self.y_tol_px / ctx.scale
        if orient == "vertical":
            res = closest_number_for_vertical_word(wbox, number_bboxes, x_tol=x_tol_pts)
            return (res[0], res[1], orient) if res else None
        else:
            res = closest_number_for_horizontal_word(wbox, number_bboxes, y_tol=y_tol_pts)
            return (res[0], res[1], orient) if res else None

    def extract(self, ctx: ExtractionContext, shared: Dict[str, object]):
        """
        Returns a list of items (PDF coords):
          - {'type':'binding', 'bbox': (x0,top,x1,bottom), 'word': <worddict>, 'number_bbox': (..)}
          - {'type':'number',  'bbox': (x0,top,x1,bottom)}
          - {'type':'word',    'bbox': (x0,top,x1,bottom)}  # target words not bound
        """
        self._set_ctx(ctx)

        # Collect words/numbers
        words_all = ctx.words()
        nums = ctx.number_bboxes()

        # Filter target words
        target_words = [w for w in words_all if is_target_word(w.get("text"))]

        # Merge BASE <-> O.D. nearest pairs
        merged_words = self._merge_base_od_pairs(ctx, target_words)

        # Attempt to bind each (merged) word to a number (one-to-one)
        used_num_idx: Set[int] = set()
        items: List[Dict[str, Any]] = []

        # Build an index for quick equality matching
        num_index: Dict[Tuple[float, float, float, float], int] = {
            nb: i for i, nb in enumerate(nums)
        }

        for w in merged_words:
            wbox = (w["x0"], w["top"], w["x1"], w["bottom"])
            res = self._bind_word_to_number(ctx, wbox, nums)
            if not res:
                # no number bound; keep the word as standalone (optional category)
                items.append({"type": "word", "bbox": wbox, "word": w})
                continue

            nbox, _, orient = res
            # Make sure we don't reuse the same number for multiple words
            idx = num_index.get(nbox)
            if idx is None or idx in used_num_idx:
                # number already used; treat word as standalone
                items.append({"type": "word", "bbox": wbox, "word": w})
                continue

            used_num_idx.add(idx)
            ub = bbox_union_pdf(wbox, nbox)
            items.append({
                "type": "binding",
                "bbox": ub,
                "word": w,
                "number_bbox": nbox,
                "orientation": orient,
            })

        # Add numbers that were not bound
        for i, nb in enumerate(nums):
            if i not in used_num_idx:
                items.append({"type": "number", "bbox": nb})

        self.result = items
        return self.result

    def plot(self, img: Any):
        """Optional: no-op here. Drawing is handled in main()."""
        return


# ================================ MAIN ======================================

def main():
    import argparse
    from pathlib import Path
    import logging
    import os
    import pdfplumber
    import matplotlib
    from PIL import Image, ImageDraw

    # Use non-interactive backend if needed
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Extract numbers, bind with target words (Opn/npO, O.D.-like, BASE/ESAB), "
                    "and visualize unified bounding boxes."
    )
    parser.add_argument("pdf", type=str, help="Path to a single PDF file")
    parser.add_argument("--page", type=int, default=0, help="Zero-based page index")
    parser.add_argument("--dpi", type=int, default=150, help="Rasterization DPI")
    parser.add_argument("--pair-radius", type=float, default=120.0, help="Max pixel distance to pair BASE with O.D.")
    parser.add_argument("--x-tol", type=float, default=7.0, help="Horizontal px tolerance for vertical words")
    parser.add_argument("--y-tol", type=float, default=7.0, help="Vertical px tolerance for horizontal words")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--out", type=str, default=None, help="Optional output PNG path")
    args = parser.parse_args()

    logging.basicConfig(level=(logging.DEBUG if args.verbose else logging.INFO),
                        format="%(levelname)s %(message)s")

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    from .data_extractor import ExtractionContext  # ensure available at runtime

    with pdfplumber.open(str(pdf_path)) as pdf:
        if not (0 <= args.page < len(pdf.pages)):
            raise IndexError(f"Page index {args.page} out of range [0..{len(pdf.pages)-1}]")
        page = pdf.pages[args.page]

        ctx = ExtractionContext(page, dpi=args.dpi)

        # Run extractor
        ex = NumberBindingExtractor(
            pair_radius_px=args.pair_radius,
            x_tol_px=args.x_tol,
            y_tol_px=args.y_tol,
        )
        items = ex.extract(ctx, shared={})

        logging.info("Produced %d unified items (bindings + standalone words + numbers)", len(items))

        # Draw overlay: bindings (green), numbers-only (blue), words-only (orange)
        base = ctx.pil().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        for it in items:
            x0, top, x1, bottom = it["bbox"]
            # Convert to pixels (top-origin)
            L = ctx.pt_to_px_x(x0)
            R = ctx.pt_to_px_x(x1)
            T = ctx.pt_to_px_y_top(top)
            B = ctx.pt_to_px_y_top(bottom)

            left, right = (min(L, R), max(L, R))
            top_px, bot_px = (min(T, B), max(T, B))

            if it["type"] == "binding":
                color = (0, 255, 0, 255)   # green
                label = f"{it['word']['text']} + num"
            elif it["type"] == "number":
                color = (0, 128, 255, 255) # blue
                label = "num"
            else:
                color = (255, 165, 0, 255) # orange
                label = it.get("word", {}).get("text", "word")

            draw.rectangle([left, top_px, right, bot_px], outline=color, width=3)
            draw.text((left, max(0, top_px - 12)), label, fill=color)

        composed = Image.alpha_composite(base, overlay).convert("RGB")

        # Show figure
        import matplotlib.pyplot as plt
        raw_rgb = ctx.pil().convert("RGB")
        plt.figure(figsize=(12, 7))
        plt.subplot(1, 2, 1); plt.title(f"RAW — {pdf_path.name} p{args.page+1}")
        plt.imshow(raw_rgb); plt.axis("off")
        plt.subplot(1, 2, 2); plt.title("Bindings (green), Numbers (blue), Words (orange)")
        plt.imshow(composed); plt.axis("off")
        plt.tight_layout(); plt.show()

        if args.out:
            out_path = Path(args.out).expanduser().resolve()
            composed.save(str(out_path))
            logging.info("Saved overlay → %s", out_path)


if __name__ == "__main__":
    main()
