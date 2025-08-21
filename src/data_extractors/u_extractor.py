from typing import List, Dict, Any, Optional, Tuple, Set
from band_extractor import Bande
from data_extractor import DataExtractor, ExtractionContext
from band_extractor import BandeExtractor
from number_binding_extractor import NumberBindingExtractor


class U:
    def __init__(self, value: float, base: list[Bande], horns: list[Bande]):
        self.value = value
        self.base = base
        self.horns = horns


class UExtractor(DataExtractor):
    def extract(self, ctx: ExtractionContext, shared: Dict[str, object]):
        """Compute and store results in self.result, also return it."""
        ...

    def plot(self, img: Any):
        """Draw overlays on a pdfplumber image object (PDF coords)."""
        ...


def find_vertical_bands_near_number(
    ctx: ExtractionContext,
    bbox_pdf: Tuple[float, float, float, float],
    *,
    tol_px: int = 10,          # vertical proximity to bbox top/bottom
    x_tol_px: int = 13,         # horizontal proximity to bbox center
    bde: "BandeExtractor" = None,
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
    vertical_bands: List["Bande"] = bde.result["vertical"]

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
        "all_vertical": vertical_bands,  # keep for debugging
    }


def main():
    import argparse
    from pathlib import Path
    import logging
    import os
    import pdfplumber
    import matplotlib

    # Use a non-interactive backend if headless (avoids Qt/GLX errors)
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from PIL import Image, ImageDraw

    # Local imports expected to exist in your project
    from data_extractor import ExtractionContext
    from band_extractor import BandeExtractor
    # NumberBindingExtractor should be defined in this file (as you added it)
    # find_vertical_bands_near_number is assumed available in scope

    parser = argparse.ArgumentParser(
        description="Visualize number/binding bboxes (from NumberBindingExtractor) and highlight nearby vertical bands."
    )
    parser.add_argument("pdf", type=str, help="Path to a single PDF file")
    parser.add_argument("--page", type=int, default=0, help="Zero-based page index")
    parser.add_argument("--dpi", type=int, default=150, help="Rasterization DPI")

    # Bande extraction params
    parser.add_argument("--thr", type=int, default=60, help="Black threshold 0..255")
    parser.add_argument("--min-thick", type=int, default=1, help="Min band thickness (px)")
    parser.add_argument("--max-thick", type=int, default=20, help="Max band thickness (px)")
    parser.add_argument("--v-min-len", type=int, default=20, help="Vertical: min length (px)")
    parser.add_argument("--h-min-len", type=int, default=20, help="Horizontal: min length (px)")
    parser.add_argument("--bridge", type=int, default=1, help="Bridge tiny gaps (px)")

    # Band association tolerances
    parser.add_argument("--tol", type=int, default=10, help="Vertical proximity to bbox top/bottom (px)")
    parser.add_argument("--x-tol", type=int, default=12, help="Horizontal proximity to bbox center (px)")

    # NumberBindingExtractor tolerances
    parser.add_argument("--pair-radius", type=float, default=120.0, help="Max pixel distance to pair BASE with O.D.")
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

        # 1) Run BandeExtractor once
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

        # 2) Run NumberBindingExtractor to get unified items (bindings + numbers + words)
        nbe = NumberBindingExtractor(
            pair_radius_px=args.pair_radius,
            x_tol_px=args.x_align,
            y_tol_px=args.y_align,
        )
        items = nbe.extract(ctx, shared={})
        logging.info("NumberBindingExtractor produced %d items", len(items))

        # We’ll associate vertical bands ONLY to items that are 'binding' or 'number'
        num_like_boxes = [it["bbox"] for it in items if it["type"] in {"binding", "number"}]
        logging.info("Found %d number/binding bboxes to check for bands", len(num_like_boxes))

        # 3) Associate vertical bands to each bbox
        associated_bands = set()
        total_matches = 0
        for i, bbx in enumerate(num_like_boxes):
            found = find_vertical_bands_near_number(
                ctx,
                bbx,
                tol_px=args.tol,
                x_tol_px=args.x_tol,
                bde=bde,
            )
            near_v = found["near_top"] + found["near_bottom"]
            total_matches += len(near_v)
            logging.info(f"[assoc] item #{i:02d} → {len(near_v)} vertical bands")
            for b in near_v:
                associated_bands.add((int(b.x0), int(b.x1), int(b.y0), int(b.y1)))

        logging.info(f"[assoc] unique associated vertical bands = {len(associated_bands)} "
                     f"(total matches counted = {total_matches})")

        # 4) Draw overlay:
        #   - bindings/numbers in green
        #   - words-only (unbound target words) in orange
        #   - associated vertical bands in red
        base = ctx.pil().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        # Draw bindings & numbers in green
        for it in items:
            if it["type"] not in {"binding", "number"}:
                continue
            x0_pt, top_pt, x1_pt, bottom_pt = it["bbox"]
            x0_px = ctx.pt_to_px_x(x0_pt)
            x1_px = ctx.pt_to_px_x(x1_pt)
            y_top_px = ctx.pt_to_px_y_top(top_pt)
            y_bottom_px = ctx.pt_to_px_y_top(bottom_pt)
            left, right = sorted([x0_px, x1_px])
            top, bottom = sorted([y_top_px, y_bottom_px])
            draw.rectangle([left, top, right, bottom], outline=(0, 255, 0, 255), width=2)

        # Draw words-only (unbound target words) in orange
        for it in items:
            if it["type"] != "word":
                continue
            x0_pt, top_pt, x1_pt, bottom_pt = it["bbox"]
            x0_px = ctx.pt_to_px_x(x0_pt)
            x1_px = ctx.pt_to_px_x(x1_pt)
            y_top_px = ctx.pt_to_px_y_top(top_pt)
            y_bottom_px = ctx.pt_to_px_y_top(bottom_pt)
            left, right = sorted([x0_px, x1_px])
            top, bottom = sorted([y_top_px, y_bottom_px])
            draw.rectangle([left, top, right, bottom], outline=(255, 165, 0, 255), width=2)

        # Draw associated vertical bands in red (pixel coords)
        for (x0, x1, y0, y1) in associated_bands:
            left, right = sorted([x0, x1])
            top, bottom = sorted([y0, y1])
            draw.rectangle([left, top, right, bottom], outline=(255, 0, 0, 255), width=3)

        composed = Image.alpha_composite(base, overlay).convert("RGB")

        # 5) Show figure side-by-side: raw vs overlay (or save if headless)
        raw_rgb = ctx.pil().convert("RGB")
        plt.figure(figsize=(12, 7))
        plt.subplot(1, 2, 1)
        plt.title(f"RAW — {pdf_path.name} p{args.page+1}")
        plt.imshow(raw_rgb)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Bindings/Numbers (green), Words (orange), Bands (red)")
        plt.imshow(composed)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        if args.out:
            out_path = Path(args.out).expanduser().resolve()
            composed.save(str(out_path))
            logging.info("Saved overlay → %s", out_path)



if __name__ == "__main__":
    main()
