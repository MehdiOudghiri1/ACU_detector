from typing import Optional, List, Dict, Tuple
from .data_extractor import ExtractionContext
from .data_extractor import DataExtractor
from PIL import ImageDraw
import pdfplumber
from typing import Any

class Segment:
    def __init__(self, value: float, coordinates: List[float], type: Optional[str], start: Optional[float], end: Optional[float]):
        """type can be either vertical or horizontal"""
        self.type = type
        self.start = start
        self.end = end
        self.value = value
        self.coordinates = coordinates

    def __repr__(self):
        return f"Segment(type={self.type}, start={self.start}, end={self.end})"

    

class SegmentChain:
    def __init__(self, segments: List[Segment] = None):
        self.segments = segments if segments is not None else []

    def add_segment(self, segment: Segment):
        self.segments.append(segment)

    def __repr__(self):
        return f"Segments(segments={self.segments})"
    

class SegmentsExtractor(DataExtractor):
    """Extracts segments from a PDF page."""
    
    def __init__(self, name: str = "segments_extractor"):
        super().__init__(name)

    def extract(self, ctx: ExtractionContext, shared: Dict[str, object]) -> List[SegmentChain]:
        self._set_ctx(ctx)  # keep ctx for plotting / conversions
        self.result = self._classify_bboxes(ctx)
        return self.result
    
    def _reverse_number_any(self, n: float) -> float:

        sign = -1 if n < 0 else 1
        s = str(abs(n))
        reversed_s = s[::-1]
        return sign * float(reversed_s)


    def plot(self, img: Any):
        if self._ctx is None or self.result is None:
            return

        # draw directly on the cached PIL image
        pil_im = self._ctx.pil().convert("RGB")
        draw = ImageDraw.Draw(pil_im)

        def pt2px(x_pt: float, y_pt: float) -> tuple[int, int]:
            return (int(round(self._ctx.pt_to_px_x(x_pt))),
                    int(round(self._ctx.pt_to_px_y(y_pt))))

        # nice distinct palette; auto-cycles if many chains
        palette = [
            (230, 57, 70),   # red-ish
            (29, 53, 87),    # dark blue
            (69, 123, 157),  # blue
            (17, 138, 178),  # teal
            (6, 214, 160),   # green
            (255, 159, 28),  # orange
            (131, 56, 236),  # violet
            (255, 93, 162),  # pink
            (0, 122, 255),   # azure
            (50, 205, 50),   # lime
        ]

        for ci, chain in enumerate(self.result):
            if not chain.segments:
                continue

            color = palette[ci % len(palette)]

            # all centers in pixel space
            pts_px = [pt2px(seg.coordinates[0], seg.coordinates[1])
                      for seg in chain.segments]
            xs = [x for x, _ in pts_px]
            ys = [y for _, y in pts_px]

            # union bbox around this chain, with padding
            pad = 8
            x0, y0 = min(xs) - pad, min(ys) - pad
            x1, y1 = max(xs) + pad, max(ys) + pad
            draw.rectangle((x0, y0, x1, y1), outline=color, width=4)

            # draw points too (helps you see the members)
            for x, y in pts_px:
                r = 3
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

            # optional: label each box with chain type + index
            label = (chain.segments[0].type or "?")[0].upper() + f"#{ci}"
            # slight offset inside the box
            draw.text((x0 + 4, y0 + 3), label, fill=color)

        
    def _classify_bboxes(self, ctx: ExtractionContext) -> List[SegmentChain]:
        """
        Regroupe les nombres détectés en chaînes de segments par centres entiers :
        - Chaînes verticales : même int(x_center), triées de haut→bas (cy croissant)
        - Chaînes horizontales : même int(y_center), triées de gauche→droite (cx croissant)
        """

        # BBoxes et textes numériques (même ordre que number_bboxes)
        boxes = ctx.number_bboxes()  # [(x0, top, x1, bottom), ...]
        num_words = [w for w in ctx.words() if ctx._is_number(w.get("text", ""))]
        texts = [w.get("text", "") for w in num_words]

        def to_int_value(s: str) -> int:
            s = s.strip()
            try:
                return int(s)
            except ValueError:
                pass
            try:
                return int(float(s))
            except ValueError:
                import re
                s2 = re.sub(r"[^0-9\.\-]+", "", s)
                return int(float(s2)) if s2 not in ("", ".", "-") else 0

        # Centres + segments de base
        centers: List[Tuple[float, float]] = []
        segments_base: List[Segment] = []
        n = min(len(boxes), len(texts))
        for i in range(n):
            x0, top, x1, bottom = boxes[i]
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (top + bottom)
            centers.append((cx, cy))
            val = to_int_value(texts[i])
            segments_base.append(Segment(value=val, coordinates=[cx, cy], type=None, start=None, end=None))

        # Binning par partie entière des centres
        already_assigned: List[bool] = [False] * n
        vert_bins: Dict[int, List[int]] = {}
        horiz_bins: Dict[int, List[int]] = {}
        for idx, (cx, cy) in enumerate(centers):
            vert_bins.setdefault(int(cx), []).append(idx)
            horiz_bins.setdefault(int(cy), []).append(idx)

        chains: List[SegmentChain] = []

        # Chaînes verticales (haut -> bas)
        for bx in sorted(vert_bins.keys()):
            idxs = sorted(vert_bins[bx], key=lambda i: centers[i][1])
            if len(idxs) < 2:
                continue  # on saute les singletons ici; ils seront traités côté horizontal
            chain = SegmentChain()
            for i in idxs:
                base = segments_base[i]
                chain.add_segment(
                    Segment(
                        value=self._reverse_number_any(base.value),
                        coordinates=base.coordinates[:],
                        type="vertical",
                        start=None,
                        end=None,
                    )
                )
                already_assigned[i] = True
            if chain.segments:
                chains.append(chain)

        # Chaînes horizontales (gauche -> droite)
        for by in sorted(horiz_bins.keys()):
            idxs = sorted(horiz_bins[by], key=lambda i: centers[i][0])
            print("[DEBUG] values = ", [segments_base[i].value for i in idxs])

            if len(idxs) == 1:
                # --- FIX: use the actual index, not 0 ---
                i0 = idxs[0]
                if already_assigned[i0]:
                    continue
                base = segments_base[i0]
                chain = SegmentChain()
                chain.add_segment(
                    Segment(
                        value=base.value,
                        coordinates=base.coordinates[:],
                        type="isolated",
                        start=None,
                        end=None,
                    )
                )
                already_assigned[i0] = True
                print("[DEBUG] the chain appended is", chain)
                chains.append(chain)
            else:
                chain = SegmentChain()
                for i in idxs:
                    if not already_assigned[i]:
                        base = segments_base[i]
                        chain.add_segment(
                            Segment(
                                value=base.value,
                                coordinates=base.coordinates[:],
                                type="horizontal",
                                start=None,
                                end=None,
                            )
                        )
                        already_assigned[i] = True
                if chain.segments:
                    print("[DEBUG] the chain appended is", chain)
                    chains.append(chain)

        return chains

        

    # ---------- quick main to try SegmentsExtractor ----------
import os, sys, random, argparse
import matplotlib.pyplot as plt

def _print_results(results: List[SegmentChain]) -> None:
    print(f"\nChains: {len(results)}")
    for k, ch in enumerate(results):
        kind = ch.segments[0].type if ch.segments else "?"
        pts  = len(ch.segments)
        vals = [s.value for s in ch.segments[:6]]
        print(f"  [{k:02d}] type={kind:10s}  points={pts:<3d}  sample_values={vals}")

def render_one(page: pdfplumber.page.Page, extractor: SegmentsExtractor, dpi: int = 150):
    ctx = ExtractionContext(page, dpi=dpi)
    res = extractor.extract(ctx, shared={})
    img = ctx.img()          # keep for compatibility if needed
    extractor.plot(img)      # draws on ctx.pil()

    shown = ctx.pil()        # <<< display the PIL with overlays
    return shown, res, page.page_number

def main_quick():
    parser = argparse.ArgumentParser(description="Quick SegmentsExtractor demo.")
    parser.add_argument("folder", help="Folder with PDF files")
    parser.add_argument("--file", dest="specific_file", help="Restrict to a single PDF (name must match)")
    parser.add_argument("--dpi", type=int, default=150, help="Rendering DPI")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Folder not found: {args.folder}", file=sys.stderr)
        sys.exit(1)

    pdfs = [f for f in os.listdir(args.folder) if f.lower().endswith(".pdf")]
    if not pdfs:
        print("No PDFs found in:", args.folder, file=sys.stderr)
        sys.exit(1)

    if args.specific_file:
        if args.specific_file not in pdfs:
            print(f"File '{args.specific_file}' not found in '{args.folder}'", file=sys.stderr)
            sys.exit(1)
        pdfs = [args.specific_file]

    extractor = SegmentsExtractor()

    print("Press Enter for next page, 'q' to quit…")
    while True:
        pdf_name = random.choice(pdfs)
        path = os.path.join(args.folder, pdf_name)
        with pdfplumber.open(path) as pdf:
            page = random.choice(pdf.pages)
            shown_img, results, pnum = render_one(page, extractor, dpi=args.dpi)

        print(f"\n{pdf_name} — page {pnum}")
        _print_results(results)

        plt.figure(figsize=(8, 10))
        plt.imshow(shown_img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        key = input().strip().lower()
        if key in {"q", "quit", "exit"}:
            plt.close("all")
            break
        plt.close("all")

if __name__ == "__main__":
    main_quick()


















