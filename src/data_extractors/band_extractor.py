from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
from .data_extractor import ExtractionContext, DataExtractor
import logging


# ───────────────────────────── Bande (segment rectangulaire) ─────────────────
# Une Bande est un fin rectangle noir: (x0, x1, y0, y1) en pixels, avec orientation.

@dataclass
class Bande:
    x0: int
    x1: int
    y0: int
    y1: int
    orientation: str  # "vertical" | "horizontal"

    # ---------- utils ----------
    @staticmethod
    def _black_mask(img: np.ndarray, thr: int = 30) -> np.ndarray:
        """Pixel 'noir' si moyenne RGB < thr (ou L < thr)."""
        if img.ndim == 2:
            m = img < thr
        else:
            m = (img[..., :3].mean(axis=2) < thr)
        logging.debug(f"[mask] thr={thr} true_ratio={m.mean():.4f} shape={m.shape}")
        return m

    @staticmethod
    def _true_runs(vec: np.ndarray) -> List[Tuple[int, int]]:
        """Runs (start, end_inclus) de True sur un vecteur 1D bool."""
        runs: List[Tuple[int, int]] = []
        in_run = False
        start = 0
        for i, v in enumerate(vec):
            if v and not in_run:
                in_run = True
                start = i
            elif not v and in_run:
                runs.append((start, i - 1))
                in_run = False
        if in_run:
            runs.append((start, len(vec) - 1))
        return runs

    @staticmethod
    def _max_true_run(vec: np.ndarray) -> int:
        best = cur = 0
        for v in vec:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    # ---------- finders: retournent des Bandes (x0,x1,y0,y1) filtrées par épaisseur & longueur ----------
    @staticmethod
    def find_vertical(
        img: np.ndarray,
        *,
        thr: int = 30,
        min_thick: int = 1,
        max_thick: int = 15,
        min_len_px: int = 8,
        max_len_px: Optional[int] = None,
        bridge_px: int = 1,
    ) -> List["Bande"]:
        """
        Détecte des BANDES VERTICALES (= segments verticaux fins):
          1) masque 'noir' (seuil sur moyenne RGB)
          2) dilatation horizontale simple (bridge_px) pour combler 1..k trous
          3) colonne marquée si sa plus longue course verticale >= min_len_px
          4) groupement des colonnes marquées -> épaisseur; filtre [min_thick..max_thick]
          5) à l'intérieur de chaque groupe, on regarde les lignes où il y a du noir
             et on renvoie les segments contigus (y0..y1) filtrés par longueur [min_len_px..max_len_px]
        """
        mask = Bande._black_mask(img, thr)
        H, W = mask.shape

        if bridge_px > 0:
            m = mask.copy()
            for s in range(1, bridge_px + 1):
                m |= np.roll(mask,  s, axis=1)
                m |= np.roll(mask, -s, axis=1)
            mask = m

        col_has_line = np.zeros(W, dtype=bool)
        for x in range(W):
            col_has_line[x] = (Bande._max_true_run(mask[:, x]) >= min_len_px)

        bands: List[Bande] = []
        for x0, x1 in Bande._true_runs(col_has_line):
            thickness = x1 - x0 + 1
            if not (min_thick <= thickness <= max_thick):
                continue

            band_cols = mask[:, x0:x1 + 1]      # (H, thickness)
            rows_any = band_cols.any(axis=1)    # (H,)

            for y0, y1 in Bande._true_runs(rows_any):
                length = (y1 + 1) - y0
                if length < min_len_px:
                    continue
                if (max_len_px is not None) and (length > max_len_px):
                    continue
                bands.append(Bande(x0, x1 + 1, y0, y1 + 1, "vertical"))

        logging.info(f"[V] thr={thr} thick=[{min_thick},{max_thick}] len=[{min_len_px},{max_len_px}] "
                     f"bridge={bridge_px} → bands={len(bands)}")
        return bands

    @staticmethod
    def find_horizontal(
        img: np.ndarray,
        *,
        thr: int = 30,
        min_thick: int = 1,
        max_thick: int = 15,
        min_len_px: int = 8,
        max_len_px: Optional[int] = None,
        bridge_px: int = 1,
    ) -> List["Bande"]:
        """
        Détecte des BANDES HORIZONTALES (= segments horizontaux fins).
        Logique identique à find_vertical mais axes échangés.
        """
        mask = Bande._black_mask(img, thr)
        H, W = mask.shape

        if bridge_px > 0:
            m = mask.copy()
            for s in range(1, bridge_px + 1):
                m |= np.roll(mask,  s, axis=0)
                m |= np.roll(mask, -s, axis=0)
            mask = m

        row_has_line = np.zeros(H, dtype=bool)
        for y in range(H):
            row_has_line[y] = (Bande._max_true_run(mask[y, :]) >= min_len_px)

        bands: List[Bande] = []
        for y0, y1 in Bande._true_runs(row_has_line):
            thickness = y1 - y0 + 1
            if not (min_thick <= thickness <= max_thick):
                continue

            band_rows = mask[y0:y1 + 1, :]      # (thickness, W)
            cols_any = band_rows.any(axis=0)    # (W,)

            for x0, x1 in Bande._true_runs(cols_any):
                length = (x1 + 1) - x0
                if length < min_len_px:
                    continue
                if (max_len_px is not None) and (length > max_len_px):
                    continue
                bands.append(Bande(x0, x1 + 1, y0, y1 + 1, "horizontal"))

        logging.info(f"[H] thr={thr} thick=[{min_thick},{max_thick}] len=[{min_len_px},{max_len_px}] "
                     f"bridge={bridge_px} → bands={len(bands)}")
        return bands


# ───────────────────────────── Extractor des bandes ──────────────────────────

class BandeExtractor(DataExtractor):
    """
    Détecteur de bandes noires fines avec contrôle:
      - Épaisseur (min_thick..max_thick, en pixels)
      - Longueur (min_len_px..max_len_px, en pixels) séparément pour V et H
      - Seuil de noirceur (thr)
      - Bridge pour combler de petits trous (1..k px)
    """
    def __init__(
        self,
        name: str = "bandes",
        *,
        thr: int = 30,
        min_thick: int = 1,
        max_thick: int = 15,
        v_min_len_px: int = 8,
        v_max_len_px: Optional[int] = None,
        h_min_len_px: int = 8,
        h_max_len_px: Optional[int] = None,
        bridge_px: int = 1,
    ):
        super().__init__(name)
        self.thr = thr
        self.min_thick = min_thick
        self.max_thick = max_thick
        self.v_min_len_px = v_min_len_px
        self.v_max_len_px = v_max_len_px
        self.h_min_len_px = h_min_len_px
        self.h_max_len_px = h_max_len_px
        self.bridge_px = bridge_px

    def extract(self, ctx: ExtractionContext, shared: Dict[str, object]):
        self._set_ctx(ctx)
        img = np.array(ctx.pil())
        H, W = img.shape[:2]
        logging.info(
            f"[extract] size={W}x{H} thr={self.thr} thick=[{self.min_thick},{self.max_thick}] "
            f"V.len=[{self.v_min_len_px},{self.v_max_len_px}] "
            f"H.len=[{self.h_min_len_px},{self.h_max_len_px}] bridge={self.bridge_px}"
        )

        v_bandes = Bande.find_vertical(
            img,
            thr=self.thr,
            min_thick=self.min_thick,
            max_thick=self.max_thick,
            min_len_px=self.v_min_len_px,
            max_len_px=self.v_max_len_px,
            bridge_px=self.bridge_px,
        )
        h_bandes = Bande.find_horizontal(
            img,
            thr=self.thr,
            min_thick=self.min_thick,
            max_thick=self.max_thick,
            min_len_px=self.h_min_len_px,
            max_len_px=self.h_max_len_px,
            bridge_px=self.bridge_px,
        )
        self.result = {"vertical": v_bandes, "horizontal": h_bandes}
        logging.info(f"[extract] bands → V={len(v_bandes)} H={len(h_bandes)}")
        return self.result

    def render_overlay(self, ctx: ExtractionContext, *, stroke_w: int = 2) -> Image.Image:
        """Dessine les bandes en couleurs (PIL RGBA) et retourne l'image composée RGB."""
        assert self.result is not None
        base = ctx.pil().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        def alpha_for(i: int, n: int, a_min=90, a_max=220) -> int:
            if n <= 1:
                return a_max
            return int(a_min + (a_max - a_min) * (i / (n - 1)))

        v_b, h_b = self.result["vertical"], self.result["horizontal"]

        # Vertical: rouge
        for i, b in enumerate(v_b):
            fill = (255, 0, 0, alpha_for(i, len(v_b)))
            draw.rectangle([b.x0, b.y0, b.x1, b.y1], outline=(255, 0, 0, 255), width=stroke_w, fill=fill)

        # Horizontal: bleu
        for j, b in enumerate(h_b):
            fill = (0, 0, 255, alpha_for(j, len(h_b)))
            draw.rectangle([b.x0, b.y0, b.x1, b.y1], outline=(0, 0, 255, 255), width=stroke_w, fill=fill)

        return Image.alpha_composite(base, overlay).convert("RGB")

    # Implémente l'abstrait pour compat DataExtractor (utile si on veut img.save(...))
    def plot(self, img: Any):
        assert self._ctx is not None and self.result is not None, "Call extract() first."
        composed = self.render_overlay(self._ctx)
        if getattr(img, "annotated", None) is None:
            img.reset()
        img.annotated = composed.convert("RGB")


# ------------------------------ Main (single PDF, display & optional save) ------------------------------

def main():
    import argparse
    from pathlib import Path
    import pdfplumber
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Detect & DISPLAY thin black bands (segments) on a single PDF page."
    )
    parser.add_argument("pdf", type=str, help="Path to a single PDF file.")
    parser.add_argument("--page", type=int, default=0, help="Zero-based page index.")
    parser.add_argument("--dpi", type=int, default=150, help="Rasterization DPI.")
    parser.add_argument("--thr", type=int, default=60, help="Black threshold 0..255 (default: 60).")

    # Épaisseur (thickness) partagée V/H
    parser.add_argument("--min-thick", type=int, default=1, help="Min band thickness in px.")
    parser.add_argument("--max-thick", type=int, default=20, help="Max band thickness in px.")

    # Longueur en pixels — séparée pour vertical et horizontal
    parser.add_argument("--v-min-len", type=int, default=50, help="Vertical: min length in px.")
    parser.add_argument("--v-max-len", type=int, default=None, help="Vertical: max length in px (None = no cap).")
    parser.add_argument("--h-min-len", type=int, default=50, help="Horizontal: min length in px.")
    parser.add_argument("--h-max-len", type=int, default=None, help="Horizontal: max length in px (None = no cap).")

    parser.add_argument("--bridge", type=int, default=1, help="Bridge pixels to fill tiny gaps (0..N).")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument("--out", type=str, default=None, help="(Optional) Save overlay PNG path.")
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

        # Contexte + extraction
        ctx = ExtractionContext(page, dpi=args.dpi)
        extractor = BandeExtractor(
            thr=args.thr,
            min_thick=args.min_thick,
            max_thick=args.max_thick,
            v_min_len_px=args.v_min_len,
            v_max_len_px=args.v_max_len,
            h_min_len_px=args.h_min_len,
            h_max_len_px=args.h_max_len,
            bridge_px=args.bridge,
        )
        res = extractor.extract(ctx, shared={})

        v_b = res["vertical"]
        h_b = res["horizontal"]
        logging.info("Detected bands: vertical=%d, horizontal=%d", len(v_b), len(h_b))

        # Affichage PIL (pas d'ambiguïté de couleurs)
        base = ctx.pil().convert("RGB")
        composed = extractor.render_overlay(ctx, stroke_w=2)

        # Sauvegarde optionnelle
        if args.out:
            out_path = Path(args.out).expanduser().resolve()
            composed.save(str(out_path))
            logging.info("Saved overlay → %s", out_path)

        # Show side-by-side: raw vs overlay
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 7))
        plt.subplot(1, 2, 1)
        plt.title(f"RAW — {pdf_path.name} p{args.page+1}")
        plt.imshow(base)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Bands (red = vertical, blue = horizontal)")
        plt.imshow(composed)
        plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
