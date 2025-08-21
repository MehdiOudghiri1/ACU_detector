from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pdfplumber
from PIL import Image
from abc import ABC, abstractmethod


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

    # add these helpers once in ExtractionContext
    def pt_to_px_y_top(self, y_pt: float) -> float:
        return y_pt * self.scale

    def px_to_pt_y_top(self, y_px: float) -> float:
        return y_px / self.scale



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
