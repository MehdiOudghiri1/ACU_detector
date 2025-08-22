# -*- coding: utf-8 -*-
# frontend.py — interactive viewer for band rectangles (not infinite lines)
# Shows ONLY:
#   • horizontal bands near vertical bands  (blue rectangles)
#   • vertical   bands near horizontal bands (red  rectangles)
# Uses UExtractor (u_extractor3.py) for band association.
# Uses WhiteBandExtractor to split the page (top/bottom) and pre-select, per half,
# the horizontal band with the lowest y and the one with the highest y.
# NEW: Also pre-select vertical bands whose bottom Y is within Δ pixels of the
#      global maximum bottom Y among all vertical bands (red).
# Bands are clickable: toggles fill + shows a centerline and the equation.
# Hold 'H' to hide all overlays temporarily.

from __future__ import annotations

import sys
from typing import Optional, List, Tuple, Any, Dict

import fitz  # PyMuPDF
import pdfplumber

from PySide6.QtCore import Qt, QRectF, QLineF, QPointF
from PySide6.QtGui import (
    QImage, QPixmap, QBrush, QPen, QColor, QGuiApplication, QFont
)
from PySide6.QtWidgets import (
    QApplication, QGraphicsScene, QGraphicsView,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsLineItem, QGraphicsSimpleTextItem
)

# ───────────────────── Backend imports ─────────────────────
from data_extractors.data_extractor import ExtractionContext
from data_extractors.u_extractor3 import UExtractor  # your pipeline producing rectangles
from data_extractors.dimension_extractor import WhiteBandExtractor  # provided by you


# ───────────────────── Utilities ─────────────────────

def qimage_from_fitz_to_size(pdf_path: str, page_index: int, target_w: int, target_h: int) -> QImage:
    """Render a PDF page with PyMuPDF to match a target pixel size (from ExtractionContext)."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    zoom_x = target_w / page.rect.width if page.rect.width > 0 else 1.0
    zoom_y = target_h / page.rect.height if page.rect.height > 0 else 1.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom_x, zoom_y), alpha=False)  # RGB
    qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888).copy()
    doc.close()
    return qimg


def qcolor_rgba(r: int, g: int, b: int, a: int = 255) -> QColor:
    c = QColor(r, g, b)
    c.setAlpha(a)
    return c


# ───────────────────── Interactive items ─────────────────────

class BandRectItem(QGraphicsRectItem):
    """
    Clickable rectangle that represents a band (with true thickness).
    On click or programmatic selection:
      • toggles fill
      • shows/hides an associated centerline
      • shows/hides an equation label: y = cy  (for horizontal) or x = cx (for vertical)
    """
    def __init__(
        self,
        rect_px: QRectF,
        orientation: str,               # "horizontal" or "vertical"
        stroke_rgb: Tuple[int, int, int],
        fill_alpha: int = 90,
        line_width_normal: float = 1.5,
        line_width_selected: float = 3.0,
    ):
        super().__init__(rect_px)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)

        self.orientation = orientation
        self.base_color = QColor(*stroke_rgb)
        self.fill_alpha = int(fill_alpha)
        self.selected = False

        # visuals
        self._pen_normal = QPen(qcolor_rgba(*stroke_rgb, 255), 2.0)
        self._pen_hover  = QPen(qcolor_rgba(*stroke_rgb, 255), 3.0)
        self.setPen(self._pen_normal)
        self.setBrush(Qt.NoBrush)

        # associated centerline + label (created by caller)
        self.center_line: Optional[QGraphicsLineItem] = None
        self.center_line_pen_normal = QPen(qcolor_rgba(*stroke_rgb, 120), line_width_normal)
        self.center_line_pen_selected = QPen(qcolor_rgba(*stroke_rgb, 255), line_width_selected)
        self.center_label: Optional[QGraphicsSimpleTextItem] = None

    def set_centerline_and_label(
        self,
        line_item: QGraphicsLineItem,
        label_item: QGraphicsSimpleTextItem,
    ):
        self.center_line = line_item
        self.center_line.setPen(self.center_line_pen_normal)
        self.center_line.setVisible(False)

        self.center_label = label_item
        self.center_label.setVisible(False)

    # programmatic selection
    def set_selected_state(self, selected: bool):
        self.selected = bool(selected)
        if self.selected:
            self.setBrush(QBrush(qcolor_rgba(self.base_color.red(), self.base_color.green(), self.base_color.blue(), self.fill_alpha)))
            if self.center_line is not None:
                self.center_line.setVisible(True)
                self.center_line.setPen(self.center_line_pen_selected)
            if self.center_label is not None:
                self.center_label.setVisible(True)
        else:
            self.setBrush(Qt.NoBrush)
            if self.center_line is not None:
                self.center_line.setVisible(False)
                self.center_line.setPen(self.center_line_pen_normal)
            if self.center_label is not None:
                self.center_label.setVisible(False)

    def mousePressEvent(self, event):
        # Toggle selection state
        self.set_selected_state(not self.selected)

        # Print equation in console too
        r = self.rect()
        if self.orientation == "horizontal":
            cy = r.center().y()
            print(f"Clicked H-band → y = {cy:.1f}")
        else:
            cx = r.center().x()
            print(f"Clicked V-band → x = {cx:.1f}")

        super().mousePressEvent(event)

    def hoverEnterEvent(self, event):
        self.setPen(self._pen_hover)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(self._pen_normal)
        super().hoverLeaveEvent(event)


class InteractiveView(QGraphicsView):
    """Hold 'H' to hide all overlays (rects/lines/labels), release to show them back."""
    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self._overlays: List[QGraphicsRectItem | QGraphicsLineItem | QGraphicsSimpleTextItem] = []
        self.setFocusPolicy(Qt.StrongFocus)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(self.renderHints())

    def register_overlays(self, items: List[Any]):
        self._overlays = items

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_H:
            for it in self._overlays:
                it.setVisible(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_H:
            for it in self._overlays:
                it.setVisible(True)
            event.accept()
            return
        super().keyReleaseEvent(event)


# ───────────────────── Main app logic ─────────────────────

def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Show rectangles for horizontal bands near vertical bands (blue) "
                    "and vertical bands near horizontal bands (red) using UExtractor. "
                    "Click to show centerline & equation. Hold 'H' to hide overlays. "
                    "Split page with WhiteBandExtractor and auto-select per half the horizontal "
                    "bands with the lowest and highest y. Also pre-select vertical bands "
                    "whose bottom Y is within Δ of the global maximum."
    )
    parser.add_argument("pdf", type=str, help="Path to PDF")
    parser.add_argument("--page", type=int, default=0, help="Zero-based page index")
    parser.add_argument("--dpi", type=int, default=150, help="Rasterization DPI")

    # Optional overrides for UExtractor (use its defaults if you don't pass these)
    parser.add_argument("--thr", type=int, default=None)
    parser.add_argument("--min-thick", type=int, default=None)
    parser.add_argument("--max-thick", type=int, default=None)
    parser.add_argument("--v-min-len", type=int, default=None)
    parser.add_argument("--h-min-len", type=int, default=None)
    parser.add_argument("--bridge", type=int, default=None)

    # Linkage tolerances (only if you want to override UExtractor defaults)
    parser.add_argument("--h-x0-tol", type=int, default=None)
    parser.add_argument("--h-right-ext-tol", type=int, default=None)
    parser.add_argument("--h-center-y-margin", type=int, default=None)
    parser.add_argument("--v-top-y0-tol", type=int, default=None)
    parser.add_argument("--v-bottom-y1-tol", type=int, default=None)
    parser.add_argument("--v-center-x-edge-tol", type=int, default=None)

    # NEW: delta (pixels) for vertical preselection near global maximum bottom Y
    parser.add_argument("--v-maxy-delta", type=int, default=5,
                        help="Preselect vertical bands whose bottom Y is within this delta of the global max bottom Y")

    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Qt app
    app = QApplication(sys.argv)

    # Build backend context (pdfplumber → ExtractionContext)
    with pdfplumber.open(str(pdf_path)) as pdf:
        if not (0 <= args.page < len(pdf.pages)):
            raise IndexError(f"Page index {args.page} out of range [0..{len(pdf.pages)-1}]")
        page = pdf.pages[args.page]
        ctx = ExtractionContext(page, dpi=args.dpi)

        # Shared pool so we reuse computed bands/NBE
        shared: Dict[str, object] = {}

        # Run UExtractor in V-mode (to get H near V = blue)
        u_v = UExtractor(horizontal_mode=False)
        if args.thr is not None: u_v.thr = args.thr
        if args.min_thick is not None: u_v.min_thick = args.min_thick
        if args.max_thick is not None: u_v.max_thick = args.max_thick
        if args.v_min_len is not None: u_v.v_min_len = args.v_min_len
        if args.h_min_len is not None: u_v.h_min_len = args.h_min_len
        if args.bridge is not None: u_v.bridge = args.bridge
        if args.h_x0_tol is not None: u_v.h_start_x0_tol_px = args.h_x0_tol
        if args.h_right_ext_tol is not None: u_v.h_right_ext_tol_px = args.h_right_ext_tol
        if args.h_center_y_margin is not None: u_v.h_center_y_margin_px = args.h_center_y_margin
        if args.v_top_y0_tol is not None: u_v.v_top_y0_tol_px = args.v_top_y0_tol
        if args.v_bottom_y1_tol is not None: u_v.v_bottom_y1_tol_px = args.v_bottom_y1_tol
        if args.v_center_x_edge_tol is not None: u_v.v_center_x_edge_tol_px = args.v_center_x_edge_tol

        res_v = u_v.extract(ctx, shared=shared)
        shared["BandeExtractor"] = u_v.band_extractor  # reuse bands

        # We only *show* blue = H near V and red = V near H.
        blue_rects = list(res_v["h_near_v"])  # set of (x0,x1,y0,y1) in px ints

        # For red rectangles (V near H), run H-mode once to compute that map
        u_h = UExtractor(horizontal_mode=True)
        # mirror overrides
        for attr in ["thr", "min_thick", "max_thick", "v_min_len", "h_min_len", "bridge",
                     "h_start_x0_tol_px", "h_right_ext_tol_px", "h_center_y_margin_px",
                     "v_top_y0_tol_px", "v_bottom_y1_tol_px", "v_center_x_edge_tol_px"]:
            setattr(u_h, attr, getattr(u_v, attr))
        res_h = u_h.extract(ctx, shared=shared)
        red_rects = list(res_h["v_near_h"])   # set of (x0,x1,y0,y1) in px ints

        # Background image rendered to match ctx pixel size exactly
        img_w, img_h = ctx.pil().size
        bg_img = qimage_from_fitz_to_size(str(pdf_path), args.page, img_w, img_h)

        # White band to split page (for H preselection per half)
        wbe = WhiteBandExtractor()
        white_res = wbe.extract(ctx, shared)
        y_top_white: Optional[int] = None
        y_bot_white: Optional[int] = None
        if white_res:
            y_top_white = int(white_res["y_top_px"])
            y_bot_white = int(white_res["y_bot_px"])

    # Scene + background
    scene = QGraphicsScene()
    bg_item = QGraphicsPixmapItem(QPixmap.fromImage(bg_img))
    bg_item.setZValue(0)
    scene.addItem(bg_item)

    overlays: List[Any] = []

    # Helper to add a band rectangle + its centerline + equation label
    def add_band_rect(rect_tuple: Tuple[int, int, int, int], orientation: str, stroke_rgb: Tuple[int, int, int]) -> BandRectItem:
        x0, x1, y0, y1 = rect_tuple
        left, right = (min(x0, x1), max(x0, x1))
        top, bottom = (min(y0, y1), max(y0, y1))
        item = BandRectItem(
            rect_px=QRectF(float(left), float(top), float(right - left), float(bottom - top)),
            orientation=orientation,
            stroke_rgb=stroke_rgb,
            fill_alpha=90,
        )
        item.setZValue(10)
        scene.addItem(item)
        overlays.append(item)

        # centerline across full image
        if orientation == "horizontal":
            cy = (top + bottom) * 0.5
            line_item = QGraphicsLineItem(QLineF(0, cy, img_w, cy))
            equation_text = f"y = {cy:.1f}"
            label_pos = QPointF(float(left), float(max(0.0, top - 16.0)))
        else:
            cx = (left + right) * 0.5
            line_item = QGraphicsLineItem(QLineF(cx, 0, cx, img_h))
            equation_text = f"x = {cx:.1f}"
            label_pos = QPointF(float(max(0.0, left - 40.0)), float(top))

        line_item.setZValue(6)
        scene.addItem(line_item)
        overlays.append(line_item)

        # label
        label_item = QGraphicsSimpleTextItem(equation_text)
        label_item.setBrush(qcolor_rgba(*stroke_rgb, 255))
        font = QFont()
        font.setPointSize(10)
        label_item.setFont(font)
        label_item.setPos(label_pos)
        label_item.setZValue(12)
        scene.addItem(label_item)
        overlays.append(label_item)

        item.set_centerline_and_label(line_item, label_item)
        return item

    # Draw blue: horizontal bands near vertical bands
    blue_items: List[BandRectItem] = []
    for rect in blue_rects:
        blue_items.append(add_band_rect(rect, "horizontal", (0, 102, 255)))  # blue

    # Draw red: vertical bands near horizontal bands
    red_items: List[BandRectItem] = []
    for rect in red_rects:
        red_items.append(add_band_rect(rect, "vertical", (255, 0, 0)))       # red

    # ───────────── Auto-select per half (top/bottom) on H bands only ─────────────
    # Split by white band: top half = cy < y_top_white ; bottom half = cy > y_bot_white
    # If no white band found, skip auto-selection.
    if y_top_white is not None and y_bot_white is not None:
        # Partition blue (horizontal) items by half
        top_half: List[Tuple[BandRectItem, float]] = []     # (item, cy)
        bottom_half: List[Tuple[BandRectItem, float]] = []

        for it in blue_items:
            r = it.rect()
            cy = r.center().y()
            # Ignore items whose center lies inside the white band itself
            if y_top_white <= cy <= y_bot_white:
                continue
            if cy < y_top_white:
                top_half.append((it, cy))
            elif cy > y_bot_white:
                bottom_half.append((it, cy))

        def select_min_max(items_with_cy: List[Tuple[BandRectItem, float]]):
            if not items_with_cy:
                return
            # lowest y (min cy) and highest y (max cy)
            min_item = min(items_with_cy, key=lambda t: t[1])[0]
            max_item = max(items_with_cy, key=lambda t: t[1])[0]
            min_item.set_selected_state(True)
            if max_item is not min_item:
                max_item.set_selected_state(True)

        select_min_max(top_half)
        select_min_max(bottom_half)

        # Optional: draw the white band area for context
        wb_rect = QRectF(0.0, float(y_top_white), float(img_w), float(y_bot_white - y_top_white))
        wb_item = QGraphicsRectItem(wb_rect)
        wb_item.setBrush(QBrush(qcolor_rgba(255, 255, 255, 40)))
        wb_item.setPen(QPen(qcolor_rgba(255, 255, 255, 80), 1.0, Qt.DashLine))
        wb_item.setZValue(4)
        scene.addItem(wb_item)
        overlays.append(wb_item)

    # ───────────── NEW: Auto-select vertical bands near global MAX bottom Y ─────────────
    if red_items:
        # Compute each vertical item's bottom Y (max of its rect's y coordinates)
        bottoms: List[Tuple[BandRectItem, float]] = []
        for it in red_items:
            r = it.rect()
            bottom_y = r.bottom()  # since rect is ordered top<=bottom, this is max Y (lower on page)
            bottoms.append((it, bottom_y))

        max_bottom = max(b for _, b in bottoms)
        delta = float(args.v_maxy_delta)
        # Preselect items whose bottom is within delta of the global max
        for it, btm in bottoms:
            if (max_bottom - btm) <= delta:
                it.set_selected_state(True)

    # View
    view = InteractiveView(scene)
    view.register_overlays(overlays)
    view.setWindowTitle("Bands (UExtractor): click to show centerline; hold 'H' to hide overlays")

    # Size ~80% of screen and center
    screen_geo = QGuiApplication.primaryScreen().availableGeometry()
    target_w = int(screen_geo.width() * 0.8)
    target_h = int(screen_geo.height() * 0.8)
    view.resize(target_w, target_h)
    view.move(
        screen_geo.x() + (screen_geo.width() - target_w) // 2,
        screen_geo.y() + (screen_geo.height() - target_h) // 2
    )

    view.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # Example:
    # python -m src.frontend "images/drawings_pdf/1 - 23H.pdf" --page 0
    main()
