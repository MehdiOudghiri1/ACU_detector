# -*- coding: utf-8 -*-
# pip install pymupdf PySide6
import sys
from typing import Optional, List

import fitz  # PyMuPDF
from PySide6.QtCore import Qt, QRectF, QLineF
from PySide6.QtGui import QImage, QPixmap, QBrush, QPen, QColor, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QGraphicsScene, QGraphicsView,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsLineItem
)


def qimage_from_pdf(pdf_path: str, page_index: int = 0, zoom: float = 2.0) -> QImage:
    """Render a PDF page to QImage (RGB)."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
    qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
    qimg = qimg.copy()
    doc.close()
    return qimg


def lighter(color: QColor, alpha: int = 90) -> QColor:
    c = QColor(color)
    c.setAlpha(alpha)
    return c


def solid(color: QColor, alpha: int = 255) -> QColor:
    c = QColor(color)
    c.setAlpha(alpha)
    return c


class ClickableRect(QGraphicsRectItem):
    """Rectangle cliquable : toggle fill, épaissit/renforce la ligne associée."""
    def __init__(self, rect: QRectF, base_color: QColor, fill_alpha: int = 90):
        super().__init__(rect)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)

        self.base_color = QColor(base_color)
        self.fill_alpha = int(fill_alpha)
        self.selected = False
        self.line: Optional[QGraphicsLineItem] = None
        self.line_pen_normal = None
        self.line_pen_selected = None

        # Pen du rectangle (contour)
        self._base_pen_width = 2.0
        pen = QPen(solid(self.base_color), self._base_pen_width)
        self.setPen(pen)
        self.setBrush(Qt.NoBrush)

    def set_associated_line(self, line_item: QGraphicsLineItem, light_width: float = 1.5, strong_width: float = 3.0):
        self.line = line_item
        light_pen = QPen(lighter(self.base_color, alpha=90), light_width)
        strong_pen = QPen(solid(self.base_color, alpha=255), strong_width)
        self.line_pen_normal = light_pen
        self.line_pen_selected = strong_pen
        self.line.setPen(self.line_pen_normal)

    def mousePressEvent(self, event):
        self.selected = not self.selected
        if self.selected:
            c = solid(self.base_color, alpha=self.fill_alpha)
            self.setBrush(QBrush(c))
            if self.line is not None:
                self.line.setPen(self.line_pen_selected)
        else:
            self.setBrush(Qt.NoBrush)
            if self.line is not None:
                self.line.setPen(self.line_pen_normal)

        sp = event.scenePos()
        r, g, b, a = self.base_color.red(), self.base_color.green(), self.base_color.blue(), self.base_color.alpha()
        print(f"Clicked at (scene): x={sp.x():.1f}, y={sp.y():.1f}; color=rgba({r},{g},{b},{a})")

        super().mousePressEvent(event)

    def hoverEnterEvent(self, event):
        pen = QPen(solid(self.base_color), self._base_pen_width + 1.0)
        self.setPen(pen)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        pen = QPen(solid(self.base_color), self._base_pen_width)
        self.setPen(pen)
        super().hoverLeaveEvent(event)


class InteractiveView(QGraphicsView):
    """Vue : quand on maintient 'd', les overlays disparaissent. Relâcher -> retour."""
    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self._overlay_items: List[QGraphicsLineItem | QGraphicsRectItem] = []
        self.setFocusPolicy(Qt.StrongFocus)

    def set_overlay_items(self, items: List[QGraphicsLineItem | QGraphicsRectItem]):
        self._overlay_items = items

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_D:
            for it in self._overlay_items:
                it.setVisible(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_D:
            for it in self._overlay_items:
                it.setVisible(True)
            event.accept()
            return
        super().keyReleaseEvent(event)


def main(pdf_path: str = "exemple.pdf", page_index: int = 0):
    app = QApplication(sys.argv)

    # 1) PDF -> image de fond
    img = qimage_from_pdf(pdf_path, page_index, zoom=2.0)
    pix = QPixmap.fromImage(img)
    img_w, img_h = pix.width(), pix.height()

    # 2) Scène + fond
    scene = QGraphicsScene()
    bg = QGraphicsPixmapItem(pix)
    bg.setZValue(0)
    scene.addItem(bg)

    # 3) Rectangles + lignes
    rects_specs = [
        (QRectF(100, 120, 180, 80), QColor(0, 180, 0),          "vertical"),
        (QRectF(420, 220, 160, 90), QColor(30, 144, 255),       "horizontal"),
        (QRectF(240, 420, 220, 110), QColor(255, 140, 0),       "vertical"),
        (QRectF(600, 140, 190, 120), QColor(138, 43, 226),      "horizontal"),
    ]

    rect_items: list[ClickableRect] = []
    line_items: list[QGraphicsLineItem] = []

    for rectf, color, mode in rects_specs:
        ritem = ClickableRect(rectf, base_color=color, fill_alpha=100)
        ritem.setZValue(10)
        scene.addItem(ritem)
        rect_items.append(ritem)

        cx = rectf.x() + rectf.width() * 0.5
        cy = rectf.y() + rectf.height() * 0.5

        if mode == "vertical":
            line = QGraphicsLineItem(QLineF(cx, 0, cx, img_h))
        else:
            line = QGraphicsLineItem(QLineF(0, cy, img_w, cy))

        line.setZValue(5)
        scene.addItem(line)
        line_items.append(line)
        ritem.set_associated_line(line, light_width=1.5, strong_width=3.0)

    # 4) Vue personnalisée
    view = InteractiveView(scene)
    view.set_overlay_items(rect_items + line_items)
    view.setRenderHints(view.renderHints())
    view.setDragMode(QGraphicsView.ScrollHandDrag)
    view.setWindowTitle("Hold 'd' to hide overlays")

    # 5) Taille = 70% écran, centré
    screen_geo = QGuiApplication.primaryScreen().availableGeometry()
    target_w = int(screen_geo.width() * 0.7)
    target_h = int(screen_geo.height() * 0.7)
    view.resize(target_w, target_h)
    view.move(
        screen_geo.x() + (screen_geo.width() - target_w) // 2,
        screen_geo.y() + (screen_geo.height() - target_h) // 2
    )

    view.show()
    sys.exit(app.exec())



if __name__ == "__main__":
    # Remplace "exemple.pdf" par ton chemin
    main("images/drawings_pdf/1 - 23H.pdf", page_index=0)
