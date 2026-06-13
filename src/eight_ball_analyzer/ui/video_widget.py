from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy


class VideoWidget(QLabel):
    # Emits (x, y) coordinates of the click on the original image
    image_clicked = pyqtSignal(int, int)

    def __init__(self) -> None:
        super().__init__("Open a practice video to begin")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(720, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#101418;color:#cbd5df;border:1px solid #26323d;")
        self._last_frame: np.ndarray | None = None
        self._rgb_buffer: np.ndarray | None = None

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        self._last_frame = frame_bgr.copy()
        self._render()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()

    def mousePressEvent(self, event) -> None:
        if self._last_frame is None or self.pixmap() is None:
            return

        img_h, img_w = self._last_frame.shape[:2]
        pixmap_size = self.pixmap().size()
        widget_size = self.size()
        
        offset_x = (widget_size.width() - pixmap_size.width()) / 2
        offset_y = (widget_size.height() - pixmap_size.height()) / 2
        
        x = event.pos().x() - offset_x
        y = event.pos().y() - offset_y
        
        if 0 <= x < pixmap_size.width() and 0 <= y < pixmap_size.height():
            orig_x = int(x * (img_w / pixmap_size.width()))
            orig_y = int(y * (img_h / pixmap_size.height()))
            self.image_clicked.emit(orig_x, orig_y)

    def _render(self) -> None:
        if self._last_frame is None:
            return
        if self._last_frame.ndim != 3 or self._last_frame.shape[2] != 3:
            return
        self._rgb_buffer = cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2RGB)
        height, width, channels = self._rgb_buffer.shape
        qimage = QImage(
            self._rgb_buffer.data, width, height, channels * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)
