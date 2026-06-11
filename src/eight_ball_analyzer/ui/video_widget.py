from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy


class VideoWidget(QLabel):
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
