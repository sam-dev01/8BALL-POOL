import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .models import Pocket, Rail, TableDetection
from .geometry import unit_vector

class CalibrationManager:
    """Manages manual table calibration using a JSON file."""

    def __init__(self, filepath: str = "calibration.json"):
        self.filepath = Path(filepath)
        self.corners: list[tuple[float, float]] = []

    def is_calibrated(self) -> bool:
        """Returns True if the calibration file exists and contains valid data."""
        if not self.filepath.exists():
            return False
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
            if all(k in data for k in ("tl", "tr", "br", "bl")):
                return True
        except Exception:
            pass
        return False

    def load_corners(self) -> np.ndarray | None:
        """Load the 4 corners from JSON into a numpy array."""
        if not self.is_calibrated():
            return None
        with open(self.filepath, "r") as f:
            data = json.load(f)
        return np.array([data["tl"], data["tr"], data["br"], data["bl"]], dtype=float)

    def save_corners(self, corners: list[tuple[float, float]]) -> None:
        """Save the 4 corners to JSON."""
        if len(corners) != 4:
            raise ValueError("Exactly 4 corners required for calibration.")
        
        # Sort them generally to ensure order (TL, TR, BR, BL)
        # Using the same sorting logic as detection.py:
        pts = np.array(corners, dtype=float).reshape(-1, 2)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]

        data = {
            "tl": tl.tolist(),
            "tr": tr.tolist(),
            "br": br.tolist(),
            "bl": bl.tolist(),
        }
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    def delete_calibration(self) -> None:
        """Delete the calibration file."""
        if self.filepath.exists():
            os.remove(self.filepath)

    def get_table_and_pockets(self) -> tuple[TableDetection | None, list[Pocket]]:
        """Constructs TableDetection and Pockets purely from the saved corners."""
        corners = self.load_corners()
        if corners is None:
            return None, []

        tl, tr, br, bl = corners

        # Calculate bounding box (x, y, w, h)
        xs = corners[:, 0]
        ys = corners[:, 1]
        x, y = int(np.min(xs)), int(np.min(ys))
        w, h = int(np.max(xs) - x), int(np.max(ys) - y)

        def _inward_normal(start: np.ndarray, end: np.ndarray, centre: np.ndarray) -> np.ndarray:
            seg = end - start
            n = np.array([-seg[1], seg[0]], dtype=float)
            n /= max(float(np.linalg.norm(n)), 1e-6)
            mid = (start + end) / 2.0
            if float(np.dot(n, centre - mid)) < 0:
                n = -n
            return n

        centre = (tl + tr + br + bl) / 4.0

        rails = [
            Rail(id=0, start=tl.copy(), end=tr.copy(), normal=_inward_normal(tl, tr, centre)),
            Rail(id=1, start=tr.copy(), end=br.copy(), normal=_inward_normal(tr, br, centre)),
            Rail(id=2, start=br.copy(), end=bl.copy(), normal=_inward_normal(br, bl, centre)),
            Rail(id=3, start=bl.copy(), end=tl.copy(), normal=_inward_normal(bl, tl, centre)),
        ]

        # Dimensions for perspective transform
        top_w = float(np.linalg.norm(tr - tl))
        bottom_w = float(np.linalg.norm(br - bl))
        left_h = float(np.linalg.norm(bl - tl))
        right_h = float(np.linalg.norm(br - tr))
        width_px = max((top_w + bottom_w) / 2.0, 1.0)
        height_px = max((left_h + right_h) / 2.0, 1.0)

        # Perspective transform matrices (same as detector logic)
        dst_pts = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ], dtype=np.float32)
        transform_matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_pts)
        inv_transform_matrix = cv2.getPerspectiveTransform(dst_pts, corners.astype(np.float32))

        table = TableDetection(
            polygon=corners.astype(np.int32),
            bounds=(x, y, w, h),
            confidence=1.0,
            corners=corners,
            width_px=width_px,
            height_px=height_px,
            transform_matrix=transform_matrix,
            inv_transform_matrix=inv_transform_matrix,
            rails=rails
        )

        # Generate Pockets mathematically
        tm = (tl + tr) / 2.0
        bm = (bl + br) / 2.0

        pockets = [
            Pocket(id=0, center=tl, mouth_radius=width_px * 0.05, label="TL"),
            Pocket(id=1, center=tm, mouth_radius=width_px * 0.04, label="TM"),
            Pocket(id=2, center=tr, mouth_radius=width_px * 0.05, label="TR"),
            Pocket(id=3, center=bl, mouth_radius=width_px * 0.05, label="BL"),
            Pocket(id=4, center=bm, mouth_radius=width_px * 0.04, label="BM"),
            Pocket(id=5, center=br, mouth_radius=width_px * 0.05, label="BR"),
        ]

        return table, pockets
