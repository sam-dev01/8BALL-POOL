from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .geometry import unit_vector
from .models import BallDetection, BallKind, Pocket, Rail, TableDetection


YOLO_CLASS_MAP = {
    "ball": BallKind.UNKNOWN,
    "cue_ball": BallKind.CUE,
    "cue": BallKind.CUE,
    "solid_ball": BallKind.SOLID,
    "solid": BallKind.SOLID,
    "stripe_ball": BallKind.STRIPE,
    "stripe": BallKind.STRIPE,
    "eight_ball": BallKind.EIGHT,
    "8_ball": BallKind.EIGHT,
    "eight": BallKind.EIGHT,
}

# Valid aspect ratio range for a pool table (width / height)
_TABLE_ASPECT_MIN = 1.6
_TABLE_ASPECT_MAX = 2.8


def _default_yolo_model_path() -> Path | None:
    candidates = [
        os.environ.get("EIGHT_BALL_YOLO_MODEL"),
        "models/8ball_yolov8.pt",
        "models/8ball_yolov8.onnx",
        "models/best.pt",
        "models/best.onnx",
    ]
    root = Path(__file__).resolve().parents[2]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class PoolDetector:
    def __init__(
        self,
        yolo_model_path: str | Path | None = None,
        yolo_confidence: float = 0.45,
    ) -> None:
        self.yolo_confidence = yolo_confidence
        self.last_corners: np.ndarray | None = None
        self._yolo_model = None
        self._yolo_model_path = (
            Path(yolo_model_path)
            if yolo_model_path is not None
            else _default_yolo_model_path()
        )
        self._yolo_load_attempted = False

    @property
    def has_yolo_model(self) -> bool:
        return self._yolo_model is not None

    def _load_yolo_model(self) -> None:
        self._yolo_load_attempted = True
        try:
            from ultralytics import YOLO
        except ImportError:
            self._yolo_model = None
            return
        try:
            self._yolo_model = YOLO(str(self._yolo_model_path))
        except Exception:
            self._yolo_model = None

    # ------------------------------------------------------------------
    # STAGE 5 — BALL DETECTION (only after table geometry is locked)
    # ------------------------------------------------------------------

    def detect_balls(
        self,
        frame: np.ndarray,
        table: TableDetection | None,
        pockets: list[Pocket] | None = None,
    ) -> list[BallDetection]:
        """Detect balls strictly inside the playable table region."""
        if table is None:
            return []

        if self._yolo_model_path is not None and not self._yolo_load_attempted:
            self._load_yolo_model()

        yolo_balls: list[BallDetection] = []
        if self.has_yolo_model:
            yolo_balls = self._detect_balls_yolo(frame, table)

        x, y, w, h = table.bounds
        # Tighter inset — ignore cushion strips and UI around the table edge
        pad = int(max(10, min(w, h) * 0.06))
        roi_x = max(0, x + pad)
        roi_y = max(0, y + pad)
        roi_x2 = min(frame.shape[1], x + w - pad)
        roi_y2 = min(frame.shape[0], y + h - pad)
        roi = frame[roi_y:roi_y2, roi_x:roi_x2]
        if roi.size == 0:
            return []

        cloth_color = np.median(roi.reshape(-1, 3), axis=0)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        min_radius = max(3, int(min(w, h) * 0.009))
        max_radius = max(min_radius + 3, int(min(w, h) * 0.058))
        if pockets is None:
            pockets = []

        circles = sorted(
            self._merge_circles(
                self._detect_circles_multi_pass(gray, min_radius, max_radius)
                + self._detect_balls_by_color_blobs(roi, cloth_color, min_radius, max_radius)
                + self._detect_white_balls(roi, gray, min_radius, max_radius)
                + self._detect_bright_cue_peaks(roi, gray, min_radius, max_radius),
                merge_distance=max_radius * 0.75,
            ),
            key=lambda item: item[3],
            reverse=True,
        )

        balls: list[BallDetection] = []
        ui_cutoff_y = int(roi.shape[0] * 0.10)
        for idx, (cx, cy, radius, base_score) in enumerate(circles[:80]):
            if cy < ui_cutoff_y:
                continue
            if radius < min_radius or radius > max_radius:
                continue
            fx = cx + roi_x
            fy = cy + roi_y
            # Strict inside-table check using the perspective transform
            if not self._inside_table(np.array([float(fx), float(fy)]), table):
                continue
            center_gray = float(gray[cy, cx]) if 0 <= cy < gray.shape[0] and 0 <= cx < gray.shape[1] else 0.0
            if center_gray > 205:
                radius = self._refine_bright_ball_radius(gray, cx, cy, radius, min_radius, max_radius)
            if center_gray > 205 and fx > x + w - int(w * 0.11):
                continue
            color = self._sample_color(frame, fx, fy, radius)
            brightness = float(sum(color)) / 3.0
            kind_hint = classify_ball(color)
            is_likely_cue = kind_hint == BallKind.CUE or brightness > 175 or center_gray > 205
            circularity, contrast = self._circle_quality(gray, cx, cy, radius)
            confidence = self._circle_confidence(circularity, contrast, base_score)
            if radius <= 10:
                min_circ, min_conf = 0.38, 0.48
            elif is_likely_cue:
                min_circ, min_conf = 0.48, 0.58
            else:
                min_circ, min_conf = 0.42, 0.52
            if circularity < min_circ or confidence < min_conf:
                continue
            if self._looks_like_cloth_or_pocket(color, cloth_color):
                continue
            
            # Reject hollow logo letters (like 'P' or 'O') where the center pixel is the cloth color
            center_color = roi[cy, cx].astype(float) if 0 <= cy < roi.shape[0] and 0 <= cx < roi.shape[1] else np.zeros(3)
            if np.linalg.norm(center_color - cloth_color.astype(float)) < 25.0:
                continue

            if self._near_pocket(np.array([float(fx), float(fy)]), pockets, radius):
                # Pockets are dark. If the object is dark, it's likely the pocket itself.
                # If it's bright/colored, it's a real ball near the pocket.
                if brightness < 65:
                    continue
            if not is_likely_cue and self._looks_like_table_marking(roi, cx, cy, radius, cloth_color):
                continue
            balls.append(
                BallDetection(
                    id=idx,
                    center=np.array([float(fx), float(fy)]),
                    radius=float(radius),
                    kind=classify_ball(color),
                    confidence=confidence,
                    color_bgr=color,
                )
            )
            if len(balls) >= 16:
                break

        balls = self._merge_ball_detections(yolo_balls + balls)
        balls = self._enforce_ball_inventory(balls)
        balls.sort(key=lambda b: (0 if b.kind == BallKind.CUE else 1, b.center[1], b.center[0]))
        for new_id, ball in enumerate(balls):
            ball.id = new_id
        return balls

    def _inside_table(self, point: np.ndarray, table: TableDetection) -> bool:
        """Check if a point lies strictly inside the table's playable cloth polygon."""
        dist = cv2.pointPolygonTest(table.polygon, (float(point[0]), float(point[1])), measureDist=True)
        # Require the ball center to be at least 2% of the table size inside the cloth
        pad = max(2.0, min(table.width_px, table.height_px) * 0.008)
        return dist >= pad

    # ------------------------------------------------------------------
    # Cue direction — reads game aim guideline (Cheto-style) then cue stick
    # ------------------------------------------------------------------

    def detect_cue_direction(
        self,
        frame: np.ndarray,
        table: TableDetection | None,
        cue_ball: BallDetection | None,
        balls: list[BallDetection],
    ) -> np.ndarray | None:
        if table is None or cue_ball is None:
            return None

        # Priority 1: dotted multicolor extended guideline (Cheto / 8BP hack style)
        guideline = self._detect_guideline_by_dot_voting(frame, table, cue_ball)
        if guideline is not None:
            return guideline

        # Priority 2: solid white/cyan guideline segments
        guideline = self._detect_game_guideline(frame, table, cue_ball)
        if guideline is not None:
            return guideline

        # Priority 3: physical cue stick edge
        stick = self._detect_cue_stick_direction(frame, table, cue_ball, balls)
        if stick is not None:
            return stick

        # Priority 4: aim at nearest object ball (open table fallback)
        return self._fallback_direction_from_balls(cue_ball, balls)

    def _guideline_color_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Mask pixels that belong to the in-game aim guideline overlay."""
        white   = cv2.inRange(hsv, np.array([0,   0,   175]), np.array([180,  70, 255]))
        cyan    = cv2.inRange(hsv, np.array([75,  35,  140]), np.array([110, 255, 255]))
        yellow  = cv2.inRange(hsv, np.array([15,  40,  150]), np.array([48,  255, 255]))
        magenta = cv2.inRange(hsv, np.array([130, 40,  120]), np.array([175, 255, 255]))
        pink    = cv2.inRange(hsv, np.array([160, 25,  140]), np.array([179, 255, 255]))
        # BUG 6 FIX: add orange range (missing from original mask)
        orange  = cv2.inRange(hsv, np.array([5,  120,  150]), np.array([18,  255, 255]))
        mask = cv2.bitwise_or(
            white,
            cv2.bitwise_or(
                cyan,
                cv2.bitwise_or(
                    yellow,
                    cv2.bitwise_or(
                        magenta,
                        cv2.bitwise_or(pink, orange),
                    ),
                ),
            ),
        )
        cloth = cv2.inRange(hsv, np.array([32, 35, 35]), np.array([98, 255, 255]))
        return cv2.bitwise_and(mask, cv2.bitwise_not(cloth))

    def _detect_guideline_by_dot_voting(
        self,
        frame: np.ndarray,
        table: TableDetection,
        cue_ball: BallDetection,
    ) -> np.ndarray | None:
        """Read aim direction from dotted pink/cyan/white guideline pixels.

        This is how Cheto-style tools detect the game's built-in aim overlay.
        """
        x, y, w, h = table.bounds
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = self._guideline_color_mask(hsv)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)

        ys, xs = np.where(mask > 0)
        if len(xs) < 6:
            return None

        cue = cue_ball.center.astype(float)
        min_dist = max(cue_ball.radius * 1.2, 6.0)
        max_dist = float(min(w, h) * 0.85)
        vectors: list[np.ndarray] = []

        for col, row in zip(xs, ys):
            px = float(col + x)
            py = float(row + y)
            delta = np.array([px, py]) - cue
            dist = float(np.linalg.norm(delta))
            if dist < min_dist or dist > max_dist:
                continue
            vectors.append(delta / dist)

        if len(vectors) < 4:
            return None

        # Dominant direction via vector sum (robust for dotted lines)
        avg = np.mean(np.array(vectors), axis=0)
        norm = float(np.linalg.norm(avg))
        if norm < 0.35:
            return None
        return avg / norm

    def find_cue_ball_on_table(
        self,
        frame: np.ndarray,
        table: TableDetection,
        existing_balls: list[BallDetection],
    ) -> BallDetection | None:
        """Find white cue ball even when circle detection missed it."""
        x, y, w, h = table.bounds
        pad = int(max(6, min(w, h) * 0.04))
        roi_x, roi_y = x + pad, y + pad
        roi = frame[roi_y: y + h - pad, roi_x: x + w - pad]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 168]), np.array([180, 80, 255]))
        _, bright = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(white, bright)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: BallDetection | None = None
        best_score = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            fx, fy = cx + roi_x, cy + roi_y
            center = np.array([fx, fy], dtype=float)
            if not self._inside_table(center, table):
                continue
            r = float(radius)
            if r < 3 or r > min(w, h) * 0.06:
                continue
            color = self._sample_color(frame, int(fx), int(fy), int(max(r, 4)))
            brightness = sum(color) / 3.0
            if brightness < 150:
                continue
            # Prefer cue balls not already matched to a colored ball
            if existing_balls:
                nearest = min(existing_balls, key=lambda b: float(np.linalg.norm(b.center - center)))
                if float(np.linalg.norm(nearest.center - center)) < max(r, nearest.radius) * 0.9:
                    if nearest.kind != BallKind.CUE and brightness < 200:
                        continue
            score = brightness + r * 2.0
            if score > best_score:
                best_score = score
                best = BallDetection(
                    id=99,
                    center=center,
                    radius=r,
                    kind=BallKind.CUE,
                    confidence=0.75,
                    color_bgr=color,
                )
        return best

    def _detect_game_guideline(
        self,
        frame: np.ndarray,
        table: TableDetection,
        cue_ball: BallDetection,
    ) -> np.ndarray | None:
        """Detect 8 Ball Pool's built-in aim line and return shot direction.

        Mobile game aim tools work by reading this line, then extending it
        with physics.  We mask bright guideline pixels (white / cyan / yellow)
        and cluster short Hough segments near the cue ball.
        """
        x, y, w, h = table.bounds
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = self._guideline_color_mask(hsv)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        blurred = cv2.GaussianBlur(mask, (3, 3), 0)
        edges = cv2.Canny(blurred, 40, 120)
        min_seg = max(8, int(min(w, h) * 0.025))
        max_gap = max(12, int(min(w, h) * 0.04))
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=18,
            minLineLength=min_seg, maxLineGap=max_gap,
        )
        if lines is None:
            return None

        cue = cue_ball.center.astype(float)
        cue_roi = cue - np.array([float(x), float(y)])
        best_score = 0.0
        best_direction: np.ndarray | None = None

        for line in lines[:, 0, :]:
            p1 = np.array([float(line[0]), float(line[1])])
            p2 = np.array([float(line[2]), float(line[3])])
            length = float(np.linalg.norm(p2 - p1))
            if length < min_seg:
                continue

            # Segment must pass close to cue ball
            cue_global = cue
            p1g = p1 + np.array([float(x), float(y)])
            p2g = p2 + np.array([float(x), float(y)])
            dist = self._point_to_segment_distance(cue_global, p1g, p2g)
            if dist > max(cue_ball.radius * 3.5, 28.0):
                continue

            # Direction: from cue toward the far endpoint of the segment
            d1 = float(np.linalg.norm(p1g - cue_global))
            d2 = float(np.linalg.norm(p2g - cue_global))
            far = p1g if d1 > d2 else p2g
            direction = far - cue_global
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            direction = direction / norm

            # Sample guideline-colored pixels along the segment
            samples = 0
            hits = 0
            for t in np.linspace(0.0, 1.0, 8):
                pt = p1 * (1.0 - t) + p2 * t
                ix, iy = int(round(pt[0])), int(round(pt[1]))
                if 0 <= ix < roi.shape[1] and 0 <= iy < roi.shape[0]:
                    samples += 1
                    if mask[iy, ix] > 0:
                        hits += 1
            color_ratio = hits / max(samples, 1)

            # Prefer segments aligned with cue ball and brightly colored
            to_seg = unit_vector(cue_roi, (p1 + p2) / 2.0)
            seg_dir = (p2 - p1) / max(length, 1e-6)
            alignment = abs(float(np.dot(to_seg, seg_dir)))
            score = length * (0.5 + color_ratio) * (0.6 + alignment) / (1.0 + dist * 0.05)

            if score > best_score:
                best_score = score
                best_direction = direction

        return best_direction

    def _detect_cue_stick_direction(
        self,
        frame: np.ndarray,
        table: TableDetection | None,
        cue_ball: BallDetection | None,
        balls: list[BallDetection],
    ) -> np.ndarray | None:
        if table is None or cue_ball is None:
            return None

        x, y, w, h = table.bounds
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 160)
        min_len = max(60, int(min(w, h) * 0.18))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, minLineLength=min_len, maxLineGap=12)
        if lines is None:
            return None

        cue = cue_ball.center.astype(float)
        best_score = 0.0
        best_direction = None
        for line in lines[:, 0, :]:
            p1 = np.array([float(line[0] + x), float(line[1] + y)])
            p2 = np.array([float(line[2] + x), float(line[3] + y)])
            length = float(np.linalg.norm(p2 - p1))
            if length < min_len:
                continue
            line_distance = self._point_to_segment_distance(cue, p1, p2)
            if line_distance > max(cue_ball.radius * 2.4, 18.0):
                continue
            if self._line_overlaps_many_balls(p1, p2, balls):
                continue
            far = p1 if np.linalg.norm(p1 - cue) > np.linalg.norm(p2 - cue) else p2
            direction = cue - far
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            score = length / (1.0 + line_distance)
            if score > best_score:
                best_score = score
                best_direction = direction / norm

        return best_direction

    def detect_power(self, frame: np.ndarray, table: TableDetection | None) -> float:
        if table is None:
            return 0.0
        x, y, w, h = table.bounds
        left_roi  = frame[max(0, y):min(frame.shape[0], y + h), max(0, x - int(w * 0.25)):max(0, x)]
        right_roi = frame[max(0, y):min(frame.shape[0], y + h), min(frame.shape[1], x + w):min(frame.shape[1], x + w + int(w * 0.25))]
        roi = left_roi if left_roi.size >= right_roi.size else right_roi
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        power_mask = cv2.inRange(hsv, np.array([5, 60, 80]), np.array([40, 255, 255]))
        contours, _ = cv2.findContours(power_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        contour = max(contours, key=cv2.contourArea)
        px, py, pw, ph = cv2.boundingRect(contour)
        if ph < max(20, h * 0.1):
            return 0.0
        filled = cv2.countNonZero(power_mask[py:py + ph, px:px + pw]) / float(max(1, pw * ph))
        return float(np.clip(filled * 100.0, 0.0, 100.0))

    # ------------------------------------------------------------------
    # YOLO ball detection
    # ------------------------------------------------------------------

    def _detect_balls_yolo(self, frame: np.ndarray, table: TableDetection) -> list[BallDetection]:
        if self._yolo_model is None:
            return []
        try:
            results = self._yolo_model.predict(frame, conf=self.yolo_confidence, iou=0.45, verbose=False)
        except Exception:
            return []
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {}) or getattr(self._yolo_model, "names", {}) or {}
        if boxes is None or len(boxes) == 0:
            return []

        x, y, w, h = table.bounds
        min_radius = max(3.0, min(w, h) * 0.009)
        max_radius = max(min_radius + 3.0, min(w, h) * 0.058)
        candidates: list[BallDetection] = []

        for index, box in enumerate(boxes):
            try:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                confidence = float(box.conf[0])
                class_id   = int(box.cls[0])
            except Exception:
                continue
            class_name = str(names.get(class_id, "")).lower().strip()
            kind = YOLO_CLASS_MAP.get(class_name)
            if kind is None:
                continue
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            radius = (bw + bh) / 4.0
            if radius < min_radius or radius > max_radius:
                continue
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
            if not self._inside_table(center, table):
                continue
            color = self._sample_color(frame, int(center[0]), int(center[1]), int(radius))
            
            # If YOLO just detected a generic 'ball', we must classify it by color
            if kind == BallKind.UNKNOWN:
                kind = classify_ball(color)
                
            candidates.append(
                BallDetection(id=index, center=center, radius=float(radius),
                              kind=kind, confidence=confidence, color_bgr=color)
            )

        candidates = self._merge_ball_detections(candidates)
        candidates = self._enforce_ball_inventory(candidates)
        for new_id, ball in enumerate(candidates):
            ball.id = new_id
        return candidates

    # ------------------------------------------------------------------
    # Circle / blob helper methods (unchanged from original)
    # ------------------------------------------------------------------

    def _detect_circles_multi_pass(self, gray, min_radius, max_radius):
        circles: list[tuple[int, int, int, float]] = []
        variants = [
            cv2.medianBlur(gray, 5),
            cv2.GaussianBlur(gray, (5, 5), 0),
            cv2.equalizeHist(cv2.medianBlur(gray, 5)),
        ]
        passes = [(1.10, 55, 10, 0.72), (1.20, 70, 12, 0.78), (1.20, 80, 15, 0.84), (1.35, 95, 18, 0.92)]
        for image in variants:
            for dp, param1, param2, score in passes:
                found = cv2.HoughCircles(
                    image, cv2.HOUGH_GRADIENT,
                    dp=dp, minDist=max(min_radius * 2, int(max_radius * 1.45)),
                    param1=param1, param2=param2,
                    minRadius=min_radius, maxRadius=max_radius,
                )
                if found is None:
                    continue
                for cx, cy, r in np.round(found[0, :]).astype(int):
                    circles.append((int(cx), int(cy), int(r), score))
        return circles

    def _refine_bright_ball_radius(self, gray, cx, cy, radius, min_radius, max_radius):
        best_radius, best_score = radius, -1.0
        for candidate in range(max(min_radius, radius - 6), min(max_radius, radius + 2) + 1):
            circ, contrast = self._circle_quality(gray, cx, cy, candidate)
            score = self._circle_confidence(circ, contrast, 0.78)
            if score > best_score:
                best_score = score
                best_radius = candidate
        return best_radius

    def _detect_white_balls(self, roi, gray, min_radius, max_radius):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, bright = cv2.threshold(blurred, 168, 255, cv2.THRESH_BINARY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_hsv = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([180, 70, 255]))
        bright = cv2.bitwise_or(bright, white_hsv)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,  kernel, iterations=1)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circles: list[tuple[int, int, int, float]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 1.0:
                continue
            circ = 4.0 * np.pi * area / (perimeter * perimeter)
            if circ < 0.35:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(contour)
            ri = int(round(r))
            if ri < min_radius or ri > max_radius:
                continue
            if area < np.pi * (ri * 0.38) ** 2:
                continue
            circles.append((int(round(cx)), int(round(cy)), ri, float(np.clip(0.62 + circ * 0.30, 0.62, 0.90))))
        return circles

    def _detect_bright_cue_peaks(self, roi, gray, min_radius, max_radius):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles: list[tuple[int, int, int, float]] = []
        for kernel_size, min_brightness in ((11, 205), (21, 190)):
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            local_max = cv2.dilate(blurred, kernel)
            peak_mask = ((blurred >= local_max) & (blurred >= min_brightness)).astype(np.uint8) * 255
            contours, _ = cv2.findContours(peak_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 8:
                    continue
                m = cv2.moments(contour)
                if m["m00"] < 1:
                    continue
                cx = int(round(m["m10"] / m["m00"]))
                cy = int(round(m["m01"] / m["m00"]))
                if cx < 0 or cy < 0 or cx >= roi.shape[1] or cy >= roi.shape[0]:
                    continue
                if float(blurred[cy, cx]) < min_brightness - 5:
                    continue
                r = int(np.clip(round(np.sqrt(cv2.contourArea(contour) / np.pi)), min_radius, max_radius))
                best_r, best_s = r, 0.0
                for cr in range(max(min_radius, r - 2), min(max_radius, r + 3) + 1):
                    circ, contrast = self._circle_quality(gray, cx, cy, cr)
                    s = self._circle_confidence(circ, contrast, 0.74)
                    if s > best_s:
                        best_s, best_r = s, cr
                circles.append((cx, cy, best_r, max(0.72, best_s)))
        return circles

    def _detect_balls_by_color_blobs(self, roi, cloth_color, min_radius, max_radius):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_diff = np.linalg.norm(roi.astype(np.float32) - cloth_color.reshape(1, 1, 3), axis=2)
        color_mask = (color_diff > 20.0).astype(np.uint8) * 255
        bright_mask    = cv2.inRange(gray, 185, 255)
        saturated_mask = cv2.inRange(hsv, np.array([0, 55, 70]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(color_mask, cv2.bitwise_or(bright_mask, saturated_mask))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circles: list[tuple[int, int, int, float]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 1.0:
                continue
            circ = 4.0 * np.pi * area / (perimeter * perimeter)
            if circ < 0.38:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(contour)
            ri = int(round(r))
            if ri < min_radius or ri > max_radius:
                continue
            if area < np.pi * (ri * 0.42) ** 2:
                continue
            circles.append((int(round(cx)), int(round(cy)), ri, float(np.clip(0.55 + circ * 0.35, 0.55, 0.88))))
        return circles

    def _merge_circles(self, circles, merge_distance):
        merged: list[tuple[int, int, int, float]] = []
        for cx, cy, radius, score in sorted(circles, key=lambda i: i[3], reverse=True):
            existing_index = None
            for idx, (mx, my, mr, _) in enumerate(merged):
                if float(np.hypot(cx - mx, cy - my)) <= max(merge_distance, min(radius, mr)):
                    existing_index = idx
                    break
            if existing_index is None:
                merged.append((cx, cy, radius, score))
                continue
            mx, my, mr, ms = merged[existing_index]
            total = score + ms
            merged[existing_index] = (
                int(round((cx * score + mx * ms) / total)),
                int(round((cy * score + my * ms) / total)),
                int(round((radius * score + mr * ms) / total)),
                min(1.0, max(score, ms) + 0.05),
            )
        return merged

    def _merge_ball_detections(self, balls):
        merged: list[BallDetection] = []
        for ball in sorted(balls, key=lambda b: b.confidence, reverse=True):
            dup = None
            for idx, existing in enumerate(merged):
                if float(np.linalg.norm(ball.center - existing.center)) < max(ball.radius, existing.radius):
                    dup = idx
                    break
            if dup is None:
                merged.append(ball)
                continue
            existing = merged[dup]
            total = max(ball.confidence + existing.confidence, 1e-6)
            existing.center = (existing.center * existing.confidence + ball.center * ball.confidence) / total
            existing.radius = float((existing.radius * existing.confidence + ball.radius * ball.confidence) / total)
            existing.confidence = max(existing.confidence, ball.confidence)
            if existing.kind == BallKind.UNKNOWN:
                existing.kind = ball.kind
        return merged

    def _enforce_ball_inventory(self, balls):
        cue     = sorted([b for b in balls if b.kind == BallKind.CUE],    key=lambda b: b.confidence, reverse=True)[:1]
        eight   = sorted([b for b in balls if b.kind == BallKind.EIGHT],  key=lambda b: b.confidence, reverse=True)[:1]
        solids  = sorted([b for b in balls if b.kind == BallKind.SOLID],  key=lambda b: b.confidence, reverse=True)[:15]
        stripes = sorted([b for b in balls if b.kind == BallKind.STRIPE], key=lambda b: b.confidence, reverse=True)[:15]
        ordered = cue + eight + solids + stripes
        return sorted(ordered[:16], key=lambda b: (0 if b.kind == BallKind.CUE else 1, b.center[1], b.center[0]))

    def _sample_color(self, frame, cx, cy, radius):
        y0 = max(0, cy - radius // 2)
        y1 = min(frame.shape[0], cy + radius // 2 + 1)
        x0 = max(0, cx - radius // 2)
        x1 = min(frame.shape[1], cx + radius // 2 + 1)
        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            return (255, 255, 255)
        mean = patch.reshape(-1, 3).mean(axis=0)
        return tuple(int(v) for v in mean)

    def _circle_quality(self, gray, cx, cy, radius):
        h, w = gray.shape[:2]
        if cx - radius < 0 or cy - radius < 0 or cx + radius >= w or cy + radius >= h:
            return 0.0, 0.0
        patch = gray[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1]
        if patch.size == 0:
            return 0.0, 0.0
        edges = cv2.Canny(patch, 60, 140)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circularity = 0.0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 1:
                circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        contrast = min(1.0, float(patch.std()) / 64.0)
        return float(circularity), float(contrast)

    def _circle_confidence(self, circularity, contrast, base_score=0.75):
        return float(np.clip(base_score * 0.35 + circularity * 0.45 + contrast * 0.20, 0.0, 0.98))

    def _looks_like_cloth_or_pocket(self, color_bgr, cloth_color):
        color = np.array(color_bgr, dtype=float)
        brightness = float(color.mean())
        cloth_distance = float(np.linalg.norm(color - cloth_color.astype(float)))
        spread = max(color_bgr) - min(color_bgr)
        if spread > 45 and brightness > 55:
            return False
        return cloth_distance < 32.0 or brightness < 32.0

    def _looks_like_table_marking(self, roi, cx, cy, radius, cloth_color):
        y0 = max(0, cy - radius)
        y1 = min(roi.shape[0], cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(roi.shape[1], cx + radius + 1)
        patch = roi[y0:y1, x0:x1]
        if patch.size == 0:
            return True
        color_distance = np.linalg.norm(patch.reshape(-1, 3).astype(float) - cloth_color.astype(float), axis=1)
        non_cloth_ratio = float(np.count_nonzero(color_distance > 42.0)) / float(color_distance.size)
        if non_cloth_ratio < 0.12:
            return True
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 150)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, radius), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, radius)))
        line_pixels = cv2.countNonZero(cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel))
        line_pixels += cv2.countNonZero(cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel))
        return line_pixels > max(14, radius * 3)

    def _near_pocket(self, center, pockets, radius):
        for pocket in pockets:
            if float(np.linalg.norm(center - pocket.center)) < max(pocket.mouth_radius * 1.35, radius * 2.6):
                return True
        return False

    def _point_to_segment_distance(self, point, start, end):
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom < 1e-6:
            return float(np.linalg.norm(point - start))
        t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
        return float(np.linalg.norm(point - (start + segment * t)))

    def _line_overlaps_many_balls(self, start, end, balls):
        overlaps = 0
        for ball in balls:
            if self._point_to_segment_distance(ball.center, start, end) < ball.radius * 1.2:
                overlaps += 1
        return overlaps > 2

    def _fallback_direction_from_balls(self, cue_ball, balls):
        candidates = [b for b in balls if b.id != cue_ball.id and b.kind != BallKind.CUE]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda b: float(np.linalg.norm(b.center - cue_ball.center)))
        direction = nearest.center.astype(float) - cue_ball.center.astype(float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        return direction / norm


# ---------------------------------------------------------------------------
# Ball colour classifier — HSV hue-sector based (BUG 5 FIX)
# ---------------------------------------------------------------------------

def classify_ball(color_bgr: tuple[int, int, int]) -> BallKind:
    """Classify ball type using HSV hue-sector mapping.

    Replaces the old RGB threshold approach which misclassified:
    - Yellow solids (1-ball) as STRIPE (bright with spread > 45)
    - Dark stripes as EIGHT (brightness < 55)
    This HSV approach uniquely maps each hue range to a ball type.
    """
    b, g, r = int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])
    brightness = (b + g + r) / 3.0
    spread = max(color_bgr) - min(color_bgr)

    # White cue ball: very low saturation, high value
    if brightness > 158 and spread < 60:
        return BallKind.CUE

    # Black 8-ball: very dark across all channels (check before hue to avoid maroon confusion)
    if brightness < 45 and spread < 50:
        return BallKind.EIGHT

    # Convert to HSV for hue classification
    pixel = np.array([[[b, g, r]]], dtype=np.uint8)
    hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])

    # Low saturation = white/gray stripe band on striped balls, or near-white
    # High saturation solid color
    if s < 40:
        # Could be a stripe (white band dominates the sample patch)
        if brightness > 155:
            return BallKind.STRIPE
        return BallKind.CUE

    # Hue-sector mapping (OpenCV H range: 0-179, half of standard 0-360)
    # Red:    H < 10 or H > 165  (wraps around)
    # Orange: H 10-20
    # Yellow: H 18-38
    # Green:  H 38-85
    # Cyan:   H 85-100
    # Blue:   H 100-130
    # Purple: H 125-155
    # Magenta:H 145-175

    if h < 10 or h > 165:  # Red
        if v < 110:
            return BallKind.SOLID   # dark red / maroon (7-ball or 15-ball)
        return BallKind.SOLID       # bright red (3-ball or 11-ball)
    elif h < 22:  # Orange
        return BallKind.SOLID       # 5-ball or 13-ball
    elif h < 40:  # Yellow
        return BallKind.SOLID       # 1-ball or 9-ball (was misclassified as STRIPE)
    elif h < 85:  # Green
        return BallKind.SOLID       # 6-ball or 14-ball
    elif h < 100:  # Cyan (not a standard ball color — likely table cloth leak)
        return BallKind.UNKNOWN
    elif h < 130:  # Blue
        return BallKind.SOLID       # 2-ball or 10-ball
    elif h < 158:  # Purple
        return BallKind.SOLID       # 4-ball or 12-ball
    else:  # Magenta / pink
        return BallKind.STRIPE

    return BallKind.SOLID
