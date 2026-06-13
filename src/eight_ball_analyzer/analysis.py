from __future__ import annotations

import numpy as np

from .detection import PoolDetector
from .geometry import (
    fill_guide_from_balls,
    predict_shot,
    recommend_best_shot,
    recommend_pocket,
    trace_ball_collision_chain,
    unit_vector,
)
from .models import AimGuide, BallDetection, BallKind, FrameAnalysis, GameState, Pocket, VisualizerConfig
from .calibration import CalibrationManager
from .tracker import BallTracker


class VideoAnalyzer:
    """TABLE-FIRST analysis pipeline with Cheto-style extended guidelines."""

    def __init__(self) -> None:
        self.detector = PoolDetector()
        self.tracker = BallTracker()
        self.calibration = CalibrationManager()
        self.target_lock_id: int | None = None
        self._frame_index: int = 0
        self._last_detection_frame: int = -999
        self.state: GameState = GameState.READY
        self.settle_frames: int = 0
        self.config = VisualizerConfig()
        self.detection_interval_frames: int = 12
        # BUG 1 FIX: attributes referenced in _select_target() but were never defined
        self.selected_object_ball_id: int | None = None
        self.selected_pocket_id: int | None = None
        self._first_frame_printed: bool = False

    def reset_tracking(self) -> None:
        self.tracker.reset()
        self._frame_index = 0
        self.target_lock_id = None
        self._last_detection_frame = -999

    def analyze_frame(self, frame, frame_index: int = 0, track: bool = True) -> FrameAnalysis:
        self._frame_index = frame_index

        if frame is None or getattr(frame, "size", 0) == 0:
            return FrameAnalysis(frame_index=frame_index, table=None, balls=[], pockets=[])

        if not self.calibration.is_calibrated():
            return FrameAnalysis(frame_index=frame_index, table=None, balls=[], pockets=[])

        table, pockets = self.calibration.get_table_and_pockets()
        if table is None:
            return FrameAnalysis(frame_index=frame_index, table=None, balls=[], pockets=[])

        # ROI Crop for YOLO performance
        fh, fw = frame.shape[:2]
        corners = self.calibration.load_corners()
        cx = max(0, int(corners[:, 0].min()))
        cy = max(0, int(corners[:, 1].min()))
        cx2 = min(fw, int(corners[:, 0].max()))
        cy2 = min(fh, int(corners[:, 1].max()))
        roi_frame = frame[cy:cy2, cx:cx2].copy() if (cx2 > cx and cy2 > cy) else frame

        # Periodic YOLO Detection
        frames_since_detection = frame_index - self._last_detection_frame
        min_retry = max(3, self.detection_interval_frames // 3)
        should_detect = (
            frames_since_detection >= self.detection_interval_frames
            or (len(self.tracker._tracks) < 2 and frames_since_detection >= min_retry)
        )
        if should_detect:
            import copy
            roi_table = copy.deepcopy(table)
            roi_table.polygon = roi_table.polygon - [cx, cy]
            rx, ry, rw, rh = roi_table.bounds
            roi_table.bounds = (max(0, rx - cx), max(0, ry - cy), rw, rh)
            if roi_table.corners is not None:
                roi_table.corners = roi_table.corners - [cx, cy]
            for rail in roi_table.rails:
                rail.start = rail.start - [cx, cy]
                rail.end = rail.end - [cx, cy]
            
            roi_pockets = copy.deepcopy(pockets)
            for p in roi_pockets:
                p.center = p.center - [cx, cy]

            detected_balls = self.detector.detect_balls(roi_frame, roi_table, pockets=roi_pockets)
            
            # Map detections back to full-frame space
            for b in detected_balls:
                b.center = np.array([b.center[0] + cx, b.center[1] + cy], dtype=float)
                
            self._last_detection_frame = frame_index
        else:
            detected_balls = []

        if track:
            # When detected_balls is empty, tracker extrpolates existing tracks
            balls = self.tracker.assign(detected_balls)
        else:
            balls = detected_balls

        ball_radius_px = self._calibrate_radius(balls)
        cue_ball = self._ensure_cue_ball(frame, balls, table)
        if cue_ball is not None:
            existing_ids = {b.id for b in balls}
            if cue_ball.id not in existing_ids:
                if track:
                    cue_ball.id = self.tracker.register_single(cue_ball)
                balls.append(cue_ball)
            else:
                for ball in balls:
                    if ball.id == cue_ball.id and ball.kind != BallKind.CUE:
                        ball.kind = BallKind.CUE
                        break

        # Always read cue direction every frame (it uses fast OpenCV edge/line detection)
        cue_direction = self.detector.detect_cue_direction(frame, table, cue_ball, balls)
        power = max(
            self.detector.detect_power(frame, table),
            self._cue_speed_from_tracker(cue_ball),
            35.0 if cue_direction is not None else 0.0,
        )

        # State Machine Update
        max_speed = 0.0
        if balls:
            max_speed = max(self.tracker.get_speed(b.id) for b in balls)

        movement_threshold = 2.0
        previous_state = self.state

        if max_speed > movement_threshold:
            self.state = GameState.SHOT_IN_PROGRESS
            self.settle_frames = 0
            self.target_lock_id = None # Clear lock on shot
        elif max_speed <= movement_threshold and self.state == GameState.SHOT_IN_PROGRESS:
            self.state = GameState.TABLE_SETTLING
        
        if self.state == GameState.TABLE_SETTLING:
            self.settle_frames += 1
            if self.settle_frames > 20:  # ~0.6 seconds at 30fps
                self.state = GameState.READY
                
        if self.state == GameState.READY and cue_direction is not None:
            self.state = GameState.AIMING
        elif self.state == GameState.AIMING and cue_direction is None:
            self.state = GameState.READY

        if previous_state != self.state:
            print(f"[STATE] {self.state.name}")

        # BUG 7 FIX: suppress per-frame print flood (only print on first frame)
        if not self._first_frame_printed:
            self._first_frame_printed = True

        if self.state in (GameState.SHOT_IN_PROGRESS, GameState.TABLE_SETTLING):
            return FrameAnalysis(
                frame_index=frame_index, table=table, balls=balls, pockets=pockets,
                state=self.state, ball_radius_px=ball_radius_px, operating_mode="Tracking...",
                config=self.config,
            )

        # Build trajectory guide
        guide = self._build_guide(cue_ball, balls, table, cue_direction, power, ball_radius_px, pockets)

        # Target Locking Logic
        if guide.first_hit_ball_id is not None:
            self.target_lock_id = guide.first_hit_ball_id

        analysis = FrameAnalysis(
            frame_index=frame_index, table=table, balls=balls, pockets=pockets,
            state=self.state, ball_radius_px=ball_radius_px,
            operating_mode="MODE A: Trajectory Visualizer",
            config=self.config,
        )
        analysis.guide = guide

        # BUG 2 FIX: _select_target() was defined but never called — wire it up now
        is_aim_mode = cue_direction is not None
        object_ball, pocket = self._select_target(
            cue_ball, balls, pockets, table, guide, ball_radius_px, is_aim_mode
        )
        if object_ball is not None and pocket is not None and cue_ball is not None:
            analysis.shot = predict_shot(
                cue_ball, object_ball, pocket, balls, table, ball_radius_px
            )
        return analysis

    def _select_target(
        self,
        cue_ball: BallDetection | None,
        balls: list[BallDetection],
        pockets: list[Pocket],
        table,
        guide: AimGuide,
        ball_radius_px: float,
        is_aim_mode: bool,
    ) -> tuple[BallDetection | None, Pocket | None]:
        object_ball: BallDetection | None = None
        pocket: Pocket | None = None

        if self.selected_object_ball_id is not None:
            object_ball = next((b for b in balls if b.id == self.selected_object_ball_id), None)

        if object_ball is None and guide.first_hit_ball_id is not None:
            object_ball = next((b for b in balls if b.id == guide.first_hit_ball_id), None)

        if is_aim_mode:
            if object_ball is None and cue_ball is not None:
                others = [
                    b for b in balls
                    if b.id != cue_ball.id and b.kind != BallKind.CUE and b.confidence >= 0.45
                ]
                if others and guide.cue_direction is not None:
                    object_ball = max(
                        others,
                        key=lambda b: float(np.dot(
                            unit_vector(cue_ball.center, b.center), guide.cue_direction
                        )),
                    )

            if self.selected_pocket_id is not None:
                pocket = next((p for p in pockets if p.id == self.selected_pocket_id), None)

            if pocket is None and object_ball is not None and guide.object_path:
                end_pt = guide.object_path[-1]
                pocket = min(
                    pockets,
                    key=lambda p: float(np.linalg.norm(end_pt - p.center)),
                )

            if pocket is None and cue_ball is not None and object_ball is not None:
                pocket = recommend_pocket(cue_ball, object_ball, pockets, balls, table, ball_radius_px)
        else:
            if object_ball is None and cue_ball is not None:
                object_ball, pocket = recommend_best_shot(
                    cue_ball, balls, pockets, table, ball_radius_px,
                )
                if object_ball is not None:
                    fill_guide_from_balls(guide, cue_ball, object_ball, table)
            elif object_ball is not None and cue_ball is not None:
                pocket = recommend_pocket(
                    cue_ball, object_ball, pockets, balls, table, ball_radius_px,
                )

        if self.selected_pocket_id is not None and pocket is None:
            pocket = next((p for p in pockets if p.id == self.selected_pocket_id), None)

        return object_ball, pocket

    def _table_is_valid(self, table) -> bool:
        if table is None:
            return False
        if table.confidence < 0.20:
            return False
        _, _, w, h = table.bounds
        if h == 0:
            return False
        aspect = w / float(h)
        return 1.3 <= aspect <= 3.2

    def _calibrate_radius(self, balls: list[BallDetection]) -> float:
        if not balls:
            return 14.0
        radii = [b.radius for b in balls if b.radius > 2.5]
        return float(np.median(radii)) if radii else 14.0

    def _build_guide(
        self,
        cue_ball,
        balls,
        table,
        cue_direction,
        power,
        ball_radius_px: float,
        pockets: list[Pocket],
    ) -> AimGuide:
        if cue_ball is None or table is None:
            return AimGuide(power=power, notes=["Cue ball or table not detected."])

        if cue_direction is not None:
            return trace_ball_collision_chain(
                cue_ball, balls, table, cue_direction, max(power, 40.0), pockets,
                ball_radius_px=ball_radius_px, max_chain_depth=2, max_bounces=self.config.max_bounces,
                locked_target_id=self.target_lock_id
            )

        # Fallback: line from cue to nearest object ball
        others = [
            b for b in balls
            if b.id != cue_ball.id and b.kind != BallKind.CUE and b.confidence >= 0.40
        ]
        if others:
            nearest = min(others, key=lambda b: float(np.linalg.norm(b.center - cue_ball.center)))
            guide = AimGuide(power=max(power, 35.0))
            fill_guide_from_balls(guide, cue_ball, nearest, table)
            guide.notes.append("Using nearest-ball fallback aim.")
            return guide

        return AimGuide(power=power, notes=["Cue direction not detected."])

    def _cue_speed_from_tracker(self, cue_ball: BallDetection | None) -> float:
        if cue_ball is None:
            return 0.0
        return float(np.clip(self.tracker.get_speed(cue_ball.id) * 4.5, 0.0, 100.0))

    def _ensure_cue_ball(
        self,
        frame,
        balls: list[BallDetection],
        table,
    ) -> BallDetection | None:
        cue = self._infer_cue_ball(balls, table)
        if cue is not None:
            return cue
        if table is None:
            return None
        visual = self.detector.find_cue_ball_on_table(frame, table, balls)
        if visual is not None:
            return visual
        return None

    def _infer_cue_ball(self, balls, table) -> BallDetection | None:
        cue_candidates = [b for b in balls if b.kind == BallKind.CUE]
        if len(cue_candidates) == 1:
            return cue_candidates[0]
        if len(cue_candidates) > 1:
            others = [b for b in balls if b.kind != BallKind.CUE]
            if others:
                return min(
                    cue_candidates,
                    key=lambda cue: min(
                        float(np.linalg.norm(cue.center - o.center)) for o in others
                    ),
                )
            return max(cue_candidates, key=lambda b: b.confidence)

        if not balls:
            return None

        bright_neutral = [
            b for b in balls
            if sum(b.color_bgr) / 3.0 > 155
            and max(b.color_bgr) - min(b.color_bgr) < 70
            and b.kind not in {BallKind.SOLID, BallKind.STRIPE, BallKind.EIGHT}
        ]
        if bright_neutral:
            return max(bright_neutral, key=lambda b: (sum(b.color_bgr), b.confidence))

        if table is not None:
            x, y, w, h = table.bounds
            table_center = np.array([x + w / 2.0, y + h / 2.0])
            neutral = [
                b for b in balls
                if sum(b.color_bgr) / 3.0 > 140 and max(b.color_bgr) - min(b.color_bgr) < 80
            ]
            if neutral:
                return max(
                    neutral,
                    key=lambda b: sum(b.color_bgr) * 0.6 + float(np.linalg.norm(b.center - table_center)) * 0.01,
                )
        return None
