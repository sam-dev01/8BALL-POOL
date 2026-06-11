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
from .models import AimGuide, BallDetection, BallKind, FrameAnalysis, Pocket
from .table_lock import TableGeometryLock
from .tracker import BallTracker


class VideoAnalyzer:
    """TABLE-FIRST analysis pipeline with Cheto-style extended guidelines."""

    def __init__(self) -> None:
        self.detector = PoolDetector()
        self.tracker = BallTracker()
        self.table_lock = TableGeometryLock()
        self.selected_object_ball_id: int | None = None
        self.selected_pocket_id: int | None = None
        self._frame_index: int = 0

    def reset_tracking(self) -> None:
        self.tracker.reset()
        self.table_lock.reset()
        self._frame_index = 0

    def analyze_frame(self, frame, frame_index: int = 0, track: bool = True) -> FrameAnalysis:
        self._frame_index = frame_index

        if frame is None or getattr(frame, "size", 0) == 0:
            return FrameAnalysis(frame_index=frame_index, table=None, balls=[], pockets=[])

        table = self.detector.detect_table(frame)
        if not self._table_is_valid(table):
            if self.table_lock.is_locked:
                table, _ = self.table_lock.apply(None, [], frame_index)
            if table is None:
                return FrameAnalysis(frame_index=frame_index, table=None, balls=[], pockets=[])

        pockets = self.detector.compute_pockets(table)  # type: ignore[arg-type]
        table, pockets = self.table_lock.apply(table, pockets, frame_index)  # type: ignore[arg-type]

        balls = self.detector.detect_balls(frame, table, pockets=pockets)
        if track:
            balls = self.tracker.assign(balls)

        ball_radius_px = self._calibrate_radius(balls)
        cue_ball = self._ensure_cue_ball(frame, balls, table)
        if cue_ball is not None:
            existing_ids = {b.id for b in balls}
            if cue_ball.id not in existing_ids:
                if track:
                    cue_ball = self.tracker.assign([cue_ball])[0]
                balls.append(cue_ball)
            else:
                for ball in balls:
                    if ball.id == cue_ball.id and ball.kind != BallKind.CUE:
                        ball.kind = BallKind.CUE
                        break

        cue_direction = self.detector.detect_cue_direction(frame, table, cue_ball, balls)
        power = max(
            self.detector.detect_power(frame, table),
            self._cue_speed_from_tracker(cue_ball),
            35.0 if cue_direction is not None else 0.0,
        )

        guide = self._build_guide(cue_ball, balls, table, cue_direction, power, ball_radius_px)

        is_aim_mode = cue_direction is not None or (
            guide.cue_direction is not None and guide.first_hit_ball_id is not None
        )
        operating_mode = "MODE A: Extended Guideline" if is_aim_mode else "MODE B: Recommendation"

        object_ball, pocket = self._select_target(
            cue_ball, balls, pockets, table, guide, ball_radius_px, is_aim_mode,
        )

        # Always try to fill metrics when we have cue + object
        if pocket is None and object_ball is not None and cue_ball is not None:
            pocket = recommend_pocket(cue_ball, object_ball, pockets, balls, table, ball_radius_px)

        if not guide.collision_paths and cue_ball is not None and object_ball is not None:
            direction = guide.cue_direction
            if direction is None:
                direction = unit_vector(cue_ball.center, object_ball.center)
            if float(np.linalg.norm(direction)) > 1e-6:
                chain_guide = trace_ball_collision_chain(
                    cue_ball, balls, table, direction, max(power, 40.0),
                    ball_radius_px=ball_radius_px, max_chain_depth=4,
                )
                if chain_guide.collision_paths:
                    guide.collision_paths = chain_guide.collision_paths
                    guide.object_path = chain_guide.object_path or guide.object_path
                    guide.cue_deflection_path = chain_guide.cue_deflection_path or guide.cue_deflection_path
                    guide.first_hit_ball_id = chain_guide.first_hit_ball_id or guide.first_hit_ball_id
                    guide.first_hit_point = chain_guide.first_hit_point or guide.first_hit_point
                    guide.cue_path = chain_guide.cue_path or guide.cue_path
                    guide.cue_direction = chain_guide.cue_direction or guide.cue_direction
                    guide.shot_speed = chain_guide.shot_speed or guide.shot_speed

        analysis = FrameAnalysis(
            frame_index=frame_index,
            table=table,
            balls=balls,
            pockets=pockets,
            ball_radius_px=ball_radius_px,
            operating_mode=operating_mode,
        )
        analysis.guide = guide
        analysis.shot = predict_shot(cue_ball, object_ball, pocket, balls, table, ball_radius_px)

        if guide.collision_paths and analysis.shot.valid:
            first_path = guide.collision_paths[0].path
            if len(first_path) >= 2:
                analysis.shot.bank_path = first_path
            if guide.cue_deflection_path:
                analysis.shot.cue_after_impact = guide.cue_deflection_path
            if guide.first_hit_point is not None and guide.cue_direction is not None:
                ghost = guide.first_hit_point - guide.cue_direction * (2.0 * ball_radius_px)
                analysis.shot.ghost_ball_center = ghost
                if cue_ball is not None:
                    analysis.shot.cue_to_ghost = (cue_ball.center, ghost)

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
    ) -> AimGuide:
        if cue_ball is None or table is None:
            return AimGuide(power=power, notes=["Cue ball or table not detected."])

        if cue_direction is not None:
            return trace_ball_collision_chain(
                cue_ball, balls, table, cue_direction, max(power, 40.0),
                ball_radius_px=ball_radius_px, max_chain_depth=4,
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
