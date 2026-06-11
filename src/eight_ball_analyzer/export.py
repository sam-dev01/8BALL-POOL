from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

import cv2
import numpy as np

from .analysis import VideoAnalyzer
from .models import BallDetection, FrameAnalysis, Pocket
from .overlays import draw_overlay


ProgressCallback = Callable[[int, int], None]
CancelCallback = Callable[[], bool]


def export_analyzed_video(
    input_path: str,
    output_path: str,
    analyzer: VideoAnalyzer,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> bool:
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    analyzer.reset_tracking()
    frame_index = 0
    cancelled = False
    try:
        while True:
            if cancel_callback is not None and cancel_callback():
                cancelled = True
                break
            ok, frame = capture.read()
            if not ok:
                break
            analysis = analyzer.analyze_frame(frame, frame_index=frame_index, track=True)
            writer.write(draw_overlay(frame, analysis))
            frame_index += 1
            if progress_callback is not None:
                progress_callback(frame_index, total)
    finally:
        capture.release()
        writer.release()
    return not cancelled


def export_analyzed_frame(frame, analysis: FrameAnalysis, output_path: str) -> None:
    if not cv2.imwrite(output_path, draw_overlay(frame, analysis)):
        raise RuntimeError(f"Could not write frame: {output_path}")


def export_analysis_report(analysis: FrameAnalysis, output_path: str) -> None:
    payload = analysis_to_dict(analysis)
    try:
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not write report: {output_path}") from exc


def analysis_to_dict(analysis: FrameAnalysis) -> dict:
    shot = analysis.shot
    return {
        "frame_index": analysis.frame_index,
        "table": None
        if analysis.table is None
        else {
            "bounds": list(analysis.table.bounds),
            "confidence": analysis.table.confidence,
            "polygon": _points(analysis.table.polygon),
        },
        "pockets": [_pocket_to_dict(pocket) for pocket in analysis.pockets],
        "balls": [_ball_to_dict(ball) for ball in analysis.balls],
        "shot": {
            "valid": shot.valid,
            "object_ball_id": shot.object_ball_id,
            "chosen_pocket_id": shot.pocket_id,
            "chosen_pocket_label": shot.pocket_label,
            "ghost_ball": _point(shot.ghost_ball_center),
            "cue_to_ghost": _segment(shot.cue_to_ghost),
            "object_to_pocket": _segment(shot.object_to_pocket),
            "cue_after_impact": _points(shot.cue_after_impact),
            "bank_path": _points(shot.bank_path),
            "cut_angle": shot.cut_angle,
            "cue_path_length": shot.cue_path_length,
            "object_path_length": shot.object_path_length,
            "blocker_ids": shot.blocker_ids,
            "rail_count": shot.rail_count,
            "difficulty": shot.difficulty,
            "difficulty_label": shot.difficulty_label,
            "success_probability": shot.success_probability,
            "recommended_power": shot.recommended_power,
            "power_label": shot.power_label,
            "notes": shot.notes,
        },
        "guide": {
            "cue_path": _points(analysis.guide.cue_path),
            "first_hit_ball_id": analysis.guide.first_hit_ball_id,
            "first_hit_point": _point(analysis.guide.first_hit_point),
            "object_path": _points(analysis.guide.object_path),
            "cue_deflection_path": _points(analysis.guide.cue_deflection_path),
            "collision_paths": [
                {
                    "ball_id": ball_path.ball_id,
                    "path": _points(ball_path.path),
                    "speed": ball_path.speed,
                    "label": ball_path.label,
                }
                for ball_path in analysis.guide.collision_paths
            ],
            "shot_speed": analysis.guide.shot_speed,
            "power": analysis.guide.power,
            "notes": analysis.guide.notes,
        },
    }


def _ball_to_dict(ball: BallDetection) -> dict:
    return {
        "id": ball.id,
        "type": ball.kind.value,
        "x": float(ball.center[0]),
        "y": float(ball.center[1]),
        "radius": ball.radius,
        "confidence": ball.confidence,
    }


def _pocket_to_dict(pocket: Pocket) -> dict:
    return {
        "id": pocket.id,
        "label": pocket.label,
        "x": float(pocket.center[0]),
        "y": float(pocket.center[1]),
        "radius": pocket.radius,
    }


def _point(point: np.ndarray | None) -> list[float] | None:
    if point is None:
        return None
    return [float(point[0]), float(point[1])]


def _points(points) -> list[list[float]]:
    return [_point(point) for point in points if point is not None]


def _segment(segment: tuple[np.ndarray, np.ndarray] | None) -> list[list[float]] | None:
    if segment is None:
        return None
    return [_point(segment[0]), _point(segment[1])]
