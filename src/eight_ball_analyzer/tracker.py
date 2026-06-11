from __future__ import annotations

import numpy as np

from .models import BallDetection, BallKind


class BallTracker:
    """Lightweight ball ID tracker — shows balls immediately (no multi-frame delay)."""

    def __init__(self, max_distance: float = 55.0, max_tracks: int = 16) -> None:
        self.max_distance = max_distance
        self.max_tracks = max_tracks
        self._tracks: dict[int, np.ndarray] = {}
        self._velocities: dict[int, np.ndarray] = {}
        self._history: dict[int, list[np.ndarray]] = {}
        self._kinds: dict[int, BallKind] = {}
        self._radii: dict[int, float] = {}
        self._colors: dict[int, tuple[int, int, int]] = {}
        self._confidences: dict[int, float] = {}

    def reset(self) -> None:
        self._tracks.clear()
        self._velocities.clear()
        self._history.clear()
        self._kinds.clear()
        self._radii.clear()
        self._colors.clear()
        self._confidences.clear()

    def assign(self, detections: list[BallDetection]) -> list[BallDetection]:
        detections = detections[: self.max_tracks]
        max_distance = self.max_distance
        if detections:
            median_r = float(np.median([d.radius for d in detections]))
            max_distance = float(np.clip(median_r * 5.5, 12.0, 55.0))

        if not self._tracks:
            for detection in detections:
                track_id = self._next_free_id()
                detection.id = track_id
                self._start_track(track_id, detection)
            return detections

        unmatched_tracks = set(self._tracks.keys())
        for detection in detections:
            best_id = None
            best_distance = max_distance
            for track_id in unmatched_tracks:
                predicted = self._tracks[track_id] + self._velocities.get(track_id, np.zeros(2))
                dist = float(np.linalg.norm(detection.center - predicted))
                if dist < best_distance:
                    best_id = track_id
                    best_distance = dist
            if best_id is None:
                track_id = self._next_free_id()
                detection.id = track_id
                if track_id in self._tracks:
                    self._drop_track(track_id)
                unmatched_tracks.discard(track_id)
                self._start_track(track_id, detection)
            else:
                detection.id = best_id
                previous = self._tracks[best_id].copy()
                self._tracks[best_id] = detection.center.copy()
                self._velocities[best_id] = detection.center.astype(float) - previous.astype(float)
                self._history.setdefault(best_id, []).append(detection.center.copy())
                self._history[best_id] = self._history[best_id][-20:]
                self._kinds[best_id] = detection.kind
                self._radii[best_id] = detection.radius
                self._colors[best_id] = detection.color_bgr
                self._confidences[best_id] = detection.confidence
                unmatched_tracks.remove(best_id)
        for track_id in unmatched_tracks:
            self._drop_track(track_id)
        return detections

    def _start_track(self, track_id: int, detection: BallDetection) -> None:
        self._tracks[track_id] = detection.center.copy()
        self._velocities[track_id] = np.zeros(2)
        self._history[track_id] = [detection.center.copy()]
        self._kinds[track_id] = detection.kind
        self._radii[track_id] = detection.radius
        self._colors[track_id] = detection.color_bgr
        self._confidences[track_id] = detection.confidence

    def _drop_track(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)
        self._velocities.pop(track_id, None)
        self._history.pop(track_id, None)
        self._kinds.pop(track_id, None)
        self._radii.pop(track_id, None)
        self._colors.pop(track_id, None)
        self._confidences.pop(track_id, None)

    def get_speed(self, track_id: int) -> float:
        velocity = self._velocities.get(track_id)
        if velocity is None:
            return 0.0
        return float(np.linalg.norm(velocity))

    def _next_free_id(self) -> int:
        for track_id in range(self.max_tracks):
            if track_id not in self._tracks:
                return track_id
        return min(self._tracks, key=lambda track_id: len(self._history.get(track_id, [])))
