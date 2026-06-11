from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class BallKind(str, Enum):
    CUE = "cue"
    OBJECT = "object"
    SOLID = "solid_ball"
    STRIPE = "stripe_ball"
    EIGHT = "eight"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class BallDetection:
    id: int
    center: np.ndarray
    radius: float
    kind: BallKind = BallKind.UNKNOWN
    confidence: float = 0.0
    color_bgr: tuple[int, int, int] = (255, 255, 255)

    @property
    def xy(self) -> tuple[int, int]:
        return int(round(float(self.center[0]))), int(round(float(self.center[1])))


@dataclass(slots=True)
class Rail:
    """One cushion rail of the pool table.

    id: 0 = top, 1 = right, 2 = bottom, 3 = left
    normal: inward-facing unit normal vector (points toward table centre).
    """
    id: int
    start: np.ndarray   # image-space pixel coordinate
    end: np.ndarray     # image-space pixel coordinate
    normal: np.ndarray  # 2-D unit normal pointing inward


@dataclass(slots=True)
class Pocket:
    """A pocket on the pool table.

    id:           0=top-left, 1=top-middle, 2=top-right,
                  3=bottom-left, 4=bottom-middle, 5=bottom-right
    center:       image-space pixel coordinate [x, y]
    mouth_radius: catchment radius – balls within this distance count as potted.
                  Corner pockets are wider than middle pockets.
    label:        human-readable name
    """
    id: int
    center: np.ndarray
    mouth_radius: float
    label: str

    # Back-compat alias so callers using pocket.radius still work.
    @property
    def radius(self) -> float:
        return self.mouth_radius

    @property
    def xy(self) -> tuple[int, int]:
        return int(round(float(self.center[0]))), int(round(float(self.center[1])))


@dataclass(slots=True)
class TableDetection:
    """Full table geometry, including perspective transform data.

    corners:              [top-left, top-right, bottom-right, bottom-left] (image space)
    width_px / height_px: extents of the perspective-corrected playable area.
    transform_matrix:     3×3 homography — image space → normalised [0,1]² space.
    inv_transform_matrix: inverse homography — normalised → image space.
    rails:                four Rail objects (top, right, bottom, left).
    """
    polygon: np.ndarray
    bounds: tuple[int, int, int, int]
    confidence: float
    # Geometry fields – populated by detect_table after corner finding.
    corners: np.ndarray = field(default_factory=lambda: np.zeros((4, 2), dtype=float))
    width_px: float = 0.0
    height_px: float = 0.0
    transform_matrix: Optional[np.ndarray] = None
    inv_transform_matrix: Optional[np.ndarray] = None
    rails: list[Rail] = field(default_factory=list)


@dataclass(slots=True)
class ShotPrediction:
    valid: bool = False
    object_ball_id: Optional[int] = None
    pocket_id: Optional[int] = None
    pocket_label: str = "--"
    ghost_ball_center: Optional[np.ndarray] = None
    cue_to_ghost: Optional[tuple[np.ndarray, np.ndarray]] = None
    object_to_pocket: Optional[tuple[np.ndarray, np.ndarray]] = None
    cue_after_impact: list[np.ndarray] = field(default_factory=list)
    bank_path: list[np.ndarray] = field(default_factory=list)
    blocker_ids: list[int] = field(default_factory=list)
    cut_angle: float = 0.0
    cue_path_length: float = 0.0
    object_path_length: float = 0.0
    rail_count: int = 0
    recommended_power: float = 0.0
    power_label: str = "--"
    difficulty: float = 0.0
    difficulty_label: str = "--"
    success_probability: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BallPath:
    ball_id: int
    path: list[np.ndarray] = field(default_factory=list)
    speed: float = 0.0
    label: str = ""


@dataclass(slots=True)
class AimGuide:
    cue_direction: Optional[np.ndarray] = None
    cue_path: list[np.ndarray] = field(default_factory=list)
    first_hit_ball_id: Optional[int] = None
    first_hit_point: Optional[np.ndarray] = None
    object_path: list[np.ndarray] = field(default_factory=list)
    collision_paths: list[BallPath] = field(default_factory=list)
    cue_deflection_path: list[np.ndarray] = field(default_factory=list)
    shot_speed: float = 0.0
    power: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FrameAnalysis:
    frame_index: int
    table: Optional[TableDetection]
    balls: list[BallDetection]
    pockets: list[Pocket]
    ball_radius_px: float = 0.0          # calibrated from detected balls this frame
    operating_mode: str = "Analysis Mode"
    shot: ShotPrediction = field(default_factory=ShotPrediction)
    guide: AimGuide = field(default_factory=AimGuide)
