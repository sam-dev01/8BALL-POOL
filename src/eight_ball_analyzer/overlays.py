from __future__ import annotations

import cv2
import numpy as np

from .models import BallKind, FrameAnalysis, TableDetection


KIND_COLORS = {
    BallKind.CUE:     (255, 255, 255),
    BallKind.OBJECT:  (30,  210, 255),
    BallKind.SOLID:   (30,  210, 255),
    BallKind.STRIPE:  (255, 210, 120),
    BallKind.EIGHT:   (35,  35,  35),
    BallKind.UNKNOWN: (180, 180, 180),
}

# HUD layout constants
_HUD_X    = 62
_HUD_Y    = 18
_HUD_W    = 380
_LINE_H   = 26
_PWR_X    = 14
_PWR_BAR_W = 18
_PWR_BAR_H = 130

# Grid lines (horizontal × vertical) for the perspective grid
_GRID_COLS = 4
_GRID_ROWS = 2


def draw_overlay(frame: np.ndarray, analysis: FrameAnalysis | None) -> np.ndarray:
    output = frame.copy()
    if analysis is None:
        return output

    table = analysis.table

    # ------------------------------------------------------------------ #
    # 1.  TABLE GEOMETRY — drawn first so everything sits on top of it    #
    # ------------------------------------------------------------------ #
    if table is not None:
        _draw_table_geometry(output, table, analysis)

    # ------------------------------------------------------------------ #
    # 2.  LOCKED POCKETS (all 6 holes — stable geometry)                  #
    # ------------------------------------------------------------------ #
    shot = analysis.shot
    for pocket in analysis.pockets:
        is_target = shot.valid and shot.pocket_id == pocket.id
        ring_color = (0, 255, 120) if is_target else (50, 50, 50)
        thickness = 3 if is_target else 1
        cv2.circle(output, _pt(pocket.center), int(pocket.mouth_radius) + 2,
                   ring_color, thickness, cv2.LINE_AA)
        cv2.circle(output, _pt(pocket.center), 3, (20, 20, 20), -1, cv2.LINE_AA)
        if is_target:
            cv2.putText(output, pocket.label,
                        (_pt(pocket.center)[0] + 6, _pt(pocket.center)[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # 3.  BALLS                                                           #
    # ------------------------------------------------------------------ #
    guide  = analysis.guide
    active_ball_id = (
        guide.first_hit_ball_id
        if guide.first_hit_ball_id is not None
        else shot.object_ball_id
    )

    for ball in analysis.balls:
        is_cue     = ball.kind == BallKind.CUE
        is_active  = ball.id == active_ball_id
        is_blocker = ball.id in shot.blocker_ids

        if is_cue:
            color, thickness = (255, 255, 255), 2
        elif is_active:
            color, thickness = (0, 255, 255), 3
        elif is_blocker:
            color, thickness = (0, 60, 255), 2
        else:
            color, thickness = KIND_COLORS.get(ball.kind, (180, 180, 180)), 1

        cv2.circle(output, ball.xy, int(ball.radius), color, thickness, cv2.LINE_AA)
        cv2.circle(output, ball.xy, 2, color, -1, cv2.LINE_AA)

        if is_cue:
            cv2.putText(output, "cue", (ball.xy[0] + 8, ball.xy[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        elif is_active:
            cv2.putText(output, "target", (ball.xy[0] + 8, ball.xy[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # ------------------------------------------------------------------ #
    # 4.  SHOT TRAJECTORIES                                               #
    # ------------------------------------------------------------------ #
    has_guide_path = len(guide.cue_path) >= 2
    config = analysis.config

    # 4a. Extended cue path (White)
    if has_guide_path:
        if config.show_cue_path:
            _draw_poly_path(output, guide.cue_path, (255, 255, 255), 3)

    # 4b. Debug Collision Geometry
    if config.show_collision and guide.first_hit_point is not None and guide.first_hit_ball_id is not None:
        target_ball = next((b for b in analysis.balls if b.id == guide.first_hit_ball_id), None)
        cue_ball = next((b for b in analysis.balls if b.kind == BallKind.CUE), None)
        
        if target_ball is not None and cue_ball is not None:
            # White circle = Cue Ball Center
            cv2.circle(output, _pt(cue_ball.center), 4, (255, 255, 255), -1, cv2.LINE_AA)
            # Yellow circle = Object Ball Center
            cv2.circle(output, _pt(target_ball.center), 4, (0, 255, 255), -1, cv2.LINE_AA)
            # Blue circle = Ghost Ball Center
            cv2.circle(output, _pt(guide.first_hit_point), 4, (255, 0, 0), -1, cv2.LINE_AA)
            
            # Red dot = Calculated Collision Point
            if guide.physical_contact_point is not None:
                cv2.circle(output, _pt(guide.physical_contact_point), 4, (0, 0, 255), -1, cv2.LINE_AA)

    # 4c. Object Trajectory (Green)
    if config.show_object_path and len(guide.object_path) >= 2:
        _draw_poly_path(output, guide.object_path, (0, 255, 0), 2)
        
    # 4d. Reflections (Yellow)
    if config.show_reflection_path and len(guide.object_reflection_path) >= 2:
        _draw_poly_path(output, guide.object_reflection_path, (0, 255, 255), 2)
        
    # 4e. Secondary Trajectory (Orange)
    if len(guide.secondary_path) >= 2:
        _draw_poly_path(output, guide.secondary_path, (0, 165, 255), 2)
        
    # 4f. Cue Deflection Trajectory (Purple)
    if config.show_deflection and len(guide.cue_deflection_path) >= 2:
        _draw_poly_path(output, guide.cue_deflection_path, (255, 0, 255), 2)

    # ------------------------------------------------------------------ #
    # 5.  HUD + POWER BAR
    _draw_hud(output, shot, guide, analysis)
    _draw_power(output, shot.recommended_power or guide.power or guide.shot_speed, analysis)
    return output


# ---------------------------------------------------------------------------
# Table geometry drawing (Stage 4 visualisation)
# ---------------------------------------------------------------------------

def _draw_table_geometry(
    output: np.ndarray,
    table: TableDetection,
    analysis: FrameAnalysis,
) -> None:
    """Draw minimal table status badge and debugging overlays."""
    # Table status badge (top-right of table area)
    tx, ty, tw, th = table.bounds
    status = f"W:{table.width_px:.0f} H:{table.height_px:.0f}  conf:{table.confidence:.2f}"
    cv2.putText(output, status,
                (tx + 4, ty - 6 if ty > 14 else ty + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 100), 1, cv2.LINE_AA)

    # BUG 9 FIX: Remove always-on red debug rectangle (coordinate bugs are now fixed).
    # Only draw the calibrated polygon outline for visual reference.
    if table.polygon is not None and len(table.polygon) > 0:
        pts = np.array(table.polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(output, [pts], True, (0, 180, 60), 1, cv2.LINE_AA)




def _draw_perspective_grid(
    output: np.ndarray,
    corners: np.ndarray,
    table: TableDetection,
) -> None:
    """Draw a grid in normalised space mapped back to image space."""
    if table.inv_transform_matrix is None:
        return
    M_inv = table.inv_transform_matrix
    W = table.width_px
    H = table.height_px

    def norm_to_img(nx: float, ny: float) -> tuple[int, int]:
        pt = np.array([[[nx, ny]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, M_inv)
        return int(round(out[0, 0, 0])), int(round(out[0, 0, 1]))

    cols = _GRID_COLS
    rows = _GRID_ROWS
    color = (60, 100, 60)

    # Vertical lines
    for c in range(1, cols):
        x_n = W * c / cols
        p1 = norm_to_img(x_n, 0.0)
        p2 = norm_to_img(x_n, H)
        cv2.line(output, p1, p2, color, 1, cv2.LINE_AA)

    # Horizontal lines
    for r in range(1, rows):
        y_n = H * r / rows
        p1 = norm_to_img(0.0,  y_n)
        p2 = norm_to_img(W, y_n)
        cv2.line(output, p1, p2, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Generic drawing helpers
# ---------------------------------------------------------------------------

def _pt(point: np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def _difficulty_color(score: float) -> tuple[int, int, int]:
    if score < 35: return (80, 255, 80)
    if score < 70: return (0, 220, 255)
    return (0, 80, 255)


def _shot_is_on_table(analysis: FrameAnalysis) -> bool:
    if analysis.table is None or analysis.shot.ghost_ball_center is None:
        return False
    x, y, w, h = analysis.table.bounds
    point  = analysis.shot.ghost_ball_center
    margin = max(8, min(w, h) * 0.04)
    return (x + margin <= point[0] <= x + w - margin
            and y + margin <= point[1] <= y + h - margin)


def _draw_poly_path(
    output: np.ndarray,
    path: list[np.ndarray],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if len(path) < 2:
        return
    for index in range(len(path) - 1):
        is_last = index == len(path) - 2
        cv2.arrowedLine(
            output,
            _pt(path[index]),
            _pt(path[index + 1]),
            color,
            thickness if index == 0 else max(1, thickness - 1),
            cv2.LINE_AA,
            tipLength=0.05 if is_last else 0.02,
        )


def _draw_hud(output: np.ndarray, shot, guide, analysis: FrameAnalysis) -> None:
    """Draw the stats box BELOW the pool table so it never obscures the playing surface."""
    hit_label   = f"ball {guide.first_hit_ball_id}" if guide.first_hit_ball_id is not None else "--"
    chain_count = len(guide.collision_paths)
    trick_shot_status = "ON" if analysis.config.trick_shot_mode else "OFF"

    if analysis.table is not None:
        tx, ty, tw, th = analysis.table.bounds
        table_line = f"Table: {analysis.table.width_px:.0f}x{analysis.table.height_px:.0f}"
        # Anchor text BELOW the table
        base_x = tx
        base_y = ty + th + 18   # 18px below the table bottom edge
    else:
        table_line = "Table: Not detected"
        base_x, base_y = 10, output.shape[0] - 120

    # Clamp so text stays inside the image
    img_h = output.shape[0]
    if base_y + 120 > img_h:
        # Not enough room below table — draw above the table instead
        base_y = max(14, ty - 10) if analysis.table is not None else 14

    lines = [
        (f"State: {analysis.state.name}  |  {table_line}  |  Hit: {hit_label}",
         (0, 230, 255), 0.45, 1),
        (f"Mode: Trajectory Visualizer  |  Trick: {trick_shot_status}  |  Speed: {guide.shot_speed:.0f}px",
         (0, 200, 120), 0.45, 1),
        (u"W=Cue  Y=Reflect  G=Object  O=Secondary  \u25cf=Collision",
         (200, 200, 200), 0.40, 1),
    ]

    line_h = 16
    for i, (text, color, scale, thickness) in enumerate(lines):
        y = base_y + i * line_h
        if 0 <= y < img_h:
            # Subtle dark shadow for readability
            cv2.putText(output, text, (base_x + 1, y + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1)
            cv2.putText(output, text, (base_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)




def _draw_power(output: np.ndarray, power: float, analysis: FrameAnalysis) -> None:
    if analysis.table is not None:
        tx, ty, tw, th = analysis.table.bounds
        x0 = max(10, tx - 30)
        y0 = ty + 10
    else:
        x0 = 10
        y0 = 60

    h  = min(150, int(output.shape[0] * 0.4))
    _PWR_BAR_W = 12
    
    cv2.rectangle(output, (x0, y0), (x0 + _PWR_BAR_W, y0 + h), (60, 60, 60), -1)
    cv2.rectangle(output, (x0, y0), (x0 + _PWR_BAR_W, y0 + h), (230, 230, 230), 1)
    fill_h = int(h * float(np.clip(power, 0.0, 100.0)) / 100.0)
    if fill_h > 0:
        fill_color = (0, 255, 100) if power < 40 else (0, 180, 255) if power < 75 else (0, 80, 255)
        cv2.rectangle(output,
                      (x0 + 2, y0 + h - fill_h),
                      (x0 + _PWR_BAR_W - 2, y0 + h - 2),
                      fill_color, -1)
    cv2.putText(output, f"{power:.0f}%", (x0 - 2, y0 + h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 1)
