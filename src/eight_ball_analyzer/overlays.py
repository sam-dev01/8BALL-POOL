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
    has_shot = shot.valid and (
        _shot_is_on_table(analysis)
        or (shot.ghost_ball_center is not None and shot.object_ball_id is not None)
    )

    # 4a. Extended cue path (yellow — Cheto-style)
    if has_guide_path:
        _draw_poly_path(output, guide.cue_path, (0, 210, 255), 3)
    elif has_shot and shot.cue_to_ghost is not None:
        start, end = shot.cue_to_ghost
        cv2.arrowedLine(output, _pt(start), _pt(end),
                        (0, 210, 255), 3, cv2.LINE_AA, tipLength=0.05)

    # 4a2. Cue deflection after impact (light blue)
    if len(guide.cue_deflection_path) >= 2:
        _draw_poly_path(output, guide.cue_deflection_path, (255, 180, 120), 2)

    # 4b. Ghost ball contact point
    if has_shot and shot.ghost_ball_center is not None:
        ghost_pt = _pt(shot.ghost_ball_center)
        ghost_r  = int(analysis.ball_radius_px) if analysis.ball_radius_px > 4 else 14
        target_ball = None
        if shot.object_ball_id is not None:
            target_ball = next((b for b in analysis.balls if b.id == shot.object_ball_id), None)
            if target_ball is not None:
                ghost_r = max(10, int(target_ball.radius))
        cv2.circle(output, ghost_pt, ghost_r, (150, 150, 0), 2, cv2.LINE_AA)
        cv2.circle(output, ghost_pt, 2, (150, 150, 0), -1, cv2.LINE_AA)
        
        # Ghost -> Target path
        if target_ball is not None:
            cv2.arrowedLine(output, ghost_pt, _pt(target_ball.center),
                            (150, 150, 0), 3, cv2.LINE_AA, tipLength=0.05)

    # 4c. Chained ball paths (green → gold → cyan for 2nd/3rd ball hits)
    if guide.collision_paths:
        chain_colors = [(0, 255, 80), (0, 200, 255), (255, 220, 0), (255, 120, 255)]
        for index, ball_path in enumerate(guide.collision_paths):
            if len(ball_path.path) < 2:
                continue
            color = chain_colors[min(index, len(chain_colors) - 1)]
            _draw_poly_path(output, ball_path.path, color, 3 if index == 0 else 2)
    elif has_shot and shot.bank_path:
        _draw_poly_path(output, shot.bank_path, (0, 255, 80), 3)
    elif guide.object_path:
        _draw_poly_path(output, guide.object_path, (0, 255, 80), 3)

    # 4d. Recommended pocket highlight (removed circular highlight, only text if needed)

    # ------------------------------------------------------------------ #
    # 5.  HUD + POWER BAR
    _draw_hud(output, shot, guide, analysis)
    _draw_power(output, shot.recommended_power or guide.power or guide.shot_speed)
    return output


# ---------------------------------------------------------------------------
# Table geometry drawing (Stage 4 visualisation)
# ---------------------------------------------------------------------------

def _draw_table_geometry(
    output: np.ndarray,
    table: TableDetection,
    analysis: FrameAnalysis,
) -> None:
    """Draw minimal table status badge (all chaotic geometric lines removed)."""
    # --- Table status badge (top-right of table area) ---
    tx, ty, tw, th = table.bounds
    status = f"W:{table.width_px:.0f} H:{table.height_px:.0f}  conf:{table.confidence:.2f}"
    cv2.putText(output, status,
                (tx + 4, ty - 6 if ty > 14 else ty + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 100), 1, cv2.LINE_AA)


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
    """Draw the stats box in the top-left corner, including table geometry info."""
    difficulty_color = _difficulty_color(shot.difficulty)
    box_x2 = _HUD_X + _HUD_W
    hit_label   = f"ball {guide.first_hit_ball_id}" if guide.first_hit_ball_id is not None else "--"
    chain_count = len(guide.collision_paths)
    mode_color = (255, 120, 0) if "Extended" in analysis.operating_mode else (0, 255, 0)

    if analysis.table is not None:
        table_line = f"Table: {analysis.table.width_px:.0f}x{analysis.table.height_px:.0f} (conf: {analysis.table.confidence:.2f})"
    else:
        table_line = "Table: Not detected"

    r_px = analysis.ball_radius_px

    lines = [
        (f"Mode: {analysis.operating_mode}",                              mode_color,      0.65, 2),
        (table_line,                                                      (80, 200, 255),  0.50, 1),
        (f"Ball radius: {r_px:.1f}px",                                    (160, 160, 160), 0.50, 1),
        (f"Pocket: {shot.pocket_label}",                                  (210, 180, 255), 0.58, 1),
        (f"Hit: {hit_label}",                                             (255, 220, 120), 0.58, 1),
        (f"Success: {shot.success_probability:.0f}%",                     (80,  255, 80),  0.58, 1),
        (f"Difficulty: {shot.difficulty:.0f}/100  {shot.difficulty_label}", difficulty_color, 0.65, 2),
        (f"Power: {shot.recommended_power:.0f}%  ({shot.power_label})",   (0,  180, 255),  0.60, 1),
        (f"Speed: {guide.shot_speed:.0f}px",                              (0,  180, 255),  0.55, 1),
        (f"Angle: {shot.cut_angle:.1f} deg",                              (230, 230, 230), 0.55, 1),
        (f"Chain: {chain_count} ball path(s)",                         (230, 230, 230), 0.55, 1),
    ]
    for i, (text, color, scale, thickness) in enumerate(lines):
        cv2.putText(output, text, (_HUD_X, _HUD_Y + _LINE_H * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def _draw_power(output: np.ndarray, power: float) -> None:
    x0 = _PWR_X
    y0 = _HUD_Y + _LINE_H * 11 + 18
    h  = _PWR_BAR_H
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
