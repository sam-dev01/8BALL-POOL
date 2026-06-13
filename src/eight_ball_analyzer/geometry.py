from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .models import AimGuide, BallDetection, BallPath, Pocket, Rail, ShotPrediction, TableDetection


VALID_CUT_ANGLE_LIMIT = 90.0


# ---------------------------------------------------------------------------
# Basic vector utilities
# ---------------------------------------------------------------------------

def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(float) - b.astype(float)))


def unit_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = b.astype(float) - a.astype(float)
    norm = np.linalg.norm(delta)
    if norm < 1e-6:
        return np.array([0.0, 0.0])
    return delta / norm


def angle_degrees(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    va = unit_vector(vertex, a)
    vb = unit_vector(vertex, b)
    dot = float(np.clip(np.dot(va, vb), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def shot_cut_angle(
    cue_ball: BallDetection,
    object_ball: BallDetection,
    ghost_center: np.ndarray,
    pocket_center: np.ndarray,
) -> float:
    approach = unit_vector(cue_ball.center, ghost_center)
    exit_dir = unit_vector(object_ball.center, pocket_center)
    dot = float(np.clip(np.dot(approach, exit_dir), -1.0, 1.0))
    return math.degrees(math.acos(dot))


# ---------------------------------------------------------------------------
# Normalised-coordinate helpers
# ---------------------------------------------------------------------------

def _to_normalised(point: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Map an image-space point to the normalised table space using homography M."""
    src = point.astype(np.float32).reshape(1, 1, 2)
    dst = cv2.perspectiveTransform(src, M)
    return dst.reshape(2).astype(float)


def _to_image(point: np.ndarray, M_inv: np.ndarray) -> np.ndarray:
    """Map a normalised-space point back to image space."""
    src = point.astype(np.float32).reshape(1, 1, 2)
    dst = cv2.perspectiveTransform(src, M_inv)
    return dst.reshape(2).astype(float)


def _path_to_image(path: list[np.ndarray], M_inv: np.ndarray) -> list[np.ndarray]:
    """Convert a list of normalised-space points to image-space points."""
    return [_to_image(p, M_inv) for p in path]


def _rail_in_normalised(rail: Rail, M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (start_n, end_n, normal_n) for a rail in normalised space."""
    s_n = _to_normalised(rail.start, M)
    e_n = _to_normalised(rail.end,   M)
    seg_n = e_n - s_n
    n = np.array([-seg_n[1], seg_n[0]], dtype=float)
    nlen = float(np.linalg.norm(n))
    if nlen > 1e-6:
        n /= nlen
    # Ensure inward (same sign convention as image-space normal through M)
    if float(np.dot(n, rail.normal)) < 0:
        n = -n
    return s_n, e_n, n


# ---------------------------------------------------------------------------
# Rail-normal bank-shot reflection   R = D - 2(D·N)N
# ---------------------------------------------------------------------------

def reflect_vector(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    d_norm = float(np.linalg.norm(direction))
    n_norm = float(np.linalg.norm(normal))
    if d_norm < 1e-6 or n_norm < 1e-6:
        return direction
    d = direction / d_norm
    n = normal / n_norm
    reflected = d - 2.0 * float(np.dot(d, n)) * n
    rlen = float(np.linalg.norm(reflected))
    return reflected / max(rlen, 1e-6)


def _ray_to_rail_in_normalised(
    pos: np.ndarray,
    direction: np.ndarray,
    rails: list[Rail],
    M: np.ndarray,
    ball_radius_n: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Find the nearest rail intersection in normalised space.

    Returns (t, contact_point, inward_normal).
    """
    best_t = 1e9
    best_pt = pos + direction * 1e9
    best_normal = direction.copy()

    for rail in rails:
        s_n, e_n, n = _rail_in_normalised(rail, M)
        # Offset the rail inward by ball_radius_n
        rail_origin = s_n + n * ball_radius_n
        rail_dir    = e_n - s_n

        denom = float(np.cross(direction, rail_dir))
        if abs(denom) < 1e-8:
            continue
        diff = rail_origin - pos
        t = float(np.cross(diff, rail_dir)) / denom
        u = float(np.cross(diff, direction)) / denom
        if t <= 1e-4 or u < 0 or u > 1.0:
            continue
        if t < best_t:
            best_t = t
            best_pt = pos + direction * t
            best_normal = n

    return best_t, best_pt, best_normal


# ---------------------------------------------------------------------------
# Core physics — ghost ball
# ---------------------------------------------------------------------------

def ghost_ball_for_target(object_ball: BallDetection, target: np.ndarray, ball_radius: float) -> np.ndarray:
    aim_dir = unit_vector(target, object_ball.center)
    return object_ball.center.astype(float) + aim_dir * (2.0 * ball_radius)


# ---------------------------------------------------------------------------
# Blocker detection (image space — straightforward segment check)
# ---------------------------------------------------------------------------

def find_blockers(
    cue_ball: BallDetection,
    object_ball: BallDetection,
    ghost_center: np.ndarray,
    pocket: Pocket,
    balls: list[BallDetection],
    ball_radius: float,
    object_path: list[np.ndarray] | None = None,
) -> list[int]:
    if object_path is None or len(object_path) < 2:
        object_path = [object_ball.center.astype(float), pocket.center.astype(float)]

    blocker_ids: list[int] = []
    ignored_ids = {cue_ball.id, object_ball.id}
    threshold = ball_radius * 2.0
    for ball in balls:
        if ball.id in ignored_ids:
            continue
        cue_blocked = _point_near_segment(ball.center, cue_ball.center, ghost_center, threshold)
        obj_blocked = any(
            _point_near_segment(ball.center, object_path[i], object_path[i + 1], threshold)
            for i in range(len(object_path) - 1)
        )
        if cue_blocked or obj_blocked:
            blocker_ids.append(ball.id)
    return blocker_ids


def count_blockers(
    object_ball: BallDetection,
    pocket: Pocket,
    balls: list[BallDetection],
    ball_radius: float,
) -> int:
    start = object_ball.center.astype(float)
    end   = pocket.center.astype(float)
    line  = end - start
    line_len_sq = float(np.dot(line, line))
    if line_len_sq < 1e-6:
        return 0
    count = 0
    for ball in balls:
        if ball.id == object_ball.id:
            continue
        point = ball.center.astype(float)
        projection = float(np.dot(point - start, line) / line_len_sq)
        if projection <= 0.05 or projection >= 0.95:
            continue
        closest = start + projection * line
        if distance(point, closest) < ball_radius * 1.7:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Shot prediction — normalised coordinate physics
# ---------------------------------------------------------------------------

def predict_shot(
    cue_ball: Optional[BallDetection],
    object_ball: Optional[BallDetection],
    pocket: Optional[Pocket],
    all_balls: list[BallDetection],
    table: Optional[TableDetection] = None,
    ball_radius_px: float = 14.0,
) -> ShotPrediction:
    if cue_ball is None or object_ball is None or pocket is None:
        return ShotPrediction(notes=["Select a detected cue ball, object ball, and pocket."])

    avg_radius = max(4.0, ball_radius_px)
    use_normalised = (
        table is not None
        and table.transform_matrix is not None
        and table.inv_transform_matrix is not None
    )

    # --- Direct path (image space) ---
    direct_path  = [object_ball.center.astype(float), pocket.center.astype(float)]
    object_path  = direct_path
    aim_target   = pocket.center.astype(float)

    object_path_blocked = count_blockers(object_ball, pocket, all_balls, avg_radius) > 0

    if object_path_blocked and table is not None:
        if use_normalised:
            bank_candidate = best_one_rail_bank_path_normalised(
                object_ball, pocket, avg_radius, table
            )
        else:
            bank_candidate = best_one_rail_bank_path_image(
                object_ball.center, pocket.center, avg_radius, table.bounds
            )
        if len(bank_candidate) == 3:
            object_path = bank_candidate
            aim_target  = bank_candidate[1]

    ghost_center    = ghost_ball_for_target(object_ball, aim_target, avg_radius)
    cue_distance    = distance(cue_ball.center, ghost_center)
    object_distance = sum(distance(object_path[i], object_path[i + 1]) for i in range(len(object_path) - 1))
    cut_angle       = shot_cut_angle(cue_ball, object_ball, ghost_center, aim_target)

    if cut_angle > VALID_CUT_ANGLE_LIMIT:
        return ShotPrediction(
            valid=False,
            object_ball_id=object_ball.id,
            pocket_id=pocket.id,
            pocket_label=pocket.label,
            ghost_ball_center=ghost_center,
            cut_angle=cut_angle,
            difficulty=100.0,
            difficulty_label="Impossible",
            notes=[f"Rejected: cut angle {cut_angle:.1f} deg is above 90 deg."],
        )

    blocker_ids = find_blockers(cue_ball, object_ball, ghost_center, pocket, all_balls, avg_radius, object_path)

    cue_after_impact = predict_cue_after_impact(
        cue_ball, object_ball, ghost_center,
        table,
        max_bounces=3,
    )
    rail_count  = max(0, len(object_path) - 2)
    power       = estimate_power(cue_distance, object_distance, cut_angle)
    difficulty  = score_difficulty(cue_distance, object_distance, cut_angle, len(blocker_ids), rail_count)

    notes = [
        f"Recommended pocket: {pocket.label}",
        f"Success probability: {success_probability(difficulty):.0f}%",
        f"Cut angle: {cut_angle:.1f} deg",
        f"Cue travel: {cue_distance:.0f}px",
        f"Object travel: {object_distance:.0f}px",
        f"Power: {power:.0f}% ({power_label(power)})",
    ]
    if blocker_ids:
        notes.append(f"Blockers: {', '.join(str(bid) for bid in blocker_ids)}")

    return ShotPrediction(
        valid=True,
        object_ball_id=object_ball.id,
        pocket_id=pocket.id,
        pocket_label=pocket.label,
        ghost_ball_center=ghost_center,
        cue_to_ghost=(cue_ball.center, ghost_center),
        object_to_pocket=(object_path[0], object_path[-1]) if len(object_path) >= 2 else (object_ball.center, pocket.center),
        cue_after_impact=cue_after_impact,
        bank_path=object_path,
        blocker_ids=blocker_ids,
        cut_angle=cut_angle,
        cue_path_length=cue_distance,
        object_path_length=object_distance,
        rail_count=rail_count,
        recommended_power=power,
        power_label=power_label(power),
        difficulty=difficulty,
        difficulty_label=difficulty_label(difficulty),
        success_probability=success_probability(difficulty),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Pocket recommendation
# ---------------------------------------------------------------------------

def recommend_pocket(
    cue_ball: Optional[BallDetection],
    object_ball: Optional[BallDetection],
    pockets: list[Pocket],
    all_balls: list[BallDetection],
    table: Optional[TableDetection] = None,
    ball_radius_px: float = 14.0,
) -> Pocket | None:
    if cue_ball is None or object_ball is None or not pockets:
        return None
    scored: list[tuple[float, Pocket]] = []
    for pocket in pockets:
        score = shot_score(cue_ball, object_ball, pocket, all_balls, table, ball_radius_px)
        if score is not None:
            scored.append((score, pocket))
    if not scored:
        return None
    return min(scored, key=lambda item: item[0])[1]


def recommend_best_shot(
    cue_ball: BallDetection,
    balls: list[BallDetection],
    pockets: list[Pocket],
    table: Optional[TableDetection] = None,
    ball_radius_px: float = 14.0,
) -> tuple[BallDetection | None, Pocket | None]:
    """Finds the absolute best valid object ball and pocket combination."""
    best_score = float('inf')
    best_target = None
    best_pocket = None
    
    from .models import BallKind
    candidates = [b for b in balls if b.id != cue_ball.id and b.kind in {BallKind.SOLID, BallKind.STRIPE, BallKind.EIGHT}]
    
    for ball in candidates:
        pocket = recommend_pocket(cue_ball, ball, pockets, balls, table, ball_radius_px)
        if pocket is not None:
            score = shot_score(cue_ball, ball, pocket, balls, table, ball_radius_px)
            if score is not None and score < best_score:
                best_score = score
                best_target = ball
                best_pocket = pocket
                
    return best_target, best_pocket


def shot_score(
    cue_ball: BallDetection,
    object_ball: BallDetection,
    pocket: Pocket,
    all_balls: list[BallDetection],
    table: Optional[TableDetection] = None,
    ball_radius_px: float = 14.0,
) -> float | None:
    avg_radius = max(4.0, ball_radius_px)
    ghost_center = ghost_ball_for_target(object_ball, pocket.center, avg_radius)
    cut_angle = shot_cut_angle(cue_ball, object_ball, ghost_center, pocket.center)
    if cut_angle > VALID_CUT_ANGLE_LIMIT:
        return None
    blockers = len(find_blockers(cue_ball, object_ball, ghost_center, pocket, all_balls, avg_radius))
    cue_distance    = distance(cue_ball.center, ghost_center)
    object_distance = distance(object_ball.center, pocket.center)
    direct_score    = score_difficulty(cue_distance, object_distance, cut_angle, 0, 0)
    
    if blockers == 0:
        return direct_score
        
    # Direct path is blocked, evaluate 1-rail bank
    bank_path = []
    if table is not None:
        if table.transform_matrix is not None and table.inv_transform_matrix is not None:
            bank_path = best_one_rail_bank_path_normalised(object_ball, pocket, avg_radius, table)
        else:
            bank_path = best_one_rail_bank_path_image(object_ball.center, pocket.center, avg_radius, table.bounds)
            
    if len(bank_path) == 3:
        bank_aim = bank_path[1]
        bank_ghost = ghost_ball_for_target(object_ball, bank_aim, avg_radius)
        bank_cut = shot_cut_angle(cue_ball, object_ball, bank_ghost, bank_aim)
        if bank_cut <= VALID_CUT_ANGLE_LIMIT:
            bank_blockers = len(find_blockers(cue_ball, object_ball, bank_ghost, pocket, all_balls, avg_radius, bank_path))
            if bank_blockers == 0:
                b_cue_dist = distance(cue_ball.center, bank_ghost)
                b_obj_dist = distance(bank_path[0], bank_path[1]) + distance(bank_path[1], bank_path[2])
                return score_difficulty(b_cue_dist, b_obj_dist, bank_cut, 0, 1) + 20.0
                
    # Impossible shot (both direct and bank are physically obstructed or invalid cut angle)
    return None


def _best_one_rail_bank_score(
    object_ball: BallDetection,
    pocket: Pocket,
    table: Optional[TableDetection],
    direct_distance: float,
    ball_radius_px: float,
) -> float | None:
    if table is None:
        return None
    use_normalised = (
        table.transform_matrix is not None and table.inv_transform_matrix is not None
    )
    if use_normalised:
        path = best_one_rail_bank_path_normalised(object_ball, pocket, ball_radius_px, table)
    else:
        path = best_one_rail_bank_path_image(object_ball.center, pocket.center, ball_radius_px, table.bounds)
    if len(path) < 3:
        return None
    bank_distance = sum(distance(path[i], path[i + 1]) for i in range(len(path) - 1))
    if bank_distance > direct_distance * 1.8:
        return None
    return min(100.0, bank_distance / 1400.0 * 100.0 + 18.0)


# ---------------------------------------------------------------------------
# Bank shot helpers — normalised space (preferred)
# ---------------------------------------------------------------------------

def best_one_rail_bank_path_normalised(
    object_ball: BallDetection,
    pocket: Pocket,
    ball_radius_px: float,
    table: TableDetection,
) -> list[np.ndarray]:
    """Compute the best single-rail bank path using normalised coordinates.

    Uses vector reflection  R = D - 2(D·N)N  with each rail's inward normal.
    Returns 3 image-space points [start, contact, end] or [] on failure.
    """
    M     = table.transform_matrix
    M_inv = table.inv_transform_matrix
    if M is None or M_inv is None or not table.rails:
        return best_one_rail_bank_path_image(object_ball.center, pocket.center, ball_radius_px, table.bounds)

    # Normalised ball radius (approx)
    norm_w = max(table.width_px, 1.0)
    norm_h = max(table.height_px, 1.0)
    ball_r_n = ball_radius_px / ((norm_w + norm_h) / 2.0)

    start_n  = _to_normalised(object_ball.center, M)
    target_n = _to_normalised(pocket.center, M)

    best_path: list[np.ndarray] = []
    best_dist = 1e9

    for rail in table.rails:
        s_n, e_n, normal_n = _rail_in_normalised(rail, M)
        # Mirror the target across the rail line (image-method in normalised space)
        mirrored_n = _mirror_across_line(target_n, s_n, e_n)
        dir_to_mirror = mirrored_n - start_n
        dir_len = float(np.linalg.norm(dir_to_mirror))
        if dir_len < 1e-6:
            continue
        dir_to_mirror /= dir_len

        # Find intersection with the rail segment
        contact_n = _ray_segment_intersect(start_n, dir_to_mirror, s_n, e_n)
        if contact_n is None:
            continue
        # Validate direction: contact must be reachable (positive t)
        t = float(np.dot(contact_n - start_n, dir_to_mirror))
        if t <= 0:
            continue

        # Verify reflection direction points toward target
        reflected = reflect_vector(dir_to_mirror, normal_n)
        to_target = unit_vector(contact_n, target_n)
        if float(np.dot(reflected, to_target)) < 0.3:
            continue

        # Map back to image space
        start_img   = _to_image(start_n,   M_inv)
        contact_img = _to_image(contact_n, M_inv)
        target_img  = _to_image(target_n,  M_inv)

        total_d = distance(start_img, contact_img) + distance(contact_img, target_img)
        if total_d < best_dist:
            best_dist = total_d
            best_path = [start_img, contact_img, target_img]

    return best_path


def _mirror_across_line(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> np.ndarray:
    """Reflect `point` across the infinite line defined by line_start→line_end."""
    d = line_end - line_start
    d_len_sq = float(np.dot(d, d))
    if d_len_sq < 1e-12:
        return point.copy()
    t = float(np.dot(point - line_start, d)) / d_len_sq
    foot = line_start + d * t
    return 2.0 * foot - point


def _ray_segment_intersect(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
) -> np.ndarray | None:
    """Return the intersection point of a ray with a finite segment, or None."""
    seg_dir = seg_end - seg_start
    denom = float(np.cross(ray_dir, seg_dir))
    if abs(denom) < 1e-8:
        return None
    diff = seg_start - ray_origin
    t = float(np.cross(diff, seg_dir)) / denom
    u = float(np.cross(diff, ray_dir)) / denom
    if t < 0 or u < 0 or u > 1.0:
        return None
    return ray_origin + ray_dir * t


# ---------------------------------------------------------------------------
# Bank shot helpers — image space (fallback)
# ---------------------------------------------------------------------------

def best_one_rail_bank_path_image(
    start: np.ndarray,
    target: np.ndarray,
    radius: float,
    bounds: tuple[int, int, int, int],
) -> list[np.ndarray]:
    """Single-rail bank using axis-aligned mirror reflection in image space."""
    x, y, w, h = bounds
    rails = [
        ("left",   x + radius),
        ("right",  x + w - radius),
        ("top",    y + radius),
        ("bottom", y + h - radius),
    ]
    candidates: list[list[np.ndarray]] = []
    for rail, value in rails:
        mirrored = target.astype(float).copy()
        if rail == "left":
            mirrored[0] = 2.0 * value - mirrored[0]
            normal = np.array([1.0, 0.0])
        elif rail == "right":
            mirrored[0] = 2.0 * value - mirrored[0]
            normal = np.array([-1.0, 0.0])
        elif rail == "top":
            mirrored[1] = 2.0 * value - mirrored[1]
            normal = np.array([0.0, 1.0])
        else:
            mirrored[1] = 2.0 * value - mirrored[1]
            normal = np.array([0.0, -1.0])

        direction = mirrored - start.astype(float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        direction /= norm

        if rail in {"left", "right"}:
            if abs(direction[0]) < 1e-6:
                continue
            t = (value - start[0]) / direction[0]
        else:
            if abs(direction[1]) < 1e-6:
                continue
            t = (value - start[1]) / direction[1]

        if t <= 0:
            continue
        contact = start.astype(float) + direction * t
        if not _point_inside_bounds(contact, bounds, radius):
            continue
        # Validate via R = D - 2(D·N)N
        reflected = reflect_vector(direction, normal)
        to_target = unit_vector(contact, target)
        if float(np.dot(reflected, to_target)) < 0.3:
            continue
        candidates.append([start.astype(float), contact, target.astype(float)])

    if not candidates:
        return []
    return min(candidates, key=lambda p: distance(p[0], p[1]) + distance(p[1], p[2]))


# ---------------------------------------------------------------------------
# Cue-path tracing
# ---------------------------------------------------------------------------

def _find_first_collision(
    pos: np.ndarray,
    direction: np.ndarray,
    balls: list[BallDetection],
    rails: list[Rail],
    pockets: list[Pocket],
    radius: float,
) -> tuple[str, np.ndarray, float, any, np.ndarray]:
    best_t = 1e9
    best_hit = ("NONE", pos, 1e9, None, np.zeros(2))

    # 1. Balls
    for ball in balls:
        C = ball.center.astype(float)
        V = pos - C
        # BUG 4 FIX: ghost-ball collision distance = sum of both radii, not max()
        effective_r = radius + ball.radius
        b = 2.0 * float(np.dot(V, direction))
        c = float(np.dot(V, V)) - effective_r ** 2
        discriminant = b*b - 4*c
        if discriminant >= 0:
            sqrt_d = math.sqrt(discriminant)
            t1 = (-b - sqrt_d) / 2.0
            t2 = (-b + sqrt_d) / 2.0
            t = None
            if c < 0:
                if b < 0:
                    t = 1e-4
            else:
                if t1 > 1e-4:
                    t = t1
                elif t2 > 1e-4:
                    t = t2
                    
            if t is not None and t < best_t:
                hit_pt = pos + direction * t
                normal = unit_vector(C, hit_pt) 
                best_t = t
                best_hit = ("BALL", hit_pt, t, ball, normal)

    # 2. Rails
    for rail in rails:
        r_dir = rail.end - rail.start
        r_len = float(np.linalg.norm(r_dir))
        if r_len < 1e-6: continue
        r_dir /= r_len
        r_start = rail.start + rail.normal * radius - r_dir * (radius * 1.5)
        r_end   = rail.end   + rail.normal * radius + r_dir * (radius * 1.5)
        
        intersect = _ray_segment_intersect(pos, direction, r_start, r_end)
        if intersect is not None:
            t = float(np.dot(intersect - pos, direction))
            if 1e-4 < t < best_t:
                best_t = t
                best_hit = ("RAIL", intersect, t, rail, rail.normal)

    # 3. Pockets
    for pocket in pockets:
        C = pocket.center.astype(float)
        V = pos - C
        pr = pocket.mouth_radius
        b = 2.0 * float(np.dot(V, direction))
        c = float(np.dot(V, V)) - pr ** 2
        discriminant = b*b - 4*c
        if discriminant >= 0:
            sqrt_d = math.sqrt(discriminant)
            t1 = (-b - sqrt_d) / 2.0
            t2 = (-b + sqrt_d) / 2.0
            t = None
            if t1 > 1e-4: t = t1
            elif t2 > 1e-4: t = t2
            if t is not None and t < best_t:
                hit_pt = pos + direction * t
                best_t = t
                best_hit = ("POCKET", hit_pt, t, pocket, np.zeros(2))

    return best_hit


def trace_cue_path(
    start: np.ndarray,
    direction: np.ndarray,
    radius: float,
    table: TableDetection,
    pockets: list[Pocket],
    balls: list[BallDetection],
    max_bounces: int,
) -> tuple[list[np.ndarray], Optional[BallDetection], Optional[np.ndarray], np.ndarray, list[dict]]:
    path = [start.copy()]
    pos  = start.copy()
    vel  = direction.copy()
    debug_markers = []

    for i in range(max_bounces + 1):
        ctype, hit_point, dist, obj, normal = _find_first_collision(
            pos, vel, balls, table.rails, pockets, radius
        )
        
        if ctype == "NONE":
            break
            
        path.append(hit_point)
        
        if i == 0:
            debug_markers.append({
                "point": hit_point,
                "type": ctype,
                "distance": dist
            })

        if ctype == "BALL":
            return path, obj, hit_point, vel, debug_markers
        elif ctype == "POCKET":
            return path, None, hit_point, vel, debug_markers
        elif ctype == "RAIL":
            vel = reflect_vector(vel, normal)
            pos = hit_point + vel * 1.5

    return path, None, None, vel, debug_markers


def trace_path_with_reflections(
    start: np.ndarray,
    direction: np.ndarray,
    radius: float,
    table: TableDetection,
    max_bounces: int = 3,
) -> list[np.ndarray]:
    path = [start.astype(float).copy()]
    pos  = start.astype(float).copy()
    vel  = direction.astype(float).copy()
    norm = float(np.linalg.norm(vel))
    if norm < 1e-6:
        return path
    vel /= norm

    for _ in range(max_bounces + 1):
        wall_t, wall_point, wall_normal = _ray_to_table_rails(pos, vel, table.rails, radius)
        if wall_t <= 0:
            break
        path.append(wall_point)
        vel = reflect_vector(vel, wall_normal)
        pos = wall_point + vel * max(1.5, radius * 0.15)
    return path


def predict_cue_after_impact(
    cue_ball: BallDetection,
    object_ball: BallDetection,
    ghost_center: np.ndarray,
    table: TableDetection | None = None,
    max_bounces: int = 3,
) -> list[np.ndarray]:
    incoming = unit_vector(cue_ball.center, ghost_center)
    object_dir = unit_vector(ghost_center, object_ball.center)
    if np.linalg.norm(incoming) < 1e-6 or np.linalg.norm(object_dir) < 1e-6:
        return []
    cue_deflection = incoming - np.dot(incoming, object_dir) * object_dir
    norm = float(np.linalg.norm(cue_deflection))
    if norm < 1e-6:
        return []
    cue_deflection /= norm
    impact_center = ghost_center.astype(float)
    if table is None:
        length = max(80.0, cue_ball.radius * 7.0)
        return [impact_center, impact_center + cue_deflection * length]
    return trace_path_with_reflections(impact_center, cue_deflection, cue_ball.radius, table, max_bounces)


def object_exit_direction(
    incoming_direction: np.ndarray,
    object_center: np.ndarray,
    hit_point: np.ndarray,
) -> np.ndarray:
    normal = unit_vector(hit_point, object_center)
    if float(np.linalg.norm(normal)) > 1e-6:
        return normal
    return unit_vector(np.zeros(2), incoming_direction)


def trace_object_after_hit(
    object_ball: BallDetection,
    hit_point: np.ndarray,
    incoming_direction: np.ndarray,
    table: TableDetection,
) -> list[np.ndarray]:
    contact_normal = unit_vector(hit_point, object_ball.center)
    if np.linalg.norm(contact_normal) < 1e-6:
        contact_normal = incoming_direction
    return trace_path_with_reflections(object_ball.center.astype(float), contact_normal, object_ball.radius, table, max_bounces=3)


# ---------------------------------------------------------------------------
# Collision chain
# ---------------------------------------------------------------------------

def power_to_shot_speed(power: float, table: TableDetection) -> float:
    x, y, w, h = table.bounds
    table_span = float(np.hypot(w, h))
    return float(np.clip(power / 100.0, 0.15, 1.0)) * table_span * 0.55


def trace_ball_collision_chain(
    cue_ball: BallDetection,
    balls: list[BallDetection],
    table: TableDetection,
    cue_direction: np.ndarray,
    power: float,
    pockets: list[Pocket],
    max_chain_depth: int = 2,
    ball_radius_px: float = 14.0,
    max_bounces: int = 3,
    locked_target_id: Optional[int] = None,
) -> AimGuide:
    guide = AimGuide(power=power)
    direction = cue_direction.astype(float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        guide.notes.append("Cue direction not detected.")
        return guide

    direction /= norm
    guide.cue_direction = direction
    guide.shot_speed = power_to_shot_speed(power, table)
    radius = max(4.0, ball_radius_px)
    remaining_balls = [b for b in balls if b.id != cue_ball.id]

    cue_path, first_hit, ghost_center, incoming, _ = trace_cue_path(
        cue_ball.center.astype(float), direction, radius, table, pockets, remaining_balls, max_bounces=0
    )
    guide.cue_path = cue_path

    # Apply Target Lock
    target_ball = first_hit
    if locked_target_id is not None and (target_ball is None or target_ball.id != locked_target_id):
        locked_ball = next((b for b in balls if b.id == locked_target_id), None)
        if locked_ball is not None:
            # Check if aim is reasonably close to the locked ball
            p1 = cue_path[0]
            p2 = p1 + direction * 2000.0
            diff = p2 - p1
            dist = float(np.linalg.norm(np.cross(diff, p1 - locked_ball.center.astype(float))) / np.linalg.norm(diff))
            
            if dist < radius * 4.0:
                target_ball = locked_ball
                # Recalculate ghost_center
                c_diff = p1 - locked_ball.center.astype(float)
                b = 2.0 * float(np.dot(c_diff, direction))
                c = float(np.dot(c_diff, c_diff)) - (2.0 * radius) ** 2
                disc = b*b - 4*c
                if disc >= 0:
                    t = (-b - math.sqrt(disc)) / 2.0
                    ghost_center = p1 + direction * t
                    guide.cue_path = [p1, ghost_center]
                    incoming = direction

    if target_ball is None or ghost_center is None:
        guide.notes.append("No ball in cue path.")
        return guide

    guide.first_hit_ball_id = target_ball.id
    guide.first_hit_point = ghost_center  # Visually, the cue ball stops at the ghost ball center

    # The game's default object-ball aim guide direction is from ghost_center to object_center
    obj_center = target_ball.center.astype(float)
    obj_dir = unit_vector(ghost_center, obj_center)
    
    # physical contact point between cue and object ball
    contact_point = obj_center - obj_dir * target_ball.radius
    guide.physical_contact_point = contact_point
    
    # Start Green Path from the physical contact point
    edge_start_pt = contact_point
    
    green_remaining = [b for b in balls if b.id != cue_ball.id and b.id != target_ball.id]
    g_type, g_hit_point, _, g_obj, g_normal = _find_first_collision(
        edge_start_pt, obj_dir, green_remaining, table.rails, pockets, radius
    )

    if g_type == "NONE":
        guide.object_path = [edge_start_pt, edge_start_pt + obj_dir * 2000.0]
    else:
        guide.object_path = [edge_start_pt, g_hit_point]

        # Yellow Path: Reflections
        if g_type == "RAIL":
            reflected_dir = reflect_vector(obj_dir, g_normal)
            yellow_path = trace_path_with_reflections(
                g_hit_point, reflected_dir, radius, table, max_bounces=max_bounces
            )
            guide.object_reflection_path = yellow_path
            
        # Orange Path: Secondary Collisions
        elif g_type == "BALL" and g_obj is not None:
            guide.secondary_ball_id = g_obj.id
            # For secondary ball, use the same geometric logic
            sec_ghost_center = g_hit_point
            sec_dir = unit_vector(sec_ghost_center, g_obj.center.astype(float))
            sec_contact_point = g_obj.center.astype(float) - sec_dir * g_obj.radius
            sec_type, sec_hit_point, _, _, _ = _find_first_collision(
                sec_contact_point, sec_dir, [], table.rails, pockets, radius
            )
            if sec_type != "NONE":
                guide.secondary_path = [sec_contact_point, sec_hit_point]
            else:
                guide.secondary_path = [sec_contact_point, sec_contact_point + sec_dir * 2000.0]

    # Calculate cue ball deflection starting from ghost_center (cue ball's position at impact)
    # BUG 4 FIX: was incorrectly starting from contact_point (object ball surface edge)
    cue_deflection_dir = incoming - np.dot(incoming, obj_dir) * obj_dir
    norm_deflection = float(np.linalg.norm(cue_deflection_dir))
    if norm_deflection > 1e-6:
        cue_deflection_dir /= norm_deflection
        deflection_start = ghost_center.astype(float)  # cue ball center at moment of impact
        deflection_remaining = [b for b in balls if b.id != cue_ball.id and b.id != target_ball.id]
        def_type, def_hit_point, _, _, def_normal = _find_first_collision(
            deflection_start, cue_deflection_dir, deflection_remaining, table.rails, pockets, radius
        )
        if def_type == "NONE":
            guide.cue_deflection_path = [deflection_start, deflection_start + cue_deflection_dir * 150.0]
        else:
            guide.cue_deflection_path = [deflection_start, def_hit_point]
            if def_type == "RAIL":
                r_dir = reflect_vector(cue_deflection_dir, def_normal)
                guide.cue_deflection_path.append(def_hit_point + r_dir * 50.0)

    return guide


def fill_guide_from_balls(
    guide: AimGuide,
    cue_ball: BallDetection,
    object_ball: BallDetection,
    table: TableDetection | None,
) -> None:
    start = cue_ball.center.astype(float)
    end   = object_ball.center.astype(float)
    direction = unit_vector(start, end)
    if float(np.linalg.norm(direction)) < 1e-6:
        return
    guide.cue_direction   = direction
    guide.cue_path        = [start, end]
    guide.first_hit_ball_id  = object_ball.id
    guide.first_hit_point    = end
    if table is not None:
        guide.object_path = trace_object_after_hit(object_ball, end, direction, table)
    guide.notes = [n for n in guide.notes if "Cue direction not detected" not in n]
    guide.notes.append("Using cue-to-target fallback aim.")


# ---------------------------------------------------------------------------
# Difficulty / power scoring
# ---------------------------------------------------------------------------

def score_difficulty(
    cue_distance: float,
    object_distance: float,
    cut_angle: float,
    blockers: int,
    rail_count: int = 0,
) -> float:
    cut_component      = min(100.0, cut_angle / 90.0 * 100.0)
    distance_component = min(100.0, (cue_distance + object_distance) / 1400.0 * 100.0)
    blocker_component  = min(100.0, blockers * 25.0)
    rail_component     = min(100.0, rail_count * 45.0)
    difficulty = (
        cut_component      * 0.40
        + distance_component * 0.20
        + blocker_component  * 0.25
        + rail_component     * 0.15
    )
    return float(np.clip(difficulty, 0.0, 100.0))


def difficulty_label(score: float) -> str:
    if score < 20: return "Easy"
    if score < 40: return "Medium"
    if score < 60: return "Hard"
    if score < 80: return "Expert"
    return "Extreme"


def estimate_power(cue_distance: float, object_distance: float, cut_angle: float) -> float:
    raw = object_distance * 0.70 + cue_distance * 0.30 + cut_angle * 0.15
    return float(np.clip(raw / 8.0, 20.0, 100.0))


def power_label(power: float) -> str:
    if power < 40: return "Soft"
    if power < 70: return "Medium"
    if power < 90: return "Firm"
    return "Power"


def success_probability(difficulty: float) -> float:
    return float(np.clip(100.0 - difficulty, 5.0, 95.0))


# ---------------------------------------------------------------------------
# Private low-level helpers
# ---------------------------------------------------------------------------

def _ray_ball_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    ball: BallDetection,
    collision_radius: float,
) -> float | None:
    hit = _first_ball_on_ray(origin, direction, [ball], collision_radius, 1e9)
    return hit[1] if hit is not None else None


def _trace_ball_path_until_hit(
    start: np.ndarray,
    direction: np.ndarray,
    radius: float,
    table: TableDetection,
    pockets: list[Pocket],
    balls: list[BallDetection],
    ignore_ids: set[int],
    max_bounces: int = 3,
) -> tuple[list[np.ndarray], BallDetection | None]:
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    pos  = start.astype(float).copy()
    path: list[np.ndarray] = [pos.copy()]
    remaining = [b for b in balls if b.id not in ignore_ids]

    for _ in range(max_bounces + 1):
        ctype, hit_point, dist, obj, normal = _find_first_collision(
            pos, direction, remaining, table.rails, pockets, radius
        )
        if ctype == "NONE" or ctype == "POCKET":
            if ctype == "POCKET":
                path.append(hit_point.astype(float))
            break
        elif ctype == "BALL":
            path.append(hit_point.astype(float))
            return path, obj
        elif ctype == "RAIL":
            path.append(hit_point.astype(float))
            direction = reflect_vector(direction, normal)
            pos = hit_point + direction * max(1.5, radius * 0.15)

    return path, None


def _first_ball_on_ray(
    pos: np.ndarray,
    direction: np.ndarray,
    balls: list[BallDetection],
    collision_radius: float,
    max_t: float,
) -> tuple[BallDetection, float] | None:
    best: tuple[BallDetection, float] | None = None
    for ball in balls:
        to_ball = ball.center.astype(float) - pos
        t = float(np.dot(to_ball, direction))
        if t <= 0 or t >= max_t:
            continue
        closest = pos + direction * t
        miss    = distance(closest, ball.center)
        allowed = max(collision_radius, ball.radius * 1.8)
        if miss <= allowed and (best is None or t < best[1]):
            best = (ball, t)
    return best


def _point_near_segment(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    threshold: float,
) -> bool:
    line = end.astype(float) - start.astype(float)
    line_len_sq = float(np.dot(line, line))
    if line_len_sq < 1e-6:
        return False
    projection = float(np.dot(point.astype(float) - start.astype(float), line) / line_len_sq)
    if projection <= 0.03 or projection >= 0.97:
        return False
    closest = start.astype(float) + projection * line
    return distance(point, closest) < threshold


def _point_inside_bounds(
    point: np.ndarray,
    bounds: tuple[int, int, int, int],
    radius: float,
) -> bool:
    x, y, w, h = bounds
    return x + radius <= point[0] <= x + w - radius and y + radius <= point[1] <= y + h - radius


def _ray_to_table_rails(
    pos: np.ndarray,
    direction: np.ndarray,
    rails: list[Rail],
    radius: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    best_t = 1e9
    best_pt = pos.copy()
    best_normal = np.array([0.0, 0.0])

    for rail in rails:
        # Extend the rail segment slightly to prevent gaps at corners
        r_dir = rail.end - rail.start
        r_len = float(np.linalg.norm(r_dir))
        if r_len < 1e-6:
            continue
        r_dir /= r_len
        
        # Offset inward
        r_start = rail.start + rail.normal * radius - r_dir * (radius * 1.5)
        r_end   = rail.end   + rail.normal * radius + r_dir * (radius * 1.5)
        
        intersect = _ray_segment_intersect(pos, direction, r_start, r_end)
        if intersect is not None:
            t = float(np.dot(intersect - pos, direction))
            if 1e-4 < t < best_t:
                best_t = t
                best_pt = intersect
                best_normal = rail.normal

    if best_t == 1e9:
        return 0.0, pos.copy(), np.array([0.0, 0.0])
    return best_t, best_pt, best_normal


# Keep old name for any external references
def build_aim_guide(
    cue_ball: Optional[BallDetection],
    balls: list[BallDetection],
    table: Optional[TableDetection],
    cue_direction: Optional[np.ndarray],
    power: float,
    max_bounces: int = 3,
) -> AimGuide:
    if cue_ball is None or table is None or cue_direction is None:
        return AimGuide(power=power, notes=["Cue direction not detected."])
    return trace_ball_collision_chain(cue_ball, balls, table, cue_direction, power, [], max_bounces=max_bounces)
