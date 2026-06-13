import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eight_ball_analyzer.export import analysis_to_dict
from eight_ball_analyzer.geometry import (
    predict_shot,
    reflect_vector,
    trace_ball_collision_chain,
)
from eight_ball_analyzer.models import (
    BallDetection,
    BallKind,
    FrameAnalysis,
    Pocket,
    TableDetection,
)


def test_predict_shot_basic_blockers_and_difficulty():
    cue = BallDetection(id=0, center=np.array([100.0, 100.0]), radius=10.0,
                        kind=BallKind.CUE, confidence=0.9)
    obj = BallDetection(id=1, center=np.array([200.0, 100.0]), radius=10.0,
                        kind=BallKind.SOLID, confidence=0.9)
    blocker = BallDetection(id=2, center=np.array([260.0, 100.0]), radius=10.0,
                            kind=BallKind.STRIPE, confidence=0.9)
    pocket = Pocket(id=0, center=np.array([400.0, 100.0]),
                    mouth_radius=18.0, label="top-left")

    shot = predict_shot(cue, obj, pocket, [cue, obj, blocker])
    assert shot.ghost_ball_center is not None
    assert shot.blocker_ids == [2]
    assert 0.0 <= shot.difficulty <= 100.0
    assert 0.0 <= shot.recommended_power <= 100.0


def test_reflect_vector_horizontal_wall():
    reflected = reflect_vector(np.array([1.0, 1.0]), np.array([-1.0, 0.0]))
    assert reflected[0] < 0


def test_analysis_to_dict_round_trip():
    cue = BallDetection(id=0, center=np.array([100.0, 100.0]), radius=10.0,
                        kind=BallKind.CUE, confidence=0.9)
    obj = BallDetection(id=1, center=np.array([200.0, 100.0]), radius=10.0,
                        kind=BallKind.SOLID, confidence=0.9)
    blocker = BallDetection(id=2, center=np.array([260.0, 100.0]), radius=10.0,
                            kind=BallKind.STRIPE, confidence=0.9)
    pocket = Pocket(id=0, center=np.array([400.0, 100.0]),
                    mouth_radius=18.0, label="top-left")
    shot = predict_shot(cue, obj, pocket, [cue, obj, blocker])

    report = analysis_to_dict(
        FrameAnalysis(frame_index=3, table=None,
                      balls=[cue, obj], pockets=[pocket], shot=shot)
    )
    assert report["shot"]["blocker_ids"] == [2]
    assert report["balls"][0]["type"] == "cue"


def test_trace_ball_collision_chain_first_hit():
    cue = BallDetection(id=0, center=np.array([100.0, 100.0]), radius=10.0,
                        kind=BallKind.CUE, confidence=0.9)
    obj = BallDetection(id=1, center=np.array([200.0, 100.0]), radius=10.0,
                        kind=BallKind.SOLID, confidence=0.9)
    blocker = BallDetection(id=2, center=np.array([300.0, 100.0]), radius=10.0,
                            kind=BallKind.STRIPE, confidence=0.9)
    pocket = Pocket(id=0, center=np.array([400.0, 100.0]),
                    mouth_radius=18.0, label="top-left")
    table = TableDetection(
        polygon=np.array([[0, 0], [500, 0], [500, 300], [0, 300]], dtype=np.int32),
        bounds=(0, 0, 500, 300),
        confidence=0.9,
    )
    direction = np.array([1.0, 0.0])
    guide = trace_ball_collision_chain(
        cue, [obj, blocker], table, direction, power=60.0, pockets=[pocket]
    )
    assert guide.first_hit_ball_id == 1
    assert len(guide.object_path) >= 2
    assert len(guide.cue_path) >= 2


def test_predict_shot_rejects_blocked_bank_first_leg():
    """BUG 3 regression: a bank candidate whose first leg is itself blocked
    must not be returned as the chosen path."""
    table = TableDetection(
        polygon=np.array([[0, 0], [500, 0], [500, 300], [0, 300]], dtype=np.int32),
        bounds=(0, 0, 500, 300),
        confidence=0.9,
    )
    cue = BallDetection(id=0, center=np.array([50.0, 150.0]), radius=10.0,
                        kind=BallKind.CUE, confidence=0.9)
    obj = BallDetection(id=1, center=np.array([250.0, 150.0]), radius=10.0,
                        kind=BallKind.SOLID, confidence=0.9)
    direct_blocker = BallDetection(id=2, center=np.array([350.0, 150.0]),
                                   radius=10.0, kind=BallKind.STRIPE, confidence=0.9)
    # Place a second blocker on the first leg of any plausible bank reflection.
    bank_blocker = BallDetection(id=3, center=np.array([260.0, 120.0]),
                                 radius=10.0, kind=BallKind.STRIPE, confidence=0.9)
    pocket = Pocket(id=0, center=np.array([480.0, 150.0]),
                    mouth_radius=18.0, label="right")

    shot = predict_shot(
        cue, obj, pocket, [cue, obj, direct_blocker, bank_blocker], table=table
    )
    # Either no rail bounce was chosen, or the path returned does not pass
    # through the bank_blocker neighbourhood.
    if shot.rail_count > 0 and len(shot.bank_path) >= 2:
        first_leg = (np.asarray(shot.bank_path[0]), np.asarray(shot.bank_path[1]))
        seg = first_leg[1] - first_leg[0]
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq > 1e-6:
            t = float(np.dot(bank_blocker.center - first_leg[0], seg) / seg_len_sq)
            t = max(0.0, min(1.0, t))
            closest = first_leg[0] + seg * t
            dist = float(np.linalg.norm(bank_blocker.center - closest))
            assert dist >= 10.0 * 1.7, (
                f"bank first leg passes through a blocker (dist={dist:.1f})"
            )


if __name__ == "__main__":
    test_predict_shot_basic_blockers_and_difficulty()
    test_reflect_vector_horizontal_wall()
    test_analysis_to_dict_round_trip()
    test_trace_ball_collision_chain_first_hit()
    test_predict_shot_rejects_blocked_bank_first_leg()
    print("Analysis math test passed")
