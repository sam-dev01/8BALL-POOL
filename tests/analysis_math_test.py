import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from eight_ball_analyzer.export import analysis_to_dict
from eight_ball_analyzer.geometry import predict_shot, reflect_vector, trace_ball_collision_chain
from eight_ball_analyzer.models import TableDetection
from eight_ball_analyzer.models import BallDetection, BallKind, FrameAnalysis, Pocket


def main() -> None:
    cue = BallDetection(id=0, center=np.array([100.0, 100.0]), radius=10.0, kind=BallKind.CUE, confidence=0.9)
    obj = BallDetection(id=1, center=np.array([200.0, 100.0]), radius=10.0, kind=BallKind.SOLID, confidence=0.9)
    blocker = BallDetection(id=2, center=np.array([260.0, 100.0]), radius=10.0, kind=BallKind.STRIPE, confidence=0.9)
    pocket = Pocket(id=0, center=np.array([400.0, 100.0]), mouth_radius=18.0, label="top-left")

    shot = predict_shot(cue, obj, pocket, [cue, obj, blocker])
    assert shot.ghost_ball_center is not None
    assert shot.blocker_ids == [2]
    assert 0.0 <= shot.difficulty <= 100.0
    assert 0.0 <= shot.recommended_power <= 100.0

    reflected = reflect_vector(np.array([1.0, 1.0]), np.array([-1.0, 0.0]))
    assert reflected[0] < 0

    report = analysis_to_dict(FrameAnalysis(frame_index=3, table=None, balls=[cue, obj], pockets=[pocket], shot=shot))
    assert report["shot"]["blocker_ids"] == [2]
    assert report["balls"][0]["type"] == "cue"

    table = TableDetection(
        polygon=np.array([[0, 0], [500, 0], [500, 300], [0, 300]], dtype=np.int32),
        bounds=(0, 0, 500, 300),
        confidence=0.9,
    )
    blocker2 = BallDetection(id=2, center=np.array([300.0, 100.0]), radius=10.0, kind=BallKind.STRIPE, confidence=0.9)
    direction = np.array([1.0, 0.0])
    guide = trace_ball_collision_chain(cue, [obj, blocker2], table, direction, power=60.0, pockets=[pocket])
    assert guide.first_hit_ball_id == 1
    assert len(guide.object_path) >= 2
    assert len(guide.cue_path) >= 2

    print("Analysis math test passed")


if __name__ == "__main__":
    main()
