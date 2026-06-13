import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np

from eight_ball_analyzer.analysis import VideoAnalyzer


def main() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[120:600, 120:1160] = (40, 120, 40)
    analyzer = VideoAnalyzer()
    analyzer.calibration.filepath = Path(tempfile.gettempdir()) / "eight_ball_smoke_calibration.json"
    analyzer.calibration.save_corners([
        (120.0, 120.0),
        (1160.0, 120.0),
        (1160.0, 600.0),
        (120.0, 600.0),
    ])
    analysis = analyzer.analyze_frame(frame, frame_index=0, track=False)
    assert analysis.table is not None
    assert len(analysis.pockets) == 6
    print("Smoke test passed")


if __name__ == "__main__":
    main()
