import sys
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
    analysis = analyzer.analyze_frame(frame, frame_index=0, track=False)
    assert analysis.table is not None
    assert len(analysis.pockets) == 6
    print("Smoke test passed")


if __name__ == "__main__":
    main()
