import sys, traceback
def eh(c, e, t):
 with open('crash_log.txt', 'w') as f: traceback.print_exception(c, e, t, file=f)
 sys.__excepthook__(c, e, t)
sys.excepthook = eh
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eight_ball_analyzer.app import main


if __name__ == "__main__":
    main()
