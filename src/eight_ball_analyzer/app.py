from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from .ui.main_window import MainWindow


def main() -> int:
    src_path = Path(__file__).resolve().parents[1]
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
