from __future__ import annotations

import cv2
import time
import win32gui
import numpy as np
import threading
import traceback
import queue
from typing import Optional

from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QComboBox, QProgressDialog, QSlider, QVBoxLayout, QWidget, QLineEdit,
    QCheckBox
)

from ..analysis import VideoAnalyzer
from ..export import export_analysis_report, export_analyzed_frame, export_analyzed_video
from ..live_capture import ChromeWebCapture
from ..overlays import draw_overlay
from .video_widget import VideoWidget


# ---------------------------------------------------------------------------
# AnalysisWorker — runs heavy AI analysis on a background QThread
# ---------------------------------------------------------------------------

class AnalysisWorker(QThread):
    analysis_finished = pyqtSignal(object, object, int, float, dict, float)
    state_changed = pyqtSignal(str)
    worker_error = pyqtSignal(str)

    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.q = queue.Queue(maxsize=1)
        self.is_running = True
        self.put_count = 0
        self.get_count = 0

    def run(self):
        self.state_changed.emit("Analysis worker started")

        # STALL FIX: Pre-warm the YOLO model here in the background thread.
        # Without this, the first frame triggers _load_yolo_model() on the
        # worker thread, but the GUI watchdog fires because the thread doesn't
        # respond to the queue for 5+ seconds while YOLO loads.
        try:
            detector = self.analyzer.detector
            if detector._yolo_model_path is not None and not detector._yolo_load_attempted:
                self.state_changed.emit("Loading YOLO model (background)...")
                detector._load_yolo_model()
                if detector.has_yolo_model:
                    self.state_changed.emit("YOLO model loaded ✓")
                else:
                    self.state_changed.emit("YOLO not found — using fallback detection")
        except Exception as _e:
            self.state_changed.emit(f"YOLO pre-warm skipped: {_e}")

        while self.is_running:
            try:
                t_dequeue_start = time.time()
                frame_bgra, frame_index, capture_ts = self.q.get(timeout=0.1)
                self.get_count += 1
                if self.get_count == 1:
                    self.state_changed.emit("First frame received by worker")

                t_dequeue_end = time.time()

                t_ana_start = time.time()
                frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                fh, fw = frame_bgr.shape[:2]

                # ----------------------------------------------------------
                # Full frame analysis
                # (ROI cropping moved to analyzer to prevent coordinate mismatch)
                # ----------------------------------------------------------
                calib = self.analyzer.calibration

                if calib.is_calibrated():
                    corners = calib.load_corners()
                    cx = max(0, int(corners[:, 0].min()))
                    cy = max(0, int(corners[:, 1].min()))
                    cx2 = min(fw, int(corners[:, 0].max()))
                    cy2 = min(fh, int(corners[:, 1].max()))

                    # BUG 7 FIX: per-frame print flood removed — only log on first frame
                    if self.get_count == 1:
                        cv2.imwrite("debug_full_frame.png", frame_bgr)
                        print(f"[DEBUG] Full frame: {fw}x{fh}  Table ROI: x={cx} y={cy} w={cx2-cx} h={cy2-cy}")
                        if cx2 > fw or cy2 > fh:
                            print(f"[WARNING] Calibration corners outside frame! Need recalibration.")


                analysis = self.analyzer.analyze_frame(frame_bgr, frame_index=frame_index, track=True)
                t_ana_end = time.time()

                fps = 1.0 / max(t_ana_end - t_ana_start, 0.001)

                # ----------------------------------------------------------
                # Draw overlays 
                # ----------------------------------------------------------
                t_overlay_start = time.time()
                display_frame = frame_bgr.copy()

                if not calib.is_calibrated():
                    # Show corner placement dots
                    for pt in calib.corners:
                        cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                        cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 15, (0, 255, 0), 2)
                    if len(calib.corners) > 0:
                        pts = np.array(calib.corners, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(display_frame, [pts], False, (0, 255, 0), 2)
                    overlay = display_frame
                else:
                    from ..overlays import draw_overlay as _draw
                    overlay = _draw(display_frame, analysis)

                t_overlay_end = time.time()

                timings = {
                    "dequeue": t_dequeue_end - t_dequeue_start,
                    "analysis": t_ana_end - t_ana_start,
                    "overlay": t_overlay_end - t_overlay_start,
                }

                self.analysis_finished.emit(overlay, analysis, frame_index, fps, timings, capture_ts)
            except queue.Empty:
                continue
            except Exception as e:
                error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
                print("AnalysisWorker error:\n", error_msg)
                self.worker_error.emit(f"Error: {str(e)}")

    def enqueue_frame(self, frame, frame_index, capture_ts):
        if self.q.full():
            try:
                self.q.get_nowait()
                self._drop_count = getattr(self, '_drop_count', 0) + 1
                if self._drop_count % 100 == 1:
                    print(f"[WARNING] Frame queue full — dropping frames (total dropped: {self._drop_count})")
            except queue.Empty:
                pass
        try:
            self.q.put_nowait((frame, frame_index, capture_ts))
            self.put_count += 1
        except queue.Full:
            pass

    def stop(self):
        self.is_running = False
        self.wait()


# ---------------------------------------------------------------------------
# MssLiveCapture — screen-region capture via mss
#
# Root cause of WGC failure:
#   LonelyScreen creates Win32 windows that cannot be converted to a
#   GraphicsCaptureItem. All three HWNDs (9177566, 852610, 66390446) raise:
#     "Capture session threw an exception: Failed to convert item to
#      GraphicsCaptureItem"
#   when passed to windows-capture's start_free_threaded().
#
#   mss uses BitBlt from the display compositor and works regardless of the
#   target application's window composition model.
# ---------------------------------------------------------------------------

class MssLiveCapture:
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.hwnd = None
        self.latest_frame = None
        self.frame_id = 0
        self.last_fetched_id = -1
        self.is_running = False
        self.thread = None
        self.startup_state = "Initialized"
        self.start_time = 0
        self.error = None

    def _find_best_hwnd(self) -> int:
        """
        Find the HWND belonging to the target application by matching the process
        executable path, not just the window title substring.  This prevents the
        searcher from accidentally picking up our own IDE window.
        """
        import ctypes
        import ctypes.wintypes

        name_lower = self.window_name.lower()

        candidates = []

        def cb(h, _):
            title = win32gui.GetWindowText(h)
            if not title:
                return True
            if name_lower not in title.lower():
                return True

            # Get the process executable path and verify it belongs to LonelyScreen
            pid = ctypes.c_ulong()
            ctypes.windll.user32.GetWindowThreadProcessId(h, ctypes.byref(pid))
            try:
                PROCESS_QUERY_INFO = 0x1000
                ph = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFO, False, pid.value)
                if ph:
                    buf = ctypes.create_unicode_buffer(512)
                    size = ctypes.c_ulong(512)
                    ctypes.windll.kernel32.QueryFullProcessImageNameW(ph, 0, buf, ctypes.byref(size))
                    ctypes.windll.kernel32.CloseHandle(ph)
                    exe_path = buf.value.lower()
                else:
                    exe_path = ""
            except Exception:
                exe_path = ""

            # Accept only if the executable path contains our search term or is LonelyScreen
            exe_ok = (name_lower in exe_path) or ("lonelyscreen" in exe_path)
            if not exe_ok:
                print(f"[HWND] Skipping HWND={h} title={repr(title)} exe={exe_path} (exe mismatch)")
                return True

            rect = win32gui.GetWindowRect(h)
            w = rect[2] - rect[0]
            ht = rect[3] - rect[1]
            print(f"[HWND] Candidate HWND={h} title={repr(title)} size={w}x{ht} exe={exe_path}")
            if w > 30 and ht > 30:
                candidates.append((w * ht, h, rect))
            return True

        win32gui.EnumWindows(cb, None)

        if not candidates:
            return 0
        candidates.sort(reverse=True)
        chosen = candidates[0]
        print(f"[HWND] Selected HWND={chosen[1]} area={chosen[0]} rect={chosen[2]}")
        return chosen[1]

    @staticmethod
    def _detect_table_roi(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """
        Detect the pool table inside a captured frame by looking for a large
        rectangle of green cloth colour.  Returns (x, y, w, h) in frame coords,
        or None if the table cannot be found.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # Pool cloth is a medium-to-dark green
        green_lo = np.array([35,  30,  30])
        green_hi = np.array([95, 255, 220])
        mask = cv2.inRange(hsv, green_lo, green_hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
        if area < frame_area * 0.05:   # table must be >5% of frame
            return None

        x, y, w, h = cv2.boundingRect(best)
        aspect = w / max(h, 1)
        if not (1.2 <= aspect <= 3.5):  # pool tables are wide
            return None
        return x, y, w, h

    @staticmethod
    def _get_window_size(hwnd: int) -> tuple[int, int]:
        """Return (width, height) of the window in its restored (non-minimized) state."""
        import ctypes, ctypes.wintypes

        class WINDOWPLACEMENT(ctypes.Structure):
            _fields_ = [
                ("length",           ctypes.c_uint),
                ("flags",            ctypes.c_uint),
                ("showCmd",          ctypes.c_uint),
                ("ptMinPosition",    ctypes.wintypes.POINT),
                ("ptMaxPosition",    ctypes.wintypes.POINT),
                ("rcNormalPosition", ctypes.wintypes.RECT),
            ]

        wp = WINDOWPLACEMENT()
        wp.length = ctypes.sizeof(wp)
        ctypes.windll.user32.GetWindowPlacement(hwnd, ctypes.byref(wp))
        r = wp.rcNormalPosition
        w = r.right  - r.left
        h = r.bottom - r.top
        if w <= 0 or h <= 0:
            # Fall back to current window rect
            rect = win32gui.GetWindowRect(hwnd)
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
        return max(w, 1), max(h, 1)

    @staticmethod
    def _capture_hwnd_bitmap(hwnd: int, width: int, height: int) -> np.ndarray | None:
        """
        Capture the window via PrintWindow(PW_RENDERFULLCONTENT).
        Works when the window is:
          - Minimized / iconic
          - Behind other windows
          - On another virtual desktop
          - Off-screen
        Returns a BGRA numpy array, or None on failure.
        """
        import ctypes
        import win32ui
        PW_RENDERFULLCONTENT = 2   # Windows 8.1+ — renders DWM content

        bmp = None
        save_dc = None
        mfc_dc = None
        hwnd_dc = None
        try:
            hwnd_dc  = win32gui.GetWindowDC(hwnd)
            mfc_dc   = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc  = mfc_dc.CreateCompatibleDC()
            bmp      = win32ui.CreateBitmap()
            bmp.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(bmp)

            result = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)

            bmp_info = bmp.GetInfo()
            bmp_bits = bmp.GetBitmapBits(True)

            img = np.frombuffer(bmp_bits, dtype=np.uint8).reshape(
                (bmp_info["bmHeight"], bmp_info["bmWidth"], 4)
            ).copy()  # BGRA

        except Exception as e:
            img = None
            result = 0
            print(f"[PrintWindow] Error: {e}")
        finally:
            try:
                if bmp:    win32gui.DeleteObject(bmp.GetHandle())
                if save_dc: save_dc.DeleteDC()
                if mfc_dc:  mfc_dc.DeleteDC()
                if hwnd_dc: win32gui.ReleaseDC(hwnd, hwnd_dc)
            except Exception:
                pass

        if img is None or result == 0:
            return None
        return img


    def start(self):
        print("Capture state: Thread starting")
        self.startup_state = "Thread starting"
        self.is_running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run_capture, daemon=True)
        self.thread.start()

    def _run_capture(self):
        try:
            print("Capture state: Finding window")
            self.startup_state = "Finding window"
            self.hwnd = self._find_best_hwnd()
            if not self.hwnd:
                self.error = f"Window not found: {self.window_name}"
                self.startup_state = "Error: Window not found"
                print(self.error)
                return

            rect = win32gui.GetWindowRect(self.hwnd)
            print(f"Capture state: Found HWND={self.hwnd} rect={rect}")
            self.startup_state = "Capture session starting"

            # --- First frame ---
            w, h = self._get_window_size(self.hwnd)
            frame = self._capture_hwnd_bitmap(self.hwnd, w, h)
            if frame is None or frame.size == 0:
                self.error = "PrintWindow returned empty frame on first capture"
                self.startup_state = "Error: first frame empty"
                print(self.error)
                return

            self.latest_frame = frame
            self.frame_id += 1
            self.startup_state = "First frame received"
            print(f"Capture state: First frame received  shape={frame.shape}")

            # --- Capture loop (works minimized / hidden) ---
            while self.is_running:
                w, h = self._get_window_size(self.hwnd)
                if w < 10 or h < 10:
                    time.sleep(0.05)
                    continue
                frame = self._capture_hwnd_bitmap(self.hwnd, w, h)
                if frame is not None and frame.size > 0:
                    self.latest_frame = frame
                    self.frame_id += 1
                time.sleep(0.033)  # ~30 fps cap

        except Exception as e:
            self.error = str(e)
            self.startup_state = f"Error: {e}"
            print("Capture failed:", e)
            traceback.print_exc()

    def get_frame(self):
        if self.frame_id == self.last_fetched_id:
            return None
        self.last_fetched_id = self.frame_id
        return self.latest_frame

    def stop(self):
        self.is_running = False






# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("8 Ball Pool Training Analyzer")
        self.resize(1180, 760)
        self.analyzer = VideoAnalyzer()
        self.capture = None
        self.video_path = None
        self.frame_index = 0
        self.frame_count = 0
        self.current_frame = None
        self.current_analysis = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._timer_tick)

        # Live capture state
        self.is_live_capture = False
        self.live_capture_wrapper = None
        self.live_target_name = ""
        self.live_frame_index = 0
        self._live_calibration_signature = None
        self._live_capture_shape = None
        self.live_timer = QTimer(self)
        self.live_timer.timeout.connect(self._live_capture_tick)

        # Diagnostic labels
        self.lbl_cap_fps = QLabel("Capture FPS: 0.0")
        self.lbl_ana_fps = QLabel("Analysis FPS: 0.0")
        self.lbl_frame_age = QLabel("Frame Age: - ms")
        self.lbl_queue = QLabel("Queue Size: 0")
        self.lbl_last_ts = QLabel("Last Frame TS: -")
        self.lbl_heartbeat = QLabel("Main Heartbeat: -")

        self.lbl_frames_cap = QLabel("Frames Captured: 0")
        self.lbl_frames_que = QLabel("Frames Queued (Put): 0")
        self.lbl_frames_ana = QLabel("Frames Dequeued: 0")
        self.lbl_frames_ren = QLabel("Frames Rendered: 0")
        self.lbl_worker_state = QLabel("Worker State: Not Started")
        self.lbl_worker_errors = QLabel("Worker Errors: None")

        self.frames_captured = 0
        self.frames_rendered = 0

        self.analysis_worker = None
        self.last_heartbeat = time.time()
        self.heartbeat_timer = QTimer(self)
        self.heartbeat_timer.timeout.connect(self._update_heartbeat)
        self.heartbeat_timer.start(100)

        self.watchdog_running = True
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()

        self.last_capture_time = time.time()

        self.video_widget = VideoWidget()
        self.open_button = QPushButton("Open Video")
        self.play_button = QPushButton("Play")

        self.live_button = QPushButton("Start Live Capture")
        self.live_button.setCheckable(True)
        self.window_name_input = QLineEdit("8 Ball Pool")
        self.auto_calibrate_cb = QCheckBox("Auto Table Lock")
        self.auto_calibrate_cb.setChecked(True)
        self.rewind_button = QPushButton("Rewind Shot")
        self.analyze_button = QPushButton("Analyze Current Frame")
        self.export_button = QPushButton("Export Overlay Video")
        self.export_frame_button = QPushButton("Export PNG Frame")
        self.export_report_button = QPushButton("Export JSON Report")
        self.reflection_combo = QComboBox()
        self.reflection_combo.addItems(["1", "2", "3", "5", "10"])
        self.reflection_combo.setCurrentText("3")

        self.show_cue_cb = QCheckBox("Show Cue Path")
        self.show_cue_cb.setChecked(True)
        self.show_reflection_cb = QCheckBox("Show Reflection Path")
        self.show_reflection_cb.setChecked(True)
        self.show_collision_cb = QCheckBox("Show Collision Prediction")
        self.show_collision_cb.setChecked(True)
        self.show_object_cb = QCheckBox("Show Object Ball Path")
        self.show_object_cb.setChecked(True)
        self.show_deflection_cb = QCheckBox("Show Cue Ball Deflection")
        self.show_deflection_cb.setChecked(True)
        self.trick_shot_cb = QCheckBox("Trick Shot Mode")
        self.trick_shot_cb.setChecked(False)

        self.state_label = QLabel("State: READY")
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self._build_layout()
        self._connect()
        self._set_video_controls_enabled(False)
        self.video_widget.image_clicked.connect(self._handle_video_click)

    # ------------------------------------------------------------------
    # Heartbeat & watchdog
    # ------------------------------------------------------------------

    def _update_heartbeat(self):
        self.last_heartbeat = time.time()
        self.lbl_heartbeat.setText(f"Main Heartbeat: {time.strftime('%H:%M:%S')}")

    def _watchdog_loop(self):
        while self.watchdog_running:
            try:
                time.sleep(0.5)
                if time.time() - self.last_heartbeat > 0.5:
                    print(f"[WATCHDOG WARNING] Main GUI thread stalled for >500ms! "
                          f"(Stall: {time.time() - self.last_heartbeat:.2f}s)")
            except Exception as e:
                print("Watchdog loop error:", e)

    # ------------------------------------------------------------------
    # Event overrides
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_R:
            self.analyzer.calibration.delete_calibration()
            self.analyzer.calibration.corners = []
            self.status_label.setText("Calibration reset. Click 4 corners (TL, TR, BR, BL).")
            self._show_current_overlay()
        super().keyPressEvent(event)

    def _handle_video_click(self, x: int, y: int) -> None:
        if self.analyzer.calibration.is_calibrated():
            return
        if len(self.analyzer.calibration.corners) < 4:
            self.analyzer.calibration.corners.append((float(x), float(y)))
            if len(self.analyzer.calibration.corners) == 4:
                self.analyzer.calibration.save_corners(self.analyzer.calibration.corners)
                self.status_label.setText("Calibration saved!")
                if self.current_frame is not None:
                    self.analyze_current_frame()
            else:
                self._show_current_overlay()

    # ------------------------------------------------------------------
    # Layout & connections
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        side = QVBoxLayout()
        side.addWidget(self.open_button)
        side.addWidget(self.play_button)
        side.addWidget(self.rewind_button)
        side.addWidget(self.analyze_button)
        side.addSpacing(10)
        side.addWidget(QLabel("Live Target Window:"))
        side.addWidget(self.window_name_input)
        side.addWidget(self.auto_calibrate_cb)
        side.addWidget(self.live_button)
        side.addSpacing(10)
        side.addWidget(QLabel("Reflection Depth"))
        side.addWidget(self.reflection_combo)
        side.addSpacing(5)
        side.addWidget(self.show_cue_cb)
        side.addWidget(self.show_reflection_cb)
        side.addWidget(self.show_collision_cb)
        side.addWidget(self.show_object_cb)
        side.addWidget(self.show_deflection_cb)
        side.addWidget(self.trick_shot_cb)
        side.addSpacing(10)
        side.addWidget(self.export_button)
        side.addWidget(self.export_frame_button)
        side.addWidget(self.export_report_button)
        side.addSpacing(16)
        side.addWidget(self.state_label)
        side.addSpacing(16)
        side.addWidget(QLabel("<b>Diagnostics</b>"))
        side.addWidget(self.lbl_cap_fps)
        side.addWidget(self.lbl_ana_fps)
        side.addWidget(self.lbl_frame_age)
        side.addWidget(self.lbl_queue)
        side.addWidget(self.lbl_last_ts)
        side.addWidget(self.lbl_heartbeat)
        side.addSpacing(10)
        side.addWidget(QLabel("<b>Counters &amp; Worker</b>"))
        side.addWidget(self.lbl_frames_cap)
        side.addWidget(self.lbl_frames_que)
        side.addWidget(self.lbl_frames_ana)
        side.addWidget(self.lbl_frames_ren)
        side.addWidget(self.lbl_worker_state)
        side.addWidget(self.lbl_worker_errors)
        self.lbl_worker_errors.setWordWrap(True)
        side.addStretch()

        side_panel = QWidget()
        side_panel.setLayout(side)
        side_panel.setFixedWidth(260)

        main = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(self.video_widget, 1)
        row.addWidget(side_panel)
        main.addLayout(row, 1)
        main.addWidget(self.timeline)
        self.status_label = QLabel("No video loaded")
        main.addWidget(self.status_label)
        root = QWidget()
        root.setLayout(main)
        self.setCentralWidget(root)

    def _connect(self) -> None:
        self.open_button.clicked.connect(self.open_video)
        self.play_button.clicked.connect(self.toggle_playback)
        self.live_button.clicked.connect(self.toggle_live_capture)
        self.rewind_button.clicked.connect(self.rewind_shot)
        self.analyze_button.clicked.connect(self.analyze_current_frame)
        self.export_button.clicked.connect(self.export_video)
        self.export_frame_button.clicked.connect(self.export_frame)
        self.export_report_button.clicked.connect(self.export_report)
        self.timeline.sliderMoved.connect(self.seek)
        self.reflection_combo.currentIndexChanged.connect(self._config_changed)
        self.show_cue_cb.stateChanged.connect(self._config_changed)
        self.show_reflection_cb.stateChanged.connect(self._config_changed)
        self.show_collision_cb.stateChanged.connect(self._config_changed)
        self.show_object_cb.stateChanged.connect(self._config_changed)
        self.show_deflection_cb.stateChanged.connect(self._config_changed)
        self.trick_shot_cb.stateChanged.connect(self._config_changed)

    # ------------------------------------------------------------------
    # Video file playback
    # ------------------------------------------------------------------

    def open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Practice Video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*.*)")
        if not path:
            return
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            QMessageBox.critical(self, "Open Video", "Could not open the selected video.")
            return
        self.video_path = path
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.timeline.setRange(0, max(0, self.frame_count - 1))
        self.frame_index = 0
        self.analyzer.reset_tracking()
        self._set_video_controls_enabled(True)
        self.read_frame(0)

    def toggle_playback(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start(33)
            self.play_button.setText("Pause")

    def rewind_shot(self) -> None:
        if self.capture is None:
            return
        target = max(0, self.frame_index - 30)
        self.read_frame(target)

    def _timer_tick(self) -> None:
        self.next_frame()

    def next_frame(self) -> None:
        if self.capture is None:
            return
        next_index = self.frame_index + 1
        if self.frame_count and next_index >= self.frame_count:
            self.timer.stop()
            self.play_button.setText("Play")
            return
        self.read_frame(next_index)

    # ------------------------------------------------------------------
    # Live capture
    # ------------------------------------------------------------------

    def toggle_live_capture(self) -> None:
        if self.live_button.isChecked():
            print("Start Live Capture clicked")
            self.status_label.setText("Start Live Capture clicked")

            if self.capture is not None:
                self.capture.release()
                self.capture = None
            self.timer.stop()

            target_name = self.window_name_input.text().strip()
            if not target_name:
                self.status_label.setText("Please enter a target window name")
                self.live_button.setChecked(False)
                return

            print(f"Starting Chrome/web capture for window: {target_name}")
            self.live_target_name = target_name
            self.is_live_capture = True
            self.live_frame_index = 0
            self.frames_captured = 0
            self.frames_rendered = 0

            self.analyzer.reset_tracking()
            self.analyzer.detection_interval_frames = 24
            self._live_calibration_signature = None
            self._live_capture_shape = None
            if self.auto_calibrate_cb.isChecked():
                self.analyzer.calibration.delete_calibration()
                self.analyzer.calibration.corners = []
            self._set_video_controls_enabled(False)

            self.live_capture_wrapper = ChromeWebCapture(target_name)
            self.live_capture_wrapper.start()

            if self.analysis_worker is not None:
                self.analysis_worker.stop()
            self.analysis_worker = AnalysisWorker(self.analyzer)
            self.analysis_worker.analysis_finished.connect(self._on_analysis_finished)
            self.analysis_worker.state_changed.connect(
                lambda s: self.lbl_worker_state.setText(f"Worker State: {s}"))
            self.analysis_worker.worker_error.connect(
                lambda e: self.lbl_worker_errors.setText(e))
            self.analysis_worker.start()

            self.live_timer.start(33)
            self.live_button.setText("Stop Live Capture")
            self.reflection_combo.setEnabled(True)
        else:
            self.live_timer.stop()
            if self.analysis_worker is not None:
                self.analysis_worker.stop()
                self.analysis_worker = None
            if self.live_capture_wrapper:
                self.live_capture_wrapper.stop()
                self.live_capture_wrapper = None
            self.is_live_capture = False
            self.analyzer.detection_interval_frames = 12
            self.status_label.setText("Live capture stopped.")
            self.live_button.setText("Start Live Capture")
            self._set_video_controls_enabled(self.video_path is not None)

    def _live_capture_tick(self) -> None:
        t_tick_start = time.time()
        try:
            if not self.is_live_capture or not self.live_capture_wrapper:
                return

            wrapper = self.live_capture_wrapper

            # Wait for startup
            if wrapper.startup_state != "First frame received":
                self.status_label.setText(f"Capture Startup: {wrapper.startup_state}")
                if wrapper.start_time > 0 and time.time() - wrapper.start_time > 5.0:
                    print(f"Capture timeout! Stalled at: {wrapper.startup_state}")
                    self.status_label.setText(f"Capture timeout! Stalled at: {wrapper.startup_state}")
                    self.live_button.setChecked(False)
                    self.toggle_live_capture()
                return

            img = wrapper.get_frame()
            if img is None:
                return

            now = time.time()
            dt = max(now - self.last_capture_time, 0.001)
            self.last_capture_time = now
            self.lbl_cap_fps.setText(f"Capture FPS: {1.0 / dt:.1f}")
            self.lbl_last_ts.setText(
                f"Last Frame TS: {time.strftime('%H:%M:%S')}.{int(now * 1000) % 1000:03d}")

            height, width = img.shape[:2]
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            self.current_frame = frame_bgr

            shape_sig = (width, height)
            if shape_sig != self._live_capture_shape:
                self._live_capture_shape = shape_sig
                self._live_calibration_signature = None
                if self.auto_calibrate_cb.isChecked():
                    self.analyzer.calibration.delete_calibration()
                    self.analyzer.calibration.corners = []

            if self.auto_calibrate_cb.isChecked():
                self._auto_calibrate_live_table(wrapper)

            if self.analysis_worker is not None:
                self.analysis_worker.enqueue_frame(img, self.live_frame_index, now)
                self.lbl_queue.setText(f"Queue Size: {self.analysis_worker.q.qsize()}")
                self.lbl_frames_que.setText(f"Frames Queued (Put): {self.analysis_worker.put_count}")
                self.lbl_frames_ana.setText(f"Frames Dequeued: {self.analysis_worker.get_count}")

            self.live_frame_index += 1
            self.frames_captured += 1
            self.lbl_frames_cap.setText(f"Frames Captured: {self.frames_captured}")
            table_status = getattr(wrapper, "table_status", "Table: unknown")
            self.status_label.setText(
                f"Live capturing: {self.live_target_name} ({width}x{height}) | {table_status}")

        except Exception as e:
            self.status_label.setText(f"Live capture error: {str(e)}")
            traceback.print_exc()

        t_tick_end = time.time()
        if t_tick_end - t_tick_start > 0.1:
            print(f"[GUI WARNING] _live_capture_tick took {(t_tick_end - t_tick_start)*1000:.1f}ms")

    def _auto_calibrate_live_table(self, wrapper) -> None:
        corners = getattr(wrapper, "table_corners", None)
        if not corners or len(corners) != 4:
            return

        signature = tuple((round(float(x) / 4) * 4, round(float(y) / 4) * 4) for x, y in corners)
        if signature == self._live_calibration_signature:
            return

        self.analyzer.calibration.save_corners(corners)
        self.analyzer.calibration.corners = []
        self.analyzer.reset_tracking()
        self._live_calibration_signature = signature
        print(f"[CALIBRATION] Live table lock updated: {signature}")

    def _on_analysis_finished(self, overlay, analysis, frame_index, fps, timings, capture_ts):
        t_start = time.time()
        
        frame_age_ms = (time.time() - capture_ts) * 1000.0
        self.lbl_frame_age.setText(f"Frame Age: {frame_age_ms:.1f} ms")

        self.frames_rendered += 1
        self.lbl_frames_ren.setText(f"Frames Rendered: {self.frames_rendered}")

        self.current_analysis = analysis
        self.lbl_ana_fps.setText(f"Analysis FPS: {fps:.1f}")
        if self.current_analysis is not None:
            self.state_label.setText(f"State: {self.current_analysis.state.name}")

        t_pixmap_start = time.time()
        self.video_widget.set_frame(overlay)
        t_pixmap_end = time.time()

        timings["qpixmap"] = t_pixmap_end - t_pixmap_start
        total_gui = time.time() - t_start
        if total_gui > 0.1:
            print(f"[GUI WARNING] _on_analysis_finished took {total_gui*1000:.1f}ms "
                  f"(Pixmap: {timings['qpixmap']*1000:.1f}ms)")

    # ------------------------------------------------------------------
    # Video frame helpers
    # ------------------------------------------------------------------

    def seek(self, index: int) -> None:
        self.read_frame(index)

    def read_frame(self, index: int) -> None:
        if self.capture is None:
            return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok:
            self.timer.stop()
            self.play_button.setText("Play")
            self.status_label.setText("End of video reached")
            return
        self.frame_index = index
        self.current_frame = frame
        self.current_analysis = self.analyzer.analyze_frame(frame, frame_index=index, track=True)
        self._show_current_overlay()
        self.timeline.blockSignals(True)
        self.timeline.setValue(index)
        self.timeline.blockSignals(False)
        self.status_label.setText(f"Frame {index + 1} / {max(1, self.frame_count)}")

    def analyze_current_frame(self) -> None:
        if self.current_frame is None:
            return
        self.current_analysis = self.analyzer.analyze_frame(
            self.current_frame, frame_index=self.frame_index, track=True)
        self._show_current_overlay()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_video(self) -> None:
        if self.video_path is None:
            QMessageBox.information(self, "Export Overlay Video",
                "Export is available after opening a recorded practice video.")
            return
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Export Analyzed Video", "analyzed_8ball_overlay.mp4", "MP4 Video (*.mp4)")
        if not output_path:
            return
        progress = QProgressDialog("Exporting analyzed video...", "Cancel",
            0, max(1, self.frame_count), self)
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)

        def on_progress(done, total):
            progress.setMaximum(max(1, total))
            progress.setValue(done)

        try:
            completed = export_analyzed_video(
                self.video_path, output_path, self.analyzer, on_progress, progress.wasCanceled)
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return
        finally:
            progress.close()
        if not completed:
            QMessageBox.information(self, "Export Cancelled", "Video export was cancelled.")
            return
        QMessageBox.information(self, "Export Complete", f"Saved analyzed video:\n{output_path}")

    def export_frame(self) -> None:
        if self.current_frame is None or self.current_analysis is None:
            return
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Export Analyzed Frame", "analyzed_8ball_frame.png", "PNG Image (*.png)")
        if not output_path:
            return
        try:
            export_analyzed_frame(self.current_frame, self.current_analysis, output_path)
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return
        QMessageBox.information(self, "Export Complete", f"Saved analyzed frame:\n{output_path}")

    def export_report(self) -> None:
        if self.current_analysis is None:
            return
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON Analysis Report", "analysis_report.json", "JSON Report (*.json)")
        if not output_path:
            return
        try:
            export_analysis_report(self.current_analysis, output_path)
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return
        QMessageBox.information(self, "Export Complete", f"Saved JSON report:\n{output_path}")

    # ------------------------------------------------------------------
    # Config & overlay
    # ------------------------------------------------------------------

    def _config_changed(self) -> None:
        self.analyzer.config.max_bounces = int(self.reflection_combo.currentText())
        self.analyzer.config.show_cue_path = self.show_cue_cb.isChecked()
        self.analyzer.config.show_reflection_path = self.show_reflection_cb.isChecked()
        self.analyzer.config.show_collision = self.show_collision_cb.isChecked()
        self.analyzer.config.show_object_path = self.show_object_cb.isChecked()
        self.analyzer.config.show_deflection = self.show_deflection_cb.isChecked()
        self.analyzer.config.trick_shot_mode = self.trick_shot_cb.isChecked()

        if self.current_frame is not None and not self.is_live_capture:
            self.current_analysis = self.analyzer.analyze_frame(
                self.current_frame, self.frame_index, track=True)
            self._show_current_overlay()

    def _show_current_overlay(self) -> None:
        if self.current_frame is None:
            return
        overlay = self.current_frame.copy()
        if not self.analyzer.calibration.is_calibrated():
            for pt in self.analyzer.calibration.corners:
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 15, (0, 255, 0), 2)
            if len(self.analyzer.calibration.corners) > 0:
                pts = np.array(self.analyzer.calibration.corners, np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], False, (0, 255, 0), 2)
        else:
            overlay = draw_overlay(self.current_frame, self.current_analysis)
        self.video_widget.set_frame(overlay)
        if self.current_analysis is not None:
            self.state_label.setText(f"State: {self.current_analysis.state.name}")

    def _set_video_controls_enabled(self, enabled: bool) -> None:
        self.play_button.setEnabled(enabled)
        self.rewind_button.setEnabled(enabled)
        self.analyze_button.setEnabled(enabled)
        self.reflection_combo.setEnabled(enabled)
        self.show_cue_cb.setEnabled(enabled)
        self.show_reflection_cb.setEnabled(enabled)
        self.show_collision_cb.setEnabled(enabled)
        self.show_object_cb.setEnabled(enabled)
        self.show_deflection_cb.setEnabled(enabled)
        self.trick_shot_cb.setEnabled(enabled)
        self.export_button.setEnabled(enabled)
        self.export_frame_button.setEnabled(enabled)
        self.export_report_button.setEnabled(enabled)
        self.timeline.setEnabled(enabled)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        if hasattr(self, "live_timer"):
            self.live_timer.stop()
        if hasattr(self, "heartbeat_timer"):
            self.heartbeat_timer.stop()
        self.watchdog_running = False
        if self.analysis_worker is not None:
            self.analysis_worker.stop()
        if self.capture is not None:
            self.capture.release()
        super().closeEvent(event)
