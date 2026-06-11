from __future__ import annotations

import cv2
import mss
import pygetwindow as gw
import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QComboBox, QProgressDialog, QSlider, QVBoxLayout, QWidget, QLineEdit
)

from ..analysis import VideoAnalyzer
from ..export import export_analysis_report, export_analyzed_frame, export_analyzed_video
from ..overlays import draw_overlay
from .video_widget import VideoWidget


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
        self.live_capture_active = False
        self.live_frame_index = 0
        self.sct = mss.mss()
        self.live_capture_timer = QTimer(self)
        self.live_capture_timer.timeout.connect(self._live_capture_tick)
        
        self.video_widget = VideoWidget()
        self.open_button = QPushButton("Open Video")
        self.play_button = QPushButton("Play")
        
        self.live_button = QPushButton("Start Live Capture")
        self.live_button.setCheckable(True)
        self.window_name_input = QLineEdit("LonelyScreen")
        self.rewind_button = QPushButton("Rewind Shot")
        self.analyze_button = QPushButton("Analyze Current Frame")
        self.export_button = QPushButton("Export Overlay Video")
        self.export_frame_button = QPushButton("Export PNG Frame")
        self.export_report_button = QPushButton("Export JSON Report")
        self.ball_combo = QComboBox()
        self.pocket_combo = QComboBox()
        self.status_label = QLabel("No video loaded")
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.recommended_pocket_label = QLabel("Pocket: --")
        self.success_label = QLabel("Success: --")
        self.difficulty_label = QLabel("Difficulty: --")
        self.cut_angle_label = QLabel("Cut angle: --")
        self.cue_travel_label = QLabel("Cue travel: --")
        self.object_travel_label = QLabel("Object travel: --")
        self.power_label_widget = QLabel("Power: --")
        self.blockers_label = QLabel("Blockers: --")
        self._build_layout()
        self._connect()
        self._set_video_controls_enabled(False)

    def _build_layout(self) -> None:
        side = QVBoxLayout()
        side.addWidget(self.open_button)
        side.addWidget(self.play_button)
        side.addWidget(self.rewind_button)
        side.addWidget(self.analyze_button)
        side.addSpacing(10)
        side.addWidget(QLabel("Live Target Window:"))
        side.addWidget(self.window_name_input)
        side.addWidget(self.live_button)
        side.addSpacing(10)
        side.addWidget(QLabel("Object Ball"))
        side.addWidget(self.ball_combo)
        side.addWidget(QLabel("Recommended Pocket"))
        side.addWidget(self.pocket_combo)
        side.addWidget(self.export_button)
        side.addWidget(self.export_frame_button)
        side.addWidget(self.export_report_button)
        side.addSpacing(16)
        for lbl in (self.recommended_pocket_label, self.success_label, self.difficulty_label, self.cut_angle_label, self.cue_travel_label,
                    self.object_travel_label, self.power_label_widget, self.blockers_label):
            side.addWidget(lbl)
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
        self.ball_combo.currentIndexChanged.connect(self._selection_changed)
        self.pocket_combo.currentIndexChanged.connect(self._selection_changed)

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

    def toggle_live_capture(self) -> None:
        if self.live_button.isChecked():
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            self.timer.stop()
            self.status_label.setText("Starting live capture...")
            self.live_capture_active = True
            self.live_frame_index = 0
            self.analyzer.reset_tracking()
            self._set_video_controls_enabled(False)
            self.live_button.setText("Stop Live Capture")
            self.live_capture_timer.start(33)
            self.ball_combo.setEnabled(True)
            self.pocket_combo.setEnabled(True)
        else:
            self.live_capture_timer.stop()
            self.live_capture_active = False
            self.status_label.setText("Live capture stopped.")
            self.live_button.setText("Start Live Capture")
            self._set_video_controls_enabled(self.video_path is not None)

    def _live_capture_tick(self) -> None:
        target_name = self.window_name_input.text()
        try:
            windows = gw.getWindowsWithTitle(target_name)
            if not windows:
                self.status_label.setText(f"Window '{target_name}' not found.")
                return
            
            win = None
            for w in windows:
                if w.width > 0 and w.height > 0:
                    win = w
                    break
                    
            if not win:
                self.status_label.setText(f"Window '{target_name}' is minimized or hidden.")
                return
                
            rect = {"top": win.top, "left": win.left, "width": win.width, "height": win.height}
            screenshot = self.sct.grab(rect)
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
            self.current_frame = frame
            self.current_analysis = self.analyzer.analyze_frame(
                frame, frame_index=self.live_frame_index, track=True,
            )
            self.live_frame_index += 1
            self._refresh_combos()
            self._show_current_overlay()
            self.status_label.setText(f"Live capturing: {target_name} ({rect['width']}x{rect['height']})")
        except Exception as e:
            self.status_label.setText(f"Live capture error: {str(e)}")

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
        self._refresh_combos()
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
        self._refresh_combos()
        self._show_current_overlay()

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

    def _selection_changed(self) -> None:
        ball_id = self.ball_combo.currentData()
        pocket_id = self.pocket_combo.currentData()
        self.analyzer.selected_object_ball_id = ball_id
        self.analyzer.selected_pocket_id = pocket_id
        if self.current_frame is not None:
            idx = self.live_frame_index if self.live_capture_active else self.frame_index
            self.current_analysis = self.analyzer.analyze_frame(
                self.current_frame, idx, track=True)
            self._show_current_overlay()

    def _refresh_combos(self) -> None:
        if self.current_analysis is None:
            return
        self.ball_combo.blockSignals(True)
        current_ball = self.ball_combo.currentData()
        self.ball_combo.clear()
        for ball in self.current_analysis.balls:
            if ball.kind.value != "cue" and ball.confidence >= 0.55:
                self.ball_combo.addItem(f"Ball {ball.id} ({ball.kind.value})", ball.id)
        detected_hit = self.current_analysis.guide.first_hit_ball_id
        preferred_ball = detected_hit if detected_hit is not None else current_ball
        self._restore_combo(self.ball_combo, preferred_ball)
        if detected_hit is not None:
            self.analyzer.selected_object_ball_id = detected_hit
        self.ball_combo.blockSignals(False)
        self.pocket_combo.blockSignals(True)
        current_pocket = self.pocket_combo.currentData()
        self.pocket_combo.clear()
        for pocket in self.current_analysis.pockets:
            self.pocket_combo.addItem(f"{pocket.id}: {pocket.label}", pocket.id)
        recommended_pocket = self.current_analysis.shot.pocket_id
        self._restore_combo(self.pocket_combo, recommended_pocket if recommended_pocket is not None else current_pocket)
        self.pocket_combo.blockSignals(False)

    def _restore_combo(self, combo, value) -> None:
        if value is None:
            return
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _show_current_overlay(self) -> None:
        if self.current_frame is None:
            return
        overlay = draw_overlay(self.current_frame, self.current_analysis)
        self.video_widget.set_frame(overlay)
        if self.current_analysis is not None:
            shot = self.current_analysis.shot
            self.recommended_pocket_label.setText(f"Pocket: {shot.pocket_label}")
            self.success_label.setText(f"Success: {shot.success_probability:.0f}%")
            self.difficulty_label.setText(
                f"Difficulty: {shot.difficulty:.0f}/100 {shot.difficulty_label}")
            self.cut_angle_label.setText(f"Cut angle: {shot.cut_angle:.1f} deg")
            self.cue_travel_label.setText(f"Cue travel: {shot.cue_path_length:.0f}px")
            self.object_travel_label.setText(f"Object travel: {shot.object_path_length:.0f}px")
            self.power_label_widget.setText(
                f"Power: {shot.recommended_power:.0f}% ({shot.power_label})")
            blocker_str = ", ".join(str(b) for b in shot.blocker_ids) if shot.blocker_ids else "none"
            self.blockers_label.setText(f"Blockers: {blocker_str}")

    def _set_video_controls_enabled(self, enabled: bool) -> None:
        self.play_button.setEnabled(enabled)
        self.rewind_button.setEnabled(enabled)
        self.analyze_button.setEnabled(enabled)
        self.ball_combo.setEnabled(enabled)
        self.pocket_combo.setEnabled(enabled)
        self.export_button.setEnabled(enabled)
        self.export_frame_button.setEnabled(enabled)
        self.export_report_button.setEnabled(enabled)
        self.timeline.setEnabled(enabled)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.live_capture_timer.stop()
        if self.capture is not None:
            self.capture.release()
        super().closeEvent(event)
