# 8 Ball Pool Training Analyzer

A Windows desktop training and post-analysis tool for recorded 8 Ball Pool practice videos and paused practice frames. The app does not provide real-time gameplay assistance; it is intended for reviewing shots after recording or while studying a paused frame.

## What It Does

- Detects the playable table cloth region with OpenCV HSV masking and contour selection.
- Estimates the six pocket positions from the detected table bounds.
- Detects balls with an optional custom Ultralytics YOLO model first, then a strict OpenCV fallback if no model is installed.
- Classifies detected balls as cue, solid, stripe, or eight ball from sampled color.
- Tracks ball identities across frames with a lightweight velocity-aware tracker.
- Computes ghost-ball contact position, cue path length, object-ball path length, cut angle, blockers, cue-ball deflection, cushion reflections, recommended power, and difficulty.
- Renders educational overlays for the cue path, object path, ghost ball, blockers, pockets, power, angle, and difficulty.
- Exports analyzed MP4 video, PNG frame overlays, and JSON analysis reports.

## Project Structure

```text
8 ball pool/
  requirements.txt
  pyproject.toml
  run.py
  src/
    eight_ball_analyzer/
      app.py
      analysis.py
      detection.py
      export.py
      geometry.py
      models.py
      overlays.py
      tracker.py
      ui/
        main_window.py
        video_widget.py
  tests/
    smoke_test.py
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
python run.py
```

Or double-click:

```text
Launch 8 Ball Analyzer.bat
```

## Basic Workflow

1. Click **Open Video** and choose a recorded practice video.
2. Use **Play**, the timeline, or **Analyze Current Frame** to inspect a frame.
3. The analyzer finds the active target from the cue ray and recommends the best pocket.
4. Review the ghost ball, predicted paths, blockers, difficulty, and recommended power.
5. Export an overlay video, PNG frame, or JSON report.

## YOLO Model

For best accuracy, train a custom Ultralytics YOLO model with these classes:

```text
cue_ball
solid_ball
stripe_ball
eight_ball
pocket
```

Place the trained model at one of these paths:

```text
models/8ball_yolov8.pt
models/8ball_yolov8.onnx
models/best.pt
models/best.onnx
```

Or set:

```powershell
$env:EIGHT_BALL_YOLO_MODEL="C:\path\to\best.pt"
```

The app uses YOLO at confidence `0.80` when a model is available. It will not download generic COCO weights automatically.

## Notes

- The OpenCV detector is a fallback. A trained pool-specific YOLO model is strongly recommended for production-quality ball detection.
- Best results come from stable videos where the whole table is visible and UI overlays do not cover the balls.
- All processing is local.

## Smoke Test

```powershell
python tests\smoke_test.py
```
