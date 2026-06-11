# Goal Description
Refactor the entire 8 Ball Pool Training Analyzer to a **TABLE-FIRST** pipeline. This ensures that the system fully understands the table geometry—establishing a stable coordinate system and mathematically derived pockets—before any ball detection or physics calculations are performed.

## User Review Required
> [!IMPORTANT]
> The physics calculations currently operate entirely in "image space". Moving them to "normalized table coordinates" means we will need to transform all ball centers into a top-down normalized space using the perspective transform matrix, perform the physics (ghost ball, blockers, banks), and then transform the paths back to image space for the overlay. This will significantly improve accuracy, especially on tilted perspectives! I will implement this transformation pipeline. Please let me know if you want to keep physics in image space instead, but normalized space is definitely the standard for billiards engines.

## Proposed Changes

---

### `models.py`
[MODIFY] [models.py](file:///c:/Users/shiva/OneDrive/Documents/8%20ball%20pool/src/eight_ball_analyzer/models.py)
* Update `TableDetection` to include:
  * `corners` (list of 4 numpy arrays: top-left, top-right, bottom-left, bottom-right)
  * `width` and `height` (normalized play area dimensions)
  * `transform_matrix` (the 3x3 perspective transform matrix `M` from `cv2.getPerspectiveTransform`)
  * `inv_transform_matrix` (the inverse of `M` for mapping normalized physics paths back to image space)
* Update `Pocket` to align with the new structure (`id`, `x`, `y`, `radius` inside normalized space or image space).

---

### `detection.py`
[MODIFY] [detection.py](file:///c:/Users/shiva/OneDrive/Documents/8%20ball%20pool/src/eight_ball_analyzer/detection.py)
* **Stage 1 (Table Detection)**: Overhaul `detect_table` to accurately find the playable cloth area, sort the 4 corners, generate normalized dimensions (`table_width`, `table_height`), and compute the perspective transform matrices.
* **Stage 2 (Pocket System)**: Remove `detect_pockets` image processing. Replace it with a mathematical calculation based solely on the 4 corners of the table (top-left, midpoints, etc.).
* **Stage 5 (Ball Detection)**: Update `detect_balls` to strictly ignore anything outside the `playable_table_region`. Transform ball centers to check against normalized bounds.

---

### `analysis.py`
[MODIFY] [analysis.py](file:///c:/Users/shiva/OneDrive/Documents/8%20ball%20pool/src/eight_ball_analyzer/analysis.py)
* **Stage 3 (Table Validation)**: Enforce the Table-First requirement. If table validation fails (e.g., corners missing, bad aspect ratio), skip ball detection and physics entirely.
* **Pipeline Restructure**: 
  1. Detect Table -> 2. Calculate Pockets -> 3. Lock Geometry -> 4. Detect Balls.
  Balls will only be detected if step 3 is successful.

---

### `geometry.py`
[MODIFY] [geometry.py](file:///c:/Users/shiva/OneDrive/Documents/8%20ball%20pool/src/eight_ball_analyzer/geometry.py)
* **Stage 6 & 7 (Shot Analysis & Bank Shots)**:
  * Refactor `predict_shot`, `best_one_rail_bank_path`, and blockers to perform calculations in the normalized coordinate system.
  * Map ball coordinates to normalized space using `transform_matrix` before physics.
  * Map resulting paths back to image space using `inv_transform_matrix`.
  * Restrict bank shots to strictly single-rail vector reflections: `R = D - 2(D·N)N`. Strip out the recursive multi-bouncing logic.

---

### `overlays.py`
[MODIFY] [overlays.py](file:///c:/Users/shiva/OneDrive/Documents/8%20ball%20pool/src/eight_ball_analyzer/overlays.py)
* **Stage 4 (Visualization)**:
  * Draw the strict green rectangle representing the exact playable area.
  * Draw corner markers.
  * Draw mathematical pocket markers with their IDs (0 to 5).
  * Draw a perspective grid representing the normalized space transformed onto the table.
  * Display "Table Width", "Table Height", and "Perspective Status" on the HUD.

## Verification Plan

### Automated Tests
- Run `.\.venv\Scripts\python.exe tests\smoke_test.py` to ensure the analyzer pipeline doesn't crash with the new Table-First logic.

### Manual Verification
- I will launch the app (`run.py`) and verify that:
  - The perspective grid overlays correctly onto the table.
  - The 6 pockets are perfectly locked to the mathematically derived corner and midpoint positions.
  - Bank shots and direct shots calculate accurately using the normalized reference frame.
