from __future__ import annotations

import copy

import numpy as np

from .models import Pocket, TableDetection


class TableGeometryLock:
    """Keep table bounds and pocket positions fixed once reliably detected.

    Lock policy
    -----------
    * Lock immediately when confidence >= LOCK_CONFIDENCE (0.40).
    * After locking, only re-evaluate the table every RECHECK_INTERVAL frames
      so that we don't waste CPU on stable footage.
    * If a re-check produces a high-confidence detection that disagrees
      substantially with the locked geometry, we update the lock.
    """

    LOCK_CONFIDENCE: float = 0.40
    RECHECK_INTERVAL: int = 60  # frames between geometry re-evaluations

    def __init__(self) -> None:
        self._locked_table: TableDetection | None = None
        self._locked_pockets: list[Pocket] | None = None
        self._lock_frame: int = -1

    def reset(self) -> None:
        self._locked_table = None
        self._locked_pockets = None
        self._lock_frame = -1

    @property
    def is_locked(self) -> bool:
        return self._locked_table is not None and self._locked_pockets is not None

    def apply(
        self,
        table: TableDetection | None,
        pockets: list[Pocket],
        frame_index: int = 0,
    ) -> tuple[TableDetection | None, list[Pocket]]:
        """Return (table, pockets), using locked values where appropriate.

        Skips re-evaluation on most frames when the geometry is already locked.
        """
        # --- Case 1: nothing detected this frame ---
        if table is None or len(pockets) != 6:
            if self.is_locked:
                return self._locked_table, copy.deepcopy(self._locked_pockets)  # type: ignore[return-value]
            return table, pockets

        # --- Case 2: geometry is locked and it is NOT a recheck frame ---
        if self.is_locked:
            frames_since_lock = frame_index - self._lock_frame
            if frames_since_lock % self.RECHECK_INTERVAL != 0:
                # Return the stable locked geometry without updating
                return self._locked_table, copy.deepcopy(self._locked_pockets)  # type: ignore[return-value]

            # It IS a recheck frame: evaluate whether the new detection is
            # substantially different from what is locked.
            if not self._geometry_changed(table):
                # Same scene — just refresh the lock frame counter
                self._lock_frame = frame_index
                return self._locked_table, copy.deepcopy(self._locked_pockets)  # type: ignore[return-value]

        # Lock on first good table detection (AirPlay / live capture)
        if not self.is_locked and table.confidence >= self.LOCK_CONFIDENCE:
            self._lock(table, pockets, frame_index)

        if self.is_locked:
            return self._locked_table, copy.deepcopy(self._locked_pockets)  # type: ignore[return-value]

        # Not yet locked (confidence still low) — use current detections
        return table, pockets

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _lock(self, table: TableDetection, pockets: list[Pocket], frame_index: int) -> None:
        """Snapshot the current table geometry."""
        self._locked_table = TableDetection(
            polygon=table.polygon.copy(),
            bounds=table.bounds,
            confidence=table.confidence,
            corners=table.corners.copy(),
            width_px=table.width_px,
            height_px=table.height_px,
            transform_matrix=table.transform_matrix.copy() if table.transform_matrix is not None else None,
            inv_transform_matrix=table.inv_transform_matrix.copy() if table.inv_transform_matrix is not None else None,
            rails=table.rails,  # Rail objects are lightweight; share references
        )
        self._locked_pockets = [
            Pocket(
                id=p.id,
                center=p.center.copy(),
                mouth_radius=p.mouth_radius,
                label=p.label,
            )
            for p in pockets
        ]
        self._lock_frame = frame_index

    def _geometry_changed(self, table: TableDetection, threshold_px: float = 20.0) -> bool:
        """Return True if the new table detection differs significantly from locked."""
        if self._locked_table is None:
            return True
        lx, ly, lw, lh = self._locked_table.bounds
        nx, ny, nw, nh = table.bounds
        centre_delta = float(np.hypot((lx + lw / 2) - (nx + nw / 2), (ly + lh / 2) - (ny + nh / 2)))
        size_delta = abs(lw - nw) + abs(lh - nh)
        return centre_delta > threshold_px or size_delta > threshold_px * 2
