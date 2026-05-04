"""
reid.py — Player Re-Identification via HSV histogram matching.
"""
import cv2
import numpy as np
from config import REID_THRESHOLD, REID_MAX_AGE


class ReIDTracker:
    """Tracks player appearances via HSV histogram features and re-identifies
    players who momentarily disappear from the frame.
    """

    def __init__(
        self,
        threshold: float = REID_THRESHOLD,
        max_age: int = REID_MAX_AGE,
    ):
        self.threshold = threshold
        self.max_age = max_age
        # Active gallery: tracker_id -> {"feature": np.ndarray, "frame": int}
        self._active: dict[int, dict] = {}
        # Lost gallery: tracker_id -> {"feature": np.ndarray, "frame": int}
        self._lost: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def compute_feature(self, frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
        """Compute a normalised HSV histogram from the torso region (middle third).

        Returns a 1-D float32 array or None if the crop is too small.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        # Torso = middle third vertically
        top = y1 + h // 3
        bot = y1 + (2 * h) // 3
        crop = frame[top:bot, x1:x2]

        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # 16 bins for H, 8 for S, 8 for V
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [16, 8, 8],
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten().astype(np.float32)

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def update(
        self,
        tracker_id: int,
        frame: np.ndarray,
        bbox: tuple,
        current_frame: int,
    ):
        """Store / refresh the feature for a known active tracker ID."""
        feat = self.compute_feature(frame, bbox)
        if feat is not None:
            self._active[tracker_id] = {"feature": feat, "frame": current_frame}
            # If it was in the lost gallery, remove it
            self._lost.pop(tracker_id, None)

    def mark_lost(self, tracker_id: int, current_frame: int):
        """Move a tracker ID from active to the lost gallery."""
        entry = self._active.pop(tracker_id, None)
        if entry is not None:
            entry["frame"] = current_frame
            self._lost[tracker_id] = entry
        # Prune stale lost entries
        self._prune(current_frame)

    def match_new_id(
        self,
        new_id: int,
        frame: np.ndarray,
        bbox: tuple,
        current_frame: int,
    ) -> int | None:
        """Compare new_id's appearance to the lost gallery.

        Returns the old tracker_id if a match above threshold is found,
        otherwise None.
        """
        if not self._lost:
            return None

        new_feat = self.compute_feature(frame, bbox)
        if new_feat is None:
            return None

        best_id = None
        best_score = -1.0

        for old_id, entry in self._lost.items():
            feat = entry["feature"]
            # Bhattacharyya correlation: higher = more similar (max 1.0)
            score = float(cv2.compareHist(new_feat, feat, cv2.HISTCMP_CORREL))
            if score > best_score:
                best_score = score
                best_id = old_id

        if best_score >= self.threshold and best_id is not None:
            # Remove from lost gallery since it has been re-identified
            self._lost.pop(best_id, None)
            return best_id

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(self, current_frame: int):
        """Remove lost-gallery entries older than max_age frames."""
        stale = [
            tid for tid, entry in self._lost.items()
            if (current_frame - entry["frame"]) > self.max_age
        ]
        for tid in stale:
            del self._lost[tid]
