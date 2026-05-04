"""
homography.py — Real-world pitch coordinate mapping via homography.
"""
import cv2
import numpy as np

PITCH_W_M = 105.0
PITCH_H_M = 68.0

# Destination points in metres: TL, TR, BR, BL
_DST_PTS = np.array([
    [0.0,       0.0      ],
    [PITCH_W_M, 0.0      ],
    [PITCH_W_M, PITCH_H_M],
    [0.0,       PITCH_H_M],
], dtype=np.float32)


class PitchHomography:
    """Maps pixel coordinates to real-world pitch coordinates (metres)."""

    def __init__(self):
        self._H: np.ndarray | None = None
        self._calibrated: bool = False
        # Stored pixel corner points (TL, TR, BR, BL)
        self._src_pts: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------

    def calibrate_interactive(self, frame: np.ndarray) -> bool:
        """Show frame and let user click 4 pitch corners (TL, TR, BR, BL).

        Returns True if calibration succeeded, False if cancelled.
        """
        clicks: list[tuple[int, int]] = []
        labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        display = frame.copy()

        def _mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append((x, y))
                cv2.circle(display, (x, y), 7, (0, 255, 0), -1)
                idx = len(clicks) - 1
                cv2.putText(display, labels[idx], (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow("Calibrate — click corners", display)

        cv2.namedWindow("Calibrate — click corners")
        cv2.setMouseCallback("Calibrate — click corners", _mouse_cb)

        instruction = "Click 4 corners: TL -> TR -> BR -> BL  |  ESC to cancel"
        cv2.putText(display, instruction, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        cv2.imshow("Calibrate — click corners", display)

        while len(clicks) < 4:
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC — cancel
                cv2.destroyWindow("Calibrate — click corners")
                return False

        cv2.destroyWindow("Calibrate — click corners")
        self.set_corners(np.array(clicks, dtype=np.float32))
        return True

    def set_corners(self, src_pts: np.ndarray):
        """Compute homography from 4 pixel points (TL, TR, BR, BL) to pitch metres.

        src_pts: shape (4, 2) float array in pixel space.
        """
        self._src_pts = src_pts.astype(np.float32)
        H, status = cv2.findHomography(self._src_pts, _DST_PTS)
        if H is not None:
            self._H = H
            self._calibrated = True
        else:
            self._H = None
            self._calibrated = False

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def to_meters(self, px: float, py: float) -> tuple[float, float] | None:
        """Convert a pixel point to real-world metres.

        Returns (mx, my) or None if not calibrated.
        """
        if not self._calibrated or self._H is None:
            return None
        pt = np.array([[[px, py]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H)
        mx, my = float(result[0][0][0]), float(result[0][0][1])
        return mx, my

    def to_minimap(
        self,
        px: float,
        py: float,
        map_w: int,
        map_h: int,
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, int]:
        """Convert pixel position to minimap pixel coordinates.

        If calibrated: uses homography to get real-world position then scales.
        Otherwise: normalises by frame dimensions.
        """
        if self._calibrated:
            result = self.to_meters(px, py)
            if result is not None:
                mx, my = result
                # Clamp to pitch
                mx = max(0.0, min(mx, PITCH_W_M))
                my = max(0.0, min(my, PITCH_H_M))
                mmx = int(mx / PITCH_W_M * map_w)
                mmy = int(my / PITCH_H_M * map_h)
                return mmx, mmy

        # Fallback — simple linear normalisation
        mmx = int(px / max(frame_w, 1) * map_w)
        mmy = int(py / max(frame_h, 1) * map_h)
        return mmx, mmy

    @property
    def calibrated(self) -> bool:
        return self._calibrated
