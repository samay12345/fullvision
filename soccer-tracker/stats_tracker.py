"""
stats_tracker.py — Per-player stat accumulation updated every frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlayerStats:
    """Holds all live per-player stats accumulated during tracking."""

    distance_m: float = 0.0          # total meters traveled
    top_speed_ms: float = 0.0        # max speed seen (m/s)
    speed_history: list = field(default_factory=list)  # rolling last 60 frames
    sprint_count: int = 0            # times exceeded 7 m/s for 10+ consecutive frames
    _sprint_frames: int = 0          # internal consecutive-sprint frame counter
    _in_sprint: bool = False         # currently in a sprint window
    time_on_ball: int = 0            # frames player bbox overlaps ball bbox
    last_pos: Optional[tuple] = None  # last (cx, cy)


class PlayerStatsTracker:
    """Accumulate per-player stats frame-by-frame.

    Parameters
    ----------
    fps : int
        Native video frame rate — used for future time-based calculations.
    pixels_per_meter : float
        Approximate conversion factor.  Default: 10 px = 1 m.
    sprint_speed_ms : float
        Speed threshold (m/s) that defines a sprint.
    sprint_frames_required : int
        Minimum consecutive frames above sprint_speed_ms to count as a sprint.
    """

    def __init__(
        self,
        fps: int = 30,
        pixels_per_meter: float = 10.0,
        sprint_speed_ms: float = 4.0,
        sprint_frames_required: int = 5,
    ):
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.sprint_speed_ms = sprint_speed_ms
        self.sprint_frames_required = sprint_frames_required
        self.stats: dict[int, PlayerStats] = {}

    # ------------------------------------------------------------------
    # Bbox overlap helper
    # ------------------------------------------------------------------

    @staticmethod
    def _bboxes_overlap(bbox_a: tuple, bbox_b: tuple) -> bool:
        """Return True if two (x1,y1,x2,y2) bboxes overlap."""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        track_id: int,
        cx: float,
        cy: float,
        speed_ms: float,
        bbox: tuple,
        ball_bbox: Optional[tuple] = None,
    ) -> None:
        """Update stats for *track_id* with the current frame's data.

        Parameters
        ----------
        track_id : int
        cx, cy : float
            Centre pixel coordinates of the player.
        speed_ms : float
            Current speed in m/s (pre-computed by PlayerTracker.get_average_speed).
        bbox : tuple
            (x1, y1, x2, y2) of the player bounding box.
        ball_bbox : tuple | None
            (x1, y1, x2, y2) of the ball bounding box (None if not detected).
        """
        if track_id not in self.stats:
            self.stats[track_id] = PlayerStats()

        ps = self.stats[track_id]

        # ── Distance ──────────────────────────────────────────────────
        if ps.last_pos is not None:
            px_dist = math.hypot(cx - ps.last_pos[0], cy - ps.last_pos[1])
            ps.distance_m += px_dist / self.pixels_per_meter

        ps.last_pos = (cx, cy)

        # ── Top speed ─────────────────────────────────────────────────
        if speed_ms > ps.top_speed_ms:
            ps.top_speed_ms = speed_ms

        # ── Speed history (rolling 60 frames) ─────────────────────────
        ps.speed_history.append(speed_ms)
        if len(ps.speed_history) > 60:
            ps.speed_history = ps.speed_history[-60:]

        # ── Sprint detection ──────────────────────────────────────────
        if speed_ms > self.sprint_speed_ms:
            ps._sprint_frames += 1
            if ps._sprint_frames >= self.sprint_frames_required and not ps._in_sprint:
                ps.sprint_count += 1
                ps._in_sprint = True
        else:
            ps._sprint_frames = 0
            ps._in_sprint = False

        # ── Time on ball ──────────────────────────────────────────────
        if ball_bbox is not None:
            if self._bboxes_overlap(bbox, ball_bbox):
                ps.time_on_ball += 1

    def get_stats(self, track_id: int) -> Optional[PlayerStats]:
        """Return the PlayerStats for *track_id*, or None if unknown."""
        return self.stats.get(track_id)

    def get_avg_speed(self, track_id: int) -> float:
        """Return the mean of the rolling speed history for *track_id*."""
        ps = self.stats.get(track_id)
        if ps is None or not ps.speed_history:
            return 0.0
        return sum(ps.speed_history) / len(ps.speed_history)

    def get_all_active(self, track_ids: list[int]) -> dict[int, PlayerStats]:
        """Return a dict of stats for only the given active track IDs."""
        return {tid: self.stats[tid] for tid in track_ids if tid in self.stats}
