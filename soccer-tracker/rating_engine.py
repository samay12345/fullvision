"""
rating_engine.py — Live player rating system (1.0–10.0 scale).

Ratings start at 6.5 and are updated by game events detected in analyzer.py.
Every 5 seconds of match time, ratings decay slightly for older events.
All events are logged to rating_log.jsonl.
"""
from __future__ import annotations

import json
import math
import time as _time


# ---------------------------------------------------------------------------
# Event deltas
# ---------------------------------------------------------------------------

EVENT_DELTAS: dict[str, float] = {
    # --- Attacking ---
    "successful_pass":      +0.04,
    "key_pass":             +0.22,   # pass directly leading to a shot
    "progressive_carry":    +0.08,
    "dribble_success":      +0.12,
    "shot_on_target":       +0.28,
    "shot_off_target":      +0.08,
    "goal_involvement":     +0.50,
    # --- Defensive ---
    "tackle_won":           +0.18,
    "defensive_intercept":  +0.12,
    "clearance":            +0.08,
    "high_press":           +0.05,
    "sprint_burst":         +0.03,
    # --- Negative ---
    "misplaced_pass":       -0.08,
    "losing_possession":    -0.10,
    "dribble_fail":         -0.10,
    "tackle_lost":          -0.08,
    "poor_positioning":     -0.05,
    "standing_still":       -0.03,
}

DEFAULT_RATING = 6.5
MIN_RATING = 1.0
MAX_RATING = 10.0
MAX_DELTA_PER_EVENT = 0.3
DECAY_INTERVAL_S = 5.0      # decay every 5 seconds of match time
DECAY_OLD_THRESHOLD_S = 90.0  # events older than 90s get exponential weight reduction


class RatingEngine:
    """
    Maintains live per-player ratings updated by game events.

    All state keyed by track_id (int).
    """

    def __init__(self, log_path: str = "rating_log.jsonl"):
        self.ratings: dict[int, float] = {}          # track_id -> rating
        self.rating_history: dict[int, list] = {}    # track_id -> [(timestamp_s, rating)]
        self.last_event: dict[int, str] = {}         # track_id -> last event string
        self._prev_rating: dict[int, float] = {}     # for delta computation
        self._event_log: list[dict] = []             # buffered events (timestamp aware)
        self._last_decay_ts: float = 0.0             # last match timestamp when decay ran
        self._player_names: dict[int, str] = {}      # track_id -> name (for logging)

        try:
            self._log_file = open(log_path, "a", encoding="utf-8")
        except OSError:
            self._log_file = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_rating(self, track_id: int) -> float:
        return self.ratings.get(track_id, DEFAULT_RATING)

    def _clamp(self, value: float) -> float:
        return max(MIN_RATING, min(MAX_RATING, value))

    def _log(self, record: dict) -> None:
        if self._log_file is not None:
            try:
                self._log_file.write(json.dumps(record) + "\n")
                self._log_file.flush()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_name(self, track_id: int, name: str) -> None:
        """Attach a display name to a track_id for log records."""
        if name:
            self._player_names[track_id] = name

    def update(
        self,
        track_id: int,
        event_type: str,
        frame_num: int,
        timestamp_s: float,
        name: str = "",
    ) -> float:
        """
        Apply an event delta to the player's rating.

        Returns the new rating.
        """
        if name:
            self._player_names[track_id] = name

        base_delta = EVENT_DELTAS.get(event_type, 0.0)
        # Cap magnitude of a single event
        delta = max(-MAX_DELTA_PER_EVENT, min(MAX_DELTA_PER_EVENT, base_delta))

        current = self._get_rating(track_id)
        self._prev_rating[track_id] = current
        new_rating = self._clamp(current + delta)
        self.ratings[track_id] = new_rating

        self.last_event[track_id] = event_type

        # Append to history (keep last 200 entries per player)
        hist = self.rating_history.setdefault(track_id, [])
        hist.append((timestamp_s, new_rating))
        if len(hist) > 200:
            self.rating_history[track_id] = hist[-200:]

        # Log the event
        record = {
            "ts": round(timestamp_s, 2),
            "track_id": track_id,
            "name": self._player_names.get(track_id, name),
            "event": event_type,
            "delta": round(delta, 4),
            "new_rating": round(new_rating, 4),
        }
        self._log(record)

        # Buffer for exponential decay weight
        self._event_log.append({
            "track_id": track_id,
            "timestamp_s": timestamp_s,
            "delta": delta,
        })
        # Keep event log bounded
        if len(self._event_log) > 5000:
            self._event_log = self._event_log[-5000:]

        return new_rating

    def get_rating(self, track_id: int) -> float:
        """Return current rating, defaulting to 6.5."""
        return round(self._get_rating(track_id), 2)

    def get_delta(self, track_id: int) -> float:
        """Return change since last update cycle (positive = improved)."""
        current = self._get_rating(track_id)
        prev = self._prev_rating.get(track_id, current)
        return round(current - prev, 3)

    def decay_all(self, timestamp_s: float) -> None:
        """
        Apply exponential decay to all players.

        Called every 5 seconds of match time. Events older than 90 s
        contribute less weight to the rating floor.
        """
        if (timestamp_s - self._last_decay_ts) < DECAY_INTERVAL_S:
            return
        self._last_decay_ts = timestamp_s

        for track_id, rating in list(self.ratings.items()):
            if rating == DEFAULT_RATING:
                continue  # nothing to decay

            # Compute a small nudge toward DEFAULT_RATING based on age
            # of recent events. The further from the default, the more we nudge.
            deviation = rating - DEFAULT_RATING
            # Decay factor: 1% per 5-second cycle
            decay_amount = deviation * 0.01
            nudged = self._clamp(rating - decay_amount)
            self.ratings[track_id] = nudged
            self._prev_rating[track_id] = rating  # so delta reflects the decay

    def get_all_ratings(self) -> dict[int, dict]:
        """Return a snapshot: track_id -> {rating, delta, last_event}."""
        result = {}
        for tid, rating in self.ratings.items():
            result[tid] = {
                "rating": round(rating, 2),
                "delta": self.get_delta(tid),
                "last_event": self.last_event.get(tid, ""),
            }
        return result

    def close(self) -> None:
        """Flush and close the log file."""
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Event detection helpers (called by analyzer.py per frame)
# ---------------------------------------------------------------------------

class EventDetector:
    """
    Stateful per-player event detector.

    Tracks passes, tackles, shots, dribbles, sprints, pressing, and carries.
    Call detect() every YOLO frame to get a list of (track_id, event_type).
    """

    SPRINT_SPEED_MS = 3.5
    SPRINT_MIN_FRAMES = 12
    STILL_SPEED_MS = 0.5
    STILL_MIN_FRAMES = 100
    PROGRESSIVE_CARRY_FRAMES = 8
    HIGH_PRESS_DIST_PX = 100

    # Shot detection
    SHOT_SPEED_PX = 18          # min ball px/frame velocity to consider a shot
    SHOT_GOAL_X_FRAC = 0.12     # goal regions occupy this fraction of frame width from each side

    # Tackle: ball changes from one team to another near two players touching
    TACKLE_IOU_THRESH = 0.05    # bboxes must overlap this much to be a tackle (not a pass)

    # Dribble: owner holds ball for N frames while an opponent is close
    DRIBBLE_OPPONENT_DIST = 90  # px — opponent must be within this distance
    DRIBBLE_MIN_FRAMES = 6      # frames holding ball near opponent = dribble success

    def __init__(self):
        self._sprint_frames: dict[int, int] = {}
        self._still_frames: dict[int, int] = {}
        self._carry_frames: dict[int, int] = {}
        self._dribble_frames: dict[int, int] = {}
        self._in_sprint: dict[int, bool] = {}
        self._in_still: dict[int, bool] = {}
        self._in_carry: dict[int, bool] = {}
        self._in_dribble: dict[int, bool] = {}

        # Ball ownership history
        self._ball_owner_this: int | None = None
        self._ball_owner_prev: int | None = None
        self._ownership_history: list[int | None] = []  # last 10 owners

        # Ball position history for velocity
        self._ball_pos_history: list[tuple[float, float]] = []

        # Team mapping
        self._player_teams: dict[int, str] = {}
        self._attacking_right: dict[str, bool] = {"A": True, "B": False}

        # Frame dimensions (set on first call)
        self._frame_w: int = 1280
        self._frame_h: int = 720

        # Cooldowns to avoid duplicate event spam
        self._last_shot_frame: int = -999
        self._last_press_frame: dict[int, int] = {}
        self._last_tackle_frame: dict[int, int] = {}

    def set_attacking_direction(self, team: str, attacks_right: bool) -> None:
        self._attacking_right[team] = attacks_right

    def _ball_speed_px(self) -> float:
        """Return current ball speed in px/frame based on last 2 positions."""
        if len(self._ball_pos_history) < 2:
            return 0.0
        p1 = self._ball_pos_history[-2]
        p2 = self._ball_pos_history[-1]
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def _ball_heading_to_goal(self) -> bool:
        """True if the ball's velocity vector points roughly toward either goal."""
        if len(self._ball_pos_history) < 2:
            return False
        p1 = self._ball_pos_history[-2]
        p2 = self._ball_pos_history[-1]
        dx = p2[0] - p1[0]
        bx = p2[0] / self._frame_w
        # Ball moving left toward left goal, or right toward right goal
        return (dx < -3 and bx < 0.35) or (dx > 3 and bx > 0.65)

    def detect(
        self,
        player_detections: list[dict],
        ball_pos: tuple | None,
        speed_map: dict[int, float],
        frame_num: int,
        frame_shape: tuple = (720, 1280),
    ) -> list[tuple[int, str]]:
        """
        Returns list of (track_id, event_type) detected this frame.
        """
        events: list[tuple[int, str]] = []
        self._frame_h, self._frame_w = frame_shape[:2]

        # Update team mapping
        for det in player_detections:
            tid = det["tracker_id"]
            if det.get("team"):
                self._player_teams[tid] = det["team"]

        # Track ball position history
        if ball_pos is not None:
            self._ball_pos_history.append((float(ball_pos[0]), float(ball_pos[1])))
            if len(self._ball_pos_history) > 10:
                self._ball_pos_history = self._ball_pos_history[-10:]

        # Determine ball owner this frame
        self._ball_owner_prev = self._ball_owner_this
        self._ball_owner_this = None
        if ball_pos is not None and player_detections:
            bx, by = ball_pos
            min_dist, closest_id = float("inf"), None
            for det in player_detections:
                cx, cy = det["center"]
                d = math.hypot(cx - bx, cy - by)
                if d < min_dist:
                    min_dist = d
                    closest_id = det["tracker_id"]
            if min_dist < 80:
                self._ball_owner_this = closest_id

        self._ownership_history.append(self._ball_owner_this)
        if len(self._ownership_history) > 10:
            self._ownership_history = self._ownership_history[-10:]

        det_map: dict[int, dict] = {d["tracker_id"]: d for d in player_detections}

        # ── Pass / Tackle / Interception detection ────────────────────────────
        prev_owner = self._ball_owner_prev
        cur_owner = self._ball_owner_this
        if prev_owner is not None and cur_owner is not None and prev_owner != cur_owner:
            prev_team = self._player_teams.get(prev_owner)
            cur_team = self._player_teams.get(cur_owner)
            if prev_team and cur_team:
                prev_det = det_map.get(prev_owner)
                cur_det = det_map.get(cur_owner)
                iou = _bbox_iou(prev_det["bbox"], cur_det["bbox"]) if (prev_det and cur_det) else 0.0

                if prev_team == cur_team:
                    # Same team: successful pass
                    # Was there a shot right before? → could be a rebound assist
                    events.append((prev_owner, "successful_pass"))
                    # Check if this pass came right after a shot → key pass candidate
                    # (handled in shot detection below)
                else:
                    # Opposition gained the ball
                    if iou > self.TACKLE_IOU_THRESH:
                        # Players overlapping → tackle
                        cooldown = self._last_tackle_frame.get(cur_owner, -999)
                        if frame_num - cooldown > 20:
                            events.append((cur_owner, "tackle_won"))
                            events.append((prev_owner, "tackle_lost"))
                            self._last_tackle_frame[cur_owner] = frame_num
                            self._last_tackle_frame[prev_owner] = frame_num
                    else:
                        # No overlap → interception or misplaced pass
                        ball_spd = self._ball_speed_px()
                        if ball_spd > 8:
                            # Fast ball that was intercepted
                            events.append((cur_owner, "defensive_intercept"))
                        events.append((prev_owner, "misplaced_pass"))

        # ── Shot detection ────────────────────────────────────────────────────
        ball_spd = self._ball_speed_px()
        if (
            ball_spd > self.SHOT_SPEED_PX
            and self._ball_owner_prev is not None
            and self._ball_owner_this is None          # ball no longer near a player
            and (frame_num - self._last_shot_frame) > 30
        ):
            shooter_id = self._ball_owner_prev
            if self._ball_heading_to_goal():
                events.append((shooter_id, "shot_on_target"))
            else:
                events.append((shooter_id, "shot_off_target"))
            self._last_shot_frame = frame_num

            # Retroactively mark the pass before this shot as a key_pass
            # (find who passed to the shooter in the last 5 frames)
            for i in range(len(self._ownership_history) - 2, -1, -1):
                prev_o = self._ownership_history[i]
                if prev_o is not None and prev_o != shooter_id:
                    prev_t = self._player_teams.get(prev_o)
                    shooter_t = self._player_teams.get(shooter_id)
                    if prev_t and prev_t == shooter_t:
                        events.append((prev_o, "key_pass"))
                    break

        # ── Per-player events ─────────────────────────────────────────────────
        for det in player_detections:
            tid = det["tracker_id"]
            speed = speed_map.get(tid, 0.0)
            cx, cy = det["center"]
            team = self._player_teams.get(tid)

            # Sprint burst
            if speed > self.SPRINT_SPEED_MS:
                self._sprint_frames[tid] = self._sprint_frames.get(tid, 0) + 1
                if (
                    self._sprint_frames[tid] >= self.SPRINT_MIN_FRAMES
                    and not self._in_sprint.get(tid, False)
                ):
                    events.append((tid, "sprint_burst"))
                    self._in_sprint[tid] = True
            else:
                self._sprint_frames[tid] = 0
                self._in_sprint[tid] = False

            # Standing still penalty
            if speed < self.STILL_SPEED_MS:
                self._still_frames[tid] = self._still_frames.get(tid, 0) + 1
                if (
                    self._still_frames[tid] >= self.STILL_MIN_FRAMES
                    and not self._in_still.get(tid, False)
                ):
                    events.append((tid, "standing_still"))
                    self._in_still[tid] = True
            else:
                self._still_frames[tid] = 0
                self._in_still[tid] = False

            # High press
            if ball_pos is not None and self._ball_owner_this is not None:
                owner_id = self._ball_owner_this
                owner_team = self._player_teams.get(owner_id)
                if owner_team and owner_team != team:
                    owner_det = det_map.get(owner_id)
                    if owner_det is not None:
                        ox, oy = owner_det["center"]
                        dist_to_holder = math.hypot(cx - ox, cy - oy)
                        cooldown = self._last_press_frame.get(tid, -999)
                        if dist_to_holder < self.HIGH_PRESS_DIST_PX and (frame_num - cooldown) > 15:
                            events.append((tid, "high_press"))
                            self._last_press_frame[tid] = frame_num

            # Progressive carry
            if tid == self._ball_owner_this and team is not None:
                self._carry_frames[tid] = self._carry_frames.get(tid, 0) + 1
                if (
                    self._carry_frames[tid] >= self.PROGRESSIVE_CARRY_FRAMES
                    and not self._in_carry.get(tid, False)
                ):
                    events.append((tid, "progressive_carry"))
                    self._in_carry[tid] = True
            else:
                carry = self._carry_frames.get(tid, 0)
                if carry > 0:
                    self._carry_frames[tid] = max(0, carry - 3)
                self._in_carry[tid] = False

            # Dribble: ball owner sustained near opponent
            if tid == self._ball_owner_this and team is not None:
                near_opponent = False
                for other in player_detections:
                    oid = other["tracker_id"]
                    if oid == tid:
                        continue
                    if self._player_teams.get(oid) != team:
                        ox, oy = other["center"]
                        if math.hypot(cx - ox, cy - oy) < self.DRIBBLE_OPPONENT_DIST:
                            near_opponent = True
                            break
                if near_opponent:
                    self._dribble_frames[tid] = self._dribble_frames.get(tid, 0) + 1
                    if (
                        self._dribble_frames[tid] >= self.DRIBBLE_MIN_FRAMES
                        and not self._in_dribble.get(tid, False)
                    ):
                        events.append((tid, "dribble_success"))
                        self._in_dribble[tid] = True
                else:
                    self._dribble_frames[tid] = 0
                    self._in_dribble[tid] = False
            else:
                # Lost the ball while dribbling = dribble fail
                if self._in_dribble.get(tid, False):
                    events.append((tid, "dribble_fail"))
                self._in_dribble[tid] = False
                self._dribble_frames[tid] = 0

        return events


def _bbox_iou(box1: tuple, box2: tuple) -> float:
    """Compute intersection-over-union of two (x1,y1,x2,y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0
