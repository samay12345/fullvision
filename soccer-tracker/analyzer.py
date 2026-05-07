"""
analyzer.py — Core analysis pipeline.

Consumes frames from frame_queue, runs YOLO + ByteTrack, classifies teams,
runs OCR for jersey numbers, builds per-player stats, annotates frames,
and pushes JSON payloads to broadcast_queue.
"""
from __future__ import annotations

import asyncio
import base64
import math
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np

from config import (
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    TRACKER,
    YOLO_MODEL,
    PERSON_CLASS_ID,
    BALL_CLASS_ID,
    SPEED_SCALE_FACTOR,
    TEAM_KIT_COLORS_BGR,
)
from tracker import PlayerRegistry, PlayerTracker, merge_player
from team_classifier import TeamClassifier
from stat_engine import StatEngine
from player_identity import PlayerIdentityManager
from reid import ReIDTracker
from stats_tracker import PlayerStatsTracker
from rating_engine import RatingEngine, EventDetector
from fc26_loader import FC26Loader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YOLO_EVERY_N = 3           # run YOLO every N frames
TEAM_A_COLOR_BGR = (255, 80, 40)    # blue
TEAM_B_COLOR_BGR = (40, 40, 220)    # red
UNKNOWN_COLOR_BGR = (160, 160, 160)
FONT = cv2.FONT_HERSHEY_SIMPLEX
MIN_PANEL_H = 30            # skip text panel for tiny bboxes
PIXELS_PER_METER = 10.0
RATING_DECAY_INTERVAL_S = 5.0


def _team_color(team: str | None) -> tuple[int, int, int]:
    if team == "A":
        return TEAM_A_COLOR_BGR
    if team == "B":
        return TEAM_B_COLOR_BGR
    return UNKNOWN_COLOR_BGR


def _rating_color(rating: float) -> tuple[int, int, int]:
    """BGR color for rating value."""
    if rating >= 7.5:
        return (80, 200, 80)   # green
    if rating >= 6.0:
        return (40, 190, 220)  # yellow (BGR)
    return (50, 50, 210)       # red


class Analyzer:
    """
    Processes frames from *frame_queue* and pushes WS payloads to
    *broadcast_queue* (an asyncio.Queue in the server event loop).

    All heavy work runs in the calling thread (a ThreadPoolExecutor worker).
    """

    def __init__(
        self,
        frame_queue: "queue.Queue[np.ndarray]",
        broadcast_queue: asyncio.Queue,
        fps: float,
        loop: asyncio.AbstractEventLoop,
        home_team: str = "",
        away_team: str = "",
    ):
        self._frame_queue = frame_queue
        self._broadcast_queue = broadcast_queue
        self._fps = max(fps, 1.0)
        self._loop = loop
        self._stop_event = threading.Event()

        # ── Tracking components ───────────────────────────────────────────────
        self._registry = PlayerRegistry()
        self._classifier = TeamClassifier()
        self._stat_engine = StatEngine(self._registry)
        self._identity_mgr = PlayerIdentityManager()
        self._reid = ReIDTracker()
        self._stats_tracker = PlayerStatsTracker(fps=int(self._fps), pixels_per_meter=PIXELS_PER_METER)
        self._rating_engine = RatingEngine()
        self._event_detector = EventDetector()
        self._fc26 = FC26Loader()

        # Load team rosters and store kit colors for classifier alignment
        self._home_kit_bgr: tuple | None = None
        self._away_kit_bgr: tuple | None = None
        self._classifier_aligned = False
        if home_team and away_team:
            home_key, away_key = self._identity_mgr.set_teams(home_team, away_team)
            if home_key:
                self._home_kit_bgr = TEAM_KIT_COLORS_BGR.get(home_key.lower())
            if away_key:
                self._away_kit_bgr = TEAM_KIT_COLORS_BGR.get(away_key.lower())

        # ── YOLO model (lazy load on first frame) ────────────────────────────
        self._model = None

        # ── State ────────────────────────────────────────────────────────────
        self._frame_num = 0
        self._last_player_detections: list[dict] = []
        self._last_ball_pos: tuple[int, int] | None = None
        self._prev_active_ids: set[int] = set()
        self._start_time = time.monotonic()
        self._last_decay_ts = 0.0
        self._key_events: dict[int, list[str]] = {}   # track_id -> recent key events
        self._match_info: str = ""
        # GK detection: collect (tid, crop) per team every GK_COLLECT_EVERY frames
        self._gk_crops: dict[str, list[tuple[int, np.ndarray]]] = {"A": [], "B": []}
        self._GK_DETECT_EVERY = 90   # run GK color detection every N frames
        self._GK_MIN_PLAYERS  = 5    # need at least this many per team

        # Possession + xG + event log + recording
        self._possession_frames: dict[str, int] = {"A": 0, "B": 0}
        self._xg_totals: dict[str, float] = {"A": 0.0, "B": 0.0}
        self._event_log: list[dict] = []   # max 200 entries
        self._should_record: bool = False
        self._recorder = None              # cv2.VideoWriter when active
        self._recording_path: str | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_xg(self, ball_pos: tuple | None, frame_shape: tuple) -> float:
        """Rule-based xG from shot distance to nearest goal (normalized coords)."""
        if ball_pos is None:
            return 0.03
        bx, by = ball_pos
        fh, fw = frame_shape[:2]
        if fw == 0 or fh == 0:
            return 0.03
        nx, ny = bx / fw, by / fh
        dist = min(
            math.sqrt(nx**2 + (ny - 0.5)**2),
            math.sqrt((1 - nx)**2 + (ny - 0.5)**2),
        )
        if dist < 0.08:  return 0.50
        if dist < 0.12:  return 0.30
        if dist < 0.18:  return 0.16
        if dist < 0.25:  return 0.08
        if dist < 0.35:  return 0.04
        return 0.02

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(YOLO_MODEL)
        return self._model

    def _timestamp_s(self) -> float:
        return (self._frame_num / self._fps)

    def _push(self, payload: dict) -> None:
        """Thread-safely enqueue payload to the async broadcast queue."""
        async def _put():
            # If queue is at capacity, drop the oldest
            if self._broadcast_queue.full():
                try:
                    self._broadcast_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                self._broadcast_queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass  # drop this frame

        asyncio.run_coroutine_threadsafe(_put(), self._loop)

    # ------------------------------------------------------------------
    # Frame annotation
    # ------------------------------------------------------------------

    def _draw_frame(
        self,
        frame: np.ndarray,
        player_detections: list[dict],
        ball_pos: tuple | None,
        highlighted_id: int | None = None,
    ) -> np.ndarray:
        """Draw all overlays onto frame in-place (no per-player copy)."""

        # Collect all background rects and labels first, blend once
        overlay_layer = np.zeros_like(frame)
        texts: list[dict] = []
        borders: list[dict] = []

        for det in player_detections:
            tid = det["tracker_id"]
            x1, y1, x2, y2 = map(int, det["bbox"])
            team = det.get("team") or "unknown"
            color = _team_color(team)

            player = self._registry.players.get(tid)
            identity = self._identity_mgr.get_cached_identity(tid) or {}
            live_stats = self._stats_tracker.get_stats(tid)
            rating = self._rating_engine.get_rating(tid)

            # ── Header label: #10 Mbappe [★7.6↑] ────────────────────────────
            name = identity.get("name") or (player.player_name if player else None)
            jersey = identity.get("jersey_number") or (player.jersey_number if player else None)
            delta = self._rating_engine.get_delta(tid)
            arrow = "+" if delta > 0.001 else ("-" if delta < -0.001 else "=")
            if name:
                surname = name.split()[-1]
                id_part = f"#{jersey} {surname}" if jersey else surname
            elif jersey:
                id_part = f"#{jersey}"
            else:
                id_part = f"#{tid}"
            star_str = f"[{rating:.1f}{arrow}]"
            top_label = f"{id_part}  {star_str}"

            # ── Bottom label: speed · distance · sprints ──────────────────
            if live_stats:
                spd = live_stats.speed_history[-1] if live_stats.speed_history else 0.0
                dist = live_stats.distance_m
                spr = live_stats.sprint_count
                bot_label = f"{spd:.1f}m/s  {dist:.0f}m  {spr}spr"
            else:
                bot_label = "0.0m/s  0m  0spr"

            bbox_h = y2 - y1
            if bbox_h >= MIN_PANEL_H:
                font_scale = max(0.27, min(0.38, bbox_h / 300.0))
                thick = 1

                # Measure label widths
                (tw1, th1), _ = cv2.getTextSize(top_label, FONT, font_scale, thick)
                (tw2, _), _ = cv2.getTextSize(bot_label, FONT, font_scale, thick)
                panel_w = max(tw1, tw2) + 8
                panel_h = (th1 + 4) * 2 + 6

                fh_frame, fw_frame = frame.shape[:2]
                px = max(0, min(x1, fw_frame - panel_w))
                py = y1 - panel_h - 3
                if py < 0:
                    py = y2 + 3
                py = max(0, min(py, fh_frame - panel_h))

                # Background rect into overlay layer
                cv2.rectangle(overlay_layer, (px, py), (px + panel_w, py + panel_h), (20, 20, 20), -1)

                rcolor = _rating_color(rating)
                borders.append({"pt1": (px, py), "pt2": (px + panel_w, py + panel_h), "color": color, "thick": 1})
                texts.append({
                    "text": top_label, "pos": (px + 3, py + th1 + 3),
                    "scale": font_scale, "color": (235, 235, 235), "thick": thick,
                })
                texts.append({
                    "text": bot_label, "pos": (px + 3, py + (th1 + 4) * 2),
                    "scale": font_scale, "color": (180, 180, 180), "thick": thick,
                })

            # Player bbox border
            thick_border = 3 if tid == highlighted_id else 2
            borders.append({"pt1": (x1, y1), "pt2": (x2, y2), "color": color, "thick": thick_border})

        # Single blend for all panel backgrounds
        mask = np.any(overlay_layer > 0, axis=2)
        if mask.any():
            blended = cv2.addWeighted(overlay_layer, 0.65, frame, 0.35, 0)
            frame[mask] = blended[mask]

        # Draw borders and text after blend
        for b in borders:
            cv2.rectangle(frame, b["pt1"], b["pt2"], b["color"], b["thick"])
        for t in texts:
            cv2.putText(frame, t["text"], t["pos"], FONT, t["scale"], t["color"], t["thick"], cv2.LINE_AA)

        # Ball indicator
        if ball_pos is not None:
            bx, by = int(ball_pos[0]), int(ball_pos[1])
            cv2.circle(frame, (bx, by), 9, (255, 255, 255), 2)
            cv2.circle(frame, (bx, by), 3, (255, 255, 255), -1)
            cv2.line(frame, (bx - 13, by), (bx + 13, by), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (bx, by - 13), (bx, by + 13), (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Build WS payload
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        frame: np.ndarray,
        player_detections: list[dict],
        ball_pos: tuple | None,
    ) -> dict:
        timestamp_s = self._timestamp_s()
        players_out: list[dict] = []

        for det in player_detections:
            tid = det["tracker_id"]
            player = self._registry.players.get(tid)
            identity = self._identity_mgr.get_cached_identity(tid) or {}
            live_stats = self._stats_tracker.get_stats(tid)
            rating = self._rating_engine.get_rating(tid)
            delta = self._rating_engine.get_delta(tid)
            last_event = self._rating_engine.last_event.get(tid, "")
            fc26_stats = self._fc26.get_for_track(tid)

            name = identity.get("name") or (player.player_name if player else None)
            jersey = identity.get("jersey_number") or (player.jersey_number if player else None)
            team = det.get("team") or "unknown"

            x1, y1, x2, y2 = map(int, det["bbox"])
            bbox = [x1, y1, x2 - x1, y2 - y1]

            speed_ms = 0.0
            top_speed_ms = 0.0
            distance_m = 0.0
            sprints = 0
            if live_stats:
                speed_ms = live_stats.speed_history[-1] if live_stats.speed_history else 0.0
                top_speed_ms = live_stats.top_speed_ms
                distance_m = live_stats.distance_m
                sprints = live_stats.sprint_count

            fc26_out: dict | None = None
            if fc26_stats:
                fc26_out = {
                    "overall":   fc26_stats.get("overall"),
                    "pace":      fc26_stats.get("pace"),
                    "shooting":  fc26_stats.get("shooting"),
                    "passing":   fc26_stats.get("passing"),
                    "dribbling": fc26_stats.get("dribbling"),
                    "defending": fc26_stats.get("defending"),
                    "physical":  fc26_stats.get("physical"),
                }
            elif identity.get("overall"):
                fc26_out = {
                    "overall":   identity.get("overall"),
                    "pace":      identity.get("pace"),
                    "shooting":  identity.get("shooting"),
                    "passing":   identity.get("passing"),
                    "dribbling": identity.get("dribbling"),
                    "defending": identity.get("defending"),
                    "physical":  identity.get("physical"),
                }

            players_out.append({
                "track_id":       tid,
                "name":           name,
                "jersey":         jersey,
                "team":           team,
                "bbox":           bbox,
                "speed_ms":       round(speed_ms, 2),
                "top_speed_ms":   round(top_speed_ms, 2),
                "distance_m":     round(distance_m, 1),
                "sprints":        sprints,
                "rating":         round(rating, 2),
                "rating_delta":   round(delta, 3),
                "last_event":     last_event,
                "fc26":           fc26_out,
            })

        ball_out = None
        if ball_pos is not None:
            ball_out = {"x": int(ball_pos[0]), "y": int(ball_pos[1])}

        # Encode frame to JPEG base64
        _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")

        return {
            "frame_id":    self._frame_num,
            "timestamp_s": round(timestamp_s, 2),
            "players":     players_out,
            "ball":        ball_out,
            "frame_b64":   frame_b64,
            "possession_a": round(100 * self._possession_frames.get("A", 0) / max(1, sum(self._possession_frames.values()))),
            "possession_b": round(100 * self._possession_frames.get("B", 0) / max(1, sum(self._possession_frames.values()))),
            "xg_a": round(self._xg_totals.get("A", 0.0), 2),
            "xg_b": round(self._xg_totals.get("B", 0.0), 2),
            "event_log": self._event_log[-25:],
            "frame_w": frame.shape[1],
            "frame_h": frame.shape[0],
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main analysis loop — runs in a background thread."""
        model = self._load_model()
        start_real = time.monotonic()
        fps_accumulator = 0.0
        fps_alpha = 0.1

        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            t0 = time.monotonic()
            run_yolo = (self._frame_num % YOLO_EVERY_N == 0)

            player_detections: list[dict] = []
            ball_pos: tuple[int, int] | None = None
            ball_bbox: tuple | None = None

            if run_yolo:
                try:
                    results = model.track(
                        source=frame,
                        conf=CONFIDENCE_THRESHOLD,
                        iou=IOU_THRESHOLD,
                        tracker=TRACKER,
                        persist=True,
                        classes=[PERSON_CLASS_ID, BALL_CLASS_ID],
                        verbose=False,
                    )[0]
                except Exception:
                    results = None

                current_active_ids: set[int] = set()

                if (
                    results is not None
                    and results.boxes is not None
                    and results.boxes.id is not None
                ):
                    for box, cls, track_id in zip(
                        results.boxes.xyxy.cpu().numpy(),
                        results.boxes.cls.cpu().numpy(),
                        results.boxes.id.cpu().numpy(),
                    ):
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        class_id = int(cls)
                        tid = int(track_id)

                        if class_id == PERSON_CLASS_ID:
                            current_active_ids.add(tid)

                            # Re-ID
                            if tid not in self._registry.players:
                                old_id = self._reid.match_new_id(tid, frame, (x1, y1, x2, y2), self._frame_num)
                                if old_id is not None:
                                    self._registry.get_or_create(tid)
                                    merge_player(self._registry, old_id, tid)
                                    # Transfer identity and rating from the old track
                                    old_identity = self._identity_mgr.get_cached_identity(old_id)
                                    if old_identity and old_identity.get("jersey_number") is not None:
                                        self._identity_mgr._id_cache[tid] = old_identity.copy()
                                    old_rating = self._rating_engine.ratings.get(old_id)
                                    if old_rating is not None:
                                        self._rating_engine.ratings[tid] = old_rating
                                        old_name = self._rating_engine._player_names.get(old_id, "")
                                        if old_name:
                                            self._rating_engine._player_names[tid] = old_name

                            player = self._registry.get_or_create(tid)
                            player.last_seen_frame = self._frame_num

                            # Team classification
                            if self._frame_num % 10 == 0:
                                self._classifier.add_sample(frame, (x1, y1, x2, y2), tid)
                            if not self._classifier.fitted and self._frame_num > 30:
                                self._classifier.fit()
                                # Align cluster 0/1 → "A"/"B" using kit colors
                                if (not self._classifier_aligned
                                        and self._home_kit_bgr and self._away_kit_bgr):
                                    self._classifier.auto_assign(
                                        list(self._home_kit_bgr), list(self._away_kit_bgr), "A", "B"
                                    )
                                    self._classifier_aligned = True
                            team = self._classifier.classify(frame, (x1, y1, x2, y2))
                            player.team = team

                            # Collect player crop for GK color detection
                            if team in ("A", "B"):
                                crop = frame[max(0,y1):max(y1+1,y2), max(0,x1):max(x1+1,x2)]
                                buf = self._gk_crops[team]
                                # Replace existing entry for this tid, or append
                                for i, (btid, _) in enumerate(buf):
                                    if btid == tid:
                                        buf[i] = (tid, crop)
                                        break
                                else:
                                    buf.append((tid, crop))

                            # OCR jersey identity
                            identity = self._identity_mgr.get_identity(
                                tid, frame, (x1, y1, x2, y2), team, self._frame_num
                            )

                            # FC26 lookup when identity resolves
                            name = identity.get("name")
                            jersey_num = identity.get("jersey_number")
                            if name:
                                self._fc26.attach_to_track(tid, name)
                                self._rating_engine.register_name(tid, name)
                            elif jersey_num and team in ("A", "B"):
                                self._fc26.attach_by_jersey(tid, team, jersey_num)

                            # Speed tracking — use instantaneous speed for responsiveness
                            player.update_position(cx, cy, self._frame_num)
                            speed_ms = player.speeds[-1] if player.speeds else 0.0
                            self._stats_tracker.update(
                                tid, cx, cy, speed_ms, (x1, y1, x2, y2),
                                ball_bbox,
                            )

                            player_detections.append({
                                "tracker_id": tid,
                                "bbox": (x1, y1, x2, y2),
                                "center": (cx, cy),
                                "team": team,
                            })

                            # Re-ID update
                            self._reid.update(tid, frame, (x1, y1, x2, y2), self._frame_num)

                        elif class_id == BALL_CLASS_ID:
                            ball_pos = (cx, cy)
                            ball_bbox = (x1, y1, x2, y2)

                # Mark disappeared IDs as lost
                disappeared = self._prev_active_ids - current_active_ids
                for lost_tid in disappeared:
                    self._reid.mark_lost(lost_tid, self._frame_num)
                self._prev_active_ids = current_active_ids

                self._last_player_detections = player_detections
                self._last_ball_pos = ball_pos

                # Periodic GK auto-detection by jersey color
                if self._frame_num % self._GK_DETECT_EVERY == 0:
                    for team_label in ("A", "B"):
                        crops = self._gk_crops[team_label]
                        if len(crops) >= self._GK_MIN_PLAYERS:
                            self._identity_mgr.detect_gk_by_color(team_label, crops)
            else:
                player_detections = self._last_player_detections
                ball_pos = self._last_ball_pos

            # Update stat engine
            if run_yolo:
                self._stat_engine.update(player_detections, ball_pos, self._frame_num)

                # Possession: assign ball to nearest player's team
                if ball_pos is not None and player_detections:
                    bx2, by2 = ball_pos
                    best_dist = float("inf")
                    poss_team = None
                    for det in player_detections:
                        cx2, cy2 = det["center"]
                        d2 = (cx2 - bx2)**2 + (cy2 - by2)**2
                        if d2 < best_dist:
                            best_dist = d2
                            poss_team = det.get("team")
                    if poss_team in ("A", "B") and best_dist < 150**2:
                        self._possession_frames[poss_team] = self._possession_frames.get(poss_team, 0) + 1

            # Build speed map for event detector
            speed_map: dict[int, float] = {}
            for det in player_detections:
                tid = det["tracker_id"]
                p = self._registry.players.get(tid)
                if p:
                    speed_map[tid] = p.speeds[-1] if p.speeds else 0.0

            # Event detection + rating updates
            if run_yolo:
                events = self._event_detector.detect(
                    player_detections, ball_pos, speed_map, self._frame_num,
                    frame_shape=frame.shape,
                )
                ts = self._timestamp_s()
                for tid, event_type in events:
                    p = self._registry.players.get(tid)
                    name = (p.player_name if p else None) or ""
                    self._rating_engine.update(tid, event_type, self._frame_num, ts, name)
                    # Track key events per player for summary
                    buf = self._key_events.setdefault(tid, [])
                    buf.append(event_type)
                    if len(buf) > 30:
                        self._key_events[tid] = buf[-30:]

                    # xG on shots
                    if event_type in ("shot_on_target", "shot_off_target"):
                        xg = self._compute_xg(ball_pos, frame.shape)
                        t_team = next((d["team"] for d in player_detections if d["tracker_id"] == tid), None)
                        if t_team in ("A", "B"):
                            self._xg_totals[t_team] = self._xg_totals.get(t_team, 0.0) + xg

                    # Event log entry
                    ev_player = self._registry.players.get(tid)
                    ev_identity = self._identity_mgr.get_cached_identity(tid) or {}
                    ev_name = ev_identity.get("name") or (ev_player.player_name if ev_player else None) or f"#{tid}"
                    ev_team = (ev_player.team if ev_player else None) or "?"
                    ev_jersey = ev_identity.get("jersey_number") or (ev_player.jersey_number if ev_player else None)
                    self._event_log.append({
                        "ts":     round(self._timestamp_s(), 1),
                        "player": ev_name,
                        "jersey": ev_jersey,
                        "team":   ev_team,
                        "event":  event_type,
                    })
                    if len(self._event_log) > 200:
                        self._event_log = self._event_log[-200:]

                # Periodic decay
                self._rating_engine.decay_all(ts)

            # Draw annotated frame
            annotated = self._draw_frame(frame, player_detections, ball_pos)

            # Recording
            if self._should_record and self._recorder is None:
                import datetime, os as _os
                _os.makedirs("recordings", exist_ok=True)
                fname = f"recordings/match_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                fh2, fw2 = annotated.shape[:2]
                self._recorder = cv2.VideoWriter(
                    fname, cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (fw2, fh2)
                )
                self._recording_path = fname
            if self._recorder is not None:
                try:
                    self._recorder.write(annotated)
                except Exception:
                    pass

            # Build and push WS payload (skip if broadcast_queue is already full)
            try:
                payload = self._build_payload(annotated, player_detections, ball_pos)
                self._push(payload)
            except Exception:
                pass  # never block the analysis loop

            self._frame_num += 1

    def get_summary_stats(self) -> list[dict]:
        """Return per-player stats for all roster players.

        Players that were tracked get real stats; players not seen in the
        footage are included with did_not_play=True so Claude can write a
        one-liner for them. Falls back to tracked-only if no roster is loaded.
        """
        # Build (jersey, team) -> stats map from tracking data
        tracked_by_jersey: dict[tuple, dict] = {}
        for tid, player in self._registry.players.items():
            identity = self._identity_mgr.get_cached_identity(tid) or {}
            jersey = identity.get("jersey_number") or player.jersey_number
            team = player.team
            if not jersey or not team:
                continue
            live_stats = self._stats_tracker.get_stats(tid)
            dist = live_stats.distance_m if live_stats else 0.0
            existing = tracked_by_jersey.get((jersey, team))
            if existing is None or dist > existing.get("distance_m", 0):
                tracked_by_jersey[(jersey, team)] = {
                    "track_id":     tid,
                    "name":         identity.get("name") or player.player_name,
                    "jersey":       jersey,
                    "team":         team,
                    "did_not_play": False,
                    "rating":       self._rating_engine.get_rating(tid),
                    "top_speed_ms": live_stats.top_speed_ms if live_stats else 0.0,
                    "distance_m":   dist,
                    "sprints":      live_stats.sprint_count if live_stats else 0,
                    "passes":       player.passes,
                    "tackles":      player.tackles,
                    "shots":        player.shots,
                    "key_events":   self._key_events.get(tid, []),
                }

        rosters = self._identity_mgr._rosters
        result: list[dict] = []

        if rosters:
            # Roster-driven: include everyone (played or not)
            for team_label in ("A", "B"):
                roster = rosters.get(team_label, {})
                for jersey_num, player_data in roster.items():
                    name = player_data.get("full_name") or f"#{jersey_num}"
                    tracked = tracked_by_jersey.get((jersey_num, team_label))
                    if tracked:
                        entry = dict(tracked)
                        entry["name"] = name  # prefer DB name over OCR
                        entry["position"] = player_data.get("position", "")
                    else:
                        entry = {
                            "name":         name,
                            "jersey":       jersey_num,
                            "team":         team_label,
                            "position":     player_data.get("position", ""),
                            "did_not_play": True,
                            "rating":       None,
                            "top_speed_ms": 0.0,
                            "distance_m":   0.0,
                            "sprints":      0,
                            "passes":       0,
                            "tackles":      0,
                            "shots":        0,
                            "key_events":   [],
                        }
                    result.append(entry)
        else:
            # No roster: fall back to tracked players only
            for tid, player in self._registry.players.items():
                identity = self._identity_mgr.get_cached_identity(tid) or {}
                live_stats = self._stats_tracker.get_stats(tid)
                name = identity.get("name") or player.player_name
                if not name and not player.jersey_number:
                    continue
                result.append({
                    "track_id":     tid,
                    "name":         name or f"#{player.jersey_number or tid}",
                    "jersey":       identity.get("jersey_number") or player.jersey_number,
                    "team":         player.team or "unknown",
                    "did_not_play": False,
                    "rating":       self._rating_engine.get_rating(tid),
                    "top_speed_ms": live_stats.top_speed_ms if live_stats else 0.0,
                    "distance_m":   live_stats.distance_m if live_stats else 0.0,
                    "sprints":      live_stats.sprint_count if live_stats else 0,
                    "passes":       player.passes,
                    "tackles":      player.tackles,
                    "shots":        player.shots,
                    "key_events":   self._key_events.get(tid, []),
                })

        return result

    def start_recording(self) -> None:
        self._should_record = True

    def stop_recording(self) -> str | None:
        self._should_record = False
        if self._recorder is not None:
            self._recorder.release()
            self._recorder = None
            path = self._recording_path
            self._recording_path = None
            return path
        return None

    def stop(self) -> None:
        """Signal the analysis loop to stop."""
        self._stop_event.set()
        self._rating_engine.close()

    @property
    def frame_count(self) -> int:
        return self._frame_num

    @property
    def player_count(self) -> int:
        active = self._registry.active_players(self._frame_num)
        return len(active)
