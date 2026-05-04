"""
overlay_renderer.py — Per-player panels and sidebar HUD.

Performance design: all semi-transparent backgrounds are collected into a
single overlay layer and blended ONCE per frame instead of once per player.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from stats_tracker import PlayerStats

_MIN_PANEL_HEIGHT = 35   # don't draw panels for tiny/distant players


class OverlayRenderer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self._bg_layer: Optional[np.ndarray] = None   # single overlay per frame
        self._pending_texts: list[dict] = []           # text to draw after blend
        self._pending_borders: list[dict] = []         # colored borders after blend

    # ------------------------------------------------------------------
    # Frame lifecycle — call begin_frame() once, then all draw_* methods,
    # then flush() to apply the blended backgrounds.
    # ------------------------------------------------------------------

    def begin_frame(self, frame: np.ndarray) -> None:
        """Allocate the background layer for this frame."""
        if self._bg_layer is None or self._bg_layer.shape != frame.shape:
            self._bg_layer = np.zeros_like(frame)
        else:
            self._bg_layer[:] = 0
        self._pending_texts = []
        self._pending_borders = []

    def flush(self, frame: np.ndarray, alpha: float = 0.68) -> np.ndarray:
        """Blend the accumulated background layer onto *frame*, then draw text."""
        if self._bg_layer is not None:
            mask = np.any(self._bg_layer > 0, axis=2)
            if mask.any():
                blended = cv2.addWeighted(self._bg_layer, alpha, frame, 1.0 - alpha, 0)
                frame[mask] = blended[mask]

        # Draw colored borders (after blend so they stay sharp)
        for b in self._pending_borders:
            cv2.rectangle(frame, b["pt1"], b["pt2"], b["color"], b["thick"])

        # Draw text last so it sits on top of everything
        for t in self._pending_texts:
            cv2.putText(
                frame, t["text"], t["pos"],
                self.font, t["scale"], t["color"], t["thick"], cv2.LINE_AA,
            )
        return frame

    def _queue_rect(self, x: int, y: int, w: int, h: int, color: tuple) -> None:
        fh, fw = self._bg_layer.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(self._bg_layer, (x1, y1), (x2, y2), color, -1)

    def _queue_text(self, text, pos, scale, color, thick=1):
        self._pending_texts.append(
            {"text": text, "pos": pos, "scale": scale, "color": color, "thick": thick}
        )

    def _queue_border(self, pt1, pt2, color, thick=2):
        self._pending_borders.append(
            {"pt1": pt1, "pt2": pt2, "color": color, "thick": thick}
        )

    # ------------------------------------------------------------------
    # Per-player panel
    # ------------------------------------------------------------------

    def draw_player_panel(
        self,
        frame: np.ndarray,
        bbox: tuple,
        track_id: int,
        identity: dict,
        live_stats: Optional[PlayerStats],
        team_color: tuple,
    ) -> None:
        x1, y1, x2, y2 = map(int, bbox)
        bbox_h = y2 - y1
        if bbox_h < _MIN_PANEL_HEIGHT:
            # Still draw the box, skip the expensive text panel
            self._queue_border((x1, y1), (x2, y2), team_color, 2)
            return

        font_scale = max(0.28, min(0.40, bbox_h / 280.0))
        thickness = 1

        jersey_number = identity.get("jersey_number")
        name = identity.get("name")
        overall = identity.get("overall")
        pac = identity.get("pace")
        sho = identity.get("shooting")
        pas = identity.get("passing")
        dri = identity.get("dribbling")
        def_ = identity.get("defending")
        phy = identity.get("physical")

        if name:
            surname = name.split()[-1]
            id_part = f"#{jersey_number} {surname}" if jersey_number else surname
        elif jersey_number:
            id_part = f"#{jersey_number}"
        else:
            id_part = f"#{track_id}"

        ovr_str = f" OVR:{overall}" if overall else ""
        line1 = f"{id_part}{ovr_str}"

        if live_stats:
            spd = live_stats.speed_history[-1] if live_stats.speed_history else 0.0
            line2 = f"{spd:.1f}m/s  TOP:{live_stats.top_speed_ms:.1f}  D:{live_stats.distance_m:.0f}m  S:{live_stats.sprint_count}"
        else:
            line2 = "0.0m/s  TOP:0.0  D:0m  S:0"

        lines = [line1, line2]
        if pac or sho:
            lines.append(
                f"PAC:{pac or '--'} SHO:{sho or '--'} PAS:{pas or '--'} "
                f"DRI:{dri or '--'} DEF:{def_ or '--'} PHY:{phy or '--'}"
            )

        # Measure panel
        line_h = int(cv2.getTextSize("Ag", self.font, font_scale, thickness)[0][1] * 1.8)
        panel_w = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, self.font, font_scale, thickness)
            panel_w = max(panel_w, tw + 8)
        panel_h = line_h * len(lines) + 6

        fh, fw = frame.shape[:2]
        px = max(0, min(x1, fw - panel_w))
        py = y1 - panel_h - 3
        if py < 0:
            py = y2 + 3
        py = max(0, min(py, fh - panel_h))

        # Queue background rect (batched)
        self._queue_rect(px, py, panel_w, panel_h, (15, 15, 15))
        # Queue colored border
        self._queue_border((px, py), (px + panel_w, py + panel_h), team_color, 1)
        # Queue player bbox border
        self._queue_border((x1, y1), (x2, y2), team_color, 2)

        # Queue text lines
        cy = py + line_h
        for line in lines:
            self._queue_text(line, (px + 3, cy), font_scale, (235, 235, 235), thickness)
            cy += line_h

    # ------------------------------------------------------------------
    # Sidebar HUD
    # ------------------------------------------------------------------

    def draw_sidebar(
        self,
        frame: np.ndarray,
        players_data: list[dict],
        frame_w: int,
        frame_h: int,
    ) -> np.ndarray:
        if not players_data:
            return frame

        sorted_p = sorted(players_data, key=lambda p: p.get("avg_speed", 0.0), reverse=True)
        rows = sorted_p[:20]

        col_w = 300
        row_h = 17
        header_h = 36
        pad = 4
        panel_h = header_h + len(rows) * row_h + pad
        px = frame_w - col_w - 6
        py = 6

        # Draw directly (sidebar uses its own blend path)
        ol = frame.copy()
        cv2.rectangle(ol, (px, py), (px + col_w, py + panel_h), (10, 10, 10), -1)
        cv2.addWeighted(ol, 0.72, frame, 0.28, 0, frame)

        cv2.rectangle(frame, (px, py), (px + col_w, py + panel_h), (55, 55, 55), 1)

        gold = (0, 215, 255)
        cv2.putText(frame, "LIVE STATS", (px + 6, py + 13),
                    self.font, 0.38, gold, 1, cv2.LINE_AA)
        hdr = f"{'NAME':<15} {'OVR':>3} {'SPD':>5} {'TOP':>5} {'D':>5} {'S':>3}"
        cv2.putText(frame, hdr, (px + 4, py + 28),
                    self.font, 0.28, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.line(frame, (px + 3, py + header_h - 2),
                 (px + col_w - 3, py + header_h - 2), (55, 55, 55), 1)

        for i, p in enumerate(rows):
            ry = py + header_h + i * row_h + row_h - 3
            tc = p.get("team_color", (180, 180, 180))
            name = p.get("name") or ""
            jnum = p.get("jersey_number")
            if name:
                display = name.split()[-1]
                display = f"#{jnum} {display}" if jnum else display
            elif jnum:
                display = f"#{jnum}"
            else:
                display = f"#{p.get('track_id', '?')}"
            if len(display) > 15:
                display = display[:14] + "."

            ovr = p.get("overall") or 0
            row_txt = (
                f"{display:<15} {ovr:>3} "
                f"{p.get('avg_speed', 0):>5.1f} "
                f"{p.get('top_speed', 0):>5.1f} "
                f"{p.get('distance', 0):>5.0f} "
                f"{p.get('sprints', 0):>3}"
            )
            cv2.circle(frame, (px + 5, ry - 5), 3, tc, -1)
            cv2.putText(frame, row_txt, (px + 11, ry),
                        self.font, 0.28, (215, 215, 215), 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Ball indicator
    # ------------------------------------------------------------------

    def draw_ball_indicator(self, frame: np.ndarray, ball_pos: tuple) -> np.ndarray:
        cx, cy = int(ball_pos[0]), int(ball_pos[1])
        r = 9
        cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
        cv2.line(frame, (cx - r - 4, cy), (cx + r + 4, cy), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - r - 4), (cx, cy + r + 4), (255, 255, 255), 1, cv2.LINE_AA)
        return frame
