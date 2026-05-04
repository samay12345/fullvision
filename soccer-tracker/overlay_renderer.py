"""
overlay_renderer.py — OpenCV drawing logic for per-player panels and sidebar HUD.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from stats_tracker import PlayerStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _semi_transparent_rect(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple = (10, 10, 10),
    alpha: float = 0.62,
) -> None:
    """Draw a semi-transparent filled rectangle on *frame* in-place."""
    x1, y1 = max(0, x), max(0, y)
    x2 = min(frame.shape[1], x + w)
    y2 = min(frame.shape[0], y + h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


# ---------------------------------------------------------------------------
# OverlayRenderer
# ---------------------------------------------------------------------------

class OverlayRenderer:
    """All OpenCV drawing logic for per-player panels and the sidebar HUD."""

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

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
    ) -> np.ndarray:
        """Draw an info panel above the player bounding box.

        Parameters
        ----------
        frame : np.ndarray
        bbox : tuple  (x1, y1, x2, y2)
        track_id : int
        identity : dict  {jersey_number, name, fc26_stats, team}
        live_stats : PlayerStats | None
        team_color : BGR tuple
        """
        x1, y1, x2, y2 = map(int, bbox)
        bbox_h = y2 - y1
        bbox_w = x2 - x1

        font_scale = max(0.3, bbox_h / 300.0)
        thickness = 1

        jersey_number = identity.get("jersey_number")
        name = identity.get("name")
        fc26 = identity.get("fc26_stats")

        # ── Build text lines ──────────────────────────────────────────
        if name:
            surname = name.split()[-1]
            id_part = f"#{jersey_number} {surname}" if jersey_number else surname
        elif jersey_number:
            id_part = f"#{jersey_number}"
        else:
            id_part = f"#{track_id}"

        ovr_str = f"  OVR {fc26['overall']}" if fc26 and fc26.get("overall") else ""
        line1 = f"{id_part}{ovr_str}"

        if live_stats:
            spd = live_stats.speed_history[-1] if live_stats.speed_history else 0.0
            line2 = f"SPD: {spd:.1f} m/s  |  TOP: {live_stats.top_speed_ms:.1f} m/s"
            line3 = f"DIST: {live_stats.distance_m:.0f}m  |  SPRINTS: {live_stats.sprint_count}"
        else:
            line2 = "SPD: 0.0 m/s  |  TOP: 0.0 m/s"
            line3 = "DIST: 0m  |  SPRINTS: 0"

        lines = [line1, line2, line3]

        if fc26:
            pac = fc26.get("pace") or 0
            sho = fc26.get("shooting") or 0
            pas = fc26.get("passing") or 0
            dri = fc26.get("dribbling") or 0
            def_ = fc26.get("defending") or 0
            phy = fc26.get("physical") or 0
            line4 = f"PAC:{pac} SHO:{sho} PAS:{pas} DRI:{dri} DEF:{def_} PHY:{phy}"
            lines.append(line4)

        # ── Measure panel dimensions ──────────────────────────────────
        line_heights = []
        line_widths = []
        for line in lines:
            (tw, th), baseline = cv2.getTextSize(line, self.font, font_scale, thickness)
            line_heights.append(th + baseline + 3)
            line_widths.append(tw)

        panel_w = max(line_widths) + 10
        line_h = max(line_heights) if line_heights else 14
        panel_h = sum(line_heights) + 8

        # ── Panel position: above bounding box ───────────────────────
        fh, fw = frame.shape[:2]
        panel_x = x1
        panel_y = y1 - panel_h - 4

        # If near top edge, shift panel below the box
        if panel_y < 2:
            panel_y = y2 + 4

        # Clamp horizontally
        if panel_x + panel_w > fw:
            panel_x = max(0, fw - panel_w)

        panel_y = max(0, panel_y)

        # ── Draw semi-transparent background ─────────────────────────
        _semi_transparent_rect(frame, panel_x, panel_y, panel_w, panel_h, (10, 10, 10), 0.70)

        # ── Draw team-colored border ──────────────────────────────────
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            team_color,
            2,
        )

        # ── Draw text ─────────────────────────────────────────────────
        cursor_y = panel_y + line_heights[0] + 2
        for i, line in enumerate(lines):
            cv2.putText(
                frame, line,
                (panel_x + 4, cursor_y),
                self.font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )
            if i + 1 < len(line_heights):
                cursor_y += line_heights[i + 1]

        return frame

    # ------------------------------------------------------------------
    # Bounding box
    # ------------------------------------------------------------------

    def draw_bounding_box(
        self,
        frame: np.ndarray,
        bbox: tuple,
        team_color: tuple,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw a colored rectangle for a player bounding box."""
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, thickness)
        return frame

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
        """Draw a right-side HUD table with player stats.

        players_data items expected keys:
            track_id, name, jersey_number, team, team_color (BGR tuple),
            overall, avg_speed, top_speed, distance, sprints
        Sorted by avg_speed descending; up to 22 rows displayed.
        """
        if not players_data:
            return frame

        # Sort by avg_speed descending
        sorted_players = sorted(players_data, key=lambda p: p.get("avg_speed", 0.0), reverse=True)
        rows = sorted_players[:22]

        col_w = 310
        row_h = 18
        header_h = 38
        pad = 4
        panel_h = header_h + len(rows) * row_h + pad
        px = frame_w - col_w - 6
        py = 6

        # Semi-transparent background
        _semi_transparent_rect(frame, px, py, col_w, panel_h, (10, 10, 10), 0.72)
        cv2.rectangle(frame, (px, py), (px + col_w, py + panel_h), (60, 60, 60), 1)

        # Header in gold
        gold = (0, 215, 255)
        cv2.putText(frame, "LIVE STATS", (px + 6, py + 14),
                    self.font, 0.42, gold, 1, cv2.LINE_AA)
        header_txt = f"{'NAME':<16} {'OVR':>4} {'SPD':>5} {'TOP':>5} {'DIST':>6} {'SPR':>4}"
        cv2.putText(frame, header_txt, (px + 4, py + 30),
                    self.font, 0.30, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.line(frame, (px + 3, py + header_h - 2), (px + col_w - 3, py + header_h - 2),
                 (60, 60, 60), 1)

        for i, p in enumerate(rows):
            ry = py + header_h + i * row_h + row_h - 3
            tc = p.get("team_color", (180, 180, 180))

            # Name
            name = p.get("name") or ""
            jnum = p.get("jersey_number")
            if name:
                display = name.split()[-1]
                if jnum:
                    display = f"#{jnum} {display}"
            elif jnum:
                display = f"#{jnum}"
            else:
                display = f"#{p.get('track_id', '?')}"
            if len(display) > 16:
                display = display[:15] + "."

            ovr = p.get("overall") or 0
            spd = p.get("avg_speed", 0.0)
            top = p.get("top_speed", 0.0)
            dist = p.get("distance", 0.0)
            spr = p.get("sprints", 0)

            row_txt = (
                f"{display:<16} "
                f"{ovr:>4} "
                f"{spd:>5.1f} "
                f"{top:>5.1f} "
                f"{dist:>6.0f} "
                f"{spr:>4}"
            )

            # Team colored dot
            cv2.circle(frame, (px + 6, ry - 5), 3, tc, -1)

            cv2.putText(frame, row_txt, (px + 12, ry),
                        self.font, 0.30, (215, 215, 215), 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Ball indicator
    # ------------------------------------------------------------------

    def draw_ball_indicator(
        self,
        frame: np.ndarray,
        ball_pos: tuple,
    ) -> np.ndarray:
        """Draw a white circle with cross-hair at ball_pos."""
        cx, cy = int(ball_pos[0]), int(ball_pos[1])
        radius = 9
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
        # Cross-hair lines
        cv2.line(frame, (cx - radius - 4, cy), (cx + radius + 4, cy), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - radius - 4), (cx, cy + radius + 4), (255, 255, 255), 1, cv2.LINE_AA)
        return frame
