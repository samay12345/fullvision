import cv2
import numpy as np
from config import TEAM_A_COLOR, TEAM_B_COLOR, UNKNOWN_COLOR, MINIMAP_W, MINIMAP_H
from tracker import PlayerRegistry, PlayerTracker

_NAMED_COLORS: dict[str, tuple] = {}


def set_team_colors(home_name: str, home_bgr: tuple, away_name: str, away_bgr: tuple):
    _NAMED_COLORS[home_name] = home_bgr
    _NAMED_COLORS[away_name] = away_bgr


def _team_color(team: str | None) -> tuple:
    if team is None:
        return UNKNOWN_COLOR
    if team in _NAMED_COLORS:
        return _NAMED_COLORS[team]
    if team == "A":
        return TEAM_A_COLOR
    if team == "B":
        return TEAM_B_COLOR
    return UNKNOWN_COLOR


def _rating_color(r: float) -> tuple:
    if r >= 65:
        return (0, 210, 0)       # green
    if r >= 40:
        return (0, 190, 255)     # amber
    return (60, 60, 200)         # red


def _panel(frame, x, y, w, h, alpha=0.62):
    ol = frame.copy()
    cv2.rectangle(ol, (x, y), (x + w, y + h), (10, 10, 10), -1)
    cv2.addWeighted(ol, alpha, frame, 1 - alpha, 0, frame)


class Overlay:
    def __init__(self):
        self.match_info: str = ""

    # ── Per-player bounding box ───────────────────────────────────────────────
    def draw_player(self, frame, bbox, tracker_id, team, stats) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        color = _team_color(team)
        rating = stats.get("rating", 0.0)
        thickness = 3 if rating >= 55 else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Name / number label above box
        name = stats.get("name") or ""
        jnum = stats.get("jersey_number")
        speed = stats.get("avg_speed_mps", 0.0)

        if name:
            short = name.split()[-1]          # surname only
            id_part = f"#{jnum} {short}" if jnum else short
        else:
            id_part = f"#{tracker_id}"

        label = f"{id_part}  {speed:.1f}m/s"
        font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
        (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
        lx = x1
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (lx, ly - th - 3), (lx + tw + 4, ly + 1), color, -1)
        cv2.putText(frame, label, (lx + 2, ly - 1), font, fs, (0, 0, 0), ft, cv2.LINE_AA)

        # Rating badge below box
        if rating > 0:
            badge = f"{rating:.0f}"
            (bw, bh), _ = cv2.getTextSize(badge, font, 0.42, 1)
            bx = (x1 + x2) // 2 - bw // 2
            by = y2 + bh + 4
            rc = _rating_color(rating)
            cv2.rectangle(frame, (bx - 3, y2 + 2), (bx + bw + 3, by + 2), rc, -1)
            cv2.putText(frame, badge, (bx, by), font, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    # ── Ball ─────────────────────────────────────────────────────────────────
    def draw_ball(self, frame, ball_pos) -> np.ndarray:
        cx, cy = ball_pos
        cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
        return frame

    # ── HUD (top-left + right leaderboard) ───────────────────────────────────
    def draw_hud(
        self,
        frame,
        registry: PlayerRegistry,
        frame_num: int,
        formations: dict | None = None,
    ) -> np.ndarray:
        """Draw the HUD panel.

        formations: optional dict like {"BAY": "4-3-3", "PSG": "4-2-3-1"}
        """
        active = registry.active_players(frame_num)
        teams: dict[str, int] = {}
        for p in active:
            if p.team and p.team != "unknown":
                teams[p.team] = teams.get(p.team, 0) + 1

        lines = [f"Frame: {frame_num}", f"On pitch: {len(active)}"]
        if self.match_info:
            lines.insert(0, self.match_info)
        for t, c in sorted(teams.items()):
            form = ""
            if formations and t in formations:
                form = f" [{formations[t]}]"
            lines.append(f"  {t}: {c}{form}")

        line_h, pad = 18, 6
        ph = len(lines) * line_h + pad * 2
        _panel(frame, 8, 8, 210, ph)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (14, 8 + pad + (i + 1) * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        self._draw_leaderboard(frame, active, frame_num)
        return frame

    # ── Right-side live leaderboard ───────────────────────────────────────────
    def _draw_leaderboard(self, frame, active: list[PlayerTracker], frame_num: int):
        # Only players with enough data and a known team
        ranked = [p for p in active
                  if p.team and p.team != "unknown" and len(p.positions) >= 15]
        if not ranked:
            return

        ranked.sort(key=lambda p: p.rating(), reverse=True)
        top = ranked[:14]

        fh, fw = frame.shape[:2]
        col_w = 275
        row_h = 21
        header_h = 42
        pad = 5
        panel_h = header_h + len(top) * row_h + pad
        px, py = fw - col_w - 8, 8

        _panel(frame, px, py, col_w, panel_h)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Column header
        cv2.putText(frame, "LIVE RATINGS", (px + 8, py + 16),
                    font, 0.42, (255, 215, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{'PLAYER':<18} {'RTG':>4} {'SPD':>5} {'P':>3} {'S':>3} {'T':>3}",
                    (px + 6, py + 34), font, 0.33, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.line(frame, (px + 4, py + header_h - 2), (px + col_w - 4, py + header_h - 2),
                 (70, 70, 70), 1)

        for i, p in enumerate(top):
            ry = py + header_h + i * row_h + row_h - 4
            color = _team_color(p.team)
            rc = _rating_color(p.rating())

            # Team dot
            cv2.circle(frame, (px + 8, ry - 6), 4, color, -1)

            # Player label
            if p.player_name:
                parts = p.player_name.split()
                display = parts[-1] if len(parts) > 1 else p.player_name
                if p.jersey_number:
                    display = f"#{p.jersey_number} {display}"
            else:
                display = f"#{p.tracker_id}"
            if len(display) > 17:
                display = display[:16] + "."

            rating_str = f"{p.rating():4.0f}"
            spd_str    = f"{p.get_average_speed():4.1f}"
            passes_str = f"{p.passes:3d}"
            shots_str  = f"{p.shots:3d}"
            tck_str    = f"{p.tackles:3d}"

            row = f"{display:<18} {' ':>4} {spd_str:>5} {passes_str:>3} {shots_str:>3} {tck_str:>3}"
            cv2.putText(frame, row, (px + 16, ry),
                        font, 0.33, (220, 220, 220), 1, cv2.LINE_AA)

            # Rating in its own color (overlaid on the gap)
            cv2.putText(frame, rating_str, (px + 155, ry),
                        font, 0.35, rc, 1, cv2.LINE_AA)

    # ── Possession bar ────────────────────────────────────────────────────────
    def draw_possession_bar(
        self,
        frame: np.ndarray,
        possession_pct: dict,
    ) -> np.ndarray:
        """Draw a possession bar at the top-centre of the frame.

        possession_pct: dict mapping team name -> percentage (0-100).
        """
        if not possession_pct:
            return frame

        fh, fw = frame.shape[:2]
        bar_w, bar_h = 300, 20
        bx = (fw - bar_w) // 2
        by = 8

        # Background
        _panel(frame, bx - 4, by - 4, bar_w + 8, bar_h + 22, alpha=0.65)

        teams = sorted(possession_pct.items())  # deterministic order
        colors = [_team_color(t) for t, _ in teams]

        total_pct = sum(p for _, p in teams)
        if total_pct <= 0:
            return frame

        # Draw split bar
        cursor = bx
        for idx, (team, pct) in enumerate(teams):
            seg_w = int(pct / total_pct * bar_w)
            if idx == len(teams) - 1:
                seg_w = bar_w - (cursor - bx)  # fill remainder to avoid rounding gap
            color = colors[idx]
            cv2.rectangle(frame, (cursor, by), (cursor + seg_w, by + bar_h), color, -1)
            cursor += seg_w

        # Border
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (80, 80, 80), 1)

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cursor = bx
        for idx, (team, pct) in enumerate(teams):
            seg_w = int(pct / total_pct * bar_w)
            label = f"{team} {pct:.0f}%"
            (lw, lh), _ = cv2.getTextSize(label, font, 0.38, 1)
            lx = cursor + max(2, (seg_w - lw) // 2)
            ly = by + bar_h + 12
            cv2.putText(frame, label, (lx, ly), font, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
            cursor += seg_w

        return frame

    # ── Minimap ───────────────────────────────────────────────────────────────
    def draw_minimap(
        self,
        frame: np.ndarray,
        active_players,
        ball_pos,
        homography,
        frame_w: int,
        frame_h: int,
        map_w: int = MINIMAP_W,
        map_h: int = MINIMAP_H,
    ) -> np.ndarray:
        """Draw a minimap in the bottom-right corner of the frame."""
        fh, fw = frame.shape[:2]
        margin = 8
        ox = fw - map_w - margin
        oy = fh - map_h - margin - 22  # offset for status bar

        # Draw semi-transparent background (green pitch)
        overlay_mm = frame.copy()
        cv2.rectangle(overlay_mm, (ox, oy), (ox + map_w, oy + map_h), (34, 139, 34), -1)
        cv2.addWeighted(overlay_mm, 0.75, frame, 0.25, 0, frame)

        # Pitch markings (white)
        # Halfway line
        mid_x = ox + map_w // 2
        cv2.line(frame, (mid_x, oy), (mid_x, oy + map_h), (255, 255, 255), 1)
        # Centre circle (approximate)
        cr = int(map_h * 0.18)
        cv2.circle(frame, (ox + map_w // 2, oy + map_h // 2), cr, (255, 255, 255), 1)
        # Penalty areas
        pa_w = int(map_w * 0.12)
        pa_h = int(map_h * 0.42)
        pa_top = oy + (map_h - pa_h) // 2
        cv2.rectangle(frame, (ox, pa_top), (ox + pa_w, pa_top + pa_h), (255, 255, 255), 1)
        cv2.rectangle(
            frame,
            (ox + map_w - pa_w, pa_top),
            (ox + map_w, pa_top + pa_h),
            (255, 255, 255), 1,
        )
        # Outer border
        cv2.rectangle(frame, (ox, oy), (ox + map_w, oy + map_h), (200, 200, 200), 1)

        # Player dots
        for player in active_players:
            if not player.positions:
                continue
            px_coord, py_coord = player.positions[-1]
            mmx, mmy = homography.to_minimap(
                px_coord, py_coord, map_w, map_h, frame_w, frame_h
            )
            dot_x = ox + int(np.clip(mmx, 0, map_w - 1))
            dot_y = oy + int(np.clip(mmy, 0, map_h - 1))
            color = _team_color(player.team)
            cv2.circle(frame, (dot_x, dot_y), 5, color, -1)
            cv2.circle(frame, (dot_x, dot_y), 5, (0, 0, 0), 1)

        # Ball dot
        if ball_pos is not None:
            bpx, bpy = ball_pos
            mmx, mmy = homography.to_minimap(bpx, bpy, map_w, map_h, frame_w, frame_h)
            dot_x = ox + int(np.clip(mmx, 0, map_w - 1))
            dot_y = oy + int(np.clip(mmy, 0, map_h - 1))
            cv2.circle(frame, (dot_x, dot_y), 4, (255, 255, 255), -1)
            cv2.circle(frame, (dot_x, dot_y), 4, (0, 0, 0), 1)

        return frame

    # ── Pass network ──────────────────────────────────────────────────────────
    def draw_pass_network(
        self,
        frame: np.ndarray,
        pass_events: list,
        current_frame: int,
        fade_frames: int = 180,
    ) -> np.ndarray:
        """Draw fading lines for recent pass events.

        Newer passes are more opaque; older ones fade out over fade_frames.
        """
        if not pass_events:
            return frame

        for event in pass_events:
            age = current_frame - event.get("frame", current_frame)
            if age > fade_frames:
                continue
            alpha = 1.0 - (age / fade_frames)
            if alpha <= 0:
                continue

            from_pos = event.get("from_pos")
            to_pos = event.get("to_pos")
            team = event.get("team")

            if from_pos is None or to_pos is None:
                continue

            color = _team_color(team)
            pt1 = (int(from_pos[0]), int(from_pos[1]))
            pt2 = (int(to_pos[0]), int(to_pos[1]))

            # Draw line onto a copy and blend
            line_layer = frame.copy()
            cv2.line(line_layer, pt1, pt2, color, 2, cv2.LINE_AA)
            cv2.addWeighted(line_layer, alpha * 0.6, frame, 1 - alpha * 0.6, 0, frame)

            # Draw endpoint circles
            cv2.circle(frame, pt2, 4, color, -1)

        return frame

    def draw_heatmap_dot(self, frame, player: PlayerTracker) -> np.ndarray:
        color = _team_color(player.team)
        for idx, (px, py) in enumerate(player.positions):
            if idx % 10 != 0:
                continue
            dot = frame.copy()
            cv2.circle(dot, (int(px), int(py)), 5, color, -1)
            cv2.addWeighted(dot, 0.3, frame, 0.7, 0, frame)
        return frame
