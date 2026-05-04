import math
import json
from typing import Optional
from config import SPEED_SCALE_FACTOR

ACTIVE_WINDOW = 90  # frames — player considered active if seen within this window


class PlayerTracker:
    def __init__(self, tracker_id: int):
        self.tracker_id = tracker_id
        self.team: Optional[str] = None
        self.positions = []
        self.speeds = []
        self.total_distance = 0.0
        self.passes = 0
        self.shots = 0
        self.tackles = 0
        self.last_ball_contact_frame = -1
        self.jersey_colors = []
        self.jersey_number: Optional[int] = None
        self.player_name: Optional[str] = None
        self.position: Optional[str] = None
        self.last_seen_frame: int = 0

    def update_position(self, x_center: float, y_center: float, frame_num: int):
        self.last_seen_frame = frame_num
        if self.positions:
            prev_x, prev_y = self.positions[-1]
            dist = math.hypot(x_center - prev_x, y_center - prev_y)
            self.total_distance += dist
            # Keep only recent speeds for a rolling average
            self.speeds.append(dist * SPEED_SCALE_FACTOR)
            if len(self.speeds) > 300:
                self.speeds = self.speeds[-150:]
        self.positions.append((x_center, y_center))

    def get_average_speed(self) -> float:
        recent = self.speeds[-60:] if self.speeds else []
        return sum(recent) / len(recent) if recent else 0.0

    def get_top_speed(self) -> float:
        return max(self.speeds) if self.speeds else 0.0

    def is_active(self, current_frame: int) -> bool:
        return (current_frame - self.last_seen_frame) <= ACTIVE_WINDOW

    def rating(self) -> float:
        """0-99 live rating. Weighted by involvement, distance, and speed."""
        if len(self.positions) < 15:
            return 0.0
        # Distance covered (normalized — ~500px = decent run)
        dist_score = min(self.total_distance / 500.0, 1.0) * 25
        # Rolling speed
        spd_score = min(self.get_average_speed() / 2.5, 1.0) * 25
        # Event involvement
        pas_score = min(self.passes * 6, 30)
        sht_score = min(self.shots * 12, 15)
        tck_score = min(self.tackles * 6, 5)
        return round(dist_score + spd_score + pas_score + sht_score + tck_score, 1)

    def to_dict(self) -> dict:
        return {
            "tracker_id": self.tracker_id,
            "team": self.team,
            "jersey_number": self.jersey_number,
            "name": self.player_name,
            "position": self.position,
            "total_distance_px": round(self.total_distance, 2),
            "avg_speed_mps": round(self.get_average_speed(), 4),
            "top_speed_mps": round(self.get_top_speed(), 4),
            "passes": self.passes,
            "shots": self.shots,
            "tackles": self.tackles,
            "rating": self.rating(),
            "frames_tracked": len(self.positions),
        }


class PlayerRegistry:
    def __init__(self):
        self.players: dict[int, PlayerTracker] = {}

    def get_or_create(self, tracker_id: int) -> PlayerTracker:
        if tracker_id not in self.players:
            self.players[tracker_id] = PlayerTracker(tracker_id)
        return self.players[tracker_id]

    def all_players(self) -> list[PlayerTracker]:
        return list(self.players.values())

    def active_players(self, current_frame: int) -> list[PlayerTracker]:
        return [p for p in self.players.values() if p.is_active(current_frame)]

    def export_stats(self, output_path: str = "stats_output.json"):
        data = [p.to_dict() for p in self.all_players()]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


def merge_player(registry: "PlayerRegistry", old_id: int, new_id: int):
    """Merge old_id's stats into new_id and remove old_id from the registry.

    Fields merged: passes, shots, tackles, total_distance, speeds, positions,
    player_name, team, jersey_number, position.
    """
    if old_id not in registry.players or new_id not in registry.players:
        return

    old_p = registry.players[old_id]
    new_p = registry.players[new_id]

    # Accumulate numeric stats
    new_p.passes += old_p.passes
    new_p.shots += old_p.shots
    new_p.tackles += old_p.tackles
    new_p.total_distance += old_p.total_distance

    # Prepend old positions/speeds so timeline is preserved
    new_p.positions = old_p.positions + new_p.positions
    new_p.speeds = old_p.speeds + new_p.speeds
    if len(new_p.speeds) > 300:
        new_p.speeds = new_p.speeds[-300:]

    # Carry over identity info if new player doesn't have it yet
    if new_p.player_name is None and old_p.player_name is not None:
        new_p.player_name = old_p.player_name
    if new_p.team is None and old_p.team is not None:
        new_p.team = old_p.team
    if new_p.jersey_number is None and old_p.jersey_number is not None:
        new_p.jersey_number = old_p.jersey_number
    if new_p.position is None and old_p.position is not None:
        new_p.position = old_p.position

    del registry.players[old_id]
