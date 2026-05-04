import math
from config import PASS_DISTANCE_THRESHOLD, SHOT_VELOCITY_THRESHOLD, GOAL_REGIONS
from tracker import PlayerRegistry


class StatEngine:
    def __init__(self, registry: PlayerRegistry):
        self.registry = registry
        self.ball_positions: list[tuple | None] = []
        self.ball_last_owner_id: int | None = None
        self.frame_count: int = 0
        # Possession tracking
        self.possession_frames: dict[str, int] = {}
        # Pass event log: each entry {from_pos, to_pos, team, frame}
        self.pass_events: list[dict] = []

    def update(self, player_detections: list[dict], ball_pos: tuple | None, frame_num: int | None = None):
        self.ball_positions.append(ball_pos)
        current = frame_num if frame_num is not None else self.frame_count

        for det in player_detections:
            tid = det["tracker_id"]
            cx, cy = det["center"]
            player = self.registry.get_or_create(tid)
            player.update_position(cx, cy, current)

        self._check_ball_ownership(player_detections, ball_pos)
        self._check_shots(ball_pos)
        self._check_tackles(player_detections)

        self.frame_count += 1

    def _check_ball_ownership(self, player_detections: list[dict], ball_pos: tuple | None):
        if ball_pos is None or not player_detections:
            return

        bx, by = ball_pos
        closest_id = None
        closest_dist = float("inf")
        closest_det = None

        for det in player_detections:
            cx, cy = det["center"]
            dist = math.hypot(cx - bx, cy - by)
            if dist < closest_dist:
                closest_dist = dist
                closest_id = det["tracker_id"]
                closest_det = det

        if closest_dist < PASS_DISTANCE_THRESHOLD:
            if (
                self.ball_last_owner_id is not None
                and self.ball_last_owner_id != closest_id
            ):
                prev_owner = self.registry.get_or_create(self.ball_last_owner_id)
                new_owner = self.registry.get_or_create(closest_id)
                prev_owner.passes += 1

                # Record pass event if both players have positions
                if prev_owner.positions and new_owner.positions:
                    from_pos = prev_owner.positions[-1]
                    to_pos = closest_det["center"] if closest_det else new_owner.positions[-1]
                    team = prev_owner.team or "unknown"
                    self.pass_events.append({
                        "from_pos": from_pos,
                        "to_pos": to_pos,
                        "team": team,
                        "frame": self.frame_count,
                    })
                    # Keep pass_events list bounded
                    if len(self.pass_events) > 500:
                        self.pass_events = self.pass_events[-500:]

            self.ball_last_owner_id = closest_id
            owner = self.registry.get_or_create(closest_id)
            owner.last_ball_contact_frame = self.frame_count

            # Increment possession counter for the current owner's team
            team = owner.team
            if team and team != "unknown":
                self.possession_frames[team] = self.possession_frames.get(team, 0) + 1

    def get_possession_pct(self) -> dict[str, float]:
        """Return possession percentage per team."""
        total = sum(self.possession_frames.values())
        if total == 0:
            return {}
        return {
            team: round(frames / total * 100.0, 1)
            for team, frames in self.possession_frames.items()
        }

    def _check_shots(self, ball_pos: tuple | None):
        if ball_pos is None or len(self.ball_positions) < 3:
            return

        recent = [p for p in self.ball_positions[-3:] if p is not None]
        if len(recent) < 2:
            return

        total_dist = 0.0
        for i in range(1, len(recent)):
            total_dist += math.hypot(
                recent[i][0] - recent[i - 1][0],
                recent[i][1] - recent[i - 1][1],
            )
        velocity = total_dist / (len(recent) - 1)

        if velocity > SHOT_VELOCITY_THRESHOLD:
            bx, by = ball_pos
            in_goal = any(
                r["x1"] <= bx <= r["x2"] and r["y1"] <= by <= r["y2"]
                for r in GOAL_REGIONS
            )
            if in_goal and self.ball_last_owner_id is not None:
                shooter = self.registry.get_or_create(self.ball_last_owner_id)
                shooter.shots += 1

    def _check_tackles(self, player_detections: list[dict]):
        if not self.ball_positions:
            return
        ball_pos = self.ball_positions[-1]

        for i in range(len(player_detections)):
            for j in range(i + 1, len(player_detections)):
                det_a = player_detections[i]
                det_b = player_detections[j]

                player_a = self.registry.get_or_create(det_a["tracker_id"])
                player_b = self.registry.get_or_create(det_b["tracker_id"])

                if player_a.team == player_b.team:
                    continue
                if player_a.team is None or player_b.team is None:
                    continue

                iou = self._bbox_iou(det_a["bbox"], det_b["bbox"])
                if iou <= 0.1:
                    continue

                if ball_pos is None:
                    continue

                bx, by = ball_pos
                radius = PASS_DISTANCE_THRESHOLD * 1.5
                near_a = math.hypot(det_a["center"][0] - bx, det_a["center"][1] - by) < radius
                near_b = math.hypot(det_b["center"][0] - bx, det_b["center"][1] - by) < radius

                if near_a or near_b:
                    player_a.tackles += 1
                    player_b.tackles += 1

    @staticmethod
    def _bbox_iou(box1: tuple, box2: tuple) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
