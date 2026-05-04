"""
formation.py — Formation detection for a soccer team.
"""
from __future__ import annotations
from tracker import PlayerTracker


def detect_formation(
    players: list[PlayerTracker],
    team: str,
    frame_h: int,
) -> str:
    """Detect the formation string for a team (e.g. '4-3-3').

    Algorithm:
    1. Collect active players for the team that have at least one position.
    2. Remove the player with the most extreme y-position (goalkeeper).
    3. Normalise remaining players' y-positions by frame_h (0=top, 1=bottom).
    4. Divide into three horizontal bands (thirds) and count players per band.
    5. Return the formation string ordered from defence to attack, e.g. '4-3-3'.

    Returns '?' if fewer than 7 eligible players are found.
    """
    team_players = [
        p for p in players
        if p.team == team and p.positions
    ]

    if len(team_players) < 7:
        return "?"

    # Use latest position for each player
    def last_y(p: PlayerTracker) -> float:
        return p.positions[-1][1]

    # Remove the goalkeeper: the player at the most extreme y (closest to either goal)
    # We remove the one with the minimum or maximum y (most extreme from centre)
    max_y = max(team_players, key=last_y)
    min_y = min(team_players, key=last_y)
    mid = frame_h / 2.0
    if abs(last_y(max_y) - mid) >= abs(last_y(min_y) - mid):
        gk = max_y
    else:
        gk = min_y

    outfield = [p for p in team_players if p is not gk]

    if not outfield:
        return "?"

    # Normalise y
    norm_ys = [last_y(p) / max(frame_h, 1) for p in outfield]

    # Split into thirds: 0.0-0.333, 0.333-0.667, 0.667-1.0
    thirds = [0, 0, 0]
    for ny in norm_ys:
        if ny < 1 / 3:
            thirds[0] += 1
        elif ny < 2 / 3:
            thirds[1] += 1
        else:
            thirds[2] += 1

    # Remove empty thirds from formation string
    formation_parts = [str(t) for t in thirds if t > 0]
    if not formation_parts:
        return "?"

    return "-".join(formation_parts)


def get_team_shape(
    players: list[PlayerTracker],
    team: str,
) -> dict:
    """Return a dict with formation_str, avg_width, avg_depth for the team."""
    team_players = [p for p in players if p.team == team and p.positions]

    formation_str = "?"
    avg_width = 0.0
    avg_depth = 0.0

    if team_players:
        xs = [p.positions[-1][0] for p in team_players]
        ys = [p.positions[-1][1] for p in team_players]
        avg_width = float(max(xs) - min(xs)) if len(xs) > 1 else 0.0
        avg_depth = float(max(ys) - min(ys)) if len(ys) > 1 else 0.0

        # Determine frame_h estimate from player y-positions
        frame_h_est = max(ys) * 1.2 if ys else 720
        formation_str = detect_formation(players, team, int(frame_h_est))

    return {
        "formation_str": formation_str,
        "avg_width": round(avg_width, 2),
        "avg_depth": round(avg_depth, 2),
    }
