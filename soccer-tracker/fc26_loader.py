"""
fc26_loader.py — Load and query FC26 player stats from fc26_cache.json and teams_db.json.

Provides fuzzy name lookup and jersey-number lookup, with permanent attachment
of stats to track IDs once a player has been identified.
"""
from __future__ import annotations

import json
import os
from difflib import SequenceMatcher


class FC26Loader:
    """
    Load FC26 stats from local cache files and expose lookup methods.

    Priority order:
        1. fc26_cache.json (name → full stats dict, scraped from sofifa)
        2. teams_db.json   (team → roster → player with overall + sub-stats)
    """

    def __init__(
        self,
        cache_path: str = "fc26_cache.json",
        teams_db_path: str = "teams_db.json",
    ):
        self._cache_path = cache_path
        self._teams_db_path = teams_db_path

        # name (lowercase) -> stats dict
        self._fc26: dict[str, dict] = {}
        # team_key (str) -> {jersey_int -> stats dict}
        self._rosters: dict[str, dict[int, dict]] = {}
        # track_id -> attached stats dict
        self._attached: dict[int, dict] = {}

        self._load_fc26_cache()
        self._load_teams_db()

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_fc26_cache(self) -> None:
        if not os.path.exists(self._cache_path):
            return
        try:
            with open(self._cache_path, encoding="utf-8") as fh:
                raw: dict = json.load(fh)
            for name, stats in raw.items():
                if stats is not None and isinstance(stats, dict):
                    self._fc26[name.lower().strip()] = stats
        except (json.JSONDecodeError, OSError):
            pass

    def _load_teams_db(self) -> None:
        if not os.path.exists(self._teams_db_path):
            return
        try:
            with open(self._teams_db_path, encoding="utf-8") as fh:
                raw: dict = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return

        teams: dict = raw.get("teams", {})
        for team_key, team_data in teams.items():
            if not isinstance(team_data, dict):
                continue
            roster: dict[int, dict] = {}
            for player in team_data.get("players", []):
                jersey = player.get("jersey")
                if jersey is None:
                    continue
                try:
                    j = int(jersey)
                except (ValueError, TypeError):
                    continue
                stats_dict = self._normalize_player(player)
                roster[j] = stats_dict
                # Also index in fc26 by player name for name lookup
                fname = (player.get("full_name") or "").lower().strip()
                if fname and fname not in self._fc26:
                    self._fc26[fname] = stats_dict
            self._rosters[team_key.lower()] = roster

    def _normalize_player(self, player: dict) -> dict:
        """Standardize a player dict to our stat schema."""
        return {
            "overall":   player.get("overall"),
            "pace":      player.get("pace"),
            "shooting":  player.get("shooting"),
            "passing":   player.get("passing"),
            "dribbling": player.get("dribbling"),
            "defending": player.get("defending"),
            "physical":  player.get("physical"),
            "position":  player.get("position", ""),
            "full_name": player.get("full_name", ""),
        }

    # ------------------------------------------------------------------
    # Lookup methods
    # ------------------------------------------------------------------

    def get_by_name(self, name: str) -> dict | None:
        """
        Fuzzy-match *name* against all known players.

        Returns a stats dict or None.
        """
        if not name or not self._fc26:
            return None
        query = name.lower().strip()
        # Exact match
        if query in self._fc26:
            return self._fc26[query]
        # Fuzzy match
        best_key: str | None = None
        best_ratio = 0.0
        for key in self._fc26:
            ratio = SequenceMatcher(None, query, key).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_key = key
        if best_ratio >= 0.72 and best_key is not None:
            return self._fc26[best_key]
        # Try last name only
        parts = query.split()
        if len(parts) > 1:
            surname = parts[-1]
            best_key2: str | None = None
            best_ratio2 = 0.0
            for key in self._fc26:
                ratio = SequenceMatcher(None, surname, key.split()[-1] if key.split() else key).ratio()
                if ratio > best_ratio2:
                    best_ratio2 = ratio
                    best_key2 = key
            if best_ratio2 >= 0.80 and best_key2 is not None:
                return self._fc26[best_key2]
        return None

    def get_by_jersey(self, team_key: str, jersey: int) -> dict | None:
        """
        Look up a player by team name (fuzzy) + jersey number.

        Returns a stats dict or None.
        """
        # Try exact key first (lowercase)
        target = team_key.lower().strip()
        roster = self._rosters.get(target)
        if roster is None:
            # Fuzzy match on team key
            best_key: str | None = None
            best_ratio = 0.0
            for key in self._rosters:
                ratio = SequenceMatcher(None, target, key).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_key = key
            if best_ratio >= 0.65 and best_key is not None:
                roster = self._rosters[best_key]
        if roster is None:
            return None
        return roster.get(jersey)

    def attach_to_track(self, track_id: int, name: str) -> None:
        """
        Permanently attach FC26 stats to a track_id by name lookup.

        Once attached, get_for_track() will always return the same dict.
        """
        if track_id in self._attached:
            return  # already attached
        stats = self.get_by_name(name)
        if stats is not None:
            self._attached[track_id] = stats

    def attach_by_jersey(self, track_id: int, team_key: str, jersey: int) -> None:
        """
        Permanently attach FC26 stats to a track_id by jersey lookup.
        """
        if track_id in self._attached:
            return
        stats = self.get_by_jersey(team_key, jersey)
        if stats is not None:
            self._attached[track_id] = stats

    def get_for_track(self, track_id: int) -> dict | None:
        """Return the attached FC26 stats for a track_id, or None."""
        return self._attached.get(track_id)

    def get_or_attach(
        self,
        track_id: int,
        name: str | None = None,
        team_key: str | None = None,
        jersey: int | None = None,
    ) -> dict | None:
        """
        Convenience method: return attached stats if available, otherwise
        try to attach by name and/or jersey.
        """
        existing = self._attached.get(track_id)
        if existing is not None:
            return existing

        if name:
            self.attach_to_track(track_id, name)
            if track_id in self._attached:
                return self._attached[track_id]

        if team_key is not None and jersey is not None:
            self.attach_by_jersey(track_id, team_key, jersey)
            if track_id in self._attached:
                return self._attached[track_id]

        return None

    def all_team_keys(self) -> list[str]:
        """Return all known team keys in the roster DB."""
        return list(self._rosters.keys())
