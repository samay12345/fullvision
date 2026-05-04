"""
player_identity.py — OCR jersey numbers and map to player names via teams_db.json.
"""
from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from typing import Optional

import numpy as np

try:
    import easyocr as _easyocr_module
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Teams DB loader
# ---------------------------------------------------------------------------

_DB_FILE = "teams_db.json"
_teams_db: dict = {}


def _load_teams_db() -> dict:
    global _teams_db
    if _teams_db:
        return _teams_db
    if not os.path.exists(_DB_FILE):
        return {}
    try:
        with open(_DB_FILE, encoding="utf-8") as fh:
            _teams_db = json.load(fh).get("teams", {})
    except (json.JSONDecodeError, OSError):
        _teams_db = {}
    return _teams_db


def find_team_in_db(query: str) -> Optional[str]:
    """Fuzzy-match *query* against team names in teams_db.  Returns exact DB key or None."""
    db = _load_teams_db()
    if not db:
        return None
    # Exact match first
    if query in db:
        return query
    q = query.lower().strip()
    # Common short-name mappings
    aliases = {
        "psg": "paris saint-germain",
        "man city": "manchester city",
        "man utd": "manchester united",
        "man united": "manchester united",
        "barca": "fc barcelona",
        "barcelona": "fc barcelona",
        "bayern": "fc bayern münchen",
        "bayern munich": "fc bayern münchen",
        "ajax": "ajax",
        "atletico": "atlético madrid",
        "atletico madrid": "atlético madrid",
        "inter milan": "inter",
        "ac milan": "ac milan",
        "dortmund": "borussia dortmund",
        "bvb": "borussia dortmund",
        "leverkusen": "bayer 04 leverkusen",
        "rb leipzig": "rb leipzig",
        "spurs": "tottenham hotspur",
        "tottenham": "tottenham hotspur",
        "wolves": "wolverhampton wanderers",
        "benfica": "sl benfica",
        "porto": "fc porto",
        "monaco": "as monaco",
        "marseille": "olympique de marseille",
        "lyon": "olympique lyonnais",
        "sporting": "sporting cp",
    }
    if q in aliases:
        canonical = aliases[q]
        for key in db:
            if key.lower() == canonical:
                return key

    best_key, best_ratio = None, 0.0
    for key in db:
        ratio = SequenceMatcher(None, q, key.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_key = key
    if best_ratio >= 0.70 and best_key:
        return best_key
    return None


def get_team_roster(team_db_key: str) -> dict[int, dict]:
    """Return {jersey_number: player_dict} for the given team key."""
    db = _load_teams_db()
    team = db.get(team_db_key, {})
    roster: dict[int, dict] = {}
    for p in team.get("players", []):
        j = p.get("jersey")
        if j is not None:
            roster[int(j)] = p
    return roster


# ---------------------------------------------------------------------------
# PlayerIdentityManager
# ---------------------------------------------------------------------------

class PlayerIdentityManager:
    """
    Identify players by OCR-ing jersey numbers and looking them up in teams_db.

    Usage:
        mgr = PlayerIdentityManager()
        mgr.set_teams("Real Madrid", "FC Bayern München")
        identity = mgr.get_identity(track_id, frame, bbox, "A")
    """

    def __init__(self):
        self._reader = None
        self._id_cache: dict[int, dict] = {}
        self._rosters: dict[str, dict[int, dict]] = {}
        self._legacy_roster: dict = {}
        # OCR throttle: track attempts and last-tried frame per player
        self._ocr_attempts: dict[int, int] = {}
        self._ocr_last_frame: dict[int, int] = {}
        self._OCR_RETRY_FRAMES = 45   # try OCR at most once every 45 frames
        self._OCR_MAX_ATTEMPTS = 6    # give up after 6 failed OCR attempts
        self._load_legacy_roster()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_legacy_roster(self) -> None:
        path = "player_roster.json"
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            for label, players in data.items():
                self._legacy_roster[label] = {
                    int(k): v for k, v in players.items()
                }
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    def set_teams(self, home_query: str, away_query: str) -> tuple[str | None, str | None]:
        """
        Fuzzy-match home/away team names against teams_db and load their rosters.

        Assigns home → team label "A", away → team label "B".
        Returns (resolved_home_name, resolved_away_name).
        """
        home_key = find_team_in_db(home_query) if home_query else None
        away_key = find_team_in_db(away_query) if away_query else None

        if home_key:
            self._rosters["A"] = get_team_roster(home_key)
            print(f"[identity] Home roster: {home_key} ({len(self._rosters['A'])} players)")
        else:
            print(f"[identity] Warning: could not find '{home_query}' in teams_db")

        if away_key:
            self._rosters["B"] = get_team_roster(away_key)
            print(f"[identity] Away roster: {away_key} ({len(self._rosters['B'])} players)")
        else:
            print(f"[identity] Warning: could not find '{away_query}' in teams_db")

        return home_key, away_key

    def set_team_label(self, label: str, team_query: str) -> str | None:
        """Assign a specific label ('A' or 'B') to a team by name."""
        key = find_team_in_db(team_query)
        if key:
            self._rosters[label] = get_team_roster(key)
        return key

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _get_reader(self):
        if self._reader is not None:
            return self._reader
        if not _EASYOCR_AVAILABLE:
            return None
        try:
            self._reader = _easyocr_module.Reader(["en"], gpu=False, verbose=False)
        except Exception:
            self._reader = None
        return self._reader

    def read_jersey_number(self, frame: np.ndarray, bbox: tuple) -> Optional[int]:
        reader = self._get_reader()
        if reader is None:
            return None
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        if h < 20 or w < 10:
            return None
        mid_y = y1 + h // 2
        crop = frame[max(0, y1): max(y1 + 1, mid_y), max(0, x1): max(x1 + 1, x2)]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        try:
            results = reader.readtext(crop, allowlist="0123456789")
        except Exception:
            return None
        for (_, text, confidence) in results:
            text = text.strip()
            if not text.isdigit() or confidence < 0.5:
                continue
            num = int(text)
            if 1 <= num <= 99:
                return num
        return None

    # ------------------------------------------------------------------
    # Roster lookup
    # ------------------------------------------------------------------

    def _lookup(self, jersey: int, team_label: str) -> Optional[dict]:
        """Return player dict from teams_db roster, or None."""
        # Try assigned label rosters (teams_db)
        roster = self._rosters.get(team_label)
        if roster:
            p = roster.get(jersey)
            if p:
                return p
        # Try short label like 'A' or 'B' against legacy roster
        legacy = self._legacy_roster.get(f"team_{team_label.lower()}")
        if legacy:
            name = legacy.get(jersey)
            if name:
                return {"full_name": name, "overall": None, "position": None}
        # Last resort: search all teams
        for roster in self._rosters.values():
            p = roster.get(jersey)
            if p:
                return p
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_identity(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: tuple,
        team_label: str,
        frame_num: int = 0,
    ) -> dict:
        cached = self._id_cache.get(track_id)
        if cached and cached.get("jersey_number") is not None:
            return cached

        # Throttle OCR: skip if we've exceeded attempt limit or tried too recently
        attempts = self._ocr_attempts.get(track_id, 0)
        last_tried = self._ocr_last_frame.get(track_id, -9999)
        should_ocr = (
            attempts < self._OCR_MAX_ATTEMPTS
            and (frame_num - last_tried) >= self._OCR_RETRY_FRAMES
        )

        jersey = None
        if should_ocr:
            self._ocr_last_frame[track_id] = frame_num
            jersey = self.read_jersey_number(frame, bbox)
            if jersey is None:
                self._ocr_attempts[track_id] = attempts + 1

        player_dict = None
        if jersey is not None:
            player_dict = self._lookup(jersey, team_label)

        result = {
            "jersey_number": jersey,
            "name": player_dict.get("full_name") if player_dict else None,
            "team": team_label,
            "overall": player_dict.get("overall") if player_dict else None,
            "position": player_dict.get("position") if player_dict else None,
            "pace": player_dict.get("pace") if player_dict else None,
            "shooting": player_dict.get("shooting") if player_dict else None,
            "passing": player_dict.get("passing") if player_dict else None,
            "dribbling": player_dict.get("dribbling") if player_dict else None,
            "defending": player_dict.get("defending") if player_dict else None,
            "physical": player_dict.get("physical") if player_dict else None,
        }

        if jersey is not None:
            self._id_cache[track_id] = result
        else:
            self._id_cache.setdefault(track_id, result)
        return result

    def get_cached_identity(self, track_id: int) -> Optional[dict]:
        return self._id_cache.get(track_id)
