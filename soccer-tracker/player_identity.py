"""
player_identity.py — OCR jersey numbers using easyocr and map to player names
via a roster JSON file.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

# easyocr is optional (2 GB+ download).  Import lazily and fall back gracefully.
try:
    import easyocr as _easyocr_module
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False

from fc26_scraper import load_fc26_cache, get_player_stats


class PlayerIdentityManager:
    """Manage per-player identity via OCR jersey-number detection.

    Roster JSON format:
        {
          "team_a": {"10": "Neymar", "7": "Kylian Mbappe", ...},
          "team_b": {"9": "Robert Lewandowski", ...}
        }
    """

    def __init__(
        self,
        roster_path: str = "player_roster.json",
        fc26_cache_path: str = "fc26_cache.json",
    ):
        self._roster: dict[str, dict[str, str]] = {}  # team_label -> {jersey_str -> name}
        self._fc26_cache: dict = {}
        self._reader = None  # easyocr.Reader — initialised lazily

        # Identity cache keyed by track_id
        self._id_cache: dict[int, dict] = {}

        self._load_roster(roster_path)
        self._load_fc26(fc26_cache_path)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_roster(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                self._roster = json.load(fh)
        except (json.JSONDecodeError, OSError):
            self._roster = {}

    def _load_fc26(self, path: str) -> None:
        # Use the helper from fc26_scraper; it already handles missing files.
        self._fc26_cache = load_fc26_cache()

    def _get_reader(self):
        """Lazily initialise easyocr.Reader on first use."""
        if self._reader is not None:
            return self._reader
        if not _EASYOCR_AVAILABLE:
            return None
        try:
            self._reader = _easyocr_module.Reader(["en"], gpu=False, verbose=False)
        except Exception:
            self._reader = None
        return self._reader

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def read_jersey_number(
        self,
        frame: np.ndarray,
        bbox: tuple,
    ) -> Optional[int]:
        """OCR the jersey number from the upper half of *bbox*.

        Returns an integer (1-99) if detected with confidence > 0.5,
        otherwise None.
        """
        reader = self._get_reader()
        if reader is None:
            return None

        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        w = x2 - x1

        if h < 20 or w < 10:
            return None

        # Crop upper half of the bounding box (shirt area, not shorts)
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
            if not text.isdigit():
                continue
            if confidence < 0.5:
                continue
            num = int(text)
            if 1 <= num <= 99:
                return num

        return None

    # ------------------------------------------------------------------
    # Roster lookup
    # ------------------------------------------------------------------

    def _lookup_roster(
        self, jersey_number: int, team_label: str
    ) -> Optional[str]:
        """Return the player name for *jersey_number* in *team_label*'s roster."""
        # Try exact team label first, then case-insensitive search
        for key, players in self._roster.items():
            if key.lower() == team_label.lower():
                return players.get(str(jersey_number))

        # Fallback: search across all teams
        for players in self._roster.values():
            name = players.get(str(jersey_number))
            if name:
                return name
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
    ) -> dict:
        """Return identity dict for *track_id*, using cache or performing OCR.

        Return structure:
            {
              "jersey_number": int | None,
              "name": str | None,
              "team": str,
              "fc26_stats": dict | None,
            }
        """
        cached = self._id_cache.get(track_id)
        if cached is not None and cached.get("jersey_number") is not None:
            return cached

        # Attempt OCR
        jersey_number = self.read_jersey_number(frame, bbox)

        name: Optional[str] = None
        fc26_stats: Optional[dict] = None

        if jersey_number is not None:
            name = self._lookup_roster(jersey_number, team_label)
            if name:
                fc26_stats = get_player_stats(name, self._fc26_cache)

        result: dict = {
            "jersey_number": jersey_number,
            "name": name,
            "team": team_label,
            "fc26_stats": fc26_stats,
        }

        # Only cache if we found something; otherwise retry next frame
        if jersey_number is not None:
            self._id_cache[track_id] = result
        else:
            # Cache minimal entry so we don't spam OCR on every frame
            # but still allow future retry if jersey still unknown
            self._id_cache.setdefault(track_id, result)

        return result

    def get_cached_identity(self, track_id: int) -> Optional[dict]:
        """Return cached identity without performing OCR.  None if unknown."""
        return self._id_cache.get(track_id)
