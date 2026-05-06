"""
player_identity.py — Jersey number OCR + roster lookup.

Improvements over the original:
- 3-zone crop strategy (front torso, mid torso, back/lower torso)
- 4× upscaling + CLAHE + Otsu binarization (4 image variants per zone)
- Roster-constrained validation with digit-confusion table (6↔9, 1↔7 …)
- Multi-frame vote accumulation: require 3 consistent reads before locking
- Automatic GK detection via jersey color distinctiveness (no OCR needed)
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Optional

import cv2
import numpy as np

try:
    import easyocr as _easyocr_module
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Digit confusion table  (OCR commonly misreads these pairs)
# ---------------------------------------------------------------------------

_CONFUSION: dict[str, list[str]] = {
    "1": ["7", "4"],
    "4": ["1", "7"],
    "6": ["9", "8"],
    "7": ["1", "4"],
    "8": ["3", "6"],
    "9": ["6", "0"],
    "3": ["8"],
    "0": ["8", "9"],
    "5": ["6"],
}


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
    db = _load_teams_db()
    if not db:
        return None
    if query in db:
        return query
    q = query.lower().strip()
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

    Key improvements:
    - Multi-zone crops tried for each detection
    - 4x upscaling + CLAHE + Otsu
    - Roster constraint: only accept numbers present in the loaded roster
    - Confusion-table correction: if OCR reads a number not in roster, try
      visually similar digits (6→9, 1→7, etc.)
    - Vote accumulation: require VOTE_THRESHOLD consistent reads to lock in
    - GK auto-detection by jersey color outlier analysis
    """

    _VOTE_THRESHOLD  = 3     # reads needed to confirm a jersey number
    _OCR_RETRY_FRAMES = 30   # minimum frames between OCR attempts per player
    _OCR_MAX_ATTEMPTS = 10   # give up after N failed OCR frames

    def __init__(self):
        self._reader = None
        self._id_cache: dict[int, dict] = {}
        self._rosters: dict[str, dict[int, dict]] = {}

        # Vote accumulation
        self._jersey_votes:  dict[int, Counter] = {}   # track_id → Counter
        self._ocr_attempts:  dict[int, int]     = {}   # track_id → failed OCR count
        self._ocr_last_frame: dict[int, int]    = {}   # track_id → last OCR frame

        # GK auto-detection bookkeeping
        self._gk_detected: dict[str, bool] = {"A": False, "B": False}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_teams(self, home_query: str, away_query: str) -> tuple[str | None, str | None]:
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
        self._gk_detected = {"A": False, "B": False}
        return home_key, away_key

    def set_team_label(self, label: str, team_query: str) -> str | None:
        key = find_team_in_db(team_query)
        if key:
            self._rosters[label] = get_team_roster(key)
        self._gk_detected[label] = False
        return key

    # ------------------------------------------------------------------
    # OCR helpers
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

    def _constrain_jersey(self, raw: int, team_label: str) -> Optional[int]:
        """
        Accept jersey if it's in the roster; try visual confusion substitutions
        if not.  If no roster is loaded for this team, accept any 1-99 number.
        """
        roster = self._rosters.get(team_label, {}) if team_label else {}
        if not roster:
            return raw  # no roster loaded — accept as-is

        if raw in roster:
            return raw

        raw_s = str(raw)
        for i, ch in enumerate(raw_s):
            for alt in _CONFUSION.get(ch, []):
                candidate = int(raw_s[:i] + alt + raw_s[i + 1:])
                if 1 <= candidate <= 99 and candidate in roster:
                    return candidate

        return None  # not in roster, no valid substitute

    def read_jersey_number(
        self,
        frame: np.ndarray,
        bbox:  tuple,
        team_label: str = "",
    ) -> Optional[int]:
        """
        Multi-strategy OCR.

        Tries 3 crop zones × 4 image variants = up to 12 attempts per call.
        Returns the highest-confidence roster-valid reading, or None.
        """
        reader = self._get_reader()
        if reader is None:
            return None

        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        if h < 24 or w < 12:
            return None

        pad_x = max(1, int(w * 0.12))

        # (y_start_frac, y_end_frac, x_pad)
        zones = [
            (0.15, 0.50, pad_x),   # front jersey (upper torso)
            (0.32, 0.68, 0),       # mid torso
            (0.48, 0.82, 0),       # back jersey (larger number on back)
        ]

        best: Optional[tuple[float, int]] = None  # (confidence, jersey)

        for fy1f, fy2f, px in zones:
            cy1 = max(0, y1 + int(h * fy1f))
            cy2 = max(cy1 + 1, y1 + int(h * fy2f))
            cx1 = max(0, x1 + px)
            cx2 = max(cx1 + 1, x2 - px)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
                continue

            # Upscale to at least 96 px tall for OCR accuracy
            target_h = 96
            scale = max(1.0, target_h / crop.shape[0])
            new_w  = max(1, int(crop.shape[1] * scale))
            crop_up = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            gray    = cv2.cvtColor(crop_up, cv2.COLOR_BGR2GRAY)
            clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)
            _, bw    = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bw_inv   = cv2.bitwise_not(bw)

            for img in [crop_up, enhanced, bw, bw_inv]:
                try:
                    results = reader.readtext(img, allowlist="0123456789", min_size=8)
                except Exception:
                    continue
                for (_, text, conf) in results:
                    text = text.strip()
                    if not text.isdigit() or conf < 0.40:
                        continue
                    num = int(text)
                    if not (1 <= num <= 99):
                        continue
                    validated = self._constrain_jersey(num, team_label)
                    if validated is not None:
                        if best is None or conf > best[0]:
                            best = (conf, validated)

        return best[1] if best else None

    # ------------------------------------------------------------------
    # Roster lookup
    # ------------------------------------------------------------------

    def _lookup(self, jersey: int, team_label: str) -> Optional[dict]:
        roster = self._rosters.get(team_label)
        if roster:
            p = roster.get(jersey)
            if p:
                return p
        for label, roster in self._rosters.items():
            if label == team_label:
                continue
            p = roster.get(jersey)
            if p:
                return p
        return None

    def _build_result(self, jersey: int, player_dict: Optional[dict], team_label: str) -> dict:
        return {
            "jersey_number": jersey,
            "name":      player_dict.get("full_name")  if player_dict else None,
            "team":      team_label,
            "overall":   player_dict.get("overall")    if player_dict else None,
            "position":  player_dict.get("position")   if player_dict else None,
            "pace":      player_dict.get("pace")       if player_dict else None,
            "shooting":  player_dict.get("shooting")   if player_dict else None,
            "passing":   player_dict.get("passing")    if player_dict else None,
            "dribbling": player_dict.get("dribbling")  if player_dict else None,
            "defending": player_dict.get("defending")  if player_dict else None,
            "physical":  player_dict.get("physical")   if player_dict else None,
        }

    def _empty_result(self, team_label: str) -> dict:
        return {k: None for k in (
            "jersey_number", "name", "overall", "position",
            "pace", "shooting", "passing", "dribbling", "defending", "physical",
        )} | {"team": team_label}

    # ------------------------------------------------------------------
    # Public identity API
    # ------------------------------------------------------------------

    def get_identity(
        self,
        track_id:   int,
        frame:      np.ndarray,
        bbox:       tuple,
        team_label: str,
        frame_num:  int = 0,
    ) -> dict:
        # Already confirmed — return immediately
        cached = self._id_cache.get(track_id)
        if cached and cached.get("jersey_number") is not None:
            return cached

        # Check if votes have reached threshold from a previous frame
        votes = self._jersey_votes.get(track_id)
        if votes:
            top, count = votes.most_common(1)[0]
            if count >= self._VOTE_THRESHOLD:
                player_dict = self._lookup(top, team_label)
                result = self._build_result(top, player_dict, team_label)
                self._id_cache[track_id] = result
                print(f"[identity] Locked #{top} for track {track_id} "
                      f"({result.get('name','?')}) after {count} votes")
                return result

        # Throttle: too many failures or tried too recently
        attempts   = self._ocr_attempts.get(track_id, 0)
        last_tried = self._ocr_last_frame.get(track_id, -9999)
        if (attempts >= self._OCR_MAX_ATTEMPTS
                or (frame_num - last_tried) < self._OCR_RETRY_FRAMES):
            return self._id_cache.setdefault(track_id, self._empty_result(team_label))

        self._ocr_last_frame[track_id] = frame_num
        jersey = self.read_jersey_number(frame, bbox, team_label)

        if jersey is not None:
            counter = self._jersey_votes.setdefault(track_id, Counter())
            counter[jersey] += 1
            top, count = counter.most_common(1)[0]
            if count >= self._VOTE_THRESHOLD:
                player_dict = self._lookup(top, team_label)
                result = self._build_result(top, player_dict, team_label)
                self._id_cache[track_id] = result
                print(f"[identity] Locked #{top} for track {track_id} "
                      f"({result.get('name','?')}) after {count} votes")
                return result
        else:
            self._ocr_attempts[track_id] = attempts + 1

        return self._id_cache.setdefault(track_id, self._empty_result(team_label))

    def get_cached_identity(self, track_id: int) -> Optional[dict]:
        return self._id_cache.get(track_id)

    # ------------------------------------------------------------------
    # GK auto-detection by jersey color outlier
    # ------------------------------------------------------------------

    def detect_gk_by_color(
        self,
        team_label: str,
        track_crops: list[tuple[int, np.ndarray]],
    ) -> None:
        """
        Find the goalkeeper without OCR by jersey color.

        The GK wears a distinctly different jersey color from teammates.
        We find the color-outlier among the team's tracked players and assign
        them the GK jersey from the loaded lineup.

        track_crops: list of (track_id, full_player_crop_BGR)
        """
        if self._gk_detected.get(team_label):
            return  # already found GK for this team

        roster = self._rosters.get(team_label, {})
        gk_jerseys = [
            j for j, p in roster.items()
            if str(p.get("position", "")).upper() == "GK"
        ]
        if not gk_jerseys:
            return
        gk_jersey = gk_jerseys[0]

        # Check if GK already identified by OCR
        for data in self._id_cache.values():
            if (data.get("jersey_number") == gk_jersey
                    and data.get("team") == team_label):
                self._gk_detected[team_label] = True
                return

        if len(track_crops) < 4:
            return  # need enough players to find an outlier

        # Compute median jersey color per player (upper torso crop)
        colors: dict[int, np.ndarray] = {}
        for tid, crop in track_crops:
            if crop is None or crop.size == 0 or crop.shape[0] < 12:
                continue
            h = crop.shape[0]
            torso = crop[int(h * 0.15): int(h * 0.55), :]
            if torso.size == 0:
                continue
            # Convert to HSV and use hue+saturation for color comparison
            hsv   = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            # Weight: emphasise saturation and hue, downweight value (lighting)
            h_med = float(np.median(hsv[:, :, 0]))
            s_med = float(np.median(hsv[:, :, 1]))
            colors[tid] = np.array([h_med * 1.5, s_med])  # weighted feature

        if len(colors) < 4:
            return

        tids = list(colors.keys())

        # For each player, compute minimum distance to any teammate
        min_dists: dict[int, float] = {}
        for tid in tids:
            c = colors[tid]
            dists = [float(np.linalg.norm(c - colors[o])) for o in tids if o != tid]
            min_dists[tid] = min(dists)

        # The GK is the player with the largest min-distance to nearest teammate
        gk_tid = max(min_dists, key=min_dists.__getitem__)
        gk_score = min_dists[gk_tid]

        # Threshold: GK should be noticeably distinct (empirically ~25 in weighted HSV)
        if gk_score < 20:
            return

        # Already have identity for this track? Don't overwrite a confirmed OCR
        existing = self._id_cache.get(gk_tid)
        if existing and existing.get("jersey_number") is not None:
            self._gk_detected[team_label] = True
            return

        player_dict = self._lookup(gk_jersey, team_label)
        result = self._build_result(gk_jersey, player_dict, team_label)
        self._id_cache[gk_tid] = result
        self._gk_detected[team_label] = True
        print(f"[identity] GK auto-detected (color): "
              f"track {gk_tid} → #{gk_jersey} "
              f"({result.get('name', '?')}) team={team_label} score={gk_score:.1f}")
