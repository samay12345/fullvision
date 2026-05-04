"""
fc26_scraper.py — Scrape player attributes from sofifa.com and cache locally.
"""
from __future__ import annotations

import json
import os
import re
import time
from difflib import SequenceMatcher

import requests
from bs4 import BeautifulSoup

CACHE_FILE = "fc26_cache.json"
ROSTER_FILE = "player_roster.json"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
BASE_URL = "https://sofifa.com"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_fc26_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2)


# ---------------------------------------------------------------------------
# Fuzzy name match
# ---------------------------------------------------------------------------

def get_player_stats(name: str, cache: dict) -> dict | None:
    if not name or not cache:
        return None
    name_lower = name.lower().strip()
    if name in cache:
        return cache[name]
    best_key, best_ratio = None, 0.0
    for key in cache:
        ratio = SequenceMatcher(None, name_lower, key.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_key = key
    if best_ratio >= 0.75 and best_key is not None:
        return cache[best_key]
    return None


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

def _search_player(name: str, session: requests.Session) -> str | None:
    """Return href of first real player result for *name*."""
    url = f"{BASE_URL}/players?keyword={requests.utils.quote(name)}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=12)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    # Player links: /player/NUMERIC_ID/slug/version/  (exclude /player/random)
    pattern = re.compile(r"^/player/\d+/")
    for a_tag in soup.find_all("a", href=pattern):
        href = a_tag.get("href", "")
        return href   # first real match
    return None


def _parse_stat_text(text: str, stat_name: str) -> int | None:
    """Find a stat value from page text where layout is: VALUE | STAT_NAME."""
    # Text is joined with '|', layout is: ...|VALUE|\n|STAT_NAME|...
    # Try pattern: number immediately before the stat name token
    parts = [p.strip() for p in text.split("|")]
    stat_lower = stat_name.lower()
    for i, part in enumerate(parts):
        if stat_lower in part.lower():
            # Search backwards for a numeric value
            for j in range(i - 1, max(i - 5, -1), -1):
                try:
                    return int(parts[j])
                except ValueError:
                    continue
    return None


def _scrape_player_page(href: str, session: requests.Session) -> dict | None:
    url = href if href.startswith("http") else BASE_URL + href
    try:
        resp = session.get(url, headers=HEADERS, timeout=12)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # --- Overall rating from JSON-LD schema ---
    overall = None
    ld = soup.find("script", type="application/ld+json")
    if ld and ld.string:
        m = re.search(r"overall rating is (\d+)", ld.string)
        if m:
            overall = int(m.group(1))

    # --- Individual attributes from page text ---
    text = soup.get_text(separator="|")

    def _stat(label: str) -> int | None:
        return _parse_stat_text(text, label)

    # Traditional 6 FC attributes (computed from sub-stats)
    acc  = _stat("Acceleration")
    spd  = _stat("Sprint speed")
    fin  = _stat("Finishing")
    spow = _stat("Shot power")
    lshot = _stat("Long shots")
    vol  = _stat("Volleys")
    spas = _stat("Short passing")
    lpas = _stat("Long passing")
    vis  = _stat("Vision")
    dri  = _stat("Dribbling")
    ball = _stat("Ball control")
    agi  = _stat("Agility")
    daw  = _stat("Defensive awareness") or _stat("Def. awareness")
    std  = _stat("Standing tackle")
    sli  = _stat("Sliding tackle")
    sta  = _stat("Stamina")
    str_ = _stat("Strength")
    jum  = _stat("Jumping")

    def _avg(*vals):
        valid = [v for v in vals if v is not None]
        return round(sum(valid) / len(valid)) if valid else None

    pace      = _avg(acc, spd)
    shooting  = _avg(fin, spow, lshot, vol)
    passing   = _avg(spas, lpas, vis)
    dribbling = _avg(dri, ball, agi)
    defending = _avg(daw, std, sli)
    physical  = _avg(sta, str_, jum)

    # Position from page text — look for short position tag near name
    position = ""
    pos_tag = soup.select_one("span.bp3-tag.bp3-round")
    if pos_tag:
        position = pos_tag.get_text(strip=True)
    if not position:
        # fallback: grab jobTitle from LD schema
        if ld and ld.string:
            m = re.search(r'"jobTitle"\s*:\s*"([^"]+)"', ld.string)
            if m:
                position = m.group(1)

    full_name = ""
    if ld and ld.string:
        m = re.search(r'"givenName"\s*:\s*"([^"]+)".*?"familyName"\s*:\s*"([^"]+)"', ld.string, re.S)
        if m:
            full_name = f"{m.group(1)} {m.group(2)}"

    if overall is None:
        return None

    return {
        "overall": overall,
        "pace": pace,
        "shooting": shooting,
        "passing": passing,
        "dribbling": dribbling,
        "defending": defending,
        "physical": physical,
        "position": position,
        "full_name": full_name,
    }


# ---------------------------------------------------------------------------
# Main scrape entry point
# ---------------------------------------------------------------------------

def scrape_players(names: list[str] | None = None) -> dict:
    cache = load_fc26_cache()

    if names is None:
        if os.path.exists(ROSTER_FILE):
            try:
                with open(ROSTER_FILE, "r", encoding="utf-8") as fh:
                    roster = json.load(fh)
                names = []
                for team_players in roster.values():
                    names.extend(team_players.values())
            except (json.JSONDecodeError, OSError):
                names = []
        else:
            names = []

    if not names:
        print("[fc26] No player names to scrape.")
        return cache

    session = requests.Session()

    for name in names:
        name = str(name).strip()
        if not name:
            continue
        if name in cache and cache[name] is not None:
            print(f"Skipping {name} (cached)")
            continue

        print(f"Scraping {name}...", end=" ", flush=True)

        href = _search_player(name, session)
        if href is None:
            print("not found (search)")
            cache[name] = None
            _save_cache(cache)
            time.sleep(1)
            continue

        stats = _scrape_player_page(href, session)
        if stats is None:
            print("not found (parse)")
            cache[name] = None
        else:
            ovr = stats.get("overall")
            print(f"done (OVR: {ovr})")
            cache[name] = stats

        _save_cache(cache)
        time.sleep(1.2)

    return cache


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = scrape_players()
    found = sum(1 for v in result.values() if v is not None)
    print(f"\n[fc26] Done — {found}/{len(result)} players cached.")
