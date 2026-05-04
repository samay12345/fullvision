"""
fc26_scraper.py — Scrape player attributes from sofifa.com and cache locally.
"""
from __future__ import annotations

import json
import os
import time
from difflib import SequenceMatcher

import requests
from bs4 import BeautifulSoup

CACHE_FILE = "fc26_cache.json"
ROSTER_FILE = "player_roster.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research bot)"}
BASE_URL = "https://sofifa.com"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_fc26_cache() -> dict:
    """Load fc26_cache.json and return its contents, or {} if missing/corrupt."""
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
    """Fuzzy-match *name* against cache keys.

    Returns the cached dict for the best match (ratio >= 0.75), or None.
    """
    if not name or not cache:
        return None

    name_lower = name.lower().strip()

    # Exact match first
    if name in cache:
        return cache[name]

    best_key = None
    best_ratio = 0.0
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
    """Search sofifa.com for *name* and return the href of the first result row."""
    url = f"{BASE_URL}/players?keyword={requests.utils.quote(name)}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # The player table rows have a link like /player/123456/...
    for a_tag in soup.select("table tbody tr td a[href*='/player/']"):
        href = a_tag.get("href", "")
        if "/player/" in href:
            return href
    return None


def _scrape_player_page(href: str, session: requests.Session) -> dict | None:
    """Scrape the player detail page at *href* and return attribute dict."""
    url = href if href.startswith("http") else BASE_URL + href
    try:
        resp = session.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    def _text(selector: str, default: str = "") -> str:
        el = soup.select_one(selector)
        return el.get_text(strip=True) if el else default

    def _int_attr(label: str) -> int | None:
        """Find a stat value by its label text in the sofifa layout."""
        # sofifa uses <li><span class="bp3-tag ...">VALUE</span><span>LABEL</span></li>
        for li in soup.select("li.bp3-text-overflow-ellipsis"):
            spans = li.find_all("span")
            if len(spans) >= 2:
                label_text = spans[-1].get_text(strip=True).lower()
                if label.lower() in label_text:
                    try:
                        return int(spans[0].get_text(strip=True))
                    except ValueError:
                        pass
        return None

    # Overall rating — try the prominent OVR badge
    overall = None
    for tag in soup.select("em[data-value]"):
        try:
            overall = int(tag["data-value"])
            break
        except (KeyError, ValueError):
            pass
    if overall is None:
        # Fallback: look for first numeric span in the rating cards
        for span in soup.select("div.columns span.label"):
            try:
                overall = int(span.get_text(strip=True))
                break
            except ValueError:
                pass

    # Individual attributes
    pace = _int_attr("pace")
    shooting = _int_attr("shooting") or _int_attr("sho")
    passing = _int_attr("passing") or _int_attr("pas")
    dribbling = _int_attr("dribbling") or _int_attr("dri")
    defending = _int_attr("defending") or _int_attr("def")
    physical = _int_attr("physicality") or _int_attr("physical") or _int_attr("phy")

    # Position
    position = ""
    pos_tag = soup.select_one("span.bp3-tag.bp3-round")
    if pos_tag:
        position = pos_tag.get_text(strip=True)

    # Full name from page title / h1
    full_name = _text("h1") or _text("title").split(" — ")[0].strip()

    if overall is None and pace is None:
        # Couldn't parse anything meaningful
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
    """Scrape FC26 attributes for all players in *names* (or from player_roster.json).

    Already-cached players are skipped.  Results are saved to fc26_cache.json.
    Returns the complete cache dict.
    """
    cache = load_fc26_cache()

    if names is None:
        # Load from roster file
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

        if name in cache:
            # Already cached — skip
            continue

        print(f"Scraping {name}...", end=" ", flush=True)

        href = _search_player(name, session)
        if href is None:
            print("not found")
            cache[name] = None
            _save_cache(cache)
            time.sleep(1)
            continue

        stats = _scrape_player_page(href, session)
        if stats is None:
            print("not found")
            cache[name] = None
        else:
            ovr = stats.get("overall")
            print(f"done (OVR: {ovr})")
            cache[name] = stats

        _save_cache(cache)
        time.sleep(1)

    return cache


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = scrape_players()
    found = sum(1 for v in result.values() if v is not None)
    print(f"\n[fc26] Done — {found}/{len(result)} players cached.")
