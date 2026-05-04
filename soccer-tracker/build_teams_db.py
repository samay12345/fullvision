"""
build_teams_db.py — Scrape the top 50 FC26 clubs and their complete rosters.

Phase 1 (fast, ~2 min): Scrape team pages → jersey numbers, names, OVR, position.
Phase 2 (slow, ~30 min): Scrape individual player pages → PAC/SHO/PAS/DRI/DEF/PHY.

The output is saved to teams_db.json.  Run phase 1 first, then optionally phase 2.

Usage:
    python build_teams_db.py            # phase 1 only (rosters)
    python build_teams_db.py --full     # phase 1 + phase 2 (all stats)
    python build_teams_db.py --resume   # resume a previously interrupted run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

DB_FILE = "teams_db.json"
BASE_URL = "https://sofifa.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
DELAY = 1.2   # seconds between requests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(session: requests.Session, url: str) -> BeautifulSoup | None:
    try:
        r = session.get(url, headers=HEADERS, timeout=14)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except requests.RequestException as exc:
        print(f"  [warn] GET failed: {url} — {exc}")
        return None


def _load_db() -> dict:
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {"teams": {}}


def _save_db(db: dict):
    with open(DB_FILE, "w", encoding="utf-8") as fh:
        json.dump(db, fh, indent=2, ensure_ascii=False)


def _slug_to_name(slug: str) -> str:
    """Convert URL slug like 'kylian-mbappe' to 'Kylian Mbappe'."""
    return " ".join(w.capitalize() for w in slug.split("-"))


def _parse_stat_text(text: str, label: str) -> int | None:
    parts = [p.strip() for p in text.split("|")]
    label_lower = label.lower()
    for i, part in enumerate(parts):
        if label_lower in part.lower():
            for j in range(i - 1, max(i - 5, -1), -1):
                try:
                    return int(parts[j])
                except ValueError:
                    continue
    return None


def _avg(*vals) -> int | None:
    valid = [v for v in vals if v is not None]
    return round(sum(valid) / len(valid)) if valid else None


# ---------------------------------------------------------------------------
# Phase 1: scrape team list and rosters
# ---------------------------------------------------------------------------

def scrape_team_list(session: requests.Session) -> list[dict]:
    """Return list of {name, sofifa_id, href} for the top 50 clubs by OVR."""
    url = f"{BASE_URL}/teams?type=club&col=oa&sort=desc"
    soup = _get(session, url)
    if soup is None:
        return []

    pattern = re.compile(r"^/team/(\d+)/([^/]+)/")
    seen: set[str] = set()
    teams = []

    for a in soup.find_all("a", href=pattern):
        href = a["href"]
        m = pattern.match(href)
        if not m:
            continue
        team_id = m.group(1)
        if team_id in seen:
            continue
        seen.add(team_id)
        name = a.get_text(strip=True)
        if not name:
            continue
        teams.append({"name": name, "sofifa_id": int(team_id), "href": href})
        if len(teams) >= 50:
            break

    return teams


def scrape_team_roster(session: requests.Session, team: dict) -> list[dict]:
    """Scrape a team page and return list of player dicts."""
    url = BASE_URL + team["href"]
    soup = _get(session, url)
    if soup is None:
        return []

    player_pattern = re.compile(r"^/player/(\d+)/([^/]+)/")
    players = []

    for row in soup.select("table tbody tr"):
        a = row.find("a", href=player_pattern)
        if not a:
            continue

        href = a["href"]
        m = player_pattern.match(href)
        if not m:
            continue

        sofifa_id = int(m.group(1))
        name_slug = m.group(2)
        full_name = _slug_to_name(name_slug)

        cells = [c.get_text(strip=True) for c in row.find_all("td")]
        if len(cells) < 5:
            continue

        # cell[3] = OVR, cell[5] = BestPos(Jersey)Contract
        try:
            ovr = int(cells[3])
        except (ValueError, IndexError):
            ovr = 0

        jersey = None
        if len(cells) > 5:
            jm = re.search(r"\((\d+)\)", cells[5])
            if jm:
                jersey = int(jm.group(1))

        # Primary position: strip name from cell[1] leaving position codes
        pos_cell = cells[1] if len(cells) > 1 else ""
        pos_match = re.sub(re.escape(a.get_text(strip=True)), "", pos_cell).strip()
        # Take first 2-3 uppercase letters as position
        pos = re.findall(r"[A-Z]{2,3}", pos_match)
        position = pos[0] if pos else ""

        players.append({
            "sofifa_id": sofifa_id,
            "full_name": full_name,
            "jersey": jersey,
            "overall": ovr,
            "position": position,
            # Phase 2 stats filled in later
            "pace": None,
            "shooting": None,
            "passing": None,
            "dribbling": None,
            "defending": None,
            "physical": None,
            "stats_fetched": False,
        })

    return players


# ---------------------------------------------------------------------------
# Phase 2: fetch individual player stats
# ---------------------------------------------------------------------------

def fetch_player_stats(session: requests.Session, sofifa_id: int) -> dict | None:
    url = f"{BASE_URL}/player/{sofifa_id}/"
    soup = _get(session, url)
    if soup is None:
        return None

    # Resolve redirect to versioned URL if needed
    if soup.title and "404" in soup.title.string:
        return None

    # Overall from JSON-LD
    overall = None
    ld = soup.find("script", type="application/ld+json")
    if ld and ld.string:
        m = re.search(r"overall rating is (\d+)", ld.string)
        if m:
            overall = int(m.group(1))

    text = soup.get_text(separator="|")

    def _stat(label):
        return _parse_stat_text(text, label)

    acc = _stat("Acceleration"); spd = _stat("Sprint speed")
    fin = _stat("Finishing"); spow = _stat("Shot power")
    lshot = _stat("Long shots"); vol = _stat("Volleys")
    spas = _stat("Short passing"); lpas = _stat("Long passing"); vis = _stat("Vision")
    dri = _stat("Dribbling"); ball = _stat("Ball control"); agi = _stat("Agility")
    daw = _stat("Defensive awareness") or _stat("Def. awareness")
    std = _stat("Standing tackle"); sli = _stat("Sliding tackle")
    sta = _stat("Stamina"); str_ = _stat("Strength"); jum = _stat("Jumping")

    return {
        "overall": overall,
        "pace": _avg(acc, spd),
        "shooting": _avg(fin, spow, lshot, vol),
        "passing": _avg(spas, lpas, vis),
        "dribbling": _avg(dri, ball, agi),
        "defending": _avg(daw, std, sli),
        "physical": _avg(sta, str_, jum),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Also fetch PAC/SHO/PAS/DRI/DEF/PHY for every player")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted run")
    args = parser.parse_args()

    db = _load_db() if args.resume else {"teams": {}}
    session = requests.Session()

    # ── Phase 1: team list ───────────────────────────────────────────────────
    print("Phase 1: Fetching top-50 team list…")
    team_list = scrape_team_list(session)
    if not team_list:
        print("ERROR: Could not fetch team list from sofifa.com")
        sys.exit(1)
    print(f"  Found {len(team_list)} teams")
    time.sleep(DELAY)

    # ── Phase 1: rosters ────────────────────────────────────────────────────
    for i, team in enumerate(team_list, 1):
        name = team["name"]
        if args.resume and name in db["teams"]:
            print(f"  [{i:2d}/{len(team_list)}] {name} — skipped (already in DB)")
            continue

        print(f"  [{i:2d}/{len(team_list)}] {name}…", end=" ", flush=True)
        players = scrape_team_roster(session, team)
        print(f"{len(players)} players")

        db["teams"][name] = {
            "sofifa_id": team["sofifa_id"],
            "href": team["href"],
            "players": players,
        }
        _save_db(db)
        time.sleep(DELAY)

    total_players = sum(len(t["players"]) for t in db["teams"].values())
    print(f"\nPhase 1 complete — {len(db['teams'])} teams, {total_players} players")

    # ── Phase 2: individual player stats (optional) ──────────────────────────
    if not args.full:
        print("\nSkipping Phase 2 (run with --full to fetch PAC/SHO/etc per player).")
        print(f"DB saved to {DB_FILE}")
        return

    print("\nPhase 2: Fetching individual player stats (this takes ~30 min)…")
    done = skipped = failed = 0
    for team_name, team_data in db["teams"].items():
        for player in team_data["players"]:
            if player.get("stats_fetched"):
                skipped += 1
                continue
            pid = player["sofifa_id"]
            stats = fetch_player_stats(session, pid)
            if stats:
                player.update(stats)
                player["stats_fetched"] = True
                done += 1
            else:
                failed += 1
            _save_db(db)
            time.sleep(DELAY)
            if (done + failed) % 50 == 0:
                print(f"  Progress: {done} fetched, {failed} failed, {skipped} skipped")

    print(f"\nPhase 2 complete — {done} players updated, {failed} failed, {skipped} skipped")
    print(f"DB saved to {DB_FILE}")


if __name__ == "__main__":
    main()
