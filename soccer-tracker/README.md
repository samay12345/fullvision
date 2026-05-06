# FullVision — Real-Time Soccer Intelligence

A real-time soccer match analysis system that streams any YouTube match, tracks every player on the pitch using computer vision, and delivers live ratings, speed/distance stats, and AI-generated post-match summaries — all in a browser.

![Architecture](https://img.shields.io/badge/YOLOv8m-Object%20Detection-blue) ![Tracking](https://img.shields.io/badge/BoT--SORT-Multi--Object%20Tracking-green) ![AI](https://img.shields.io/badge/Claude%20Vision-Lineup%20%26%20Summary-purple)

---

## What It Does

- **Streams any YouTube match URL** and decodes it at full 1080p via ffmpeg (no quality loss)
- **Detects and tracks all players** on screen using YOLOv8m + BoT-SORT with appearance-based re-association
- **Names players automatically** by OCR-ing jersey numbers against a 1,872-player FC26 database (50 top clubs)
- **Identifies goalkeepers without OCR** — automatically finds the color-outlier in each team's jersey distribution
- **Calculates live 1–10 ratings** updated in real time by detected events: passes, tackles, shots, sprints, dribbles, key passes, and more
- **Extracts lineups from TV screenshots** using Claude Vision — upload a photo of the broadcast lineup card and Claude reads every jersey number, name, and formation
- **Generates post-match player summaries** via Claude — every player on the roster gets a written performance review; players who didn't appear are marked DNP
- **Streams to the browser over WebSocket** at ~10 fps with annotated video + a live ratings sidebar

---

## Architecture

```
YouTube URL
    │
    ▼
yt-dlp  ──► ffmpeg (1080p DASH decode, raw BGR pipe)
                │
                ▼
         StreamCapture thread
                │  frame queue
                ▼
            Analyzer
         ┌──────────────────────────────────────────────┐
         │  YOLOv8m detection (every 3 frames)          │
         │  BoT-SORT tracking (appearance + Kalman)     │
         │  K-means team classification (A / B)         │
         │  Auto-align clusters to known kit colors     │
         │  GK detection by jersey color outlier        │
         │  Multi-zone OCR + roster-constrained read    │
         │  EventDetector → RatingEngine (1–10 scale)   │
         │  JPEG encode (quality 95) → base64           │
         └──────────────────────────────────────────────┘
                │  asyncio.Queue
                ▼
         WebSocket broadcast ──► Browser (canvas + sidebar)
```

### Key Files

| File | Role |
|------|------|
| `server.py` | FastAPI server — REST endpoints + WebSocket broadcast + pipeline lifecycle |
| `stream_capture.py` | yt-dlp + ffmpeg frame capture; pipes raw BGR at native FPS |
| `analyzer.py` | Core loop: YOLO → tracking → teams → OCR → events → overlay → WS push |
| `player_identity.py` | Multi-zone OCR, roster-constraint + confusion table, GK color detection |
| `rating_engine.py` | Live 1–10 ratings; `EventDetector` fires on passes/tackles/shots/sprints |
| `team_classifier.py` | K-means jersey color clustering with kit-color auto-alignment |
| `reid.py` | HSV histogram Re-ID — re-links tracks after occlusion |
| `lineup_analyzer.py` | Claude Vision → structured lineup JSON; Claude → match summaries |
| `fc26_loader.py` | Looks up FC26 OVR / pace / shooting / etc. by player name |
| `teams_db.json` | 50-club, 1,872-player database (jersey numbers, names, positions, stats) |
| `frontend/index.html` | Single-page app — 3-tab sidebar (Live / Lineup / Summary) + annotated canvas |

---

## How Player Identity Works

Getting names on players from a single broadcast camera is hard. FullVision uses a layered approach:

1. **Lineup card upload** (most reliable) — take a screenshot of the broadcast lineup graphic and upload it. Claude Vision extracts every jersey number, name, and position. These are injected directly into the tracker.

2. **Automatic GK detection** (no OCR) — every 90 frames the system extracts the HSV color signature from each player's torso. The goalkeeper always wears a distinct jersey color. The color-outlier in each team is automatically assigned the GK jersey number from the lineup.

3. **Roster-constrained OCR** (fallback) — three crop zones per player (front torso, mid torso, back number), each processed at 4× upscaled resolution with CLAHE + Otsu binarization. Any read not present in the loaded roster is rejected; visually-ambiguous digits (6↔9, 1↔7, 3↔8) are tried as alternatives before giving up. Three consistent reads of the same number are required before locking a jersey to a track.

---

## Live Ratings

Ratings start at 6.5 and update instantly on detected events:

| Event | Δ Rating |
|-------|----------|
| Shot on target | +0.28 |
| Key pass | +0.22 |
| Tackle won | +0.18 |
| Dribble success | +0.12 |
| Defensive intercept | +0.12 |
| Progressive carry | +0.08 |
| Sprint burst | +0.03 |
| Misplaced pass | −0.08 |
| Tackle lost | −0.08 |
| Losing possession | −0.10 |
| Dribble fail | −0.10 |
| Standing still | −0.03 |

Ratings decay gently toward 6.5 every 5 seconds to reflect current form rather than cumulative stats.

---

## Setup

### Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org) (`brew install ffmpeg` on macOS)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- An [Anthropic API key](https://console.anthropic.com) (for lineup card analysis and match summaries — optional)

### Install

```bash
git clone https://github.com/samay12345/fullvision.git
cd fullvision/soccer-tracker
pip install -r requirements.txt
brew install ffmpeg        # macOS — skip if already installed
```

### Build the player database (one time)

```bash
python build_teams_db.py          # scrape 50-club roster DB (~2 min)
python fc26_scraper.py            # scrape FC26 stats cache from sofifa.com (~30 min, optional)
```

This creates `teams_db.json` and `fc26_cache.json` locally. They are not committed to the repo.

### Run

```bash
python server.py
```

Open **http://localhost:8000** in a browser.

---

## Usage

### Basic tracking

1. Paste a YouTube match URL into the URL bar at the bottom
2. Enter the home and away team names (e.g. `Bayern` and `PSG`)
3. Click **▶ Analyze**

The video starts in the canvas; detected players get annotated bounding boxes with their name and live rating. The **Live** sidebar tab shows all players sorted by rating.

### Lineup cards (recommended)

1. Pause the match broadcast when the lineup card graphic appears
2. Take a screenshot
3. Switch to the **Lineup** tab and upload the screenshot for Home / Away
4. Claude Vision extracts every jersey number, name, and formation instantly

Once lineups are loaded, the tracker maps team colors to known jersey numbers without relying on OCR.

### Match summary

1. After watching some of the match, click the **Summary** tab
2. Click **Generate Match Summary**
3. Claude writes a 2–3 sentence performance review for every player in both rosters, with individual ratings, strengths, and one area to improve. Players who didn't appear are marked **DNP**

### API key

Click the 🔑 icon in the status bar to enter your Anthropic API key. It is sent only to your local server and never stored in the browser.

---

## REST API

| Method | Path | Body / Response |
|--------|------|-----------------|
| `GET` | `/` | Frontend HTML |
| `POST` | `/analyze` | `{"url":"…","home_team":"…","away_team":"…"}` |
| `GET` | `/status` | `{"state":"running","frame_count":…,"player_count":…}` |
| `POST` | `/stop` | Stop the pipeline |
| `POST` | `/set_api_key` | `{"api_key":"sk-ant-…"}` |
| `POST` | `/lineup/{home\|away}` | Multipart image upload → lineup JSON |
| `GET` | `/lineup` | Current lineups for both teams |
| `GET` | `/summary` | Claude-generated match summary |
| `WS` | `/ws/feed` | JSON frames: `{frame_id, players[], ball, frame_b64}` |

### WebSocket frame schema

```json
{
  "frame_id": 1042,
  "timestamp_s": 34.7,
  "players": [
    {
      "track_id": 7,
      "name": "Mbappe",
      "jersey": 9,
      "team": "A",
      "bbox": [x, y, w, h],
      "speed_ms": 8.3,
      "top_speed_ms": 9.1,
      "distance_m": 412.0,
      "sprints": 14,
      "rating": 7.8,
      "rating_delta": 0.04,
      "last_event": "sprint_burst"
    }
  ],
  "ball": {"x": 640, "y": 380},
  "frame_b64": "..."
}
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection | YOLOv8m (Ultralytics) |
| Tracking | BoT-SORT (appearance + Kalman filter) |
| OCR | EasyOCR with multi-zone preprocessing |
| Re-ID | HSV histogram Bhattacharyya correlation |
| Team classification | K-means + kit-color alignment |
| Stream decoding | yt-dlp + ffmpeg (1080p DASH) |
| AI lineup extraction | Claude claude-opus-4-7 Vision |
| AI match summaries | Claude claude-opus-4-7 |
| Backend | FastAPI + uvicorn |
| Frontend | Vanilla JS + Canvas API (no framework) |
| Player database | FC26 scrape — 50 clubs, 1,872 players |
