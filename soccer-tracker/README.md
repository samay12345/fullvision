# Football Live Tracker

Real-time football tracking system that analyses YouTube match streams and
displays player stats, live ratings, and annotated video in a browser.

## Setup

```bash
pip install fastapi uvicorn[standard] ultralytics opencv-python easyocr yt-dlp requests beautifulsoup4
```

## First run (one time only)

```bash
# Scrape rosters for the top 50 FC26 clubs (~2 min)
python build_teams_db.py

# Scrape FC26 individual player stats (optional, ~30 min)
python build_teams_db.py --full

# Scrape FC26 player attributes cache from sofifa.com
python fc26_scraper.py
```

## Start

```bash
python server.py
```

## Open

```
http://localhost:8000
```

## Usage

1. Paste a YouTube football match URL into the URL bar
2. Click **Analyze**
3. Watch real-time tracking with player names and live ratings in the browser

## Architecture

| File | Role |
|------|------|
| `server.py` | FastAPI server, WebSocket broadcast, pipeline lifecycle |
| `stream_capture.py` | yt-dlp + OpenCV frame capture in background thread |
| `analyzer.py` | YOLO detection, tracking, team classification, overlay drawing |
| `rating_engine.py` | Live 1–10 player ratings updated by game events |
| `fc26_loader.py` | FC26 stats lookup from local cache files |
| `frontend/index.html` | Single-page app — canvas video + ratings sidebar |

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Frontend HTML |
| POST | `/analyze` | `{"url": "..."}` — start pipeline |
| GET | `/status` | Pipeline state |
| POST | `/stop` | Stop pipeline |
| WS | `/ws/feed` | JSON frame stream at ~10 fps |
