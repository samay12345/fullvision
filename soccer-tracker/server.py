"""
server.py — FastAPI + uvicorn server for the Football Live Tracker.

Endpoints:
    GET  /           → serve frontend/index.html
    POST /analyze    → start analysis pipeline
    GET  /status     → pipeline status
    POST /stop       → stop pipeline
    WS   /ws/feed    → broadcast JSON frames at ~10fps
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from stream_capture import StreamCapture
from analyzer import Analyzer
from lineup_analyzer import analyze_lineup_image, generate_match_summary


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Football Live Tracker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

pipeline_state: dict = {
    "state": "idle",         # "idle" | "running" | "error"
    "stream_id": None,
    "stream_capture": None,
    "analyzer": None,
    "frame_count": 0,
    "player_count": 0,
    "error": None,
    "_frame_queue": None,
    "_analyzer_thread": None,
    "_capture_thread_ref": None,
}

# Lineups extracted by Claude Vision — shared with the analyzer
_lineups: dict[str, dict] = {}   # "home" | "away" -> lineup dict
_api_key: str = ""

# asyncio.Queue for WS broadcast (maxsize=3)
_broadcast_queue: asyncio.Queue | None = None
_broadcast_loop: asyncio.AbstractEventLoop | None = None
_ws_clients: list[WebSocket] = []
_ws_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# WebSocket broadcast task
# ---------------------------------------------------------------------------

async def _ws_broadcaster():
    """Continuously dequeue payloads and broadcast to all WS clients."""
    global _broadcast_queue
    while True:
        if _broadcast_queue is None:
            await asyncio.sleep(0.05)
            continue
        try:
            payload = await asyncio.wait_for(_broadcast_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue

        # Update frame/player count in pipeline_state
        pipeline_state["frame_count"] = payload.get("frame_id", 0)
        pipeline_state["player_count"] = len(payload.get("players", []))

        message = json.dumps(payload)
        disconnected: list[WebSocket] = []
        async with _ws_lock:
            clients = list(_ws_clients)
        for ws in clients:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        if disconnected:
            async with _ws_lock:
                for ws in disconnected:
                    if ws in _ws_clients:
                        _ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup():
    global _broadcast_queue, _broadcast_loop
    _broadcast_loop = asyncio.get_event_loop()
    _broadcast_queue = asyncio.Queue(maxsize=3)
    asyncio.create_task(_ws_broadcaster())


@app.on_event("shutdown")
async def _shutdown():
    await _stop_pipeline()


# ---------------------------------------------------------------------------
# Pipeline management
# ---------------------------------------------------------------------------

def _stop_pipeline_sync():
    """Stop capture + analyzer synchronously (called from any thread)."""
    sc: StreamCapture | None = pipeline_state.get("stream_capture")
    an: Analyzer | None = pipeline_state.get("analyzer")

    if an is not None:
        an.stop()
    if sc is not None:
        sc.stop()

    at: threading.Thread | None = pipeline_state.get("_analyzer_thread")
    if at is not None and at.is_alive():
        at.join(timeout=3.0)

    pipeline_state["state"] = "idle"
    pipeline_state["stream_id"] = None
    pipeline_state["stream_capture"] = None
    pipeline_state["analyzer"] = None
    pipeline_state["_frame_queue"] = None
    pipeline_state["_analyzer_thread"] = None


async def _stop_pipeline():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _stop_pipeline_sync)


def _inject_lineup_into_analyzer(an: "Analyzer", lineup: dict, label: str) -> None:
    """Push extracted lineup data directly into the analyzer's identity manager."""
    try:
        mgr = an._identity_mgr
        roster: dict[int, dict] = {}
        for p in lineup.get("starters", []):
            j = p.get("jersey")
            name = p.get("name", "")
            if j and name:
                roster[int(j)] = {
                    "full_name": name,
                    "overall": None,
                    "position": p.get("position", ""),
                }
        for p in lineup.get("substitutes", []):
            j = p.get("jersey")
            name = p.get("name", "")
            if j and name:
                roster[int(j)] = {"full_name": name, "overall": None, "position": ""}
        mgr._rosters[label] = roster
        print(f"[lineup] Injected {len(roster)} players into team {label}")
    except Exception as exc:
        print(f"[lineup] Inject error: {exc}")


def _start_pipeline(url: str, stream_id: str, home_team: str = "", away_team: str = ""):
    """
    Build StreamCapture + Analyzer, start both, update pipeline_state.
    Runs in a ThreadPoolExecutor worker called from the /analyze endpoint.
    """
    global _broadcast_queue, _broadcast_loop

    frame_q: queue.Queue = queue.Queue(maxsize=10)

    sc = StreamCapture(url, frame_q)
    sc.start()

    # Wait briefly for stream to open and expose fps
    import time
    for _ in range(30):
        if sc.error:
            pipeline_state["state"] = "error"
            pipeline_state["error"] = sc.error
            sc.stop()
            return
        if sc.fps > 0 and sc.frame_width > 0:
            break
        time.sleep(0.2)

    if sc.error:
        pipeline_state["state"] = "error"
        pipeline_state["error"] = sc.error
        sc.stop()
        return

    an = Analyzer(
        frame_queue=frame_q,
        broadcast_queue=_broadcast_queue,
        fps=sc.fps,
        loop=_broadcast_loop,
        home_team=home_team,
        away_team=away_team,
    )

    # Inject any already-uploaded lineups into the new analyzer
    if _lineups.get("home"):
        _inject_lineup_into_analyzer(an, _lineups["home"], "A")
    if _lineups.get("away"):
        _inject_lineup_into_analyzer(an, _lineups["away"], "B")

    analyzer_thread = threading.Thread(target=an.run, daemon=True, name="Analyzer")
    analyzer_thread.start()

    pipeline_state["state"] = "running"
    pipeline_state["stream_id"] = stream_id
    pipeline_state["stream_capture"] = sc
    pipeline_state["analyzer"] = an
    pipeline_state["_frame_queue"] = frame_q
    pipeline_state["_analyzer_thread"] = analyzer_thread
    pipeline_state["error"] = None

    # Monitor for capture death
    def _watchdog():
        import time as _t
        while True:
            _t.sleep(2.0)
            if pipeline_state["state"] != "running":
                break
            if not sc.is_alive():
                if sc.error:
                    pipeline_state["state"] = "error"
                    pipeline_state["error"] = sc.error
                else:
                    pipeline_state["state"] = "idle"
                break

    threading.Thread(target=_watchdog, daemon=True, name="CaptureWatchdog").start()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    if FRONTEND_PATH.exists():
        return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


class AnalyzeRequest(BaseModel):
    url: str
    home_team: str = ""
    away_team: str = ""


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if not req.url:
        return JSONResponse({"error": "url is required"}, status_code=400)

    # Stop any running pipeline first
    await _stop_pipeline()

    stream_id = uuid.uuid4().hex[:8]
    pipeline_state["state"] = "running"
    pipeline_state["stream_id"] = stream_id
    pipeline_state["frame_count"] = 0
    pipeline_state["player_count"] = 0
    pipeline_state["error"] = None

    # Start pipeline in thread pool
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _start_pipeline, req.url, stream_id, req.home_team, req.away_team)

    return JSONResponse({"status": "started", "stream_id": stream_id})


@app.get("/status")
async def status():
    an: Analyzer | None = pipeline_state.get("analyzer")
    frame_count = an.frame_count if an is not None else pipeline_state["frame_count"]
    player_count = an.player_count if an is not None else pipeline_state["player_count"]
    return JSONResponse({
        "state":        pipeline_state["state"],
        "frame_count":  frame_count,
        "player_count": player_count,
        "error":        pipeline_state["error"],
    })


@app.post("/stop")
async def stop():
    await _stop_pipeline()
    return JSONResponse({"status": "stopped"})


class ApiKeyRequest(BaseModel):
    api_key: str


@app.post("/set_api_key")
async def set_api_key(req: ApiKeyRequest):
    global _api_key
    _api_key = req.api_key.strip()
    os.environ["ANTHROPIC_API_KEY"] = _api_key
    return JSONResponse({"status": "ok"})


@app.post("/lineup/{side}")
async def upload_lineup(side: str, file: UploadFile = File(...)):
    """
    Upload a lineup card image (home or away).
    Claude Vision extracts the roster and returns it.
    """
    global _lineups
    if side not in ("home", "away"):
        return JSONResponse({"error": "side must be home or away"}, status_code=400)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return JSONResponse({"error": "API key not set — click the key icon first"}, status_code=400)

    image_bytes = await file.read()
    content_type = file.content_type or "image/jpeg"

    loop = asyncio.get_event_loop()
    try:
        lineup = await loop.run_in_executor(
            None, analyze_lineup_image, image_bytes, content_type
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    _lineups[side] = lineup

    # If analyzer is running, push lineup into identity manager
    an: Analyzer | None = pipeline_state.get("analyzer")
    if an is not None:
        label = "A" if side == "home" else "B"
        _inject_lineup_into_analyzer(an, lineup, label)

    return JSONResponse({"status": "ok", "lineup": lineup})


@app.get("/lineup")
async def get_lineups():
    return JSONResponse(_lineups)


@app.get("/summary")
async def get_summary():
    an: Analyzer | None = pipeline_state.get("analyzer")
    if an is None:
        return JSONResponse({"error": "No active analysis session"}, status_code=400)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return JSONResponse({"error": "API key not set"}, status_code=400)

    stats = an.get_summary_stats()
    if not stats:
        return JSONResponse({"error": "Not enough data yet"}, status_code=400)

    loop = asyncio.get_event_loop()
    try:
        summary = await loop.run_in_executor(None, generate_match_summary, stats, "")
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    return JSONResponse(summary)


@app.websocket("/ws/feed")
async def ws_feed(ws: WebSocket):
    await ws.accept()
    async with _ws_lock:
        _ws_clients.append(ws)
    try:
        # Keep the connection alive; client sends nothing
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send a ping to keep connection alive
                try:
                    await ws.send_text('{"ping":1}')
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        async with _ws_lock:
            if ws in _ws_clients:
                _ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
