"""
web_server.py — Flask MJPEG streaming server for live soccer tracker dashboard.
"""
from __future__ import annotations
import threading
import io

import cv2
import numpy as np

# ------------------------------------------------------------------
# Shared state
# ------------------------------------------------------------------
_frame_buf: bytes | None = None
_stats_buf: dict = {}
_lock = threading.Lock()

# Flask is imported lazily so that importing this module does not
# hard-crash when Flask is unavailable during testing.
_app = None


def _build_app():
    """Lazily create and return the Flask application."""
    global _app
    if _app is not None:
        return _app

    from flask import Flask, Response, jsonify

    app = Flask(__name__)

    _DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Soccer Tracker — Live Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d0d0d; color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; }
  header { background: #1a1a2e; padding: 14px 24px; display: flex; align-items: center; gap: 16px; border-bottom: 2px solid #16213e; }
  header h1 { font-size: 1.4rem; color: #00d4aa; letter-spacing: 1px; }
  .badge { background: #e94560; color: #fff; border-radius: 4px; font-size: 0.7rem; padding: 2px 7px; text-transform: uppercase; }
  .container { display: flex; gap: 16px; padding: 16px; flex-wrap: wrap; }
  .video-panel { flex: 2; min-width: 320px; }
  .video-panel img { width: 100%; border-radius: 8px; border: 2px solid #16213e; background: #111; }
  .stats-panel { flex: 1; min-width: 260px; }
  .stats-panel h2 { font-size: 0.9rem; color: #00d4aa; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
  #stats-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
  #stats-table th { background: #16213e; color: #aaa; padding: 6px 8px; text-align: left; border-bottom: 1px solid #222; }
  #stats-table td { padding: 5px 8px; border-bottom: 1px solid #1a1a1a; }
  #stats-table tr:hover td { background: #1a1a2e; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
  #last-update { font-size: 0.65rem; color: #555; margin-top: 8px; }
  footer { text-align: center; padding: 12px; font-size: 0.7rem; color: #333; border-top: 1px solid #111; }
</style>
</head>
<body>
<header>
  <h1>Soccer Tracker</h1>
  <span class="badge">Live</span>
</header>
<div class="container">
  <div class="video-panel">
    <img src="/video" alt="Live feed" id="stream">
  </div>
  <div class="stats-panel">
    <h2>Player Stats</h2>
    <table id="stats-table">
      <thead>
        <tr>
          <th>#</th><th>Name</th><th>Team</th><th>Dist</th>
          <th>Spd</th><th>Pass</th><th>Shts</th><th>Tkl</th><th>Rtg</th>
        </tr>
      </thead>
      <tbody id="stats-body"><tr><td colspan="9" style="color:#555;text-align:center">Loading…</td></tr></tbody>
    </table>
    <div id="last-update"></div>
  </div>
</div>
<footer>Soccer Tracker &mdash; Live Dashboard</footer>

<script>
function teamColor(team) {
  if (!team) return '#888';
  const h = [...team].reduce((a,c)=>a+c.charCodeAt(0),0);
  const colors = ['#e94560','#00d4aa','#f5a623','#4a90e2','#7ed321'];
  return colors[h % colors.length];
}

async function refreshStats() {
  try {
    const resp = await fetch('/stats');
    const data = await resp.json();
    const players = Array.isArray(data) ? data : Object.values(data);
    const tbody = document.getElementById('stats-body');
    if (!players.length) {
      tbody.innerHTML = '<tr><td colspan="9" style="color:#555;text-align:center">No data yet…</td></tr>';
      return;
    }
    players.sort((a,b) => (b.rating||0) - (a.rating||0));
    tbody.innerHTML = players.slice(0, 20).map(p => {
      const color = teamColor(p.team);
      const name = p.name ? p.name.split(' ').pop() : ('#' + p.tracker_id);
      const num  = p.jersey_number ? '#'+p.jersey_number+' ' : '';
      return '<tr>' +
        '<td><span class="dot" style="background:' + color + '"></span>' + (p.tracker_id||'') + '</td>' +
        '<td>' + num + name + '</td>' +
        '<td>' + (p.team||'?') + '</td>' +
        '<td>' + (p.total_distance_px||0).toFixed(0) + '</td>' +
        '<td>' + (p.avg_speed_mps||0).toFixed(1) + '</td>' +
        '<td>' + (p.passes||0) + '</td>' +
        '<td>' + (p.shots||0) + '</td>' +
        '<td>' + (p.tackles||0) + '</td>' +
        '<td><b style="color:' + (p.rating>=65?'#00d4aa':p.rating>=40?'#f5a623':'#e94560') + '">' + (p.rating||0).toFixed(0) + '</b></td>' +
        '</tr>';
    }).join('');
    document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString();
  } catch(e) {
    console.warn('stats fetch error', e);
  }
}

refreshStats();
setInterval(refreshStats, 2000);
</script>
</body>
</html>"""

    @app.route("/")
    def index():
        return _DASHBOARD_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

    @app.route("/video")
    def video_stream():
        def _generate():
            while True:
                with _lock:
                    buf = _frame_buf
                if buf is None:
                    # Send a black placeholder frame
                    blank = np.zeros((360, 640, 3), dtype=np.uint8)
                    _, enc = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    buf = enc.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf + b"\r\n"
                )
        return Response(
            _generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/stats")
    def stats():
        with _lock:
            data = dict(_stats_buf)
        return jsonify(data)

    _app = app
    return _app


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def update_frame(frame: np.ndarray):
    """JPEG-encode frame and store for streaming."""
    global _frame_buf
    _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    with _lock:
        _frame_buf = enc.tobytes()


def update_stats(stats_dict: dict):
    """Store a stats dict for the /stats endpoint."""
    global _stats_buf
    with _lock:
        _stats_buf = dict(stats_dict)


def start_server(port: int = 5000):
    """Start Flask in a background daemon thread."""
    app = _build_app()

    def _run():
        # Silence Flask startup banner
        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print(f"[web] Dashboard available at  http://localhost:{port}")
