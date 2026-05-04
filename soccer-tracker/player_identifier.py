import base64
import json
import queue
import threading
import time
import cv2
import numpy as np
import anthropic


class PlayerIdentifier:
    """
    Background thread that sends player bbox crops to Claude Vision,
    reads the jersey number, determines the team, and stores results
    for the main loop to pick up.
    """

    def __init__(self, match_data: dict, jersey_lookup: dict):
        self.match_data = match_data
        self.jersey_lookup = jersey_lookup
        self._client = anthropic.Anthropic()
        self._queue: queue.Queue = queue.Queue(maxsize=30)
        self._results: dict[int, dict] = {}
        self._queued: set[int] = set()
        self._lock = threading.Lock()

        home = match_data.get("home_team", {})
        away = match_data.get("away_team", {})
        self._squad_json = json.dumps({
            "home_team": {
                "name": home.get("name"),
                "short": home.get("short"),
                "primary_color_bgr": home.get("primary_color_bgr"),
                "squad": home.get("squad", []),
            },
            "away_team": {
                "name": away.get("name"),
                "short": away.get("short"),
                "primary_color_bgr": away.get("primary_color_bgr"),
                "squad": away.get("squad", []),
            },
        }, indent=2)

        # Verify API key is present before starting
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[vision] WARNING: ANTHROPIC_API_KEY not set — player identification disabled")
            self._enabled = False
        else:
            self._enabled = True

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def queue_crop(self, tracker_id: int, frame: np.ndarray, bbox: tuple):
        """Submit a player crop for identification. Safe to call every frame — deduped internally."""
        if not self._enabled or tracker_id in self._queued:
            return
        x1, y1, x2, y2 = bbox
        if (x2 - x1) < 40 or (y2 - y1) < 70:
            return
        crop = frame[max(0, y1):y2, max(0, x1):x2].copy()
        try:
            self._queue.put_nowait((tracker_id, crop))
            self._queued.add(tracker_id)
        except queue.Full:
            pass

    def get_result(self, tracker_id: int) -> dict | None:
        with self._lock:
            return self._results.get(tracker_id)

    def _worker(self):
        while True:
            tracker_id, crop = self._queue.get()
            try:
                result = self._identify(crop)
                if result:
                    with self._lock:
                        self._results[tracker_id] = result
                    print(f"[vision] #{tracker_id} → {result}")
            except Exception as exc:
                print(f"[vision] #{tracker_id} failed: {exc}")
            finally:
                self._queue.task_done()
            time.sleep(0.1)  # gentle rate limiting

    def _identify(self, crop: np.ndarray) -> dict | None:
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        b64 = base64.b64encode(buf.tobytes()).decode()

        prompt = (
            "You are analyzing a single soccer player cropped from a broadcast frame.\n\n"
            f"Match squads (JSON):\n{self._squad_json}\n\n"
            "Tasks:\n"
            "1. Read the jersey number printed on the shirt (it may be partially visible or motion-blurred — do your best).\n"
            "2. Identify which team this player belongs to based on the kit color compared to the primary_color_bgr values above.\n\n"
            "Reply with ONLY a JSON object, no markdown, no extra text:\n"
            '{"jersey_number": <integer or null>, "team_short": "<short code>", "confidence": "high|medium|low"}'
        )

        response = self._client.messages.create(
            model="claude-opus-4-7",
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )

        text = next((b.text for b in response.content if b.type == "text"), "")
        data = json.loads(text.strip())
        if data.get("confidence") not in ("high", "medium"):
            return None

        # Enrich with name/position from jersey_lookup
        jnum = data.get("jersey_number")
        if jnum is not None and int(jnum) in self.jersey_lookup:
            info = self.jersey_lookup[int(jnum)]
            data["name"] = info["name"]
            data["position"] = info["position"]

        return data
