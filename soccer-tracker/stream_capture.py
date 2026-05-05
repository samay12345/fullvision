"""
stream_capture.py — Capture frames from a YouTube URL using yt-dlp + OpenCV.

Runs in a background thread and pushes frames into a queue.
"""
from __future__ import annotations

import queue
import subprocess
import threading
import time

import cv2


class StreamCapture:
    """
    Capture frames from a YouTube (or direct) URL.

    Resolved stream URL is obtained via yt-dlp subprocess call.
    Frames are decoded with OpenCV and pushed to *frame_queue*.
    If the queue is full the frame is dropped (non-blocking put).
    """

    def __init__(self, youtube_url: str, frame_queue: "queue.Queue[cv2.typing.MatLike]"):
        self._url = youtube_url
        self._frame_queue = frame_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Video properties filled after the stream is opened
        self._fps: float = 30.0
        self._frame_width: int = 1280
        self._frame_height: int = 720
        self._stream_url: str | None = None
        self._error: str | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_width(self) -> int:
        return self._frame_width

    @property
    def frame_height(self) -> int:
        return self._frame_height

    @property
    def error(self) -> str | None:
        return self._error

    # ------------------------------------------------------------------
    # Stream URL resolution
    # ------------------------------------------------------------------

    def _resolve_stream_url(self) -> str:
        """
        Use yt-dlp subprocess to get a direct streamable URL.
        Falls back to treating the input as a direct URL if yt-dlp fails.
        """
        # Check if this looks like a YouTube URL
        if "youtube.com" not in self._url and "youtu.be" not in self._url:
            return self._url  # treat as direct URL

        cmd = [
            "yt-dlp",
            "-f", "best[height<=720][ext=mp4]/best[height<=720]/best",
            "--get-url",
            "--no-playlist",
            self._url,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                raise RuntimeError(
                    f"yt-dlp failed (exit {result.returncode}): {stderr[:300]}"
                )
            url = result.stdout.strip().splitlines()[0]
            if not url:
                raise RuntimeError("yt-dlp returned an empty URL")
            return url
        except FileNotFoundError:
            raise RuntimeError(
                "yt-dlp not found. Install with: pip install yt-dlp"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("yt-dlp timed out while resolving stream URL")

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._stream_url = self._resolve_stream_url()
        except RuntimeError as exc:
            self._error = str(exc)
            return

        cap = cv2.VideoCapture(self._stream_url)
        if not cap.isOpened():
            self._error = f"OpenCV could not open stream: {self._stream_url[:80]}..."
            return

        # Read video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 0 else 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0:
            self._frame_width = w
        if h > 0:
            self._frame_height = h

        frame_delay = 1.0 / self._fps
        last_frame_time = 0.0

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # Try to recover from transient network hiccup
                time.sleep(0.5)
                if not cap.isOpened():
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES))
                ret2, frame = cap.read()
                if not ret2:
                    break

            # Throttle to native fps
            now = time.monotonic()
            elapsed = now - last_frame_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
            last_frame_time = time.monotonic()

            # Push to queue — drop if full
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # drop frame to avoid blocking

        cap.release()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the capture thread (returns immediately)."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="StreamCapture")
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def is_alive(self) -> bool:
        """Return True if the capture thread is still running."""
        return self._thread is not None and self._thread.is_alive()
