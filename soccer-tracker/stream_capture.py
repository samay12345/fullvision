"""
stream_capture.py — Capture frames from a YouTube URL using yt-dlp + ffmpeg.

yt-dlp resolves the best available 1080p stream URL(s).
ffmpeg decodes the video and pipes raw BGR frames to this process.
Frames are pushed into a queue for the analyzer to consume.
"""
from __future__ import annotations

import queue
import subprocess
import threading
import time
import json
import numpy as np


class StreamCapture:
    """
    Capture frames from a YouTube (or direct) URL.

    Uses yt-dlp to resolve the direct stream URL, then ffmpeg to decode
    and pipe raw BGR frames. This allows true 1080p quality by using
    separate DASH video+audio streams that ffmpeg merges on the fly.

    Frames are pushed to *frame_queue*. If the queue is full the frame
    is dropped (non-blocking put).
    """

    def __init__(self, youtube_url: str, frame_queue: "queue.Queue[np.ndarray]"):
        self._url = youtube_url
        self._frame_queue = frame_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Video properties filled after the stream is opened
        self._fps: float = 30.0
        self._frame_width: int = 1920
        self._frame_height: int = 1080
        self._error: str | None = None
        self._proc: subprocess.Popen | None = None

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

    def _resolve_stream_info(self) -> tuple[str, str | None, float, int, int]:
        """
        Use yt-dlp to get direct stream URL(s) and video properties.

        Returns (video_url, audio_url_or_None, fps, width, height).
        For combined streams, audio_url is None.
        """
        is_youtube = "youtube.com" in self._url or "youtu.be" in self._url

        if not is_youtube:
            # Direct URL — probe with ffprobe for dimensions/fps
            return self._probe_direct(self._url)

        # Try best video+audio merged first (requires ffmpeg, gets 1080p)
        # yt-dlp format selection: prefer 1080p60, fall back to 1080p, then 720p
        fmt = (
            "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]"
            "/bestvideo[height<=1080]+bestaudio"
            "/best[height<=1080]"
            "/best"
        )
        cmd = [
            "yt-dlp",
            "-f", fmt,
            "--get-url",
            "--no-playlist",
            "--no-warnings",
            self._url,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"yt-dlp failed (exit {result.returncode}): {result.stderr.strip()[:300]}"
                )
            urls = result.stdout.strip().splitlines()
            if not urls or not urls[0]:
                raise RuntimeError("yt-dlp returned an empty URL")

            video_url = urls[0]
            audio_url = urls[1] if len(urls) > 1 else None

            # Probe the video stream for fps/dimensions
            fps, w, h = self._ffprobe_stream(video_url)
            return video_url, audio_url, fps, w, h

        except FileNotFoundError:
            raise RuntimeError("yt-dlp not found. Install with: pip install yt-dlp")
        except subprocess.TimeoutExpired:
            raise RuntimeError("yt-dlp timed out while resolving stream URL")

    def _ffprobe_stream(self, url: str) -> tuple[float, int, int]:
        """Run ffprobe to get fps, width, height from a stream URL."""
        try:
            r = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,r_frame_rate",
                    "-of", "json",
                    url,
                ],
                capture_output=True, text=True, timeout=15,
            )
            info = json.loads(r.stdout)
            streams = info.get("streams", [{}])
            s = streams[0] if streams else {}
            w = int(s.get("width", 1920))
            h = int(s.get("height", 1080))
            rfr = s.get("r_frame_rate", "30/1")
            num, den = rfr.split("/")
            fps = float(num) / float(den) if float(den) else 30.0
            return fps, w, h
        except Exception:
            return 30.0, 1920, 1080

    def _probe_direct(self, url: str) -> tuple[str, None, float, int, int]:
        """For non-YouTube URLs, probe directly."""
        fps, w, h = self._ffprobe_stream(url)
        return url, None, fps, w, h

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            video_url, audio_url, fps, w, h = self._resolve_stream_info()
        except RuntimeError as exc:
            self._error = str(exc)
            return

        self._fps = fps if fps > 0 else 30.0
        self._frame_width = w
        self._frame_height = h

        frame_size = w * h * 3  # BGR bytes per frame

        # Build ffmpeg command to pipe raw BGR frames
        # -re: read at native rate (important for live streams)
        # -i video: primary video input
        # -i audio (optional): merge audio so ffmpeg doesn't stall on demux
        # -vf scale: preserve resolution (no downscale)
        # -f rawvideo -pix_fmt bgr24: pipe BGR bytes
        ff_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]

        if audio_url:
            ff_cmd += ["-i", video_url, "-i", audio_url]
        else:
            ff_cmd += ["-i", video_url]

        ff_cmd += [
            "-vf", f"scale={w}:{h}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",          # no audio output
            "pipe:1",
        ]

        try:
            proc = subprocess.Popen(
                ff_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=frame_size * 4,
            )
            self._proc = proc
        except FileNotFoundError:
            self._error = "ffmpeg not found. Install with: brew install ffmpeg"
            return

        frame_delay = 1.0 / self._fps
        last_frame_time = 0.0

        try:
            while not self._stop_event.is_set():
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    # Stream ended or error
                    break

                frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()

                # Throttle to native fps to avoid flooding the queue
                now = time.monotonic()
                elapsed = now - last_frame_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                last_frame_time = time.monotonic()

                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    pass  # drop frame to avoid blocking
        finally:
            proc.stdout.close()
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            self._proc = None

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
        if self._proc is not None:
            self._proc.terminate()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def is_alive(self) -> bool:
        """Return True if the capture thread is still running."""
        return self._thread is not None and self._thread.is_alive()
