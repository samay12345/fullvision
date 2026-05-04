import yt_dlp


def get_stream_url(youtube_url: str) -> str:
    """Extract a direct streamable video URL from a YouTube link using yt-dlp."""
    ydl_opts = {
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]/best[height<=720]",
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        # For merged formats yt-dlp returns a direct URL
        if "url" in info:
            return info["url"]
        # For split video+audio formats pick the best video-only stream
        for fmt in reversed(info.get("formats", [])):
            if fmt.get("vcodec") != "none" and fmt.get("url"):
                return fmt["url"]

    raise RuntimeError(f"Could not extract stream URL from: {youtube_url}")


def is_youtube_url(source: str) -> bool:
    return "youtube.com" in source or "youtu.be" in source
