"""
lineup_analyzer.py — Claude Vision API lineup card analyzer.

Sends a TV lineup graphic to Claude and extracts jersey numbers,
player names, positions, formation, and substitutes as structured JSON.
"""
from __future__ import annotations

import base64
import json
import os
import re

import anthropic


def analyze_lineup_image(image_bytes: bytes, media_type: str = "image/jpeg") -> dict:
    """
    Extract lineup data from a TV lineup card image using Claude Vision.

    Returns dict:
    {
        "team_name": str,
        "formation": str,          # e.g. "4-3-3"
        "starters": [
            {"jersey": int, "name": str, "position": str}
        ],
        "substitutes": [
            {"jersey": int, "name": str}
        ]
    }
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "This is a soccer/football TV lineup card. "
                            "Extract ALL players shown.\n\n"
                            "Return ONLY valid JSON — no markdown, no explanation:\n"
                            "{\n"
                            '  "team_name": "Full Team Name",\n'
                            '  "formation": "4-3-3",\n'
                            '  "starters": [\n'
                            '    {"jersey": 9, "name": "First Last", "position": "ST"}\n'
                            "  ],\n"
                            '  "substitutes": [\n'
                            '    {"jersey": 1, "name": "First Last"}\n'
                            "  ]\n"
                            "}\n\n"
                            "Positions: GK CB LB RB LWB RWB CDM CM CAM LW RW ST.\n"
                            "Use the jersey numbers shown on the shirt icons for starters. "
                            "Use the numbers listed beside substitute names. "
                            "Spell names exactly as shown."
                        ),
                    },
                ],
            }
        ],
    )

    raw = msg.content[0].text.strip()

    # Strip markdown code fences if Claude wrapped the JSON
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
        else:
            raise ValueError(f"Claude returned non-JSON: {raw[:300]}")

    return data


def generate_match_summary(player_stats: list[dict], match_info: str = "") -> dict:
    """
    Generate per-player end-of-match summaries using Claude.

    player_stats: list of dicts with keys:
        name, team, rating, speed_ms, distance_m, sprints,
        passes, tackles, shots, events (list of event type strings)

    Returns:
    {
        "players": [
            {
                "name": str,
                "rating": float,
                "summary": str,       # 2-3 sentence narrative
                "strengths": str,     # what they did well
                "improve": str        # one area to work on
            }
        ],
        "match_narrative": str        # 2-sentence overall match summary
    }
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Split into played vs DNA
    played = [p for p in player_stats if not p.get("did_not_play")]
    dnp    = [p for p in player_stats if p.get("did_not_play")]

    compact = []
    for p in played:
        compact.append({
            "name":         p.get("name") or f"Player #{p.get('jersey', '?')}",
            "team":         p.get("team", ""),
            "position":     p.get("position", ""),
            "rating":       round(p.get("rating") or 6.5, 1),
            "top_speed_ms": round(p.get("top_speed_ms", 0), 1),
            "distance_m":   round(p.get("distance_m", 0)),
            "sprints":      p.get("sprints", 0),
            "passes":       p.get("passes", 0),
            "tackles":      p.get("tackles", 0),
            "shots":        p.get("shots", 0),
            "key_events":   p.get("key_events", [])[:8],
        })

    dnp_names = [
        {"name": p.get("name") or f"#{p.get('jersey','?')}", "team": p.get("team", "")}
        for p in dnp
    ]

    stats_json = json.dumps(compact, indent=2)
    dnp_json   = json.dumps(dnp_names, indent=2)
    context    = f"Match: {match_info}\n\n" if match_info else ""

    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{context}"
                    "You are a professional soccer analyst. Based on these player stats, "
                    "write a brief performance summary for every player who played.\n\n"
                    f"Players with tracking data:\n{stats_json}\n\n"
                    f"Players NOT seen in the footage (did not play / unused sub):\n{dnp_json}\n\n"
                    "Rules:\n"
                    "- For players WITH data: write summary (2-3 sentences referencing specific numbers), "
                    "strengths (one thing done well), improve (one area to work on).\n"
                    "- For players WITHOUT data: set summary to 'Did not play.', "
                    "strengths to '' and improve to ''.\n"
                    "- Include ALL players from both lists in the output.\n\n"
                    "Also write a 2-sentence match_narrative about the overall game.\n\n"
                    "Return ONLY valid JSON:\n"
                    '{"players": [{"name": "...", "team": "...", "position": "...", "rating": 7.2, '
                    '"did_not_play": false, "summary": "...", "strengths": "...", "improve": "..."}], '
                    '"match_narrative": "..."}'
                ),
            }
        ],
    )

    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError(f"Claude returned non-JSON: {raw[:300]}")
