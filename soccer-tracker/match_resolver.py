import json
import anthropic

_CLIENT = None

_SYSTEM_PROMPT = """\
You are a professional football (soccer) data assistant with deep knowledge of all major \
leagues, clubs, and international competitions up to your knowledge cutoff.

When the user describes a match (e.g. "Bayern Munich vs PSG last week", \
"Argentina vs France World Cup 2022 final"), return ONLY a valid JSON object — \
no markdown fences, no extra text — with this exact schema:

{
  "home_team": {
    "name": "<Club or national team full name>",
    "short": "<3-letter abbreviation>",
    "primary_color_bgr": [B, G, R],
    "secondary_color_bgr": [B, G, R],
    "squad": [
      {
        "jersey_number": <int>,
        "name": "<Full name>",
        "position": "<GK|DEF|MID|FWD>"
      }
    ]
  },
  "away_team": { <same structure as home_team> },
  "competition": "<Competition name>",
  "date": "<YYYY-MM-DD or approximate>",
  "notes": "<any relevant context, e.g. injury absences, lineup source>"
}

primary_color_bgr is the dominant jersey color for that team in this specific match \
(OpenCV BGR order, values 0-255). Include all players in the starting XI plus key \
substitutes if known. If you are uncertain about exact colors use the team's most \
well-known home/away kit color for this season."""


def _client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic()
    return _CLIENT


def resolve_match(description: str) -> dict:
    """
    Given a natural language match description, return a dict with squad data
    for both teams, using Claude claude-opus-4-7 with prompt caching.
    """
    response = _client().messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": description,
            }
        ],
    )

    # Extract text block (thinking blocks come first; skip them)
    text_block = next(
        (b for b in response.content if b.type == "text"), None
    )
    if text_block is None:
        raise RuntimeError("Claude returned no text block in response")

    try:
        data = json.loads(text_block.text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Claude response was not valid JSON: {text_block.text[:500]}"
        ) from exc

    cache_created = getattr(response.usage, "cache_creation_input_tokens", 0)
    cache_read = getattr(response.usage, "cache_read_input_tokens", 0)
    print(
        f"[match_resolver] cache_creation={cache_created} "
        f"cache_read={cache_read} "
        f"output_tokens={response.usage.output_tokens}"
    )

    return data


def build_jersey_lookup(match_data: dict) -> dict[int, dict]:
    """
    Flatten both squads into {jersey_number: {name, position, team}} for fast
    lookup during tracking.
    """
    lookup: dict[int, dict] = {}
    for side_key in ("home_team", "away_team"):
        team = match_data.get(side_key, {})
        team_name = team.get("short", side_key)
        for player in team.get("squad", []):
            num = player.get("jersey_number")
            if num is not None:
                lookup[int(num)] = {
                    "name": player.get("name", ""),
                    "position": player.get("position", ""),
                    "team": team_name,
                }
    return lookup
