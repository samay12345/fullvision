"""
match_report.py — AI-powered match report generator using Claude.
"""
from __future__ import annotations
import os
import json
from tracker import PlayerRegistry


def generate_report(
    registry: PlayerRegistry,
    match_info: str,
    frame_count: int,
    possession: dict,
) -> str:
    """Generate a detailed match report.

    Uses Claude claude-opus-4-7 when ANTHROPIC_API_KEY is set; otherwise produces a
    plain-text report from the collected stats.

    The report is saved to 'match_report.txt' and also returned as a string.
    """
    # ------------------------------------------------------------------
    # Build stats summary
    # ------------------------------------------------------------------
    players_data = [p.to_dict() for p in registry.all_players() if len(p.to_dict().get("frames_tracked", 0) and [1] or []) or True]
    # Keep only players with meaningful data
    players_data = [p for p in players_data if p.get("frames_tracked", 0) >= 10]

    teams: dict[str, list[dict]] = {}
    for p in players_data:
        t = p.get("team") or "unknown"
        teams.setdefault(t, []).append(p)

    stats_summary: dict = {
        "match_info": match_info,
        "total_frames": frame_count,
        "possession_pct": possession,
        "teams": {},
    }

    for team, plist in teams.items():
        plist_sorted_dist = sorted(plist, key=lambda x: x.get("total_distance_px", 0), reverse=True)
        plist_sorted_pass = sorted(plist, key=lambda x: x.get("passes", 0), reverse=True)
        plist_sorted_rtg  = sorted(plist, key=lambda x: x.get("rating", 0), reverse=True)
        plist_sorted_tck  = sorted(plist, key=lambda x: x.get("tackles", 0), reverse=True)
        stats_summary["teams"][team] = {
            "player_count": len(plist),
            "total_passes": sum(p.get("passes", 0) for p in plist),
            "total_shots": sum(p.get("shots", 0) for p in plist),
            "total_tackles": sum(p.get("tackles", 0) for p in plist),
            "top_rated": plist_sorted_rtg[:3],
            "top_distance": plist_sorted_dist[:3],
            "top_passes": plist_sorted_pass[:3],
            "top_tackles": plist_sorted_tck[:3],
        }

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if api_key:
        report = _claude_report(stats_summary, api_key)
    else:
        report = _simple_report(stats_summary)

    # Save to file
    with open("match_report.txt", "w", encoding="utf-8") as fh:
        fh.write(report)
    print("[report] Saved to match_report.txt")

    return report


# ------------------------------------------------------------------
# Claude report
# ------------------------------------------------------------------

def _claude_report(stats: dict, api_key: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    stats_json = json.dumps(stats, indent=2)

    prompt = f"""You are an expert soccer analyst. Below is raw tracking data from an automated computer-vision system that analysed a match.

Match info: {stats.get('match_info', 'Unknown match')}
Total frames processed: {stats.get('total_frames', 0)}
Possession breakdown: {json.dumps(stats.get('possession_pct', {}), indent=2)}

Team and player stats (JSON):
{stats_json}

Please write a detailed, engaging match report covering:
1. **Overview** — brief match summary including possession and pace.
2. **Top performers per team** — highlight standout players with their key stats.
3. **Possession analysis** — which team dominated and what it meant tactically.
4. **Key stats breakdown** — most passes, most distance covered, most tackles.
5. **Tactical observations** — formation tendencies, pressing intensity, etc.
6. **Conclusion** — overall assessment of the match.

Write in a professional sports journalism style. Use the player names/numbers where available.
"""

    message = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


# ------------------------------------------------------------------
# Fallback plain-text report
# ------------------------------------------------------------------

def _simple_report(stats: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("MATCH REPORT")
    lines.append("=" * 60)
    lines.append(f"Match: {stats.get('match_info', 'Unknown')}")
    lines.append(f"Frames analysed: {stats.get('total_frames', 0)}")

    poss = stats.get("possession_pct", {})
    if poss:
        lines.append("\nPossession:")
        for team, pct in poss.items():
            lines.append(f"  {team}: {pct:.1f}%")

    for team, tdata in stats.get("teams", {}).items():
        lines.append(f"\n--- {team} ---")
        lines.append(f"  Players tracked : {tdata['player_count']}")
        lines.append(f"  Total passes    : {tdata['total_passes']}")
        lines.append(f"  Total shots     : {tdata['total_shots']}")
        lines.append(f"  Total tackles   : {tdata['total_tackles']}")

        top = tdata.get("top_rated", [])
        if top:
            lines.append("  Top rated players:")
            for p in top:
                name = p.get("name") or f"#{p.get('tracker_id')}"
                lines.append(
                    f"    {name} — Rating: {p.get('rating', 0):.0f}, "
                    f"Passes: {p.get('passes', 0)}, "
                    f"Distance: {p.get('total_distance_px', 0):.0f}px"
                )

        top_dist = tdata.get("top_distance", [])
        if top_dist:
            lines.append("  Most distance covered:")
            for p in top_dist:
                name = p.get("name") or f"#{p.get('tracker_id')}"
                lines.append(
                    f"    {name} — {p.get('total_distance_px', 0):.0f}px"
                )

    lines.append("\n" + "=" * 60)
    lines.append("(Note: Claude API key not set — simple report generated.)")
    lines.append("=" * 60)
    return "\n".join(lines)
