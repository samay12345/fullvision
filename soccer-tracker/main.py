import argparse
import time
import cv2
from ultralytics import YOLO
from config import (
    VIDEO_SOURCE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    TRACKER, YOLO_MODEL, PERSON_CLASS_ID, BALL_CLASS_ID,
    FRAME_WIDTH, FRAME_HEIGHT, RECORD_FPS, WEB_PORT,
)

YOLO_EVERY_N_FRAMES = 3
from tracker import PlayerRegistry, merge_player
from team_classifier import TeamClassifier
from stat_engine import StatEngine
from overlay import Overlay, set_team_colors
from youtube_input import is_youtube_url, get_stream_url
from homography import PitchHomography
from reid import ReIDTracker
from formation import detect_formation


def _resolve_match(description: str) -> dict | None:
    try:
        from match_resolver import resolve_match
        print(f"[match] Querying Claude for squad data: {description!r}")
        data = resolve_match(description)
        home = data.get("home_team", {})
        away = data.get("away_team", {})
        print(
            f"[match] {home.get('name')} vs {away.get('name')} "
            f"| {data.get('competition')} | {data.get('date')}"
        )
        return data
    except Exception as exc:
        print(f"[match] Warning: could not resolve match — {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Soccer Tracker")
    parser.add_argument("--source", default=VIDEO_SOURCE)
    parser.add_argument("--match", default="", help="e.g. 'Bayern vs PSG 2024'")
    parser.add_argument("--output", default="stats_output.json")
    parser.add_argument("--record", action="store_true",
                        help="Save annotated video to output.mp4")
    parser.add_argument("--web", action="store_true",
                        help="Start live web dashboard on port 5000")
    parser.add_argument("--calibrate", action="store_true",
                        help="Interactively calibrate pitch homography on first frame")
    args = parser.parse_args()

    # ── Match resolution via Claude text ─────────────────────────────────────
    match_data = None
    jersey_lookup: dict[int, dict] = {}
    home_meta = away_meta = None
    identifier = None

    if args.match:
        match_data = _resolve_match(args.match)

    if match_data:
        from match_resolver import build_jersey_lookup
        from player_identifier import PlayerIdentifier
        jersey_lookup = build_jersey_lookup(match_data)
        home_meta = match_data["home_team"]
        away_meta = match_data["away_team"]
        set_team_colors(
            home_meta["short"], tuple(home_meta["primary_color_bgr"]),
            away_meta["short"], tuple(away_meta["primary_color_bgr"]),
        )
        identifier = PlayerIdentifier(match_data, jersey_lookup)
        print("[vision] Player identifier started (background thread)")

    # ── Video source ──────────────────────────────────────────────────────────
    source = args.source
    if is_youtube_url(source):
        print("[source] Extracting stream URL from YouTube…")
        source = get_stream_url(source)
        print("[source] Stream URL obtained")

    # ── Pipeline setup ────────────────────────────────────────────────────────
    model = YOLO(YOLO_MODEL)
    registry = PlayerRegistry()
    classifier = TeamClassifier()
    stat_engine = StatEngine(registry)
    overlay = Overlay()
    homography = PitchHomography()
    reid = ReIDTracker()

    if match_data:
        overlay.match_info = f"{home_meta['short']} vs {away_meta['short']}"

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"Error: could not open video source '{source}'")
        return

    # ── Optional web dashboard ────────────────────────────────────────────────
    if args.web:
        import web_server
        web_server.start_server(port=WEB_PORT)

    # ── Optional video recording ──────────────────────────────────────────────
    video_writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            "output.mp4", fourcc, RECORD_FPS, (FRAME_WIDTH, FRAME_HEIGHT)
        )
        print("[record] Recording to output.mp4")

    frame_num = 0
    last_player_detections = []
    last_ball_pos = None
    fps_t = time.time()
    fps_display = 0.0
    paused = False
    speed = 1.0  # playback multiplier

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    SEEK_SECONDS = 10  # how many seconds to jump per keypress

    # Track which IDs were active in the previous YOLO frame
    prev_active_ids: set[int] = set()

    # ── Calibrate homography on first frame ───────────────────────────────────
    first_frame_grabbed = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calibrate on first frame if requested
        if not first_frame_grabbed:
            first_frame_grabbed = True
            if args.calibrate:
                print("[calibrate] Click the 4 pitch corners: TL → TR → BR → BL")
                success = homography.calibrate_interactive(frame.copy())
                if success:
                    print("[calibrate] Homography calibrated successfully.")
                else:
                    print("[calibrate] Calibration cancelled — using frame-normalised minimap.")

        run_yolo = (frame_num % YOLO_EVERY_N_FRAMES == 0)

        if run_yolo:
            results = model.track(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                tracker=TRACKER,
                persist=True,
                classes=[PERSON_CLASS_ID, BALL_CLASS_ID],
                verbose=False,
            )[0]
        else:
            results = None

        player_detections = []
        ball_pos = None

        if run_yolo and results is not None and results.boxes is not None and results.boxes.id is not None:
            current_active_ids: set[int] = set()

            for box, cls, track_id in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.cls.cpu().numpy(),
                results.boxes.id.cpu().numpy(),
            ):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                class_id = int(cls)
                tracker_id = int(track_id)

                if class_id == PERSON_CLASS_ID:
                    current_active_ids.add(tracker_id)

                    # ── Re-ID: handle new tracker IDs ─────────────────────────
                    if tracker_id not in registry.players:
                        old_id = reid.match_new_id(tracker_id, frame, (x1, y1, x2, y2), frame_num)
                        if old_id is not None:
                            # Create the new slot first, then merge
                            registry.get_or_create(tracker_id)
                            merge_player(registry, old_id, tracker_id)

                    player = registry.get_or_create(tracker_id)
                    player.last_seen_frame = frame_num  # keep active window in sync with main frame_num

                    # ── Team classification ───────────────────────────────────
                    if match_data:
                        team = classifier.classify_direct(
                            frame, (x1, y1, x2, y2),
                            home_meta["primary_color_bgr"],
                            away_meta["primary_color_bgr"],
                            home_meta["short"],
                            away_meta["short"],
                        )
                    else:
                        if frame_num % 10 == 0:
                            classifier.add_sample(frame, (x1, y1, x2, y2), tracker_id)
                        if not classifier.fitted and frame_num > 30:
                            classifier.fit()
                        team = classifier.classify(frame, (x1, y1, x2, y2))

                    player.team = team

                    # ── Vision identification (background) ────────────────────
                    if identifier and player.player_name is None:
                        identifier.queue_crop(tracker_id, frame, (x1, y1, x2, y2))

                    if identifier:
                        result = identifier.get_result(tracker_id)
                        if result and player.player_name is None:
                            jnum = result.get("jersey_number")
                            if jnum is not None and int(jnum) in jersey_lookup:
                                info = jersey_lookup[int(jnum)]
                                player.jersey_number = int(jnum)
                                player.player_name = info["name"]
                                player.position = info["position"]
                            team_from_vision = result.get("team_short")
                            if team_from_vision:
                                player.team = team_from_vision
                                team = team_from_vision

                    player_detections.append({
                        "tracker_id": tracker_id,
                        "bbox": (x1, y1, x2, y2),
                        "center": (cx, cy),
                        "team": player.team or team,
                    })

                    # ── Re-ID: update active gallery ──────────────────────────
                    reid.update(tracker_id, frame, (x1, y1, x2, y2), frame_num)

                elif class_id == BALL_CLASS_ID:
                    ball_pos = (cx, cy)

            # ── Mark disappeared IDs as lost ──────────────────────────────────
            disappeared_ids = prev_active_ids - current_active_ids
            for tid in disappeared_ids:
                reid.mark_lost(tid, frame_num)

            prev_active_ids = current_active_ids
            last_player_detections = player_detections
            last_ball_pos = ball_pos
        else:
            player_detections = last_player_detections
            ball_pos = last_ball_pos

        if run_yolo:
            stat_engine.update(player_detections, ball_pos, frame_num)

        # ── Compute formations ────────────────────────────────────────────────
        fh, fw = frame.shape[:2]
        active = registry.active_players(frame_num)
        all_teams = {p.team for p in active if p.team and p.team != "unknown"}
        formations: dict[str, str] = {}
        for t in all_teams:
            formations[t] = detect_formation(active, t, fh)

        # ── Draw detections ───────────────────────────────────────────────────
        for det in player_detections:
            tid = det["tracker_id"]
            player = registry.get_or_create(tid)
            stats = player.to_dict()
            frame = overlay.draw_player(frame, det["bbox"], tid, player.team, stats)

        if ball_pos:
            frame = overlay.draw_ball(frame, ball_pos)

        # ── Possession & pass network overlays ───────────────────────────────
        possession_pct = stat_engine.get_possession_pct()
        frame = overlay.draw_possession_bar(frame, possession_pct)
        frame = overlay.draw_pass_network(frame, stat_engine.pass_events, frame_num)
        frame = overlay.draw_minimap(frame, active, ball_pos, homography, fw, fh)

        now = time.time()
        fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - fps_t, 1e-6))
        fps_t = now

        # ── Status bar (bottom of frame) ──────────────────────────────────────
        status = f"FPS:{fps_display:.1f}  Speed:{speed:.1f}x"
        if paused:
            status += "  [PAUSED]"
        status += "  SPACE=pause  LEFT=-10s  RIGHT=+10s  ]=faster  [=slower  q=quit"
        cv2.rectangle(frame, (0, fh - 22), (fw, fh), (0, 0, 0), -1)
        cv2.putText(frame, status, (6, fh - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        frame = overlay.draw_hud(frame, registry, frame_num, formations=formations)
        cv2.imshow("Soccer Tracker", frame)

        # ── Optional outputs ──────────────────────────────────────────────────
        if args.web:
            web_stats = {
                str(p.tracker_id): p.to_dict()
                for p in active
            }
            import web_server
            web_server.update_frame(frame)
            web_server.update_stats(web_stats)

        if video_writer is not None:
            out_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            video_writer.write(out_frame)

        # ── Input handling ────────────────────────────────────────────────────
        # On macOS: left arrow = 2, right arrow = 3  (after & 0xFF)
        delay = max(1, int(1000 / (native_fps * speed))) if not paused else 30
        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q"):
            break
        elif key == 32:  # Space — toggle pause
            paused = not paused
            while paused:
                key2 = cv2.waitKey(30) & 0xFF
                if key2 == 32:
                    paused = False
                elif key2 == ord("q"):
                    paused = False
                    cap.release()
                    cv2.destroyAllWindows()
                    if video_writer:
                        video_writer.release()
                    registry.export_stats(args.output)
                    return
                elif key2 == 3:  # right arrow — seek forward while paused
                    target = cap.get(cv2.CAP_PROP_POS_FRAMES) + native_fps * SEEK_SECONDS
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target))
                elif key2 == 2:  # left arrow — seek back while paused
                    target = cap.get(cv2.CAP_PROP_POS_FRAMES) - native_fps * SEEK_SECONDS
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target))
        elif key == 3:  # right arrow — forward 10s
            target = cap.get(cv2.CAP_PROP_POS_FRAMES) + native_fps * SEEK_SECONDS
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            last_player_detections = []
            last_ball_pos = None
        elif key == 2:  # left arrow — rewind 10s
            target = cap.get(cv2.CAP_PROP_POS_FRAMES) - native_fps * SEEK_SECONDS
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target))
            last_player_detections = []
            last_ball_pos = None
        elif key == ord("]"):
            speed = min(speed * 2, 8.0)
        elif key == ord("["):
            speed = max(speed / 2, 0.25)

        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()

    if video_writer is not None:
        video_writer.release()
        print("[record] Video saved to output.mp4")

    registry.export_stats(args.output)
    print(f"Stats saved to {args.output}  ({frame_num} frames processed)")

    # ── Match report ──────────────────────────────────────────────────────────
    if args.match:
        from match_report import generate_report
        possession_pct = stat_engine.get_possession_pct()
        match_info_str = overlay.match_info or args.match
        print("[report] Generating match report…")
        report = generate_report(registry, match_info_str, frame_num, possession_pct)
        print("\n" + "=" * 60)
        print(report)
        print("=" * 60)


if __name__ == "__main__":
    main()
