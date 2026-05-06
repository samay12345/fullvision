VIDEO_SOURCE = "assets/test_clip.mp4"

CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
TRACKER = "botsort.yaml"
YOLO_MODEL = "yolov8m.pt"

PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32

TEAM_A_COLOR = (0, 165, 255)   # orange
TEAM_B_COLOR = (60, 60, 220)   # red
UNKNOWN_COLOR = (200, 200, 200)

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

SPEED_SCALE_FACTOR = 1.0
PASS_DISTANCE_THRESHOLD = 80
SHOT_VELOCITY_THRESHOLD = 25

GOAL_REGIONS = [
    {"x1": 0,    "y1": 250, "x2": 80,   "y2": 470},
    {"x1": 1200, "y1": 250, "x2": 1280, "y2": 470},
]

MINIMAP_W = 200
MINIMAP_H = 130
RECORD_FPS = 30
WEB_PORT = 5000
REID_THRESHOLD = 0.82
REID_MAX_AGE = 180

# Kit colors (BGR) for team color assignment when using --teams flag
TEAM_KIT_COLORS_BGR: dict[str, tuple] = {
    "paris saint-germain":        (180, 30,  10),
    "fc barcelona":               (160, 0,   10),
    "real madrid":                (230, 230, 230),
    "arsenal":                    (30,  30,  200),
    "liverpool":                  (20,  20,  210),
    "manchester city":            (220, 180, 0),
    "fc bayern münchen":          (20,  20,  210),
    "inter":                      (140, 0,   0),
    "atlético madrid":            (30,  30,  200),
    "aston villa":                (80,  0,   140),
    "chelsea":                    (210, 50,  20),
    "newcastle united":           (30,  30,  30),
    "borussia dortmund":          (0,   220, 255),
    "ac milan":                   (20,  20,  200),
    "napoli":                     (220, 210, 10),
    "manchester united":          (20,  20,  200),
    "tottenham hotspur":          (230, 230, 230),
    "juventus":                   (30,  30,  30),
    "roma":                       (20,  40,  200),
    "galatasaray sk":             (20,  80,  220),
    "athletic club":              (20,  20,  200),
    "crystal palace":             (200, 0,   0),
    "nottingham forest":          (20,  20,  200),
    "bayer 04 leverkusen":        (20,  20,  200),
    "atalanta":                   (0,   0,   160),
    "rb leipzig":                 (180, 0,   150),
    "real betis balompié":        (0,   180, 0),
    "olympique de marseille":     (230, 230, 230),
    "villarreal cf":              (0,   230, 255),
    "sporting cp":                (0,   200, 0),
    "everton":                    (210, 50,  20),
    "brighton & hove albion":     (220, 160, 20),
    "lazio":                      (220, 210, 100),
    "as monaco":                  (20,  20,  200),
    "fenerbahçe sk":              (0,   210, 255),
    "al hilal":                   (20,  50,  180),
    "brentford":                  (20,  20,  200),
    "fulham fc":                  (230, 230, 230),
    "afc bournemouth":            (20,  20,  180),
    "real sociedad":              (20,  160, 210),
    "sl benfica":                 (20,  20,  200),
    "fc porto":                   (20,  50,  180),
    "west ham united":            (60,  0,   150),
    "eintracht frankfurt":        (30,  30,  30),
    "vfb stuttgart":              (20,  20,  200),
    "rc lens":                    (20,  100, 210),
    "olympique lyonnais":         (230, 230, 230),
    "sunderland":                 (20,  20,  200),
    "ca osasuna":                 (20,  20,  180),
    "rayo vallecano":             (230, 230, 230),
}
