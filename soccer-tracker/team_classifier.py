import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    def __init__(self, n_teams: int = 2):
        self.kmeans = KMeans(n_clusters=n_teams, n_init=10, random_state=42)
        self.fitted = False
        self.color_samples: list[np.ndarray] = []
        self.sample_tracker_ids: list[int] = []
        # Maps cluster index (0/1) → team label ("A"/"B" or custom name)
        self.label_map: dict[int, str] = {0: "A", 1: "B"}

    def extract_jersey_color(self, frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1

        # Take the center third vertically to skip head and legs
        top = y1 + h // 3
        bot = y1 + (2 * h) // 3
        crop = frame[top:bot, x1:x2]

        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_color = hsv.mean(axis=(0, 1))
        return mean_color.astype(np.float32)

    def add_sample(self, frame: np.ndarray, bbox: tuple, tracker_id: int):
        color = self.extract_jersey_color(frame, bbox)
        if color is not None:
            self.color_samples.append(color)
            self.sample_tracker_ids.append(tracker_id)

    def fit(self):
        if len(self.color_samples) >= 10:
            X = np.array(self.color_samples)
            self.kmeans.fit(X)
            self.fitted = True

    def classify(self, frame: np.ndarray, bbox: tuple) -> str:
        color = self.extract_jersey_color(frame, bbox)
        if color is None or not self.fitted:
            return "unknown"
        cluster = int(self.kmeans.predict(color.reshape(1, -1))[0])
        return self.label_map.get(cluster, str(cluster))

    def classify_direct(
        self,
        frame: np.ndarray,
        bbox: tuple,
        home_color_bgr: list[int],
        away_color_bgr: list[int],
        home_name: str,
        away_name: str,
    ) -> str:
        """Classify immediately using known kit colors — no fitting required."""
        color = self.extract_jersey_color(frame, bbox)
        if color is None:
            return "unknown"

        def bgr_to_hsv(bgr):
            px = np.array([[bgr]], dtype=np.uint8)
            return cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0].astype(float)

        def dist(a, b):
            dh = min(abs(a[0] - b[0]), 180 - abs(a[0] - b[0]))
            return dh * 2 + abs(a[1] - b[1]) + abs(a[2] - b[2])

        home_hsv = bgr_to_hsv(home_color_bgr)
        away_hsv = bgr_to_hsv(away_color_bgr)
        return home_name if dist(color, home_hsv) <= dist(color, away_hsv) else away_name

    def auto_assign(
        self,
        home_color_bgr: list[int],
        away_color_bgr: list[int],
        home_name: str,
        away_name: str,
    ):
        """
        After fitting, map cluster centroids to home/away team names by
        comparing each centroid's HSV representation to the known kit colors.
        """
        if not self.fitted:
            return

        def bgr_to_hsv_vec(bgr):
            pixel = np.array([[bgr]], dtype=np.uint8)
            return cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0].astype(float)

        home_hsv = bgr_to_hsv_vec(home_color_bgr)
        away_hsv = bgr_to_hsv_vec(away_color_bgr)
        centroids = self.kmeans.cluster_centers_  # shape (2, 3), already HSV

        def dist(a, b):
            # Hue is circular (0-179 in OpenCV); weight it appropriately
            dh = min(abs(a[0] - b[0]), 180 - abs(a[0] - b[0]))
            return dh * 2 + abs(a[1] - b[1]) + abs(a[2] - b[2])

        # Cluster 0 vs home
        c0_home = dist(centroids[0], home_hsv)
        c0_away = dist(centroids[0], away_hsv)

        if c0_home <= c0_away:
            self.label_map = {0: home_name, 1: away_name}
        else:
            self.label_map = {0: away_name, 1: home_name}
