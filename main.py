"""
Interactive Digital Web — Real-time ASCII with Hand Tracking
Renders webcam as ASCII art and draws pulsing "web" lines connecting
corresponding fingertips between two detected hands.

Install dependencies:
    pip install opencv-python mediapipe numpy

Model (auto-downloaded on first run):
    hand_landmarker.task — MediaPipe hand landmark detection
"""

import math
import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BLOCK_SIZE = 10
ASCII_CHARS = " .,:;=+*#%@"
NUM_CHARS = len(ASCII_CHARS)

# Hand-based displacement
HAND_INFLUENCE_RADIUS = 120.0        # Pixels — how far hands push ASCII chars
HAND_DISPLACEMENT_STRENGTH = 5.0     # Max push in pixels
HAND_DAMPING = 0.35                  # Damping factor on displacement
MAX_OFFSET_PX = 5                    # Hard clamp

# Organic wave layered on top of hand displacement
WAVE_AMPLITUDE = 1.0
WAVE_SPATIAL_FREQ = 0.06
WAVE_TIME_FREQ = 3.0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FONT = cv2.FONT_HERSHEY_SIMPLEX
BASE_FONT_SCALE = 0.35
FONT_THICKNESS = 1
FONT_COLOR = (255, 255, 255)

# Web (teia) aesthetics
WEB_BASE_COLOR = np.array([255, 255, 0], dtype=np.float64)   # Cyan (BGR)
WEB_PULSE_COLOR = np.array([255, 100, 50], dtype=np.float64) # Bright blue pulse
WEB_THICKNESS = 2
WEB_PULSE_SPEED = 4.0               # Pulse oscillation speed (Hz)

# Hand circle
HAND_CIRCLE_COLOR = (200, 200, 200)
HAND_CIRCLE_THICKNESS = 1

# MediaPipe fingertip landmark indices
FINGERTIP_IDS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

# Finger anatomy: (TIP, PIP) pairs for curl detection (excluding thumb — unreliable)
# A finger is "curled" when its tip is closer to the wrist than its PIP joint.
FINGER_TIP_PIP = [(8, 6), (12, 10), (16, 14), (20, 18)]
FIST_CURL_THRESHOLD = 3             # Min curled fingers (out of 4) to count as fist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model(path, url):
    """Download a MediaPipe model if not already present."""
    if not os.path.exists(path):
        name = os.path.basename(path)
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, path)
        print(f"{name} downloaded.")


# ──────────────────────────────────────────────
# ASCII Processor
# ──────────────────────────────────────────────

class ASCIIProcessor:
    """Converts grayscale frames into ASCII art with hand-proximity displacement."""

    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.grid_w = frame_w // BLOCK_SIZE
        self.grid_h = frame_h // BLOCK_SIZE

        # Precompute pixel-space center of each grid cell
        cy, cx = np.mgrid[0:self.grid_h, 0:self.grid_w]
        self.base_x = (cx * BLOCK_SIZE + BLOCK_SIZE // 2).astype(np.float64)
        self.base_y = (cy * BLOCK_SIZE + BLOCK_SIZE // 2).astype(np.float64)

    def build_char_grid(self, gray):
        """Map each block's mean brightness → ASCII char index (vectorized)."""
        cropped = gray[:self.grid_h * BLOCK_SIZE, :self.grid_w * BLOCK_SIZE]
        blocks = cropped.reshape(self.grid_h, BLOCK_SIZE, self.grid_w, BLOCK_SIZE)
        means = blocks.mean(axis=(1, 3))
        return np.clip((means / 255.0 * (NUM_CHARS - 1)).astype(int), 0, NUM_CHARS - 1)

    def compute_hand_displacement(self, hand_points, t):
        """Compute displacement where each hand landmark pushes nearby ASCII chars.

        Displacement math (per hand landmark):
        ───────────────────────────────────────
        1. For each grid cell, compute distance to every hand landmark.
        2. Within HAND_INFLUENCE_RADIUS, apply a radial push: the character
           gets pushed AWAY from the hand landmark.
        3. Strength falls off with a smooth inverse-square curve:
              influence = (1 - dist/radius)^2  (clamped to 0 outside radius)
           This creates a soft, natural force field — strong at the center,
           fading smoothly to zero at the edge.
        4. The landmark's Z-axis modulates strength: closer hand (smaller z,
           negated to positive) = stronger push. This adds depth realism.
        5. A subtle sine wave is layered on top, modulated by total influence,
           so only the area near hands undulates.
        6. Final offset is hard-clamped to ±MAX_OFFSET_PX.

        Args:
            hand_points: list of (px, py, z) in pixel coords for all landmarks
            t: current time in seconds
        Returns:
            dx, dy: displacement arrays (grid_h, grid_w)
            influence: normalized influence map (0..1) for rendering modulation
        """
        dx = np.zeros((self.grid_h, self.grid_w), dtype=np.float64)
        dy = np.zeros((self.grid_h, self.grid_w), dtype=np.float64)
        total_influence = np.zeros((self.grid_h, self.grid_w), dtype=np.float64)

        if not hand_points:
            return dx, dy, total_influence

        for (px, py, z) in hand_points:
            # Distance from this landmark to every grid cell center
            dist_x = self.base_x - px
            dist_y = self.base_y - py
            dist = np.sqrt(dist_x ** 2 + dist_y ** 2) + 1e-6

            # Smooth falloff: (1 - d/R)^2, zero outside radius
            falloff = np.maximum(1.0 - dist / HAND_INFLUENCE_RADIUS, 0.0) ** 2

            # Z modulation: closer hand (z negated, so positive = close) adds strength
            z_factor = max(0.2, min(1.0, 0.5 - z * 5.0))

            strength = HAND_DISPLACEMENT_STRENGTH * HAND_DAMPING * falloff * z_factor

            # Radial push: unit vector away from landmark × strength
            dx += (dist_x / dist) * strength
            dy += (dist_y / dist) * strength
            total_influence += falloff * z_factor

        # Organic sine wave layered on influenced areas
        dx += WAVE_AMPLITUDE * np.sin(WAVE_SPATIAL_FREQ * self.base_y + WAVE_TIME_FREQ * t) * total_influence
        dy += WAVE_AMPLITUDE * np.cos(WAVE_SPATIAL_FREQ * self.base_x + WAVE_TIME_FREQ * t + 0.7) * total_influence

        # Hard clamp
        dx = np.clip(dx, -MAX_OFFSET_PX, MAX_OFFSET_PX)
        dy = np.clip(dy, -MAX_OFFSET_PX, MAX_OFFSET_PX)

        # Normalize influence for rendering use
        inf_max = total_influence.max()
        if inf_max > 1e-6:
            total_influence = np.clip(total_influence / inf_max, 0.0, 1.0)

        return dx, dy, total_influence

    def render(self, char_indices, dx, dy, influence):
        """Draw displaced ASCII characters on a black canvas."""
        canvas = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)

        for r in range(self.grid_h):
            for c in range(self.grid_w):
                idx = char_indices[r, c]
                if idx == 0:
                    continue

                draw_x = int(self.base_x[r, c] + dx[r, c])
                draw_y = int(self.base_y[r, c] + dy[r, c])

                if 0 <= draw_x < self.frame_w and 0 <= draw_y < self.frame_h:
                    # Chars near hands are slightly larger
                    inf = influence[r, c]
                    scale = BASE_FONT_SCALE * (1.0 + 0.2 * inf)
                    cv2.putText(
                        canvas, ASCII_CHARS[idx], (draw_x, draw_y),
                        FONT, scale, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA,
                    )

        return canvas


# ──────────────────────────────────────────────
# Hand Tracker
# ──────────────────────────────────────────────

class HandTracker:
    """Detects hands via MediaPipe HandLandmarker.
    Returns structured data: fingertips, all landmarks, handedness."""

    def __init__(self):
        ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL)

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb_frame, timestamp_ms):
        """Return (hands_data, all_landmark_points).

        hands_data: dict mapping handedness ("Left"/"Right") to a dict with:
            - "fingertips": list of 5 (px, py) for each fingertip
            - "landmarks": list of 21 (px, py, z) for all landmarks
            - "center": (px, py) palm center
            - "radius": int adaptive radius

        all_landmark_points: flat list of (px, py, z) for displacement computation
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return self._parse_result(result, rgb_frame.shape)

    @staticmethod
    def _is_hand_open(hand_lms):
        """Determine if a hand is open or closed (fist) using a curl heuristic.

        For each of the 4 fingers (index, middle, ring, pinky), we compare:
            dist(fingertip, wrist)  vs  dist(PIP joint, wrist)

        If the tip is CLOSER to the wrist than the PIP joint, the finger
        is curled inward. When 3+ fingers are curled → fist (closed).

        The thumb is excluded because its curl axis is perpendicular to
        the other fingers, making this distance metric unreliable for it.
        """
        wrist = hand_lms[0]
        curled = 0

        for tip_id, pip_id in FINGER_TIP_PIP:
            tip = hand_lms[tip_id]
            pip = hand_lms[pip_id]

            # Euclidean distance in normalized coords (cheaper than pixel conversion)
            dist_tip = math.sqrt((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2)
            dist_pip = math.sqrt((pip.x - wrist.x) ** 2 + (pip.y - wrist.y) ** 2)

            if dist_tip < dist_pip:
                curled += 1

        return curled < FIST_CURL_THRESHOLD

    def _parse_result(self, result, shape):
        h, w = shape[:2]
        hands_data = {}
        all_points = []

        if not result.hand_landmarks or not result.handedness:
            return hands_data, all_points

        for i, hand_lms in enumerate(result.hand_landmarks):
            # Handedness label (note: MediaPipe mirrors, so "Left" in result
            # means user's left hand when the image is flipped)
            label = result.handedness[i][0].category_name

            # All 21 landmarks in pixel coordinates
            landmarks_px = [
                (lm.x * w, lm.y * h, lm.z) for lm in hand_lms
            ]

            # Fist detection — open hands contribute to displacement and web
            hand_open = self._is_hand_open(hand_lms)

            # Only open-hand landmarks push the ASCII grid
            if hand_open:
                all_points.extend(landmarks_px)

            # 5 fingertips
            fingertips = [
                (int(hand_lms[fid].x * w), int(hand_lms[fid].y * h))
                for fid in FINGERTIP_IDS
            ]

            # Palm center (wrist 0 + middle MCP 9)
            cx = int((hand_lms[0].x + hand_lms[9].x) / 2.0 * w)
            cy = int((hand_lms[0].y + hand_lms[9].y) / 2.0 * h)

            # Adaptive radius from bounding box
            xs = [lm.x * w for lm in hand_lms]
            ys = [lm.y * h for lm in hand_lms]
            diag = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
            radius = max(int(diag * 0.55), 30)

            hands_data[label] = {
                "fingertips": fingertips,
                "landmarks": landmarks_px,
                "center": (cx, cy),
                "radius": radius,
                "is_open": hand_open,
            }

        return hands_data, all_points

    def close(self):
        self.landmarker.close()


# ──────────────────────────────────────────────
# Web Renderer (Teia Digital)
# ──────────────────────────────────────────────

class WebRenderer:
    """Draws pulsing web lines connecting corresponding fingertips
    between left and right hands."""

    def draw_web(self, canvas, hands_data, t):
        """Draw the web only when both hands are present.

        Pulse effect math:
        ──────────────────
        The web color oscillates between WEB_BASE_COLOR and WEB_PULSE_COLOR
        using a sine wave over time:
            blend = 0.5 + 0.5 * sin(2π * PULSE_SPEED * t)
            color = lerp(base, pulse, blend)

        Each of the 5 finger pairs also gets a slight phase offset
        (finger_index * 0.4 radians) so the pulse cascades across
        fingers like a traveling wave rather than all pulsing in unison.

        Additionally, small intermediate "energy nodes" are drawn along
        each web line to reinforce the digital aesthetic.
        """
        # Both hands must be present AND open — closed fist kills the web
        if "Left" not in hands_data or "Right" not in hands_data:
            return
        if not hands_data["Left"]["is_open"] or not hands_data["Right"]["is_open"]:
            return

        left_tips = hands_data["Left"]["fingertips"]
        right_tips = hands_data["Right"]["fingertips"]

        for i in range(5):
            pt_left = left_tips[i]
            pt_right = right_tips[i]

            # Per-finger phase offset for cascading RGB cycle
            phase = i * 0.4
            angle = 2.0 * math.pi * WEB_PULSE_SPEED * t + phase

            # RGB cycle: each channel is a sine wave offset by 120° (2π/3)
            r = int(127.5 + 127.5 * math.sin(angle))
            g = int(127.5 + 127.5 * math.sin(angle + 2.094))   # +2π/3
            b = int(127.5 + 127.5 * math.sin(angle + 4.189))   # +4π/3
            color_bgr = (b, g, r)

            # Draw the main web line
            cv2.line(canvas, pt_left, pt_right, color_bgr, WEB_THICKNESS, cv2.LINE_AA)

            # Energy nodes along the line (5 evenly spaced dots)
            for j in range(1, 6):
                frac = j / 6.0
                nx = int(pt_left[0] + (pt_right[0] - pt_left[0]) * frac)
                ny = int(pt_left[1] + (pt_right[1] - pt_left[1]) * frac)

                # Node size pulses with a different phase
                node_pulse = 0.5 + 0.5 * math.sin(
                    2.0 * math.pi * WEB_PULSE_SPEED * t + phase + j * 0.8
                )
                node_radius = int(2 + 3 * node_pulse)
                cv2.circle(canvas, (nx, ny), node_radius, color_bgr, -1, cv2.LINE_AA)

    def draw_hand_circles(self, canvas, hands_data):
        """Draw subtle circles around each detected hand."""
        for _, data in hands_data.items():
            cx, cy = data["center"]
            radius = data["radius"]
            cv2.circle(
                canvas, (cx, cy), radius,
                HAND_CIRCLE_COLOR, HAND_CIRCLE_THICKNESS, cv2.LINE_AA,
            )


# ──────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────

class MainApp:
    """Ties together camera, ASCII rendering, hand tracking, and digital web."""

    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam. Check camera connection.")

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.ascii = ASCIIProcessor(w, h)
        self.hands = HandTracker()
        self.web = WebRenderer()

        self.frame_count = 0
        self.timestamp_ms = 0
        self.ascii_mode = True           # Toggle with 'w' key

    def run(self):
        """Main loop: capture → hands → displacement → ASCII → web → display."""
        prev_time = time.time()
        fps = 0.0
        window = "Digital Web — ASCII"

        print("Digital Web — ASCII. Press 'q' to quit, 'w' to toggle ASCII mode.")
        print("Show both hands to activate the web effect!")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame. Exiting.")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.timestamp_ms += 33
            t = time.time()

            # ── Hand detection ──
            hands_data, all_points = self.hands.detect(rgb, self.timestamp_ms)

            if self.ascii_mode:
                # ── Full ASCII pipeline ──
                char_indices = self.ascii.build_char_grid(gray)
                dx, dy, influence = self.ascii.compute_hand_displacement(all_points, t)
                canvas = self.ascii.render(char_indices, dx, dy, influence)
            else:
                # ── Raw webcam feed ──
                canvas = frame.copy()

            # ── Draw hand circles + web (on top of either mode) ──
            self.web.draw_hand_circles(canvas, hands_data)
            self.web.draw_web(canvas, hands_data, t)

            # ── FPS ──
            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_time = now

            cv2.putText(
                canvas, f"FPS: {fps:.1f}", (10, 25),
                FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )

            # Status indicator
            mode_tag = "[ASCII]" if self.ascii_mode else "[CAM]"
            open_hands = [l for l, d in hands_data.items() if d["is_open"]]
            closed_hands = [l for l, d in hands_data.items() if not d["is_open"]]
            if len(open_hands) >= 2:
                web_tag = "Web active!"
            elif closed_hands:
                fist_names = " + ".join(closed_hands)
                web_tag = f"Fist: {fist_names}"
            else:
                web_tag = "Show both hands for web"
            status = f"{mode_tag} {web_tag}  |  'w' toggle  'q' quit"
            cv2.putText(
                canvas, status, (10, self.ascii.frame_h - 15),
                FONT, 0.5, (100, 100, 100), 1, cv2.LINE_AA,
            )

            cv2.imshow(window, canvas)
            self.frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("w"), ord("W")):
                self.ascii_mode = not self.ascii_mode

        self._cleanup()

    def _cleanup(self):
        self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()


def main():
    app = MainApp(camera_index=0)
    app.run()


if __name__ == "__main__":
    main()
