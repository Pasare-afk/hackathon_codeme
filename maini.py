"""
Complete demo (simplified face lines):
- MediaPipe hands with colored fingers
- Simple black-line face (like original)
- Motion detection (frame differencing)
- Hand motion / gestures -> subtitles (wave, swipe, point, pinch)
- Subtitle manager with timing + fade + translucent box
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, defaultdict
import textwrap

# ------------------ Config ------------------
CAM_IDX = 1  # change to 0 or 1 depending on your webcam
MAX_HANDS = 2
MP_MIN_DET_CONF = 0.5
FPS_SMOOTH = 0.9

# Gesture thresholds (tune to your setup)
MOTION_SCORE_THRESHOLD = 4000        # for frame diff
SWIPE_SPEED_THRESHOLD = 30.0         # pixels/frame
WAVE_OSCILLATION_COUNT = 4          # how many direction changes constitute a wave
WAVE_MIN_AMPLITUDE = 20             # min px amplitude for wave
PINCH_DIST_THRESHOLD = 35           # pixels (thumb tip to index tip)
POINT_MIN_EXTEND = 30               # distance index tip to pip to detect pointing
GESTURE_COOLDOWN = 1.2              # seconds between same gesture triggers per hand

# ------------------ MediaPipe init ------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=MAX_HANDS, min_detection_confidence=MP_MIN_DET_CONF)
mp_draw = mp.solutions.drawing_utils

# ------------------ Colors & finger mapping ------------------
finger_colors = {
    'thumb': (0, 0, 255),    # red
    'index': (0, 255, 0),    # green
    'middle': (255, 0, 0),   # blue
    'ring': (0, 255, 255),   # yellow
    'pinky': (255, 0, 255)   # magenta
}
fingers = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20]
}

# ------------------ Subtitle manager ------------------
class SubtitleManager:
    def __init__(self, max_lines=3, font=cv2.FONT_HERSHEY_SIMPLEX):
        self.queue = deque()  # elements = (text, start_time, duration)
        self.font = font
        self.max_lines = max_lines

    def add(self, text, duration=3.0):
        now = time.time()
        if self.queue and self.queue[-1][0] == text and now - self.queue[-1][1] < 0.8:
            t, s, d = self.queue[-1]
            self.queue[-1] = (t, s, max(d, duration))
            return
        self.queue.append((text, now, duration))

    def draw(self, img):
        now = time.time()
        # remove expired
        while self.queue and now - self.queue[0][1] > self.queue[0][2]:
            self.queue.popleft()
        if not self.queue:
            return img

        visible = list(self.queue)[-self.max_lines:]
        joined = " | ".join([t for (t, s, d) in visible])
        wrapped = textwrap.wrap(joined, width=40)
        h, w = img.shape[:2]
        pad = 12
        line_h = 28
        box_h = line_h * len(wrapped) + 2 * pad
        box_y0 = h - box_h - 35
        overlay = img.copy()
        cv2.rectangle(overlay, (20, box_y0), (w - 20, h - 20), (0, 0, 0), -1)
        alpha = 0.45
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        y = box_y0 + pad + 20
        for line in wrapped:
            cv2.putText(img, line, (35, y), self.font, 0.7, (255,255,255), 2, cv2.LINE_AA)
            y += line_h
        return img

subtitle_mgr = SubtitleManager()

# ------------------ Motion detection (frame differencing) ------------------
prev_gray_small = None
def detect_motion(frame, downscale=0.5):
    global prev_gray_small
    small = cv2.resize(frame, (0,0), fx=downscale, fy=downscale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    if prev_gray_small is None:
        prev_gray_small = gray
        return 0
    diff = cv2.absdiff(prev_gray_small, gray)
    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_score = int(np.sum(th) / 255)
    prev_gray_small = gray
    return motion_score

# ------------------ Gesture tracking state ------------------
hand_state = defaultdict(lambda: {
    'wrist_deque': deque(maxlen=12),
    'prev_wrist': None,
    'last_gesture_time': {},
    'smoothed_wrist': None
})

def now_s():
    return time.time()

# ------------------ Utilities ------------------
def euclid(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

def get_label_from_handedness(handedness_proto):
    if not handedness_proto:
        return "Unknown"
    label = handedness_proto.classification[0].label
    return label  # 'Left' or 'Right'

# ------------------ Simple black-line face drawing ------------------
def draw_simple_face(canvas):
    h, w, _ = canvas.shape
    # place face near the top center (similar to original)
    cx, cy = w // 2, h // 5
    # eyes
    eye_offset_x = 40
    eye_offset_y = -10
    eye_r = 12
    cv2.circle(canvas, (cx - eye_offset_x, cy + eye_offset_y), eye_r, (0,0,0), -1)
    cv2.circle(canvas, (cx + eye_offset_x, cy + eye_offset_y), eye_r, (0,0,0), -1)
    # nose as small triangle
    nose_pts = np.array([
        [cx, cy + 2],
        [cx - 10, cy + 28],
        [cx + 10, cy + 28]
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [nose_pts], (0,0,0))
    # mouth (arc)
    cv2.ellipse(canvas, (cx, cy + 55), (35, 15), 0, 0, 180, (0,0,0), 3, cv2.LINE_AA)
    # optional simple eyebrows
    cv2.line(canvas, (cx - eye_offset_x - 18, cy + eye_offset_y - 18), (cx - eye_offset_x + 8, cy + eye_offset_y - 20), (0,0,0), 2, cv2.LINE_AA)
    cv2.line(canvas, (cx + eye_offset_x - 8, cy + eye_offset_y - 20), (cx + eye_offset_x + 18, cy + eye_offset_y - 18), (0,0,0), 2, cv2.LINE_AA)
    return canvas

# ------------------ Main loop ------------------
cap = cv2.VideoCapture(CAM_IDX)
if not cap.isOpened():
    print(f"Cannot open camera index {CAM_IDX}. Try changing CAM_IDX.")
    raise SystemExit

prev_time = time.time()
display_fps = 0.0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Camera read failed.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # white canvas
        canvas = 255 * np.ones((h, w, 3), dtype=np.uint8)

        # draw simple black-line face
        canvas = draw_simple_face(canvas)

        # motion detection (downscaled)
        motion_score = detect_motion(frame, downscale=0.5)
        if motion_score > MOTION_SCORE_THRESHOLD:
            subtitle_mgr.add("Motion detected", duration=1.2)

        # MediaPipe hands
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                handedness = None
                if result.multi_handedness and idx < len(result.multi_handedness):
                    handedness = result.multi_handedness[idx]

                # landmarks -> pixels
                lm = hand_landmarks.landmark
                pts = [(int(p.x * w), int(p.y * h)) for p in lm]

                # draw black skeleton first
                mp_draw.draw_landmarks(
                    canvas,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0,0,0), thickness=2)
                )

                # color each finger
                for name, idxs in fingers.items():
                    color = finger_colors.get(name, (255,255,255))
                    for i in range(len(idxs) - 1):
                        cv2.line(canvas, pts[idxs[i]], pts[idxs[i+1]], color, 4)
                    for i in idxs:
                        cv2.circle(canvas, pts[i], 6, color, -1)

                # ---- Gesture detection ----
                label = get_label_from_handedness(handedness)
                key = f"{label}_{idx}"
                st = hand_state[key]

                wrist = pts[0]  # wrist index 0
                # smoothing wrist
                if st['smoothed_wrist'] is None:
                    st['smoothed_wrist'] = wrist
                else:
                    sx = int(0.75 * st['smoothed_wrist'][0] + 0.25 * wrist[0])
                    sy = int(0.75 * st['smoothed_wrist'][1] + 0.25 * wrist[1])
                    st['smoothed_wrist'] = (sx, sy)

                st['wrist_deque'].append(st['smoothed_wrist'])

                # velocity
                if st['prev_wrist'] is None:
                    st['prev_wrist'] = st['smoothed_wrist']
                dx = st['smoothed_wrist'][0] - st['prev_wrist'][0]
                dy = st['smoothed_wrist'][1] - st['prev_wrist'][1]
                speed = euclid(st['smoothed_wrist'], st['prev_wrist'])

                # Swipe detection
                if speed > SWIPE_SPEED_THRESHOLD:
                    if abs(dx) > abs(dy):
                        dir = "right" if dx > 0 else "left"
                        last = st['last_gesture_time'].get(f"swipe_{dir}", 0)
                        if now_s() - last > GESTURE_COOLDOWN:
                            subtitle_mgr.add(f"Swipe {dir}", duration=1.2)
                            st['last_gesture_time'][f"swipe_{dir}"] = now_s()
                    else:
                        dir = "down" if dy > 0 else "up"
                        last = st['last_gesture_time'].get(f"swipe_{dir}", 0)
                        if now_s() - last > GESTURE_COOLDOWN:
                            subtitle_mgr.add(f"Swipe {dir}", duration=1.2)
                            st['last_gesture_time'][f"swipe_{dir}"] = now_s()

                # Wave detection
                if len(st['wrist_deque']) >= st['wrist_deque'].maxlen:
                    xs = [p[0] for p in st['wrist_deque']]
                    diffs = np.sign(np.diff(xs))
                    if len(diffs) >= 3:
                        changes = np.sum(diffs[:-1] != diffs[1:])
                    else:
                        changes = 0
                    amplitude = (max(xs) - min(xs)) if xs else 0
                    if changes >= WAVE_OSCILLATION_COUNT and amplitude > WAVE_MIN_AMPLITUDE:
                        last_wave = st['last_gesture_time'].get("wave", 0)
                        if now_s() - last_wave > GESTURE_COOLDOWN:
                            subtitle_mgr.add("Wave", duration=1.5)
                            st['last_gesture_time']['wave'] = now_s()

                # Pinch detection (thumb tip 4, index tip 8)
                thumb_tip = pts[4]
                index_tip = pts[8]
                thumb_index_dist = euclid(thumb_tip, index_tip)
                if thumb_index_dist < PINCH_DIST_THRESHOLD:
                    last_pin = st['last_gesture_time'].get("pinch", 0)
                    if now_s() - last_pin > GESTURE_COOLDOWN:
                        subtitle_mgr.add("Pinch/Grab", duration=1.5)
                        st['last_gesture_time']['pinch'] = now_s()

                # Point detection (index tip vs pip)
                index_pip = pts[6]
                index_ext_dist = euclid(index_tip, index_pip)
                other_tips = [pts[i] for i in (4, 12, 16, 20) if i != 8]
                other_close = sum(1 for ot in other_tips if euclid(ot, pts[0]) < 70)
                if index_ext_dist > POINT_MIN_EXTEND and other_close >= 3:
                    last_point = st['last_gesture_time'].get("point", 0)
                    if now_s() - last_point > GESTURE_COOLDOWN:
                        subtitle_mgr.add("Pointing", duration=1.4)
                        st['last_gesture_time']['point'] = now_s()

                st['prev_wrist'] = st['smoothed_wrist']

        # Draw subtitles
        canvas = subtitle_mgr.draw(canvas)

        # FPS display
        cur_time = time.time()
        dt = cur_time - prev_time
        prev_time = cur_time
        if dt > 0:
            display_fps = display_fps * FPS_SMOOTH + (1.0 / dt) * (1 - FPS_SMOOTH)
        cv2.putText(canvas, f"FPS: {int(display_fps)}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30,30,30), 2)

        cv2.imshow("Hands + Simple Face + Subtitles", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
