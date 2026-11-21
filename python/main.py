"""
ANTI MICROSLEEP - camera part (Linux side of the Arduino UNO Q)

Pipeline per frame:
  USB camera -> face mesh (478 points, MediaPipe models run with LiteRT, see face_engine.py)
             -> EAR  (eye aspect ratio)   : eyes open / closed
             -> PERCLOS (% eyes closed over last 60 s)
             -> MAR  (mouth aspect ratio)  : yawning
             -> brightness / face found    : camera confidence
  => alarm level 0/1/2 + status code, sent to the MCU sketch via Bridge.
"""

import base64
import hashlib
import hmac
import http.client
import json
import os
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import wave
from collections import deque

import numpy as np
import cv2
from fastapi import Body, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from arduino.app_utils import App, Bridge, Logger
from arduino.app_peripherals.camera import Camera
from arduino.app_peripherals.speaker import Speaker
from arduino.app_bricks.web_ui import WebUI

from face_engine import FaceMeshEngine
from nav import Navigator

logger = Logger("anti-microsleep")

# --------------------------------------------------------------------------
# Settings (tune these during testing)
# --------------------------------------------------------------------------
CAMERA_SOURCE = "usb:0"       # first USB webcam. Other options: None (auto), "ws://0.0.0.0:8080"
RESOLUTION = (640, 480)
FPS = 15

# Eye threshold is learned automatically in the background (no calibration step):
EAR_DEFAULT_THRESHOLD = 0.12  # "clearly closed" default until the driver's eyes are learned
EAR_DEFAULT_NARROW = 0.17     # heavy-eyelid default until learned
EAR_DEFAULT_SQUINT = 0.21   # squint zone upper bound default
EAR_THRESHOLD_RATIO = 0.45    # EAR < baseline*0.45 -> clearly closed
EAR_SQUINT_RATIO = 0.78       # baseline*0.45..0.78 (glasses: closed eyes still read ~0.6-0.75 x open)
NARROW_RATIO = 0.65           # "heavy eyelids" warning: EAR below baseline*0.65 for NARROW_WARN_S -> squint zone: closed only if no iris visible
IRIS_DARK_MIN = 0.25          # eye middle >= 25 % darker than its corners = iris visible
SMILE_RATIO = 1.10            # mouth width >= 110 % of the driver's normal = smiling/laughing
NARROW_WARN_S = 4.0          # eyes narrow (squint zone or closed) this long -> WARNING
LEARN_WINDOW_S = 120.0        # baseline = 75th percentile of the last 2 min of frontal EAR
LEARN_MIN_S = 10.0           # need this much face time before the first update
LEARN_EVERY_S = 2.0
BASELINE_DROP_PER_MIN = 0.03  # baseline may fall only slowly (drowsy eyes must not lower it)
NEW_DRIVER_NO_FACE_S = 30.0   # no face this long -> forget baseline (driver may change)
EYES_CLOSED_WARN_S = 1.0      # eyes closed this long -> WARNING
EYES_CLOSED_ALARM_S = 2.0     # eyes closed this long -> ALARM (microsleep)
SQUINT_CLOSED_WARN_S = 1.5    # closure seen only as "narrow + no iris" (glasses / squint):
SQUINT_CLOSED_ALARM_S = 3.0   # ...warn / alarm a bit later, so a squint is not a microsleep
SQUINT_EXCUSE_MAX_S = 8.0     # a closure that looks like a hard squint (squint_check.py) is excused
                              # this long at most; after that it counts as closed again (safety net)
REOPEN_MARGIN = 1.00          # a closure ends only when EAR rises above the squint threshold
HOLD_AFTER_S = 0.5            # closures longer than this also ride over short "open" flickers
REOPEN_GRACE_S = 0.3          # ...eyes must look open this long to end the closure

PERCLOS_WINDOW_S = 60.0
PERCLOS_MIN_DATA_S = 20.0
PERCLOS_WARN = 0.20
PERCLOS_ALARM = 0.35          # "very drowsy" note (still WARNING level, see _decide_level)
PERCLOS_MAX_EPISODE_S = 1.0   # one eye closure adds at most 1 s to PERCLOS; longer closures
                              # already trigger the microsleep alarm and must not keep the
                              # warning on for a minute after the eyes reopen

MAR_YAWN = 0.55               # mouth open ratio counted as yawn
YAWN_MIN_S = 1.2
YAWN_WINDOW_S = 300.0
YAWNS_WARN = 3

CLOSED_WINDOW_S = 2.0         # short window to ride over EAR flicker
CLOSED_FRAC_ALARM = 0.80      # eyes closed in >= 80 % of the last 2 s -> microsleep ALARM
HEAD_EYES_FRAC = 0.50         # head down/tilted + eyes closed >= 50 % of last 2 s -> ALARM
HEAD_EYES_MIN_S = 1.5         # ...once the head has been down at least this long
HEAD_DOWN_DEG = 18.0         # head pitched down this much below the driver's normal pose
HEAD_TILT_DEG = 25.0          # head tilted sideways (roll) this much vs normal pose
HEAD_BACK_DEG = 12.0          # head fallen BACK (chin up, asleep against the seat) this much vs normal
YAW_POSE_MAX = 0.75           # with the head dropped/back/tilted, eyes are still judged up to this yaw
                              # (a fallen head distorts the yaw estimate; never stop watching the eyes then)
HEAD_WARN_S = 2.0             # head down / tilted this long -> WARNING
HEAD_ALARM_S = 5.0            # head down / tilted this long -> ALARM (nodding off)
DROP_WARN_S = 1.0             # face vanished right after the head went down -> WARNING
DROP_ALARM_S = 3.0            # ... and is still gone after this long -> ALARM
PERSON_EVERY_S = 0.5          # person detector (YOLOX) runs only while no face is seen
PERSON_MIN_CONF = 0.45
PERSON_MIN_AREA = 0.15        # person box >= 15 % of the frame = sitting right in front
PERSON_FRESH_S = 1.5          # a person result is trusted this long
DROP_PITCH_RISE = 8.0        # head pitched down by this much in the last 2 s before losing the face
DROP_FACE_SLIDE = 0.06        # or the face slid down by 6 % of the frame height
PITCH_DOWN_SIGN = 1.0        # +1 if pitch grows when the head goes down (checked on device)
YAW_MAX = 0.35                # |yaw| above this = head turned (looking-away timer)
YAW_EYES_MAX = 0.50           # eyes are still judged up to this |yaw| (yaw estimate is jumpy)
YAW_LEARN_MAX = 0.15          # learning only uses frames facing the driver's normal direction
YAW_BASE_MAX = 0.55           # camera may sit off to the side: the driver's normal yaw is learned
                              # (median of the last 2 min) and yaw is judged relative to it, up to this
YAW_BASE_MIN_FRAMES = 20      # face frames needed before the learned normal yaw is used (~1.5 s)
HAAR_CROP_W = 96              # eye crop width for the open-eye image detector
HAAR_MIN_SAMPLES = 20         # open-eye checks needed before trusting the detector for a driver
HAAR_RELIABLE = 0.75          # detector must find this driver's open eyes >= 75 % of the time
HAAR_LEARN_EVERY_S = 0.3
HAAR_DOUBT_RATIO = 0.88       # EAR below baseline*0.88 (and not clearly shut) -> ask the detector
LOOK_AWAY_WARN_S = 4.0        # head turned away this long -> WARNING (distraction)
LOOK_AWAY_ALARM_S = 10.0      # ... this long -> ALARM (eyes off the road far too long, or asleep turned)
NO_FACE_WARN_S = 4.0         # good light but no face this long -> WARNING
LOW_LIGHT_BRIGHTNESS = 35     # mean gray level below this = camera "blind"

LEVEL_HOLD_S = 1.5            # keep an alarm at least this long (no flicker)
HEARTBEAT_S = 1.0             # resend state to MCU at least this often

# Status codes shown on the MCU (LED4 colour)
ST_OK, ST_CALIBRATING, ST_NO_CAMERA, ST_NO_FACE, ST_LOW_LIGHT = range(5)
STATUS_NAMES = ["OK", "CALIBRATING", "NO_CAMERA", "NO_FACE", "LOW_LIGHT"]
LEVEL_NAMES = ["OK", "WARNING", "ALARM"]

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(APP_DIR, "models", "face_landmarker.task")
# Roboflow cloud model "eye_close_project/2" = second opinion on eyes open/closed.
# API key: put it in <app>/roboflow_api_key.txt (or env ROBOFLOW_API_KEY). No key = EAR only.
ROBOFLOW_HOST = "serverless.roboflow.com"
ROBOFLOW_PATH = "/eye_close_project/2"
ROBOFLOW_URL = f"https://{ROBOFLOW_HOST}{ROBOFLOW_PATH}"
ROBOFLOW_KEY_FILE = os.path.join(APP_DIR, "roboflow_api_key.txt")
ROBOFLOW_URGENT_EVERY_S = 0.7  # while EAR says "closed": ask often to confirm / veto
ROBOFLOW_IDLE_EVERY_S = 3.0    # otherwise: occasional check (borderline EAR, stats)
ROBOFLOW_TEST_MODE = False     # True: log every answer + 1 request/s, to evaluate the model
SQUINT_TEST_LOG = False         # TEMP: log EAR + iris value in the squint zone (tuning)
ROBOFLOW_MIN_CONF = 0.45
ROBOFLOW_VERDICT_TTL_S = 2.5   # trust a verdict for this long after its frame was taken
ROBOFLOW_TIMEOUT_S = 3.0
AI_VETO_MAX_S = 1.5            # AI "open" may cancel an EAR closure only during its first 1.5 s
EAR_BORDERLINE = 1.20         # EAR < threshold*1.2 + model says closed -> closed
ALARM_SOUND = os.path.join(APP_DIR, "sounds", "beep.wav")  # looped only while level = ALARM
READY_SOUND = os.path.join(APP_DIR, "sounds", "ready.wav")      # once: camera + face AI running
LEARNED_SOUND = os.path.join(APP_DIR, "sounds", "learned.wav")  # once each time a driver's eyes are learned
SPEAKER_DEVICE = Speaker.USB_SPEAKER_1
LIVE_JPEG_QUALITY = 70       # live view at http://<board-ip>:7000
DEBUG_SNAPSHOT = False        # also save debug/frame.jpg every 2 s
DEBUG_DIR = os.path.join(APP_DIR, "debug")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# MediaPipe face-mesh landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]   # corner, top, top, corner, bottom, bottom
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 81, 178, 13, 14, 311, 402]  # 2 corners + 3 vertical pairs
NOSE_TIP, FACE_LEFT, FACE_RIGHT = 1, 234, 454
# full eye outlines (16 points each, in order around the lid margin)
RIGHT_EYE_OUTLINE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_OUTLINE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def _d(a, b):
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts, idx):
    p = pts[idx]
    horiz = _d(p[0], p[3])
    if horiz < 1e-6:
        return 0.0
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2.0 * horiz)


# generic 3D face model (mm) for head pose: nose tip, chin, eye outer corners, mouth corners
_HEAD_MODEL = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
                        (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
                        (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], dtype=np.float64)
_HEAD_IDX = [1, 152, 33, 263, 61, 291]


def head_pose(pts, w, h):
    """Returns (pitch_deg, roll_deg). pitch > 0 = nose pointing down (head dropped).
    solvePnP starts from 'facing the camera' so it can't flip to the mirror solution,
    and pitch comes from the nose direction (no Euler-angle wrap-around)."""
    img_pts = pts[_HEAD_IDX].astype(np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    rvec0 = np.array([[np.pi], [0.0], [0.0]])          # model y-up -> image y-down, facing camera
    tvec0 = np.array([[0.0], [0.0], [2000.0]])
    ok, rvec, _ = cv2.solvePnP(_HEAD_MODEL, img_pts, cam, np.zeros(4), rvec0, tvec0,
                               useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    pitch = None
    if ok:
        R, _ = cv2.Rodrigues(rvec)
        nose_dir = R @ np.array([0.0, 0.0, 1.0])        # (0,0,-1) when facing the camera
        if nose_dir[2] < 0:                             # sane: face points toward camera
            pitch = float(np.degrees(np.arcsin(np.clip(nose_dir[1], -1.0, 1.0))))
    dx, dy = pts[263] - pts[33]
    roll = float(np.degrees(np.arctan2(dy, dx)))
    return pitch, roll


def iris_visible_fraction(gray, pts):
    """How much darker the MIDDLE of the eye is than its corners (0 = no difference).
    Open / squinting eye: the iris is a dark spot in the middle -> high value.
    Closed eye: the lash line is a thin dark line across the whole width -> low value."""
    scores = []
    for outline, (c0, c1) in ((RIGHT_EYE_OUTLINE, (33, 133)), (LEFT_EYE_OUTLINE, (362, 263))):
        # straighten the eye (corner to corner horizontal) and cut a small patch around it
        p0, p1 = pts[c0], pts[c1]
        width = float(np.linalg.norm(p1 - p0))
        if width < 8:
            continue
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        cx, cy = (p0 + p1) / 2
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(ang), 1.0)
        pw, ph = int(width * 1.1), max(6, int(width * 0.5))
        M[0, 2] += pw / 2 - cx
        M[1, 2] += ph / 2 - cy
        patch = cv2.warpAffine(gray, M, (pw, ph), flags=cv2.INTER_LINEAR)
        poly = cv2.transform(pts[outline].reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
        mask = np.zeros((ph, pw), np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        cols = [patch[:, i][mask[:, i] > 0] for i in range(pw)]
        means = np.array([c.mean() if c.size else np.nan for c in cols])
        valid = ~np.isnan(means)
        if valid.sum() < 6:
            continue
        idx = np.where(valid)[0]
        span = idx[-1] - idx[0] + 1
        mid = means[idx[0] + int(span * 0.3): idx[0] + int(span * 0.7) + 1]
        edges = np.concatenate([means[idx[0]: idx[0] + int(span * 0.2) + 1],
                                means[idx[-1] - int(span * 0.2): idx[-1] + 1]])
        mid, edges = mid[~np.isnan(mid)], edges[~np.isnan(edges)]
        if mid.size == 0 or edges.size == 0:
            continue
        ref = float(np.median(edges))
        scores.append(max(0.0, (ref - float(np.min(mid))) / max(ref, 1.0)))
    return max(scores) if scores else None


_EYE_CASCADE = None


def open_eyes_seen(gray, pts):
    """Image check that does not trust the eyelid landmarks (they stay "open" behind glasses):
    OpenCV's open-eye detector for glasses runs on a crop around each eye's corners. An open
    eye is usually found, a closed eye usually not. Returns the fraction of eyes found (0..1),
    or None if the face is too small to check."""
    global _EYE_CASCADE
    if _EYE_CASCADE is None:
        _EYE_CASCADE = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_eye_tree_eyeglasses.xml"))
    if _EYE_CASCADE.empty():
        return None
    H, W = gray.shape[:2]
    tried = found = 0
    for c0, c1 in ((33, 133), (362, 263)):
        p0, p1 = pts[c0], pts[c1]
        w = float(np.linalg.norm(p1 - p0))
        if w < 10:
            continue
        cx, cy = (p0 + p1) / 2
        x0, x1 = max(0, int(cx - w)), min(W, int(cx + w))
        y0, y1 = max(0, int(cy - w * 0.8)), min(H, int(cy + w * 0.8))
        if x1 - x0 < 10 or y1 - y0 < 10:
            continue
        crop = gray[y0:y1, x0:x1]
        scale = HAAR_CROP_W / crop.shape[1]
        crop = cv2.equalizeHist(cv2.resize(crop, None, fx=scale, fy=scale,
                                           interpolation=cv2.INTER_LINEAR))
        ch, cw = crop.shape[:2]
        eyes = _EYE_CASCADE.detectMultiScale(crop, 1.1, 3,
                                             minSize=(int(cw * 0.3), int(cw * 0.3)))
        tried += 1
        ex, ey = (cx - x0) * scale, (cy - y0) * scale          # where the eye should be
        for (x, y, ew, eh) in eyes:
            if abs(x + ew / 2 - ex) < cw * 0.25 and abs(y + eh / 2 - ey) < ch * 0.3:
                found += 1
                break
    return found / tried if tried else None


def mouth_aspect_ratio(pts):
    p = pts[MOUTH]
    horiz = _d(p[0], p[1])
    if horiz < 1e-6:
        return 0.0
    return (_d(p[2], p[3]) + _d(p[4], p[5]) + _d(p[6], p[7])) / (3.0 * horiz)


# --------------------------------------------------------------------------
# Drowsiness logic
# --------------------------------------------------------------------------
class DrowsinessDetector:
    def __init__(self):
        self.learn_samples = deque()        # (t, ear) frontal frames
        self.pose_samples = deque()         # (t, pitch, roll) frontal frames
        self.recent_closed = deque()        # (t, closed) last 2 s
        self.narrow_since = None
        self.mouth_samples = deque()        # (t, mouth width / eye distance)
        self.base_mouth = None
        self.pitch_buf = deque(maxlen=5)
        self.yaw_buf = deque(maxlen=5)
        self.recent_face = deque()          # (t, pitch, face centre y / frame h), last 2 s
        self.last_head_down = False
        self.dropped = False                # face vanished downward = head dropped
        self.base_pitch = self.base_roll = None
        self.yaw_samples = deque()          # (t, raw yaw) all face frames, last 2 min
        self.base_yaw = 0.0                 # driver's normal yaw (camera off-centre)
        self.head_since = None
        self.baseline = None
        self.last_learn = 0.0
        self.ear_threshold = EAR_DEFAULT_THRESHOLD
        self.squint_threshold = EAR_DEFAULT_SQUINT
        self.narrow_threshold = EAR_DEFAULT_NARROW
        self.calibrated = False             # True once the driver's eyes are learned
        self.ear_buf = deque(maxlen=3)
        self.closed_since = None            # effective closure (after AI veto/confirm)
        self.ear_closed_since = None        # closure according to EAR only
        self.reopen_since = None            # eyes look open again since (hysteresis)
        self.squint_face_since = None       # closure excused as a hard squint since
        self.ep_min_ear = None              # lowest EAR in the current closure
        self.haar_open = deque(maxlen=60)   # open-eye detector results on clearly open eyes
        self.haar_learn_t = 0.0
        self.closed_hist = deque()          # (t, closed: bool)
        self.yawn_since = None
        self.yawn_counted = False
        self.yawn_times = deque()
        self.no_face_since = None
        self.away_since = None
        self.level = 0
        self.level_raised_at = 0.0

    def forget_driver(self):
        self.learn_samples.clear()
        self.pose_samples.clear()
        self.mouth_samples.clear()
        self.base_mouth = None
        self.base_pitch = self.base_roll = None
        self.yaw_samples.clear()
        self.base_yaw = 0.0
        self.baseline = None
        self.ear_threshold = EAR_DEFAULT_THRESHOLD
        self.squint_threshold = EAR_DEFAULT_SQUINT
        self.narrow_threshold = EAR_DEFAULT_NARROW
        self.calibrated = False
        self.haar_open.clear()
        if globals().get("squint_checker") is not None:
            squint_checker.reset()

    def haar_rate(self):
        """How often the open-eye detector finds this driver's open eyes (None = not known yet)."""
        if len(self.haar_open) < HAAR_MIN_SAMPLES:
            return None
        return float(np.mean(self.haar_open))

    def _relative_yaw(self, t, yaw):
        """Yaw relative to the driver's learned normal direction (camera may be off to the side)."""
        self.yaw_samples.append((t, yaw))
        while self.yaw_samples and t - self.yaw_samples[0][0] > LEARN_WINDOW_S:
            self.yaw_samples.popleft()
        if len(self.yaw_samples) >= YAW_BASE_MIN_FRAMES and t - self.last_learn >= LEARN_EVERY_S:
            # the driver looks at the road most of the time: the median is "facing forward"
            self.base_yaw = float(np.clip(np.median([y for _, y in self.yaw_samples]),
                                          -YAW_BASE_MAX, YAW_BASE_MAX))
        return yaw - self.base_yaw

    def recalibrate(self):
        """Reset button: forget learned eyes and clear PERCLOS / yawns."""
        self.forget_driver()
        self.closed_hist.clear()
        self.yawn_times.clear()
        logger.info("Reset: re-learning the driver's eyes in the background")

    def _learn(self, t, ear, yaw, pitch=None, roll=None, mouth=None):
        if abs(yaw) < YAW_LEARN_MAX:
            self.learn_samples.append((t, ear))
            if pitch is not None:
                self.pose_samples.append((t, pitch, roll))
            if mouth is not None:
                self.mouth_samples.append((t, mouth))
        while self.mouth_samples and t - self.mouth_samples[0][0] > LEARN_WINDOW_S:
            self.mouth_samples.popleft()
        if len(self.mouth_samples) >= 100 and t - self.last_learn >= LEARN_EVERY_S:
            # normal (relaxed) mouth width; 40th percentile so smiles don't raise it
            self.base_mouth = float(np.percentile([m for _, m in self.mouth_samples], 40))
        while self.learn_samples and t - self.learn_samples[0][0] > LEARN_WINDOW_S:
            self.learn_samples.popleft()
        while self.pose_samples and t - self.pose_samples[0][0] > LEARN_WINDOW_S:
            self.pose_samples.popleft()
        if len(self.pose_samples) >= 100 and t - self.last_learn >= LEARN_EVERY_S:
            # driver's normal head pose (camera may be mounted at an angle)
            self.base_pitch = float(np.median([p for _, p, _ in self.pose_samples]))
            self.base_roll = float(np.median([r for _, _, r in self.pose_samples]))
        if (not self.learn_samples or t - self.last_learn < LEARN_EVERY_S
                or t - self.learn_samples[0][0] < LEARN_MIN_S or len(self.learn_samples) < 100):
            return
        dt, self.last_learn = t - self.last_learn, t
        # 75th percentile ignores blinks / short eye closures
        new = float(np.percentile([e for _, e in self.learn_samples], 75))
        if self.baseline is None or new >= self.baseline:
            self.baseline = new
        else:
            max_drop = BASELINE_DROP_PER_MIN * min(dt, LEARN_EVERY_S * 2) / 60.0
            self.baseline = max(new, self.baseline - max_drop)
        self.ear_threshold = float(np.clip(self.baseline * EAR_THRESHOLD_RATIO, 0.07, 0.22))
        self.squint_threshold = float(np.clip(self.baseline * EAR_SQUINT_RATIO,
                                              self.ear_threshold + 0.02, 0.30))
        self.narrow_threshold = float(np.clip(self.baseline * NARROW_RATIO,
                                              self.ear_threshold + 0.02, 0.30))
        if not self.calibrated:
            self.calibrated = True
            logger.info(f"Learned driver's eyes: baseline EAR={self.baseline:.3f} "
                        f"-> threshold={self.ear_threshold:.3f}")

    def update(self, pts, frame, t):
        """pts: (478, 2) landmark pixel coordinates, or None if no face."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        light_ok = brightness >= LOW_LIGHT_BRIGHTNESS
        light_conf = float(np.clip((brightness - LOW_LIGHT_BRIGHTNESS) / 50.0, 0.0, 1.0))

        s = {"t": t, "face": pts is not None, "brightness": brightness,
             "ear": None, "mar": None, "closed_for": 0.0, "perclos": None,
             "yawns": 0, "reasons": []}

        s["yaw"] = None
        s["looking_away"] = False
        s["ai"], s["ai_conf"], s["ai_note"] = None, None, None
        s["iris"], s["squint"], s["eyes_closed"] = None, False, False
        s["pitch"], s["roll"], s["head_down"], s["head_tilt"] = None, None, False, False
        s["head_back"] = False
        s["mouth"], s["smiling"] = None, False
        if pts is not None:
            self.no_face_since = None
            raw_ear = (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2.0
            self.ear_buf.append(raw_ear)
            ear = float(np.median(self.ear_buf))     # median of 3 frames: removes 1-frame jitter
            mar = mouth_aspect_ratio(pts)
            s["ear"], s["mar"] = ear, mar

            # head turn left/right: 0 = facing camera, +-1 = full profile.
            # A turned head makes the eyes look narrow (low EAR) -> ignore eyes then.
            dl, dr = _d(pts[NOSE_TIP], pts[FACE_LEFT]), _d(pts[NOSE_TIP], pts[FACE_RIGHT])
            raw_yaw = (dl - dr) / max(dl + dr, 1e-6)
            self.yaw_buf.append(raw_yaw)
            raw_yaw = float(np.median(self.yaw_buf))   # landmark jitter: median of 5 frames
            yaw = self._relative_yaw(t, raw_yaw)     # 0 = the driver's normal direction
            s["yaw"], s["raw_yaw"] = yaw, raw_yaw
            frontal = abs(yaw) < YAW_MAX
            eyes_ok = abs(yaw) < YAW_EYES_MAX         # eyes still readable (a bit turned is fine)
            s["looking_away"] = not frontal
            if frontal:
                self.away_since = None
            elif self.away_since is None:
                self.away_since = t

            # --- head pose: dropped (pitch) or tilted sideways (roll) = nodding off ---
            fh, fw = frame.shape[:2]
            pitch, roll = head_pose(pts, fw, fh)
            if pitch is not None:
                self.pitch_buf.append(pitch)
                pitch = float(np.median(self.pitch_buf))     # smooth over 5 frames
            s["pitch"], s["roll"] = pitch, roll
            # remember the last 2 s of head motion, to judge a disappearing face
            self.dropped = False
            self.recent_face.append((t, pitch, float(pts[:, 1].mean()) / fh))
            while self.recent_face and t - self.recent_face[0][0] > 2.0:
                self.recent_face.popleft()
            pd = ((pitch - self.base_pitch) * PITCH_DOWN_SIGN
                  if pitch is not None and self.base_pitch is not None else None)
            down = pd is not None and pd > HEAD_DOWN_DEG
            back = pd is not None and pd < -HEAD_BACK_DEG      # chin up: asleep leaning back
            tilt = abs(roll - (self.base_roll or 0.0)) > HEAD_TILT_DEG
            s["head_down"], s["head_tilt"], s["head_back"] = down, tilt, back
            self.last_head_down = down
            head_off = down or tilt or back
            if head_off:
                if self.head_since is None:
                    self.head_since = t
            else:
                self.head_since = None

            # --- smiling / laughing: mouth much wider than normal (laughing squeezes the eyes) ---
            mouth = _d(pts[61], pts[291]) / max(_d(pts[33], pts[263]), 1e-6)
            smiling = self.base_mouth is not None and mouth >= self.base_mouth * SMILE_RATIO
            s["mouth"], s["smiling"] = mouth, smiling

            # --- learn this driver's open-eye EAR + normal head pose in the background ---
            self._learn(t, ear, yaw, pitch if not head_off else None, roll, mouth)

            # --- eyes closed duration ---
            # Normally only when facing the camera (a turned head makes eyes look narrow).
            # But a head that dropped / fell back / tilted distorts the yaw estimate, and that
            # is exactly when a microsleep happens: keep judging the eyes then.
            judge_eyes = eyes_ok or (head_off and abs(yaw) < YAW_POSE_MAX)
            if judge_eyes:
                # --- image check (works through glasses): learn how well it sees this
                #     driver's open eyes, then ask it whenever EAR is in doubt ---
                haar = None
                doubt_thr = max(self.squint_threshold,
                                (self.baseline or 0.0) * HAAR_DOUBT_RATIO)
                if (frontal and ear > doubt_thr and not head_off
                        and t - self.haar_learn_t >= HAAR_LEARN_EVERY_S):
                    self.haar_learn_t = t
                    h = open_eyes_seen(gray, pts)
                    if h is not None:
                        self.haar_open.append(1.0 if h > 0 else 0.0)   # >= 1 open eye seen
                rate = self.haar_rate()
                s["haar_rate"] = rate
                use_haar = rate is not None and rate >= HAAR_RELIABLE
                if use_haar and not head_off and self.ear_threshold <= ear < doubt_thr:
                    haar = open_eyes_seen(gray, pts)
                s["haar"] = haar
                # 3 zones: clearly closed / squint zone (check iris) / open
                if ear < self.ear_threshold:
                    ear_closed = True
                elif haar is not None:
                    # detector is reliable for this driver: no open eye found = closed
                    ear_closed = haar == 0
                    s["squint"] = not ear_closed and ear < self.squint_threshold
                elif ear < self.squint_threshold:
                    if head_off:
                        # head dropped + narrow eyes = nodding off; no squint excuse
                        # (lash line looks dark from above and fools the iris check)
                        ear_closed = True
                    else:
                        iris = iris_visible_fraction(gray, pts)
                        s["iris"] = iris
                        ear_closed = iris is not None and iris < IRIS_DARK_MIN
                        s["squint"] = not ear_closed  # narrow but iris visible = squint
                else:
                    ear_closed = False
                # --- hard squint vs shut eyes: the eyelid points look shut in both cases, but
                #     brows, lid curve and the AI expression model differ (squint_check.py) ---
                sq_face = False
                checker = globals().get("squint_checker")
                if checker is not None and frontal and not head_off and self.calibrated:
                    if ear >= self.squint_threshold:
                        checker.learn_open(pts)
                    elif ear_closed or self.ear_closed_since is not None:
                        try:
                            sq_face, sq_info = checker.check(pts, t)
                        except Exception as e:                    # never block an alarm
                            sq_face, sq_info = False, None
                            logger.warning(f"squint check error: {e}")
                        s["sq_info"] = sq_info
                if sq_face:
                    if self.squint_face_since is None:
                        self.squint_face_since = t
                        logger.info(f"Hard squint, not closed: {s.get('sq_info')}")
                    if t - self.squint_face_since < SQUINT_EXCUSE_MAX_S:
                        ear_closed = False
                        s["squint"] = True
                    else:
                        sq_face = False                           # too long: count as closed
                else:
                    self.squint_face_since = None
                # Hysteresis: once the eyes are closed they count as closed until they clearly
                # reopen. Glasses make EAR / iris flicker around the squint edge, which used to
                # reset the timer every few frames so the alarm never came.
                if ear_closed:
                    self.reopen_since = None
                elif self.ear_closed_since is not None:
                    if haar is not None and haar > 0 and t - self.ear_closed_since < HOLD_AFTER_S:
                        pass                                  # detector sees an open eye: blink over
                    elif (ear < self.squint_threshold * REOPEN_MARGIN
                          and not (haar is not None and haar > 0) and not sq_face):
                        ear_closed, self.reopen_since = True, None
                        s["squint"] = False
                    elif t - self.ear_closed_since >= HOLD_AFTER_S:
                        if self.reopen_since is None:
                            self.reopen_since = t
                        ear_closed = t - self.reopen_since < REOPEN_GRACE_S
                if ear_closed:
                    if self.ear_closed_since is None:
                        self.ear_closed_since, self.ep_min_ear = t, ear
                    self.ep_min_ear = min(self.ep_min_ear, ear)
                else:
                    self.ear_closed_since, self.reopen_since, self.ep_min_ear = None, None, None
                # clearly shut at some point in this closure? (else: only narrow = maybe a squint)
                s["closed_clear"] = self.ep_min_ear is not None and self.ep_min_ear < self.ear_threshold
                closed = ear_closed

                # --- AI second opinion (Roboflow), only if a fresh verdict exists ---
                v = eye_model.current(t) if eye_model is not None else None
                if v is not None:
                    state, conf, frame_t = v
                    s["ai"], s["ai_conf"] = state, conf
                    # veto only borderline closures: a clearly shut eye (very low EAR)
                    # always counts, so a wrong cloud answer can't hide a real microsleep
                    # Glasses fool the cloud model (it keeps saying "open"), so the veto
                    # only covers short closures: past AI_VETO_MAX_S the eyes win.
                    if (ear_closed and state == "open" and conf >= 0.70
                            and ear > self.ear_threshold * 0.6
                            and frame_t >= self.ear_closed_since - 0.3
                            and t - self.ear_closed_since < AI_VETO_MAX_S):
                        closed = False                      # AI saw open eyes during this "closure"
                        s["ai_note"] = "veto"
                    # (no "AI confirms closed" rule: tests showed the cloud model
                    #  calls squinting eyes "closed eye", so it may only veto)
                if closed and smiling and not head_off:
                    closed = False                          # laughing squeezes the eyes shut
                    s["squint"] = True
                s["eyes_closed"] = closed
                self.recent_closed.append((t, closed))
                # heavy eyelids: narrow (squint zone) or closed for a long stretch
                if ear < self.narrow_threshold:
                    if self.narrow_since is None:
                        self.narrow_since = t
                else:
                    self.narrow_since = None

                if closed:
                    if self.closed_since is None:
                        # a closure the AI vetoed at first counts from when the eyes shut
                        self.closed_since = (self.ear_closed_since
                                             if ear_closed and self.ear_closed_since is not None else t)
                    s["closed_for"] = t - self.closed_since
                else:
                    self.closed_since = None
                self.closed_hist.append((t, closed and s["closed_for"] <= PERCLOS_MAX_EPISODE_S))
            else:
                self.closed_since = None
                self.ear_closed_since = None

            # --- yawns ---
            if mar > MAR_YAWN:
                if self.yawn_since is None:
                    self.yawn_since, self.yawn_counted = t, False
                elif not self.yawn_counted and t - self.yawn_since >= YAWN_MIN_S:
                    self.yawn_times.append(t)
                    self.yawn_counted = True
                    logger.info("Yawn detected")
            else:
                self.yawn_since = None
        else:
            self.closed_since = None
            self.ear_closed_since = None
            self.yawn_since = None
            self.away_since = None
            self.head_since = None
            self.narrow_since = None
            self.ear_buf.clear()
            self.pitch_buf.clear()
            if self.no_face_since is None:
                self.no_face_since = t
                # did the face vanish DOWNWARD (head dropping) rather than turn away / leave?
                rf = self.recent_face
                if len(rf) >= 3:
                    pitches = [p for _, p, _ in rf if p is not None]
                    rise = (pitches[-1] - min(pitches)) * PITCH_DOWN_SIGN if len(pitches) >= 2 else 0.0
                    slide = rf[-1][2] - min(c for _, _, c in rf)
                    self.dropped = (self.last_head_down or rise >= DROP_PITCH_RISE
                                    or slide >= DROP_FACE_SLIDE)
                    if self.dropped:
                        logger.info(f"Face lost while dropping (pitch +{rise:.0f} deg, "
                                    f"slide {slide:.2f}) -> head-drop watch")
                self.recent_face.clear()
            elif self.calibrated and t - self.no_face_since >= NEW_DRIVER_NO_FACE_S:
                self.forget_driver()
                logger.info("No face for 30 s: forgetting learned eyes (driver may change)")

        # --- short window: share of closed frames in the last 2 s (flicker-proof) ---
        while self.recent_closed and t - self.recent_closed[0][0] > CLOSED_WINDOW_S:
            self.recent_closed.popleft()
        s["closed_frac"] = None
        if self.recent_closed and t - self.recent_closed[0][0] >= CLOSED_WINDOW_S * 0.85:
            s["closed_frac"] = sum(c for _, c in self.recent_closed) / len(self.recent_closed)

        # --- PERCLOS over sliding window ---
        while self.closed_hist and t - self.closed_hist[0][0] > PERCLOS_WINDOW_S:
            self.closed_hist.popleft()
        if self.closed_hist and t - self.closed_hist[0][0] >= PERCLOS_MIN_DATA_S:
            s["perclos"] = sum(c for _, c in self.closed_hist) / len(self.closed_hist)

        while self.yawn_times and t - self.yawn_times[0] > YAWN_WINDOW_S:
            self.yawn_times.popleft()
        s["yawns"] = len(self.yawn_times)

        # camera confidence: how much we trust the camera right now (for sensor fusion later)
        s["cam_conf"] = light_conf * (1.0 if s["face"] else 0.5)

        # --- status code ---
        if not light_ok:
            s["status"] = ST_LOW_LIGHT
        elif not s["face"]:
            s["status"] = ST_NO_FACE
        else:
            s["status"] = ST_OK

        s["level"] = self._decide_level(s, light_ok)
        return s

    def _decide_level(self, s, light_ok):
        lvl, r = 0, s["reasons"]
        # clearly shut eyes: alarm at 2 s. Only narrow (glasses / maybe squinting): 3 s.
        clear = s.get("closed_clear", True)
        alarm_s = EYES_CLOSED_ALARM_S if clear else SQUINT_CLOSED_ALARM_S
        warn_s = EYES_CLOSED_WARN_S if clear else SQUINT_CLOSED_WARN_S
        if s["closed_for"] >= alarm_s:
            lvl = 2; r.append(f"microsleep {s['closed_for']:.1f}s")
        elif s["closed_for"] >= warn_s:
            lvl = max(lvl, 1); r.append(f"eyes closed {s['closed_for']:.1f}s")
        p = s["perclos"]
        # PERCLOS looks back 60 s, so it only WARNS ("you've been drowsy lately").
        # The siren (ALARM) is reserved for eyes closed right now (microsleep).
        if p is not None and p >= PERCLOS_WARN:
            lvl = max(lvl, 1)
            r.append(f"PERCLOS {p:.0%}" + (" (very drowsy)" if p >= PERCLOS_ALARM else ""))
        if s["yawns"] >= YAWNS_WARN:
            lvl = max(lvl, 1); r.append(f"{s['yawns']} yawns/5min")
        if light_ok and self.no_face_since is not None and s["t"] - self.no_face_since >= NO_FACE_WARN_S:
            lvl = max(lvl, 1); r.append("face not visible")
        hidden = (self.no_face_since is not None and person_checker is not None
                  and person_checker.present_now(s["t"]))
        s["person_no_face"] = hidden
        if hidden and not self.dropped:
            self.dropped = True                     # person in front, face hidden = head down
            logger.info("Person in front of camera but no face -> head-down watch")
        if self.dropped and self.no_face_since is not None:
            dur = s["t"] - self.no_face_since
            if dur >= DROP_ALARM_S:
                lvl = 2; r.append(f"head dropped, face hidden {dur:.0f}s")
            elif dur >= DROP_WARN_S:
                lvl = max(lvl, 1); r.append(f"head dropped {dur:.0f}s")
        if self.narrow_since is not None and s["t"] - self.narrow_since >= NARROW_WARN_S:
            lvl = max(lvl, 1); r.append(f"heavy eyelids {s['t'] - self.narrow_since:.0f}s")
        cf = s.get("closed_frac")
        if cf is not None and cf >= CLOSED_FRAC_ALARM and lvl < 2 and clear:
            lvl = 2; r.append(f"eyes closed {cf:.0%} of last 2s")
        if self.head_since is not None:
            dur = s["t"] - self.head_since
            what = ("head down" if s.get("head_down") else
                    "head back" if s.get("head_back") else "head tilted")
            if dur >= HEAD_EYES_MIN_S and cf is not None and cf >= HEAD_EYES_FRAC:
                lvl = 2; r.append(f"{what} + eyes closed")          # nodding off
            elif dur >= HEAD_ALARM_S:
                lvl = 2; r.append(f"{what} {dur:.0f}s")
            elif dur >= HEAD_WARN_S:
                lvl = max(lvl, 1); r.append(f"{what} {dur:.0f}s")
        if self.away_since is not None:
            dur = s["t"] - self.away_since
            if dur >= LOOK_AWAY_ALARM_S:
                lvl = 2; r.append(f"looking away {dur:.0f}s")
            elif dur >= LOOK_AWAY_WARN_S:
                lvl = max(lvl, 1); r.append(f"looking away {dur:.0f}s")

        # hold alarms for a minimum time to avoid flicker
        t = s["t"]
        if lvl >= self.level:
            if lvl > self.level:
                self.level_raised_at = t
            self.level = lvl
        elif t - self.level_raised_at >= LEVEL_HOLD_S:
            self.level = lvl
        return self.level


# --------------------------------------------------------------------------
# MediaPipe model
# --------------------------------------------------------------------------
def ensure_model():
    if os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    logger.info("Downloading face_landmarker.task model (~3.7 MB)...")
    tmp = MODEL_PATH + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    os.replace(tmp, MODEL_PATH)
    logger.info("Model downloaded.")


# --------------------------------------------------------------------------
# Roboflow eye model (cloud, second opinion)
# --------------------------------------------------------------------------
def load_roboflow_key():
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key and os.path.isfile(ROBOFLOW_KEY_FILE):
        with open(ROBOFLOW_KEY_FILE) as f:
            key = f.read().strip()
    return key or None


class EyeModelChecker(threading.Thread):
    """Sends a face crop to Roboflow at a limited rate; keeps the latest verdict."""

    def __init__(self, key):
        super().__init__(daemon=True)
        self.key = key
        self.wake = threading.Event()
        self.lock = threading.Lock()
        self.pending = None            # (jpeg_bytes, frame_time)
        self.last_submit = 0.0
        self.pause_until = 0.0
        self.verdict = None            # "open" / "closed" / None
        self.verdict_conf = 0.0
        self.verdict_frame_t = 0.0
        self.latency = None
        self.calls = self.errors = 0
        self.status = "starting"
        self.seen_classes = set()
        self.conn = None               # persistent HTTPS connection
        self.pending_ear = None

    def submit(self, frame, pts, t, urgent, ear=None):
        every = ROBOFLOW_URGENT_EVERY_S if urgent else ROBOFLOW_IDLE_EVERY_S
        if ROBOFLOW_TEST_MODE:
            every = 1.0
        self.pending_ear = ear
        if t - self.last_submit < every or t < self.pause_until or self.pending is not None:
            return
        self.last_submit = t
        h, w = frame.shape[:2]
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        px, py = (x1 - x0) * 0.3, (y1 - y0) * 0.3
        x0, y0 = int(max(0, x0 - px)), int(max(0, y0 - py))
        x1, y1 = int(min(w, x1 + px)), int(min(h, y1 + py))
        if x1 - x0 < 20 or y1 - y0 < 20:
            return
        crop = frame[y0:y1, x0:x1]
        scale = 256.0 / max(crop.shape[:2])        # small upload = faster round trip
        if scale < 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self.lock:
                self.pending = (buf.tobytes(), t)
            self.wake.set()

    def current(self, t):
        """Fresh verdict: (state, confidence, frame_time) or None."""
        if self.verdict and t - self.verdict_frame_t <= ROBOFLOW_VERDICT_TTL_S:
            return self.verdict, self.verdict_conf, self.verdict_frame_t
        return None

    def _post(self, body):
        """POST over ONE persistent HTTPS connection (keep-alive).
        A new connection costs DNS + TLS and often a cold server (~1-5 s);
        a reused one answers in ~0.3 s."""
        headers = {"Authorization": f"Bearer {self.key}",
                   "Content-Type": "application/x-www-form-urlencoded",
                   "Connection": "keep-alive"}
        for attempt in (1, 2):                 # retry once on a dropped idle connection
            if self.conn is None:
                self.conn = http.client.HTTPSConnection(ROBOFLOW_HOST, timeout=ROBOFLOW_TIMEOUT_S)
            try:
                path = ROBOFLOW_PATH + ("?confidence=10" if ROBOFLOW_TEST_MODE else "")
                self.conn.request("POST", path, body=body, headers=headers)
                r = self.conn.getresponse()
                payload = r.read()
            except (http.client.HTTPException, OSError):
                self.conn.close()
                self.conn = None
                if attempt == 2:
                    raise
                continue
            if r.status >= 400:
                raise urllib.error.HTTPError(ROBOFLOW_URL, r.status, r.reason, r.headers, None)
            return json.loads(payload)

    def _parse(self, data):
        best = {"open": 0.0, "closed": 0.0}
        for p in data.get("predictions", []):
            name = str(p.get("class", "")).lower()
            self.seen_classes.add(name)
            conf = float(p.get("confidence", 0.0))
            # check "awake" words first: "Non Drowsy" / "not_sleepy" must not count as closed
            if any(k in name for k in ("open", "awake", "alert", "non", "not", "natural")):
                best["open"] = max(best["open"], conf)
            elif any(k in name for k in ("clos", "sleep", "drows")):
                best["closed"] = max(best["closed"], conf)
        state = max(best, key=best.get)
        return (state, best[state]) if best[state] >= ROBOFLOW_MIN_CONF else (None, best[state])

    def run(self):
        try:                                    # pre-open: DNS + TLS done before 1st image
            self.conn = http.client.HTTPSConnection(ROBOFLOW_HOST, timeout=ROBOFLOW_TIMEOUT_S)
            self.conn.connect()
        except Exception:
            self.conn = None
        while True:
            self.wake.wait()
            self.wake.clear()
            with self.lock:
                job, self.pending = self.pending, None
            if job is None:
                continue
            jpeg, frame_t = job
            t_start = time.monotonic()
            try:
                data = self._post(base64.b64encode(jpeg))
                self.latency = time.monotonic() - t_start
                self.calls += 1
                state, conf = self._parse(data)
                self.verdict, self.verdict_conf, self.verdict_frame_t = state, conf, frame_t
                if ROBOFLOW_TEST_MODE:
                    preds = [(p.get("class"), round(float(p.get("confidence", 0)), 2))
                             for p in data.get("predictions", [])]
                    ear_txt = f"{self.pending_ear:.3f}" if self.pending_ear is not None else "-"
                    logger.info(f"RFTEST ear={ear_txt} preds={preds} -> {state} "
                                f"{conf:.2f} ({self.latency * 1000:.0f} ms)")
                if self.status != "ok":
                    logger.info(f"Roboflow OK ({self.latency * 1000:.0f} ms), classes seen: "
                                f"{sorted(self.seen_classes) or '-'}")
                self.status = "ok"
            except urllib.error.HTTPError as e:
                self.errors += 1
                if e.code in (401, 403):
                    self.status = "bad API key"
                    self.pause_until = time.monotonic() + 60
                else:
                    self.status = f"HTTP {e.code}"
                    self.pause_until = time.monotonic() + 5
                logger.warning(f"Roboflow: {self.status}")
            except Exception as e:
                self.errors += 1
                if self.status != "offline":
                    logger.warning(f"Roboflow unreachable ({e}); using EAR only")
                self.status = "offline"
                self.pause_until = time.monotonic() + 5


# --------------------------------------------------------------------------
# Person detector: "someone is in front of the camera but no face" = head down
# --------------------------------------------------------------------------
class PersonChecker(threading.Thread):
    """Runs the App Lab Object Detection brick (YOLOX-nano, COCO 'person') in the
    background, only while the face tracker sees no face."""

    def __init__(self):
        super().__init__(daemon=True)
        from arduino.app_bricks.object_detection import ObjectDetection
        self.od = ObjectDetection(confidence=PERSON_MIN_CONF)
        self.wake = threading.Event()
        self.pending = None
        self.last_submit = 0.0
        self.present = False           # big person right in front of the camera
        self.result_t = 0.0            # frame time of the latest result
        self.box = None
        self.ms = None
        self.error_logged = False

    def submit(self, frame, t):
        if t - self.last_submit < PERSON_EVERY_S or self.pending is not None:
            return
        self.last_submit = t
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.pending = (buf.tobytes(), t, frame.shape[1], frame.shape[0])
            self.wake.set()

    def present_now(self, t):
        return self.present and t - self.result_t <= PERSON_FRESH_S

    def run(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            job, self.pending = self.pending, None
            if job is None:
                continue
            jpeg, frame_t, w, h = job
            t0 = time.monotonic()
            try:
                out = self.od.detect(jpeg, image_type="jpg") or {}
                self.ms = (time.monotonic() - t0) * 1000
                best = None
                for d in out.get("detection", []):
                    if str(d.get("class_name", "")).lower() != "person":
                        continue
                    x0, y0, x1, y1 = [float(v) for v in d.get("bounding_box_xyxy", [0, 0, 0, 0])]
                    area = max(0.0, x1 - x0) * max(0.0, y1 - y0) / float(w * h)
                    cx = (x0 + x1) / 2 / w
                    if area >= PERSON_MIN_AREA and 0.15 < cx < 0.85 and (best is None or area > best[0]):
                        best = (area, (x0, y0, x1, y1), float(d.get("confidence", 0)))
                self.present, self.result_t = best is not None, frame_t
                self.box = best[1] if best else None
                self.error_logged = False
            except Exception as e:
                self.present = False
                if not self.error_logged:
                    logger.warning(f"Person detector failed: {e}")
                    self.error_logged = True
                time.sleep(1.0)


# --------------------------------------------------------------------------
# Alarm sound (USB speaker on the Linux side)
# --------------------------------------------------------------------------
class AlarmSound(threading.Thread):
    """Loops a WAV file on the USB speaker while active (ALARM level only).
    Also plays short one-shot chimes (ready / driver learned) when no alarm is sounding;
    an alarm always cuts a chime off."""

    BLOCK_S = 0.05     # 50 ms blocks -> stops within ~0.1 s when the alarm ends

    def __init__(self, path):
        super().__init__(daemon=True)
        with wave.open(path, "rb") as w:
            self.rate, self.channels = w.getframerate(), w.getnchannels()
            if w.getsampwidth() != 2:
                raise ValueError("alarm WAV must be 16-bit PCM")
            self.pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()
        self.block = int(self.rate * self.BLOCK_S) * self.channels
        self.active = threading.Event()
        self.chimes = deque(maxlen=3)
        self.spk = None
        self.next_try = 0.0
        self.warned = False
        logger.info(f"Alarm sound loaded: {os.path.basename(path)} "
                    f"{len(self.pcm) / self.channels / self.rate:.1f}s {self.rate}Hz x{self.channels}")

    def load_clip(self, path):
        """Load a one-shot WAV (must match the alarm's rate/channels, 16-bit). None if unusable."""
        try:
            with wave.open(path, "rb") as w:
                if (w.getsampwidth(), w.getframerate(), w.getnchannels()) != (2, self.rate, self.channels):
                    raise ValueError(f"needs 16-bit {self.rate}Hz x{self.channels}")
                return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy()
        except Exception as e:
            logger.warning(f"Sound {os.path.basename(path)} not used: {e}")
            return None

    def play_once(self, pcm):
        if pcm is not None:
            self.chimes.append(pcm)

    def set_active(self, on):
        if on:
            self.active.set()
        else:
            self.active.clear()

    def _speaker_ready(self):
        if self.spk is None and time.monotonic() >= self.next_try:
            self._open()
        return self.spk is not None

    def _write(self, chunk):
        try:
            self.spk.play(chunk)
            return True
        except Exception as e:
            logger.warning(f"Speaker write failed ({e}); reopening")
            try:
                self.spk.stop()
            except Exception:
                pass
            self.spk = None
            return False

    def _play_chime(self, pcm):
        if not self._speaker_ready():
            return                            # no speaker: skip the chime (never queue forever)
        for pos in range(0, len(pcm), self.block):
            if self.active.is_set() or not self._write(pcm[pos:pos + self.block]):
                return

    def _open(self):
        try:
            spk = Speaker(SPEAKER_DEVICE, sample_rate=self.rate, channels=self.channels, format=np.int16)
            spk.start()
            self.spk, self.warned = spk, False
            logger.info("USB speaker ready")
        except Exception as e:
            self.spk = None
            self.next_try = time.monotonic() + 10.0
            if not self.warned:
                logger.warning(f"No USB speaker for alarm sound ({e}); retrying every 10 s")
                self.warned = True

    def run(self):
        pos = 0
        while True:
            if not self.active.is_set():
                pos = 0                       # next alarm starts from the beginning
                if self.chimes:
                    self._play_chime(self.chimes.popleft())
                else:
                    self.active.wait(timeout=0.1)
                continue
            if not self._speaker_ready():
                time.sleep(0.2)
                continue
            chunk = self.pcm[pos:pos + self.block]
            pos = (pos + self.block) % len(self.pcm)
            self._write(chunk)


# --------------------------------------------------------------------------
# MCU link
# --------------------------------------------------------------------------
class McuLink:
    def __init__(self):
        self.sent = None
        self.last_send = 0.0
        self.error_logged = False

    def send(self, level, status, force=False):
        now = time.monotonic()
        if not force and (level, status) == self.sent and now - self.last_send < HEARTBEAT_S:
            return
        try:
            Bridge.notify("set_state", int(level), int(status))
            self.sent, self.last_send, self.error_logged = (level, status), now, False
        except Exception as e:
            if not self.error_logged:
                logger.warning(f"Bridge send failed: {e}")
                self.error_logged = True


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
ensure_model()
engine = FaceMeshEngine(MODEL_PATH, threads=4)
try:
    from squint_check import SquintCheck
    squint_checker = SquintCheck(MODEL_PATH)
    logger.info("Squint check ready (blendshapes + brows + lid curve)")
except Exception as e:                      # without it: every closure counts (safe default)
    squint_checker = None
    logger.warning(f"Squint check disabled: {e}")
# smoke test: run once on a blank frame so model errors show up at startup
engine.process(np.zeros((RESOLUTION[1], RESOLUTION[0], 3), np.uint8))
logger.info("Face engine ready (LiteRT, 478-point face mesh)")

detector = DrowsinessDetector()
mcu = McuLink()
person_checker = None
try:
    person_checker = PersonChecker()
    person_checker.start()
    logger.info("Person detector ready (Object Detection brick)")
except Exception as e:
    logger.warning(f"Person detector disabled: {e}")

eye_model = None
_rf_key = load_roboflow_key()
if _rf_key:
    eye_model = EyeModelChecker(_rf_key)
    eye_model.start()
    logger.info("Roboflow eye model enabled (second opinion)")
else:
    logger.info(f"Roboflow eye model off: no API key in {ROBOFLOW_KEY_FILE}")

alarm_sound = None
ready_pcm = learned_pcm = None
try:
    alarm_sound = AlarmSound(ALARM_SOUND)
    ready_pcm = alarm_sound.load_clip(READY_SOUND)
    learned_pcm = alarm_sound.load_clip(LEARNED_SOUND)
    alarm_sound.start()
except Exception as e:
    logger.warning(f"Alarm sound disabled: {e}")
ready_chime_done = False          # "system ready" chime plays once per app start
was_calibrated = False            # for the "driver learned" chime
camera = None
next_camera_try = 0.0
last_log = 0.0
current_fps = 0.0
frames = 0
last_level = 0


def annotate(frame, pts, s):
    """Camera frame with face mesh + eye contours drawn on it."""
    out = frame.copy()
    if pts is not None:
        for x, y in pts[::2].astype(int):
            cv2.circle(out, (int(x), int(y)), 1, (90, 200, 90), -1)
        eye_col = ((40, 40, 255) if s.get("eyes_closed")          # red = closed
                   else (0, 165, 255) if s.get("squint")          # orange = squint
                   else (255, 220, 60))                           # cyan = open
        for eye in (LEFT_EYE, RIGHT_EYE):
            poly = pts[[eye[0], eye[1], eye[2], eye[3], eye[4], eye[5]]].astype(np.int32)
            cv2.polylines(out, [poly], True, eye_col, 2)
    return out


def save_debug_frame(frame, pts, s):
    """Write the latest annotated frame to <app>/debug/frame.jpg."""
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        tmp = os.path.join(DEBUG_DIR, "frame.tmp.jpg")
        cv2.imwrite(tmp, annotate(frame, pts, s))
        os.replace(tmp, os.path.join(DEBUG_DIR, "frame.jpg"))
    except Exception as e:
        logger.warning(f"debug snapshot failed: {e}")


class LiveView:
    """Latest annotated JPEG + stats for the web page (MJPEG stream + JSON)."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg = None
        self.seq = 0
        self.last_fetch = 0.0
        self.stats = {"status": "NO_CAMERA", "level": "OK", "level_num": 0}

    def publish_stats(self, s, fps):
        self.stats = {
            "level": LEVEL_NAMES[s["level"]], "level_num": s["level"],
            "status": STATUS_NAMES[s["status"]], "face": s["face"],
            "ear": s["ear"], "ear_threshold": detector.ear_threshold,
            "calibrated": detector.calibrated, "mar": s["mar"], "mar_yawn": MAR_YAWN,
            "closed_for": s["closed_for"], "perclos": s["perclos"],
            "perclos_warn": PERCLOS_WARN, "perclos_alarm": PERCLOS_ALARM,
            "yawns": s["yawns"], "brightness": s["brightness"], "cam_conf": s["cam_conf"],
            "reasons": s["reasons"], "fps": fps, "time": time.time(),
            "yaw": s.get("yaw"), "looking_away": s.get("looking_away", False),
            "squint": s.get("squint", False), "eyes_closed": s.get("eyes_closed", False),
            "smiling": s.get("smiling", False),
            "pitch_delta": (s["pitch"] - detector.base_pitch) * PITCH_DOWN_SIGN
                           if s.get("pitch") is not None and detector.base_pitch is not None else None,
            "roll_delta": s["roll"] - (detector.base_roll or 0.0) if s.get("roll") is not None else None,
            "head_down": s.get("head_down", False), "head_tilt": s.get("head_tilt", False),
            "head_back": s.get("head_back", False),
            "iris": s.get("iris"), "squint_threshold": detector.squint_threshold,
            "ai": s.get("ai"), "ai_conf": s.get("ai_conf"), "ai_note": s.get("ai_note"),
            "ai_status": eye_model.status if eye_model else "off (no API key)",
            "ai_latency": eye_model.latency if eye_model else None,
            "ai_calls": eye_model.calls if eye_model else 0,
        }

    def publish_frame(self, frame, pts, s):
        # only spend CPU on JPEG encoding while someone is watching
        if self.jpeg is not None and time.monotonic() - self.last_fetch > 3.0:
            return
        ok, buf = cv2.imencode(".jpg", annotate(frame, pts, s),
                               [cv2.IMWRITE_JPEG_QUALITY, LIVE_JPEG_QUALITY])
        if ok:
            with self.cond:
                self.jpeg, self.seq = buf.tobytes(), self.seq + 1
                self.cond.notify_all()

    def frames(self):
        last = -1
        while True:
            self.last_fetch = time.monotonic()
            with self.cond:
                self.cond.wait_for(lambda: self.seq != last, timeout=1.0)
                jpeg, seq = self.jpeg, self.seq
            if jpeg is None or seq == last:
                continue
            last = seq
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


# --------------------------------------------------------------------------
# Password for the live (mechanic) page: the video shows the driver's face, so
# /stream, /api/stats, /api/recalibrate and /api/roboflow_key need a login.
# The driver page (driver.html) stays open: the driver must see it without a password.
#
#   live_auth.json            salt + PBKDF2 hash of the password + cookie secret (never plaintext)
#   live_password.txt         put a password here over SSH and restart the app -> it becomes the
#                             new password and this file is deleted (also the way to reset it)
#   live_password_initial.txt written on first start with a random password; deleted once the
#                             password is changed on the web page
# --------------------------------------------------------------------------
LIVE_AUTH_FILE = os.path.join(APP_DIR, "live_auth.json")
LIVE_PASSWORD_FILE = os.path.join(APP_DIR, "live_password.txt")
LIVE_PASSWORD_INITIAL_FILE = os.path.join(APP_DIR, "live_password_initial.txt")
LIVE_COOKIE = "ams_live"
LIVE_COOKIE_DAYS = 30
LOGIN_MAX_FAILS = 5           # this many wrong passwords in a row ...
LOGIN_LOCK_S = 60.0           # ... locks the login for this long
PBKDF2_ROUNDS = 200_000


def _write_private(path, text):
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, 0o600)


class LiveAuth:
    def __init__(self):
        self.lock = threading.Lock()
        self.fails = 0
        self.locked_until = 0.0
        self.cfg = None
        try:
            with open(LIVE_AUTH_FILE) as f:
                self.cfg = json.load(f)
        except Exception:
            pass
        new_pw = None
        try:
            with open(LIVE_PASSWORD_FILE) as f:
                new_pw = f.read().strip()
            os.remove(LIVE_PASSWORD_FILE)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"cannot read {LIVE_PASSWORD_FILE}: {e}")
        if new_pw:
            self.set_password(new_pw)
            self._remove_initial()
            logger.info("Live page password set from live_password.txt")
        elif not self.cfg:
            pw = secrets.token_urlsafe(6)          # 8 characters, e.g. "q3Zk-8Pa"
            self.set_password(pw)
            try:
                _write_private(LIVE_PASSWORD_INITIAL_FILE, pw + "\n")
            except Exception as e:
                logger.warning(f"cannot save initial password file: {e}")
            logger.info(f"Live page password created: {pw} (also in live_password_initial.txt)")

    @staticmethod
    def _hash(pw, salt):
        return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), PBKDF2_ROUNDS).hex()

    @staticmethod
    def _remove_initial():
        try:
            os.remove(LIVE_PASSWORD_INITIAL_FILE)
        except FileNotFoundError:
            pass

    def set_password(self, pw):
        salt = secrets.token_hex(16)
        # a new secret also logs out every browser that had the old password
        cfg = {"salt": salt, "hash": self._hash(pw, salt), "secret": secrets.token_hex(32)}
        try:
            _write_private(LIVE_AUTH_FILE, json.dumps(cfg))
        except Exception as e:
            logger.warning(f"cannot save live password: {e}")
        self.cfg = cfg

    def check_password(self, pw):
        """Returns (ok, error). Wrong passwords are rate limited."""
        with self.lock:
            now = time.monotonic()
            if now < self.locked_until:
                return False, f"ใส่ผิดหลายครั้ง รอ {int(self.locked_until - now) + 1} วินาที"
            ok = hmac.compare_digest(self._hash(str(pw), self.cfg["salt"]), self.cfg["hash"])
            if ok:
                self.fails = 0
                return True, None
            self.fails += 1
            if self.fails >= LOGIN_MAX_FAILS:
                self.fails, self.locked_until = 0, now + LOGIN_LOCK_S
            return False, "รหัสผ่านไม่ถูกต้อง"

    def token(self):
        return hmac.new(bytes.fromhex(self.cfg["secret"]), b"live", hashlib.sha256).hexdigest()

    def ok(self, request: Request):
        return hmac.compare_digest(request.cookies.get(LIVE_COOKIE, ""), self.token())

    def login_response(self):
        r = JSONResponse({"ok": True})
        r.set_cookie(LIVE_COOKIE, self.token(), max_age=LIVE_COOKIE_DAYS * 86400,
                     httponly=True, samesite="strict", path="/")
        return r


live_auth = LiveAuth()


def _unauthorized():
    return JSONResponse({"ok": False, "error": "login required"}, status_code=401)


def _login(body: dict = Body(...)):
    ok, err = live_auth.check_password(body.get("password", ""))
    if not ok:
        return JSONResponse({"ok": False, "error": err}, status_code=401)
    return live_auth.login_response()


def _logout():
    r = JSONResponse({"ok": True})
    r.delete_cookie(LIVE_COOKIE, path="/")
    return r


def _change_password(request: Request, body: dict = Body(...)):
    if not live_auth.ok(request):
        return _unauthorized()
    ok, err = live_auth.check_password(body.get("old", ""))
    if not ok:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    new = str(body.get("new", ""))
    if len(new) < 6 or len(new) > 100:
        return JSONResponse({"ok": False, "error": "รหัสใหม่ต้องยาว 6–100 ตัวอักษร"}, status_code=400)
    live_auth.set_password(new)
    live_auth._remove_initial()
    logger.info("Live page password changed from web page")
    return live_auth.login_response()        # keep this browser logged in with the new secret


def _stream(request: Request):
    if not live_auth.ok(request):
        return _unauthorized()
    return StreamingResponse(live.frames(), media_type="multipart/x-mixed-replace; boundary=frame")


def _stats(request: Request):
    if not live_auth.ok(request):
        return _unauthorized()
    return live.stats


live = LiveView()
ui = WebUI()
ui.expose_api("POST", "/api/login", _login)
ui.expose_api("POST", "/api/logout", _logout)
ui.expose_api("POST", "/api/password", _change_password)
ui.expose_api("GET", "/stream", _stream)
ui.expose_api("GET", "/api/stats", _stats)


def _recalibrate(request: Request):
    if not live_auth.ok(request):
        return _unauthorized()
    detector.recalibrate()
    return {"ok": True}


ui.expose_api("POST", "/api/recalibrate", _recalibrate)


def _set_roboflow_key(request: Request, body: dict = Body(...)):
    """Save the Roboflow API key typed on the web page and enable the AI model."""
    global eye_model
    if not live_auth.ok(request):
        return _unauthorized()
    key = str(body.get("key", "")).strip()
    if not key or len(key) > 200 or any(c.isspace() for c in key):
        return {"ok": False, "error": "invalid key"}
    try:
        with open(ROBOFLOW_KEY_FILE, "w") as f:
            f.write(key + "\n")
        os.chmod(ROBOFLOW_KEY_FILE, 0o600)
    except Exception as e:
        return {"ok": False, "error": f"cannot save: {e}"}
    if eye_model is None:
        eye_model = EyeModelChecker(key)
        eye_model.start()
    else:
        eye_model.key, eye_model.status, eye_model.pause_until = key, "starting", 0.0
    logger.info("Roboflow API key saved from web page; AI model enabled")
    return {"ok": True}


ui.expose_api("POST", "/api/roboflow_key", _set_roboflow_key)


# --------------------------------------------------------------------------
# Driver page (http://<board>:7000/driver.html): truck number, shift timer,
# today's alerts. Everything is kept on the board, so it works with no internet.
# --------------------------------------------------------------------------
DATA_DIR = os.path.join(APP_DIR, "data")
TRUCK_ID_FILE = os.path.join(APP_DIR, "truck_id.txt")   # e.g. "7" (one per board)
EVENTS_FILE = os.path.join(DATA_DIR, "events.jsonl")
SHIFT_FILE = os.path.join(DATA_DIR, "shift.json")
EVENTS_KEEP = 500             # newest events kept in memory / served to the page
BREAK_EVERY_S = 2 * 3600      # remind the driver to rest after 2 h of driving


def _read_truck_id():
    try:
        with open(TRUCK_ID_FILE) as f:
            tid = f.read().strip()
        return tid[:20] or "1"
    except Exception:
        return "1"


class DriverLog:
    """Alert events (level rises to WARNING/ALARM) + shift start, saved under data/."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events = deque(maxlen=EVENTS_KEEP)
        self.shift_start = None
        self.last_break = None
        self.pending_alert_at = None
        self.pending_level = 0
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            with open(EVENTS_FILE) as f:
                for line in f:
                    try:
                        self.events.append(json.loads(line))
                    except ValueError:
                        pass
        except FileNotFoundError:
            pass
        try:
            with open(SHIFT_FILE) as f:
                d = json.load(f)
            self.shift_start, self.last_break = d.get("start"), d.get("last_break")
        except Exception:
            pass

    def add(self, level, reasons):
        now = time.time()
        ev = {"t": now, "level": level, "reasons": list(reasons)}
        only_no_face = reasons and all(str(r).startswith("face not visible") for r in reasons)
        with self.lock:
            self.events.append(ev)
            if not only_no_face:
                if self.pending_alert_at is None:
                    self.pending_alert_at = now
                self.pending_level = max(self.pending_level, int(level))
            try:
                with open(EVENTS_FILE, "a") as f:
                    f.write(json.dumps(ev) + "\n")
            except Exception as e:
                logger.warning(f"cannot save event: {e}")

    def ack(self):
        with self.lock:
            at = self.pending_alert_at
            self.pending_alert_at = None
            self.pending_level = 0
            for i in range(len(self.events) - 1, -1, -1):
                if not self.events[i].get("acked"):
                    self.events[i]["acked"] = True
                    break
        return None if at is None else max(1, round(time.time() - at))

    def _save_shift(self):
        try:
            with open(SHIFT_FILE, "w") as f:
                json.dump({"start": self.shift_start, "last_break": self.last_break}, f)
        except Exception as e:
            logger.warning(f"cannot save shift: {e}")

    def shift(self, action):
        now = time.time()
        with self.lock:
            if action == "start":
                self.shift_start, self.last_break = now, now
            elif action == "break":
                self.last_break = now
            elif action == "end":
                self.shift_start = self.last_break = None
            self._save_shift()

    def snapshot(self):
        with self.lock:
            return (list(self.events), self.shift_start, self.last_break,
                    self.pending_alert_at, self.pending_level)


driver_log = DriverLog()
TRUCK_ID = _read_truck_id()


def _driver_state():
    events, start, last_break, pending_at, pending_lvl = driver_log.snapshot()
    s = live.stats
    return {
        "truck": TRUCK_ID, "now": time.time(),
        "level": s.get("level_num", 0), "status": s.get("status"),
        "reasons": s.get("reasons", []), "perclos": s.get("perclos"), "yawns": s.get("yawns"),
        "shift_start": start, "last_break": last_break, "break_every_s": BREAK_EVERY_S,
        "pending_alert_at": pending_at, "pending_level": pending_lvl,
        "events": events[-100:],
    }


def _driver_shift(body: dict = Body(...)):
    action = str(body.get("action", ""))
    if action not in ("start", "break", "end"):
        return {"ok": False, "error": "bad action"}
    driver_log.shift(action)
    if action == "break":
        sec = driver_log.ack()
        company_sync.enqueue_ack(
            f"คนขับกดแวะพักรถแล้ว (ตอบสนองใน {sec} วิ)" if sec else "คนขับกดแวะพักรถแล้ว"
        )
    logger.info(f"Driver shift: {action}")
    return {"ok": True}


def _driver_ack(body: dict = Body(...)):
    kind = str(body.get("kind", "awake"))
    sec = driver_log.ack()
    if kind == "break":
        driver_log.shift("break")
        note = f"คนขับแวะพักรถแล้ว (ตอบสนองใน {sec} วิ)" if sec else "คนขับแวะพักรถแล้ว"
    else:
        note = f"คนขับกดยืนยันว่าตื่นแล้ว (ตอบสนองใน {sec} วิ)" if sec else "คนขับกดยืนยันว่าตื่นแล้ว"
    detector.closed_hist.clear()
    detector.yawn_times.clear()
    detector.narrow_since = None
    company_sync.enqueue_ack(note)
    logger.info(f"Driver ACK ({kind}): {note}")
    return {"ok": True, "note": note}


def _driver_test_alert(body: dict = Body(...)):
    lvl = 2 if int(body.get("level", 2)) >= 2 else 1
    reasons = ["microsleep 2.2s"] if lvl == 2 else ["eyes closed 1.3s"]
    driver_log.add(lvl, reasons)
    company_sync.enqueue(lvl, reasons)
    logger.info(f"Simulated driver alert level={lvl}")
    return {"ok": True}


ui.expose_api("GET", "/api/driver", _driver_state)
ui.expose_api("POST", "/api/shift", _driver_shift)
ui.expose_api("POST", "/api/driver/ack", _driver_ack)
ui.expose_api("POST", "/api/driver/test_alert", _driver_test_alert)


# --------------------------------------------------------------------------
# Map: GPS location (MCU -> Bridge "gps"), nearest rest stops, offline map file
# --------------------------------------------------------------------------
MAPS_DIR = os.path.join(APP_DIR, "maps")
MAP_FILE = os.path.join(MAPS_DIR, "thailand.pmtiles")    # offline map (protomaps vector tiles)
os.makedirs(MAPS_DIR, exist_ok=True)
navigator = Navigator(MAPS_DIR, logger)
try:
    Bridge.provide("gps", navigator.on_gps)
except Exception as e:
    logger.warning(f"Cannot receive GPS from MCU: {e}")


def _nav_state():
    d = navigator.state()
    d["offline_map"] = os.path.exists(MAP_FILE)
    return d


def _nav_pois(s: float, w: float, n: float, e: float):
    if n - s > 2.0 or e - w > 2.0:                   # zoomed far out: too many to draw
        return {"pois": [], "too_wide": True}
    return {"pois": navigator.pois_in(s, w, n, e)}


def _nav_test_location(body: dict = Body(...)):
    try:
        navigator.set_test_location(body.get("lat"), body.get("lon"), body.get("course"))
    except (TypeError, ValueError) as e:
        return {"ok": False, "error": str(e)}
    logger.info(f"Test location set: {body.get('lat')}, {body.get('lon')} course {body.get('course')}")
    company_sync.wake()
    return {"ok": True}


def _map_file():
    if not os.path.exists(MAP_FILE):
        return JSONResponse({"error": "offline map not installed"}, status_code=404)
    return FileResponse(MAP_FILE, media_type="application/octet-stream")


def _nav_indoor_location(body: dict = Body(...)):
    try:
        navigator.set_indoor_location(
            body.get("lat"), body.get("lon"), body.get("speed"), body.get("course")
        )
    except (TypeError, ValueError) as e:
        return {"ok": False, "error": str(e)}
    company_sync.wake()
    return {"ok": True}


ui.expose_api("GET", "/api/nav", _nav_state)
ui.expose_api("GET", "/api/pois", _nav_pois)
ui.expose_api("POST", "/api/test_location", _nav_test_location)
ui.expose_api("POST", "/api/indoor_location", _nav_indoor_location)
ui.expose_api("GET", "/maps/thailand.pmtiles", _map_file)


# --------------------------------------------------------------------------
# Company server sync (S1 events, S2 offline SQLite queue, S3 real-time heartbeat)
# --------------------------------------------------------------------------
COMPANY_URL_DEFAULT = "https://antimicrosleep-head.vercel.app"
COMPANY_URL_FILE = os.path.join(APP_DIR, "company_url.txt")
COMPANY_KEY_FILE = os.path.join(APP_DIR, "company_api_key.txt")
COMPANY_QUEUE_DB = os.path.join(DATA_DIR, "company_queue.sqlite")
COMPANY_HEARTBEAT_S = 5.0
APP_VERSION = "2026-11-20"


HOST_IP_FILE = os.path.join(DATA_DIR, "host_ip.json")   # written by tools/host_ip.sh on the board


def _host_network():
    """Board's Wi-Fi IP / network name as seen by the board itself (Docker hides it from us)."""
    try:
        with open(HOST_IP_FILE) as f:
            d = json.load(f)
        if time.time() - float(d.get("t", 0)) < 60 and d.get("ip"):
            return {"local_ip": str(d["ip"])[:40], "wifi": str(d.get("wifi") or "")[:64]}
    except Exception:
        pass
    return {}


class CompanySync:
    """Background sender: heartbeats + SQLite-queued alert events to company-web."""

    def __init__(self):
        self.wake_ev = threading.Event()
        self.db_lock = threading.Lock()
        self.pending_ack_note = None
        self.last_ok = None
        self.last_err = None
        os.makedirs(DATA_DIR, exist_ok=True)
        with self._db() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS queue ("
                "id TEXT PRIMARY KEY, t REAL, level INTEGER, reasons TEXT, "
                "lat REAL, lon REAL, speed REAL)"
            )

    def _db(self):
        return sqlite3.connect(COMPANY_QUEUE_DB, timeout=10.0)

    def enqueue_ack(self, note):
        with self.db_lock:
            self.pending_ack_note = note
        self.wake()

    def _read_key(self):
        try:
            with open(COMPANY_KEY_FILE) as f:
                k = f.read().strip()
            return k if k.startswith("ams_") else ""
        except Exception:
            return ""

    def _read_url(self):
        try:
            with open(COMPANY_URL_FILE) as f:
                u = f.read().strip().rstrip("/")
            if u.startswith("http"):
                return u
        except Exception:
            pass
        return COMPANY_URL_DEFAULT

    def enqueue(self, level, reasons):
        loc = navigator.location()
        uid = f"{int(time.time()*1000)}-{secrets.token_hex(4)}"
        with self.db_lock, self._db() as con:
            con.execute(
                "INSERT OR IGNORE INTO queue (id, t, level, reasons, lat, lon, speed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (uid, time.time(), int(level), json.dumps(list(reasons)),
                 loc.get("lat"), loc.get("lon"), loc.get("speed")),
            )
        self.wake()

    def wake(self):
        self.wake_ev.set()

    def _post(self, url, key, payload):
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-Truck-Code": _read_truck_id(),
            "User-Agent": f"anti-microsleep/{APP_VERSION}",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=8.0) as r:
            return json.loads(r.read().decode("utf-8"))

    def _step(self):
        key = self._read_key()
        base = self._read_url()
        loc = navigator.location()
        st = live.stats
        hb = {
            "lat": loc.get("lat"),
            "lon": loc.get("lon"),
            "speed": loc.get("speed"),
            "level": st.get("level_num", 0),
            "camera_ok": st.get("status") != "NO_CAMERA",
            "version": APP_VERSION,
        }
        hb.update(_host_network())                  # board Wi-Fi IP + network name (for /go QR)
        try:
            self._post(f"{base}/api/ingest/heartbeat", key, hb)
            self.last_ok, self.last_err = time.time(), None
        except Exception as e:
            self.last_err = str(e)
            return

        with self.db_lock, self._db() as con:
            rows = con.execute(
                "SELECT id, t, level, reasons, lat, lon, speed FROM queue ORDER BY t ASC LIMIT 50"
            ).fetchall()
        if rows:
            events = [
                {
                    "id": r[0], "t": round(r[1]), "level": r[2],
                    "reasons": json.loads(r[3] or "[]"),
                    "lat": r[4], "lon": r[5], "speed": r[6],
                }
                for r in rows
            ]
            try:
                self._post(f"{base}/api/ingest/event", key, {"events": events})
                with self.db_lock, self._db() as con:
                    con.executemany("DELETE FROM queue WHERE id = ?", [(r[0],) for r in rows])
                logger.info(f"Synced {len(rows)} alert event(s) to company server")
            except Exception as e:
                self.last_err = str(e)
                return

        with self.db_lock:
            ack_note = self.pending_ack_note
        if ack_note:
            try:
                self._post(f"{base}/api/ingest/ack", key, {"note": ack_note})
                with self.db_lock:
                    if self.pending_ack_note == ack_note:
                        self.pending_ack_note = None
                logger.info(f"Synced driver ACK to company server: {ack_note}")
            except Exception as e:
                self.last_err = str(e)

    def start(self):
        def _run():
            while True:
                self._step()
                self.wake_ev.wait(timeout=COMPANY_HEARTBEAT_S)
                self.wake_ev.clear()
        threading.Thread(target=_run, daemon=True).start()


company_sync = CompanySync()
company_sync.start()


def open_camera():
    global camera, next_camera_try
    try:
        cam = Camera(CAMERA_SOURCE, resolution=RESOLUTION, fps=FPS)
        cam.start()
        camera = cam
        logger.info(f"Camera started: {CAMERA_SOURCE or 'auto'} {RESOLUTION}@{FPS}")
    except Exception as e:
        camera = None
        next_camera_try = time.monotonic() + 5.0
        logger.warning(f"No camera yet ({e}). Retrying in 5 s...")


def loop():
    global camera, last_log, frames, last_level, current_fps, ready_chime_done, was_calibrated

    if camera is None:
        mcu.send(0, ST_NO_CAMERA)
        if alarm_sound is not None:
            alarm_sound.set_active(False)
        live.stats = {"status": "NO_CAMERA", "level": "OK", "level_num": 0, "time": time.time()}
        if time.monotonic() >= next_camera_try:
            open_camera()
        else:
            time.sleep(0.2)
        return

    try:
        frame = camera.capture()
    except Exception as e:
        logger.warning(f"Camera read failed ({e}); reopening")
        try:
            camera.stop()
        except Exception:
            pass
        camera = None
        return
    if frame is None:
        time.sleep(0.01)
        return
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    t = time.monotonic()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pts, _presence = engine.process(rgb)
    if person_checker is not None and pts is None:
        person_checker.submit(frame, t)          # only look for a body when no face is found
    if eye_model is not None and pts is not None:
        eye_model.submit(frame, pts, t, urgent=detector.ear_closed_since is not None,
                         ear=detector.ear_buf[-1] if detector.ear_buf else None)

    s = detector.update(pts, frame, t)
    try:                                   # research recorder (only while data/rec_on exists)
        global _eye_rec
        if "_eye_rec" not in globals():
            from eye_research import Recorder
            _eye_rec = Recorder(APP_DIR, MODEL_PATH, logger)
        _eye_rec(frame, pts, s, detector, t)
    except Exception as e:                 # never let research code break detection
        if t - getattr(loop, "_rec_err_t", 0) > 10:
            loop._rec_err_t = t
            logger.warning(f"eye recorder error: {e}")
    if SQUINT_TEST_LOG and pts is not None and s["ear"] is not None and t - getattr(loop, "_sq_t", 0) >= 0.4:
        loop._sq_t = t
        iris = iris_visible_fraction(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), pts)
        zone = ("CLOSED" if s["eyes_closed"] else "SQUINT" if s["squint"] else "open")
        bp = detector.base_pitch
        logger.info(f"SQTEST ear={s['ear']:.3f} thr={detector.ear_threshold:.3f}/"
                    f"{detector.squint_threshold:.3f} iris={iris if iris is None else round(iris, 2)} -> {zone}"
                    f" | pitch={s['pitch'] if s['pitch'] is None else round(s['pitch'], 1)}"
                    f" base={bp if bp is None else round(bp, 1)} roll={s['roll']:.0f}"
                    f" down={s['head_down']} tilt={s['head_tilt']}"
                    f" yaw={s['yaw'] if s['yaw'] is None else round(s['yaw'], 2)}"
                    f" ai={s.get('ai')}/{s.get('ai_note')}"
                    f" raw={s.get('raw_yaw') if s.get('raw_yaw') is None else round(s['raw_yaw'], 2)}"
                    f" byaw={round(detector.base_yaw, 2)} fx={float(pts[:, 0].mean()) / frame.shape[1]:.2f}"
                    f" haar={s.get('haar')} rate={s.get('haar_rate') if s.get('haar_rate') is None else round(s['haar_rate'], 2)}"
                    f" cf={s['closed_for']:.1f}"
                    f" | mouth={s['mouth'] if s['mouth'] is None else round(s['mouth'], 2)}"
                    f" base={detector.base_mouth if detector.base_mouth is None else round(detector.base_mouth, 2)}"
                    f" smile={s['smiling']}")
    mcu.send(s["level"], s["status"])
    if alarm_sound is not None:
        alarm_sound.set_active(s["level"] == 2)       # RED only, never on WARNING
    if not ready_chime_done:                          # first frame through camera + face AI
        ready_chime_done = True
        logger.info("System ready (camera + face AI running) - ready chime")
        if alarm_sound is not None:
            alarm_sound.play_once(ready_pcm)
    if detector.calibrated and not was_calibrated:
        logger.info("Driver's eyes learned - learned chime")
        if alarm_sound is not None:
            alarm_sound.play_once(learned_pcm)
    was_calibrated = detector.calibrated
    live.publish_stats(s, current_fps)
    live.publish_frame(frame, pts, s)
    frames += 1

    if s["level"] != last_level:
        logger.info(f"LEVEL {LEVEL_NAMES[last_level]} -> {LEVEL_NAMES[s['level']]}  {', '.join(s['reasons'])}")
        if s["level"] > last_level:               # record only rises (OK->WARN, ->ALARM)
            driver_log.add(s["level"], s["reasons"])
            company_sync.enqueue(s["level"], s["reasons"])
        else:
            company_sync.wake()
        last_level = s["level"]

    if DEBUG_SNAPSHOT and t - last_log >= 2.0:
        save_debug_frame(frame, pts, s)

    if t - last_log >= 2.0:
        fps = frames / (t - last_log) if last_log else 0.0
        current_fps = fps
        ear = f"{s['ear']:.3f}" if s["ear"] is not None else "-"
        mar = f"{s['mar']:.2f}" if s["mar"] is not None else "-"
        perclos = f"{s['perclos']:.0%}" if s["perclos"] is not None else "-"
        logger.info(
            f"[{STATUS_NAMES[s['status']]}] level={LEVEL_NAMES[s['level']]} "
            f"EAR={ear} (thr {detector.ear_threshold:.3f}) MAR={mar} PERCLOS={perclos} "
            f"yawns={s['yawns']} light={s['brightness']:.0f} conf={s['cam_conf']:.2f} fps={fps:.1f}")
        last_log, frames = t, 0


App.run(user_loop=loop)
