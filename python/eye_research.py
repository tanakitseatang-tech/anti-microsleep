"""
Research recorder: squint vs closed eyes.

Active only while the file data/rec_on exists. Every frame it appends one CSV row with many
measurements (MediaPipe blendshapes + face geometry + image statistics) to data/rec/rec.csv and
every ~0.4 s saves a crop of both eyes to data/rec/img/. Nothing here changes the alarms.
"""

import csv
import os
import time
import zipfile

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# MediaPipe face_blendshapes model input: these 146 of the 478 landmarks, in pixel coordinates
BS_IDX = [0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63,
          65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133,
          136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163,
          168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270,
          276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317,
          318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379,
          380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466,
          468, 469, 470, 471, 472, 473, 474, 475, 476, 477]

BS_NAMES = ["_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
            "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
            "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
            "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
            "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
            "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
            "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
            "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
            "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
            "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
            "noseSneerLeft", "noseSneerRight"]
BS_KEEP = ["eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight", "cheekSquintLeft",
           "cheekSquintRight", "browDownLeft", "browDownRight", "browInnerUp", "eyeWideLeft",
           "eyeWideRight", "mouthSmileLeft", "mouthSmileRight", "noseSneerLeft", "noseSneerRight",
           "eyeLookDownLeft", "eyeLookDownRight"]

R_OUT = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
L_OUT = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# per eye: outline, outer corner, inner corner, upper-lid mid, lower-lid mid, brow mid, cheek below
EYES = (("R", R_OUT, 33, 133, 159, 145, 105, 118),
        ("L", L_OUT, 263, 362, 386, 374, 334, 347))


class Blendshapes:
    def __init__(self, task_path):
        data = zipfile.ZipFile(task_path).read("face_blendshapes.tflite")
        self.it = Interpreter(model_content=data, num_threads=1)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]
        assert len(BS_IDX) == 146, len(BS_IDX)

    def __call__(self, pts):
        x = pts[BS_IDX, :2].astype(np.float32)[None]
        self.it.set_tensor(self.inp, x)
        self.it.invoke()
        v = self.it.get_tensor(self.out).reshape(-1)
        return {n: float(v[BS_NAMES.index(n)]) for n in BS_KEEP}


def _eye_image_stats(gray, pts, outline, c_out, c_in, cheek):
    """Image statistics a person would look at: dark iris / white sclera inside the lid opening,
    wrinkles beside the outer corner (crow's feet) and under the eye."""
    p = pts[outline].astype(np.float32)
    w = float(np.linalg.norm(pts[c_out] - pts[c_in]))
    H, W = gray.shape
    res = {}
    # skin reference: patch on the cheek below the eye
    cx, cy = pts[cheek].astype(int)
    r = max(3, int(w * 0.12))
    skin = gray[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    skin_med = float(np.median(skin)) if skin.size else 128.0
    # lid opening (slightly widened) mask
    mask = np.zeros_like(gray, np.uint8)
    cen = p.mean(axis=0)
    poly = (cen + (p - cen) * np.array([1.0, 1.6])).astype(np.int32)
    cv2.fillPoly(mask, [poly], 1)
    vals = gray[mask > 0]
    if vals.size >= 10:
        res["dark_frac"] = float(np.mean(vals < skin_med * 0.55))
        res["white_frac"] = float(np.mean(vals > skin_med * 1.12))
        res["p5_rel"] = float(np.percentile(vals, 5) / max(skin_med, 1))
        res["std_rel"] = float(np.std(vals) / max(skin_med, 1))
    # crow's feet: patch just outside the outer corner; under-eye patch
    ox, oy = pts[c_out]
    ix = pts[c_in][0]
    side = 1 if ox > ix else -1
    for name, (px, py) in (("crow", (ox + side * w * 0.35, oy)),
                           ("under", ((ox + ix) / 2, oy + w * 0.35))):
        s = max(4, int(w * 0.25))
        x0, y0 = int(px - s), int(py - s)
        patch = gray[max(0, y0):min(H, y0 + 2 * s), max(0, x0):min(W, x0 + 2 * s)]
        if patch.size >= 25:
            lap = cv2.Laplacian(patch.astype(np.float32), cv2.CV_32F)
            res[name + "_tex"] = float(np.mean(np.abs(lap)) / max(float(patch.mean()), 1))
    return res


class Recorder:
    def __init__(self, app_dir, task_path, logger=None):
        self.flag = os.path.join(app_dir, "data", "rec_on")
        self.dir = os.path.join(app_dir, "data", "rec")
        self.task_path = task_path
        self.logger = logger
        self.bs = None
        self.writer = None
        self.fh = None
        self.cols = None
        self.last_img = 0.0
        self.last_check = 0.0
        self.on = False

    def _active(self, t):
        if t - self.last_check >= 1.0:
            self.last_check = t
            on = os.path.exists(self.flag)
            if on and not self.on:
                os.makedirs(os.path.join(self.dir, "img"), exist_ok=True)
                if self.bs is None:
                    self.bs = Blendshapes(self.task_path)
                if self.logger:
                    self.logger.info("REC on")
            if not on and self.on:
                if self.fh:
                    self.fh.close()
                self.fh = self.writer = None
                if self.logger:
                    self.logger.info("REC off")
            self.on = on
        return self.on

    def __call__(self, frame, pts, s, detector, t):
        if pts is None or not self._active(t):
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        row = {"t": round(time.time(), 2), "ear": s.get("ear"), "closed": int(bool(s.get("eyes_closed"))),
               "squint": int(bool(s.get("squint"))), "level": s.get("level"),
               "thr": round(detector.ear_threshold, 4), "base": detector.baseline,
               "yaw": s.get("yaw"), "pitch": s.get("pitch")}
        row.update(self.bs(pts))
        D = float(np.linalg.norm(pts[33] - pts[263])) or 1.0
        for tag, outline, c_out, c_in, up, lo, brow, cheek in EYES:
            row[tag + "_brow_lid"] = float(np.linalg.norm(pts[brow] - pts[up])) / D
            row[tag + "_lid_cheek"] = float(np.linalg.norm(pts[lo] - pts[cheek])) / D
            row[tag + "_open"] = float(np.linalg.norm(pts[up] - pts[lo])) / D
            row[tag + "_width"] = float(np.linalg.norm(pts[c_out] - pts[c_in])) / D
            row[tag + "_corner_dy"] = float(pts[c_out][1] - pts[c_in][1]) / D
            # how far the lid line sags below the corner-to-corner line (shape of the closed lid)
            p = pts[outline].astype(np.float64)
            v = p[8] - p[0]
            n = np.array([-v[1], v[0]]) / (np.linalg.norm(v) or 1.0)
            if n[1] < 0:
                n = -n
            mid = (p[0] + p[8]) / 2
            row[tag + "_curve_up"] = float(np.mean((p[9:16] - mid) @ n)) / D
            row[tag + "_curve_lo"] = float(np.mean((p[1:8] - mid) @ n)) / D
            for k, v2 in _eye_image_stats(gray, pts, outline, c_out, c_in, cheek).items():
                row[tag + "_" + k] = v2
        for k, v in list(row.items()):
            if isinstance(v, float):
                row[k] = round(v, 4)
        if self.writer is None:
            self.cols = list(row.keys())
            path = os.path.join(self.dir, "rec.csv")
            new = not os.path.exists(path)
            self.fh = open(path, "a", newline="")
            self.writer = csv.DictWriter(self.fh, fieldnames=self.cols, extrasaction="ignore")
            if new:
                self.writer.writeheader()
        self.writer.writerow(row)
        self.fh.flush()
        if t - self.last_img >= 0.4:
            self.last_img = t
            xs = pts[[70, 300, 118, 347, 33, 263], 0]
            ys = pts[[70, 300, 118, 347, 33, 263], 1]
            m = int(D * 0.15)
            x0, x1 = int(xs.min()) - m, int(xs.max()) + m
            y0, y1 = int(ys.min()) - m, int(ys.max()) + m
            H, W = frame.shape[:2]
            crop = frame[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
            if crop.size:
                crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                name = f"{row['t']:.1f}_ear{(s.get('ear') or 0):.3f}.jpg"
                cv2.imwrite(os.path.join(self.dir, "img", name), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
