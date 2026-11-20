"""
Face mesh engine using LiteRT (TensorFlow Lite) directly.

Why not the `mediapipe` package? Its ARM64 wheels need ARMv8.2 CPU features
(dotprod, fp16, ...). The UNO Q's Cortex-A53 is ARMv8.0, so mediapipe crashes
with SIGILL. LiteRT runs the very same models (taken from face_landmarker.task)
on the A53 at ~25 ms per frame.

Pipeline (same as MediaPipe Face Landmarker):
  1. BlazeFace short-range detector (128x128) -> face box + eye keypoints
  2. Rotated square crop around the face (1.5x box), eyes horizontal
  3. Face landmarks model (256x256) -> 478 points (same indices as Face Mesh)
  4. Next frame: crop is computed from the previous landmarks (tracking),
     the detector runs again only when the face is lost.
"""

import math
import os
import zipfile

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

DET_SIZE = 128
LM_SIZE = 256
DET_SCORE_MIN = 0.5
PRESENCE_MIN = 0.5
ROI_SCALE = 1.5


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _blazeface_anchors():
    # SSD anchors for BlazeFace short range: strides [8,16,16,16], 2 anchors each
    strides = [8, 16, 16, 16]
    anchors, i = [], 0
    while i < len(strides):
        stride, n = strides[i], 0
        while i < len(strides) and strides[i] == stride:
            n += 2
            i += 1
        fm = math.ceil(DET_SIZE / stride)
        for y in range(fm):
            for x in range(fm):
                anchors.extend([((x + 0.5) / fm, (y + 0.5) / fm)] * n)
    return np.array(anchors, dtype=np.float32)          # (896, 2)


def _upright(theta):
    # never produce an upside-down crop
    if math.cos(theta) < 0:
        theta += math.pi
    return theta


class FaceMeshEngine:
    def __init__(self, task_path, threads=4):
        model_dir = os.path.join(os.path.dirname(task_path), "tflite")
        det_path = os.path.join(model_dir, "face_detector.tflite")
        lm_path = os.path.join(model_dir, "face_landmarks_detector.tflite")
        if not (os.path.isfile(det_path) and os.path.isfile(lm_path)):
            with zipfile.ZipFile(task_path) as z:
                z.extract("face_detector.tflite", model_dir)
                z.extract("face_landmarks_detector.tflite", model_dir)

        self.det = Interpreter(model_path=det_path, num_threads=threads)
        self.det.allocate_tensors()
        self.det_in = self.det.get_input_details()[0]["index"]
        self.det_boxes = self.det_scores = None
        for o in self.det.get_output_details():
            if o["shape"][-1] == 16:
                self.det_boxes = o["index"]
            else:
                self.det_scores = o["index"]

        self.lm = Interpreter(model_path=lm_path, num_threads=threads)
        self.lm.allocate_tensors()
        self.lm_in = self.lm.get_input_details()[0]["index"]
        self.lm_points = self.lm_presence = None
        for o in self.lm.get_output_details():
            size = int(np.prod(o["shape"]))
            if size == 478 * 3:
                self.lm_points = o["index"]
            elif o["name"] == "Identity_1":
                self.lm_presence = o["index"]

        self.anchors = _blazeface_anchors()
        self.roi = None          # (cx, cy, size, theta) in pixels / radians
        self.used_detector = False

    # ---------------------------------------------------------------- detect
    def _detect(self, rgb):
        h, w = rgb.shape[:2]
        side = max(w, h)
        px, py = (side - w) // 2, (side - h) // 2
        sq = cv2.copyMakeBorder(rgb, py, side - h - py, px, side - w - px, cv2.BORDER_CONSTANT)
        x = cv2.resize(sq, (DET_SIZE, DET_SIZE)).astype(np.float32) / 127.5 - 1.0
        self.det.set_tensor(self.det_in, x[None])
        self.det.invoke()
        raw = self.det.get_tensor(self.det_boxes)[0]            # (896, 16)
        scores = _sigmoid(self.det.get_tensor(self.det_scores)[0, :, 0])
        best = int(np.argmax(scores))
        if scores[best] < DET_SCORE_MIN:
            return None
        r, a = raw[best] / DET_SIZE, self.anchors[best]
        cx, cy = (r[0] + a[0]) * side - px, (r[1] + a[1]) * side - py
        bw, bh = r[2] * side, r[3] * side
        # keypoint 0 = right eye, 1 = left eye (image left -> right)
        ex0, ey0 = (r[4] + a[0]) * side - px, (r[5] + a[1]) * side - py
        ex1, ey1 = (r[6] + a[0]) * side - px, (r[7] + a[1]) * side - py
        theta = _upright(math.atan2(ey1 - ey0, ex1 - ex0))
        return (cx, cy, max(bw, bh) * ROI_SCALE, theta)

    # ------------------------------------------------------------- landmarks
    @staticmethod
    def _affine(roi):
        cx, cy, s, th = roi
        c, si = math.cos(th), math.sin(th)
        k = s / LM_SIZE
        A = np.array([[c * k, -si * k], [si * k, c * k]], dtype=np.float32)
        b = np.array([cx, cy], dtype=np.float32) - A @ np.array([LM_SIZE / 2, LM_SIZE / 2], np.float32)
        return A, b                                            # crop(u,v) -> image(x,y)

    def _landmarks(self, rgb, roi):
        A, b = self._affine(roi)
        M = np.hstack([A, b[:, None]])
        crop = cv2.warpAffine(rgb, M, (LM_SIZE, LM_SIZE),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT)
        self.lm.set_tensor(self.lm_in, (crop.astype(np.float32) / 255.0)[None])
        self.lm.invoke()
        uv = self.lm.get_tensor(self.lm_points).reshape(478, 3)[:, :2]
        presence = float(_sigmoid(self.lm.get_tensor(self.lm_presence).ravel()[0]))
        pts = uv @ A.T + b                                     # back to image pixels
        return pts, presence

    @staticmethod
    def _roi_from_landmarks(pts):
        p0, p1 = pts[33], pts[263]                             # outer eye corners
        th = _upright(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, s], [-s, c]], dtype=np.float32)      # image -> rotated frame
        q = pts @ R.T
        lo, hi = q.min(axis=0), q.max(axis=0)
        center = R.T @ ((lo + hi) / 2)
        size = float(max(hi - lo)) * ROI_SCALE
        return (float(center[0]), float(center[1]), size, th)

    # ----------------------------------------------------------------- public
    def process(self, rgb):
        """rgb: HxWx3 uint8 RGB. Returns (points (478,2) in pixels or None, presence)."""
        self.used_detector = False
        if self.roi is None:
            self.roi = self._detect(rgb)
            self.used_detector = True
            if self.roi is None:
                return None, 0.0
        pts, presence = self._landmarks(rgb, self.roi)
        if presence < PRESENCE_MIN:
            self.roi = None
            return None, presence
        self.roi = self._roi_from_landmarks(pts)
        return pts, presence
