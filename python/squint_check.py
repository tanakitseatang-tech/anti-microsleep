"""
Squint vs closed eyes, the way a person tells them apart.

When the eyelid points say "closed", a forced squint and a real closure can look the same
(EAR ~0.03-0.09 for both). Recorded tests (21 Nov 2026) showed four cues that differ:

  cue                         closed eyes          hard squint
  eyelid curve (lid line)     +0.013..+0.016       -0.009..+0.004   (closed lid arcs over the eyeball)
  AI browDown                 0.05..0.17           0.23..0.55       (squinting lowers the brows)
  AI eyeBlink - eyeSquint     +0.10..+0.16         -0.07..+0.03
  brow-to-lid gap vs normal   +0.05..+0.09         +0.01..+0.04     (closing drops the lid, brows stay)

With glasses the brows move less (browDown stays low), so the other three cues carry it.
Replayed on 4 recordings (2 with glasses): no real closure was excused.

A closure is excused as a squint only when at least SQUINT_VOTES_MIN of the cues agree
(median over the last 0.6 s). Any doubt -> it stays "closed" (safety first).
"""

import zipfile
from collections import deque

import numpy as np
from ai_edge_litert.interpreter import Interpreter

# MediaPipe face_blendshapes model input: these 146 of the 478 landmarks, in pixel coordinates
BS_IDX = [0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63,
          65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133,
          136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163,
          168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270,
          276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317,
          318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379,
          380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
          466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477]
I_BROW_DOWN = (1, 2)       # browDownLeft, browDownRight
I_BLINK = (9, 10)          # eyeBlinkLeft, eyeBlinkRight
I_SQUINT = (19, 20)        # eyeSquintLeft, eyeSquintRight

R_OUT = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
L_OUT = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
BROW_LID = ((105, 159), (334, 386))          # brow middle -> upper lid middle, per eye

# thresholds (from the recordings above)
CURVE_SQUINT_MAX = 0.006        # lid curve at or below this = flat lid (squint)
BROW_DOWN_SQUINT_MIN = 0.20     # brows lowered
BLINK_MINUS_SQUINT_MAX = 0.06   # AI says squint about as much as (or more than) blink
BROW_GAP_RISE_MAX = 0.045       # brow-to-lid gap grew less than this vs the open-eye normal
                                # (closed: +0.05..+0.09, squint: +0.01..+0.04 incl. glasses)
SQUINT_VOTES_MIN = 3            # cues that must agree
WINDOW_S = 0.6
MIN_FRAMES = 4


def _lid_curve(pts, D):
    ups = []
    for outline in (R_OUT, L_OUT):
        p = pts[outline].astype(np.float64)
        v = p[8] - p[0]
        n = np.array([-v[1], v[0]]) / (np.linalg.norm(v) or 1.0)
        if n[1] < 0:
            n = -n                                   # normal pointing down in the image
        mid = (p[0] + p[8]) / 2
        ups.append(float(np.mean((p[9:16] - mid) @ n)) / D)
    return float(np.mean(ups))


def _brow_gap(pts, D):
    return float(np.mean([np.linalg.norm(pts[b] - pts[l]) for b, l in BROW_LID])) / D


class SquintCheck:
    def __init__(self, task_path):
        data = zipfile.ZipFile(task_path).read("face_blendshapes.tflite")
        self.it = Interpreter(model_content=data, num_threads=1)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]
        assert len(BS_IDX) == 146
        self.open_gap = deque(maxlen=300)      # brow-to-lid gap while the eyes are open
        self.recent = deque()                  # (t, cues dict)

    def reset(self):
        self.open_gap.clear()
        self.recent.clear()

    def learn_open(self, pts):
        """Call on frames with clearly open eyes, face frontal (cheap: geometry only)."""
        D = float(np.linalg.norm(pts[33] - pts[263])) or 1.0
        self.open_gap.append(_brow_gap(pts, D))

    def check(self, pts, t):
        """Call on frames where the eyes look closed. Returns (is_squint, info)."""
        D = float(np.linalg.norm(pts[33] - pts[263])) or 1.0
        self.it.set_tensor(self.inp, pts[BS_IDX, :2].astype(np.float32)[None])
        self.it.invoke()
        bs = self.it.get_tensor(self.out).reshape(-1)
        cues = {"curve": _lid_curve(pts, D),
                "brow_down": float(np.mean(bs[list(I_BROW_DOWN)])),
                "blink_minus_squint": float(np.mean(bs[list(I_BLINK)]) - np.mean(bs[list(I_SQUINT)])),
                "gap_rise": (_brow_gap(pts, D) - float(np.median(self.open_gap))
                             if len(self.open_gap) >= 30 else None)}
        self.recent.append((t, cues))
        while self.recent and t - self.recent[0][0] > WINDOW_S:
            self.recent.popleft()
        if len(self.recent) < MIN_FRAMES:
            return False, None
        med = {}
        for k in cues:
            vals = [c[k] for _, c in self.recent if c[k] is not None]
            med[k] = float(np.median(vals)) if vals else None
        votes = [med["curve"] <= CURVE_SQUINT_MAX,
                 med["brow_down"] >= BROW_DOWN_SQUINT_MIN,
                 med["blink_minus_squint"] <= BLINK_MINUS_SQUINT_MAX]
        if med["gap_rise"] is not None:
            votes.append(med["gap_rise"] <= BROW_GAP_RISE_MAX)
        # without the open-eye gap learned yet, all 3 remaining cues must agree
        need = SQUINT_VOTES_MIN if len(votes) == 4 else 3
        med["votes"] = sum(votes)
        return sum(votes) >= need, med
