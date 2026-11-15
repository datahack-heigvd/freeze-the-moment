"""How well the rendered people's colours match the real broadcast, in the source camera: per-pixel colour difference (CIE76 ΔE in
Lab) inside each cast member's own outline, eroded so the outline's edge doesn't count, wherever the render drew a person. Render
with TEX_HOLDOUT so no frame's own pixels colour its bodies (as in a novel view). Also the share of those pixels within ΔE 10 of the
real colour, and ΔE at quarter resolution, where per-vertex detail doesn't count, only the colour of each body region.
usage (from pipeline/, YOLO venv): texcheck.py <render_dir> [bundle]   e.g. after TEX_HOLDOUT=8 RES=960x544 OUT=/tmp/x render.py shot check"""

import json
import sys

import cv2
import numpy as np

d, B = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "bundle"
S = json.load(open(f"{B}/scene.json"))
T = json.load(open(f"{B}/cast.json"))["tracks"]
first = json.load(open(f"{d}/camera.json"))["source_start"] - S["start"]
first = round(first * S["fps"])
PEOPLE = np.array([[60, 110, 255], [255, 210, 40], [40, 220, 40]])[:, ::-1]  # home, away, other (BGR)
rgb, sem = cv2.VideoCapture(f"{d}/rgb.mp4"), cv2.VideoCapture(f"{d}/semantic.mp4")
de, de4, k = [], [], 0
while True:
    ok, f = rgb.read()
    ok2, s = sem.read()
    if not (ok and ok2):
        break
    i = first + k
    k += 1
    h, w = f.shape[:2]
    real = cv2.resize(cv2.imread(f"{B}/frames/{i + 1:03d}.jpg"), (w, h), interpolation=cv2.INTER_AREA)
    own = np.zeros((h, w), np.uint8)
    for t in T:
        m = t["mask"][i]
        if m >= 0:
            cv2.fillPoly(own, [np.round(np.array(S["outlines"][i][m]) * [w / 1920, h / 1080]).astype(np.int32)], 1)
    drawn = (np.abs(s[..., None, :].astype(int) - PEOPLE).sum(-1) < 60).any(-1)
    m = cv2.erode(own, np.ones((5, 5), np.uint8)).astype(bool) & drawn
    if m.sum() < 50:
        continue
    lab = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32) * [100 / 255, 1, 1] - [0, 128, 128]
    de.append(np.linalg.norm(lab(f) - lab(real), axis=-1)[m])
    q = lambda x: cv2.resize(cv2.GaussianBlur(x, (0, 0), 2), (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    m4 = cv2.resize(m.astype(np.uint8), (w // 4, h // 4), interpolation=cv2.INTER_AREA).astype(bool)
    de4.append(np.linalg.norm(lab(q(f)) - lab(q(real)), axis=-1)[m4])
de, de4 = np.concatenate(de), np.concatenate(de4)
print(
    json.dumps(
        dict(
            render=d,
            frames=k,
            pixels=int(de.size),
            dE=round(float(de.mean()), 2),
            dE_median=round(float(np.median(de)), 2),
            within10=round(float((de < 10).mean()), 3),
            dE_quarter=round(float(de4.mean()), 2),
        )
    )
)
