"""Rendered people vs the real outlines in the source camera: IoU and area ratio (render / real) of the cast's people per frame, and
stick cover: the share of the stick outlines' pixels outside everyone's outline (sticks.json) within 2 px of the render (a shaft is
about a pixel wide), and spill: the rendered pixels outside both, per real outline pixel.
usage (from pipeline/, YOLO venv): silhouette.py <render_dir> [bundle]   e.g. after RES=960x540 OUT=/tmp/x render.py shot check"""

import json
import sys

import cv2
import numpy as np

d, B = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "bundle"
S = json.load(open(f"{B}/scene.json"))
T = json.load(open(f"{B}/cast.json"))["tracks"]
K = json.load(open(f"{B}/sticks.json"))
SEM = np.array([[60, 110, 255], [255, 210, 40], [40, 220, 40], [220, 40, 220]])[:, ::-1]  # home, away, other, stick (BGR)
cap = cv2.VideoCapture(f"{d}/semantic.mp4")
ious, ratios = [], []
px = dict(stick=0, covered=0, real=0, spill=0)
i = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    h, w = f.shape[:2]
    real = np.zeros((h, w), np.uint8)
    for t in T:
        m = t["mask"][i]
        if m >= 0:
            cv2.fillPoly(real, [np.round(np.array(S["outlines"][i][m]) * [w / 1920, h / 1080]).astype(np.int32)], 1)
    near = np.abs(f[..., None, :].astype(int) - SEM).sum(-1) < 60
    ren = near[..., :3].any(-1)
    drawn = near.any(-1)
    stick = np.zeros_like(real)
    for k in K[i]:
        cv2.fillPoly(stick, [np.round(np.array(k["poly"]) * [w / 1920, h / 1080]).astype(np.int32)], 1)
    stick &= real == 0
    px["stick"] += stick.sum()
    px["covered"] += ((cv2.dilate(drawn.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (stick > 0)).sum()
    px["real"] += real.sum()
    px["spill"] += (drawn & (real == 0) & (stick == 0)).sum()
    if real.sum() > 500:
        ious.append((ren & (real > 0)).sum() / (ren | (real > 0)).sum())
        ratios.append(ren.sum() / real.sum())
    i += 1
print(
    f"{d}: frames {len(ious)}  IoU {np.mean(ious):.3f}  area render/real {np.median(ratios):.3f}  "
    f"stick cover {px['covered'] / max(px['stick'], 1):.3f}  spill {px['spill'] / max(px['real'], 1):.3f}"
)
