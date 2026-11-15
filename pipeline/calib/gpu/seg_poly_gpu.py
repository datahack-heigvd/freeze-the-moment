"""Runs on the GPU machine (full_run.py copies it there). Person instance segmentation (players, referees, staff, spectators) on every frame of a folder; stores one simplified outline per person.
usage: seg_poly_gpu.py <frames_dir> <out.npz> [weights] [imgsz]
out.npz: dets (n, 7) float32 [frame, x1, y1, x2, y2, conf, n_points], pts int16 (sum n_points, 2) outline vertices in 1/8 px, concatenated in dets order"""

import glob
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

frames_dir, out = sys.argv[1], sys.argv[2]
weights = sys.argv[3] if len(sys.argv) > 3 else "yolo11x-seg.pt"
imgsz = int(sys.argv[4]) if len(sys.argv) > 4 else 1920
B, EPS = 4, 1.0  # frames per batch; outline simplification tolerance in px
files = sorted(glob.glob(f"{frames_dir}/*.jpg"))
model = YOLO(weights)
t0 = time.time()
dets, pts = [], []
for s in range(0, len(files), B):
    for k, r in enumerate(
        model.predict(
            files[s : s + B], imgsz=imgsz, conf=0.25, iou=0.5, classes=[0], retina_masks=True, half=True, verbose=False, device=0
        )
    ):
        if r.masks is None:
            continue
        for box, c, poly in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.masks.xy):
            if len(poly) < 3:
                continue
            a = cv2.approxPolyDP(poly.astype(np.float32).reshape(-1, 1, 2), EPS, True).reshape(-1, 2)
            if len(a) < 3:
                continue
            dets.append([s + k, *box, c, len(a)])
            pts.append(np.round(a * 8).astype(np.int16))
    if (s // B) % 100 == 0:
        print(f"{s}/{len(files)} {time.time() - t0:.0f}s persons={len(dets)}", flush=True)
np.savez(out, dets=np.array(dets, np.float32), pts=np.concatenate(pts))
print("done", len(dets), f"{time.time() - t0:.0f}s")
