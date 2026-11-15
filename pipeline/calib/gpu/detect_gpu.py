"""Runs on the GPU machine (full_run.py copies it there). Person detection (YOLO) on every frame in a folder + jersey colour features per detection.
usage: detect_gpu.py <frames_dir> <out.npz> [weights] [imgsz]"""

import glob
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

frames_dir, out = sys.argv[1], sys.argv[2]
weights = sys.argv[3] if len(sys.argv) > 3 else "yolo11x.pt"
imgsz = int(sys.argv[4]) if len(sys.argv) > 4 else 1920
files = sorted(glob.glob(f"{frames_dir}/*.jpg"))
model = YOLO(weights)
t0 = time.time()
rows = []


def color_feat(img, b):
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 8:
        return [0] * 6
    c = img[max(y1 + int(0.2 * h), 0) : max(y1 + int(0.55 * h), 1), max(x1 + int(0.25 * w), 0) : max(x2 - int(0.25 * w), 1)]
    if c.size == 0:
        return [0] * 6
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    hh, ss, vv = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    yellow = ((hh > 15) & (hh < 35) & (ss > 110) & (vv > 120)).mean()
    blue = ((hh > 100) & (hh < 130) & (ss > 60) & (vv > 40)).mean()
    dark = (vv < 60).mean()
    white = ((ss < 40) & (vv > 170)).mean()
    return [yellow, blue, dark, white, ss.mean() / 255, vv.mean() / 255]


B = 16
for s in range(0, len(files), B):
    chunk = files[s : s + B]
    res = model.predict(chunk, imgsz=imgsz, conf=0.12, iou=0.5, classes=[0], half=True, verbose=False, device=0)
    for k, r in enumerate(res):
        img = None
        for bx, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if img is None:
                img = cv2.imread(chunk[k])
            rows.append([s + k, *bx, cf, *color_feat(img, bx)])
    if (s // B) % 25 == 0:
        print(f"{s}/{len(files)} {time.time() - t0:.0f}s dets={len(rows)}", flush=True)
np.save(out.replace(".npz", ".npy"), np.array(rows, np.float32))
print("done", len(rows), f"{time.time() - t0:.0f}s")
