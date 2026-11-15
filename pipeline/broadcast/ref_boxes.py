"""Runs on the GPU machine. HockeyAI person boxes (skaters, goalies, referees) on every frame of a folder; bundle.py uses them to tell
referees from players, which jersey colour cannot (stripes, black jerseys with white trim and motion blur share one histogram).
usage: ref_boxes.py <frames_dir> <out.npz>
out.npz: dets (n, 7) float32 [frame, x1, y1, x2, y2, conf, referee (1) or player/goalie (0)]"""

import glob
import os
import sys
import time

import numpy as np
from ultralytics import YOLO

WEIGHTS = os.path.expanduser(f"{os.environ.get('MODELS', '~/store/models')}/simula/HockeyAI_model_weight.pt")
B = 8  # frames per batch
frames_dir, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{frames_dir}/*.jpg"))
model = YOLO(WEIGHTS)
t0 = time.time()
dets = []
cls = {k for k, n in model.names.items() if n in ("player", "goalie", "referee")}
for s in range(0, len(files), B):
    for k, r in enumerate(
        model.predict(files[s : s + B], imgsz=1920, conf=0.25, classes=sorted(cls), half=True, verbose=False, device=0)
    ):
        for box, c, j in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            dets.append([s + k, *box, c, model.names[int(j)] == "referee"])
    if (s // B) % 50 == 0:
        print(f"{s}/{len(files)} {time.time() - t0:.0f}s boxes={len(dets)}", flush=True)
np.savez(out, dets=np.array(dets, np.float32).reshape(-1, 7))
print("done", len(dets), f"{time.time() - t0:.0f}s")
