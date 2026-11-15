"""2D body keypoints for the cast from an independent detector (YOLO11x-pose, COCO-17), as evidence for fit.py next to the body
model's own keypoints: a second opinion that does not share SAM 3D Body's mistakes under motion blur.

Each detection goes to the cast member whose outline holds most of its confident keypoints (dilated 6 px, at least half of them),
each person at most once per frame, best first.

  bundle/pose2d.json   per frame {key: [[x, y, conf] x 17]} in source pixels

usage (from pipeline/, hockey venv):  pose2d.py [bundle]"""

import json
import sys

import cv2
import numpy as np
from ultralytics import YOLO

B = sys.argv[1] if len(sys.argv) > 1 else "bundle"
from paths import MODELS

MODEL = f"{MODELS}/yolo11x-pose.pt"


def main():
    S = json.load(open(f"{B}/scene.json"))
    tracks = json.load(open(f"{B}/cast.json"))["tracks"]
    model = YOLO(MODEL)
    out = []
    for i in range(S["n"]):
        res = model.predict(f"{B}/frames/{i + 1:03d}.jpg", imgsz=1920, conf=0.2, verbose=False)[0]
        kp = res.keypoints.data.cpu().numpy() if res.keypoints is not None else np.zeros((0, 17, 3))
        cands = []
        for t in tracks:
            m = t["mask"][i]
            if m < 0:
                continue
            mk = np.zeros((1080, 1920), np.uint8)
            cv2.fillPoly(mk, [np.round(np.array(S["outlines"][i][m])).astype(np.int32)], 1)
            mk = cv2.dilate(mk, np.ones((13, 13), np.uint8))
            for d, k in enumerate(kp):
                ok = k[:, 2] > 0.3
                if ok.sum() < 4:
                    continue
                xy = np.clip(np.round(k[ok, :2]).astype(int), 0, [1919, 1079])
                share = mk[xy[:, 1], xy[:, 0]].mean()
                if share >= 0.5:
                    cands.append((share * ok.sum(), t["key"], d))
        frame, used = {}, set()
        for _, key, d in sorted(cands, reverse=True):
            if str(key) in frame or d in used:
                continue
            frame[str(key)] = np.round(kp[d], 2).tolist()
            used.add(d)
        out.append(frame)
        if i % 50 == 0:
            print(i, len(kp), "detections", len(frame), "assigned", flush=True)
    json.dump(out, open(f"{B}/pose2d.json", "w"))


if __name__ == "__main__":
    main()
