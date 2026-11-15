"""A detector that tells the teams apart: every person on the ice boxed as a skater of team 0, of team 1, or an official, trained on
labels from the tracking feed.

On a frame of the match (outside the goal windows, HOLD_OUT s around them: those are the evaluation) every person outline the AR
pipeline found (store/ar/<seg>/masks.bin) is matched to the feed's tracked players (tracks.json ctl: position and team) by projecting
their feet through the frame's camera: the outline whose bottom centre is nearest, in outline heights, one to one. A match counts only
if it is close (MATCH) and clearly the best (the next candidate SECOND times further), since the feed runs 0.2 to 0.5 s behind the
video and players in a pack swap. An outline whose ice point is further than FREE m from every tracked or detected skater is an official.
Every other outline is left unlabelled and its box blanked out of the image (grey), so it teaches neither class nor background.

  labels <frames> <out>       (YOLO venv) YOLO dataset from the frames (puck_train frames: NNNNNN.jpg, match frame N)
  train  <dataset> <init> <name>   fine-tune at IMGSZ
  eval   <weights> <dataset>  on the validation frames: per labelled person, whether a box of the right class covers it (IoU >
                              0.5), against a kit-colour baseline (each box's upper half, nearest team colour from the training labels)"""

import glob
import json
import os
import sys

import numpy as np
from paths import AR, GOALS

HOLD_OUT, SPLIT, IMGSZ = 15, 0.8, 1280
MATCH, SECOND, FREE, MIN_H = 0.35, 2.0, 1.5, 24  # outline heights, ratio, m, px
NAMES = {0: "team0", 1: "team1", 2: "official"}


def read_masks(path):
    raw = open(path, "rb").read()
    n = int(np.frombuffer(raw, np.uint32, 1)[0])
    offs = np.frombuffer(raw, np.uint32, n + 1, 4)
    data = np.frombuffer(raw, np.int16, offset=4 * (n + 2))
    frames = []
    for i in range(n):
        chunk, polys, j = data[offs[i] : offs[i + 1]], [], 0
        while j < len(chunk):
            m = int(chunk[j])
            polys.append(chunk[j + 1 : j + 1 + 2 * m].reshape(-1, 2).astype(np.float32) / 8)
            j += 1 + 2 * m
        frames.append(polys)
    return frames


def project(cam, T, P):
    f, k, R, C = cam[0], cam[1], cam[2:11].reshape(3, 3), cam[11:14]
    pc = (np.atleast_2d(P) - C) @ R.T
    uv = f * pc[:, :2] / pc[:, 2:3]
    return uv * (1 + k * (uv**2).sum(1, keepdims=True) / T["fref"] ** 2) + [T["cx"], T["cy"]]


def to_ice(cam, T, px):
    f, k, R, C = cam[0], cam[1], cam[2:11].reshape(3, 3), cam[11:14]
    d = np.atleast_2d(px) - [T["cx"], T["cy"]]
    u = d.copy()
    for _ in range(8):
        u = d / (1 + k * (u**2).sum(1, keepdims=True) / T["fref"] ** 2)
    ray = np.c_[u / f, np.ones(len(u))] @ R
    t = -C[2] / ray[:, 2]
    return (C + t[:, None] * ray)[:, :2]


def frame_labels(T, cam, polys, a):
    """(box [x0, y0, x1, y1], class or -1 for unlabelled) per outline of frame a of the segment."""
    ctl = np.array(T["ctl"][a]).reshape(-1, 4)
    ctl = ctl[ctl[:, 0] >= 0]
    rings = np.array(T["rings"][a]).reshape(-1, 4)
    boxes = [np.r_[p.min(0), p.max(0)] for p in polys]
    out = [[b, -1] for b in boxes]
    if not polys or not np.abs(cam).sum():
        return out
    feet = project(cam, T, np.c_[ctl[:, 1:3] / 100, np.zeros(len(ctl))]) if len(ctl) else np.zeros((0, 2))
    bottoms = np.array([[p[:, 0].mean(), p[:, 1].max()] for p in polys])
    heights = np.array([max(np.ptp(p[:, 1]), 20) for p in polys])
    if len(ctl):
        D = np.linalg.norm(feet[:, None] - bottoms[None], axis=2) / heights[None]
        for j in range(len(ctl)):
            m = int(np.argmin(D[j]))
            others = np.delete(D[:, m], j)
            rival = np.delete(D[j], m)
            if (
                D[j, m] < MATCH
                and (not len(others) or others.min() > SECOND * D[j, m])
                and (not len(rival) or rival.min() > SECOND * D[j, m])
            ):
                out[m][1] = int(ctl[j, 3])
    skaters = np.r_[ctl[:, 1:3], rings[:, 1:3]] / 100 if len(rings) else ctl[:, 1:3] / 100
    ice = to_ice(cam, T, bottoms)
    for m in range(len(polys)):
        if (
            out[m][1] < 0
            and (not len(skaters) or np.linalg.norm(skaters - ice[m], axis=1).min() > FREE)
            and abs(ice[m][0]) < 30
            and abs(ice[m][1]) < 15
        ):
            out[m][1] = 2
    return [(b, c) if b[3] - b[1] >= MIN_H else (b, -1) for b, c in out]


def labels(frames_dir, out):
    import cv2

    goals = [json.load(open(p))["start"] for p in glob.glob(f"{GOALS}/*/bundle/scene.json")]
    segs = []
    for d in sorted(glob.glob(f"{AR}/c*/")):
        T = json.load(open(d + "tracks.json"))
        segs.append((T, np.fromfile(d + "camera.bin", np.float32).reshape(-1, 14).astype(np.float64), d))
    have = {int(f[:6]) for f in os.listdir(frames_dir) if f.endswith(".jpg")}
    keep = sorted(N for N in have if N - 1 in have and N + 1 in have)  # the middle frame of each triple
    keep = [N for N in keep if not any(-HOLD_OUT < N / 25 - g < 20 + HOLD_OUT for g in goals)]
    split = np.quantile(keep, SPLIT)
    stats = np.zeros(4, int)
    masks = {}
    for part in ("train", "val"):
        os.makedirs(f"{out}/{part}/images", exist_ok=True)
        os.makedirs(f"{out}/{part}/labels", exist_ok=True)
    for N in keep:
        seg = next(((T, cam, d) for T, cam, d in segs if T["n0"] <= N < T["n0"] + len(cam)), None)
        if seg is None:
            continue
        T, cams, d = seg
        a = N - T["n0"]
        if d not in masks:
            masks = {d: read_masks(d + "masks.bin")}
        lab = frame_labels(T, cams[a], masks[d][a], a)
        if not any(c >= 0 for _, c in lab):
            continue
        img = cv2.imread(f"{frames_dir}/{N:06d}.jpg")
        H, W = img.shape[:2]
        part = "train" if N < split else "val"
        lines = []
        for (x0, y0, x1, y1), c in lab:
            stats[c] += 1
            if c < 0:
                img[int(max(y0, 0)) : int(y1) + 1, int(max(x0, 0)) : int(x1) + 1] = 114
            else:
                lines.append(f"{c} {(x0 + x1) / 2 / W:.5f} {(y0 + y1) / 2 / H:.5f} {(x1 - x0) / W:.5f} {(y1 - y0) / H:.5f}")
        cv2.imwrite(f"{out}/{part}/images/{N:06d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        open(f"{out}/{part}/labels/{N:06d}.txt", "w").write("\n".join(lines) + "\n")
    names = ", ".join(f"{k}: {v}" for k, v in NAMES.items())
    open(f"{out}/data.yaml", "w").write(f"path: {out}\ntrain: train/images\nval: val/images\nnames: {{{names}}}\n")
    print(
        dict(frames=len(keep), team0=int(stats[0]), team1=int(stats[1]), official=int(stats[2]), unlabelled=int(stats[3])),
        flush=True,
    )


def train(data, init, name):
    from ultralytics import YOLO

    YOLO(init).train(
        data=f"{data}/data.yaml",
        imgsz=IMGSZ,
        epochs=80,
        batch=8,
        mosaic=1.0,
        fliplr=0.5,
        hsv_h=0.0,  # hue carries the team
        project=os.path.join(os.path.dirname(data), "runs"),
        name=name,
        exist_ok=True,
        patience=20,
        workers=8,
        verbose=False,
    )


def iou(a, b):
    x0, y0, x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1]), np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter)


def evaluate(weights, data):
    import cv2
    from ultralytics import YOLO

    m = YOLO(weights)

    def read(part):
        for p in sorted(glob.glob(f"{data}/{part}/labels/*.txt")):
            img = cv2.imread(p.replace("/labels/", "/images/").replace(".txt", ".jpg"))
            H, W = img.shape[:2]
            L = np.loadtxt(p, ndmin=2)
            yield (
                img,
                np.c_[
                    L[:, 0],
                    (L[:, 1] - L[:, 3] / 2) * W,
                    (L[:, 2] - L[:, 4] / 2) * H,
                    (L[:, 1] + L[:, 3] / 2) * W,
                    (L[:, 2] + L[:, 4] / 2) * H,
                ],
            )

    def kit(img, b):
        x0, y0, x1, y1 = b.astype(int)
        crop = img[max(y0, 0) : max(y0 + (y1 - y0) // 2, y0 + 1), max(x0, 0) : max(x1, x0 + 1)]
        return cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0) if crop.size else np.zeros(3)

    colours = {0: [], 1: []}
    for img, L in read("train"):
        for c, *b in L:
            if c < 2 and len(colours[int(c)]) < 4000:
                colours[int(c)].append(kit(img, np.array(b)))
    centre = {c: np.median(v, 0) for c, v in colours.items()}
    hit = {k: np.zeros(3) for k in ("detector", "found", "colour")}
    total = np.zeros(3)  # colour: every labelled skater, found or not
    for img, L in read("val"):
        x = m.predict(img, imgsz=IMGSZ, conf=0.25, verbose=False)[0].boxes
        db, dc = x.xyxy.cpu().numpy(), x.cls.cpu().numpy().astype(int)
        for c, *b in L:
            c = int(c)
            b = np.array(b)
            total[c] += 1
            o = iou(b, db) if len(db) else np.zeros(0)
            if len(o) and o.max() > 0.5:
                hit["found"][c] += 1
                hit["detector"][c] += dc[o.argmax()] == c
            if c < 2:
                hit["colour"][c] += min(centre, key=lambda t: np.linalg.norm(kit(img, b) - centre[t])) == c
    res = {k: (v / np.maximum(total, 1)).round(3).tolist() for k, v in hit.items()}
    print(
        json.dumps(
            dict(
                people=total.astype(int).tolist(),
                found=res["found"],
                right_class=res["detector"],
                colour_baseline_teams=res["colour"][:2],
                team_accuracy_when_found=[round(float(hit["detector"][c] / max(hit["found"][c], 1)), 3) for c in range(3)],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    a = sys.argv[2:]
    {"labels": lambda: labels(*a), "train": lambda: train(*a), "eval": lambda: evaluate(*a)}[sys.argv[1]]()
