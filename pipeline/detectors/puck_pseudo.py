"""Self-training for the puck detector on the 2023 and 2024 Spengler Cup broadcasts (same rink, no feed): the detector's own boxes,
kept only where they form a puck track, become labels for crops like puck_train.py's; trained together with the feed-labelled 2025
crops. A puck track: a box in at least RUN frames of LINK px steps, confident (PEAK) somewhere along it, alone in its frames, in a
game view (at least ICE of the frame is ice), on ice or boards (the median brightness BRIGHT round it) and moving against the camera's
own pan (spanning MOVE px once the frame-to-frame shift of the whole picture is taken out): the detector's false boxes are crowd
lights, faces and marks on the boards.

  clips   <out>                               (local) WINDOWS windows of SECONDS from each game (between 10% and 95% of its length,
                                              fixed seed), every frame, to <out>/<game>/<window>/<frame>.jpg
  detect  <clips> <weights> <out.json>        (YOLO venv) puck boxes per frame of every window, with the camera's pan
  label   <clips> <boxes.json> <out.json>     the boxes of the puck tracks
  dataset <clips> <labels.json> <base> <out>  <base>'s YOLO crops (puck_train.py dataset) plus a puck crop for every EVERY-th
                                              tracked frame, to train, but for the HELD_OUT games
  compare <clips> <out.jpg> <weights>...      on the HELD_OUT games, each detector's most confident box (over CONF_SHOWN) in
                                              every SHEET_EVERY-th frame: how many frames have one, and a sheet of them (one row per
                                              detector, the same frames) to count real pucks by eye"""

import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from paths import STORE

GAMES = sorted(glob.glob(f"{STORE}/data/thirdparty/spengler-cup-202[34]/*.mkv"))
WINDOWS, SECONDS = 12, 3.0
CONF, LINK, GAP, RUN, PEAK, EVERY, CROP, MIN_BOX = 0.05, 60, 2, 10, 0.5, 2, 640, 8
ICE, BRIGHT, MOVE, NEAR = 0.25, 140, 15, 20  # share of the frame; grey level; px; px either side of the box
HELD_OUT = ("AHjuTBhfoOM", "x7LF4LzjnBE", "Dfi7vg7ty30", "NEAje00Mqak")  # two 2023 and two 2024 games never trained on
CONF_SHOWN, SHEET_EVERY, SHEET_N = 0.3, 7, 60


def clips(out):
    rng = np.random.default_rng(0)
    for g in GAMES:
        key = g.rsplit("[", 1)[1][:11]
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", g]))
        for w, t in enumerate(np.sort(rng.uniform(0.1, 0.95, WINDOWS)) * dur):
            d = f"{out}/{key}/{w:02d}"
            if os.path.isdir(d) and len(os.listdir(d)) > 0:
                continue
            os.makedirs(d)
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", g, "-t", str(SECONDS), "-q:v", "3", f"{d}/%04d.jpg"],
                check=True,
            )
        print(key, flush=True)


def tracks(boxes):
    """Boxes per frame [[u, v, w, h, conf], ...] -> tracks [[(frame, box index), ...], ...]: each box joins the nearest track that
    ended at most GAP frames before within LINK px per frame, else starts one."""
    out, live = [], []
    for f, bs in enumerate(boxes):
        taken = set()
        for i in np.argsort([-b[4] for b in bs]):
            u, v = bs[i][:2]
            best = None
            for t in live:
                lf, li = t[-1]
                gap = f - lf
                if 0 < gap <= GAP + 1 and id(t) not in taken:
                    d = np.hypot(u - boxes[lf][li][0], v - boxes[lf][li][1]) / gap
                    if d < LINK and (best is None or d < best[0]):
                        best = (d, t)
            if best:
                best[1].append((f, i))
                taken.add(id(best[1]))
            else:
                t = [(f, i)]
                live.append(t)
                out.append(t)
                taken.add(id(t))
        live = [t for t in live if f - t[-1][0] <= GAP]
    return out


def ice(img):
    """Share of the picture that is ice: bright and unsaturated."""
    import cv2

    h = cv2.cvtColor(cv2.resize(img, (480, 270)), cv2.COLOR_BGR2HSV)
    return float(((h[..., 2] > 170) & (h[..., 1] < 40)).mean())


def detect(clips_dir, weights, out):
    import cv2
    from ultralytics import YOLO

    m = YOLO(weights)
    pk = next(k for k, v in m.names.items() if v.lower() == "puck")
    res = {}
    for d in sorted(glob.glob(f"{clips_dir}/*/*/")):
        win = res[os.path.relpath(d, clips_dir)] = dict(files=[], boxes=[], pan=[], ice=[])
        prev = None
        for p in sorted(glob.glob(f"{d}*.jpg")):
            img = cv2.imread(p)
            x = m.predict(img, imgsz=1920, conf=CONF, classes=[pk], verbose=False)[0].boxes
            g = np.float32(cv2.cvtColor(cv2.resize(img, (480, 270)), cv2.COLOR_BGR2GRAY))
            s = img.shape[1] / 480
            (dx, dy), _ = cv2.phaseCorrelate(prev, g) if prev is not None else ((0.0, 0.0), 0)
            prev = g
            box = [
                [
                    *map(float, b),
                    float(c),
                    float(
                        np.median(
                            cv2.cvtColor(
                                img[max(int(b[1]) - NEAR, 0) : int(b[1]) + NEAR, max(int(b[0]) - NEAR, 0) : int(b[0]) + NEAR],
                                cv2.COLOR_BGR2GRAY,
                            )
                        )
                    ),
                ]
                for b, c in zip(x.xywh.tolist(), x.conf.tolist())
            ]
            win["files"].append(os.path.basename(p))
            win["boxes"].append(box)
            win["pan"].append([dx * s, dy * s])
            win["ice"].append(ice(img))
        print(d, sum(map(len, win["boxes"])), "boxes", flush=True)
    json.dump(res, open(out, "w"))


def label(clips_dir, boxes_path, out):
    res = {}
    stats = dict(windows=0, frames=0, tracks=0, kept=0)
    for w, win in json.load(open(boxes_path)).items():
        boxes, pan = win["boxes"], np.cumsum(win["pan"], 0)
        good = []
        for t in tracks(boxes):
            if len(t) < RUN or max(boxes[f][i][4] for f, i in t) < PEAK:
                continue
            if np.mean([win["ice"][f] for f, _ in t]) < ICE or np.median([boxes[f][i][5] for f, i in t]) < BRIGHT:
                continue
            path = np.array([boxes[f][i][:2] for f, i in t]) - pan[[f for f, _ in t]]
            if np.ptp(path, 0).max() < MOVE:
                continue
            good.append(t)
        count = {}
        for t in good:
            for f, _ in t:
                count[f] = count.get(f, 0) + 1
        keep = {f"{w}/{win['files'][f]}": boxes[f][i][:4] for t in good for f, i in t if count[f] == 1}
        res.update(keep)
        stats["windows"] += 1
        stats["frames"] += len(boxes)
        stats["tracks"] += len(good)
        stats["kept"] += len(keep)
    json.dump(res, open(out, "w"))
    print(stats)


def dataset(clips_dir, labels_path, base, out):
    import cv2

    L = json.load(open(labels_path))
    rng = np.random.default_rng(0)
    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(base, out, ignore=shutil.ignore_patterns("*.cache"))
    n = 0
    for k, rel in enumerate(sorted(r for r in L if r.split("/")[0] not in HELD_OUT)):
        if k % EVERY:
            continue
        (u, v, w, h), img = L[rel], cv2.imread(f"{clips_dir}/{rel}")
        H, W = img.shape[:2]
        name = "p_" + rel.replace("/", "_")[:-4]
        ox, oy = (
            int(np.clip(u - rng.uniform(60, CROP - 60), 0, W - CROP)),
            int(np.clip(v - rng.uniform(60, CROP - 60), 0, H - CROP)),
        )
        cv2.imwrite(f"{out}/train/images/{name}.jpg", img[oy : oy + CROP, ox : ox + CROP], [cv2.IMWRITE_JPEG_QUALITY, 95])
        s = max(MIN_BOX, w, h)
        open(f"{out}/train/labels/{name}.txt", "w").write(
            f"0 {(u - ox) / CROP:.5f} {(v - oy) / CROP:.5f} {s / CROP:.5f} {s / CROP:.5f}\n"
        )
        n += 1
    open(f"{out}/data.yaml", "w").write(f"path: {out}\ntrain: train/images\nval: val/images\nnames: {{0: puck}}\n")
    print("pseudo crops", n)


def compare(clips_dir, out, *weights):
    import cv2
    from ultralytics import YOLO

    files = [f for g in HELD_OUT for f in sorted(glob.glob(f"{clips_dir}/{g}/*/*.jpg"))][::SHEET_EVERY]
    tops = []
    for w in weights:
        m = YOLO(w)
        pk = next(k for k, v in m.names.items() if v.lower() == "puck")
        top = []
        for p in files:
            x = m.predict(p, imgsz=1920, conf=CONF_SHOWN, classes=[pk], verbose=False)[0].boxes
            top.append(x.xywh[int(x.conf.argmax())][:2].tolist() if len(x) else None)
        tops.append(top)
        print(w, "frames with a box", sum(t is not None for t in top), "of", len(files), flush=True)
    both = [i for i in range(len(files)) if any(t[i] is not None for t in tops)]
    pick = [both[i] for i in np.linspace(0, len(both) - 1, min(SHEET_N, len(both))).astype(int)]
    rows = []
    for top in tops:
        tiles = []
        for i in pick:
            img = cv2.imread(files[i])
            H, W = img.shape[:2]
            if top[i] is None:
                tiles.append(np.zeros((96, 96, 3), np.uint8))
                continue
            u, v = top[i]
            x0, y0 = int(np.clip(u - 48, 0, W - 96)), int(np.clip(v - 48, 0, H - 96))
            t = img[y0 : y0 + 96, x0 : x0 + 96].copy()
            cv2.circle(t, (int(u - x0), int(v - y0)), 14, (0, 0, 255), 1)
            tiles.append(t)
        rows.append(np.vstack([np.hstack(tiles[j : j + 20]) for j in range(0, len(tiles), 20)]))
    cv2.imwrite(out, np.vstack([np.pad(r, ((0, 12), (0, 0), (0, 0))) for r in rows]))
    print("sheet", len(pick), "frames per row block")


if __name__ == "__main__":
    {"clips": clips, "detect": detect, "label": label, "dataset": dataset, "compare": compare}[sys.argv[1]](*sys.argv[2:])
