"""A puck detector for this broadcast, trained with labels from the tracking feed, and its check against the feed.

The feed has the puck's rink position at event snapshots (live/game.json frames; it runs 0 to 0.1 s behind the video). Each snapshot
outside the goal windows (HOLD_OUT s around them: those are the evaluation) is projected through that frame's camera (store/ar/<seg>/
camera.bin) and refined to the nearest puck box of the given detectors within REACH px, kept only if no other box is that close
(projected snapshots land 20 to 60 px from the puck; dark blobs there are mostly skates and gloves, so a box is required). The frame
around it is cut into a 640 px crop; detector boxes further than FAR px from the puck are cut out as hard negatives (they are what made
HockeyAI's tracks wrong). The last 20% of the match by time validates. Round 1 starts from HockeyAI's boxes, round 2 adds round 1's.
With TEMPORAL=1 (dataset, detect) the detector sees three frames at once: the grey frames n - 1, n and n + 1 as its three channels, so a
moving puck stands out as a coloured streak against the still ice (labels from frame n only: its neighbours lack theirs).

  labels  <labels.json>                       (local) usable snapshots: N, t, x, y, projected u, v, puck radius r (px)
  frames  <labels.json> <dir>                 (local, calib venv with av) frames N-1, N, N+1 of every label from the match video
  dataset <labels.json> <frames> <out> <weights>...   (YOLO venv) YOLO crops refined with these detectors
  train   <dataset> <init weights> <name> [arch]   fine-tune at 640 px (UP=2: crops enlarged to 1280 px, frames detected at 3840);
                                              with an architecture yaml (yolov8m-p2.yaml: a stride-4 head for the few-px puck) the init
                                              weights fill the layers it shares with them
  detect  <weights> <out name> <bundle>...    puck boxes per bundle frame, like puck.py detect -> bundle/<out name>
  check   <bundle> <dets name>                leave-one-out over the feed's snapshots in the window: each held-out snapshot
                                              against the straight-line feed path through the others, the video track alone, and the
                                              video track pulled toward that path by a pull of SIG0 + SIG_PER_S per s from the nearest
                                              snapshot (cap SIG_MAX) m, and whether that frame would be drawn (shown());
                                              prints one JSON line"""

import glob
import json
import os
import sys

import numpy as np
from paths import AR, GAME, SCENES, VIDEO

HOLD_OUT, SPLIT = 15, 0.8  # s around each goal window kept out of training; share of match time that trains
REACH, FAR, CROP, HARD, MIN_BOX, QUIET_PX = 70, 200, 640, 2, 8, 25
SIG0, SIG_PER_S, SIG_MAX = 1.0, 4.0, 8.0
TEMPORAL = os.environ.get("TEMPORAL", "0") == "1"
MIN_RUN, GAP = (
    int(os.environ.get("MIN_RUN", 5)),
    3,
)  # frames: shortest detector-backed run shown; longest gap bridged between runs
UP = int(os.environ.get("UP", 1))  # the detector sees crops and frames enlarged UP times (the puck is a few px)
SEED = int(os.environ.get("SEED", 0))  # training seed (repeats tell a real gain from the noise of 83 snapshots)


def labels(out):
    G = json.load(open(GAME))
    sn = np.array([f[:3] for f in G["frames"] if f[1] is not None])
    goals = [json.load(open(p))["source"]["start"] for p in glob.glob(f"{SCENES}/*/scene.json")]
    segs = [
        (json.load(open(d + "tracks.json")), np.fromfile(d + "camera.bin", np.float32).reshape(-1, 14))
        for d in sorted(glob.glob(f"{AR}/c*/"))
    ]
    res = []
    for t, x, y in sn:
        if any(-HOLD_OUT < t - g < 20 + HOLD_OUT for g in goals):
            continue
        N = int(round(t * 25))
        for T, cam in segs:
            if not T["n0"] <= N < T["n0"] + len(cam):
                continue
            c = cam[N - T["n0"]].astype(np.float64)
            if not c[0]:
                break
            R, C = c[2:11].reshape(3, 3), c[11:14]
            pc = (np.array([x, y, 0]) - C) @ R.T
            uv = c[0] * pc[:2] / pc[2]
            uv = uv * (1 + c[1] * (uv**2).sum() / T["fref"] ** 2) + [T["cx"], T["cy"]]
            if 40 < uv[0] < 1880 and 40 < uv[1] < 1040:
                res.append(
                    dict(
                        N=N,
                        t=round(float(t), 2),
                        x=float(x),
                        y=float(y),
                        u=round(float(uv[0]), 1),
                        v=round(float(uv[1]), 1),
                        r=round(float(c[0] * 0.038 / pc[2]), 2),
                    )
                )
            break
    json.dump(res, open(out, "w"))
    print("snapshots", len(sn), "labels", len(res))


def frames(labels_path, out):
    import av
    import cv2

    want = {n for l in json.load(open(labels_path)) for n in (l["N"] - 1, l["N"], l["N"] + 1)}
    want -= {int(f[:6]) for f in os.listdir(out)} if os.path.isdir(out) else set()
    os.makedirs(out, exist_ok=True)
    if not want:
        return
    c = av.open(VIDEO)
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    last = max(want)
    for fr in c.decode(s):
        n = int(round(fr.pts * s.time_base * 25))
        if n in want:
            cv2.imwrite(f"{out}/{n:06d}.jpg", fr.to_ndarray(format="bgr24"), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if n >= last:
            break


def stacked(paths):
    """Grey frames n - 1, n, n + 1 as the channels of one image (missing neighbours repeat frame n)."""
    import cv2

    grey = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None for p in paths]
    grey = [g if g is not None else grey[1] for g in grey]
    return np.dstack(grey)


def puck_class(m):
    return next(k for k, v in m.names.items() if v.lower() == "puck")


def dataset(labels_path, frames_dir, out, weights):
    import cv2
    from ultralytics import YOLO

    L = json.load(open(labels_path))
    models = [YOLO(w) for w in weights]
    cls = [puck_class(m) for m in models]
    split = np.quantile([l["t"] for l in L], SPLIT)
    rng = np.random.default_rng(0)
    stats = dict(pos=0, ambiguous=0, none=0, hard=0)
    for part in ("train", "val"):
        os.makedirs(f"{out}/{part}/images", exist_ok=True)
        os.makedirs(f"{out}/{part}/labels", exist_ok=True)
    by_n = {}
    for l in L:
        by_n.setdefault(l["N"], l)
    for N, l in sorted(by_n.items()):
        for dn in (0,) if TEMPORAL else (0, -1, 1):
            n = N + dn
            path = f"{frames_dir}/{n:06d}.jpg"
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
            src = stacked([f"{frames_dir}/{k:06d}.jpg" for k in (n - 1, n, n + 1)]) if TEMPORAL else img
            p0 = np.array([l["u"], l["v"]])
            xy = np.concatenate(
                [
                    m.predict(img, imgsz=1920, conf=0.05, classes=[c], verbose=False)[0].boxes.xywh[:, :2].cpu().numpy()
                    for m, c in zip(models, cls)
                ]
            )
            d = np.linalg.norm(xy - p0, axis=1)
            near = xy[d < REACH]
            if len(near) == 0:
                stats["none"] += 1
                continue
            c = near[np.argmin(np.linalg.norm(near - p0, axis=1))]
            if (np.linalg.norm(near - c, axis=1) > 6).any():
                stats["ambiguous"] += 1
                continue  # two different boxes near the puck
            if dn and np.linalg.norm(c - p0) > QUIET_PX:
                continue  # a neighbour frame only where the puck sits
            far = xy[d > FAR]
            far = far[rng.permutation(len(far))[:HARD]]
            stats["pos"] += 1
            stats["hard"] += len(far)
            size = max(MIN_BOX, 2.6 * l["r"])
            part = "train" if l["t"] < split else "val"
            H, W = img.shape[:2]
            for j in range(2 + len(far)):  # the puck, a random crop, hard negatives
                if j == 0:
                    ox, oy = (c - rng.uniform(60, CROP - 60, 2)).astype(int)
                elif j > 1:
                    ox, oy = (far[j - 2] - rng.uniform(100, CROP - 100, 2)).astype(int)
                else:
                    ox, oy = rng.integers(0, W - CROP), rng.integers(0, H - CROP)
                ox, oy = int(np.clip(ox, 0, W - CROP)), int(np.clip(oy, 0, H - CROP))
                if j > 0 and ox - 20 < c[0] < ox + CROP + 20 and oy - 20 < c[1] < oy + CROP + 20:
                    continue
                name = f"{n:06d}_{j}"
                cv2.imwrite(
                    f"{out}/{part}/images/{name}.jpg", src[oy : oy + CROP, ox : ox + CROP], [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                with open(f"{out}/{part}/labels/{name}.txt", "w") as f:
                    if j == 0:
                        f.write(f"0 {(c[0] - ox) / CROP:.5f} {(c[1] - oy) / CROP:.5f} {size / CROP:.5f} {size / CROP:.5f}\n")
    open(f"{out}/data.yaml", "w").write(f"path: {out}\ntrain: train/images\nval: val/images\nnames: {{0: puck}}\n")
    print(stats)


def train(data, init, name, arch=None):
    from ultralytics import YOLO

    (YOLO(arch).load(init) if arch else YOLO(init)).train(
        data=f"{data}/data.yaml",
        imgsz=CROP * UP,
        epochs=100,
        batch=32 // UP,
        scale=0.3,
        mosaic=1.0,
        project=os.path.join(os.path.dirname(data), "runs"),
        name=name,
        exist_ok=True,
        patience=30,
        workers=3,
        verbose=False,
        seed=SEED,  # 3 workers bound the RAM use
        **(dict(hsv_h=0.0, hsv_s=0.0) if TEMPORAL else {}),
    )  # hue and saturation would mix the frames up


def detect(weights, out, bundles):
    from ultralytics import YOLO

    m = YOLO(weights)
    pk = puck_class(m)
    for b in bundles:
        n = len(os.listdir(f"{b}/frames"))
        res = []
        for i in range(n):
            src = (
                stacked([f"{b}/frames/{min(max(k, 1), n):03d}.jpg" for k in (i, i + 1, i + 2)])
                if TEMPORAL
                else f"{b}/frames/{i + 1:03d}.jpg"
            )
            x = m.predict(src, imgsz=1920 * UP, conf=0.05, classes=[pk], verbose=False)[0].boxes
            k = x.conf.argsort(descending=True)[:5]
            res.append(
                [
                    [round(float(u), 1), round(float(v), 1), round(float(c), 3)]
                    for (u, v), c in zip(x.xywh[k, :2].tolist(), x.conf[k].tolist())
                ]
            )
        json.dump(res, open(f"{b}/{out}", "w"))
        print(b, "frames with a box", sum(bool(r) for r in res), "of", n, flush=True)


def shown(seen):
    """Frames where the tracked puck may be drawn: runs of at least MIN_RUN detector-backed frames, joined across gaps of up to GAP."""
    k = np.flatnonzero(seen)
    out = np.zeros(len(seen), bool)
    if not len(k):
        return out
    runs = np.split(k, np.flatnonzero(np.diff(k) > GAP + 1) + 1)
    for r in runs:
        if r[-1] - r[0] + 1 >= MIN_RUN and seen[r[0] : r[-1] + 1].sum() >= MIN_RUN:
            out[r[0] : r[-1] + 1] = True
    return out


def check(bundle, dets):
    os.environ["BUNDLE"] = bundle
    from goal import puck as P

    P.DETS = json.load(open(f"{bundle}/{dets}"))
    P.PRIOR_COST = lambda s, p: (
        0.0 if s is None or not np.isfinite(p[:2]).all() else (np.hypot(s[0] - p[0], s[1] - p[1]) / p[2]) ** 2
    )
    G = json.load(open(GAME))
    sn = np.array([f[:3] for f in G["frames"] if f[1] is not None])
    fr = (sn[:, 0] - P.S["start"]) * P.FPS
    k = (fr >= 0) & (fr <= P.N - 1)
    fr, xy = fr[k], sn[k, 1:]
    cands = [P.candidates(i) for i in range(P.N)]
    t = np.arange(P.N)

    def track(prior):
        c = [
            [
                q
                for q in cs
                if not np.isfinite(prior[i][:2]).all() or np.hypot(q[0] - prior[i][0], q[1] - prior[i][1]) < 3 * prior[i][2]
            ]
            for i, cs in enumerate(cands)
        ]
        p = P.viterbi(c, prior)
        seen = np.array([s is not None for s in p])
        x = np.array([s[:2] if s else (np.nan, np.nan) for s in p])
        kn = np.flatnonzero(seen)
        return (np.stack([np.interp(t, kn, x[kn, d]) for d in range(2)], 1) if len(kn) else x), seen

    video, vseen = track(np.c_[np.full((P.N, 2), np.nan), np.full(P.N, np.inf)])
    rows = []
    for j in range(len(fr)):
        o = np.arange(len(fr)) != j
        i = int(round(fr[j]))
        if not o.any():
            continue
        path = np.stack([np.interp(t, fr[o], xy[o, d]) for d in range(2)], 1)
        sig = np.minimum(SIG0 + SIG_PER_S * np.abs(t[:, None] - fr[o][None]).min(1) / P.FPS, SIG_MAX)
        fused, fseen = track(np.c_[path, sig])
        rows.append(
            dict(
                gap=round(float(np.abs(fr[o] - fr[j]).min() / P.FPS), 2),
                feed=float(np.hypot(*(path[i] - xy[j]))),
                fused=float(np.hypot(*(fused[i] - xy[j]))),
                fused_seen=bool(fseen[i]),
                fused_shown=bool(shown(fseen)[i]),
                video=float(np.hypot(*(video[i] - xy[j]))),
                video_seen=bool(vseen[i]),
            )
        )
    print(json.dumps(dict(goal=os.path.basename(os.path.dirname(os.path.abspath(bundle))), dets=dets, rows=rows)))


if __name__ == "__main__":
    a = sys.argv[2:]
    {
        "labels": lambda: labels(a[0]),
        "frames": lambda: frames(a[0], a[1]),
        "dataset": lambda: dataset(a[0], a[1], a[2], a[3:]),
        "train": lambda: train(*a),
        "detect": lambda: detect(a[0], a[1], a[2:]),
        "check": lambda: check(*a),
    }[sys.argv[1]]()
