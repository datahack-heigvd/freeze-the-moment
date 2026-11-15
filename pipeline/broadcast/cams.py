"""Per-frame cameras for two-second windows of a match's wide game-camera shots, through the 2025 Kam1 calibration.

The 2024 broadcast's wide camera stands where the 2025 Kam1 stands (mapmatch.py), so every frame is solved against the 2025
camera map: the window's frames vote for the map nodes they look at, each frame is matched to the best of those nodes and its
(f, rotation) solved with calib's localize.py; calib's self-calibration then gives the full camera row (f, k, R, C) the renderers
use, and camsmooth.py smooths the window. Frames are taken at 25 fps from the 30 fps stream so every later step sees the 2025 rate.
A goal is one longer window over the play (`goal`), set by hand from the broadcast.

  <out>/win/<id>/frames/NNN.jpg   n source frames (49 per window, any for a goal), 1920x1080, 25 fps
  <out>/win/<id>/cams.npz         cam [n, 14] smoothed, cam_raw, inl, med (px)
  <out>/windows.json              [{id, t0, ok, inl, med}] for every window tried

usage (from pipeline/broadcast/):  cams.py <video> <out_dir> [max_windows] [workers]
                                  cams.py goal <video> <out_dir> <t0> <seconds>"""

import json
import os
import subprocess
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from paths import PIPELINE

HERE = f"{PIPELINE}/broadcast"
os.chdir(f"{PIPELINE}/calib")
sys.path.insert(0, "scripts")
import camsmooth as CS
import export_ar as EA
import localize as Lc

FRAMES, FPS, STEP, MARGIN = 49, 25, 6.0, 0.5  # window length; s between window starts inside a shot; s kept clear of cuts
TOP, MIN_INL, MAX_MED = 6, 40, 3.0  # nodes a window matches against; per-frame acceptance
_z = np.load("cache/cam_pano300.npz")
Z = {k: _z[k] for k in _z.files}
LENS = Z["lens"]
G3 = np.load("cache/selfcal_v1.npz")["V3"]
PTS = np.load("cache/mapfeat_pano300_pts.npy")
DES = np.load("cache/mapfeat_pano300_des.npy")
OFF = np.load("cache/mapfeat_pano300_off.npy")


def windows(shots):
    out = []
    for s in shots:
        if not s["wide"]:
            continue
        t = s["t0"] + MARGIN
        while t + FRAMES / FPS < s["t1"] - MARGIN:
            out.append(round(t, 2))
            t += STEP
    return out


def extract(video, t0, d, n=FRAMES):
    os.makedirs(f"{d}/frames", exist_ok=True)
    if not os.path.exists(f"{d}/frames/{n:03d}.jpg"):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                f"{t0:.3f}",
                "-i",
                video,
                "-vf",
                f"fps={FPS}",
                "-frames:v",
                str(n),
                "-start_number",
                "1",
                "-q:v",
                "2",
                f"{d}/frames/%03d.jpg",
            ],
            check=True,
        )
    return [cv2.resize(cv2.imread(f"{d}/frames/{i + 1:03d}.jpg"), (960, 540)) for i in range(n)]


def node(n):
    return PTS[OFF[n] : OFF[n + 1]], DES[OFF[n] : OFF[n + 1]].astype(np.float32)


def votes(fl, node_of, imgs):
    """How many of the map's features in each node the frames match."""
    v = np.zeros(len(OFF) - 1)
    for im in imgs:
        im = im.copy()
        im[:70, 280:680] = 0
        im[440:, 880:] = 0
        m = [
            x
            for x in fl.knnMatch(Lc.feats(im)[1].astype(np.float32), k=2)
            if len(x) == 2 and x[0].distance < 0.85 * x[1].distance
        ]
        v += np.bincount(node_of[[x[0].trainIdx for x in m]], minlength=len(v))
    return v


def matcher():
    fl = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
    fl.add([DES.astype(np.float32)])
    fl.train()
    return fl, np.searchsorted(OFF, np.arange(len(PTS)), side="right") - 1


def solve_window(args):
    video, t0, d, nodes, n = args
    Lc._init()
    imgs = extract(video, t0, d, n)
    for im in imgs:
        im[:70, 280:680] = 0
        im[440:, 880:] = 0  # score bug and channel logo
    F = [Lc.feats(im) for im in imgs]
    N = [node(n) for n in nodes]
    x_prev, rows, inl, med = None, np.zeros((n, 14), np.float32), np.zeros(n, int), np.full(n, 99.0)
    for i, (pb, db) in enumerate(F):
        cons = []
        for nd, (pa, da) in zip(nodes, N):
            mt = Lc.match(da, db, pa, pb)
            if mt is not None:
                cons.append((len(mt[0]), (mt[0], Z["f"][nd], cv2.Rodrigues(Z["rv"][nd])[0], mt[1]), nd))
        if not cons:
            continue
        cons.sort(key=lambda c: -c[0])
        n0 = cons[0][2]
        x0 = x_prev if x_prev is not None else np.r_[Z["f"][n0], Z["rv"][n0]]
        x, ninl, m = Lc.solve(x0, [c[1] for c in cons[:3]], LENS)
        inl[i], med[i] = ninl, m
        if ninl >= MIN_INL and m <= MAX_MED and np.isfinite(x).all():
            rows[i] = EA.camera_row(G3, x[0], x[1:4])[0]
            x_prev = x
    ok = bool((np.abs(rows).sum(1) > 0).all())
    cam = CS.smooth_cameras(rows.astype(np.float64), CS.LAM)[0].astype(np.float32) if ok else rows
    np.savez(f"{d}/cams.npz", cam=cam, cam_raw=rows, inl=inl, med=med)
    return dict(t0=t0, ok=ok, inl=int(np.median(inl)), med=round(float(np.median(med)), 2))


def main(video, out, max_windows="60", workers="6"):
    out = f"{HERE}/{out}"
    ts = windows(json.load(open(f"{out}/shots.json")))
    ts = [ts[int(k)] for k in np.linspace(0, len(ts) - 1, min(int(max_windows), len(ts)))]
    print(f"{len(ts)} windows", flush=True)
    fl = None
    Lc._init()
    jobs = []
    for t0 in ts:
        d = f"{out}/win/{int(round(t0 * 100)):07d}"
        if os.path.exists(f"{d}/nodes.json"):
            jobs.append((video, t0, d, json.load(open(f"{d}/nodes.json")), FRAMES))
            continue
        if fl is None:
            fl, node_of = matcher()
        imgs = extract(video, t0, d)
        v = votes(fl, node_of, [imgs[i] for i in (0, FRAMES // 2, FRAMES - 1)])
        jobs.append((video, t0, d, [int(n) for n in np.argsort(-v)[:TOP]], FRAMES))
        json.dump(jobs[-1][3], open(f"{d}/nodes.json", "w"))
    del fl
    res = []
    with Pool(int(workers)) as pool:
        for r, j in zip(pool.imap(solve_window, jobs), jobs):
            res.append(dict(id=os.path.basename(j[2]), **r))
            print(res[-1], flush=True)
    json.dump(res, open(f"{out}/windows.json", "w"), indent=0)
    print(f"{sum(r['ok'] for r in res)} of {len(res)} windows calibrated on every frame")


def goal(video, out, t0, seconds):
    """One window over a goal's play: the camera pans and zooms more over a long play, so the nodes are voted for every second and
    each frame is solved against the most voted ones."""
    out, t0 = f"{HERE}/{out}", float(t0)
    n = round(float(seconds) * FPS)
    d = f"{out}/win/{int(round(t0 * 100)):07d}"
    Lc._init()
    fl, node_of = matcher()
    imgs = extract(video, t0, d, n)
    v = votes(fl, node_of, imgs[::FPS])
    nodes = [int(k) for k in np.argsort(-v)[: 2 * TOP]]
    del fl
    json.dump(nodes, open(f"{d}/nodes.json", "w"))
    res = [dict(id=os.path.basename(d), **solve_window((video, t0, d, nodes, n)))]
    print(res[0])
    json.dump(res, open(f"{out}/windows.json", "w"), indent=0)


if __name__ == "__main__":
    goal(*sys.argv[2:]) if sys.argv[1] == "goal" else main(*sys.argv[1:])
