"""Bundle one source window for the novel-view renderer: frames, per-frame camera, people and puck on the rink.

Coordinates are the calibration's rink frame (metres, origin at centre ice, z up), which is the CSV metadata frame negated.
  bundle/frames/NNN.jpg  source frames (1920x1080, 25 fps)
  bundle/atlas_frames    link to atlas/<segment>/, frames every 2 s of the calibrated segment for the background atlas
  bundle/scene.npz       cam [n,14] (f, k, R row-major, C; up to 1 s without a solve is interpolated, those frames get no outlines), puck [n,2]
  bundle/scene.json      rink, intrinsics, roster, per-frame people [{id, team, xy, mask}] and mask outlines

usage (calib venv, in a goal dir; just bundle):  [BUNDLE=dir] prep.py [start_s] [dur_s]"""

import json
import os
import subprocess
import sys

import numpy as np
from paths import AR, GAME, PIPELINE, VIDEO
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation, Slerp

FPS = 25
OUT = os.environ.get("BUNDLE", "bundle")
ATLAS_STEP = 50  # every 2 s of the segment


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


def project(cam, fref, cx, cy, P):
    f, k, R, C = cam[0], cam[1], cam[2:11].reshape(3, 3), cam[11:14]
    pc = (np.atleast_2d(P) - C) @ R.T
    uv = f * pc[:, :2] / pc[:, 2:3]
    return uv * (1 + k * (uv**2).sum(1, keepdims=True) / fref**2) + [cx, cy]


def ray_to_ice(cam, fref, cx, cy, px):
    """Source pixel -> rink point on z=0."""
    f, k, R, C = cam[0], cam[1], cam[2:11].reshape(3, 3), cam[11:14]
    d = np.atleast_2d(px) - [cx, cy]
    u = d.copy()
    for _ in range(8):
        u = d / (1 + k * (u**2).sum(1, keepdims=True) / fref**2)
    ray = np.c_[u / f, np.ones(len(u))] @ R
    t = -C[2] / ray[:, 2]
    return (C + t[:, None] * ray)[:, :2]


def segment_of(n):
    """The calibrated segment containing source frame n."""
    for d in sorted(os.listdir(AR)):
        T = json.load(open(f"{AR}/{d}/tracks.json"))
        if T["n0"] <= n < T["n0"] + len(T["ctl"]):
            return f"{AR}/{d}"
    raise SystemExit(f"no calibrated segment holds frame {n}")


def puck_path(game, t):
    """Puck snapshots from the game feed (rink frame, about every 0.6 s), interpolated to source times t; held across gaps."""
    snaps = np.array([f[:3] for f in game["frames"] if f[1] is not None], float)
    return np.c_[np.interp(t, snaps[:, 0], snaps[:, 1]), np.interp(t, snaps[:, 0], snaps[:, 2])]


def fill_cams(cams, a, n, longest=25):
    """The cameras of frames a..a+n with short gaps (a fast pan blurs the frame past the solver) filled between their neighbours:
    rotation by slerp, focal length, distortion and position linearly; and which frames were filled. Longer gaps stop the bundle."""
    cams = cams.copy()
    ok = np.abs(cams).sum(1) > 0
    filled = np.zeros(len(cams), bool)
    i = a
    while i < a + n:
        if ok[i]:
            i += 1
            continue
        j = i
        while j < len(cams) and not ok[j]:
            j += 1
        if i == 0 or j == len(cams) or j - i > longest:
            raise SystemExit(f"frames {i}..{j - 1} of the segment have no camera")
        w = (np.arange(i, j) - (i - 1)) / (j - (i - 1))
        rot = Slerp([0, 1], Rotation.from_matrix([cams[i - 1, 2:11].reshape(3, 3), cams[j, 2:11].reshape(3, 3)]))(w).as_matrix()
        lin = (1 - w)[:, None] * cams[i - 1] + w[:, None] * cams[j]
        cams[i:j] = np.c_[lin[:, :2], rot.reshape(-1, 9), lin[:, 11:]]
        filled[i:j] = True
        i = j
    return cams, filled


def main(start=5310.2, dur=10.0):
    N0 = int(round(start * FPS))
    n = int(round(dur * FPS))
    segment = segment_of(N0)
    T = json.load(open(f"{segment}/tracks.json"))
    game = json.load(open(GAME))
    cams = np.fromfile(f"{segment}/camera.bin", np.float32).reshape(-1, 14).astype(np.float64)
    masks = read_masks(f"{segment}/masks.bin")
    a = N0 - T["n0"]
    cams, filled = fill_cams(cams, a, n)
    cx, cy, fref = T["cx"], T["cy"], T["fref"]
    roster = [dict(jersey=r["jersey"], name=r["name"], team=int(r["team"] == "Away"), goalie=r["goalie"]) for r in game["roster"]]

    people, outlines = [], []
    for i in range(n):
        cam, polys = cams[a + i], [] if filled[a + i] else masks[a + i]  # outlines need the frame's own camera
        named = {p: (x / 100, y / 100, team) for p, x, y, team in np.array(T["ctl"][a + i]).reshape(-1, 4).tolist() if p >= 0}
        for p, x, y, fl in np.array(T["rings"][a + i]).reshape(-1, 4).tolist():
            if p >= 0 and p not in named:
                named[p] = (x / 100, y / 100, fl >> 1 & 1)
        anon = [(x / 100, y / 100, fl >> 1 & 1) for p, x, y, fl in np.array(T["rings"][a + i]).reshape(-1, 4).tolist() if p < 0]
        ents = [dict(id=p, team=int(t), xy=[x, y]) for p, (x, y, t) in named.items()] + [
            dict(id=-1, team=int(t), xy=[x, y]) for x, y, t in anon
        ]
        # a person's outline: the mask whose bottom-centre is nearest to the projected foot point
        feet = project(cam, fref, cx, cy, np.array([[*e["xy"], 0.0] for e in ents])) if ents else np.zeros((0, 2))
        bottoms = np.array([[p[:, 0].mean(), p[:, 1].max()] for p in polys]) if polys else np.zeros((0, 2))
        heights = np.array([np.ptp(p[:, 1]) for p in polys]) if polys else np.zeros(0)
        for e in ents:
            e["mask"] = -1
        if len(ents) and len(polys):
            D = np.linalg.norm(feet[:, None] - bottoms[None], axis=2) / np.maximum(heights[None], 20)
            r, c = linear_sum_assignment(D)
            for j, m in zip(r, c):
                if D[j, m] < 0.6:
                    ents[j]["mask"] = int(m)
        used = {e["mask"] for e in ents}
        tracked = len(ents)
        for m in range(len(polys)):
            if m in used:
                continue
            xy = ray_to_ice(cam, fref, cx, cy, bottoms[m])[0]
            dist = [np.hypot(*(np.array(e["xy"]) - xy)) for e in ents[:tracked]]
            free = [j for j in range(tracked) if ents[j]["mask"] < 0 and dist[j] < 1.5]
            if free:
                ents[min(free, key=lambda j: dist[j])]["mask"] = m  # a tracked player the matching missed
            elif not dist or min(dist) > 1.0:  # referees and untracked skaters
                ents.append(dict(id=-2, team=-1, xy=xy.tolist(), mask=m))
        people.append(ents)
        outlines.append([np.round(p, 1).tolist() for p in polys])

    os.makedirs(f"{OUT}/frames", exist_ok=True)
    if not os.path.exists(f"{OUT}/frames/{n:03d}.jpg"):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                f"{(N0 - 0.5) / FPS:.4f}",
                "-i",
                VIDEO,
                "-frames:v",
                str(n),
                "-fps_mode",
                "passthrough",
                "-start_number",
                "1",
                "-q:v",
                "2",
                f"{OUT}/frames/%03d.jpg",
            ],
            check=True,
        )
    # the static background atlas also uses frames from the whole segment, where the camera pans over the entire rink; they are
    # shared by every bundle of the segment
    shared = f"{PIPELINE}/atlas/{os.path.basename(segment)}"
    os.makedirs(shared, exist_ok=True)
    picks = [k for k in range(0, len(cams), ATLAS_STEP) if np.abs(cams[k]).sum() > 0]
    for j, k in enumerate(picks):
        if not os.path.exists(f"{shared}/{j:03d}.jpg"):
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    f"{(T['n0'] + k - 0.5) / FPS:.4f}",
                    "-i",
                    VIDEO,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    f"{shared}/{j:03d}.jpg",
                ],
                check=True,
            )
    if not os.path.lexists(f"{OUT}/atlas_frames"):
        os.symlink(os.path.relpath(shared, OUT), f"{OUT}/atlas_frames")
    np.savez(f"{OUT}/scene.npz", cam=cams[a : a + n], puck=puck_path(game, start + np.arange(n) / FPS), atlas_cam=cams[picks])
    json.dump(
        dict(
            start=start,
            fps=FPS,
            n=n,
            cx=cx,
            cy=cy,
            fref=fref,
            hx=T["hx"],
            hy=T["hy"],
            corner=T["corner"],
            segment=os.path.basename(segment),
            roster=roster,
            people=people,
            outlines=outlines,
            atlas_outlines=[[np.round(p, 1).tolist() for p in masks[k]] for k in picks],
        ),
        open(f"{OUT}/scene.json", "w"),
    )
    print(
        f"{n} frames, people/frame {np.mean([len(p) for p in people]):.1f}, with outline {np.mean([sum(e['mask'] >= 0 for e in p) for p in people]):.1f}, "
        f"untracked outlines {np.mean([sum(e['id'] == -2 for e in p) for p in people]):.1f}"
    )


if __name__ == "__main__":
    main(*map(float, sys.argv[1:]))
