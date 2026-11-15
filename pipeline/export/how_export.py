"""The "How it works" page's data for one goal (app/public/pages/How.tsx): the moment's broadcast frames undistorted into one
pinhole camera per frame, so that the rink's lines, the outlines, the puck and three.js renders of the 3D scene all line up with the
picture with no lens model on the page; the people, their outlines and how well their bodies line up; the puck; and how the arena's
atlas filled in.

  <out>/frames/<i>.webp    the moment's frames FROM..TO, undistorted, OUT_W × OUT_H
  <out>/how.json           cameras, rink, people, puck and atlas views (see the end of main)
  <out>/pose.json          each fitted body's 2D keypoints (OpenPose 18) per frame of the moment, and the limbs that join them
  <out>/sticks.json        the stick outlines found in each frame of the moment (sticks.py)
  <out>/atlas_ice.webp     the ice atlas, +y up;  atlas_out.webp: boards and stands, up the stands upward
  <out>/first_ice.png      per atlas texel (coarser), the index of the first atlas view that saw it, 255 never; first_out.png
  <out>/views/<n>.webp     each atlas view's frame, small

Undistorting: the source camera is a pinhole with one radial term (render.py's src_project). Each output pixel o is the pinhole
pixel u = (o - centre) / s, read from the source at u (1 + k |u|² / fref²) + (cx, cy). One scale s for the whole moment, the least
that keeps every output pixel of every frame inside the source frame, so nothing is padded and the zoom stays the camera's.

usage (LTX venv, CPU, in a goal dir):  CUDA_VISIBLE_DEVICES= BUNDLE=bundle how_export.py <out> [from to]"""

import json
import math
import os
import sys

import cv2
import numpy as np
import torch
from export.pipeline_figure import rink_lines
from goal import render as R

OUT_W, OUT_H = 960, 540  # page frames; the source is 1920 × 1080
SRC_W, SRC_H = 1920, 1080
FIRST_ICE, FIRST_OUT = 4, 4  # atlas texels per first-seen texel
QUALITY = 72


def border(n=48):
    """Output pixels round the frame's edge (full-size units)."""
    t = np.linspace(0, 1, n)
    return np.concatenate(
        [
            np.c_[t * SRC_W, np.zeros(n)],
            np.c_[t * SRC_W, np.full(n, SRC_H)],
            np.c_[np.zeros(n), t * SRC_H],
            np.c_[np.full(n, SRC_W), t * SRC_H],
        ]
    )


def distort(u, k):
    """Pinhole pixels (from the principal point) -> source pixels."""
    return u * (1 + k * (u**2).sum(-1, keepdims=True) / R.S["fref"] ** 2) + (R.S["cx"], R.S["cy"])


def undistort(d, k):
    """Source pixels -> pinhole pixels (from the principal point), by fixed-point steps."""
    d = d - (R.S["cx"], R.S["cy"])
    u = d.copy()
    for _ in range(12):
        u = d / (1 + k * (u**2).sum(-1, keepdims=True) / R.S["fref"] ** 2)
    return u


def least_scale(k):
    """The least s for which every output pixel reads from inside the source frame."""
    o = border() - (SRC_W / 2, SRC_H / 2)
    lo, hi = 0.3, 3.0
    for _ in range(40):
        s = (lo + hi) / 2
        d = distort(o / s, k)
        inside = (d[:, 0] >= 0) & (d[:, 0] <= SRC_W - 1) & (d[:, 1] >= 0) & (d[:, 1] <= SRC_H - 1)
        lo, hi = (lo, s) if inside.all() else (s, hi)
    return hi


def main():
    out = sys.argv[1]
    S, Z = R.S, R.Z
    shot = R.SHOT
    release = shot["release"] if shot else S["n"] // 2
    a, b = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (max(release - 22, 0), min(release + 37, S["n"] - 1))
    window = range(a, b + 1)
    for d in ("frames", "views"):
        os.makedirs(f"{out}/{d}", exist_ok=True)

    # one scale for the moment, then each frame resampled
    s = max(least_scale(float(Z["cam"][i][1])) for i in window)
    gx, gy = np.meshgrid(np.arange(SRC_W) + 0.5, np.arange(SRC_H) + 0.5)
    o = np.stack([gx, gy], -1) - (SRC_W / 2, SRC_H / 2)
    cams = []
    for i in window:
        cam = Z["cam"][i]
        k = float(cam[1])
        d = distort(o / s, k).astype(np.float32)
        src = cv2.imread(f"{R.B}/frames/{i + 1:03d}.jpg")
        # sampled at full size, then down with area averaging
        big = cv2.remap(src, d[..., 0], d[..., 1], cv2.INTER_LINEAR)
        cv2.imwrite(
            f"{out}/frames/{i}.webp",
            cv2.resize(big, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA),
            [cv2.IMWRITE_WEBP_QUALITY, QUALITY],
        )
        # the pinhole camera of the output frame: focal in page pixels, rows of R (rink to OpenCV camera axes), centre
        cams.append(
            dict(
                f=round(float(cam[0]) * s * OUT_W / SRC_W, 3),
                R=np.round(cam[2:11].reshape(3, 3), 6).tolist(),
                C=np.round(cam[11:14], 4).tolist(),
            )
        )

    page = lambda d, k: (undistort(np.asarray(d, np.float64), k) * s + (SRC_W / 2, SRC_H / 2)) * OUT_W / SRC_W
    point = lambda q: (
        [round(float(q[0]), 3), round(float(q[1]), 3)] if q is not None and np.isfinite(np.asarray(q, float)).all() else None
    )

    # people: rink paths over the whole scene, outlines and alignment over the moment
    C = json.load(open(f"{R.B}/cast.json"))
    A = json.load(open(f"{R.B}/align.json")) if os.path.exists(f"{R.B}/align.json") else {}
    people = []
    for t in C["tracks"]:
        xy = [point(q) for q in t["xy"]]
        outline, iou = [], []
        for i in window:
            m = t["mask"][i]
            if not t["seen"][i] or m < 0:
                outline.append(None)
            else:
                p = page(S["outlines"][i][m], float(Z["cam"][i][1])).astype(np.float32)
                p = cv2.approxPolyDP(p.reshape(-1, 1, 2), 0.8, True).reshape(-1, 2)
                outline.append(np.round(p, 1).flatten().tolist())
            v = (A.get(str(t["key"])) or [None] * (i + 1))[i] if A else None
            iou.append(round(float(v[0]), 3) if isinstance(v, list) and v else None)
        people.append(
            dict(
                key=t["key"],
                team=t["team"],
                jersey=t.get("jersey"),
                name=t.get("name"),
                goalie=bool(t.get("goalie")),
                xy=xy,
                seen=[bool(t["seen"][i]) for i in window],
                outline=outline,
                iou=iou,
            )
        )

    # the fitted bodies' 2D keypoints, as pipeline_figure.py draws them, and the stick outlines
    bodies = np.load(f"{R.B}/{R.BODIES}.npz")
    bf, bk = bodies["frame"], bodies["key"]
    kp2d = bodies["kp2d"]
    pose = {}
    for t in C["tracks"]:
        rows = []
        for i in window:
            m = np.flatnonzero((bf == i) & (bk == t["key"]))
            rows.append(np.round(page(kp2d[m[0]][R.OPENPOSE], float(Z["cam"][i][1])), 1).flatten().tolist() if len(m) else None)
        if any(r is not None for r in rows):
            pose[str(t["key"])] = rows
    json.dump(dict(limbs=R.LIMBS, colours=R.POSE_COLORS, people=pose), open(f"{out}/pose.json", "w"), separators=(",", ":"))
    sticks = json.load(open(f"{R.B}/sticks.json")) if os.path.exists(f"{R.B}/sticks.json") else [[] for _ in range(S["n"])]
    json.dump(
        [[np.round(page(st["poly"], float(Z["cam"][i][1])), 1).flatten().tolist() for st in sticks[i]] for i in window],
        open(f"{out}/sticks.json", "w"),
        separators=(",", ":"),
    )

    # the atlas, and per texel the first view that saw it (render.py's build_atlas test, without its occlusion march)
    at = np.load(f"{R.B}/atlas.npz")
    cv2.imwrite(
        f"{out}/atlas_ice.webp",
        cv2.cvtColor(cv2.resize(at["ice"][::-1], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_WEBP_QUALITY, 78],
    )
    cv2.imwrite(
        f"{out}/atlas_out.webp",
        cv2.cvtColor(cv2.resize(at["out"][::-1], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_WEBP_QUALITY, 78],
    )
    dev = R.DEV
    ix = torch.arange(-R.HX, R.HX, R.ICE_RES * FIRST_ICE, device=dev)
    iy = torch.arange(-R.HY, R.HY, R.ICE_RES * FIRST_ICE, device=dev)
    IY, IX = torch.meshgrid(iy, ix, indexing="ij")
    ice = torch.stack([IX, IY, torch.zeros_like(IX)], -1)
    inside = R.sdf(IX, IY) < -0.05
    gs = torch.arange(0, R.PERIM, R.OUT_RES * FIRST_OUT, device=dev)
    gh = torch.arange(0, 34, R.OUT_RES * FIRST_OUT, device=dev)
    OH, OS = torch.meshgrid(gh, gs, indexing="ij")
    ox, oy, oz = R.out_world(OS, OH)
    outside = torch.stack([ox, oy, oz], -1)
    _, _, nx, ny = R.outline_point(OS)
    k_ = math.sqrt(1 + R.SLOPE**2)
    on_wall = OH < R.WALL
    normal = torch.stack(
        [
            torch.where(on_wall, -nx, -R.SLOPE * nx / k_),
            torch.where(on_wall, -ny, -R.SLOPE * ny / k_),
            torch.where(on_wall, torch.zeros_like(nx), torch.full_like(nx, 1 / k_)),
        ],
        -1,
    )
    normal = torch.where(
        ((OH >= R.WALL) & (OH < R.WALL + R.LEDGE))[..., None], torch.tensor([0, 0, 1.0], device=dev).expand_as(normal), normal
    )
    views = [("atlas_frames", j, Z["atlas_cam"][j], S["atlas_outlines"][j]) for j in range(len(Z["atlas_cam"]))]
    views += [("frames", i, Z["cam"][i], S["outlines"][i]) for i in range(0, S["n"], 10)]
    first = {
        "ice": torch.full(ice.shape[:2], 255, dtype=torch.uint8),
        "out": torch.full(outside.shape[:2], 255, dtype=torch.uint8),
    }
    for n, (folder, j, cam, outl) in enumerate(views):
        m = torch.from_numpy(R.person_mask(outl)).to(dev).float()[None]
        Cc = torch.tensor(cam[11:14], device=dev, dtype=torch.float32)
        for key, P in (("ice", ice), ("out", outside)):
            px, depth = R.src_project(cam, P)
            ok = (depth > 1) & (px[..., 0] > 2) & (px[..., 0] < SRC_W - 2) & (px[..., 1] > 2) & (px[..., 1] < SRC_H - 2)
            ok &= R.sample(m, px)[..., 0] < 0.5
            if key == "ice":
                ok &= inside
            else:
                ok &= ((Cc - P) * normal).sum(-1) / (Cc - P).norm(dim=-1) > 0.3
            f = first[key]
            f[ok.cpu() & (f == 255)] = n
        thumb = cv2.resize(
            cv2.cvtColor((R.load_frame(j, folder).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            (320, 180),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(f"{out}/views/{n}.webp", thumb, [cv2.IMWRITE_WEBP_QUALITY, 70])
    cv2.imwrite(f"{out}/first_ice.png", first["ice"].numpy()[::-1])
    cv2.imwrite(f"{out}/first_out.png", first["out"].numpy()[::-1])

    P = json.load(open(f"{R.B}/puck.json"))
    how = dict(
        fps=S["fps"],
        frames=[a, b],
        size=[OUT_W, OUT_H],
        cams=cams,
        rink=dict(hx=R.HX, hy=R.HY, corner=R.CR, goal_x=R.GOAL_X, lines=[np.round(L, 3).tolist() for L in rink_lines()]),
        people=people,
        puck=dict(xy=[point(q) for q in P["xy"]], show=[round(float(v), 3) for v in P["show"]], shot=P["shot"]),
        atlas=dict(
            views=len(views),
            moment_from=len(Z["atlas_cam"]),
            ice_res=R.ICE_RES * FIRST_ICE,
            out_res=R.OUT_RES * FIRST_OUT,
            perim=R.PERIM,
        ),
    )
    json.dump(how, open(f"{out}/how.json", "w"), separators=(",", ":"))
    print(out, f"frames {a}..{b}, scale {s:.4f}, {len(people)} people, {len(views)} atlas views")


if __name__ == "__main__":
    main()
