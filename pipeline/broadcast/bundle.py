"""Turn a match's calibrated windows into novel-view bundles (the format of prep.py), without any tracking feed.

People come from the video only: person outlines whose feet stand on the ice are
the cast. SAM 3.1 (sam_seg.py) is the primary outline: far steadier over time than YOLO11x-seg (0.15 vs 1.97 on/off flips per second
over 980 frames, seg_eval.py); YOLO finds ~4% more people, so a YOLO outline that overlaps no SAM outline is added. An outline is a
referee when the HockeyAI box (ref_boxes.py) that overlaps it most is a referee box: colour cannot tell stripes from black jerseys
with white trim or from motion blur, the detector can. cast.py links the outlines, makes a track a referee when most of its outlines
are, and assigns the others a team by jersey colour against this match's colour groups, found here: every on-ice player outline's
upper body becomes a colour histogram and the histograms fall into groups labelled home, away or junk from their own colours and
sizes (kit_groups.py; on the 2025 goals 99.8% of people land in their own team). out/<match>/teams.jpg shows the groups for
inspection, out/<match>/group_labels.json their labels. The training caption's jerseys are named from the teams' histograms
(jerseys). The broadcast's graphics (score bug, channel logo) are score_bug.py's steady overlays (out/<match>/score_bug.json), masked
out of the atlas, and each window's crop keeps below and beside them. The background atlas is shared by the match: frames about 2 s apart over every
window, with people, the score bug and the channel logo masked out. No puck is drawn (its position is unknown). Each window is
listed with the crop that keeps the broadcast graphics out of the training pair.

  pipeline/<SET>/<match>_<win>/     frames (link), scene.json, scene.npz        SET defaults to train24
  pipeline/atlas/<match>/           atlas frames (links)
  pipeline/<SET>/windows.json       [{id, segment, start, crop}]

usage (from pipeline/broadcast/):  bundle.py seg <match>      SAM 3.1 outlines and HockeyAI referee boxes for every window (GPU_HOST)
                                  bundle.py build <match>    bundles and windows.json"""

import fcntl
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from paths import PIPELINE

HERE = f"{PIPELINE}/broadcast"
sys.path.insert(0, f"{PIPELINE}/calib/scripts")
os.chdir(f"{PIPELINE}/calib")
import selfcal as SC
from broadcast.kit_groups import labels, team_groups, upper_px

GPU, SAM_PY = (
    os.environ.get("GPU_HOST", "gpu"),
    os.environ.get("SAM_PY", "python"),
)  # the GPU machine (ssh) and its SAM 3.1 python
REMOTE = "broadcast"
FPS, MARGIN, ASPECT = 25, 4, 960 / 544  # px round each overlay; the crop's width over height
SET = os.environ.get("SET", "train24")


def ok_windows(match):
    """The match's calibrated windows, each with its frame count n (49, or a goal's length)."""
    ws = [w for w in json.load(open(f"{HERE}/out/{match}/windows.json")) if w["ok"]]
    return [dict(w, n=len(os.listdir(f"{HERE}/out/{match}/win/{w['id']}/frames"))) for w in ws]


def seg(match):
    """SAM 3.1 per window, YOLO11x-seg and HockeyAI boxes over all window frames, on the GPU machine; results copied back next to the frames."""
    ws = ok_windows(match)
    src = f"{HERE}/out/{match}/win"
    dst = f"{REMOTE}/{match}/win"
    flat = f"{REMOTE}/{match}/flat"
    subprocess.run(["ssh", GPU, f"mkdir -p {dst} {flat}"], check=True)
    subprocess.run(
        ["rsync", "-a", "--relative"] + [f"{src}/./{w['id']}/frames" for w in ws] + [f"{HERE}/./sam_seg.py", f"{GPU}:{dst}/"],
        check=True,
    )
    ids = " ".join(w["id"] for w in ws)
    links = f"for w in {ids}; do for f in ../win/$w/frames/*.jpg; do ln -sf $f ${{w}}_$(basename $f); done; done"
    subprocess.run(["rsync", "-a", f"{HERE}/ref_boxes.py", f"{PIPELINE}/calib/gpu/seg_poly_gpu.py", f"{GPU}:hockey/"], check=True)
    subprocess.run(
        [
            "ssh",
            GPU,
            f"cd {flat} && find . -maxdepth 1 -name '*.jpg' -delete && {links} && cd ~/hockey && .venv/bin/python -u seg_poly_gpu.py ~/{flat} ~/{REMOTE}/{match}/yolo.npz"
            f" && .venv/bin/python -u ref_boxes.py ~/{flat} ~/{REMOTE}/{match}/refs.npz",
        ],
        check=True,
    )
    subprocess.run(["ssh", GPU, f"cd {dst} && {SAM_PY} -u sam_seg.py " + " ".join(w["id"] for w in ws)], check=True)
    subprocess.run(
        ["rsync", "-a", f"{GPU}:{REMOTE}/{match}/yolo.npz", f"{GPU}:{REMOTE}/{match}/refs.npz", f"{HERE}/out/{match}/"],
        check=True,
    )
    subprocess.run(["rsync", "-a", "--include=*/", "--include=sam3.json", "--exclude=*", f"{GPU}:{dst}/", f"{src}/"], check=True)


def raster(poly):
    m = np.zeros((1080, 1920), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m


def outlines(match, ws):
    """{(win, frame): [polygon px]}: SAM 3.1 outlines, plus YOLO outlines that overlap none of them (IoU < 0.2)."""
    z = np.load(f"{HERE}/out/{match}/yolo.npz")
    names = sorted(f"{w['id']}_{i:03d}" for w in ws for i in range(1, w["n"] + 1))
    yolo, k = {}, 0
    for d in z["dets"]:
        n = int(d[6])
        poly = z["pts"][k : k + n].astype(np.float32) / 8
        k += n
        win, fr = names[int(d[0])].rsplit("_", 1)
        yolo.setdefault((win, int(fr) - 1), []).append(poly)
    out = {}
    for w in ws:
        for i, fr in enumerate(json.load(open(f"{HERE}/out/{match}/win/{w['id']}/sam3.json"))):
            sam = [np.array(o["poly"], np.float32) for o in fr]
            ms = [raster(p) for p in sam]
            extra = [
                p
                for p, r in ((p, raster(p)) for p in yolo.get((w["id"], i), []))
                if all((r & m).sum() / max((r | m).sum(), 1) < 0.2 for m in ms)
            ]
            out[w["id"], i] = sam + extra
    return out


def referees(match, ws, polys):
    """{(win, frame): [bool per outline]}: whether the HockeyAI box overlapping the outline's box most (IoU > 0.3) is a referee."""
    z = np.load(f"{HERE}/out/{match}/refs.npz")["dets"]
    names = sorted(f"{w['id']}_{i:03d}" for w in ws for i in range(1, w["n"] + 1))
    boxes = {}
    for d in z:
        win, fr = names[int(d[0])].rsplit("_", 1)
        boxes.setdefault((win, int(fr) - 1), []).append(d[1:])
    out = {}
    for key, ps in polys.items():
        b = np.array(boxes.get(key, np.zeros((0, 6))), np.float32).reshape(-1, 6)
        flags = []
        for p in ps:
            x0, y0 = p.min(0)
            x1, y1 = p.max(0)
            iw = np.clip(np.minimum(x1, b[:, 2]) - np.maximum(x0, b[:, 0]), 0, None)
            ih = np.clip(np.minimum(y1, b[:, 3]) - np.maximum(y0, b[:, 1]), 0, None)
            iou = iw * ih / ((x1 - x0) * (y1 - y0) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - iw * ih)
            flags.append(bool(len(b) and iou.max() > 0.3 and b[iou.argmax(), 5] > 0))
        out[key] = flags
    return out


def rink(d):
    return dict(
        hx=float(d["hx"]),
        hy=float(d["hy"]),
        corner=float(d["corner"]),
        cx=960 + float(d["cxo"]),
        cy=540 + float(d["cyo"]),
        fref=SC.F_REF,
    )


def to_ice(cam, R_, px):
    f, k, R, C = cam[0], cam[1], cam[2:11].reshape(3, 3), cam[11:14]
    d = np.atleast_2d(px) - [R_["cx"], R_["cy"]]
    u = d.copy()
    for _ in range(8):
        u = d / (1 + k * (u**2).sum(1, keepdims=True) / R_["fref"] ** 2)
    ray = np.c_[u / f, np.ones(len(u))] @ R
    return (C + (-C[2] / ray[:, 2])[:, None] * ray)[:, :2]


def sdf(xy, R_):
    q = np.abs(xy) - [R_["hx"] - R_["corner"], R_["hy"] - R_["corner"]]
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0) - R_["corner"]


def overlays(match):
    """The broadcast's graphics as outline polygons (px), and the crop (x, y, w, h, ASPECT) below the ones at the top and left of the
    ones at the right below that."""
    O = [
        [x0 - MARGIN, y0 - MARGIN, x1 + MARGIN, y1 + MARGIN]
        for x0, y0, x1, y1 in json.load(open(f"{HERE}/out/{match}/score_bug.json"))["overlays"]
    ]
    top = max([b[3] for b in O if b[1] < 540] + [0])
    right = min([b[0] for b in O if b[3] > top and b[0] > 960] + [1920])
    w, h = min(right, round((1080 - top) * ASPECT)), 1080 - top
    h = min(h, round(w / ASPECT))
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]] for x0, y0, x1, y1 in O], [0, top, w, h]


HUES = ["red", "orange", "yellow", "lime", "green", "teal", "blue", "purple"]
GREYS = ["black", "dark grey", "light grey", "white"]


def jerseys(teams, labs):
    """One line naming both teams' jersey colours (the two strongest bins of each team's histogram)."""
    name = lambda h: " and ".join((HUES + GREYS)[b] for b in np.argsort(h)[::-1][:2] if h[b] > 0.15)
    hist = {
        side: np.average(
            np.array(teams["groups"])[[g for g, l in enumerate(labs) if l == side]],
            axis=0,
            weights=np.array(teams["sizes"])[[g for g, l in enumerate(labs) if l == side]],
        )
        for side in ("home", "away")
    }
    return f"one team in {name(hist['home'])} jerseys, the other in {name(hist['away'])}"


def group_sheet(path, crops, rank, per=14):
    """One row of jersey crops per group."""
    rows = []
    for g in range(rank.max() + 1):
        idx = np.flatnonzero(rank == g)
        idx = idx[np.linspace(0, len(idx) - 1, min(len(idx), per)).astype(int)]
        tiles = [cv2.resize(crops[j], (64, 128)) for j in idx] + [np.zeros((128, 64, 3), np.uint8)] * (per - len(idx))
        rows.append(np.hstack(tiles))
    cv2.imwrite(path, np.vstack(rows))


def build(match):
    ws = ok_windows(match)
    polys = outlines(match, ws)
    refs = referees(match, ws, polys)
    d = dict(zip(SC.GNAMES, np.load("cache/selfcal_v1.npz")["V3"]))
    R_ = rink(d)
    W = {
        w["id"]: dict(
            w,
            dir=f"{HERE}/out/{match}/win/{w['id']}",
            cam=np.load(f"{HERE}/out/{match}/win/{w['id']}/cams.npz")["cam"].astype(np.float64),
        )
        for w in ws
    }
    people, pxs, crops, ref_crops = {}, [], [], []
    for wid, w in W.items():
        for i in range(w["n"]):
            ps = polys.get((wid, i), [])
            ents = []
            if not ps:
                people[wid, i] = ents
                continue
            feet = to_ice(w["cam"][i], R_, np.array([[p[:, 0].mean(), p[:, 1].max()] for p in ps]))
            for m, (p, xy) in enumerate(zip(ps, feet)):
                if sdf(xy, R_) < -0.2:
                    ents.append(dict(id=-2, team=-1, xy=xy.round(3).tolist(), mask=m, ref=refs[wid, i][m]))
            people[wid, i] = ents
            if i % 8 == 0 and ents:
                bgr = cv2.imread(f"{w['dir']}/frames/{i + 1:03d}.jpg")
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
                for e in ents:
                    p = ps[e["mask"]]
                    x0, y0 = np.maximum(p.min(0).astype(int), 0)
                    x1, y1 = p.max(0).astype(int) + 1
                    if e["ref"]:
                        ref_crops.append(bgr[y0:y1, x0:x1])
                        continue
                    px = upper_px(img, p)
                    if px is None:
                        continue
                    pxs.append(px)
                    crops.append(bgr[y0:y1, x0:x1])
    teams, rank = team_groups(pxs)
    k = len(teams["groups"])
    group_sheet(f"{HERE}/out/{match}/teams.jpg", crops + ref_crops, np.r_[rank, np.full(len(ref_crops), k)])
    teams["labels"] = labels(teams)
    caption = jerseys(teams, teams["labels"])
    OVERLAYS, CROP = overlays(match)
    json.dump(teams["labels"], open(f"{HERE}/out/{match}/group_labels.json", "w"))
    print(match, teams["labels"], caption)
    atlas = [
        (wid, i)
        for wid, w in W.items()
        for i in np.linspace(0, w["n"] - 1, max(3, w["n"] // 50 + 1)).round().astype(int).tolist()
    ]
    adir = f"{PIPELINE}/atlas/{match}"
    os.makedirs(adir, exist_ok=True)
    for j, (wid, i) in enumerate(atlas):
        dst = f"{adir}/{j:03d}.jpg"
        if not os.path.lexists(dst):
            os.symlink(f"{W[wid]['dir']}/frames/{i + 1:03d}.jpg", dst)
    atlas_cam = np.array([W[wid]["cam"][i] for wid, i in atlas])
    atlas_outlines = [[np.round(p, 1).tolist() for p in polys.get(a, [])] + OVERLAYS for a in atlas]
    rows = []
    for wid, w in W.items():
        b = f"{PIPELINE}/{SET}/{match}_{wid}"
        os.makedirs(b, exist_ok=True)
        if not os.path.lexists(f"{b}/frames"):
            os.symlink(f"{w['dir']}/frames", f"{b}/frames")
        if not os.path.lexists(f"{b}/atlas_frames"):
            os.symlink(os.path.relpath(adir, b), f"{b}/atlas_frames")
        n = w["n"]
        np.savez(f"{b}/scene.npz", cam=w["cam"], puck=np.full((n, 2), np.nan), atlas_cam=atlas_cam)
        json.dump(
            dict(
                start=w["t0"],
                fps=FPS,
                n=n,
                **R_,
                segment=match,
                roster=[],
                teams=teams,
                people=[people[wid, i] for i in range(n)],
                outlines=[[np.round(p, 1).tolist() for p in polys.get((wid, i), [])] for i in range(n)],
                atlas_outlines=atlas_outlines,
            ),
            open(f"{b}/scene.json", "w"),
        )
        rows.append(dict(id=f"{match}_{wid}", segment=match, start=w["t0"], crop=CROP, jerseys=caption))
    path = f"{PIPELINE}/{SET}/windows.json"
    with open(f"{path}.lock", "w") as lock:  # builds of several matches may run at once
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = [r for r in (json.load(open(path)) if os.path.exists(path) else []) if r["segment"] != match]
        json.dump(old + rows, open(f"{path}.tmp", "w"), indent=0)
        os.replace(f"{path}.tmp", path)
    print(
        f"{match}: {len(rows)} bundles, people/frame {np.mean([len(v) for v in people.values()]):.1f}, atlas frames {len(atlas)}"
    )


if __name__ == "__main__":
    {"seg": seg, "build": build}[sys.argv[1]](sys.argv[2])
