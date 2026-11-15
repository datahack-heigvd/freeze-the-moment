"""Export the per-frame camera and tracker output for the live/ AR overlay.
  store/ar/camera.bin : float32 x 14 per frame [f, k, R (9, row major), C (3)]  (pc = R (P - C), u = f pc_xy / pc_z, radial k); all zeros where the pose is unreliable.
                       The per-frame solves smoothed into a physical head trajectory (camsmooth.py); camera_raw.bin keeps the solves
  store/ar/tracks.json: {n0, fps, cx, cy, hx, hy, corner, rings: [[idx, x_cm, y_cm, flags]...] per frame, ctl: [[idx, x_cm, y_cm, team]...] per frame}
rings: players to draw (idx = roster index, -1 for an unnamed player); flags bit0 = number shown, bit1 = team (0 home, 1 away)
ctl  : every on-ice player position for zone control, including players out of view (idx = roster index, -1 unnamed)
car  : per frame, roster index of the puck carrier seen in the video inside a metadata gap, -1 elsewhere
When gapid.py has solved the chunk (cache/full/id/result_<N0>.pkl), its rings are used: the tracker's numbers plus the appearance
solver's where the tracker had none, and the video carrier.
usage: [WIN=<window>] export_ar.py   (writes store/ar/<window>/ and refreshes store/ar/index.json)"""

import json
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
import camsmooth as CS
import ptrack as PT
import selfcal as S
from window import W as WIN

GAME = "../../store/site/game.json"
AR = os.environ.get(
    "AR_ROOT", "../../store/ar"
)  # where segments are written (the full-game run stages them elsewhere and swaps them in at the end)
OUT = f"{AR}/{os.path.basename(WIN['out'])}"
MIN_INL, MAX_RMS = (
    20,
    6.0,
)  # a frame whose pose has fewer inliers or a worse fit gets an all-zero camera row: the app draws nothing there
ANON_SAME = 1.5  # m


def camera_row(g, f_pano, rv):
    d = dict(zip(S.GNAMES, g))
    f = d["s"] * f_pano
    R = cv2.Rodrigues(rv)[0] @ cv2.Rodrigues(np.array([d["rg0"], d["rg1"], d["rg2"]]))[0]
    C = S.C_BASE + np.array([d["dC0"], d["dC1"], d["dC2"]]) + np.array([d["dCf0"], d["dCf1"], d["dCf2"]]) * (f / S.F_REF - 1)
    k = d["kb"] * (S.F_REF / f) ** d["pw"]
    return np.concatenate([[f, k], R.ravel(), C]).astype(np.float32), d


def write_index():
    """store/ar/index.json lists every exported window; the app starts on the first one that contains a goal."""
    import glob

    segs = sorted((json.load(open(f)) for f in glob.glob(f"{AR}/*/meta.json")), key=lambda m: m["t0"])
    goals = [e["t"] for e in json.load(open(GAME))["events"] if e["kind"] == "goal"]
    start = next((m["t0"] for m in segs if any(m["t0"] <= t <= m["t1"] for t in goals)), segs[0]["t0"])
    json.dump(dict(segments=segs, defaultStart=start), open(f"{AR}/index.json", "w"))


def main():
    W = PT.World()
    res = pickle.load(open(PT.CACHE.format(""), "rb"))
    show = PT.display(res, W)
    DT = W.DT
    n0, n1 = W.n0, W.n1
    cam = np.zeros((n1 - n0, 14), np.float32)
    bad = 0
    for N in range(n0, n1):
        k = DT.rowN.get(N)
        if k is None or not (DT.Z["inl"][k] >= MIN_INL and DT.Z["rms"][k] <= MAX_RMS and np.isfinite(DT.Z["f"][k])):
            bad += 1
            continue
        cam[N - n0] = camera_row(DT.g, DT.Z["f"][k], DT.Z["rv"][k])[0]
    d = dict(zip(S.GNAMES, DT.g))
    state = {}  # roster player -> {frame: position} of every smoothed state (seen or not)
    for p, spans in res["tracks"].items():
        for s_ in spans:
            for i, q in enumerate(s_["pos"]):
                state.setdefault(s_["N0"] + i, []).append((p, PT.clamp_inside(q, W.hx, W.hy, W.c)))
    gap_path = f"cache/full/id/result_{WIN['frame0']}.pkl"
    G_ = pickle.load(open(gap_path, "rb")) if os.path.exists(gap_path) else dict(rings={}, car={})
    rings, ctl = [], []
    cm = lambda v: int(round(float(v) * 100))
    for N in range(n0, n1):
        fr = G_["rings"].get(N) or PT.frame_rings(show, W, N)
        rings.append([v for p, xy, num, team in fr for v in (max(p, -1), cm(xy[0]), cm(xy[1]), int(num) | team << 1)])
        c = [v for p, q in state.get(N, []) for v in (p, cm(q[0]), cm(q[1]), int(W.team_p[p]))]
        own = state.get(N, [])
        c += [
            v
            for p, xy, num, team in fr
            if p < 0 and all(W.team_p[q] != team or np.linalg.norm(xy - qxy) > ANON_SAME for q, qxy in own)
            for v in (-1, cm(xy[0]), cm(xy[1]), team)
        ]  # an unnamed ring on a tracked teammate is the same person
        ctl.append(c)
    os.makedirs(OUT, exist_ok=True)
    cam.tofile(f"{OUT}/camera_raw.bin")
    CS.smooth_cameras(cam.astype(np.float64), CS.LAM)[0].astype(np.float32).tofile(f"{OUT}/camera.bin")
    car = [int(G_["car"].get(N, -1)) for N in range(n0, n1)]
    json.dump(
        dict(
            n0=n0,
            fps=25,
            fref=S.F_REF,
            cx=960 + d["cxo"],
            cy=540 + d["cyo"],
            hx=d["hx"],
            hy=d["hy"],
            corner=d["corner"],
            rings=rings,
            ctl=ctl,
            car=car,
        ),
        open(f"{OUT}/tracks.json", "w"),
        separators=(",", ":"),
    )
    json.dump(
        dict(id=os.path.basename(OUT), base=os.path.basename(WIN["out"]), t0=n0 / 25, t1=n1 / 25, bad=bad),
        open(f"{OUT}/meta.json", "w"),
    )
    write_index()
    print(
        "frames",
        n1 - n0,
        "no pose",
        bad,
        f"rings/frame {np.mean([len(r) / 4 for r in rings]):.1f}",
        f"ctl/frame {np.mean([len(c) / 4 for c in ctl]):.1f}",
        "video carrier frames",
        sum(c >= 0 for c in car),
    )


if __name__ == "__main__":
    main()
