"""Per-frame bodies (body.py) vs refitted bodies (fit.py), per person and overall (median over people):
  jitter    3D joint acceleration between consecutive frames (cm per frame²), all body joints / the wrists
  size      variation of the summed bone lengths over the clip (coefficient of variation, %)
  2D        median distance (px) of the projected joints to YOLO11x-pose's (pose2d.json, conf > 0.5) and to SAM 3D Body's own
  ice       median height (cm) of the lowest body point

usage (from pipeline/, YOLO venv):  fitcheck.py [bundle]"""

import json
import sys

import numpy as np

B = sys.argv[1] if len(sys.argv) > 1 else "bundle"
BODY = list(range(21)) + [41, 62]
COCO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
BONES = [(5, 6), (5, 7), (6, 8), (9, 11), (10, 12), (11, 13), (12, 14)]


def project(S, cam, P):
    R, C = cam[2:11].reshape(3, 3), cam[11:14]
    pc = (P - C) @ R.T
    uv = cam[0] * pc[:, :2] / pc[:, 2:3]
    return uv * (1 + cam[1] * (uv**2).sum(1, keepdims=True) / S["fref"] ** 2) + [S["cx"], S["cy"]]


def distort(S, cam, px):
    c = np.array([S["cx"], S["cy"]])
    u = px - c
    return c + u * (1 + cam[1] * (u**2).sum(1, keepdims=True) / S["fref"] ** 2)


def measure(S, cams, yolo, raw, b):
    out = {}
    for key in np.unique(b["key"]):
        rows = np.flatnonzero(b["key"] == key)
        rows = rows[np.argsort(b["frame"][rows])]
        fr = b["frame"][rows]
        J = b["joints"][rows]
        ok = (np.diff(fr)[1:] == 1) & (np.diff(fr)[:-1] == 1)
        acc = np.linalg.norm(J[2:] - 2 * J[1:-1] + J[:-2], axis=-1)[ok] * 100
        size = sum(np.linalg.norm(J[:, a] - J[:, c], axis=-1) for a, c in BONES)
        dy, ds = [], []
        for r, f, j in zip(rows, fr, J):
            px = project(S, cams[f], j.astype(np.float64))
            ds.extend(np.linalg.norm(px[BODY] - distort(S, cams[f], raw["kp2d"][r][BODY].astype(np.float64)), axis=1))
            y = yolo[f].get(str(int(key)))
            if y:
                y = np.array(y)
                m = y[:, 2] > 0.5
                dy.extend(np.linalg.norm(px[COCO][m] - y[m, :2], axis=1))
        low = b["verts"][rows][:, ::25, 2].astype(np.float32).min(1) * 100
        if len(acc):
            out[int(key)] = dict(
                jitter=np.median(acc[:, BODY]),
                wrist=np.median(acc[:, [41, 62]]),
                size=size.std() / size.mean() * 100,
                yolo=np.median(dy) if dy else np.nan,
                sam=np.median(ds),
                ice=np.median(low),
            )
    return out


def main():
    S = json.load(open(f"{B}/scene.json"))
    cams = np.load(f"{B}/scene.npz")["cam"].astype(np.float64)
    yolo = json.load(open(f"{B}/pose2d.json"))
    raw = np.load(f"{B}/bodies.npz")
    fit = np.load(f"{B}/bodies_fit.npz")
    m = {name: measure(S, cams, yolo, raw, b) for name, b in (("per-frame", raw), ("refit", fit))}
    cols = ("jitter", "wrist", "size", "yolo", "sam", "ice")
    print(f"{'person':>6} " + " ".join(f"{c:>17}" for c in cols))
    for key in m["per-frame"]:
        print(f"{key:>6} " + " ".join(f"{m['per-frame'][key][c]:7.2f} -> {m['refit'][key][c]:6.2f}" for c in cols))
    print(
        f"{'median':>6} "
        + " ".join(
            f"{np.nanmedian([v[c] for v in m['per-frame'].values()]):7.2f} -> {np.nanmedian([v[c] for v in m['refit'].values()]):6.2f}"
            for c in cols
        )
    )


if __name__ == "__main__":
    main()
