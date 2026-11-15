"""How well each reconstructed person lines up with the footage, per frame: the IoU of their body (render.py's Cast, as exported)
projected through the broadcast camera with their own outline (cast.json's mask), both drawn at SCALE of the source; the other
people's outlines are left out of both (an occluder hides the body there, and the outline stops at it); and the offset of the
body's centre from the outline's (source px), which, shared by everyone in a frame, is the camera's error rather than the body's.
Frames where the person has no outline, or it is under MIN_AREA source pixels, have none.

  bundle/align.json   {key: [[IoU, dx, dy] or null per frame]}   (align_<BODIES>.json for other bodies than the per-frame fits)

usage (LTX venv, in a goal dir):  BUNDLE=bundle BODIES=bodies align.py"""

import json

import cv2
import numpy as np
import torch
from goal import render as R

SCALE, MIN_AREA = 0.5, 800


def main():
    R.load_cast()
    with torch.no_grad():
        cast = R.Cast()
    faces = cast.faces.long()
    w, h = int(1920 * SCALE), int(1080 * SCALE)
    outlines = R.S["outlines"]
    out = {}
    for t in cast.tracks:
        k = t["key"]
        ious = []
        for i in range(R.S["n"]):
            m = t["mask"][i]
            if m < 0 or not t["seen"][i] or cv2.contourArea(np.array(outlines[i][m], np.float32)) < MIN_AREA:
                ious.append(None)
                continue
            real, others, body = (np.zeros((h, w), np.uint8) for _ in range(3))
            cv2.fillPoly(real, [np.round(np.array(outlines[i][m]) * SCALE).astype(np.int32)], 1)
            for j, p in enumerate(outlines[i]):
                if j != m:
                    cv2.fillPoly(others, [np.round(np.array(p) * SCALE).astype(np.int32)], 1)
            with torch.no_grad():
                uv, z = R.src_project(R.Z["cam"][i], cast.verts[k][i])
            tri = (uv[faces] * SCALE).round().int().cpu().numpy()
            front = (z[faces] > 0).all(1).cpu().numpy()
            cv2.fillPoly(body, list(tri[front]), 1)
            keep = others == 0
            a, b = (real > 0) & keep, (body > 0) & keep
            ca, cb = np.argwhere(a).mean(0), np.argwhere(b).mean(0) if b.any() else np.full(2, np.nan)
            ious.append(
                [round(float((a & b).sum() / max((a | b).sum(), 1)), 3), *[round(float(x), 1) for x in (cb - ca)[::-1] / SCALE]]
            )
        out[k] = ious
        v = [x[0] for x in ious if x is not None]
        if v:
            print(
                k,
                t["team"],
                f"frames {len(v)} IoU median {np.median(v):.2f} p10 {np.percentile(v, 10):.2f} under 0.5: {np.mean(np.array(v) < 0.5):.0%}",
                flush=True,
            )
    json.dump(out, open(f"{R.B}/align.json" if R.BODIES == "bodies" else f"{R.B}/align_{R.BODIES}.json", "w"))
    allv = [x[0] for v in out.values() for x in v if x is not None]
    print(
        f"all: {len(allv)} person-frames, IoU median {np.median(allv):.2f}, p10 {np.percentile(allv, 10):.2f}, under 0.5: {np.mean(np.array(allv) < 0.5):.0%}"
    )


if __name__ == "__main__":
    main()
