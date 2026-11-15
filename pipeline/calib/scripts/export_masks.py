"""Export person outlines for the live/ AR overlay, so ice-painted graphics can be cut out around everybody standing on the ice.
Input: segmentation of every frame (seg_poly_gpu.py). Keeps the people whose outline touches the rink in the image.
  <segment>/masks.bin: uint32 n_frames, uint32 offsets[n_frames + 1] (in int16 units, from the start of the data), int16 data;
                       a frame holds its outlines back to back as [n_points, x0, y0, x1, y1, ...] with coordinates in 1/8 video pixel
usage: [WIN=<window>] [AR_ROOT=<dir>] export_masks.py <seg.npz>"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
import selfcal as S
from opt2 import FRAME0, Detections
from window import W as WIN

AR = os.environ.get("AR_ROOT", "../../store/ar")
OUT = f"{AR}/{os.path.basename(WIN['out'])}"
MARGIN, SCALE = 0.5, 4  # the rink outline is grown by this many metres; its image mask is rasterised at 1/SCALE resolution
W_, H_ = 1920, 1080


def rink_outline(hx, hy, c, margin):
    out = []
    for sx, sy, a0 in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        for k in range(13):
            a = np.radians(a0 + 90 * k / 12)
            out.append([sx * (hx - c) + (c + margin) * np.cos(a), sy * (hy - c) + (c + margin) * np.sin(a)])
    return np.array(out)


def main(seg):
    Z = np.load(seg)
    dets, pts = Z["dets"], Z["pts"]
    DT = Detections()
    d = dict(zip(S.GNAMES, DT.g))
    ring = np.c_[rink_outline(d["hx"], d["hy"], d["corner"], MARGIN), np.zeros(52)]
    first = np.searchsorted(dets[:, 0], np.arange(WIN["n"] + 1))
    start = np.r_[0, np.cumsum(dets[:, 6]).astype(np.int64)]
    n0 = int(DT.Z["N"][0])
    n1 = n0 + WIN["n"]
    offs, data, kept = [0], [], 0
    for N in range(n0, n1):
        k = N - FRAME0
        row = DT.rowN.get(N)
        chunk = []
        if row is not None and 0 <= k < WIN["n"]:
            px = S.project(DT.g, DT.Z["f"][row], DT.Z["rv"][row], ring)
            mask = np.zeros((H_ // SCALE, W_ // SCALE), np.uint8)
            if np.isfinite(px).all():
                cv2.fillPoly(mask, [np.round(px / SCALE).astype(np.int32)], 1)
            for j in range(first[k], first[k + 1]):
                p = pts[start[j] : start[j + 1]]
                q = np.clip((p / (8 * SCALE)).astype(np.int32), 0, [W_ // SCALE - 1, H_ // SCALE - 1])
                if mask[q[:, 1], q[:, 0]].any():
                    chunk += [len(p), *p.ravel().tolist()]
                    kept += 1
        data.append(np.array(chunk, np.int16))
        offs.append(offs[-1] + len(chunk))
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/masks.bin", "wb") as f:
        f.write(np.array([n1 - n0], np.uint32).tobytes())
        f.write(np.array(offs, np.uint32).tobytes())
        f.write(np.concatenate(data).tobytes())
    print(
        f"{n1 - n0} frames, {kept / (n1 - n0):.1f} people on the ice per frame (of {len(dets) / WIN['n']:.1f} detected), {os.path.getsize(f'{OUT}/masks.bin') / 1e6:.1f} MB"
    )


if __name__ == "__main__":
    main(sys.argv[1])
