"""Runs on the GPU machine. Cuts every detection of dets.npz out of its frame: person crops (80x160, for identity) and wide crops around the
lower body (128x128, stick reach, for 'has the puck').
usage: crops_gpu.py <frames_dir> <dets.npz> <out_prefix>"""

import sys
from multiprocessing import Pool

import cv2
import numpy as np

PW, PH, WS = 80, 160, 128


def person(img, b):
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    x1, x2, y1, y2 = x1 - 0.05 * w, x2 + 0.05 * w, y1 - 0.03 * h, y2 + 0.03 * h
    return cut(img, x1, y1, x2, y2, PW, PH)


def wide(img, b):
    """Square around the feet, 1.6 box heights wide: the stick blade and a puck on it are inside."""
    x1, y1, x2, y2 = b
    h = y2 - y1
    cx = (x1 + x2) / 2
    s = 0.8 * h
    return cut(img, cx - s, y2 - 1.2 * s, cx + s, y2 + 0.4 * s, WS, WS)


def cut(img, x1, y1, x2, y2, w, h):
    M = np.array([[w / (x2 - x1), 0, -x1 * w / (x2 - x1)], [0, h / (y2 - y1), -y1 * h / (y2 - y1)]], np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)


def job(i):
    sel = np.flatnonzero(IMG == i)
    if not len(sel):
        return sel, None, None
    img = cv2.imread(f"{frames_dir}/{i:05d}.jpg")
    return sel, np.stack([person(img, BOX[k]) for k in sel]), np.stack([wide(img, BOX[k]) for k in sel])


if __name__ == "__main__":
    frames_dir, dets, out = sys.argv[1:4]
    Z = np.load(dets)
    IMG, BOX = Z["img"], Z["box"]
    P = np.lib.format.open_memmap(f"{out}_person.npy", "w+", np.uint8, (len(IMG), PH, PW, 3))
    W = np.lib.format.open_memmap(f"{out}_wide.npy", "w+", np.uint8, (len(IMG), WS, WS, 3))
    with Pool(10) as pool:
        for n, (sel, p, w) in enumerate(pool.imap_unordered(job, sorted(set(IMG.tolist())), chunksize=8)):
            if p is not None:
                P[sel] = p
                W[sel] = w
            if n % 1000 == 0:
                print(n, flush=True)
    P.flush()
    W.flush()
    print("done", len(IMG))
