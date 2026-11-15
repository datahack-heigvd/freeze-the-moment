"""Detect the yellow kickplate strip (boards/ice junction) and report how much of it is visible per frame."""

import pathlib
import sys
from multiprocessing import Pool

import cv2
import numpy as np


def detect(path):
    img = cv2.imread(str(path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (15, 110, 140), (35, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return 0, 0.0
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    w, a = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_AREA]
    return int(w), float(a)


if __name__ == "__main__":
    paths = sorted(pathlib.Path(sys.argv[1]).glob("*.jpg"))
    with Pool(6) as p:
        res = p.map(detect, paths, chunksize=8)
    w = np.array([r[0] for r in res])
    frac = w / 960
    print(
        f"{len(paths)} frames: longest yellow strip spans >50% of width in {(frac > 0.5).mean() * 100:.0f}%, >80% in {(frac > 0.8).mean() * 100:.0f}%, <25% in {(frac < 0.25).mean() * 100:.0f}%"
    )
