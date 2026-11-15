"""Fit rink outline (half-length, half-width, corner radius) to the bottom edge of the yellow kickplate using accurate per-frame cameras."""

import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
import camera
from check_overlay import cam_at, frame_at
from rink import boards
from scipy.optimize import least_squares


def kick_edge(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (15, 110, 140), (35, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    below = np.zeros_like(m)
    below[:-6] = m[6:]
    edge = (m > 0) & (below == 0)  # lower boundary of the strip
    return edge.astype(np.uint8)


def main(name, times):
    Z = np.load(f"cache/cam_{name}.npz")
    camera.LENS = tuple(Z["lens"][:2])
    data = []
    for t in times:
        img = frame_at(t)
        cam = cam_at(Z, t)
        e = kick_edge(img)
        dt = cv2.distanceTransform(1 - e, cv2.DIST_L2, 3)
        data.append((cam, np.minimum(dt, 25.0)))

    def pts(p):
        return boards(length=2 * p[0], width=2 * p[1], corner=p[2])

    def dense(b):
        return np.concatenate([np.linspace(b[i], b[i + 1], 3, endpoint=False) for i in range(len(b) - 1)])

    def res(p):
        P = dense(pts(p))
        out = []
        for cam, dt in data:
            q = camera.project_ice(cam, P)
            ok = (q[:, 0] > 3) & (q[:, 0] < 1916) & (q[:, 1] > 3) & (q[:, 1] < 1076)
            d = np.zeros(len(P))
            qi = np.round(q[ok]).astype(int)
            d[ok] = dt[qi[:, 1], qi[:, 0]]
            out.append(d)
        return np.concatenate(out)

    p0 = np.array([30.0, 14.7, 8.5])
    r0 = res(p0)
    print("start cost", (r0.sum() / (r0 > 0).sum()).round(2))
    best = None
    for hx in (28.5, 29.5, 30.0, 30.5):
        for r in (5.0, 8.5, 11.0):
            s = least_squares(
                res, [hx, 14.7, r], loss="soft_l1", f_scale=3, bounds=([27, 13.5, 3], [31.5, 16.5, 14]), diff_step=1e-3
            )
            r_ = res(s.x)
            c = r_.sum() / max((r_ > 0).sum(), 1)
            if best is None or c < best[0]:
                best = (c, s.x)
    print("best half-length, half-width, corner:", best[1].round(3), "cost", best[0].round(2))
    np.save("cache/rink_fit.npy", best[1])


if __name__ == "__main__":
    main("pano300_w2", [940, 960, 985, 1010, 1020, 1030, 1075, 1100.1, 1115, 1125, 1140, 1150, 1180, 1200])
