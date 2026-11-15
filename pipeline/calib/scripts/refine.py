"""Absolute per-frame refinement: align projected rink paint to red/blue paint evidence (chamfer), fixed camera position."""

import camera
import cv2
import numpy as np
from masks import ridge_masks
from rink import lines
from scipy.optimize import least_squares

PAINT = ("centre", "blue_l", "blue_r", "goal_l", "goal_r", "centre_circle", "fo_-20_-7", "fo_-20_7", "fo_20_-7", "fo_20_7")


def evidence(img, cam0, scale=0.5, cutoff=40):
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    red, blue = ridge_masks(small, 8, 8)
    ice = np.zeros(small.shape[:2], np.uint8)
    poly = camera.project_ice(cam0, lines(30.0)["boards"]) * scale
    cv2.fillPoly(ice, [poly.astype(np.int32)], 1)
    ice = cv2.erode(ice, np.ones((9, 9), np.uint8))
    return [np.minimum(cv2.distanceTransform((1 - m * ice).astype(np.uint8), cv2.DIST_L2, 3), cutoff) for m in (red, blue)]


def paint_points(width=30.0):
    L = lines(width)
    red, blue = [], []
    for n in PAINT:
        p = L[n]
        # densify
        q = np.concatenate([np.linspace(p[i], p[i + 1], 4, endpoint=False) for i in range(len(p) - 1)])
        (blue if n.startswith("blue") else red).append(q)
    return np.concatenate(red), np.concatenate(blue)


RED, BLUE = paint_points()
CUT = 40.0


def residuals(x, cam0, dts, scale):
    cam = cam0.copy()
    cam[0] = x[0]
    cam[1:4] = x[1:4]
    out = []
    for pts, dt in ((RED, dts[0]), (BLUE, dts[1])):
        q = camera.project_ice(cam, pts) * scale
        ok = (q[:, 0] > 2) & (q[:, 0] < dt.shape[1] - 3) & (q[:, 1] > 2) & (q[:, 1] < dt.shape[0] - 3)
        d = np.full(len(pts), CUT)
        qi = np.round(q[ok]).astype(int)
        d[ok] = np.minimum(dt[qi[:, 1], qi[:, 0]], CUT)
        out.append(d)
    return np.concatenate(out)


def refine(img, cam0):
    x0 = np.concatenate([[cam0[0]], cam0[1:4]])
    for scale, cut in ((0.25, 40.0), (0.5, 16.0), (0.5, 8.0)):
        global CUT
        CUT = cut
        dts = evidence(img, cam0, scale, cut)
        sol = least_squares(
            residuals,
            x0,
            args=(cam0, dts, scale),
            x_scale=[60, 0.005, 0.005, 0.005],
            loss="soft_l1",
            f_scale=3.0 * scale * 2,
            max_nfev=80,
            diff_step=1e-4,
        )
        x0 = sol.x
    c = cam0.copy()
    c[0] = x0[0]
    c[1:4] = x0[1:4]
    return c, float(np.mean(residuals(x0, cam0, dts, scale) < 3))


def search(img, cam0, n=4000, df=0.15, dr=0.07, seed=0, top=8):
    """Global search around cam0 over (f, rotvec) using a smooth coarse chamfer, then LM refinement of the best candidates."""
    rng = np.random.default_rng(seed)
    sc = 0.25
    dts = evidence(img, cam0, sc, 60)
    x0 = np.concatenate([[cam0[0]], cam0[1:4]])
    global CUT
    CUT = 60.0
    cands = x0 + rng.uniform(-1, 1, (n, 4)) * [x0[0] * df, dr, dr, dr]
    cands[0] = x0
    costs = np.array([np.mean(residuals(x, cam0, dts, sc)) for x in cands])
    order = np.argsort(costs)[:top]
    out = []
    for i in order:
        x = cands[i]
        for scale, cut in ((0.25, 30.0), (0.5, 12.0), (0.5, 6.0)):
            CUT = cut
            d = evidence(img, cam0, scale, cut)
            sol = least_squares(
                residuals,
                x,
                args=(cam0, d, scale),
                x_scale=[60, 0.005, 0.005, 0.005],
                loss="soft_l1",
                f_scale=2.0,
                max_nfev=50,
                diff_step=1e-4,
            )
            x = sol.x
        CUT = 6.0
        out.append((float(np.mean(residuals(x, cam0, d, 0.5) < 2.5)), x))
    out.sort(key=lambda t: -t[0])
    c = cam0.copy()
    c[0] = out[0][1][0]
    c[1:4] = out[0][1][1:4]
    return c, out[0][0], out
