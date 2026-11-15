import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "scripts")
from camera import from_homography, project_ice
from overlay import draw
from rink import BLUE_LINE

img = cv2.imread("cache/faceoff_920.png")
L, R, C = (np.load(f"cache/{n}_pts.npy") for n in ("blue_l", "blue_r", "centre"))
dots_px = np.array([(531.6, 412.9), (1373.4, 409.9), (332.2, 923.4), (1572.9, 910.0)])  # TL, TR, BL, BR
far_px = np.array([(493, 243), (953, 243), (1408, 243)], float)
far_w = np.array([(-BLUE_LINE, 15), (0, 15), (BLUE_LINE, 15)], float)
ys = np.linspace(-15, 15, 300)
far_edge = np.load("cache/far_pts.npy").astype(float)
xs_w = np.linspace(-29, 29, 300)


def line_resid(cam, xw, pts):
    q = project_ice(cam, np.stack([np.full(len(ys), xw), ys], 1))
    order = np.argsort(q[:, 1])
    return np.interp(pts[:, 1], q[order, 1], q[order, 0]) - pts[:, 0]


def resid(p, with_k1=True):
    cam, dx, hb, b = p[:10].copy(), p[10], p[11], BLUE_LINE

    out = [line_resid(cam, -b, L), line_resid(cam, b, R), line_resid(cam, 0, C) * 0.5]
    q = project_ice(cam, np.stack([xs_w, np.full(len(xs_w), hb)], 1))
    order = np.argsort(q[:, 0])
    out.append(np.interp(far_edge[:, 0], q[order, 0], q[order, 1]) - far_edge[:, 1])
    out.append([cam[7] * 40.0, cam[8] * 0.3, cam[9] * 0.3])
    dots_w = np.array([(-dx, 7), (dx, 7), (-dx, -7), (dx, -7)], float)
    out.append((project_ice(cam, dots_w) - dots_px).ravel() * 0.7)
    return np.concatenate(out)


p0 = np.concatenate([from_homography(np.load("cache/H_faceoff_init.npy")), [0.0, 0.0, 0.0, 5.64, 15.0]])
sol = least_squares(resid, p0, x_scale="jac", diff_step=1e-6, loss="linear", max_nfev=2000)
r = resid(sol.x)
print(sol.status, sol.nfev, "rms", np.sqrt((resid(p0) ** 2).mean()).round(2), "->", np.sqrt((r**2).mean()).round(2))
print("f,rv,C,k1,dx,hb:", np.round(sol.x, 4))
cam = sol.x[:10]
np.save("cache/cam_faceoff.npy", cam)
np.save("cache/hb_faceoff.npy", sol.x[11:12])
cv2.imwrite("cache/overlay_refined.jpg", draw(img, None, cam=cam, width=2 * sol.x[11]))
