"""Joint calibration of the static camera (position, distortion, principal point) and rink dims from
landmarks in the centre-ice frame and two end-zone frames. Per-frame unknowns: focal + rotation."""

import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "scripts")
from camera import project_ice, project_world
from rink import circle

BLUE = 7.14
cam0 = np.load("cache/cam_faceoff.npy")  # f, rv, C, k1, cxo, cyo
P = np.load("cache/poses_seg0.npz")

# ---- frame A: centre ice (920.6s) ------------------------------------------------------------
L, R, C = (np.load(f"cache/{n}_pts.npy") for n in ("blue_l", "blue_r", "centre"))
dots_px = np.array([(531.6, 412.9), (1373.4, 409.9), (332.2, 923.4), (1572.9, 910.0)])  # TL, TR, BL, BR
far_edge = np.load("cache/far_pts.npy").astype(float)
ys = np.linspace(-15, 15, 300)
xs_w = np.linspace(-29, 29, 300)


def line_resid(cam, xw, pts):
    q = project_ice(cam, np.stack([np.full(len(ys), xw), ys], 1))
    o = np.argsort(q[:, 1])
    return np.interp(pts[:, 1], q[o, 1], q[o, 0]) - pts[:, 0]


def ring_resid(cam, center, r, pts):
    q = project_ice(cam, circle(center[0], center[1], r, 240))
    return np.array([np.min(np.linalg.norm(q - p, axis=1)) for p in pts])


def resid_A(cam, s):
    dx, hb = s["dx"], s["hb"]
    out = [line_resid(cam, -BLUE, L), line_resid(cam, BLUE, R), line_resid(cam, 0, C) * 0.5]
    q = project_ice(cam, np.stack([xs_w, np.full(len(xs_w), hb)], 1))
    o = np.argsort(q[:, 0])
    out.append(np.interp(far_edge[:, 0], q[o, 0], q[o, 1]) - far_edge[:, 1])
    dots_w = np.array([(-dx, 7), (dx, 7), (-dx, -7), (dx, -7)], float)
    out.append((project_ice(cam, dots_w) - dots_px).ravel() * 0.7)
    return np.concatenate(out)


# ---- frame B: left end zone (1100.1s) --------------------------------------------------------
def resid_B(cam, s):
    out = []
    out.append((project_world(cam, [(-26, -0.915, 0)]) - [(250, 610)]).ravel() * 2)
    out.append((project_ice(cam, [(-20, -7), (-20, 7)]) - [(346.5, 866), (912.5, 500)]).ravel())
    out.append(ring_resid(cam, (-20, 7), s["rc"], [(635, 487), (1220, 512), (915, 415), (900, 605)]) * 0.7)
    out.append(line_resid_goal(cam, -26, [(150, 665), (50, 710)]))
    return np.concatenate(out)


def line_resid_goal(cam, xw, pts):
    q = project_ice(cam, np.stack([np.full(100, xw), np.linspace(-15, 15, 100)], 1))
    return np.array([np.min(np.linalg.norm(q - np.array(p), axis=1)) for p in pts])


# ---- frame C: right end zone (1020.6s) -------------------------------------------------------
def resid_C(cam, s):
    out = []
    out.append(
        (project_world(cam, [(26, 0.915, 0), (26, -0.915, 0), (26, 0.915, 1.22)]) - [(875, 330), (956, 355), (875, 252)]).ravel()
        * 2
    )
    out.append((project_ice(cam, [(20, 7)]) - [(300, 310)]).ravel())
    out.append(ring_resid(cam, (20, 7), s["rc"], [(350, 218), (350, 420), (612, 300)]) * 0.7)
    return np.concatenate(out)


frames = [(resid_A, 0), (resid_B, 359), (resid_C, 200)]
SHARED = ["Cx", "Cy", "Cz", "k1", "cxo", "cyo", "hb", "dx", "rc"]


def unpack(x):
    s = dict(zip(SHARED, x[:9]))
    s["hb"] = s["hb"]
    cams = []
    for i, (_, _idx) in enumerate(frames):
        f, rv = x[9 + 4 * i], x[10 + 4 * i : 13 + 4 * i]
        cams.append(np.concatenate([[f], rv, [s["Cx"], s["Cy"], s["Cz"]], [s["k1"], s["cxo"], s["cyo"]]]))
    return s, cams


def resid(x):
    s, cams = unpack(x)
    out = [fn(c, s) for (fn, _), c in zip(frames, cams)]
    out.append([s["k1"] * 40, s["cxo"] * 0.3, s["cyo"] * 0.3])
    return np.concatenate(out)


if __name__ == "__main__":
    x0 = [*cam0[4:7], cam0[7], cam0[8], cam0[9], 14.8, 5.65, 4.5]
    for _, idx in frames:
        if idx == 0:
            x0 += [cam0[0], *cam0[1:4]]
        else:
            x0 += [P["f"][idx], *P["rv"][idx]]
    x0 = np.array(x0, float)
    print("rms0", np.sqrt((resid(x0) ** 2).mean()))
    sol = least_squares(resid, x0, x_scale="jac", diff_step=1e-6, loss="soft_l1", f_scale=6.0, max_nfev=3000)
    s, cams = unpack(sol.x)
    print("rms", np.sqrt((resid(sol.x) ** 2).mean()), sol.status, sol.nfev)
    print({k: round(float(v), 3) for k, v in s.items()})
    for (fn, idx), c in zip(frames, cams):
        print(idx, "f", round(c[0]), "resid rms", np.sqrt((fn(c, s) ** 2).mean()).round(2))
    np.savez("cache/calib_global.npz", x=sol.x, cams=np.array(cams), **{k: v for k, v in s.items()})
