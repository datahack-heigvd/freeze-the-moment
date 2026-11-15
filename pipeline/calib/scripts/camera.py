"""Pinhole camera over the ice plane. params = [f, rotvec(3), C(3)] with principal point at image centre."""

import cv2
import numpy as np

W, H_IMG = 1920, 1080
CX, CY = W / 2, H_IMG / 2


def K_of(f):
    return np.array([[f, 0, CX], [0, f, CY], [0, 0, 1.0]])


def homography(params):
    f, rv, C = params[0], params[1:4], params[4:7]
    R, _ = cv2.Rodrigues(np.asarray(rv, float))
    t = -R @ C
    return K_of(f) @ np.column_stack([R[:, 0], R[:, 1], t])


def project3d(params, pts):
    f, rv, C = params[0], params[1:4], params[4:7]
    R, _ = cv2.Rodrigues(np.asarray(rv, float))
    pc = (np.asarray(pts, float) - C) @ R.T
    uv = pc @ K_of(f).T
    return uv[:, :2] / uv[:, 2:3]


def from_homography(H, f_range=(800, 8000)):
    """Decompose H = K[r1 r2 t] with principal point fixed."""
    H = H / np.linalg.norm(np.linalg.inv(K_of(2000)) @ H[:, 0])
    best = None
    for f in np.linspace(*f_range, 600):
        A = np.linalg.inv(K_of(f)) @ H
        a1, a2 = A[:, 0], A[:, 1]
        err = abs(a1 @ a2) / (np.linalg.norm(a1) * np.linalg.norm(a2)) + abs(
            np.linalg.norm(a1) - np.linalg.norm(a2)
        ) / np.linalg.norm(a1)
        if best is None or err < best[0]:
            best = (err, f)
    f = best[1]
    A = np.linalg.inv(K_of(f)) @ H
    lam = (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1])) / 2
    r1, r2, t = A[:, 0] / lam, A[:, 1] / lam, A[:, 2] / lam
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if t[2] < 0:  # camera must be in front of the plane
        R = np.column_stack([-R[:, 0], -R[:, 1], R[:, 2]])
        t = -t
    rv, _ = cv2.Rodrigues(R)
    C = -R.T @ t
    return np.concatenate([[f], rv.ravel(), C])


def project_ice(params, xy):
    """World ice-plane points (N,2) -> pixels, with radial distortion k1 = params[7] if present."""
    xy = np.asarray(xy, float)
    return project_world(params, np.column_stack([xy, np.zeros(len(xy))]))


LENS = (1.0, 0.0)  # k1 multiplier and zoom exponent: k1_eff = k1 * km * (F_REF / f) ** pw
F_REF = 3050.0  # distortion is defined on pixel radius / F_REF, independent of the current zoom


def project_world(params, pts):
    pts = np.asarray(pts, float)
    f, rv, C = params[0], params[1:4], params[4:7]
    k1 = params[7] * LENS[0] * (F_REF / f) ** LENS[1] if len(params) > 7 else 0.0
    cx, cy = (CX + params[8], CY + params[9]) if len(params) > 9 else (CX, CY)
    R, _ = cv2.Rodrigues(np.asarray(rv, float))
    pc = (pts - C) @ R.T
    u = pc[:, :2] / pc[:, 2:3] * f  # undistorted pixel offsets
    d = u * (1 + k1 * (u**2).sum(1, keepdims=True) / F_REF**2)
    return d + [cx, cy]
