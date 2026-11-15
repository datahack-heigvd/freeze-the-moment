import camera
import cv2
import numpy as np
from rink import crease, goal_frame, lines


def project(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def draw(img, H, color=(0, 255, 255), w=2, only=None, cam=None, width=None):
    out = img.copy()
    for name, poly in (lines() if width is None else lines(width)).items():
        if only and name not in only:
            continue
        q = camera.project_ice(cam, poly) if cam is not None else project(H, poly)
        cv2.polylines(out, [q.astype(np.int32)], False, color, w, cv2.LINE_AA)
    if cam is not None:
        for sign, col in ((1, (0, 120, 255)), (-1, (0, 120, 255))):
            for poly in goal_frame(sign):
                q = camera.project_world(cam, poly)
                cv2.polylines(out, [q.astype(np.int32)], False, col, w, cv2.LINE_AA)
            cv2.polylines(out, [camera.project_ice(cam, crease(sign)).astype(np.int32)], False, (0, 120, 255), w, cv2.LINE_AA)
    return out
