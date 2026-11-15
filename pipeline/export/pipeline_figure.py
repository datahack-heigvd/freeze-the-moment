"""The pipeline on one broadcast frame, as panels to stack: the frame; the solved camera (the rink's lines drawn through it); the
tracked people (boxes from their outlines, team colours, track keys); their segmentation outlines; the fitted bodies' 2D keypoints;
the fitted 3D bodies projected back; how well each lines up with its outline (align.py's IoU); and everyone's rink position from
above with their last second's path and the puck.

  <out>/panel_<n>.jpg   PANEL_W wide, each with its title bar

usage (LTX venv, in a goal dir):  BUNDLE=bundle BODIES=bodies pipeline_figure.py <frame> <out>"""

import json
import math
import os
import sys

import cv2
import numpy as np
import torch
from goal import render as R

PANEL_W, BAR = 1280, 44
TEAM = [(40, 170, 250), (210, 110, 40), (160, 160, 160)]  # BGR: team 0, team 1, officials
SCALE = 30  # px per m on the rink map


def team(t):
    return TEAM[t["team"]] if t["team"] in (0, 1) else TEAM[2]


def titled(img, title):
    h = int(img.shape[0] * PANEL_W / img.shape[1])
    img = cv2.resize(img, (PANEL_W, h), interpolation=cv2.INTER_AREA)
    bar = np.full((BAR, PANEL_W, 3), 24, np.uint8)
    cv2.putText(bar, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 240), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def rink_lines():
    """World polylines of the rink's markings that render.py defines: its outline, the centre line and circle, the goal lines."""
    s = np.linspace(0, 2 * math.pi, 400)
    hx, hy, c = R.HX, R.HY, R.CR
    out = []
    a, b = hx - c, hy - c
    ring = []
    for q in range(4):  # rounded rectangle, quarter by quarter
        cx, cy = (a if q in (0, 3) else -a), (b if q in (0, 1) else -b)
        for t in np.linspace(q * math.pi / 2, (q + 1) * math.pi / 2, 30):
            ring.append((cx + c * math.cos(t), cy + c * math.sin(t)))
    out.append(np.array(ring + ring[:1]))
    out.append(np.array([(0, -hy), (0, hy)]))
    out.append(np.c_[4.5 * np.cos(s), 4.5 * np.sin(s)])
    for x in (-R.GOAL_X, R.GOAL_X):
        y = math.sqrt(max(c * c - (abs(x) - a) ** 2, 0)) + b if abs(x) > a else hy
        out.append(np.array([(x, -y), (x, y)]))
    return out


def project(i, P):
    with torch.no_grad():
        uv, z = R.src_project(R.Z["cam"][i], torch.tensor(P, device=R.DEV, dtype=torch.float32))
    return uv.cpu().numpy(), z.cpu().numpy()


def main():
    i, out = int(sys.argv[1]), sys.argv[2]
    os.makedirs(out, exist_ok=True)
    R.load_cast()
    tracks = json.load(open(f"{R.B}/cast.json"))["tracks"]
    O = R.S["outlines"][i]
    with torch.no_grad():
        cast = R.Cast()
    frame = cv2.imread(f"{R.B}/frames/{i + 1:03d}.jpg")
    dim = (frame * 0.55).astype(np.uint8)
    here = [t for t in tracks if t["seen"][i] and t["mask"][i] >= 0]
    panels = [titled(frame, f"1  Broadcast frame (source frame {i})")]

    img = dim.copy()
    for L in rink_lines():
        P = np.c_[L, np.zeros(len(L))]
        dense = np.concatenate([np.linspace(P[k], P[k + 1], 20) for k in range(len(P) - 1)])
        uv, z = project(i, dense)
        uv = uv[z > 0]
        cv2.polylines(img, [np.round(uv).astype(np.int32)], False, (80, 255, 255), 3, cv2.LINE_AA)
    panels.append(titled(img, "2  Camera calibration: the rink's lines drawn through the solved camera"))

    img = frame.copy()
    for t in here:
        x, y, w, h = cv2.boundingRect(np.round(np.array(O[t["mask"][i]])).astype(np.int32))
        col = team(t)
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 3)
        cv2.putText(
            img,
            f"#{t['key']}" + (f" {t['jersey']}" if t.get("jersey") else ""),
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            col,
            2,
            cv2.LINE_AA,
        )
    panels.append(titled(img, f"3  Detection and tracking: {len(here)} people, boxes in team colours with track keys"))

    img = frame.copy()
    over = frame.copy()
    for t in here:
        p = np.round(np.array(O[t["mask"][i]])).astype(np.int32)
        cv2.fillPoly(over, [p], team(t))
        cv2.polylines(img, [p], True, (255, 255, 255), 2, cv2.LINE_AA)
    img = cv2.addWeighted(over, 0.45, img, 0.55, 0)
    panels.append(titled(img, "4  Segmentation: each person's outline"))

    b = np.load(f"{R.B}/{R.BODIES}.npz")
    img = dim.copy()
    for m in np.flatnonzero(b["frame"] == i):
        kp = b["kp2d"][m][R.OPENPOSE].astype(np.float32)
        for (p, q), col in zip(R.LIMBS, R.POSE_COLORS):
            cv2.line(img, tuple(np.round(kp[p]).astype(int)), tuple(np.round(kp[q]).astype(int)), col[::-1], 3, cv2.LINE_AA)
    panels.append(titled(img, "5  2D pose: each fitted body's keypoints"))

    img = dim.copy()
    faces = cast.faces.long()
    for t in here:
        if t["key"] not in cast.verts:
            continue
        uv, z = project(i, cast.verts[t["key"]][i].cpu().numpy())
        tri = np.round(uv[faces.cpu().numpy()]).astype(np.int32)
        front = (z[faces.cpu().numpy()] > 0).all(1)
        over = img.copy()
        cv2.fillPoly(over, list(tri[front]), team(t))
        img = cv2.addWeighted(over, 0.75, img, 0.25, 0)
    panels.append(titled(img, "6  3D bodies (SAM 3D Body, placed on the rink), projected back through the camera"))

    A = json.load(open(f"{R.B}/align.json")) if os.path.exists(f"{R.B}/align.json") else {}
    img = frame.copy()
    for t in here:
        a = (A.get(str(t["key"])) or [None] * (i + 1))[i]
        if not a or isinstance(a, (int, float)):
            continue
        p = np.round(np.array(O[t["mask"][i]])).astype(np.int32)
        g = float(np.clip((a[0] - 0.35) / 0.45, 0, 1))
        col = (0, int(220 * g), int(230 * (1 - g)))
        cv2.polylines(img, [p], True, col, 3, cv2.LINE_AA)
        x, y, w, h = cv2.boundingRect(p)
        cv2.putText(img, f"{a[0]:.2f}", (x, y + h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    panels.append(titled(img, "7  Alignment: body against outline (IoU; under 0.35 for 5 frames the person is shown as a ghost)"))

    W, H = int(2 * R.HX * SCALE) + 40, int(2 * R.HY * SCALE) + 40
    img = np.full((H, W, 3), 245, np.uint8)
    px = lambda x, y: (int(W / 2 + x * SCALE), int(H / 2 - y * SCALE))
    for L in rink_lines():
        cv2.polylines(img, [np.array([px(x, y) for x, y in L], np.int32)], False, (180, 140, 120), 2, cv2.LINE_AA)
    for t in tracks:
        xy = np.array(t["xy"], float)
        if not np.isfinite(xy[i]).all():
            continue
        tail = [px(*p) for p in xy[max(i - 25, 0) : i + 1] if np.isfinite(p).all()]
        if len(tail) > 1:
            cv2.polylines(img, [np.array(tail, np.int32)], False, team(t), 2, cv2.LINE_AA)
        cv2.circle(img, px(*xy[i]), 9, team(t), -1, cv2.LINE_AA)
        cv2.circle(img, px(*xy[i]), 9, (60, 60, 60), 1, cv2.LINE_AA)
    pk = np.asarray(R.PUCK)[i]
    if np.isfinite(pk).all():
        cv2.circle(img, px(*pk), 6, (20, 20, 20), -1, cv2.LINE_AA)
    panels.append(titled(img, "8  Rink positions from above: everyone's place and last second's path, and the puck"))

    for n, p in enumerate(panels, 1):
        cv2.imwrite(f"{out}/panel_{n}.jpg", p, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(out, len(panels), "panels")


if __name__ == "__main__":
    main()
