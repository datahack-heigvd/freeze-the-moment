import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
from camera import project_ice

d = json.load(open("../../store/site/game.json"))
frames = d["frames"]
roster = d["roster"]
ts = np.array([f[0] for f in frames])


def players_at(t):
    i = int(np.searchsorted(ts, t)) - 1
    f, n = frames[i], frames[min(i + 1, len(frames) - 1)]
    k = (t - f[0]) / (n[0] - f[0]) if n[0] - f[0] <= 3 else 0
    nxt = {n[j]: (n[j + 1], n[j + 2]) for j in range(3, len(n), 3)}
    out = []
    for j in range(3, len(f), 3):
        idx = f[j]
        x, y = f[j + 1], f[j + 2]
        if idx in nxt and k > 0:
            x += (nxt[idx][0] - x) * k
            y += (nxt[idx][1] - y) * k
        out.append((idx, x, y))
    return out


def draw_players(img, cam, t, color_scale=1.0):
    out = img.copy()
    for idx, x, y in players_at(t):
        q = project_ice(cam, [(x, y)])[0] * color_scale
        r = roster[idx]
        col = (0, 200, 255) if r["team"] == "Away" else (255, 120, 40)
        cv2.circle(out, tuple(int(v) for v in q), 9, col, 3, cv2.LINE_AA)
        cv2.putText(out, r["jersey"], (int(q[0]) + 10, int(q[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return out


if __name__ == "__main__":
    G = np.load("cache/calib_global.npz")
    P = np.load("cache/poses_seg0.npz")
    t, idx, k = 1100.1, 359, 1
    img = cv2.imread("cache/full_1100.1.png")
    cam_l = G["cams"][k]
    cam_c = cam_l.copy()
    cam_c[0] = P["f"][idx]
    cam_c[1:4] = P["rv"][idx]
    a = draw_players(img, cam_l, t)
    b = draw_players(img, cam_c, t)
    cv2.putText(a, "landmark pose", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(b, "chain pose", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.imwrite("cache/players_cmp.jpg", np.vstack([cv2.resize(a, (1440, 810)), cv2.resize(b, (1440, 810))]))
