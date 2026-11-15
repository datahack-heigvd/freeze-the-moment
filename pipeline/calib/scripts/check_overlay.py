import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
import camera
from overlay import draw
from players_check import draw_players

from pipeline import VIDEO, G

HB = float(G["hb"])
CAM0 = G["cams"][0]


def frame_at(t):
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(t), "-i", VIDEO, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True,
    ).stdout
    return cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)


def cam_at(C, t):
    i = int(np.argmin(np.abs(C["t"] - t)))
    c = CAM0.copy()
    c[0] = C["f"][i]
    c[1:4] = C["rv"][i]
    if "lens" in C.files and len(C["lens"]) == 4:
        c[8] = C["lens"][2] - 960
        c[9] = C["lens"][3] - 540
    if "C" in C.files:
        c[4:7] = C["C"]
        c[7] = float(C["k1"])
    return c


def montage(name, times, out, players=True):
    C = np.load(f"cache/cam_{name}.npz")
    tiles = []
    if "lens" in C.files:
        camera.LENS = tuple(C["lens"][:2])
    for t in times:
        img = frame_at(t)
        cam = cam_at(C, t)
        o = draw(img, None, cam=cam, w=3)
        if players:
            o = draw_players(o, cam, t)
        cv2.putText(o, f"t={t:.1f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        tiles.append(cv2.resize(o, (960, 540)))
    rows = [np.hstack(tiles[k : k + 2]) for k in range(0, len(tiles), 2)]
    cv2.imwrite(out, np.vstack(rows))


if __name__ == "__main__":
    name = sys.argv[1]
    times = [float(x) for x in sys.argv[2].split(",")]
    montage(name, times, f"cache/ov_{name}.jpg")
