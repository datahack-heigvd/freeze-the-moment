"""A goal's reel: the whole clip of the broadcast undistorted into one pinhole camera per frame (how_export.py's resampling, one
scale for the clip), so a three.js render of the scene from that camera lines up with the picture. The submission film
(app/scripts/build-film.ts, docs/video) renders its tv take from these cameras and cuts between the footage and the 3D with nothing
moving.

  <out>/broadcast.mp4   the clip's frames, undistorted, SRC_W × SRC_H, H.264 at the clip's fps with a keyframe every second (cuts start
                        mid-clip), no sound
  <out>/reel.json       fps, width, height, and per frame i (the scene's exported frame i) the camera {fx (at width), R (rows,
                        rink to OpenCV camera axes), C}, as scene.json's shots
  <out>/masks.bin       everyone's outline per frame in broadcast.mp4's pixels, the live player's cutout format (app/public/player/
                        cutout.ts): u32 frames n, u32 offsets[n + 1] into the data (int16 units), then per frame its polygons as
                        int16 [points, x0, y0, x1, y1...] in eighths of a pixel; graphics on the ice are erased inside them, so they
                        lie under the players

usage (LTX venv, CPU, in a goal dir):  CUDA_VISIBLE_DEVICES= BUNDLE=bundle reel_export.py <out>"""

import json
import os
import subprocess
import sys

import cv2
import numpy as np
from export.how_export import SRC_H, SRC_W, distort, least_scale, undistort
from goal import render as R

CRF = 22


def main():
    out = sys.argv[1]
    S, Z = R.S, R.Z
    n = S["n"]
    os.makedirs(out, exist_ok=True)
    s = max(least_scale(float(Z["cam"][i][1])) for i in range(n))
    gx, gy = np.meshgrid(np.arange(SRC_W) + 0.5, np.arange(SRC_H) + 0.5)
    o = np.stack([gx, gy], -1) - (SRC_W / 2, SRC_H / 2)
    enc = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{SRC_W}x{SRC_H}",
            "-r",
            str(S["fps"]),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(CRF),
            "-g",
            str(S["fps"]),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            f"{out}/broadcast.mp4",
        ],
        stdin=subprocess.PIPE,
    )
    cams = []
    for i in range(n):
        cam = Z["cam"][i]
        d = distort(o / s, float(cam[1])).astype(np.float32)
        enc.stdin.write(cv2.remap(cv2.imread(f"{R.B}/frames/{i + 1:03d}.jpg"), d[..., 0], d[..., 1], cv2.INTER_LINEAR).tobytes())
        cams.append(
            dict(
                fx=round(float(cam[0]) * s, 3),
                R=np.round(cam[2:11].reshape(3, 3), 6).tolist(),
                C=np.round(cam[11:14], 4).tolist(),
            )
        )
    enc.stdin.close()
    enc.wait()
    offsets, data = [0], []
    for i in range(n):
        k = float(Z["cam"][i][1])
        for poly in S["outlines"][i]:
            q = undistort(np.asarray(poly, np.float64).reshape(-1, 2), k) * s + (SRC_W / 2, SRC_H / 2)
            q = cv2.approxPolyDP(q.astype(np.float32).reshape(-1, 1, 2), 0.8, True).reshape(-1, 2)
            if len(q) >= 3:
                data += [len(q), *np.round(np.clip(q, -512, [SRC_W + 512, SRC_H + 512]) * 8).astype(int).flatten().tolist()]
        offsets.append(len(data))
    with open(f"{out}/masks.bin", "wb") as fh:
        fh.write(np.array([n, *offsets], np.uint32).tobytes())
        fh.write(np.array(data, np.int16).tobytes())
    json.dump(dict(fps=S["fps"], width=SRC_W, height=SRC_H, cams=cams), open(f"{out}/reel.json", "w"), separators=(",", ":"))
    print(out, n, "frames, scale", round(s, 3))


if __name__ == "__main__":
    main()
