"""Map each 4 fps panorama node to its exact 25 fps source frame index by pixel-exact matching."""

import hashlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, "scripts")
from pipeline import VIDEO


def stream(start, dur, vf):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        str(start),
        "-t",
        str(dur),
        "-i",
        VIDEO,
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    n = 960 * 540 * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield hashlib.md5(b).hexdigest()
    p.wait()


if __name__ == "__main__":
    start, dur = float(sys.argv[1]), float(sys.argv[2])
    full = list(stream(start, dur, "scale=960:540"))
    pos = {h: i for i, h in enumerate(full)}
    nodes = list(stream(start, dur, "fps=4,scale=960:540"))
    idx = np.array([pos.get(h, -1) for h in nodes])
    print("full frames", len(full), "nodes", len(nodes), "matched", (idx >= 0).sum())
    d = np.diff(idx[idx >= 0])
    print("spacing counts", dict(zip(*np.unique(d, return_counts=True))))
    print("first idx", idx[:8], "offset of node k vs nominal k*6.25:", (idx[:8] - np.arange(8) * 6.25).round(2))
    np.save("cache/node_index_pano300.npy", idx)
