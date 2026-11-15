"""Per-source-frame localization against bundle-adjusted nodes (pure rotation about a fixed point).
Every 25 fps frame gets (f, rotvec) from feature matches to its two neighbouring nodes, with the panorama lens model fixed.
Poses are therefore tied to image content, not to timestamps.

usage: localize.py <name> <pano_name> <start_s> <dur_s> <workers>   (needs cache/node_index_<pano>.npy from node_index.py)"""

import subprocess
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "scripts")
import pipeline as P

FPS = 25.0
_sift = None


def _init():
    global _sift
    cv2.setNumThreads(1)
    _sift = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.02)
    P._init_worker()


def abs_first_frame(start):
    return int(np.ceil(start * FPS - 1e-6))


def read_range(N0, n):
    ss = (N0 - 0.5) / FPS
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{ss:.4f}",
        "-i",
        P.VIDEO,
        "-frames:v",
        str(n),
        "-vf",
        "scale=960:540",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    sz = 960 * 540 * 3
    while True:
        b = p.stdout.read(sz)
        if len(b) < sz:
            break
        yield np.frombuffer(b, np.uint8).reshape(540, 960, 3)
    p.wait()


def feats(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = _sift.detectAndCompute(g, None)
    if des is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.array([k.pt for k in kp], np.float32) * 2.0, des


def undist(d, f, lens):
    km, pw, cx, cy = lens
    d = d - [cx, cy]
    u = d.copy()
    k = P.K1 * km * (P.F_REF / f) ** pw
    for _ in range(6):
        u = d / (1 + k * (u**2).sum(1, keepdims=True) / P.F_REF**2)
    return u


def match(da, db, pa, pb):
    fl = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))
    if len(da) < 15 or len(db) < 15:
        return None
    m = [x for x in fl.knnMatch(da, db, k=2) if len(x) == 2 and x[0].distance < 0.8 * x[1].distance]
    if len(m) < 20:
        return None
    a = pa[[x[0].queryIdx for x in m]]
    b = pb[[x[0].trainIdx for x in m]]
    H, inl = cv2.findHomography(a, b, cv2.RANSAC, 4.0, maxIters=2000)
    if H is None or inl.sum() < 20:
        return None
    k = inl.ravel().astype(bool)
    return a[k], b[k]


def solve(x0, constraints, lens):
    """constraints: list of (pts_node_raw, f_node, R_node, pts_frame_raw). Solve frame (f, rotvec)."""
    prep = [(undist(a, fa, lens), fa, Ra, b) for a, fa, Ra, b in constraints]

    def res(x):
        f = x[0]
        R = cv2.Rodrigues(x[1:4])[0]
        out = []
        for au, fa, Ra, b in prep:
            bu = undist(b, f, lens)
            d = np.c_[au / fa, np.ones(len(au))]
            p = d @ (R @ Ra.T).T
            out.append((f * p[:, :2] / p[:, 2:3] - bu).ravel())
        return np.concatenate(out)

    sol = least_squares(res, x0, x_scale=[50, 0.005, 0.005, 0.005], loss="soft_l1", f_scale=1.5, max_nfev=30)
    e = np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)
    return sol.x, int((e < 3).sum()), float(np.median(e))


def work(args):
    j0, j1, nodeN, nodeX, lens, N0 = args  # absolute source frame numbers [j0, j1], node arrays (absolute N, [f, rv...])
    imgs = list(read_range(j0, j1 - j0 + 1))
    F = [feats(im) for im in imgs]
    out = {}
    node_at = {int(n): i for i, n in enumerate(nodeN)}
    for j in range(j0, j1 + 1):
        if j in node_at:
            x = nodeX[node_at[j]]
            out[j] = (x[0], x[1:4].copy(), 999, 0.0)
            continue
        k = int(np.searchsorted(nodeN, j))
        a, b = max(k - 1, 0), min(k, len(nodeN) - 1)
        w = (j - nodeN[a]) / max(nodeN[b] - nodeN[a], 1)
        x0 = (1 - w) * nodeX[a] + w * nodeX[b]
        cons = []
        for ni in {a, b}:
            nj = int(nodeN[ni])
            if not (j0 <= nj <= j1):
                continue
            pa, da = F[nj - j0]
            pb, db = F[j - j0]
            mt = match(da, db, pa, pb)
            if mt is None:
                continue
            cons.append((mt[0], nodeX[ni][0], cv2.Rodrigues(nodeX[ni][1:4])[0], mt[1]))
        if not cons:
            out[j] = (x0[0], x0[1:4], 0, 99.0)
            continue
        x, ninl, med = solve(x0, cons, lens)
        out[j] = (x[0], x[1:4], ninl, med)
    return out


def main(name, pano, start, dur, workers):
    t0 = time.time()
    Z = np.load(f"cache/cam_{pano}.npz")
    idx = np.load(f"cache/node_index_{pano}.npy")
    lens = Z["lens"]
    Nf = abs_first_frame(start)
    nodeN = Nf + idx
    nodeX = np.column_stack([Z["f"], Z["rv"]])
    last = Nf + int(dur * FPS) - 1
    # chunks start/end on node frames so every frame has both neighbours inside its chunk
    bounds = nodeN[::12]
    chunks = []
    for i in range(len(bounds)):
        j0 = int(bounds[i])
        j1 = int(bounds[i + 1]) if i + 1 < len(bounds) else last
        chunks.append((j0, j1, nodeN, nodeX, lens, Nf))
    chunks = [c for c in chunks if c[1] > c[0]]
    res = {}
    with Pool(workers, initializer=_init) as pool:
        for k, r in enumerate(pool.imap_unordered(work, chunks)):
            res.update(r)
            if k % 5 == 0:
                print(f"[{name}] chunk {k + 1}/{len(chunks)} {time.time() - t0:.0f}s", flush=True)
    N = np.array(sorted(res))
    f = np.array([res[n][0] for n in N])
    rv = np.array([res[n][1] for n in N])
    ninl = np.array([res[n][2] for n in N])
    med = np.array([res[n][3] for n in N])
    np.savez(f"cache/cam_{name}.npz", t=N / FPS, N=N, f=f, rv=rv, rms=med, inl=ninl, lens=lens)
    print(
        f"[{name}] {len(N)} frames, median inliers {np.median(ninl):.0f}, frames with <20 inliers {np.mean(ninl < 20) * 100:.1f}%, median fit {np.median(med):.2f}px, {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]))
