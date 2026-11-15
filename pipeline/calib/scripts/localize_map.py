"""Per-frame localization of any stretch of the game against the panorama nodes of an already solved window (the map).
A frame is matched to the few nodes whose views overlap its predicted view; (f, rotvec) come out in the map's frame and lens model,
so the shared self-calibration (selfcal V3) applies unchanged. Each chunk starts with a retrieval over a subsample of nodes.

usage: localize_map.py <name> <map_pano> <map_start_s> <start_s> <dur_s> <workers>   (needs cache/node_index_<map_pano>.npy)
output: cache/cam_<name>.npz  (t, N, f, rv, rms, inl, lens)"""

import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, "scripts")
import localize as L

import pipeline as P

FPS = 25.0
CHUNK = 250  # frames per independent chunk (each starts with a retrieval)
TOP = 4  # nodes matched per frame
MIN_INL = 20
RETRIEVE_STEP = 5  # nodes tried at retrieval
MAP = {}


def map_features(pano, map_start):
    """SIFT features of every map node (the 4 fps stream is exactly the node sequence), cached as arrays."""
    base = f"cache/mapfeat_{pano}"
    if os.path.exists(base + "_des.npy"):
        return
    L._init()
    pts, des, off = [], [], [0]
    for _k, img in enumerate(P.read_frames(map_start, 300, 4)):
        p, d = L.feats(img)
        pts.append(p)
        des.append(d.astype(np.uint8))
        off.append(off[-1] + len(p))
    np.save(base + "_pts.npy", np.concatenate(pts).astype(np.float32))
    np.save(base + "_des.npy", np.concatenate(des))
    np.save(base + "_off.npy", np.array(off))
    print(f"map features: {len(pts)} nodes, {off[-1]} keypoints", flush=True)


def _init(pano):
    L._init()
    base = f"cache/mapfeat_{pano}"
    MAP["pts"] = np.load(base + "_pts.npy", mmap_mode="r")
    MAP["des"] = np.load(base + "_des.npy", mmap_mode="r")
    MAP["off"] = np.load(base + "_off.npy")
    Z = np.load(f"cache/cam_{pano}.npz")
    MAP["X"] = np.column_stack([Z["f"], Z["rv"]])
    MAP["lens"] = Z["lens"]
    MAP["grid"] = np.array([P.proj(P.H_of(x[0], x[1:4]), P.GRID) for x in MAP["X"]])
    ok = lambda q: (q[..., 0] > 0) & (q[..., 0] < 1920) & (q[..., 1] > 0) & (q[..., 1] < 1080)
    MAP["ok"] = ok(MAP["grid"])
    MAP["okf"] = ok


def node(k):
    a, b = MAP["off"][k], MAP["off"][k + 1]
    return np.asarray(MAP["pts"][a:b]), np.asarray(MAP["des"][a:b]).astype(np.float32)


def match_node(k, fp, fd):
    pa, da = node(k)
    mt = L.match(da, fd, pa, fp)
    return (mt, k) if mt is not None else (None, k)


def choose(x0):
    """Nodes whose views overlap the predicted view most (spread out in time)."""
    A = P.proj(P.H_of(x0[0], x0[1:4]), P.GRID)
    okA = MAP["okf"](A)
    both = MAP["ok"] & okA[None]
    n = both.sum(1)
    disp = np.where(both, np.linalg.norm(MAP["grid"] - A[None], axis=2), 0).sum(1) / np.maximum(n, 1)
    score = n - 0.02 * disp
    out = []
    for k in np.argsort(-score):
        if n[k] < 12:
            break
        if all(abs(k - c) > 8 for c in out):
            out.append(int(k))
        if len(out) >= TOP:
            break
    return out


def retrieve(fp, fd):
    best = []
    for k in range(0, len(MAP["X"]), RETRIEVE_STEP):
        mt, _ = match_node(k, fp, fd)
        if mt is not None:
            best.append((len(mt[0]), k, mt))
    best.sort(key=lambda r: -r[0])
    return best[:2]


def work(args):
    j0, n = args
    out = {}
    prev = None
    miss = 0
    frames = list(L.read_range(j0, n))
    F = [L.feats(im) for im in frames]
    for i, (fp, fd) in enumerate(F):
        N = j0 + i
        cons = []
        if prev is None or miss >= 3:
            best = retrieve(fp, fd)
            if not best:
                out[N] = (np.nan, np.full(3, np.nan), 0, 99.0)
                continue
            cons = [(mt[0], MAP["X"][k][0], cv2.Rodrigues(MAP["X"][k][1:4])[0], mt[1]) for _, k, mt in best]
            x0 = MAP["X"][best[0][1]].copy()
            miss = 0
        else:
            x0 = prev.copy()
        for k in choose(x0):
            mt, _ = match_node(k, fp, fd)
            if mt is not None:
                cons.append((mt[0], MAP["X"][k][0], cv2.Rodrigues(MAP["X"][k][1:4])[0], mt[1]))
        if not cons:
            out[N] = (prev[0], prev[1:4], 0, 99.0) if prev is not None else (np.nan, np.full(3, np.nan), 0, 99.0)
            miss += 1
            continue
        x, ninl, med = L.solve(x0, cons, MAP["lens"])
        if ninl < MIN_INL:
            miss += 1
        else:
            miss = 0
            prev = x
        out[N] = (x[0], x[1:4], ninl, med)
    return out


def main(name, pano, map_start, start, dur, workers):
    t0 = time.time()
    map_features(pano, map_start)
    Nf = L.abs_first_frame(start)
    last = Nf + int(dur * FPS)
    chunks = [(j, min(CHUNK, last - j)) for j in range(Nf, last, CHUNK)]
    res = {}
    with Pool(workers, initializer=_init, initargs=(pano,)) as pool:
        for k, r in enumerate(pool.imap_unordered(work, chunks)):
            res.update(r)
            if k % 3 == 0:
                print(f"[{name}] chunk {k + 1}/{len(chunks)} {time.time() - t0:.0f}s", flush=True)
    N = np.array(sorted(res))
    f = np.array([res[n][0] for n in N])
    rv = np.array([res[n][1] for n in N])
    ninl = np.array([res[n][2] for n in N])
    med = np.array([res[n][3] for n in N])
    np.savez(
        f"cache/cam_{name}.npz", t=N / FPS, N=N, f=f, rv=rv, rms=med, inl=ninl, lens=np.load(f"cache/cam_{pano}.npz")["lens"]
    )
    print(
        f"[{name}] {len(N)} frames, median inliers {np.median(ninl):.0f}, frames with <{MIN_INL} inliers {np.mean(ninl < MIN_INL) * 100:.1f}%, median fit {np.median(med):.2f}px, {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]))
