"""Appearance data for the gap benchmark: every on-ice person detection of the window, with identity labels taken from the metadata.
  lab_full    roster index from the full-metadata run (training labels for the OTHER window, and a rough referee in this one)
  lab_vis     roster index from the masked run, only on frames of visible blocks (the labels a method may train on in this window)
usage (WIN=300|2160):  appear.py index     -> cache/bench/<win>/dets.npz"""

import pickle
import sys

import numpy as np

sys.path.insert(0, "scripts")
import idbench as B
from opt2 import Detections
from window import W as WIN

MATCH = 0.5  # m between a tracklet's smoothed position and the detection it was built from


def labels_from(R, DT, rows, Ns, xy):
    """Per detection row: roster index of the identified tracklet sitting on it (or -1)."""
    lab = np.full(len(rows), -1)
    byN = {}
    for i, N in enumerate(Ns):
        byN.setdefault(int(N), []).append(i)
    for ti, tr in enumerate(R["tracks"]):
        p = int(R["ident"][ti])
        if p < 0:
            continue
        for j, N in enumerate(range(tr["a"], tr["b"] + 1)):
            if not tr["meas"][j] or N not in byN:
                continue
            c = byN[N]
            d = np.linalg.norm(xy[c] - tr["pos"][j], axis=1)
            k = int(np.argmin(d))
            if d[k] < MATCH:
                lab[c[k]] = p
    return lab


def index():
    DT = Detections()
    rows, Ns, xy, occ = [], [], [], []
    for N in range(DT.N0, DT.N0 + WIN["n"]):
        r = DT.evaluate(N)
        if r is None:
            continue
        D, p, o, on, fits, _ = r
        k = np.flatnonzero(on & fits)
        a = DT.start[N - DT.F0]
        rows += list(a + k)
        Ns += [N] * len(k)
        xy += list(p[k])
        occ += list(o[k])
    rows, Ns, xy, occ = np.array(rows), np.array(Ns), np.array(xy), np.array(occ)
    full = labels_from(pickle.load(open(WIN["gtrack"], "rb")), DT, rows, Ns, xy)
    vis = labels_from(pickle.load(open(f"{B.OUT}/gtrack.pkl", "rb")), DT, rows, Ns, xy)
    vis[np.array([B.hidden_at(N / B.FPS) for N in Ns])] = -1
    box = DT.D[rows, 1:5]
    img = Ns - DT.F0
    np.savez(
        f"{B.OUT}/dets.npz", row=rows, N=Ns, img=img, box=box, xy=xy, occ=occ, conf=DT.D[rows, 5], lab_full=full, lab_vis=vis
    )
    print(
        f"{len(rows)} detections, labelled: full {(full >= 0).mean() * 100:.0f}%, visible-only {(vis >= 0).mean() * 100:.0f}%, players {len(set(full[full >= 0]))}"
    )


if __name__ == "__main__":
    {"index": index}[sys.argv[1]]()
