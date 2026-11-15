"""Option 2: person detections (GPU) projected to the ice with the per-frame camera, matched to snapshot players."""

import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
import selfcal as S
from opt1 import load_snaps
from window import W as WIN

F_REF = S.F_REF
NFR = WIN["n"]
FRAME0 = WIN["frame0"]  # source frame number of extracted image / detection index 0 (verified against the source video)
HEAD_H = 1.8  # head height used to locate players whose skates are hidden by the near boards
OCC_TOL = 0.12  # box bottom more than this fraction of the box height above where the head implies it should be
NEAR_BAND = 3.0  # metres from the near boards where occlusion by the boards happens
RINK_MARGIN = 0.5  # a foot point may be this far outside the boards (detection and calibration error) before the person counts as not on the ice
HEIGHT_RANGE = (0.5, 1.45)  # accepted box height as a fraction of the pixel height of a HEAD_H person standing at that spot


def backproject(g, f, rv, px, h=0.0):
    """Image pixels (N,2) at full resolution -> (x, y) on the horizontal plane z=h (ice by default) using the self-calibrated model."""
    d = dict(zip(S.GNAMES, g))
    fz = d["s"] * f
    R = cv2.Rodrigues(rv)[0] @ cv2.Rodrigues(np.array([d["rg0"], d["rg1"], d["rg2"]]))[0]
    C = S.C_BASE + np.array([d["dC0"], d["dC1"], d["dC2"]]) + np.array([d["dCf0"], d["dCf1"], d["dCf2"]]) * (fz / F_REF - 1)
    k = d["kb"] * (F_REF / fz) ** d["pw"]
    dd = px - [960 + d["cxo"], 540 + d["cyo"]]
    u = dd.copy()
    for _ in range(8):
        u = dd / (1 + k * (u**2).sum(1, keepdims=True) / F_REF**2)
    dirc = np.c_[u / fz, np.ones(len(u))]
    dirw = dirc @ R  # R maps world->camera, so R^T applied to row vectors is dirc @ R
    s = (h - C[2]) / dirw[:, 2]
    return (C[:2] + dirw[:, :2] * s[:, None]), s


class Detections:
    def __init__(self, path=WIN["det"], loc=WIN["loc"], g=None):
        self.D = np.load(path)
        self.Z = np.load(f"cache/cam_{loc}.npz")
        self.N0 = int(self.Z["N"][0])
        self.F0 = FRAME0
        self.g = g if g is not None else np.load("cache/selfcal_v1.npz")["V3"]
        order = np.argsort(self.D[:, 0], kind="stable")
        self.D = self.D[order]
        self.start = np.searchsorted(self.D[:, 0], np.arange(0, NFR + 1))
        self.rowN = {int(n): i for i, n in enumerate(self.Z["N"])}
        self.cache = {}
        self.stat = dict(total=0, off_rink=0, bad_height=0)

    def evaluate(self, N):
        """Every raw detection at frame N with its rink position and gating flags: (D, xy, occ, on_ice, fits, box height / expected height of a 1.8 m person at that spot) or None."""
        k = N - self.F0
        if k < 0 or k >= NFR or N not in self.rowN:
            return None
        a, b = self.start[k], self.start[k + 1]
        D = self.D[a:b]
        if len(D) == 0:
            return None
        i = self.rowN[N]
        f, rv = self.Z["f"][i], self.Z["rv"][i]
        foot = np.c_[(D[:, 1] + D[:, 3]) / 2, D[:, 4]]
        xy, s = backproject(self.g, f, rv, foot)
        d = dict(zip(S.GNAMES, self.g))
        # skates hidden by the near boards: the box bottom sits above where the head says the feet are, so trust the head
        xyh, _ = backproject(self.g, f, rv, np.c_[(D[:, 1] + D[:, 3]) / 2, D[:, 2]], h=HEAD_H)
        foot_px = S.project(self.g, f, rv, np.c_[xyh, np.zeros(len(xyh))])
        occ = (D[:, 4] < foot_px[:, 1] - OCC_TOL * (D[:, 4] - D[:, 2])) & (xy[:, 1] < -d["hy"] + NEAR_BAND) & (s > 0)
        xy = np.where(occ[:, None], xyh, xy)
        c = d["corner"]
        dx = np.maximum(np.abs(xy[:, 0]) - (d["hx"] - c), 0)
        dy = np.maximum(np.abs(xy[:, 1]) - (d["hy"] - c), 0)
        on_ice = (
            (s > 0)
            & (dx**2 + dy**2 <= (c + RINK_MARGIN) ** 2)
            & (np.abs(xy[:, 0]) < d["hx"] + RINK_MARGIN)
            & (np.abs(xy[:, 1]) < d["hy"] + RINK_MARGIN)
        )
        px0 = S.project(self.g, f, rv, np.c_[xy, np.zeros(len(xy))])
        px1 = S.project(self.g, f, rv, np.c_[xy, np.full(len(xy), HEAD_H)])
        ratio = (D[:, 4] - D[:, 2]) / np.maximum(np.abs(px0[:, 1] - px1[:, 1]), 1.0)
        fits = (ratio > HEIGHT_RANGE[0]) & (ratio < HEIGHT_RANGE[1])
        return D, xy, occ, on_ice, fits, ratio

    def at_frame(self, N):
        """Detections at absolute frame N: array rows [x, y, conf, yellow, blue, dark, white, head_based, sat, val] in rink metres.
        Rejected: feet outside the (rounded) rink, or a box whose height does not fit a person standing at that spot (crowd, bench, glass reflections)."""
        if N in self.cache:
            return self.cache[N]
        r = self.evaluate(N)
        if r is None:
            return np.zeros((0, 10))
        D, xy, occ, on_ice, fits, _ = r
        ok = on_ice & fits
        self.stat["total"] += len(D)
        self.stat["off_rink"] += int((~on_ice).sum())
        self.stat["bad_height"] += int((on_ice & ~fits).sum())
        out = np.c_[xy, D[:, 5], D[:, 6], D[:, 7], D[:, 8], D[:, 9], occ, D[:, 10], D[:, 11]][ok]
        self.cache[N] = out
        return out


def team_of(det):
    """0 = Home (blue), 1 = Away (yellow), -1 = unknown/referee."""
    y, b = det[:, 3], det[:, 4]
    t = np.full(len(det), -1)
    t[(y > 0.22) & (y > b)] = 1
    t[(b > 0.22) & (b > y)] = 0
    return t


if __name__ == "__main__":
    g, snaps = load_snaps()
    DT = Detections()
    roster = g["roster"]
    team = {i: (0 if r["team"] == "Home" else 1) for i, r in enumerate(roster)}
    lo, hi = 925.0, 1215.0
    idx = [i for i, (t, d) in enumerate(snaps) if lo < t < hi and len(d) >= 11]
    print("snapshots in window:", len(idx))
    offs = np.arange(-0.6, 0.61, 0.04)
    med = []
    for dlt in offs:
        errs = []
        for i in idx[::3]:
            t, d = snaps[i]
            N = int(round((t + dlt) * 25))
            det = DT.at_frame(N)
            if len(det) == 0:
                continue
            for _p, xy in d.items():
                errs.append(np.min(np.linalg.norm(det[:, :2] - xy, axis=1)))
        med.append((np.median(errs), np.mean(np.array(errs) < 1.0)))
    for dlt, (m, f) in zip(offs, med):
        print(f"offset {dlt:+.2f}s  median nearest-detection distance {m:.2f} m   within 1 m: {f * 100:.0f}%")
