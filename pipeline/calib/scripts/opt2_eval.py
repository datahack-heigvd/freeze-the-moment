"""Compare option 1 (interpolation) and option 2 (snapshot-anchored vision tracker) by hiding a time window around a snapshot."""

import sys

sys.path.insert(0, "scripts")
import numpy as np
import selfcal as S
from opt1 import kalman, linear, load_snaps
from opt2 import Detections, team_of
from scipy.optimize import linear_sum_assignment

BIAS = np.array([-0.14, -0.38])
Q, R_SNAP, R_DET = 6.0, 0.1, 0.6
FPS_USE = 12.5


def kf_predict(x, P, dt):
    F = np.array([[1, dt], [0, 1]])
    Qm = Q**2 * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
    return x @ F.T, np.einsum("ab,kbc,dc->kad", F, P, F) + Qm


def kf_update(x, P, z, r, rows):
    """Update only `rows`. x: (K,2,2) axis-major [axis][pos,vel]; here x[k, axis, :]"""
    for k, zk in zip(rows, z):
        S_ = P[k, 0, 0] + r**2
        K = P[k, :, 0] / S_
        for ax in range(2):
            x[k, ax] = x[k, ax] + K * (zk[ax] - x[k, ax, 0])
        P[k] = P[k] - np.outer(K, P[k, 0])
    return x, P


def init_state(snaps, i0, players, direction):
    """Initialise from the last known snapshot(s) before (direction=+1) or after (-1) the hidden window."""
    K = len(players)
    x = np.zeros((K, 2, 2))
    P = np.tile(np.diag([R_SNAP**2, 4.0]), (K, 1, 1)).astype(float)
    ok = np.zeros(K, bool)
    for k, p in enumerate(players):
        pts = []
        j = i0
        while 0 <= j < len(snaps) and len(pts) < 2 and abs(snaps[j][0] - snaps[i0][0]) < 8:
            if p in snaps[j][1]:
                pts.append((snaps[j][0], snaps[j][1][p]))
            j -= direction
        if not pts:
            continue
        t0, p0 = pts[0]
        v = np.zeros(2)
        if len(pts) == 2 and abs(pts[0][0] - pts[1][0]) > 0.05:
            v = (pts[0][1] - pts[1][1]) / (pts[0][0] - pts[1][0])
        for ax in range(2):
            x[k, ax] = [p0[ax], v[ax]]
        ok[k] = True
    return x, P, ok


def run_pass(snaps, DT, players, teams, i_start, t_start, t_end, direction, use_vision=True):
    """Filter from the snapshot at index i_start (time t_start) towards t_end through the hidden window; returns state at t_end."""
    x, P, ok = init_state(snaps, i_start, players, direction)
    t = t_start
    step = 1.0 / FPS_USE
    n = int(abs(t_end - t_start) / step)
    for _s in range(n):
        tn = t + direction * step
        dt = step
        x, P = kf_predict(x, P, dt) if True else (x, P)
        t = tn
        if not use_vision:
            continue
        det = DT.at_frame(int(round(t * 25)))
        if len(det) == 0:
            continue
        tm = team_of(det)
        pos = det[:, :2] - BIAS
        kk = DT.rowN.get(int(round(t * 25)))
        if kk is None:
            continue
        px = S.project(DT.g, DT.Z["f"][kk], DT.Z["rv"][kk], np.c_[x[:, :, 0], np.zeros(len(x))])
        inview = (px[:, 0] > 40) & (px[:, 0] < 1880) & (px[:, 1] > 40) & (px[:, 1] < 1040)
        rows = np.flatnonzero(ok & inview)
        if len(rows) == 0:
            continue
        pred = x[rows, :, 0]
        var = P[rows, 0, 0] + R_DET**2
        d2 = ((pred[:, None, :] - pos[None]) ** 2).sum(2) / var[:, None]
        cost = d2.copy()
        for a, k in enumerate(rows):
            bad = (tm >= 0) & (tm != teams[k])
            cost[a, bad] = 1e6
            cost[a, tm < 0] += 2.0
        cost[d2 > 9.0] = 1e6
        r_idx, c_idx = linear_sum_assignment(cost)
        good = [(rows[a], c) for a, c in zip(r_idx, c_idx) if cost[a, c] < 1e5]
        if good:
            x, P = kf_update(x, P, [pos[c] for _, c in good], R_DET, [k for k, _ in good])
    # final predict to exactly t_end
    rem = abs(t_end - t)
    if rem > 1e-6:
        x, P = kf_predict(x, P, rem)
    return x, P, ok


def fuse(xf, Pf, okf, xb, Pb, okb):
    out = np.full((len(xf), 2), np.nan)
    for k in range(len(xf)):
        if okf[k] and okb[k]:
            for ax in range(2):
                a = np.linalg.inv(Pf[k])
                b = np.linalg.inv(Pb[k])
                Pc = np.linalg.inv(a + b)
                xc = Pc @ (a @ xf[k, ax] + b @ xb[k, ax])
                out[k, ax] = xc[0]
    return out


def evaluate(W_list=(2, 4, 6, 8), t_lo=930.0, t_hi=1210.0):
    g, snaps = load_snaps()
    DT = Detections()
    team = {i: (0 if r["team"] == "Home" else 1) for i, r in enumerate(g["roster"])}
    centres = [i for i, (t, d) in enumerate(snaps) if t_lo < t < t_hi and len(d) >= 11]
    res = {W: [] for W in W_list}
    for W in W_list:
        for i in centres:
            tc, dc = snaps[i]
            a = tc - W / 2
            b = tc + W / 2
            # indices of last snapshot before the window and first after
            ia = max(j for j in range(i) if snaps[j][0] <= a) if any(snaps[j][0] <= a for j in range(i)) else None
            ib = (
                min(j for j in range(i + 1, len(snaps)) if snaps[j][0] >= b)
                if any(snaps[j][0] >= b for j in range(i + 1, len(snaps)))
                else None
            )
            if ia is None or ib is None or snaps[ib][0] - snaps[ia][0] > W + 6:
                continue
            players = [p for p in dc if p in snaps[ia][1] and p in snaps[ib][1]]
            if not players:
                continue
            teams = [team[p] for p in players]
            # visibility of each player at the centre frame
            N = int(round(tc * 25))
            k = DT.rowN.get(N)
            if k is None:
                continue
            vis = []
            for p in players:
                px = S.project(DT.g, DT.Z["f"][k], DT.Z["rv"][k], np.array([[dc[p][0], dc[p][1], 0.0]]))[0]
                vis.append(60 < px[0] < 1860 and 60 < px[1] < 1020)
            # option 1: kalman using snapshots outside window only
            e1, e1l = [], []
            for p in players:
                ts, xs = [], []
                for j in range(max(ia - 3, 0), min(ib + 4, len(snaps))):
                    tj, dj = snaps[j]
                    if (tj <= a or tj >= b) and p in dj:
                        ts.append(tj)
                        xs.append(dj[p])
                e1.append(np.linalg.norm(kalman(np.array(ts), np.array(xs), tc, q=Q, r=R_SNAP) - dc[p]))
                e1l.append(np.linalg.norm(linear(np.array(ts), np.array(xs), tc) - dc[p]))
            # option 2: two-filter vision tracker
            xf, Pf, okf = run_pass(snaps, DT, players, teams, ia, snaps[ia][0], tc, +1)
            xb, Pb, okb = run_pass(snaps, DT, players, teams, ib, snaps[ib][0], tc, -1)
            est = fuse(xf, Pf, okf, xb, Pb, okb)
            e2 = [np.linalg.norm(est[kk] - dc[p]) if np.isfinite(est[kk]).all() else np.nan for kk, p in enumerate(players)]
            for kk, p in enumerate(players):
                res[W].append((vis[kk], e1l[kk], e1[kk], e2[kk]))
        print(f"W={W}s done: {len(res[W])} player-cases", flush=True)
    return res


def report(res):
    for W, rows in res.items():
        A = np.array(rows, float)
        for name, m in (("visible players", A[:, 0] == 1), ("all players", np.ones(len(A), bool))):
            B = A[m]
            B = B[np.isfinite(B[:, 3])]
            if len(B) == 0:
                continue
            harm = np.mean(B[:, 3] > B[:, 2] + 0.5)
            help_ = np.mean(B[:, 3] < B[:, 2] - 0.5)
            print(
                f"hidden window {W}s | {name:15s} n={len(B):4d} | linear med {np.median(B[:, 1]):.2f} p90 {np.percentile(B[:, 1], 90):.2f} | opt1 kalman med {np.median(B[:, 2]):.2f} p90 {np.percentile(B[:, 2], 90):.2f} | opt2 vision med {np.median(B[:, 3]):.2f} p90 {np.percentile(B[:, 3], 90):.2f} | opt2 better by >0.5m {help_ * 100:.0f}% worse by >0.5m {harm * 100:.0f}%"
            )


if __name__ == "__main__":
    res = evaluate()
    report(res)
    np.save("cache/opt2_eval.npy", {W: np.array(v) for W, v in res.items()}, allow_pickle=True)
