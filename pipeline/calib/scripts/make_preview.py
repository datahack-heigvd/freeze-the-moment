"""Render a preview clip: one tracker per player fusing snapshots (precise, give identity) and vision detections (noisy, fill gaps).
usage: make_preview.py <t0> <dur> <out_prefix>"""

import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
import selfcal as S
from opt1 import load_snaps
from opt2 import Detections, team_of
from opt2_eval import BIAS, R_DET, R_SNAP, kf_predict
from scipy.optimize import linear_sum_assignment

FPS = 25.0


def track(snaps, DT, team, t0, t1, use_vision, pad=5.0, ACTIVE_S=4.0):
    """Joint forward Kalman filter (+ RTS smoothing) over 25 fps frames. Returns pos (T,K,2), source (T,K) 0=pred 1=vision 2=snapshot, active (T,K), players."""
    ta, tb = t0 - pad, t1 + pad
    frames = np.arange(int(round(ta * FPS)), int(round(tb * FPS)) + 1)
    T = len(frames)
    times = frames / FPS
    ss = [(t, d) for t, d in snaps if ta - ACTIVE_S <= t <= tb + ACTIVE_S]
    players = sorted({p for _, d in ss for p in d})
    K = len(players)
    pid = {p: k for k, p in enumerate(players)}
    # activity: a player exists at time t if some snapshot containing them is within ACTIVE_S
    snap_t = {p: np.array([t for t, d in ss if p in d]) for p in players}
    active = np.zeros((T, K), bool)
    for p in players:
        for k_t, t in enumerate(times):
            active[k_t, pid[p]] = np.abs(snap_t[p] - t).min() <= ACTIVE_S
    X = np.zeros((T, K, 2, 2))
    Pm = np.zeros((T, K, 2, 2))
    Xp = np.zeros_like(X)
    Pp = np.zeros_like(Pm)
    started = np.zeros(K, bool)
    src = np.zeros((T, K), int)
    last_vis = np.full(K, -9.0)
    last_snap = np.full(K, -9.0)
    x = np.zeros((K, 2, 2))
    P = np.tile(np.diag([R_SNAP**2, 4.0]), (K, 1, 1)).astype(float)
    snap_by_frame = {}
    for t, d in ss:
        snap_by_frame.setdefault(int(round(t * FPS)), []).append(d)
    for fi, N in enumerate(frames):
        if fi > 0:
            x, P = kf_predict(x, P, 1.0 / FPS)
        Xp[fi], Pp[fi] = x, P
        # snapshot updates (start tracks that have not started yet)
        for d in snap_by_frame.get(int(N), []):
            for p, xy in d.items():
                k = pid[p]
                if not started[k]:
                    x[k] = np.array([[xy[0], 0.0], [xy[1], 0.0]])
                    P[k] = np.diag([R_SNAP**2, 4.0])
                    started[k] = True
                else:
                    S_ = P[k, 0, 0] + R_SNAP**2
                    Kg = P[k, :, 0] / S_
                    for ax in range(2):
                        x[k, ax] = x[k, ax] + Kg * (xy[ax] - x[k, ax, 0])
                    P[k] = P[k] - np.outer(Kg, P[k, 0])
                src[fi, k] = 2
                last_snap[k] = times[fi]
        # vision update
        rows_ok = started & active[fi]
        if use_vision and rows_ok.any():
            det = DT.at_frame(int(N))
            kk = DT.rowN.get(int(N))
            if len(det) and kk is not None:
                px = S.project(DT.g, DT.Z["f"][kk], DT.Z["rv"][kk], np.c_[x[:, :, 0], np.zeros(K)])
                inview = (px[:, 0] > 40) & (px[:, 0] < 1880) & (px[:, 1] > 40) & (px[:, 1] < 1040)
                rows = np.flatnonzero(rows_ok & inview)
                if len(rows):
                    tm = team_of(det)
                    pos = det[:, :2] - BIAS
                    var = P[rows, 0, 0] + R_DET**2
                    d2 = ((x[rows, :, 0][:, None, :] - pos[None]) ** 2).sum(2) / var[:, None]
                    cost = d2.copy()
                    for a, k in enumerate(rows):
                        cost[a, (tm >= 0) & (tm != team[players[k]])] = 1e6
                        cost[a, tm < 0] += 2.0
                    cost[d2 > 9.0] = 1e6
                    ri, ci = linear_sum_assignment(cost)
                    for a, c in zip(ri, ci):
                        if cost[a, c] >= 1e5:
                            continue
                        k = rows[a]
                        S_ = P[k, 0, 0] + R_DET**2
                        Kg = P[k, :, 0] / S_
                        for ax in range(2):
                            x[k, ax] = x[k, ax] + Kg * (pos[c][ax] - x[k, ax, 0])
                        P[k] = P[k] - np.outer(Kg, P[k, 0])
                        if src[fi, k] == 0:
                            src[fi, k] = 1
                        last_vis[k] = times[fi]
        X[fi], Pm[fi] = x, P
    # RTS smoother
    Xs = X.copy()
    dt = 1.0 / FPS
    F = np.array([[1, dt], [0, 1]])
    for fi in range(T - 2, -1, -1):
        for k in range(K):
            if not started[k]:
                continue
            C = Pm[fi, k] @ F.T @ np.linalg.inv(Pp[fi + 1, k])
            Xs[fi, k] = X[fi, k] + (Xs[fi + 1, k] - Xp[fi + 1, k]) @ C.T
    pos = Xs[:, :, :, 0]
    # source labels for rendering: recent snapshot (<0.5 s) / recent vision (<0.3 s)
    lab = np.zeros((T, K), int)
    for k in range(K):
        for fi in range(T):
            t = times[fi]
            lab[fi, k] = (
                2
                if np.abs(snap_t[players[k]] - t).min() < 0.5
                else (1 if (use_vision and src[:, k][max(fi - 8, 0) : fi + 9].max() == 1) else 0)
            )
    return frames, pos, lab, active & started[None, :], players


def draw_players(img, g, f, rv, pos, lab, act, players, roster, label="", team_col=((255, 120, 0), (0, 215, 255))):
    out = img.copy()
    ov = out.copy()
    for k, p in enumerate(players):
        if not act[k]:
            continue
        r = roster[p]
        col = team_col[0 if r["team"] == "Home" else 1]
        ring = np.array([[pos[k, 0] + 0.7 * np.cos(a), pos[k, 1] + 0.7 * np.sin(a), 0.0] for a in np.linspace(0, 2 * np.pi, 28)])
        q = S.project(g, f, rv, ring)
        if not np.isfinite(q).all() or np.abs(q).max() > 4000:
            continue
        th = {2: 4, 1: 3, 0: 2}[lab[k]]
        cv2.polylines(ov, [q.astype(np.int32)], True, col, th, cv2.LINE_AA)
        head = S.project(g, f, rv, np.array([[pos[k, 0], pos[k, 1], 2.15]]))[0]
        if 0 < head[0] < 1920 and 0 < head[1] < 1080:
            txt = ("G" if r.get("goalie") else "") + str(r["jersey"])
            (tw, th_), _ = cv2.getTextSize(txt, 0, 0.9, 2)
            x0, y0 = int(head[0] - tw / 2), int(head[1])
            cv2.rectangle(ov, (x0 - 5, y0 - th_ - 5), (x0 + tw + 5, y0 + 5), col, -1)
            cv2.putText(ov, txt, (x0, y0), 0, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    out = cv2.addWeighted(ov, 0.9, out, 0.1, 0)
    if label:
        cv2.putText(out, label, (30, 60), 0, 1.4, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(out, label, (30, 60), 0, 1.4, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def render(t0, dur, prefix):
    g, snaps = load_snaps()
    DT = Detections()
    roster = g["roster"]
    team = {i: (0 if r["team"] == "Home" else 1) for i, r in enumerate(roster)}
    fa, pa, la, aa, players = track(snaps, DT, team, t0, t0 + dur, True)
    fb, pb, lb, ab, _ = track(snaps, DT, team, t0, t0 + dur, False)
    N0 = DT.F0

    def writer(path, w, h):
        return subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{w}x{h}",
                "-r",
                "25",
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                path,
            ],
            stdin=subprocess.PIPE,
        )

    w1 = writer(f"{prefix}_tracker.mp4", 1920, 1080)
    w2 = writer(f"{prefix}_compare.mp4", 1920, 540)
    for N in range(int(round(t0 * 25)), int(round((t0 + dur) * 25))):
        fi = int(np.argmin(abs(fa - N)))
        img = cv2.imread(f"cache/f300/{N - N0:05d}.jpg")
        kk = DT.rowN[N]
        f, rv = DT.Z["f"][kk], DT.Z["rv"][kk]
        A = draw_players(img, DT.g, f, rv, pa[fi], la[fi], aa[fi], players, roster, "snapshots + vision")
        B = draw_players(img, DT.g, f, rv, pb[fi], lb[fi], ab[fi], players, roster, "snapshots only (interpolation)")
        w1.stdin.write(A.tobytes())
        w2.stdin.write(np.hstack([cv2.resize(B, (960, 540)), cv2.resize(A, (960, 540))]).tobytes())
    for w in (w1, w2):
        w.stdin.close()
        w.wait()
    print("wrote", prefix)


if __name__ == "__main__":
    render(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
