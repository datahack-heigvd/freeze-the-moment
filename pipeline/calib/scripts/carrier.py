"""Puck carrier inside metadata gaps (the live app marks the carrier only from metadata, so it vanishes with it).
Truth: the live app's rule on each snapshot (closest to the puck within 2 m, 0.7 m clear). Training uses the other window and the visible blocks only.
  K1 layout   per ring: where it stands relative to everyone else (gradient boosting on tracked positions and velocities)
  K2 crop     per detection: a CNN looking at the feet and stick area says 'has the puck' (id_gpu.py K)
  K1+K2       both scores summed as log-odds, then a Viterbi pass over the rings: the carrier changes only when the evidence insists
usage (WIN=300|2160):  carrier.py label            -> cache/bench/<win>/carrier_lab.npz (per detection: 1 carrier, 0 other, -1 unknown)
                       carrier.py run <ident method>  K0 (hold), K1, K2, K1+K2 on the rings of that identity method"""

import pickle
import sys

import numpy as np

sys.path.insert(0, "scripts")
import idbench as B
from sklearn.ensemble import HistGradientBoostingClassifier
from window import NAME
from window import W as WIN

OTHER = {"300": "2160", "2160": "300"}.get(NAME)
POS_R, SPREAD = (
    1.2,
    12,
)  # m from the carrier's position to its tracklet; frames either side of a snapshot labelled along that tracklet
SWITCH = 4.0  # Viterbi: log-odds cost of handing the puck to another ring
VEL = 5  # frames over which a ring's velocity is measured


def corr_of(win, masked):

    return pickle.load(
        open(
            f"cache/bench/{win}/gtrack.pkl" if masked else {"300": "cache/gtrack.pkl", "2160": "cache/gtrack2160.pkl"}[win], "rb"
        )
    )["corr"]


def carrier_lab(Z, R, snaps, car, keep=lambda t: True):
    """Per detection: 1 carrier, 0 someone else in the same frame, -1 unknown. At each snapshot with a carrier (keep(t) true), the
    carrier's tracklet (nearest within POS_R) is followed SPREAD frames either way."""
    byN = {}
    for i, N in enumerate(Z["N"]):
        byN.setdefault(int(N), []).append(i)
    corr, tracks = R["corr"], R["tracks"]
    y = np.full(len(Z["N"]), -1)
    for t, d in snaps:
        N = int(round(t * B.FPS))
        p = car.get(N)
        if p is None or p not in d or not keep(t) or not any(M in byN for M in range(N - SPREAD, N + SPREAD + 1)):
            continue
        c = corr(d[p], p)
        alive = [tr for tr in tracks if tr["a"] <= N <= tr["b"] and tr.get("team", 0) != -2]
        if not alive:
            continue
        dist = [np.linalg.norm(tr["pos"][N - tr["a"]] - c) for tr in alive]
        tr = alive[int(np.argmin(dist))]
        if min(dist) >= POS_R:
            continue
        for M in range(max(N - SPREAD, tr["a"]), min(N + SPREAD, tr["b"]) + 1):
            ix = byN.get(M)
            if not ix:
                continue
            dd = np.linalg.norm(Z["xy"][ix] - tr["pos"][M - tr["a"]], axis=1)
            k = int(np.argmin(dd))
            if dd[k] < 0.5:
                y[ix] = np.where(y[ix] == 1, 1, 0)
                y[ix[k]] = 1
    return y


def label():
    """Per detection of this window: y_full from every snapshot, y_vis from visible-block snapshots only."""
    Z = np.load(f"{B.OUT}/dets.npz")
    car = B.carriers()
    g, snaps = B.load_snaps()
    out = dict(
        y_full=carrier_lab(Z, pickle.load(open(WIN["gtrack"], "rb")), snaps, car),
        y_vis=carrier_lab(Z, pickle.load(open(f"cache/bench/{NAME}/gtrack.pkl", "rb")), snaps, car, lambda t: not B.hidden_at(t)),
    )
    np.savez(f"{B.OUT}/carrier_lab.npz", **out)
    print(
        f"{NAME}: positives full {(out['y_full'] == 1).sum()}, visible {(out['y_vis'] == 1).sum()}; negatives full {(out['y_full'] == 0).sum()}"
    )


def ring_tracks(win, method):
    """{frame: [(track key, roster index or -1, xy)]} from a saved method plus per-frame velocity: rings matched frame to frame by nearest position."""
    R = pickle.load(open(f"cache/bench/{win}/{method}.pkl", "rb"))["rings"]
    return R


def features(rings, N):
    """Per ring at frame N: position, speed, nearest teammate / opponent / anyone distance, people within 2/4 m, distance to the centroid,
    rank of closeness to the centroid, distance to the nearest goal line point."""
    rs = rings.get(N, [])
    prev = rings.get(N - VEL, [])
    if not rs:
        return np.zeros((0, 12))
    xy = np.array([r[1] for r in rs])
    pv = np.array([r[1] for r in prev]) if prev else xy
    v = np.array([(q - pv[np.argmin(np.linalg.norm(pv - q, axis=1))]) * B.FPS / VEL for q in xy])
    sp = np.linalg.norm(v, axis=1)
    D = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    np.fill_diagonal(D, 99)
    cen = xy.mean(0)
    dc = np.linalg.norm(xy - cen, axis=1)
    rank = np.argsort(np.argsort(dc)) / max(len(rs) - 1, 1)
    goal = np.abs(np.abs(xy[:, 0]) - 26.0)
    return np.c_[
        np.abs(xy[:, 0]),
        np.abs(xy[:, 1]),
        sp,
        D.min(1),
        (D < 2).sum(1),
        (D < 4).sum(1),
        dc,
        rank,
        goal,
        np.full(len(rs), len(rs)),
        np.sort(D, 1)[:, 1] if len(rs) > 2 else np.full(len(rs), 99),
        v[:, 0] * np.sign(xy[:, 0]),
    ]


def layout_set(win, method, masked, corr):
    """(features, label, frame, ring index) at snapshot frames with a carrier: label 1 for the ring within POS_R of the carrier's position."""
    rings = ring_tracks(win, method)
    g, snaps = B.load_snaps()
    car = B.carriers()
    X, y = [], []
    for t, d in snaps:
        N = int(round(t * B.FPS))
        p = car.get(N)
        if p is None or p not in d or N not in rings or (masked and B.hidden_at(t)):
            continue
        f = features(rings, N)
        if not len(f):
            continue
        dist = np.linalg.norm(np.array([r[1] for r in rings[N]]) - corr(d[p], p), axis=1)
        k = int(np.argmin(dist))
        if dist[k] >= POS_R:
            continue
        X.append(f)
        y.append(np.arange(len(f)) == k)
    return np.concatenate(X), np.concatenate(y).astype(int)


def k2_scores(rings, N, Z, logit, byN):
    """Per ring at frame N: the crop model's mean log-odds over the detections under it in frames N-2..N+2."""
    out = np.zeros(len(rings.get(N, [])))
    for j, (_, xy, *_) in enumerate(rings.get(N, [])):
        v = []
        for M in range(N - 2, N + 3):
            ix = byN.get(M)
            if not ix:
                continue
            d = np.linalg.norm(Z["xy"][ix] - xy, axis=1)
            k = int(np.argmin(d))
            if d[k] < 0.8:
                v.append(float(logit[ix[k]]))
        out[j] = np.mean(v) if v else -3.0
    return out


def viterbi(rings, frames, score):
    """Carrier ring per frame: maximise the summed log-odds of the chosen ring minus SWITCH per hand-over (rings followed by position)."""
    best, path = None, {}
    for N in frames:
        rs = rings.get(N, [])
        if not rs:
            best = None
            continue
        xy = np.array([r[1] for r in rs])
        s = score[N]
        if best is None:
            cur = s.copy()
        else:
            v, pxy = best
            d = np.linalg.norm(xy[:, None] - pxy[None], axis=2)
            same = d.argmin(1)
            stay = v[same] + np.where(d.min(1) < 1.0, 0.0, -SWITCH)
            cur = s + np.maximum(stay, v.max() - SWITCH)
        best = (cur, xy)
        path[N] = int(np.argmax(cur))
    return path


def posterior(rings, frames, score, stay=0.99):
    """Forward-backward over the rings (log domain): per frame, the most likely carrier ring and its margin, the log posterior ratio to the
    second most likely ring (a confidence that does not saturate). Same model as viterbi(): the puck stays with the ring continuing the
    previous one (nearest within 1 m) with probability stay per frame step, else jumps to any ring; a ring's evidence is exp(its log-odds)."""
    from scipy.special import logsumexp

    fw, steps = [], []
    for N in frames:
        rs = rings.get(N, [])
        if not rs:
            fw.append(None)
            steps.append(None)
            continue
        xy = np.array([r[1] for r in rs])
        e = np.asarray(score[N], float)
        prev = fw[-1] if fw else None
        if prev is None:
            la = e - logsumexp(e)
            LT = None
        else:
            pla, pxy = prev
            d = np.linalg.norm(xy[:, None] - pxy[None], axis=2)
            T = np.full((len(pxy), len(xy)), (1 - stay) / len(xy))
            cont = d.argmin(0)
            ok = d.min(0) < 1.0
            T[np.flatnonzero(ok), cont[ok]] += stay
            LT = np.log(T / T.sum(1, keepdims=True))
            la = e + logsumexp(pla[:, None] + LT, axis=0)
            la -= logsumexp(la)
        fw.append((la, xy))
        steps.append(LT)
    out = {}
    lb = None
    for N, f, LT in zip(reversed(frames), reversed(fw), reversed(steps)):
        if f is None:
            lb = None
            continue
        la, _ = f
        g = la + (lb if lb is not None else 0.0)
        o = np.argsort(-g)
        out[N] = (int(o[0]), float(g[o[0]] - g[o[1]]) if len(g) > 1 else 99.0)
        if LT is None:
            lb = None
            continue
        e = np.asarray(score[N], float)
        lb = logsumexp(LT + (e + (lb if lb is not None else 0.0))[None, :], axis=1)
        lb -= logsumexp(lb)
    return out


def run(method):
    import os

    rings = ring_tracks(NAME, method)
    hidden_N = [N for N in sorted(rings) if B.hidden_at(N / B.FPS)]
    Xo, yo = layout_set(
        OTHER, method if os.path.exists(f"cache/bench/{OTHER}/{method}.pkl") else "baseline_gtrack", False, corr_of(OTHER, False)
    )
    Xv, yv = layout_set(NAME, method, True, corr_of(NAME, True))
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(np.r_[Xo, Xv], np.r_[yo, yv])
    Z = np.load(f"{B.OUT}/dets.npz")
    byN = {}
    for i, N in enumerate(Z["N"]):
        byN.setdefault(int(N), []).append(i)
    k2 = np.load(f"{B.OUT}/out_K.npz")["K"].astype(np.float32) if os.path.exists(f"{B.OUT}/out_K.npz") else None
    s1, s2 = {}, {}
    for N in hidden_N:
        f = features(rings, N)
        if len(f):
            p = np.clip(m.predict_proba(f)[:, 1], 1e-4, 1 - 1e-4)
            s1[N] = np.log(p / (1 - p))
        else:
            s1[N] = np.zeros(0)
        if k2 is not None:
            s2[N] = k2_scores(rings, N, Z, k2, byN)
    variants = {"K1": s1}
    if k2 is not None:
        variants["K2"] = s2
        variants["K1+K2"] = {N: s1[N] + s2[N] for N in hidden_N}
    last = {}  # K0: the last carrier seen before the gap, held through it
    car = B.carriers()
    held = None
    for N in sorted(rings):
        if not B.hidden_at(N / B.FPS) and N in car:
            held = car[N]
        if B.hidden_at(N / B.FPS) and held is not None:
            r = [x for x in rings[N] if x[0] == held]
            if r:
                last[N] = (held, r[0][1])
    out = {"K0": last}
    for name, sc in variants.items():
        path = viterbi(rings, hidden_N, sc)
        out[name] = {N: (rings[N][k][0], rings[N][k][1]) for N, k in path.items()}
    for name, c in out.items():
        B.save(f"{method}|{name}", rings, c)
        B.score(f"{method}|{name}")
    if k2 is not None:  # K2 with a confidence: precision and coverage per threshold
        post = posterior(rings, hidden_N, s2)
        car = B.carriers()
        corr = B.truth_corr()
        hs = {int(round(t * B.FPS)): d for t, d in B.hidden_snaps()}
        for thr in (0, 2, 4, 6, 8, 12, 16):
            n = shown = right = 0
            for N, d in hs.items():
                if N not in car or car[N] not in d or N not in post:
                    continue
                n += 1
                k, mg = post[N]
                if mg < thr:
                    continue
                shown += 1
                right += np.linalg.norm(rings[N][k][1] - corr(d[car[N]], car[N])) < B.NEAR
            print(
                f"{NAME:>5} K2 margin >= {thr:2d} nats: shown {100 * shown / max(n, 1):5.1f}% of {n} carrier moments, right when shown {100 * right / max(shown, 1):5.1f}%"
            )
        c = {N: (rings[N][k][0], rings[N][k][1]) for N, (k, pr) in post.items()}
        B.save(f"{method}|K2fb", rings, c)
        B.score(f"{method}|K2fb")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "label":
        label()
    elif cmd == "run":
        run(sys.argv[2])
