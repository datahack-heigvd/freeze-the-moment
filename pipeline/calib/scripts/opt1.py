"""Option 1: snapshot interpolation models, scored by hiding snapshots (leave-one-out) and predicting them from neighbours."""

import json

import numpy as np

HX, HY = 29.9, 15.0


def load_snaps(path="../../store/site/game.json"):
    g = json.load(open(path))
    snaps = []
    for f in g["frames"]:
        d = {}
        for k in range(3, len(f), 3):
            if f[k] is None:
                break
            d[f[k]] = np.array([f[k + 1], f[k + 2]], float)
        snaps.append((f[0], d))
    return g, snaps


def linear(ts, xs, t):
    return np.array([np.interp(t, ts, xs[:, 0]), np.interp(t, ts, xs[:, 1])])


def hermite(ts, xs, t):
    """Cubic Hermite between the two bracketing samples; velocities from central differences over the neighbours."""
    i = np.searchsorted(ts, t)
    if i == 0 or i == len(ts):
        return xs[min(i, len(ts) - 1)]
    v = np.gradient(xs, ts, axis=0) if len(ts) > 2 else np.zeros_like(xs)
    t0, t1 = ts[i - 1], ts[i]
    h = t1 - t0
    s = (t - t0) / h
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * xs[i - 1] + h10 * h * v[i - 1] + h01 * xs[i] + h11 * h * v[i]


def kalman(ts, xs, t, q=4.0, r=0.2):
    """Constant-velocity Kalman filter + RTS smoother per axis; q = acceleration noise (m/s^2), r = measurement noise (m).
    Returns the smoothed position at time t (t need not be a measurement time)."""
    times = np.unique(np.r_[ts, t])
    n = len(times)
    out = np.zeros(2)
    for ax in range(2):
        m = np.zeros((n, 2))
        P = np.zeros((n, 2, 2))
        mp = np.zeros((n, 2))
        Pp = np.zeros((n, 2, 2))
        F = [None] * n
        x = np.array([xs[0, ax], 0.0])
        Pc = np.diag([r**2, 25.0])
        meas = {float(a): b for a, b in zip(ts, xs[:, ax])}
        for k in range(n):
            if k > 0:
                dt = times[k] - times[k - 1]
                Fk = np.array([[1, dt], [0, 1]])
                Q = q**2 * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
                x = Fk @ x
                Pc = Fk @ Pc @ Fk.T + Q
                F[k] = Fk
            mp[k] = x
            Pp[k] = Pc
            z = meas.get(float(times[k]))
            if z is not None:
                S = Pc[0, 0] + r**2
                K = Pc[:, 0] / S
                x = x + K * (z - x[0])
                Pc = Pc - np.outer(K, Pc[0])
            m[k] = x
            P[k] = Pc
        xs_s = m.copy()
        for k in range(n - 2, -1, -1):
            C = P[k] @ F[k + 1].T @ np.linalg.inv(Pp[k + 1])
            xs_s[k] = m[k] + C @ (xs_s[k + 1] - mp[k + 1])
        out[ax] = xs_s[int(np.searchsorted(times, t))][0]
    return out


METHODS = {"linear": linear, "hermite": hermite, "kalman": kalman}


def neighbours(snaps, i, p, span=4, hide=1):
    """Measurements of player p from up to `span` snapshots on each side of i, excluding i .. i+hide-1 (all hidden)."""
    ts, xs = [], []
    for j in list(range(max(i - span, 0), i)) + list(range(i + hide, min(i + hide + span, len(snaps)))):
        tj, dj = snaps[j]
        if p in dj and abs(tj - snaps[i][0]) < 12:
            ts.append(tj)
            xs.append(dj[p])
    return np.array(ts), np.array(xs)


def loo(snaps, methods, hide=1, lo=0, hi=None, params=None):
    """For each snapshot i (and player), hide snapshots i..i+hide-1 and predict snapshot i. Returns list of (span_s, {method: err})."""
    out = []
    hi = hi or len(snaps)
    for i in range(max(lo, 1), min(hi, len(snaps) - hide - 1)):
        ti, di = snaps[i]
        for p, truth in di.items():
            ts, xs = neighbours(snaps, i, p, hide=hide)
            if len(ts) < 2 or not (ts[0] < ti < ts[-1]):
                continue
            span = ts[ts > ti].min() - ts[ts < ti].max()
            if span > 12:
                continue
            e = {}
            for name in methods:
                kw = (params or {}).get(name, {})
                e[name] = float(np.linalg.norm(METHODS[name](ts, xs, ti, **kw) - truth))
            out.append((span, p, i, e))
    return out


def summarize(res, methods, bins=((0, 1), (1, 2), (2, 3), (3, 5), (5, 12))):
    S = np.array([r[0] for r in res])
    lines = []
    for lo, hi in bins:
        m = (S >= lo) & (S < hi)
        if not m.any():
            continue
        cells = []
        for name in methods:
            e = np.array([r[3][name] for r in res])[m]
            cells.append(f"{name}: med {np.median(e):.2f} p90 {np.percentile(e, 90):.2f}")
        lines.append(f"span {lo}-{hi}s (n={m.sum():5d}) | " + " | ".join(cells))
    return "\n".join(lines)


if __name__ == "__main__":
    g, snaps = load_snaps()
    sub = range(0, len(snaps), 7)  # tuning subset
    best = None
    for q in (2.0, 4.0, 6.0, 10.0):
        for r in (0.1, 0.25, 0.5):
            res = []
            for i in sub:
                res += loo(snaps, ["kalman"], lo=i, hi=i + 1, params={"kalman": dict(q=q, r=r)})
            e = np.array([x[3]["kalman"] for x in res])
            sc = np.mean(np.minimum(e, 5.0))
            print(f"q={q} r={r}: mean(min(err,5m)) {sc:.3f} median {np.median(e):.3f}", flush=True)
            if best is None or sc < best[0]:
                best = (sc, q, r)
    print("best", best)
    q, r = best[1], best[2]
    for hide in (1, 2):
        res = loo(snaps, list(METHODS), hide=hide, params={"kalman": dict(q=q, r=r)})
        print(f"\n== hide {hide} snapshot(s) at a time; n={len(res)}\n" + summarize(res, list(METHODS)))
