"""Metadata-gap benchmark: the metadata is hidden in alternating BLOCK_S blocks of the window, the trackers see only the rest,
and every method is scored inside the hidden blocks against the hidden snapshots (corrected with the full-metadata map).
usage (WIN=300|2160):  idbench.py baseline       gtrack + ptrack on the visible metadata only
                       idbench.py score <method>  print the gap scores of a saved method (baseline_ptrack, baseline_gtrack, ...)"""

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "scripts")
import selfcal as S
from opt1 import load_snaps
from opt2 import Detections
from window import NAME
from window import W as WIN

FPS = 25.0
BLOCK_S = 30.0
NEAR = 2.5  # m: a ring this close to a player's corrected metadata position is on that player
OUT = f"cache/bench/{NAME}"
GAP_BINS = ((0, 10), (10, 20), (20, 30))


def hidden_at(t):
    """Hidden blocks: every second block of BLOCK_S seconds, starting with the second block of the window."""
    return int((t - WIN["t0"]) // BLOCK_S) % 2 == 1


def gap_age(t):
    """Seconds since the hidden block containing t started."""
    return (t - WIN["t0"]) % BLOCK_S


def masked_snaps():
    g, snaps = load_snaps()
    return g, [s for s in snaps if not hidden_at(s[0])]


def hidden_snaps(visible=False):
    """Snapshots of the window inside the hidden blocks (or, with visible=True, inside the visible ones)."""
    g, snaps = load_snaps()
    return [s for s in snaps if hidden_at(s[0]) != visible and WIN["t0"] <= s[0] < WIN["t0"] + WIN["dur"]]


def truth_corr():
    """metadata -> video map fitted by the full-metadata run (the referee for every method)."""
    return pickle.load(open(WIN["gtrack"], "rb"))["corr"]


def carriers():
    """{frame: roster index} of the puck carrier per snapshot (whole game), by the live app's rule: closest to the puck within 2 m and 0.7 m clear of the next."""
    g = load_snaps()[0]
    out = {}
    for f in g["frames"]:
        t, px, py = f[0], f[1], f[2]
        if px is None:
            continue
        d = sorted((np.hypot(f[k + 1] - px, f[k + 2] - py), f[k]) for k in range(3, len(f), 3) if f[k] is not None)
        if d and d[0][0] <= 2.0 and (len(d) == 1 or d[1][0] - d[0][0] >= 0.7):
            out[int(round(t * FPS))] = d[0][1]
    return out


def patched():
    """gtrack and ptrack modules wired to the visible metadata and to bench caches."""
    import gtrack as G
    import ptrack as P

    os.makedirs(OUT, exist_ok=True)
    G.load_snaps = P.load_snaps = masked_snaps
    G.CACHE = f"{OUT}/gtrack.pkl"
    P.CACHE = f"{OUT}/ptrack{{}}.pkl"
    return G, P


def baseline():
    G, P = patched()
    G.build(verbose=False)
    res = P.run(verbose=False)
    pickle.dump(res, open(P.CACHE.format(""), "wb"))
    W = P.World()
    show = P.display(res, W)
    rings = {N: [(p, xy, num) for p, xy, num, _ in P.frame_rings(show, W, N)] for N in range(W.n0, W.n1)}
    save("baseline_ptrack", rings)
    R = G.load()
    team_p = np.array([0 if r["team"] == "Home" else 1 for r in W.roster])
    items = G.player_items(R["tracks"], R["ident"], team_p, W.roster)
    rings = {}
    for N in range(W.n0, W.n1):
        rings[N] = [
            (int(R["ident"][i]) if R["ident"][i] >= 0 else -1, xy, R["ident"][i] >= 0)
            for i, xy in G.frame_items(R["tracks"], items, N)
        ]
    save("baseline_gtrack", rings)


def save(method, rings, carrier=None):
    """rings: {frame: [(roster index or -1, rink xy, number shown)]}; carrier: {frame: (roster index or -1, rink xy of the carrier's ring)} (optional)."""
    os.makedirs(OUT, exist_ok=True)
    pickle.dump(dict(rings=rings, carrier=carrier), open(f"{OUT}/{method}.pkl", "wb"))


def score(method, verbose=True, visible=False):
    """Inside the hidden blocks, per in-view player of a hidden snapshot: has a ring, has the right number; numbers shown on the wrong person.
    visible=True scores the visible blocks instead (a method must not get worse where the metadata is there)."""
    R = pickle.load(open(f"{OUT}/{method}.pkl", "rb"))
    rings = R["rings"]
    corr = truth_corr()
    DT = Detections()
    hs = hidden_snaps(visible)
    car = carriers()
    method = method + (" [visible]" if visible else "")
    t = dict(n=0, ring=0, num=0, shown=0, wrong=0, car_n=0, car_ok=0, car_pos=0, car_any=0)
    by = {b: np.zeros(3) for b in GAP_BINS}
    for t_, d in hs:
        N = int(round(t_ * FPS))
        k = DT.rowN.get(N)
        if k is None or N not in rings:
            continue
        meta = {p: corr(m, p) for p, m in d.items()}
        rs = rings[N]
        age = gap_age(t_)
        b = next(b for b in GAP_BINS if b[0] <= age < b[1])
        for p, c in meta.items():
            px = S.project(DT.g, DT.Z["f"][k], DT.Z["rv"][k], np.array([[*c, 0.0]]))[0]
            if not (60 < px[0] < 1860 and 60 < px[1] < 1020):
                continue
            has = any(np.linalg.norm(xy - c) < NEAR for _, xy, _ in rs)
            good = any(q == p and num and np.linalg.norm(xy - c) < NEAR for q, xy, num in rs)
            t["n"] += 1
            t["ring"] += has
            t["num"] += good
            by[b] += (1, has, good)
        if len(d) >= 11:
            for q, xy, num in rs:
                if not num:
                    continue
                t["shown"] += 1
                t["wrong"] += q not in meta or np.linalg.norm(xy - meta[q]) >= NEAR
        if R["carrier"] is not None and N in car:
            got = R["carrier"].get(N)
            t["car_n"] += 1
            if got is not None and car[N] in meta:
                t["car_any"] += 1
                t["car_ok"] += got[0] == car[N]
                t["car_pos"] += np.linalg.norm(got[1] - meta[car[N]]) < NEAR
    if verbose:
        n = max(t["n"], 1)
        print(
            f"{NAME:>5} {method:24s} rings {100 * t['ring'] / n:5.1f}%  right number {100 * t['num'] / n:5.1f}%  wrong numbers {t['wrong']}/{t['shown']} ({100 * t['wrong'] / max(t['shown'], 1):.1f}%)  of {t['n']} hidden in-view players"
            + "  | by gap age: "
            + "  ".join(f"{a}-{b_}s {100 * v[2] / max(v[0], 1):.0f}%" for (a, b_), v in by.items())
            + (
                f"  | carrier: right ring {100 * t['car_pos'] / max(t['car_n'], 1):.1f}%, right number {100 * t['car_ok'] / max(t['car_n'], 1):.1f}%, marked {100 * t['car_any'] / max(t['car_n'], 1):.0f}% (n={t['car_n']})"
                if R["carrier"] is not None
                else ""
            )
        )
    record(method, t, by)
    return t, by


def record(method, t, by):
    """Keep the scores of every method in cache/bench/<win>/scores.json (read by the report)."""
    import json

    path = f"{OUT}/scores.json"
    S_ = json.load(open(path)) if os.path.exists(path) else {}
    S_[method] = dict({k: int(v) for k, v in t.items()}, by_age={f"{a}-{b_}": [int(x) for x in v] for (a, b_), v in by.items()})
    json.dump(S_, open(path, "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "baseline":
        baseline()
    elif cmd == "score":
        score(sys.argv[2], visible="--visible" in sys.argv)
