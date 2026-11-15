"""Identity inside metadata gaps: the masked tracker's tracklets + appearance evidence from the GPU models (id_gpu.py).
Evidence per (tracklet, player) is a log-likelihood ratio against a uniform guess, averaged over the tracklet's crops (every STEP-th frame)
and scaled by lam * min(crops, CAP): a long, clearly seen tracklet counts more, but never like hundreds of independent looks.
  D  player classifier         B  jersey number reader (number + team)         A  embedding vs player prototypes from the visible blocks
Solvers: greedy = gtrack's propagation rounds (posterior >= THR); C = one integer program over all tracklets and players
(one place per player at a time, motion links between consecutive tracklets of a player rewarded, metadata seeds fixed).
A ':split' suffix first cuts tracklets where the appearance says the person changed (two players crossing inside a gap, where no
metadata can cut them): Viterbi over the crops with states = the tracklet's team players + 'someone else', SPLIT_SW nats per switch.
usage (WIN=300|2160):  idsolve.py run <method>...   methods: greedy:<cues>[:split] | ilp:<cues>[:split], cues '+'-joined from D,B,A,A0 or 'none';
                       a cue may carry a weight (D*0.5): cues that look at the same pixels (D and A) share one budget instead of double counting"""

import pickle
import sys

import numpy as np

sys.path.insert(0, "scripts")
import gtrack as G
import idbench as B
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from vtrack import FPS, NULL, TEMPER, THR, snap_list, team_of_tr

STEP, CAP, CLIP = 3, 30, 5.0
LAM = dict(D=0.15, B=0.15, A=0.15, A0=0.15)
TAU = 0.1  # cosine temperature of the embedding-to-prototype softmax
EDGE_GAP, TOPK = 10.0, 6  # ILP: longest gap (s) bridged by a motion link; candidate players kept per tracklet
SPLIT_SW, SPLIT_MIN = 20.0, 8  # appearance split: switch penalty (nats), fewest crops in a piece


class Ctx:
    """Tracklets, metadata, detections and appearance outputs of one stretch of video. Defaults: the benchmark window with masked metadata;
    the full-game run passes its chunk's tracker result, all snapshots, detections and model outputs."""

    def __init__(self, R=None, snaps=None, Z=None, out=None):
        self.R = R or pickle.load(open(f"{B.OUT}/gtrack.pkl", "rb"))
        self.tracks = self.R["tracks"]
        self.corr = self.R["corr"]
        self.g, masked = B.masked_snaps()
        self.roster = self.g["roster"]
        self.npl = len(self.roster)
        self.sl = snap_list(snaps if snaps is not None else masked, self.R["n0"], self.R["n1"])
        self.team_p = np.array([0 if r["team"] == "Home" else 1 for r in self.roster])
        self.goalie = np.array([bool(r.get("goalie")) for r in self.roster])
        self.Z = Z if Z is not None else np.load(f"{B.OUT}/dets.npz")
        self.rows = track_rows(self.tracks, self.Z)
        self.out = out
        if out is None:
            self.out = {}
            for t in ("D", "B", "A"):
                try:
                    self.out.update(dict(np.load(f"{B.OUT}/out_{t}.npz")))
                except FileNotFoundError:
                    pass
        self.ident0, self.conf0 = self.R["ident0"], self.R["conf0"]
        self.pool = pool_gap(self.tracks, self.sl, self.npl)

    def split(self, cues):
        """Cut tracklets where the appearance changes player; pieces keep the metadata seed only if their appearance does not contradict it."""
        L = sum(lam * crop_llr(self, c) for c, lam in weighted(cues))
        out, i0, c0, ncut = [], [], [], 0
        for ti, (tr, r) in enumerate(zip(self.tracks, self.rows)):
            if len(r) < 2 * SPLIT_MIN or tr.get("team", -1) < 0:
                out.append(tr)
                i0.append(self.ident0[ti])
                c0.append(self.conf0[ti])
                continue
            ps = np.flatnonzero((self.team_p == tr["team"]) | self.goalie)
            Lk = np.c_[L[r][:, ps], np.zeros(len(r))]
            dp = Lk[0].copy()
            bp = np.zeros(Lk.shape, int)
            for k in range(1, len(r)):
                top = int(np.argmax(dp))
                sw = dp[top] - SPLIT_SW
                bp[k] = np.where(dp >= sw, np.arange(Lk.shape[1]), top)
                dp = Lk[k] + np.maximum(dp, sw)
            path = np.zeros(len(r), int)
            path[-1] = int(np.argmax(dp))
            for k in range(len(r) - 1, 0, -1):
                path[k - 1] = bp[k, path[k]]
            Nk = self.Z["N"][r]
            edges = [tr["a"]]
            states = [path[0]]
            for k in range(1, len(r)):
                if path[k] != path[k - 1]:
                    edges.append(int((Nk[k - 1] + Nk[k]) // 2))
                    states.append(path[k])
            edges.append(tr["b"] + 1)
            keep = [
                (e0, e1, st) for e0, e1, st in zip(edges[:-1], edges[1:], states) if ((Nk >= e0) & (Nk < e1)).sum() >= SPLIT_MIN
            ]
            if len(keep) < 2:
                out.append(tr)
                i0.append(self.ident0[ti])
                c0.append(self.conf0[ti])
                continue
            ncut += len(keep) - 1
            keep[0] = (tr["a"], *keep[0][1:])
            keep[-1] = (keep[-1][0], tr["b"] + 1, keep[-1][2])
            for e0, e1, st in keep:
                out.append(
                    {
                        **tr,
                        "a": e0,
                        "b": e1 - 1,
                        "pos": tr["pos"][e0 - tr["a"] : e1 - tr["a"]],
                        "meas": tr["meas"][e0 - tr["a"] : e1 - tr["a"]],
                    }
                )
                ok = self.ident0[ti] >= 0 and (st == len(ps) or ps[st] == self.ident0[ti])
                i0.append(self.ident0[ti] if ok else -1)
                c0.append(self.conf0[ti] if ok else 0.0)
        self.tracks, self.ident0, self.conf0 = out, np.array(i0), np.array(c0)
        self.rows = track_rows(self.tracks, self.Z)
        self.pool = pool_gap(self.tracks, self.sl, self.npl)
        return ncut


def track_rows(tracks, Z):
    """Detection indices (dets.npz order) a tracklet was built from, on the measured frames that are multiples of STEP."""
    byN = {}
    for i, N in enumerate(Z["N"]):
        byN.setdefault(int(N), []).append(i)
    out = []
    for tr in tracks:
        r = []
        for N in range(tr["a"] + (-tr["a"]) % STEP, tr["b"] + 1, STEP):
            j = N - tr["a"]
            c = byN.get(N)
            if not tr["meas"][j] or not c:
                continue
            d = np.linalg.norm(Z["xy"][c] - tr["pos"][j], axis=1)
            k = int(np.argmin(d))
            if d[k] < 0.5:
                r.append(c[k])
        out.append(np.array(r, int))
    return out


def pool_gap(tracks, sl, npl):
    """Candidates: seen in a snapshot within POOL_S of the tracklet, or in the last snapshot before it / the first after it (a player on the ice when a gap starts or ends)."""
    pool = G.pool_matrix(tracks, sl, npl)
    Ns = np.array([N for N, _ in sl])
    for ti, tr in enumerate(tracks):
        i = np.searchsorted(Ns, tr["a"]) - 1
        j = np.searchsorted(Ns, tr["b"], "right")
        for k in (i, j):
            if 0 <= k < len(sl):
                pool[ti, list(sl[k][1])] = True
    return pool


def crop_llr(ctx, cue):
    """(detections, players) per-crop log-likelihood ratio against a uniform guess, clipped below at -CLIP."""
    o = ctx.out
    if cue == "D":
        l = o["D"].astype(np.float32) + np.log(ctx.npl)
    elif cue == "B":
        lp = o["B"].astype(np.float32)
        J = [r["jersey"] for r in ctx.roster]
        T = np.array([int(j) // 10 if len(j) == 2 else 10 for j in J])
        U = np.array([int(j) % 10 for j in J])
        l = lp[:, T] + lp[:, 11 + U] + np.log(110)
    else:
        e = o[cue].astype(np.float32)
        lab = ctx.Z["lab_vis"]
        P = np.zeros((ctx.npl, e.shape[1]), np.float32)
        have = np.zeros(ctx.npl, bool)
        for p in range(ctx.npl):
            k = lab == p
            if k.sum() >= 20:
                P[p] = e[k].mean(0)
                P[p] /= np.linalg.norm(P[p])
                have[p] = True
        z = e @ P.T / TAU
        z[:, ~have] = -1e4
        z = z - z.max(1, keepdims=True)
        l = z - np.log(np.exp(z).sum(1, keepdims=True)) + np.log(have.sum())
    return np.maximum(l, -CLIP)


def weighted(cues):
    """'D*0.5' -> ('D', 0.5 * LAM['D'])."""
    return [(c.split("*")[0], LAM[c.split("*")[0]] * (float(c.split("*")[1]) if "*" in c else 1.0)) for c in cues]


def evidence(ctx, cues):
    E = np.zeros((len(ctx.tracks), ctx.npl))
    for cue, lam in weighted(cues):
        l = crop_llr(ctx, cue)
        for ti, r in enumerate(ctx.rows):
            if len(r):
                E[ti] += lam * min(len(r), CAP) * l[r].mean(0)
    return E


def greedy(ctx, E):
    ident, conf = G.propagate(
        ctx.tracks,
        ctx.ident0.copy(),
        ctx.conf0.copy(),
        ctx.sl,
        ctx.corr,
        ctx.team_p,
        ctx.goalie,
        verbose=False,
        extra=E,
        pool=ctx.pool,
    )
    return ident


def ilp(ctx, E):
    """max sum x[t,p] (U[t,p] - null[t]) + sum y[i,j,p] r[i,j]  s.t. one player per tracklet, conflicting tracklets never share a player,
    at most one motion link into and out of each (tracklet, player); metadata seeds fixed. Shown only if the local posterior agrees (>= THR)."""
    tracks, npl = ctx.tracks, ctx.npl
    nT = len(tracks)
    Sc, ns = G.evidence(tracks, ctx.sl, ctx.corr, npl)
    U = Sc + E
    cf = G.conflicts(tracks)
    null = -(NULL * ns + G.LINK_NULL * 2)
    ttm = np.array([team_of_tr(t) for t in tracks])
    seed = ctx.ident0
    allowed = ctx.pool.copy()
    for ti in range(nT):
        if ttm[ti] >= 0:
            allowed[ti] &= (ctx.team_p == ttm[ti]) | ctx.goalie
        if ttm[ti] == -2:
            allowed[ti] = False
        if seed[ti] >= 0:
            allowed[ti] = False
            allowed[ti, seed[ti]] = True
    cand = []
    for ti in range(nT):
        ps = np.flatnonzero(allowed[ti])
        if len(ps):
            cand += [(ti, int(p)) for p in ps[np.argsort(-U[ti, ps])][:TOPK]]
    xi = {c: k for k, c in enumerate(cand)}
    obj = [-(U[t, p] - null[t]) for t, p in cand]
    byp = {}
    for t, p in cand:
        byp.setdefault(p, []).append(t)
    edges = []
    for p, ts in byp.items():
        ts = sorted(ts, key=lambda t: tracks[t]["a"])
        for i in ts:
            for j in ts:
                gap = (tracks[j]["a"] - tracks[i]["b"]) / FPS
                if j == i or tracks[j]["a"] <= tracks[i]["a"] or gap > EDGE_GAP or cf[i, j]:
                    continue
                r = 2 * (G.LINK_NONE + G.link_ll(tracks[i], tracks[j]))
                if r > 0:
                    edges.append((i, j, p, r))
    nx = len(cand)
    nv = nx + len(edges)
    obj += [-r for *_, r in edges]
    rows, cols, vals, lo, hi = [], [], [], [], []

    def con(entries, l, h):
        k = len(lo)
        lo.append(l)
        hi.append(h)
        for c, v in entries:
            rows.append(k)
            cols.append(c)
            vals.append(v)

    byt = {}
    for k, (t, p) in enumerate(cand):
        byt.setdefault(t, []).append(k)
    for t, ks in byt.items():
        con([(k, 1) for k in ks], 0, 1)
    for p, ts in byp.items():
        for a in range(len(ts)):
            for b in range(a + 1, len(ts)):
                if cf[ts[a], ts[b]]:
                    con([(xi[(ts[a], p)], 1), (xi[(ts[b], p)], 1)], 0, 1)
    out_e, in_e = {}, {}
    for k, (i, j, p, _) in enumerate(edges):
        out_e.setdefault((i, p), []).append(nx + k)
        in_e.setdefault((j, p), []).append(nx + k)
    for (t, p), ks in out_e.items():
        con([(k, 1) for k in ks] + [(xi[(t, p)], -1)], -np.inf, 0)
    for (t, p), ks in in_e.items():
        con([(k, 1) for k in ks] + [(xi[(t, p)], -1)], -np.inf, 0)
    lb = np.zeros(nv)
    lb[[xi[(t, int(seed[t]))] for t in range(nT) if seed[t] >= 0]] = 1
    A = coo_matrix((vals, (rows, cols)), shape=(len(lo), nv)).tocsr()
    res = milp(
        np.array(obj),
        constraints=LinearConstraint(A, lo, hi),
        integrality=np.ones(nv),
        bounds=Bounds(lb, np.ones(nv)),
        options=dict(time_limit=300, mip_rel_gap=1e-3),
    )
    x = np.round(res.x[:nx]).astype(bool)
    ident = np.full(nT, -1)
    chains = [[] for _ in range(npl)]
    for k in np.flatnonzero(x):
        t, p = cand[k]
        ident[t] = p
        chains[p].append(t)
    for p in range(npl):
        chains[p].sort(key=lambda j: tracks[j]["a"])
    shown = ident.copy()
    for t in range(nT):  # display rule: the solution's choice must also be locally certain
        if ident[t] < 0 or seed[t] >= 0:
            continue
        ps = np.flatnonzero(allowed[t])
        s = np.array([U[t, p] + G.link_score(tracks, [j for j in chains[p] if j != t], t) for p in ps])
        z = np.r_[s, null[t]] * TEMPER
        z = np.exp(z - z.max())
        post = z / z.sum()
        k = int(np.argmax(post[:-1]))
        if ps[k] != ident[t] or post[k] < THR:
            shown[t] = -1
    return shown


def merge(base, ctx, ident, near=1.0):
    """base rings (the production tracker's, {frame: [(player, xy, number shown, ...)]}) keep their numbers; a ring without one takes the
    solver's identity of the tracklet under it (within near m), unless that player is already numbered elsewhere in the frame.
    Extra fields of a ring (its team) are kept, the team taken from the roster when the ring gets a player."""
    items = G.player_items(ctx.tracks, ident, ctx.team_p, ctx.roster)
    out = {}
    for N, rs in base.items():
        sol = [(int(ident[i]), xy) for i, xy in G.frame_items(ctx.tracks, items, N) if ident[i] >= 0]
        taken = {e[0] for e in rs if e[2]}
        new = []
        for p, xy, num, *rest in rs:
            if not num and sol:
                q, qxy = min(sol, key=lambda e: np.linalg.norm(e[1] - xy))
                if np.linalg.norm(qxy - xy) < near and q not in taken:
                    p, num = q, True
                    taken.add(q)
                    rest = [int(ctx.team_p[q])] * len(rest)
            new.append((p, xy, num, *rest))
        out[N] = new
    return out


def rings_of(ctx, ident):
    items = G.player_items(ctx.tracks, ident, ctx.team_p, ctx.roster)
    return {
        N: [(int(ident[i]) if ident[i] >= 0 else -1, xy, ident[i] >= 0) for i, xy in G.frame_items(ctx.tracks, items, N)]
        for N in range(ctx.R["n0"], ctx.R["n1"])
    }


def run(ctx, method, save=True):
    solver, cues, *flags = method.split(":")
    cues = [] if cues == "none" else cues.split("+")
    if "split" in flags:
        print(f"  appearance split: {ctx.split(cues)} cuts", flush=True)
    E = evidence(ctx, cues)
    ident = (greedy if solver == "greedy" else ilp)(ctx, E)
    if save:
        B.save(method, rings_of(ctx, ident), None)
        pickle.dump(ident, open(f"{B.OUT}/ident_{method}.pkl", "wb"))
    return ident


if __name__ == "__main__":
    cmd = sys.argv[1]
    ctx = Ctx()
    if cmd == "run":
        for m in sys.argv[2:]:
            run(Ctx() if m.endswith(":split") else ctx, m)
            B.score(m)
    elif cmd == "merge":  # merge <base method> <solver method>: production rings + solver numbers
        base, m = sys.argv[2:4]
        c = Ctx()
        if m.endswith(":split"):
            c.split(m.split(":")[1].split("+"))
        name = f"{base}+{m}"
        B.save(
            name,
            merge(pickle.load(open(f"{B.OUT}/{base}.pkl", "rb"))["rings"], c, pickle.load(open(f"{B.OUT}/ident_{m}.pkl", "rb"))),
        )
        B.score(name)
        B.score(name, visible=True)
        B.score(base, visible=True)
        B.score(m, visible=True)
