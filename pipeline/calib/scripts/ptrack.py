"""Player-centric tracker: one state per on-ice roster player, matched to at most one detection per frame.
  - who is on the ice comes from the snapshots (seen within ON_ICE_S), the corrected metadata is an identity prior, never the drawn position
  - association (Hungarian over players x detections, with 'missed' and 'not a player' outcomes) replaces geometric gating
  - a wide box shared by two nearby teammates supports both (merged detections)
  - forward Kalman filter + backward RTS smoothing over the whole range
  - a number is shown only when the pairing is unambiguous (margin to the best alternative assignment) and agrees with the metadata
usage: ptrack.py run [parity] | eval | render <t0> <dur> <out_prefix>"""

import pickle
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
import gtrack as G
import roles
import selfcal as S
from opt1 import load_snaps
from opt2 import Detections
from scipy.optimize import linear_sum_assignment
from vtrack import snap_list

FPS = 25.0
DT_ = 1.0 / FPS
F_ = np.array([[1, DT_], [0, 1]])
Q_ = 6.0
QM = Q_**2 * np.array([[DT_**4 / 4, DT_**3 / 2], [DT_**3 / 2, DT_**2]])
R_DET, R_OCC, R_SHARE, R_META = 0.35, 0.7, 1.1, 0.9
GATE, C_MISS, W_META, META_CAP, TEAM_PEN, REF_PEN = 16.0, 14.0, 1.0, 30.0, 6.0, 10.0
P_CAP = 9.0  # position variance cap (m^2) of a state nobody has seen for a while
ON_ICE_S, GAP_S, UNSUPPORTED_AFTER, SUPPORT_GAP = 3.0, 60.0, 12, 20
WIDE_RATIO, RINK_SOFT, MARGIN_MIN, META_OK, MARGIN_CAP = 0.8, 2.0, 3.0, 2.0, 30.0
SHARE_PRIOR, SHARE_META, SHARE_MIN_SEP = (
    1.5,
    2.0,
    0.5,
)  # a second teammate may share a wide box only if it was already this close (m), agrees with its metadata, and ends up this far from the first
META_NUM = 2.5
RESET_D, RESET_N = 4.0, 8  # a state this far (m) from its metadata for this many frames is re-anchored on the metadata
NUM_GAP_S, NUM_EDGE_S = (
    30.0,
    3.0,
)  # a number needs metadata on both sides agreeing with the track (at most this far apart), or on one side within NUM_EDGE_S at the span edge
ANON_GATE, ANON_MISS, ANON_PN = (
    9.0,
    15,
    0.2,
)  # unnamed players: association gate (d2), frames unseen before the track ends, max not-a-player probability to start one
ANON_CONF, ANON_KEEP_CONF, ANON_INSIDE = (
    0.45,
    0.25,
    0.0,
)  # detector confidence to start / keep an unnamed track (spectators behind the glass score low), and min distance inside the boards (m)
ANON_MAX_OCC, ANON_ROLE_PN = (
    0.4,
    0.3,
)  # a track whose detections mostly have hidden feet (spectators behind the glass), or whose median colours the role classifier calls not-a-player (referee, staff), is not drawn
REF_PURITY, REF_FAR, REF_FRAC, REF_STATES, REF_JUDGED = (
    0.35,
    2.5,
    0.7,
    10,
    5,
)  # an unnamed track with no dominant team colour that keeps standing >2.5 m from every roster player (judged on frames where >=10 are tracked) is an official
ANON_MIN_SUP, ANON_BRIDGE, ANON_DUP = (
    8,
    8,
    1.0,
)  # supported frames needed before a ring shows, detector misses bridged, separation (m) from a named ring
INF = 1e6
from window import W as WIN

CACHE = WIN["ptrack"]


def rink_dist(xy, hx, hy, c):
    """Signed distance to the boards of the rounded rink (positive inside)."""
    x, y = xy[..., 0], xy[..., 1]
    qx = np.abs(x) - (hx - c)
    qy = np.abs(y) - (hy - c)
    return np.where((qx > 0) & (qy > 0), c - np.hypot(qx, qy), np.minimum(hx - np.abs(x), hy - np.abs(y)))


def clamp_inside(xy, hx, hy, c, m=0.1):
    x, y = xy
    qx, qy = abs(x) - (hx - c), abs(y) - (hy - c)
    if qx > 0 and qy > 0:
        r = np.hypot(qx, qy)
        if r > c - m:
            qx, qy = qx * (c - m) / r, qy * (c - m) / r
        return np.array([np.sign(x) * (qx + hx - c), np.sign(y) * (qy + hy - c)])
    return np.array([np.clip(x, -(hx - m), hx - m), np.clip(y, -(hy - m), hy - m)])


def predict(x, P, dt=DT_):
    Fm = np.array([[1, dt], [0, 1]])
    return x @ Fm.T, Fm @ P @ Fm.T + QM


def update(x, P, z, r):
    S_ = P[0, 0] + r**2
    K = P[:, 0] / S_
    for ax in range(2):
        x[ax] = x[ax] + K * (z[ax] - x[ax, 0])
    return x, P - np.outer(K, P[0])


class World:
    def __init__(self, parity=None):
        self.g, self.snaps = load_snaps()
        self.DT = Detections()
        self.roster = self.g["roster"]
        self.npl = len(self.roster)
        self.team_p = np.array([0 if r["team"] == "Home" else 1 for r in self.roster])
        self.goalie = np.array([bool(r.get("goalie")) for r in self.roster])
        R = G.load()
        self.corr = R["corr"]
        self.n0, self.n1 = R["n0"], R["n1"]
        d = dict(zip(S.GNAMES, self.DT.g))
        self.hx, self.hy, self.c = d["hx"], d["hy"], d["corner"]
        self.sl = snap_list(self.snaps, self.n0, self.n1, parity)
        self.snapN = [[] for _ in range(self.npl)]
        self.snapXY = [[] for _ in range(self.npl)]
        for N, dd in self.sl:
            for p, m in dd.items():
                self.snapN[p].append(N)
                self.snapXY[p].append(self.corr(m, p))
        self.snapN = [np.array(a) for a in self.snapN]
        self.snapXY = [np.array(a) for a in self.snapXY]
        # per-detection role probabilities from the tracklet classifier (home, away, not a player)
        tracks = R["tracks"]
        sl_all = snap_list(self.snaps, self.n0, self.n1)
        y = roles.labels(tracks, R["ident"], sl_all, self.corr, self.team_p)
        _, self.model = roles.fit_predict(tracks, y, verbose=False)
        model = self.model
        D = self.DT.D
        self.PD = model.predict_proba(roles.feature_matrix(D[:, 6:12]))

    def active(self, p, N):
        """On the ice: seen in a snapshot within ON_ICE_S, or seen before and after a metadata gap shorter than GAP_S (stoppages)."""
        t = self.snapN[p]
        if len(t) == 0:
            return False
        i = np.searchsorted(t, N)
        w = ON_ICE_S * FPS
        if (i < len(t) and t[i] - N <= w) or (i > 0 and N - t[i - 1] <= w):
            return True
        return 0 < i < len(t) and t[i] - t[i - 1] <= GAP_S * FPS

    def meta_at(self, p, N):
        """Corrected metadata position of player p at frame N (interpolated between neighbouring snapshots), or None."""
        t = self.snapN[p]
        if len(t) == 0:
            return None
        i = np.searchsorted(t, N)
        if 0 < i < len(t) and t[i] - t[i - 1] <= 3 * FPS:
            w = (N - t[i - 1]) / max(t[i] - t[i - 1], 1)
            return (1 - w) * self.snapXY[p][i - 1] + w * self.snapXY[p][i]
        for j in (i - 1, i):
            if 0 <= j < len(t) and abs(t[j] - N) <= 0.5 * FPS:
                return self.snapXY[p][j]
        return None

    def candidates(self, N):
        """Detections usable for association: height fits a person, feet within RINK_SOFT of the boards. Returns dict of arrays or None."""
        r = self.DT.evaluate(N)
        if r is None:
            return None
        D, xy, occ, on_ice, fits, ratio = r
        k = N - self.DT.F0
        a = self.DT.start[k]
        ok = fits & (rink_dist(xy, self.hx, self.hy, self.c) > -RINK_SOFT)
        if not ok.any():
            return None
        P = self.PD[a : a + len(D)][ok]
        D = D[ok]
        return dict(
            xy=xy[ok],
            occ=occ[ok],
            conf=D[:, 5],
            wide=(D[:, 3] - D[:, 1]) > WIDE_RATIO * (D[:, 4] - D[:, 2]),
            ph=P[:, 0],
            pa=P[:, 1],
            pn=P[:, 2],
            feat=D[:, 6:12],
            ratio=ratio[ok],
            idx=np.flatnonzero(ok),
        )


def trim(s_, last):
    """Cut a finished span after its last supported frame."""
    n = last - s_["N0"] + 1
    return {k: (v[:n] if isinstance(v, list) and k != "feat" else v) for k, v in s_.items()}


def smooth(s_):
    xf, Pf, xp, Pp = (np.array(s_[k]) for k in ("xf", "Pf", "xp", "Pp"))
    xs = xf.copy()
    for i in range(len(xf) - 2, -1, -1):
        Cg = Pf[i] @ F_.T @ np.linalg.inv(Pp[i + 1])
        xs[i] = xf[i] + (xs[i + 1] - xp[i + 1]) @ Cg.T
    n = len(xf)
    return dict(
        N0=s_["N0"],
        pos=xs[:, :, 0],
        sup=np.array(s_["sup"]),
        margin=np.array(s_.get("margin", np.zeros(n))),
        dmeta=np.array(s_.get("dmeta", np.full(n, np.nan))),
        ph=s_.get("ph", 0.0),
        pa=s_.get("pa", 0.0),
        pn=s_.get("pn", 0.0),
        occ=s_.get("occ", 0.0),
        n=s_.get("n", 1),
        feat=np.median(np.array(s_["feat"]), axis=0) if "feat" in s_ else None,
    )


def run(parity=None, verbose=True):
    W = World(parity)
    npl = W.npl
    n0, n1 = W.n0, W.n1
    T = n1 - n0
    alive = np.zeros(npl, bool)
    x = np.zeros((npl, 2, 2))
    P = np.zeros((npl, 2, 2))
    last_sup = np.full(npl, -(10**6))
    spans = [None] * npl
    rec = {p: [] for p in range(npl)}  # per player: finished spans of per-frame records
    bad = np.zeros(npl, int)
    anon = []
    anon_rec = []  # unnamed player tracks (detections no roster player claimed)
    for N in range(n0, n1):
        act = [p for p in range(npl) if W.active(p, N)]
        for p in range(npl):
            if alive[p] and p not in act:
                rec[p].append(spans[p])
                alive[p] = False
                spans[p] = None
        for p in act:
            if not alive[p]:
                m = W.meta_at(p, N)
                if m is None:
                    j = np.searchsorted(W.snapN[p], N)
                    m = W.snapXY[p][min(j, len(W.snapN[p]) - 1)]
                x[p] = np.array([[m[0], 0.0], [m[1], 0.0]])
                P[p] = np.diag([R_META**2, 9.0])
                alive[p] = True
                last_sup[p] = -(10**6)
                bad[p] = 0
                spans[p] = dict(N0=N, xf=[], Pf=[], xp=[], Pp=[], sup=[], margin=[], dmeta=[], first=True)
        for p in act:
            if not spans[p]["first"]:
                x[p], P[p] = predict(x[p], P[p])
                if N - last_sup[p] > UNSUPPORTED_AFTER:
                    x[p, :, 1] *= 0.98
            spans[p]["xp"].append(x[p].copy())
            spans[p]["Pp"].append(P[p].copy())
            spans[p]["first"] = False
        sup = {p: 0 for p in act}
        margin = {p: 0.0 for p in act}
        dm = {p: np.nan for p in act}
        meta = {p: W.meta_at(p, N) for p in act}
        cand = W.candidates(N)
        used = set()
        if cand is not None and act:
            M = len(cand["xy"])
            K = len(act)
            C = np.full((K, M + K), INF)
            for a, p in enumerate(act):
                var = P[p][0, 0] + np.where(cand["occ"], R_OCC, R_DET) ** 2
                diff = cand["xy"] - x[p][:, 0]
                d2 = (diff**2).sum(1) / var
                c = d2 + 2 * np.log(var / R_DET**2)  # Gaussian likelihood: an uncertain state does not win by being vague
                if meta[p] is not None:
                    c += W_META * np.minimum(((cand["xy"] - meta[p]) ** 2).sum(1) / R_META**2, META_CAP)
                pt = cand["ph"] if W.team_p[p] == 1 else cand["pa"]
                tot = cand["ph"] + cand["pa"] + 1e-9  # probability of the OTHER team
                c = c + TEAM_PEN * pt / tot + REF_PEN * cand["pn"]
                c[d2 > GATE] = INF
                C[a, :M] = c
                C[a, M + a] = C_MISS
            r, cc = linear_sum_assignment(C)
            total = C[r, cc].sum()
            pairs = [(a, m) for a, m in zip(r, cc) if m < M and C[a, m] < INF / 2]
            for a, m in pairs:
                p = act[a]
                x[p], P[p] = update(x[p], P[p], cand["xy"][m], R_OCC if cand["occ"][m] else R_DET)
                sup[p] = 1
                last_sup[p] = N
                if meta[p] is not None:
                    dm[p] = np.linalg.norm(cand["xy"][m] - meta[p])
                C2 = C.copy()
                C2[a, m] = INF
                r2, c2 = linear_sum_assignment(C2)
                t2 = C2[r2, c2].sum()
                margin[p] = MARGIN_CAP if t2 >= INF / 2 else min(t2 - total, MARGIN_CAP)
            matched = {m: act[a] for a, m in pairs}
            claimed = set(act[a] for a, _ in pairs)
            used = set(matched)
            for a, p in enumerate(act):  # a wide box shared by two teammates supports both
                if p in claimed:
                    continue
                for m in np.flatnonzero(cand["wide"]):
                    q = matched.get(m)
                    if q is None or W.team_p[q] != W.team_p[p]:
                        continue
                    if np.linalg.norm(cand["xy"][m] - x[p][:, 0]) <= SHARE_PRIOR and (
                        meta[p] is None or np.linalg.norm(cand["xy"][m] - meta[p]) <= SHARE_META
                    ):
                        x[p], P[p] = update(x[p], P[p], cand["xy"][m], R_SHARE)
                        sup[p] = 2
                        last_sup[p] = N
                        used.add(m)
                        break
        for a_ in anon:
            if not a_["first"]:
                a_["x"], a_["P"] = predict(a_["x"], a_["P"])
                a_["x"][:, 1] *= 0.98 if N - a_["last"] > UNSUPPORTED_AFTER else 1.0
            a_["first"] = False
            a_["span"]["xp"].append(a_["x"].copy())
            a_["span"]["Pp"].append(a_["P"].copy())
            a_["sup"] = 0
        if cand is not None:
            left = [
                m for m in range(len(cand["xy"])) if m not in used and cand["pn"][m] < 0.5 and cand["conf"][m] >= ANON_KEEP_CONF
            ]
            if left and anon:
                K2 = len(anon)
                C2 = np.full((K2, len(left) + K2), INF)
                for i, a_ in enumerate(anon):
                    var = a_["P"][0, 0] + np.where(cand["occ"][left], R_OCC, R_DET) ** 2
                    d2 = ((cand["xy"][left] - a_["x"][:, 0]) ** 2).sum(1) / var
                    c = d2 + 2 * np.log(var / R_DET**2)
                    c[d2 > ANON_GATE] = INF
                    C2[i, : len(left)] = c
                    C2[i, len(left) + i] = C_MISS
                for i, j in zip(*linear_sum_assignment(C2)):
                    if j >= len(left) or C2[i, j] >= INF / 2:
                        continue
                    m = left[j]
                    a_ = anon[i]
                    a_["x"], a_["P"] = update(a_["x"], a_["P"], cand["xy"][m], R_OCC if cand["occ"][m] else R_DET)
                    a_["sup"] = 1
                    a_["last"] = N
                    used.add(m)
                    sp = a_["span"]
                    sp["ph"] += cand["ph"][m]
                    sp["pa"] += cand["pa"][m]
                    sp["pn"] += cand["pn"][m]
                    sp["occ"] += cand["occ"][m]
                    sp["n"] += 1
                    sp["feat"].append(cand["feat"][m])
            live = [x[p][:, 0] for p in range(npl) if alive[p]] + [a_["x"][:, 0] for a_ in anon]
            for m in range(len(cand["xy"])):
                if (
                    m in used
                    or cand["pn"][m] >= ANON_PN
                    or cand["occ"][m]
                    or cand["conf"][m] < ANON_CONF
                    or rink_dist(cand["xy"][m], W.hx, W.hy, W.c) < ANON_INSIDE
                ):
                    continue
                if live and min(np.linalg.norm(q - cand["xy"][m]) for q in live) < ANON_DUP:
                    continue
                sp = dict(
                    N0=N,
                    xf=[],
                    Pf=[],
                    xp=[np.array([[cand["xy"][m][0], 0.0], [cand["xy"][m][1], 0.0]])],
                    Pp=[np.diag([R_DET**2, 9.0])],
                    sup=[],
                    ph=float(cand["ph"][m]),
                    pa=float(cand["pa"][m]),
                    pn=float(cand["pn"][m]),
                    occ=0.0,
                    n=1,
                    feat=[cand["feat"][m]],
                )
                anon.append(dict(x=sp["xp"][0].copy(), P=sp["Pp"][0].copy(), first=True, last=N, sup=1, span=sp))
                live.append(cand["xy"][m])
                used.add(m)
        for a_ in list(anon):
            if a_["P"][0, 0] > P_CAP:
                a_["P"] = a_["P"] * (P_CAP / a_["P"][0, 0])
            s_ = a_["span"]
            s_["xf"].append(a_["x"].copy())
            s_["Pf"].append(a_["P"].copy())
            s_["sup"].append(a_["sup"])
            if N - a_["last"] > ANON_MISS:
                anon_rec.append(trim(s_, a_["last"]))
                anon = [o for o in anon if o is not a_]
        for p in act:
            if meta[p] is not None and alive[p]:
                bad[p] = bad[p] + 1 if np.linalg.norm(x[p][:, 0] - meta[p]) > RESET_D else 0
        reset = [p for p in act if bad[p] >= RESET_N]
        for p in act:
            if sup[p] == 0 and N - last_sup[p] > UNSUPPORTED_AFTER and meta[p] is not None:
                x[p], P[p] = update(x[p], P[p], meta[p], R_META)  # not seen: follow the metadata
            if P[p][0, 0] > P_CAP:
                P[p] = P[p] * (P_CAP / P[p][0, 0])
            s_ = spans[p]
            s_["xf"].append(x[p].copy())
            s_["Pf"].append(P[p].copy())
            s_["sup"].append(sup[p])
            s_["margin"].append(margin[p])
            s_["dmeta"].append(dm[p])
        for p in reset:
            rec[p].append(spans[p])
            spans[p] = None
            alive[p] = False
            bad[p] = 0
        if verbose and (N - n0) % 1500 == 0:
            print(f"  frame {N - n0}/{T}  active players {len(act)}", flush=True)
    for p in range(npl):
        if spans[p] is not None:
            rec[p].append(spans[p])
    out = {p: [smooth(s_) for s_ in rec[p]] for p in range(npl)}  # backward smoothing per span
    anon_out = [
        smooth(trim(s_, s_["N0"] + len(s_["sup"]) - 1 - int(np.argmax(np.array(s_["sup"])[::-1] > 0))))
        for s_ in anon_rec + [a_["span"] for a_ in anon]
        if any(v > 0 for v in s_["sup"])
    ]
    for s_ in anon_out:
        s_["role"] = W.model.predict_proba(roles.feature_matrix(s_["feat"][None]))[
            0
        ]  # one decision per track, from its median colours (what the classifier was trained on)
    return dict(tracks=out, anon=anon_out, n0=n0, n1=n1)


def verified(has, agree):
    """Per frame: the track is vouched for by the metadata on both sides (or by one side at a span edge)."""
    n = len(has)
    idx = np.flatnonzero(has)
    ok = np.zeros(n, bool)
    if len(idx) == 0:
        return ok
    fr = np.arange(n)
    j = np.searchsorted(idx, fr, "right") - 1
    k = np.searchsorted(idx, fr, "left")
    for i in range(n):
        pv = idx[j[i]] if j[i] >= 0 else None
        nx = idx[k[i]] if k[i] < len(idx) else None
        if pv is not None and nx is not None:
            ok[i] = agree[pv] and agree[nx] and nx - pv <= NUM_GAP_S * FPS
        elif pv is not None:
            ok[i] = agree[pv] and i - pv <= NUM_EDGE_S * FPS
        else:
            ok[i] = agree[nx] and nx - i <= NUM_EDGE_S * FPS
    return ok


def anon_ok(s_):
    """An unnamed track is drawn if it was seen long enough, has visible feet, and its median colours say home or away."""
    return (s_["sup"] > 0).sum() >= ANON_MIN_SUP and s_["occ"] / s_["n"] <= ANON_MAX_OCC and s_["role"][2] <= ANON_ROLE_PN


def is_official(s_, state):
    """Referee test: weak team colour, and far from every tracked roster player on most frames where the roster is well tracked."""
    if max(s_["feat"][:2]) >= REF_PURITY:
        return False
    far = [
        min(np.linalg.norm(q - o) for o in state[N]) > REF_FAR
        for i, q in enumerate(s_["pos"])
        if s_["sup"][i] > 0 and len(state.get(N := s_["N0"] + i, [])) >= REF_STATES
    ]
    return len(far) >= REF_JUDGED and np.mean(far) >= REF_FRAC


def display(res, W):
    """{frame: [(player or -1-k for an unnamed track, ring position (inside the rink), number shown, shared box, team)]} of rings to draw."""
    byN = {}
    put = lambda N, e: byN.setdefault(N, []).append(e)
    for p, spans in res["tracks"].items():
        for s_ in spans:
            n = len(s_["sup"])
            sup = s_["sup"] > 0
            ring = sup.copy()
            idx = np.flatnonzero(sup)
            for a, b in zip(idx[:-1], idx[1:]):
                if 1 < b - a <= SUPPORT_GAP:
                    ring[a : b + 1] = True  # short detector miss: smoothed path bridges it
            cert = (s_["sup"] == 1) & (s_["margin"] >= MARGIN_MIN) & (np.isnan(s_["dmeta"]) | (s_["dmeta"] <= META_OK))
            cert = fill_gaps(cert, SUPPORT_GAP)
            cert = drop_short(cert, 5) & ring  # the number stays through a short detector miss
            meta = [W.meta_at(p, s_["N0"] + i) for i in range(n)]
            has = np.array([m is not None for m in meta])
            agree = np.array([m is not None and np.linalg.norm(q - m) <= META_NUM for q, m in zip(s_["pos"], meta)])
            cert &= verified(has, agree)
            pos = np.array([clamp_inside(q, W.hx, W.hy, W.c) for q in s_["pos"]])
            for i in np.flatnonzero(ring):
                put(s_["N0"] + i, (p, pos[i], bool(cert[i]), s_["sup"][i] == 2, int(W.team_p[p])))
    state = {}
    for spans in res["tracks"].values():
        for s_ in spans:
            for i, q in enumerate(s_["pos"]):
                state.setdefault(s_["N0"] + i, []).append(q)
    for k, s_ in enumerate(res.get("anon", [])):
        sup = s_["sup"] > 0
        if not anon_ok(s_) or is_official(s_, state):
            continue
        ring = sup.copy()
        idx = np.flatnonzero(sup)
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= ANON_BRIDGE:
                ring[a : b + 1] = True
        team = int(s_["role"][1] > s_["role"][0])
        for i in np.flatnonzero(ring):
            put(s_["N0"] + i, (-1 - k, clamp_inside(s_["pos"][i], W.hx, W.hy, W.c), False, False, team))
    return byN


def fill_gaps(b, gap):
    b = b.copy()
    idx = np.flatnonzero(b)
    for a, c in zip(idx[:-1], idx[1:]):
        if 1 < c - a <= gap + 1:
            b[a : c + 1] = True
    return b


def drop_short(b, n):
    b = b.copy()
    i = 0
    while i < len(b):
        if b[i]:
            j = i
            while j < len(b) and b[j]:
                j += 1
            if j - i < n:
                b[i:j] = False
            i = j
        else:
            i += 1
    return b


def frame_rings(show, W, N):
    """[(player or negative for an unnamed track, xy, number shown, team)] visible at frame N."""
    es = show.get(N, [])
    named = [e for e in es if e[0] >= 0]
    keep = [
        e for e in named if not (e[3] and any(o is not e and np.linalg.norm(o[1] - e[1]) < SHARE_MIN_SEP for o in named))
    ]  # a shared-box ring sitting on another ring is the same person
    for e in sorted((e for e in es if e[0] < 0), key=lambda e: -e[0]):
        if all(np.linalg.norm(o[1] - e[1]) >= ANON_DUP for o in keep):
            keep.append(e)
    return [(p, xy, num, team) for p, xy, num, _, team in keep]


def main_run(parity=None):
    res = run(parity)
    name = "" if parity is None else f"_odd{parity}"
    pickle.dump(res, open(CACHE.format(name), "wb"))
    print("saved", CACHE.format(name))


def evaluate(parity=None, windows=range(int(WIN["t0"]), int(WIN["t0"]) + 300, 20), verbose=True):
    """parity=None: all snapshots used by the tracker and for the check (optimistic). parity=0/1: the tracker saw only that half, the check uses the other half (honest)."""
    W = World(parity)
    name = "" if parity is None else f"_odd{parity}"
    res = pickle.load(open(CACHE.format(name), "rb"))
    show = display(res, W)
    ev = snap_list(W.snaps, W.n0, W.n1, None if parity is None else 1 - parity)
    DT = W.DT
    bins = [(-9, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 99)]
    bc = {b: np.zeros(3) for b in bins}
    tot = np.zeros(6)
    print(f"check snapshots: {len(ev)} ({'held-out half' if parity is not None else 'same as tracker input'})")
    if verbose:
        print("window      | in-view players: ring | right number | numbered but misplaced | rings on nobody")
    for f0 in windows:
        row = np.zeros(6)
        for N, d in ev:
            if not f0 * FPS <= N < (f0 + 20) * FPS:
                continue
            k = DT.rowN.get(N)
            if k is None:
                continue
            rings = frame_rings(show, W, N)
            meta = {p: W.corr(m, p) for p, m in d.items()}
            for p, c in meta.items():
                px = S.project(DT.g, DT.Z["f"][k], DT.Z["rv"][k], np.array([[*c, 0.0]]))[0]
                if not (60 < px[0] < 1860 and 60 < px[1] < 1020):
                    continue
                row[0] += 1
                has = any(np.linalg.norm(xy - c) < 2.5 for _, xy, _, _ in rings)
                good = any(q == p and num and np.linalg.norm(xy - c) < 2.5 for q, xy, num, _ in rings)
                row[1] += has
                row[2] += good
                b = rink_dist(c, W.hx, W.hy, W.c)
                bk = [x for x in bins if x[0] <= b < x[1]][0]
                bc[bk] += (1, has, good)
            for q, xy, num, _ in rings:
                if num and q in meta:
                    row[3] += 1
                    row[4] += np.linalg.norm(xy - meta[q]) >= 2.5
            if len(d) >= 11:
                row[5] += sum(1 for q, xy, _, _ in rings if q >= 0 and min(np.linalg.norm(xy - c) for c in meta.values()) > 3.0)
        tot += row
        if verbose:
            print(
                f"{f0}-{f0 + 20}s  | {int(row[0]):4d}  {100 * row[1] / max(row[0], 1):5.1f}%   {100 * row[2] / max(row[0], 1):5.1f}%     {int(row[4]):3d} of {int(row[3]):4d}              {int(row[5]):3d}"
            )
    if not verbose:
        return tot, bc
    print(
        f"all windows: rings {100 * tot[1] / tot[0]:.1f}%, right numbers {100 * tot[2] / tot[0]:.1f}% of {int(tot[0])} in-view players; numbered but misplaced {int(tot[4])} of {int(tot[3])} ({100 * tot[4] / max(tot[3], 1):.2f}%); rings on nobody {int(tot[5])}"
    )
    print(
        "by distance to boards:  "
        + "  ".join(
            f"{b[0]}..{b[1]}m: n={int(v[0])} ring {100 * v[1] / max(v[0], 1):.0f}% number {100 * v[2] / max(v[0], 1):.0f}%"
            for b, v in bc.items()
        )
    )


def render(t0, dur, prefix):
    W = World()
    res = pickle.load(open(CACHE.format(""), "rb"))
    show = display(res, W)
    DT = W.DT
    f0, f1 = int(round(t0 * FPS)), int(round((t0 + dur) * FPS))
    w = subprocess.Popen(
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
            "1920x1080",
            "-r",
            "25",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "22",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            f"{prefix}.mp4",
        ],
        stdin=subprocess.PIPE,
    )
    last_k = DT.rowN[min(DT.rowN, key=lambda n: abs(n - f0))]
    for N in range(f0, f1):
        img = cv2.imread(f"{WIN['frames']}/{N - DT.F0:05d}.jpg")
        kk = DT.rowN.get(N, last_k)
        last_k = kk
        f, rv = DT.Z["f"][kk], DT.Z["rv"][kk]
        rings = [
            (xy, team, (("G" if W.goalie[p] else "") + str(W.roster[p]["jersey"])) if num else None)
            for p, xy, num, team in frame_rings(show, W, N)
        ]
        out = G.draw_scene(img, DT.g, f, rv, rings)
        txt = f"t={N / FPS:.1f}s  player-centric tracker: ring = player, number = unambiguous identity"
        cv2.putText(out, txt, (30, 60), 0, 1.1, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(out, txt, (30, 60), 0, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
        w.stdin.write(out.tobytes())
    w.stdin.close()
    w.wait()
    print("wrote", f"{prefix}.mp4")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        main_run(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "eval":
        evaluate(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "render":
        render(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
