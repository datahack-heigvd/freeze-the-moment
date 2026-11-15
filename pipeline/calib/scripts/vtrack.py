"""Vision-first tracker: positions come from person detections, the metadata only gives identity.
1) detections -> tracklets (25 fps Kalman + Hungarian, RTS smoothed)
2) tracklet identity = posterior over the roster given noisy snapshots (corrected by an affine + per-player offset fitted by EM),
   same-team prior, no two simultaneous tracklets share a player; assigned only when certain, candidates shrink by elimination.
usage: vtrack.py <t0> <dur> <out_prefix>"""

import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
import selfcal as S
from make_preview import draw_players
from make_preview import track as old_track
from opt1 import load_snaps
from opt2 import Detections, team_of
from scipy.optimize import linear_sum_assignment
from window import W as WIN

FEAT = [3, 4, 5, 6, 8, 9]  # detection columns: yellow, blue, dark, white, saturation, value (torso colour)
FPS = 25.0
DT_ = 1.0 / FPS
F_ = np.array([[1, DT_], [0, 1]])
Q_ = 6.0
R_ = 0.3
MAXMISS = 15
MINMEAS = 10
MINCONF = 0.25
QM = Q_**2 * np.array([[DT_**4 / 4, DT_**3 / 2], [DT_**3 / 2, DT_**2]])
SIGMA, CAP, ABSENT, NULL, TEMPER, THR = 0.8, 8.0, 1.5, 3.0, 0.35, 0.9
MATCH_LL = -3.0  # a snapshot 'matches' a tracklet when its log-likelihood is above this (about 2.2 m)
DUP_DIST = 1.2
SNAP_DT = 0.0  # seconds added to snapshot timestamps to line them up with the video
HANDOVER = 20  # frames two fragments of the same person may overlap (lost and re-found tracks) before they count as two people
NONE_LL = 2.0  # per-snapshot log-likelihood of 'this tracklet is nobody on the roster'


def team_of_col(col):
    y, b = col
    return 1 if (y > 0.22 and y > b) else (0 if (b > 0.22 and b > y) else -1)


def colour_cuts(colf, win=25, minrun=20):
    """Frame indices where the jersey colour of a tracklet switches team for good (the track jumped to a person of the other team)."""
    ok = np.isfinite(colf[:, 0])
    c = np.where(ok[:, None], colf, 0.0)
    cs = np.vstack([np.zeros(2), np.cumsum(c, 0)])
    ns = np.r_[0, np.cumsum(ok)]
    lo = np.clip(np.arange(len(colf)) - win // 2, 0, len(colf))
    hi = np.clip(np.arange(len(colf)) + win // 2 + 1, 0, len(colf))
    cnt = ns[hi] - ns[lo]
    lab = np.array([team_of_col((cs[h] - cs[l]) / n) if n else -1 for l, h, n in zip(lo, hi, cnt)])
    runs, st = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[st]:
            runs.append((st, i, lab[st]))
            st = i
    long = [r for r in runs if r[2] >= 0 and r[1] - r[0] >= minrun]
    return [(a[1] + b[0]) // 2 for a, b in zip(long[:-1], long[1:]) if a[2] != b[2]]


def team_of_tr(t):
    """0 home, 1 away, -1 unknown, -2 not a player (referee, staff). Set by the role classifier, jersey colour until then."""
    return t["team"] if "team" in t else team_of_col(t["col"])


def build_tracklets(DT, n0, n1):
    active, done = [], []

    def finish(tr):
        k = max(i for i, m in enumerate(tr["meas"]) if m) + 1
        if sum(tr["meas"]) < MINMEAS or np.mean(tr["conf"]) < MINCONF:
            return
        xf, Pf, xp, Pp = (np.array(tr[n][:k]) for n in ("xf", "Pf", "xp", "Pp"))
        xs = xf.copy()
        for i in range(k - 2, -1, -1):
            C = Pf[i] @ F_.T @ np.linalg.inv(Pp[i + 1])
            xs[i] = xf[i] + (xs[i + 1] - xp[i + 1]) @ C.T
        colf = np.array([c if c is not None else np.full(len(FEAT), np.nan) for c in tr["colf"][:k]], float)
        meas = np.array(tr["meas"][:k])
        edges = [0] + colour_cuts(colf[:, :2]) + [k]
        for e0, e1 in zip(edges[:-1], edges[1:]):
            if meas[e0:e1].sum() < MINMEAS:
                continue
            feat = np.nanmedian(colf[e0:e1], 0)
            done.append(dict(a=tr["a"] + e0, b=tr["a"] + e1 - 1, pos=xs[e0:e1, :, 0], meas=meas[e0:e1], col=feat[:2], feat=feat))

    for N in range(n0, n1):
        det = DT.at_frame(N)
        for tr in active:
            tr["x"] = tr["x"] @ F_.T
            tr["P"] = F_ @ tr["P"] @ F_.T + QM
            tr["xp"].append(tr["x"].copy())
            tr["Pp"].append(tr["P"].copy())
            tr["hit"] = False
        used = np.zeros(len(det), bool)
        if active and len(det):
            pred = np.array([tr["x"][:, 0] for tr in active])
            var = np.array([tr["P"][0, 0] for tr in active]) + R_**2
            dist2 = ((pred[:, None] - det[None, :, :2]) ** 2).sum(2)
            cost = dist2 / var[:, None]
            d2 = cost.copy()
            dteam = team_of(det)
            for a, tr in enumerate(active):
                tt = team_of_col(tr["col"])
                cost[a, (tt >= 0) & (dteam >= 0) & (dteam != tt)] += 6.0
            cost[(d2 > 9) | (dist2 > 2.5**2)] = 1e6
            for a, c in zip(*linear_sum_assignment(cost)):
                if cost[a, c] >= 1e5:
                    continue
                tr = active[a]
                z = det[c, :2]
                used[c] = True
                tr["hit"] = True
                S_ = tr["P"][0, 0] + R_**2
                K = tr["P"][:, 0] / S_
                for ax in range(2):
                    tr["x"][ax] = tr["x"][ax] + K * (z[ax] - tr["x"][ax, 0])
                tr["P"] = tr["P"] - np.outer(K, tr["P"][0])
                tr["conf"].append(det[c, 2])
                tr["col"] = 0.9 * tr["col"] + 0.1 * det[c, 3:5]
                tr["cur"] = det[c, FEAT].copy()
        still = []
        for tr in active:
            tr["meas"].append(tr["hit"])
            tr["xf"].append(tr["x"].copy())
            tr["Pf"].append(tr["P"].copy())
            tr["colf"].append(tr.pop("cur", None))
            miss = 0
            for m in reversed(tr["meas"]):
                if m:
                    break
                miss += 1
            if miss > MAXMISS:
                finish(tr)
            else:
                still.append(tr)
        active = still
        apos = [tr["x"][:, 0] for tr in active]
        for c in np.flatnonzero(~used):
            if apos and (((np.array(apos) - det[c, :2]) ** 2).sum(1) < 0.6**2).any():
                continue
            apos.append(det[c, :2])
            x0 = np.array([[det[c, 0], 0.0], [det[c, 1], 0.0]])
            P0 = np.diag([R_**2, 9.0])
            active.append(
                dict(
                    a=N,
                    x=x0,
                    P=P0,
                    meas=[True],
                    conf=[det[c, 2]],
                    col=det[c, 3:5].copy(),
                    colf=[det[c, FEAT].copy()],
                    xf=[x0.copy()],
                    Pf=[P0.copy()],
                    xp=[x0.copy()],
                    Pp=[P0.copy()],
                    hit=True,
                )
            )
    for tr in active:
        finish(tr)
    return done


# ---------------------------------------------------------------- identity
class Corr:
    """metadata -> video frame correction: affine + shrunk per-player offset."""

    def __init__(self, n_players):
        self.A = np.eye(2)
        self.b = np.zeros(2)
        self.pb = np.zeros((n_players, 2))

    def __call__(self, xy, p):
        return self.A @ xy + self.b + self.pb[p]


def snap_list(snaps, n0, n1, parity=None):
    out = []
    for i, (t, d) in enumerate(snaps):
        N = int(round((t + SNAP_DT) * FPS))
        if n0 - 4 <= N <= n1 + 4 and (parity is None or i % 2 == parity):
            out.append((N, d))
    return out


def score_tracklets(tracks, sl, corr, npl):
    nT = len(tracks)
    Sc = np.zeros((nT, npl))
    Nm = np.zeros((nT, npl), int)
    ns = np.zeros(nT)
    pairs = []
    for ti, tr in enumerate(tracks):
        for N, d in sl:
            if not (tr["a"] - 4 <= N <= tr["b"] + 4):
                continue
            xy = tr["pos"][int(np.clip(N - tr["a"], 0, len(tr["pos"]) - 1))]
            row = np.full(npl, -ABSENT)
            ns[ti] += 1
            for p, m in d.items():
                dd = np.linalg.norm(xy - corr(m, p))
                row[p] = -min(dd * dd / (2 * SIGMA**2), CAP)
                pairs.append((ti, p, xy, m))
                Nm[ti, p] += row[p] > MATCH_LL
            Sc[ti] += row
    return Sc, ns, Nm, pairs


def assign(tracks, Sc, ns, Nm, team_p, thr=THR, goalie=None):
    """Greedy posterior-driven assignment. A player cannot be in two time-overlapping tracklets; a tracklet whose same-team candidates shrink to one becomes certain."""
    nT, npl = Sc.shape
    ident = np.full(nT, -1)
    conf = np.zeros(nT)
    a = np.array([t["a"] for t in tracks])
    b = np.array([t["b"] for t in tracks])
    ttm = np.array([team_of_tr(t) for t in tracks])
    overlap = np.zeros(
        (nT, nT), bool
    )  # two tracklets conflict only if they coexist in different places (same-person duplicates may overlap)
    for i in range(nT):
        for j in range(i + 1, nT):
            lo, hi = max(a[i], a[j]), min(b[i], b[j])
            if hi - lo >= HANDOVER:
                overlap[i, j] = overlap[j, i] = (
                    np.linalg.norm(
                        tracks[i]["pos"][lo - a[i] : hi - a[i] + 1] - tracks[j]["pos"][lo - a[j] : hi - a[j] + 1], axis=1
                    ).mean()
                    > DUP_DIST
                )
    taken = [[] for _ in range(npl)]
    allowed = np.ones((nT, npl), bool)
    for ti in range(nT):
        if ttm[ti] >= 0:
            allowed[ti] = (team_p == ttm[ti]) | (goalie if goalie is not None else False)
        if ttm[ti] == -2:
            allowed[ti] = False
        allowed[ti] &= Nm[ti] >= min(2, ns[ti])
    while True:
        best = (0.0, -1, -1)
        for ti in range(nT):
            if ident[ti] >= 0 or ns[ti] == 0:
                continue
            ok = allowed[ti].copy()
            for p in range(npl):
                if ok[p] and any(overlap[ti, tj] for tj in taken[p]):
                    ok[p] = False
            if not ok.any():
                continue
            s = np.where(ok, Sc[ti], -np.inf)
            z = np.r_[s, -NULL * ns[ti]] * TEMPER
            z = np.exp(z - z.max())
            post = z / z.sum()
            k = int(np.argmax(post[:-1]))
            if post[k] > best[0]:
                best = (post[k], ti, k)
        if best[1] < 0 or best[0] < thr:
            break
        c, ti, p = best
        ident[ti] = p
        conf[ti] = c
        taken[p].append(ti)
    return ident, conf


def fit_corr(tracks, ident, pairs, corr, npl, shrink=8.0):
    sel = [(ti, p, xy, m) for ti, p, xy, m in pairs if ident[ti] == p]
    if len(sel) < 30:
        return corr
    M = np.array([m for _, _, _, m in sel])
    X = np.array([xy for _, _, xy, _ in sel])
    for _ in range(3):
        res = np.array([np.linalg.norm(X[i] - corr(M[i], sel[i][1])) for i in range(len(sel))])
        w = res < 2.5
        D = np.c_[M[w], np.ones(w.sum())]
        cx = np.linalg.lstsq(D, X[w, 0] - corr.pb[[s[1] for s, ww in zip(sel, w) if ww], 0], rcond=None)[0]
        cy = np.linalg.lstsq(D, X[w, 1] - corr.pb[[s[1] for s, ww in zip(sel, w) if ww], 1], rcond=None)[0]
        corr.A = np.array([cx[:2], cy[:2]])
        corr.b = np.array([cx[2], cy[2]])
        res_v = np.array([X[i] - (corr.A @ M[i] + corr.b) for i in range(len(sel))])
        pl = np.array([s[1] for s in sel])
        for p in range(npl):
            m = (pl == p) & w
            corr.pb[p] = res_v[m].sum(0) / (m.sum() + shrink) if m.any() else 0.0
    return corr


def split_tracklets(tracks, sl, corr, team_p, lam=8.0, goalie=None):
    """A tracklet can change who it follows: two players cross, or a bench person walks onto the ice and the track continues as the skater.
    Viterbi over the snapshots inside a tracklet (states: players seen at least twice, plus 'nobody'; switch penalty lam nats) finds the changes;
    the tracklet is cut where the new identity joins (or leaves) the old one."""
    npl = len(team_p)
    NONE = npl
    out = []
    ncut = 0
    for tr in tracks:
        obs = [(N, d) for N, d in sl if tr["a"] - 4 <= N <= tr["b"] + 4]
        if len(obs) < 3:
            out.append(tr)
            continue
        L = np.full((len(obs), npl + 1), -ABSENT)
        L[:, NONE] = -NONE_LL
        pres = np.zeros(npl, int)
        at = lambda N: tr["pos"][int(np.clip(N - tr["a"], 0, len(tr["pos"]) - 1))]
        for k, (N, d) in enumerate(obs):
            xy = at(N)
            for p, m in d.items():
                L[k, p] = -min(np.sum((xy - corr(m, p)) ** 2) / (2 * SIGMA**2), CAP)
                pres[p] += 1
        tm = team_of_tr(tr)
        if tm == -2:
            out.append(tr)
            continue
        L[:, np.flatnonzero((pres < 2) | ((team_p != tm) & (tm >= 0) & ~(goalie if goalie is not None else False)))] = -1e3
        dp = np.zeros_like(L)
        bp = np.zeros(L.shape, int)
        dp[0] = L[0]
        for k in range(1, len(obs)):
            top = int(np.argmax(dp[k - 1]))
            stay = dp[k - 1]
            sw = dp[k - 1, top] - lam
            bp[k] = np.where(stay >= sw, np.arange(npl + 1), top)
            dp[k] = L[k] + np.maximum(stay, sw)
        path = np.zeros(len(obs), int)
        path[-1] = int(np.argmax(dp[-1]))
        for k in range(len(obs) - 1, 0, -1):
            path[k - 1] = bp[k, path[k]]
        good = lambda k, st: st == NONE or (st in obs[k][1] and L[k, st] > MATCH_LL)
        cuts = []
        for k in range(1, len(obs)):
            old, new = path[k - 1], path[k]
            if old == new or not (good(k - 1, old) and good(k, new)) or (old == NONE and new == NONE):
                continue
            (N0_, d0), (N1_, d1) = obs[k - 1], obs[k]
            c = (N0_ + N1_) // 2
            if N1_ > N0_:
                fr = np.arange(N0_, N1_ + 1)
                w = (fr - N0_) / (N1_ - N0_)
                if (
                    old != NONE and new != NONE and all(q in dd for q in (old, new) for dd in (d0, d1))
                ):  # two players: where they are closest
                    gap = [
                        np.linalg.norm(
                            (1 - ww) * (corr(d0[old], old) - corr(d0[new], new)) + ww * (corr(d1[old], old) - corr(d1[new], new))
                        )
                        for ww in w
                    ]
                    c = int(fr[int(np.argmin(gap))])
                elif old == NONE or new == NONE:  # nobody <-> player: where the track meets the player's path
                    q = new if old == NONE else old
                    if q in d0 and q in d1:
                        dist = np.array(
                            [np.linalg.norm(at(f_) - ((1 - ww) * corr(d0[q], q) + ww * corr(d1[q], q))) for f_, ww in zip(fr, w)]
                        )
                        near = np.flatnonzero(dist < 2.0)
                        if len(near):
                            c = int(fr[near[0]]) if old == NONE else int(fr[near[-1]]) + 1
            cuts.append(c)
        edges = [tr["a"]] + [c for c in sorted(set(cuts)) if tr["a"] + 5 < c < tr["b"] - 5] + [tr["b"] + 1]
        ncut += len(edges) - 2
        for e0, e1 in zip(edges[:-1], edges[1:]):
            if e1 - e0 < 8:
                continue
            out.append(
                {
                    **tr,
                    "a": e0,
                    "b": e1 - 1,
                    "pos": tr["pos"][e0 - tr["a"] : e1 - tr["a"]],
                    "meas": tr["meas"][e0 - tr["a"] : e1 - tr["a"]],
                }
            )
    return out, ncut


def identify(tracks, snaps, team_p, n0, n1, parity=None, iters=3, verbose=True, corr=None, goalie=None):
    npl = len(team_p)
    corr = corr or Corr(npl)
    sl = snap_list(snaps, n0, n1, parity)
    for it in range(iters):
        Sc, ns, Nm, pairs = score_tracklets(tracks, sl, corr, npl)
        ident, conf = assign(tracks, Sc, ns, Nm, team_p, goalie=goalie)
        r = [np.linalg.norm(xy - corr(m, p)) for ti, p, xy, m in pairs if ident[ti] == p]
        if verbose:
            print(
                f"  EM iter {it}: identified {int((ident >= 0).sum())}/{len(tracks)} tracklets, median |res| {np.median(r):.2f} m (n={len(r)})",
                flush=True,
            )
        if it < iters - 1:
            corr = fit_corr(tracks, ident, pairs, corr, npl)
    return ident, conf, corr


def identify_split(tracks, snaps, team_p, n0, n1, parity=None, verbose=True, goalie=None):
    """identify -> cut tracklets at identity switches -> identify again on the pieces (correction carried over)."""
    _, _, corr = identify(tracks, snaps, team_p, n0, n1, parity, iters=2, verbose=verbose, goalie=goalie)
    pieces, ncut = split_tracklets(tracks, snap_list(snaps, n0, n1, parity), corr, team_p, goalie=goalie)
    if verbose:
        print(f"  split: {ncut} identity switches cut -> {len(pieces)} pieces", flush=True)
    ident, conf, corr = identify(pieces, snaps, team_p, n0, n1, parity, iters=3, verbose=verbose, corr=corr, goalie=goalie)
    return pieces, ident, conf, corr


# ---------------------------------------------------------------- render
def per_identity(tracks, ident, GAP=25):
    """Concatenate tracklets of the same player; short gaps are bridged linearly (flag 1), tracklet frames flag 2."""
    out = {}
    for p in sorted(set(int(i) for i in ident if i >= 0)):
        ts = sorted([tracks[i] for i in range(len(tracks)) if ident[i] == p], key=lambda t: t["a"])
        d = {}
        for k, tr in enumerate(ts):
            for j, xy in enumerate(tr["pos"]):
                d[tr["a"] + j] = (xy, 2)
            if k + 1 < len(ts):
                nx = ts[k + 1]
                gap = nx["a"] - tr["b"]
                if 1 < gap <= GAP and np.linalg.norm(nx["pos"][0] - tr["pos"][-1]) <= 1.0 + 8.0 * gap / FPS:
                    for g_ in range(1, gap):
                        d[tr["b"] + g_] = (tr["pos"][-1] + (nx["pos"][0] - tr["pos"][-1]) * g_ / gap, 1)
        out[p] = d
    return out


def render(t0, dur, prefix):
    g, snaps = load_snaps()
    DT = Detections()
    roster = g["roster"]
    npl = len(roster)
    team_p = np.array([0 if r["team"] == "Home" else 1 for r in roster])
    team = {i: int(team_p[i]) for i in range(npl)}
    goalie = np.array([bool(r.get("goalie")) for r in roster])
    n0, n1 = DT.N0, DT.N0 + WIN["n"]
    print("tracklets ...")
    tracks = build_tracklets(DT, n0, n1)
    raw = tracks
    print(" ", len(tracks), "tracklets; mean length %.1f s" % (np.mean([t["b"] - t["a"] + 1 for t in tracks]) / FPS), flush=True)
    print("identity (all snapshots):")
    tracks, ident, conf, corr = identify_split(tracks, snaps, team_p, n0, n1, goalie=goalie)
    print(
        f"  affine A={np.round(corr.A, 3).tolist()} b={np.round(corr.b, 2).tolist()} ; per-player offset spread {corr.pb[np.abs(corr.pb).sum(1) > 0].std():.2f} m"
    )
    # self-consistency: identities from even vs odd snapshots only
    pe, ie, _, _ = identify_split(raw, snaps, team_p, n0, n1, parity=0, verbose=False, goalie=goalie)
    po, io, _, _ = identify_split(raw, snaps, team_p, n0, n1, parity=1, verbose=False, goalie=goalie)
    agree = tot_ = 0
    for N in range(n0, n1, 5):
        a_ = {int(i) for i in [ie[j] for j, t in enumerate(pe) if t["a"] <= N <= t["b"] and ie[j] >= 0]}
        b_ = {int(i) for i in [io[j] for j, t in enumerate(po) if t["a"] <= N <= t["b"] and io[j] >= 0]}
        agree += len(a_ & b_)
        tot_ += len(a_ | b_)
    print("  even/odd snapshot split: identity sets per frame agree %.1f%% (|A and B| / |A or B|)" % (100 * agree / max(tot_, 1)))
    _, _, _, pr = score_tracklets(tracks, snap_list(snaps, n0, n1), corr, npl)
    for name, m in (("near boards y<-11", lambda y: y < -11), ("rest", lambda y: y >= -11)):
        r = [np.linalg.norm(xy - corr(mm, p)) for ti, p, xy, mm in pr if ident[ti] == p and m(xy[1])]
        print(f"  residual vs metadata, {name}: median {np.median(r):.2f} m (n={len(r)})")
    ln = np.array([t["b"] - t["a"] + 1 for t in tracks])
    tm = np.array([team_of_col(t["col"]) for t in tracks])
    print(
        f"  identified share of tracked person-time: team-known {100 * ln[(ident >= 0) & (tm >= 0)].sum() / ln[tm >= 0].sum():.0f}%, all {100 * ln[ident >= 0].sum() / ln.sum():.0f}%"
    )
    ids = per_identity(tracks, ident)
    f0, f1 = int(round(t0 * FPS)), int(round((t0 + dur) * FPS))
    players = sorted(ids)
    K = len(players)
    T = f1 - f0
    pos = np.full((T, K, 2), np.nan)
    lab = np.zeros((T, K), int)
    for k, p in enumerate(players):
        for N, (xy, fl) in ids[p].items():
            if f0 <= N < f1:
                pos[N - f0, k] = xy
                lab[N - f0, k] = fl
    act = np.isfinite(pos[:, :, 0])
    pos = np.nan_to_num(pos)
    # per-frame coverage vs snapshots in the clip: in-view snapshot players with an identified ring within 2.5 m of corrected metadata
    hit = tot = 0
    for N, d in snap_list(snaps, f0, f1):
        if not (f0 <= N < f1):
            continue
        kk = DT.rowN[N]
        for p, m in d.items():
            px = S.project(DT.g, DT.Z["f"][kk], DT.Z["rv"][kk], np.array([[*corr(m, p), 0.0]]))[0]
            if not (60 < px[0] < 1860 and 60 < px[1] < 1020):
                continue
            tot += 1
            if (
                p in players
                and act[N - f0, players.index(p)]
                and np.linalg.norm(pos[N - f0, players.index(p)] - corr(m, p)) < 2.5
            ):
                hit += 1
    print(
        f"  clip coverage: {hit}/{tot} in-view snapshot players have a ring within 2.5 m of their (corrected) metadata position ({100 * hit / max(tot, 1):.0f}%)"
    )
    fa, pa, la, aa, pl_old = old_track(snaps, DT, team, t0, t0 + dur, True)

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

    w1 = writer(f"{prefix}_vision.mp4", 1920, 1080)
    w2 = writer(f"{prefix}_compare2.mp4", 1920, 540)
    for N in range(f0, f1):
        img = cv2.imread(f"{WIN['frames']}/{N - DT.F0:05d}.jpg")
        kk = DT.rowN[N]
        f, rv = DT.Z["f"][kk], DT.Z["rv"][kk]
        new = draw_players(
            img, DT.g, f, rv, pos[N - f0], lab[N - f0], act[N - f0], players, roster, "vision-first (identity only when certain)"
        )
        for i, tr in enumerate(tracks):  # unidentified people: small grey dot at the detected foot point
            if ident[i] < 0 and tr["a"] <= N <= tr["b"] and tr["meas"][N - tr["a"]]:
                q = S.project(DT.g, f, rv, np.array([[*tr["pos"][N - tr["a"]], 0.0]]))[0]
                if 0 < q[0] < 1920 and 0 < q[1] < 1080:
                    cv2.circle(new, (int(q[0]), int(q[1])), 6, (170, 170, 170), -1, cv2.LINE_AA)
        fi = int(np.argmin(abs(fa - N)))
        old = draw_players(img, DT.g, f, rv, pa[fi], la[fi], aa[fi], pl_old, roster, "previous: snapshots (0.1 m) + vision")
        w1.stdin.write(new.tobytes())
        w2.stdin.write(np.hstack([cv2.resize(old, (960, 540)), cv2.resize(new, (960, 540))]).tobytes())
    for w in (w1, w2):
        w.stdin.close()
        w.wait()
    print("wrote", prefix)


if __name__ == "__main__":
    render(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
