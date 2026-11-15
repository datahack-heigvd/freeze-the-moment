"""Players on the ice over the whole detected range: a ring for every player, a number only when the identity is certain.
Rings come from detections (tracklets) classified as home / away / not a player (referees, staff). Identities spread from certain anchors through the whole 300 s.
Evidence for 'tracklet T is player P':
  - snapshots inside T and just outside it (variance grows with the time distance), corrected by the fitted metadata->video map
  - continuity with the nearest already-identified tracklets of P before and after T (position vs gap time)
  - same team (goalies exempt), P on the ice near that time (seen in a snapshot within POOL_S), no simultaneous different place
Rounds repeat until nothing new becomes certain; elimination happens through the conflict rule and the shrinking pool.
usage: gtrack.py build | eval | render <t0> <dur> <out_prefix>"""

import pickle
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
import roles
from vtrack import *

EXT_S, Q_EXT, POOL_S = (
    4.0,
    5.0,
    15.0,
)  # Q_EXT: m/s of drift allowed when a snapshot just outside a tracklet is compared with its end (skaters reach 8 m/s)
LINK_SIG, LINK_V, LINK_CAP, LINK_NONE, LINK_NULL, GMAX = 0.7, 2.5, 6.0, 1.5, 3.0, 20.0
from window import W as WIN

CACHE = WIN["gtrack"]


def evidence(tracks, sl, corr, npl):
    """Snapshot evidence per (tracklet, player), including snapshots up to EXT_S outside the tracklet."""
    nT = len(tracks)
    Sc = np.zeros((nT, npl))
    ns = np.zeros(nT)
    Nsn = np.array([N for N, _ in sl])
    e = int(EXT_S * FPS)
    for ti, tr in enumerate(tracks):
        a, b = tr["a"], tr["b"]
        for k in range(np.searchsorted(Nsn, a - e), np.searchsorted(Nsn, b + e, side="right")):
            N, d = sl[k]
            if N < a:
                dt, xy = (a - N) / FPS, tr["pos"][0]
            elif N > b:
                dt, xy = (N - b) / FPS, tr["pos"][-1]
            else:
                dt, xy = 0.0, tr["pos"][N - a]
            s2 = SIGMA**2 + (Q_EXT * dt) ** 2
            row = np.full(npl, -ABSENT * (1.0 if dt == 0 else 0.5))
            ns[ti] += 1
            for p, m in d.items():
                row[p] = -min(np.sum((xy - corr(m, p)) ** 2) / (2 * s2), CAP)
            Sc[ti] += row
    return Sc, ns


def pool_matrix(tracks, sl, npl):
    times = [[] for _ in range(npl)]
    for N, d in sl:
        for p in d:
            times[p].append(N)
    times = [np.array(t) for t in times]
    pool = np.zeros((len(tracks), npl), bool)
    w = int(POOL_S * FPS)
    for ti, tr in enumerate(tracks):
        for p in range(npl):
            t = times[p]
            if len(t):
                j = np.searchsorted(t, tr["a"] - w)
                pool[ti, p] = j < len(t) and t[j] <= tr["b"] + w
    return pool


def conflicts(tracks):
    nT = len(tracks)
    a = np.array([t["a"] for t in tracks])
    b = np.array([t["b"] for t in tracks])
    cf = np.zeros((nT, nT), bool)
    for i in range(nT):
        for j in range(i + 1, nT):
            lo, hi = max(a[i], a[j]), min(b[i], b[j])
            if hi - lo >= HANDOVER:
                cf[i, j] = cf[j, i] = (
                    np.linalg.norm(
                        tracks[i]["pos"][lo - a[i] : hi - a[i] + 1] - tracks[j]["pos"][lo - a[j] : hi - a[j] + 1], axis=1
                    ).mean()
                    > DUP_DIST
                )
    return cf


def link_ll(A, B):
    """Log-likelihood that tracklet B (later) continues tracklet A (earlier)."""
    gap = (B["a"] - A["b"]) / FPS
    lo, hi = max(A["a"], B["a"]), min(A["b"], B["b"])
    if hi - lo >= 3:
        d = np.linalg.norm(A["pos"][lo - A["a"] : hi - A["a"] + 1] - B["pos"][lo - B["a"] : hi - B["a"] + 1], axis=1).mean()
        s2 = LINK_SIG**2
    else:
        d = np.linalg.norm(A["pos"][-1] - B["pos"][0])
        s2 = LINK_SIG**2 + (LINK_V * max(gap, 0.0)) ** 2
    return -min(d * d / (2 * s2), LINK_CAP)


def link_score(tracks, chain, ti):
    """Continuity with the tracklets already assigned to the player (chain: tracklet indices sorted by start): nearest before and after ti."""
    t = tracks[ti]
    pred = succ = None
    for j in chain:
        o = tracks[j]
        if o["a"] < t["a"] or (o["a"] == t["a"] and o["b"] < t["b"]):
            pred = j
        elif succ is None and o["a"] > t["a"]:
            succ = j
    s = 0.0
    for j, early in ((pred, True), (succ, False)):
        if j is None:
            s -= LINK_NONE
            continue
        A, B = (tracks[j], t) if early else (t, tracks[j])
        s += link_ll(A, B) if abs(B["a"] - A["b"]) / FPS <= GMAX else -LINK_NONE
    return s


def propagate(tracks, ident, conf, sl, corr, team_p, goalie, verbose=True, max_rounds=25, thr=THR, extra=None, pool=None):
    """extra: (tracklets, players) log-likelihoods added to the snapshot evidence (appearance); pool: candidate players per tracklet (default: seen near it in a snapshot)."""
    npl = len(team_p)
    Sc, ns = evidence(tracks, sl, corr, npl)
    cf = conflicts(tracks)
    if pool is None:
        pool = pool_matrix(tracks, sl, npl)
    if extra is not None:
        Sc = Sc + extra
    ttm = np.array([team_of_tr(t) for t in tracks])
    allowed = np.ones((len(tracks), npl), bool)
    for ti in range(len(tracks)):
        if ttm[ti] >= 0:
            allowed[ti] = (team_p == ttm[ti]) | goalie
        if ttm[ti] == -2:
            allowed[ti] = False
    allowed &= pool
    chains = [[] for _ in range(npl)]
    for ti in np.argsort([t["a"] for t in tracks]):
        if ident[ti] >= 0:
            chains[ident[ti]].append(int(ti))
    for rnd in range(max_rounds):
        cand = []
        for ti in range(len(tracks)):
            if ident[ti] >= 0:
                continue
            ok = allowed[ti].copy()
            for p in np.flatnonzero(ok):
                if any(cf[ti, tj] for tj in chains[p]):
                    ok[p] = False
            if not ok.any():
                continue
            s = np.full(npl, -np.inf)
            for p in np.flatnonzero(ok):
                s[p] = Sc[ti, p] + link_score(tracks, chains[p], ti)
            z = np.r_[s, -(NULL * ns[ti] + LINK_NULL * 2)] * TEMPER
            z = np.exp(z - z.max())
            post = z / z.sum()
            k = int(np.argmax(post[:-1]))
            if post[k] >= thr:
                cand.append((post[k], ti, k))
        n_new = 0
        for c, ti, p in sorted(cand, reverse=True):
            if ident[ti] >= 0 or any(cf[ti, tj] for tj in chains[p]):
                continue
            ident[ti] = p
            conf[ti] = c
            chains[p].append(ti)
            chains[p].sort(key=lambda j: tracks[j]["a"])
            n_new += 1
        if verbose:
            print(f"  round {rnd}: +{n_new} -> {int((ident >= 0).sum())}/{len(tracks)} identified", flush=True)
        if n_new == 0:
            break
    return ident, conf


RING_COL = {0: (255, 120, 0), 1: (0, 215, 255), -1: (230, 230, 230)}


def player_items(tracks, ident, team_p, roster):
    """Per tracklet: (team for the ring colour, number text or None). Not-a-player tracklets are dropped."""
    out = {}
    for i, t in enumerate(tracks):
        tm = team_of_tr(t)
        if tm == -2:
            continue
        if ident[i] >= 0:
            r = roster[ident[i]]
            out[i] = (int(team_p[ident[i]]), ("G" if r.get("goalie") else "") + str(r["jersey"]))
        else:
            out[i] = (tm, None)
    return out


def frame_items(tracks, items, N, min_sep=0.8):
    """Rings visible at frame N, duplicates (two tracklets on one person) removed in favour of the longer one."""
    cand = [(tracks[i]["b"] - tracks[i]["a"], i) for i in items if tracks[i]["a"] <= N <= tracks[i]["b"]]
    kept = []
    for _, i in sorted(cand, reverse=True):
        xy = tracks[i]["pos"][N - tracks[i]["a"]]
        if all(np.linalg.norm(xy - k[1]) > min_sep for k in kept):
            kept.append((i, xy))
    return kept


def draw_scene(img, g, f, rv, rings):
    """rings: list of (xy, team, text or None). Ring on the ice in the team colour, number above the head only when given."""
    out = img.copy()
    ov = out.copy()
    for xy, team, text in rings:
        col = RING_COL[team]
        ring = np.array([[xy[0] + 0.7 * np.cos(a), xy[1] + 0.7 * np.sin(a), 0.0] for a in np.linspace(0, 2 * np.pi, 28)])
        q = S.project(g, f, rv, ring)
        if not np.isfinite(q).all() or np.abs(q).max() > 4000:
            continue
        cv2.polylines(ov, [q.astype(np.int32)], True, col, 4, cv2.LINE_AA)
        if text:
            head = S.project(g, f, rv, np.array([[xy[0], xy[1], 2.15]]))[0]
            if 0 < head[0] < 1920 and 0 < head[1] < 1080:
                (tw, th), _ = cv2.getTextSize(text, 0, 0.9, 2)
                x0, y0 = int(head[0] - tw / 2), int(head[1])
                cv2.rectangle(ov, (x0 - 5, y0 - th - 5), (x0 + tw + 5, y0 + 5), col, -1)
                cv2.putText(ov, text, (x0, y0), 0, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return cv2.addWeighted(ov, 0.9, out, 0.1, 0)


def build(verbose=True):
    g, snaps = load_snaps()
    DT = Detections()
    roster = g["roster"]
    team_p = np.array([0 if r["team"] == "Home" else 1 for r in roster])
    goalie = np.array([bool(r.get("goalie")) for r in roster])
    n0, n1 = DT.N0, DT.N0 + WIN["n"]
    sl = snap_list(snaps, n0, n1)
    print("tracklets ...")
    raw = build_tracklets(DT, n0, n1)
    print(
        f"  {len(raw)} tracklets; detections rejected: off the ice {DT.stat['off_rink']} ({100 * DT.stat['off_rink'] / DT.stat['total']:.0f}%), implausible height {DT.stat['bad_height']}",
        flush=True,
    )
    print("pass 1: identity from snapshots and jersey colour")
    tracks, ident, conf, corr = identify_split(raw, snaps, team_p, n0, n1, verbose=False, goalie=goalie)
    print("roles (home / away / not a player):")
    P, _ = roles.fit_predict(tracks, roles.labels(tracks, ident, sl, corr, team_p))
    roles.assign_roles(tracks, P)
    print("pass 2: identity with roles")
    tracks, ident, conf, corr = identify_split(tracks, snaps, team_p, n0, n1, verbose=verbose, goalie=goalie)
    ident0, conf0 = ident.copy(), conf.copy()
    print("global propagation:")
    ident, conf = propagate(tracks, ident0.copy(), conf0.copy(), sl, corr, team_p, goalie, verbose=verbose)
    print(f"  certain identities (posterior >= {THR}): {(ident0 >= 0).sum()} -> {(ident >= 0).sum()} of {len(tracks)} tracklets")
    pickle.dump(
        dict(tracks=tracks, ident=ident, conf=conf, ident0=ident0, conf0=conf0, corr=corr, n0=n0, n1=n1, P=P), open(CACHE, "wb")
    )


def load():
    return pickle.load(open(CACHE, "rb"))


def evaluate(windows=range(int(WIN["t0"]), int(WIN["t0"]) + 300, 20)):
    """Per window, over in-view snapshot players: ring coverage, number coverage; and rings that sit on nobody in the metadata."""
    g, snaps = load_snaps()
    DT = Detections()
    roster = g["roster"]
    R = load()
    tracks, ident, corr = R["tracks"], R["ident"], R["corr"]
    team_p = np.array([0 if r["team"] == "Home" else 1 for r in roster])
    sl = snap_list(snaps, R["n0"], R["n1"])
    items = player_items(tracks, ident, team_p, roster)
    base = {
        i: (team_of_col(t["col"]), None) for i, t in enumerate(tracks) if team_of_col(t["col"]) >= 0
    }  # old rule: any tracklet with a jersey colour
    print(
        "window       snaps | rings on in-view snapshot players | numbers (certain) | rings at snapshot times not near any metadata player: new rule / old colour rule"
    )
    tot = np.zeros(6)
    for f0 in windows:
        ring = num = tt = fa_new = fa_old = n_new = n_old = 0
        for N, d in sl:
            if not f0 * FPS <= N < (f0 + 20) * FPS:
                continue
            k = DT.rowN.get(N)
            if k is None:
                continue
            pos_new = frame_items(tracks, items, N)
            pos_old = frame_items(tracks, base, N)
            meta = np.array([corr(m, p) for p, m in d.items()]) if d else np.zeros((0, 2))
            if len(d) >= 11:
                n_new += len(pos_new)
                n_old += len(pos_old)
                fa_new += sum(1 for _, xy in pos_new if np.linalg.norm(meta - xy, axis=1).min() > 3.0)
                fa_old += sum(1 for _, xy in pos_old if np.linalg.norm(meta - xy, axis=1).min() > 3.0)
            for p, m in d.items():
                c = corr(m, p)
                px = S.project(DT.g, DT.Z["f"][k], DT.Z["rv"][k], np.array([[*c, 0.0]]))[0]
                if not (60 < px[0] < 1860 and 60 < px[1] < 1020):
                    continue
                tt += 1
                if any(np.linalg.norm(xy - c) < 2.5 for _, xy in pos_new):
                    ring += 1
                if any(ident[i] == p and np.linalg.norm(xy - c) < 2.5 for i, xy in pos_new):
                    num += 1
        tot += (ring, num, tt, fa_new, fa_old, 0)
        tot[5] += 0
        ns = sum(1 for N, d in sl if f0 * FPS <= N < (f0 + 20) * FPS)
        print(
            f"{f0}-{f0 + 20}s  {ns:3d}   |  {ring:4d}/{tt:<4d} ({100 * ring / max(tt, 1):3.0f}%)  |  {num:4d}/{tt:<4d} ({100 * num / max(tt, 1):3.0f}%)  |  {fa_new:3d} / {fa_old:3d}   (of {n_new} / {n_old} rings)"
        )
    print(
        f"all windows: rings {100 * tot[0] / tot[2]:.1f}%  numbers {100 * tot[1] / tot[2]:.1f}% of {int(tot[2])} in-view snapshot players;  rings on nobody: new {int(tot[3])}, old colour rule {int(tot[4])}"
    )


def render(t0, dur, prefix):
    g, snaps = load_snaps()
    DT = Detections()
    roster = g["roster"]
    R = load()
    tracks, ident = R["tracks"], R["ident"]
    team_p = np.array([0 if r["team"] == "Home" else 1 for r in roster])
    items = player_items(tracks, ident, team_p, roster)
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
        rings = [(xy, items[i][0], items[i][1]) for i, xy in frame_items(tracks, items, N)]
        out = draw_scene(img, DT.g, f, rv, rings)
        cv2.putText(
            out,
            f"t={N / FPS:.1f}s  ring = player (team colour), number = certain identity",
            (30, 60),
            0,
            1.2,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"t={N / FPS:.1f}s  ring = player (team colour), number = certain identity",
            (30, 60),
            0,
            1.2,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        w.stdin.write(out.tobytes())
    w.stdin.close()
    w.wait()
    print("wrote", f"{prefix}.mp4")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "eval":
        evaluate()
    elif cmd == "render":
        render(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
