"""Stabilise the cast: one continuous, smoothed track per person on the ice for the whole clip.

Detections (tracked players, anonymous rings and untracked outlines from prep.py) are linked frame to frame by position continuity
(Hungarian matching on constant-velocity predictions), so tracker id swaps cannot teleport anyone. Outlines whose feet land beyond
OFF_ICE outside the rink, or below the foreground boards' top edge, are not player sightings. Every outline's own jersey colour
(team_of_track on that frame alone, unless the frame edge cuts the outline) must agree: a tracked player's identity is not given to an outline in the other team's or a
referee's colours, and a fragment never takes, or joins with, outlines whose colour contradicts its own (COLOUR_VOTES or more). Short-lived or off-ice
fragments are dropped, compatible fragments are joined, and every track gets an identity and team by majority vote, with jersey
colour deciding the team of untracked people (referees by the HockeyAI detector where the match has its boxes, else when neither
team's colour fits). Gaps are interpolated, the ends are held,
and paths are smoothed.

A broadcast without a feed names its people from bundle/identities.json (by eye): [{frame, px [x, y], jersey, name,
goalie}], each given to the track whose outline holds that source pixel at that frame. A team with no goalie named has one when a
track stays near a crease: within CREASE m of the point a metre in front of either goal line for 90% of the frames it is seen (on
the 2025 goals every goalie the feed names is within 2.4 m, every skater beyond 2.9).

  bundle/cast.json  tracks [{key, id, jersey, name, team, goalie, xy [n][2], mask [n] (outline index or -1), seen [n]}]

usage (calib venv, in a goal dir; just bundle):  [BUNDLE=dir] cast.py"""

import json
import os
from functools import lru_cache

import cv2
import numpy as np
from goal.prep import project, ray_to_ice
from scipy.optimize import linear_sum_assignment

B = os.environ.get("BUNDLE", "bundle")
with open(f"{B}/scene.json") as f:
    S = json.load(f)
Z = np.load(f"{B}/scene.npz")
N, HX, HY, CR = S["n"], S["hx"], S["hy"], S["corner"]
GATE, GATE_MISS, MAX_MISS = 1.2, 0.25, 12  # metres; extra metres per missed frame; frames a track may coast
MIN_LEN = 15  # detections a fragment needs to be kept
OFF_ICE = 0.5  # m outside the rink outline where an outline's feet mean it is no player
BOARD_HEIGHT = 1.07  # opaque boards, as in render.py; glass above them is transparent
CREASE = 2.5  # m from the crease a goalie stays within (90% of the frames)
COLOUR_VOTES = 3  # outlines of one colour that set a fragment's colour for linking and joining
REF = -1  # team of referees and linesmen
TEAMS = S.get("teams")  # the match's labelled jersey groups (broadcast/bundle.py), for matches without a tracking feed


def crease_goalies(tracks):
    """Marks as goalie, for each team without one, its track that stays nearest a crease, when within CREASE."""
    crease = np.array([HX - 5.0, 0.0])
    for team in (0, 1):
        if any(t["goalie"] for t in tracks if t["team"] == team):
            continue
        stay = [
            (np.percentile(np.linalg.norm(np.abs(np.array(t["xy"])[np.array(t["seen"])]) - crease, axis=1), 90), t["key"])
            for t in tracks
            if t["team"] == team and any(t["seen"])
        ]
        if stay and min(stay)[0] < CREASE:
            tracks[min(stay)[1]]["goalie"] = True


def sdf(xy):
    """Signed distance to the rink outline (negative inside)."""
    q = np.abs(xy) - [HX - CR, HY - CR]
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0) - CR


def feet_visible(cam, xy):
    """The camera-to-foot ray must clear the foreground boards. Staff behind them can project onto the ice, but below this edge."""
    C = cam[11:14]
    if sdf(C[:2]) <= 0:
        return np.ones(len(xy), bool)
    lo, hi = np.zeros(len(xy)), np.ones(len(xy))
    for _ in range(24):
        mid = (lo + hi) / 2
        outside = sdf(C[:2] + mid[:, None] * (xy - C[:2])) > 0
        lo = np.where(outside, mid, lo)
        hi = np.where(outside, hi, mid)
    return C[2] * (1 - hi) > BOARD_HEIGHT


def smooth(a, sigma):
    r = int(3 * sigma)
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    w /= w.sum()
    pad = np.concatenate([a[:1].repeat(r, 0), a, a[-1:].repeat(r, 0)])
    return np.stack([np.convolve(pad[:, j], w, "valid") for j in range(a.shape[1])], 1)


def detections():
    """Per frame: one detection per outline, at its foot point on the ice, with the tracked player it was assigned to (if any)."""
    out = []
    for i, P in enumerate(S["people"]):
        polys = S["outlines"][i]
        if not polys:
            out.append([])
            continue
        bottoms = np.array([[np.mean(np.array(p)[:, 0]), np.max(np.array(p)[:, 1])] for p in polys])
        feet = ray_to_ice(Z["cam"][i], S["fref"], S["cx"], S["cy"], bottoms)
        owner = {e["mask"]: e for e in P if e["mask"] >= 0}
        frame, off = [], (sdf(feet) > OFF_ICE) | ~feet_visible(Z["cam"][i], feet)
        for m in range(len(polys)):
            if off[m]:
                continue  # spectators, bench and camera staff behind the boards
            o = owner.get(m, {})
            p = np.array(polys[m])
            h, w = frame_lab(i).shape[:2]
            cut = (p.min(0) < 3).any() or p[:, 0].max() > w - 4 or p[:, 1].max() > h - 4
            px = None if cut else jersey_px(i, polys[m])  # an outline cut by the frame edge may show only a helmet
            ct = team_of_track([px], float(o.get("ref", False))) if px is not None else None
            named = o.get("id", -1) >= 0 and (ct is None or ct == S["roster"][o["id"]]["team"])
            frame.append(
                dict(
                    xy=feet[m].tolist(),
                    mask=m,
                    id=o["id"] if named else -1,
                    team=o.get("team", -1),
                    ref=o.get("ref", False),
                    ct=ct,
                )
            )
        out.append(frame)
    return out


def colour(seq):
    """A fragment's jersey colour: the most common outline colour once it has COLOUR_VOTES, else None."""
    vals = [e["ct"] for _, e in seq if e["ct"] is not None]
    if not vals:
        return None
    v, c = np.unique(vals, return_counts=True)
    return int(v[c.argmax()]) if c.max() >= COLOUR_VOTES else None


def sensor_paths():
    """Tracking-system positions per roster id: frame -> xy. They lag the video (interpolated between event snapshots)."""
    paths = {}
    for i, P in enumerate(S["people"]):
        for e in P:
            if e["id"] >= 0:
                paths.setdefault(e["id"], {})[i] = np.array(e["xy"])
    return paths


def link(dets):
    """Frame-to-frame association by position: fragments as lists of (frame, detection)."""
    live, done = [], []
    for i, P in enumerate(dets):
        pred = np.array([t["pos"] + t["vel"] * (i - t["last"]) for t in live]).reshape(-1, 2)
        xy = np.array([e["xy"] for e in P]).reshape(-1, 2)
        D = np.linalg.norm(pred[:, None] - xy[None], axis=2)
        gate = np.array([GATE + GATE_MISS * (i - t["last"] - 1) for t in live])[:, None]
        colours = [colour(t["seq"]) for t in live]
        clash = np.array([[ct is not None and e["ct"] is not None and ct != e["ct"] for e in P] for ct in colours], bool).reshape(
            len(live), len(P)
        )
        D = np.where((D < gate) & ~clash, D, 1e6)
        hit = set()
        if len(live) and len(P):
            for r, c in zip(*linear_sum_assignment(D)):
                if D[r, c] >= 1e6:
                    continue
                t = live[r]
                dt = i - t["last"]
                v = (xy[c] - t["pos"]) / dt
                t["vel"] = 0.6 * t["vel"] + 0.4 * v
                t["pos"] = xy[c]
                t["last"] = i
                t["seq"].append((i, P[c]))
                hit.add(c)
        for c in set(range(len(P))) - hit:
            live.append(dict(pos=xy[c], vel=np.zeros(2), last=i, seq=[(i, P[c])]))
        done += [t["seq"] for t in live if i - t["last"] > MAX_MISS]
        live = [t for t in live if i - t["last"] <= MAX_MISS]
    return done + [t["seq"] for t in live]


def vote(seq, key):
    vals = [e[key] for _, e in seq if e[key] >= 0]
    if not vals:
        return -1, 0.0
    v, c = np.unique(vals, return_counts=True)
    return int(v[c.argmax()]), c.max() / len(seq)


def join(frags):
    """Greedily join fragments end-to-start when the gap is short, the motion plausible and identities compatible."""
    frags = sorted(frags, key=lambda s: s[0][0])
    joined = True
    while joined:
        joined = False
        best = None
        for a, A in enumerate(frags):
            for b, Bq in enumerate(frags):
                gap = Bq[0][0] - A[-1][0]
                if a == b or not 0 < gap <= 25:
                    continue
                ia, ib = vote(A, "id")[0], vote(Bq, "id")[0]
                if ia >= 0 and ib >= 0 and ia != ib:
                    continue
                ca, cb = colour(A), colour(Bq)
                if ca is not None and cb is not None and ca != cb:
                    continue
                d = np.hypot(*np.subtract(Bq[0][1]["xy"], A[-1][1]["xy"]))
                if d < 1.0 + 0.3 * gap and (best is None or d < best[0]):
                    best = (d, a, b)
        if best:
            _, a, b = best
            frags[a] = frags[a] + frags[b]
            del frags[b]
            joined = True
    return frags


def colour_hist(px):
    """12-bin jersey histogram of Lab pixels (OpenCV 8-bit): coloured pixels (chroma > 12) by hue in 8 bins, neutral ones by lightness in
    4 (black, dark, light, white). Same as broadcast/bundle.py."""
    a, b = px[:, 1] - 128, px[:, 2] - 128
    col = np.hypot(a, b) > 12
    bins = np.where(
        col, (np.arctan2(b, a) % (2 * np.pi) / (np.pi / 4)).astype(int) % 8, 8 + np.digitize(px[:, 0], [60, 120, 190])
    )
    return np.bincount(bins, minlength=12) / len(px)


def team_of_track(pxs, ref):
    """Team of a track from its upper-body pixels in several frames and the share of its outlines the detector calls a referee. With the
    match's labelled jersey groups (broadcast/bundle.py): a referee when most outlines are, None if the track's mean histogram is nearest
    a junk group (empty ice, spectators), else the label of the nearest home or away group (Hellinger distance). Without them: the
    Spengler Cup 2025 colours on the median colour (yellow away, navy home), striped neutral shirts are referees, None for anything
    else (spectators, bench staff)."""
    if TEAMS:
        if ref > 0.5:
            return REF
        h = np.sqrt(np.mean([colour_hist(p) for p in pxs], 0))
        d = [np.linalg.norm(h - np.sqrt(g)) for g in TEAMS["groups"]]
        if TEAMS["labels"][int(np.argmin(d))] == "junk":
            return None
        return {"home": 0, "away": 1}[
            TEAMS["labels"][min((j for j, l in enumerate(TEAMS["labels"]) if l in ("home", "away")), key=lambda j: d[j])]
        ]
    L, a, b_, spread = np.median(
        [np.r_[p.mean(0), p[:, 0].std()] for p in pxs], 0
    )  # lightness spread separates striped referee shirts
    if b_ > 145:
        return 1
    if b_ < 118 and L > 60:
        return 0
    return REF if abs(b_ - 128) < 8 and L > 85 and spread > 50 else None


def jersey_px(i, poly):
    """Lab pixels of the upper body (top half of the outline below the helmet, eroded) in source frame i."""
    img = frame_lab(i)
    poly = np.array(poly)
    y0, y1 = poly[:, 1].min(), poly[:, 1].max()
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    m[int(y0 + 0.55 * (y1 - y0)) :] = 0
    m[: int(y0 + 0.12 * (y1 - y0))] = 0
    px = img[cv2.erode(m, np.ones((5, 5), np.uint8)) > 0].astype(np.float32)
    return px if len(px) > 30 else None


@lru_cache(maxsize=1)
def frame_lab(i):
    img = cv2.imread(f"{B}/frames/{i + 1:03d}.jpg")
    if img is None:
        raise FileNotFoundError(f"{B}/frames/{i + 1:03d}.jpg")
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


def identify(tracks):
    """Jersey, name and goalie from identities.json onto the tracks whose outline holds each labelled pixel."""
    labels = []
    if os.path.exists(f"{B}/identities.json"):
        with open(f"{B}/identities.json") as f:
            labels = json.load(f)
    for e in labels:
        i = e["frame"]
        hit = [
            t
            for t in tracks
            if t["mask"][i] >= 0
            and cv2.pointPolygonTest(np.array(S["outlines"][i][t["mask"][i]], np.float32), tuple(map(float, e["px"])), False) >= 0
        ]
        if len(hit) != 1:
            raise SystemExit(f"identity #{e['jersey']} at frame {i} {e['px']}: {len(hit)} outlines hold it")
        hit[0].update(jersey=e["jersey"], name=e["name"], goalie=e.get("goalie", False))


def main():
    frags = [s for s in join(link(detections())) if len(s) >= MIN_LEN]
    frags = [s for s in frags if np.median(sdf(np.array([e["xy"] for _, e in s]))) < 0.3]

    # untracked people: referees by the detector, players by jersey colour over the track; anything else (spectators and bench staff seen through the glass)
    # is dropped
    named = {k: vote(s, "id") for k, s in enumerate(frags)}
    team_of = {}
    for k, s in enumerate(frags):
        pxs = [p for p in (jersey_px(i, S["outlines"][i][e["mask"]]) for i, e in s[::3]) if p is not None]
        team_of[k] = team_of_track(pxs, np.mean([e["ref"] for _, e in s])) if pxs else None
    keep = [k for k in range(len(frags)) if named[k][0] >= 0 or team_of[k] is not None]
    frags, named, team_of = (
        [frags[k] for k in keep],
        {j: named[k] for j, k in enumerate(keep)},
        {j: team_of.get(k) for j, k in enumerate(keep)},
    )

    # one id per roster player at a time: the strongest claim keeps it; a fragment overlapping the player's stronger ones only
    # briefly (a split outline track) gives up just those frames
    owner = {}

    def taken(pid):
        return {i for o in owner.get(pid, []) for i, _ in frags[o]}

    def fits(k, pid):
        return sum(i in taken(pid) for i, _ in frags[k]) <= 0.25 * len(frags[k])

    def claim(k, pid):
        t = taken(pid)
        frags[k] = [(i, e) for i, e in frags[k] if i not in t]
        owner.setdefault(pid, []).append(k)

    for _, k in sorted(((named[k][1] * len(s), k) for k, s in enumerate(frags) if named[k][0] >= 0), reverse=True):
        if fits(k, named[k][0]):
            claim(k, named[k][0])
        else:
            named[k] = (-1, 0.0)
    # an outline the per-frame matching never gave a name: the free tracked player whose (lagging) path stays nearest to it
    sensor = sensor_paths()
    for k, s in enumerate(frags):
        if named[k][0] >= 0:
            continue
        best = None
        ct = colour(s)
        if ct is None:
            ct = team_of[k]
        for pid, path in sensor.items():
            if ct is not None and ct != S["roster"][pid]["team"]:
                continue
            if not fits(k, pid):
                continue
            d = [np.linalg.norm(path[i] - e["xy"]) for i, e in s if i in path]
            if len(d) > 0.8 * len(s) and np.median(d) < 2.5 and (best is None or np.median(d) < best[0]):
                best = (np.median(d), pid)
        if best:
            named[k] = (best[1], 0.0)
            claim(k, best[1])

    tracks, t_all = [], np.arange(N)

    def visual(ks):
        fr = np.array([i for k in ks for i, _ in frags[k]], int)
        xy = np.array([e["xy"] for k in ks for _, e in frags[k]]).reshape(-1, 2)
        mask = np.full(N, -1)
        mask[fr] = [e["mask"] for k in ks for _, e in frags[k]]
        return fr, xy, mask

    # tracked players: the video position where they are visible; elsewhere their sensor path, shifted by the offset to the
    # nearest visible moment (fading over half a second), so entering and leaving the picture is seamless
    for pid, path in sorted(sensor.items()):
        fr, xy, mask = visual(owner.get(pid, []))
        sf = np.array(sorted(path))
        sxy = np.array([path[i] for i in sf])
        alive = np.zeros(N, bool)
        alive[sf.min() : sf.max() + 1] = True
        if len(fr):
            alive[fr.min() : fr.max() + 1] = True
        base = np.c_[np.interp(t_all, sf, sxy[:, 0]), np.interp(t_all, sf, sxy[:, 1])]
        full = base.copy()
        if len(fr):
            off = xy - base[fr]
            near = np.abs(t_all[:, None] - fr[None]).argmin(1)
            w = np.exp(-np.abs(t_all - fr[near]) / (0.5 * S["fps"]))
            full = base + off[near] * w[:, None]
            full[fr] = xy
        r = S["roster"][pid]
        tracks.append(
            dict(
                key=len(tracks),
                id=pid,
                jersey=r["jersey"],
                name=r["name"],
                team=r["team"],
                goalie=r["goalie"],
                xy=np.round(smooth(full, 2.5), 3).tolist(),
                mask=mask.tolist(),
                seen=alive.tolist(),
            )
        )
    # everyone else (referees, linesmen, untracked skaters): their outline path, continued before and after it along the entry and
    # exit motion (for up to 2 s), and present only where the broadcast camera could not have seen them
    for k, s in enumerate(frags):
        if named[k][0] >= 0 or team_of[k] is None:
            continue
        fr, xy, mask = visual([k])
        full = np.c_[np.interp(t_all, fr, xy[:, 0]), np.interp(t_all, fr, xy[:, 1])]
        m = min(10, len(fr) - 1)
        v0, v1 = (xy[m] - xy[0]) / (fr[m] - fr[0]), (xy[-1] - xy[-1 - m]) / (fr[-1] - fr[-1 - m])
        before, after = t_all < fr[0], t_all > fr[-1]
        full[before] = xy[0] + v0 * np.maximum(t_all[before] - fr[0], -50)[:, None]
        full[after] = xy[-1] + v1 * np.minimum(t_all[after] - fr[-1], 50)[:, None]
        full = smooth(full, 3.0)
        seen = (t_all >= fr[0]) & (t_all <= fr[-1])
        for i in np.flatnonzero(~seen):
            px = project(Z["cam"][i], S["fref"], S["cx"], S["cy"], np.array([[*full[i], 0.0], [*full[i], 1.8]]))
            seen[i] = not ((px[:, 0] > -20) & (px[:, 0] < 1940) & (px[:, 1] > -20) & (px[:, 1] < 1100)).any()
        tracks.append(
            dict(
                key=len(tracks),
                id=-1,
                jersey=None,
                name=None,
                team=team_of[k],
                goalie=False,
                xy=np.round(full, 3).tolist(),
                mask=mask.tolist(),
                seen=seen.tolist(),
            )
        )
    identify(tracks)
    crease_goalies(tracks)
    with open(f"{B}/cast.json", "w") as f:
        json.dump(dict(n=N, tracks=tracks), f)

    for t in tracks:
        f = np.flatnonzero(t["seen"])
        v = np.linalg.norm(np.diff(np.array(t["xy"]), axis=0), axis=1) * S["fps"]
        print(
            f"vmax {v.max():4.1f} m/s key {t['key']:3d} id {t['id']:3d} #{t['jersey'] or '-':>3} team {t['team']:2d} {'G' if t['goalie'] else ' '} "
            f"frames {f[0]:3d}-{f[-1]:3d} masks {sum(m >= 0 for m in t['mask']):3d}"
        )
    raw = [sum(e["mask"] >= 0 for e in P) for P in S["people"]]
    print(f"{len(tracks)} tracks (raw outlines/frame {np.mean(raw):.1f}, min {min(raw)}, max {max(raw)})")


if __name__ == "__main__":
    main()
