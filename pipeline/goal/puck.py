"""The puck as seen in the broadcast, on the rink, for every frame of the bundle (replaces the event-interpolated path from prep).

The tracking feed only has puck positions at event snapshots; between them prep interpolates straight lines, which is fine while a
player carries the puck and wrong for passes and shots. Here the puck is found in the video: HockeyAI (a hockey detector with a puck
class) at DET_SIZE px gives each frame's DET_TOP puck boxes (puck.py detect, bundle/puck_dets.json), and those on visible ice are the
candidates, scored by confidence. Candidates are linked by a Viterbi pass over rink positions: steps cost their acceleration, a frame
may be "unseen" (occluded or blurred by a shot) at a fixed price, and the event path pulls weakly. Unseen stretches are bridged in a
straight line between their ends, which is what a puck does. A goal in the window fixes the end: at the shot's release the puck is on
the scorer's blade (a stick's length in front of them, toward the net) and from there it is the shot below; tracking stops at the release.

That path places the puck for planning shots (director.py) and is never drawn: from the video alone it is mostly wrong (against the
feed's puck snapshots, median 10 m off: the detector's best box is the puck about a quarter of the time), and a puck that jumps about
is worse than none.

Only the goal shot is drawn, and only when its release is confirmed in the video: bundle/release.json {"frame", "how"} is the frame
the scorer's stick strikes the puck, read off the broadcast (the feed's shot time is 0.2 to 0.5 s early, by a varying amount), else
bundle/release_auto.json, the strike of the scorer's fitted stick (release.py). Without either the shot keeps the feed's time for
planning and no puck is shown. The release found in the track (release(): the last frame the
puck is seen on the scorer's stick before it is next seen RELEASE_GONE m away toward the net) is kept in the shot for inspection only:
against the six releases read by eye it is within 2 frames for three. From the release the puck leaves the blade (a stick's length in
front of the scorer, toward the net) at SHOT_V, flies straight over the goal line (inside the posts) and on into the net, strikes the
mesh (back NET_BACK behind the line, sides at the posts), keeps RESTITUTION of its speed across the mesh and ALONG of it along the
mesh, slows at NET_DRAG and comes to rest inside. It is aimed at the middle of the back of the net: where it ends up is rarely visible
(the net's own markings look like a lying puck), and the middle is the least wrong guess. show is its opacity per frame: ramping in
over FADE frames from the release, held until HOLD frames after the puck has stopped, then fading out.

A broadcast without a tracking feed (broadcast/bundle.py) has no event path to pull toward, and its goal is bundle/goal.json
{"jersey", "team", "release"}, read off the broadcast and its replays (the scorer is the cast track carrying that jersey, cast.py
identities). The release is an estimate for
planning that only a release.json (or release_auto.json) confirms.

  bundle/puck_dets.json  per frame [[u, v, conf], ...] (puck.py detect)
  bundle/puck.json       per frame {xy, seen, show}, the shot, and the candidates for inspection
  check/puck.jpg         crops of the broadcast around the tracked puck

usage (from pipeline/):  puck.py detect   (YOLO venv, GPU)
                          puck.py"""

import json
import os
import sys

import cv2
import numpy as np
from paths import GAME, MODELS

B = os.environ.get("BUNDLE", "bundle")
S = json.load(open(f"{B}/scene.json"))
Z = np.load(f"{B}/scene.npz")
N, FPS = S["n"], S["fps"]
GOAL_X, POST = S["hx"] - 4.0, 0.915
UNSEEN, ACCEL, PRIOR, GATE = (
    4.0,
    0.6,
    2.0,
    6.0,
)  # cost of an unseen frame; m/s^2 scale of a free step; event-path tolerance and hard gate (m)
MAXV = 50.0  # m/s: faster than any shot
SHOT_V, BLADE = 25.0, 1.0  # m/s of a close-range shot; m from the scorer to the blade
NET_BACK, PUCK_R = 1.12, 0.038  # m from the goal line to the back of the net at ice level; puck radius
RESTITUTION, ALONG, NET_DRAG = (
    0.1,
    0.05,
    8.0,
)  # share of its speed across and along the mesh a puck keeps when it hits it; m/s^2 slowing it inside the net
FADE, HOLD = 3, 20  # frames the puck fades in from the release and out at the end; frames it stays shown at rest
HOCKEYAI = f"{MODELS}/simula/HockeyAI_model_weight.pt"
DET_SIZE, DET_TOP, DET_MIN = 1920, 5, 0.05  # detector input px, boxes kept per frame, lowest confidence kept
REACH, RELEASE_GONE, RELEASE_SPAN = (
    2.5,
    3.0,
    (-8, 25),
)  # m: on the scorer's stick; m: gone toward the net; frames around the feed's shot


def project(cam, P):
    f, k, R, C = cam[0], cam[1], cam[2:11].reshape(3, 3), cam[11:14]
    pc = (np.atleast_2d(P) - C) @ R.T
    uv = f * pc[:, :2] / pc[:, 2:3]
    uv = uv * (1 + k * (uv**2).sum(1, keepdims=True) / S["fref"] ** 2)
    return uv + [S["cx"], S["cy"]], pc[:, 2]


def to_ice(cam, px):
    d = np.atleast_2d(px) - [S["cx"], S["cy"]]
    u = d.copy()
    for _ in range(8):
        u = d / (1 + cam[1] * (u**2).sum(1, keepdims=True) / S["fref"] ** 2)
    ray = np.c_[u / cam[0], np.ones(len(u))] @ cam[2:11].reshape(3, 3)
    return (cam[11:14] + (-cam[13] / ray[:, 2])[:, None] * ray)[:, :2]


def inside_rink(xy, margin=0.1):
    q = np.abs(xy) - [S["hx"] - S["corner"], S["hy"] - S["corner"]]
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0) - S["corner"] < -margin


def over_boards(cam, xy, wall=1.25):
    """True where the line of sight from the camera to the ice point clears the boards (with glass rail), i.e. the ice is visible there."""
    C = cam[11:14]
    t = np.linspace(0, 1, 200)[:, None]
    P = C + t * (np.r_[xy, 0] - C)
    q = np.abs(P[:, :2]) - [S["hx"] - S["corner"], S["hy"] - S["corner"]]
    sd = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0) - S["corner"]
    k = np.flatnonzero(sd > 0)
    return len(k) == 0 or P[k[-1], 2] > wall


def detect():
    from ultralytics import YOLO

    m = YOLO(HOCKEYAI)
    pk = next(k for k, v in m.names.items() if v.lower() == "puck")
    out = []
    for i in range(N):
        b = m.predict(f"{B}/frames/{i + 1:03d}.jpg", imgsz=DET_SIZE, conf=DET_MIN, classes=[pk], verbose=False)[0].boxes
        k = b.conf.argsort(descending=True)[:DET_TOP]
        out.append(
            [
                [round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                for (x, y), c in zip(b.xywh[k, :2].tolist(), b.conf[k].tolist())
            ]
        )
    json.dump(out, open(f"{B}/puck_dets.json", "w"))
    print("frames with a puck box", sum(bool(o) for o in out), "of", N)


DETS = json.load(open(f"{B}/puck_dets.json")) if os.path.exists(f"{B}/puck_dets.json") else None


def candidates(i):
    """[(x, y on the rink, score)] for frame i: the detector's puck boxes on visible ice, scored by confidence."""
    cam = Z["cam"][i].astype(np.float64)
    out = []
    for u, v, conf in DETS[i]:
        xy = to_ice(cam, (u, v))[0]
        if inside_rink(xy) and over_boards(cam, xy):
            out.append((float(xy[0]), float(xy[1]), 2.0 + 6.0 * conf))
    return out


def viterbi(cands, prior):
    """Best path through candidates (or 'unseen') per frame. Positions of unseen frames are filled afterwards."""
    states = [[(c[0], c[1], c[2]) for c in cs] + [None] for cs in cands]
    cost = [np.array([PRIOR_COST(s, prior[0]) - (s[2] if s else -UNSEEN) for s in states[0]])]
    back, last_pos = [], [[(s[:2] if s else None, None) for s in states[0]]]
    for i in range(1, len(states)):
        c_i, b_i, lp_i = [], [], []
        for s in states[i]:
            best, arg, lp = np.inf, 0, None
            for j, _s0 in enumerate(states[i - 1]):
                pos0, gap0 = last_pos[-1][j]
                if s is None:
                    step = 0.0
                    pos = pos0
                    gap = (gap0 or 0) + 1
                elif pos0 is None:
                    step = 0.0
                    pos = s[:2]
                    gap = None
                else:
                    dt = ((gap0 or 0) + 1) / FPS
                    v = np.hypot(s[0] - pos0[0], s[1] - pos0[1]) / dt
                    if v > MAXV:
                        continue
                    step = (v * dt / (ACCEL * dt + 0.15)) ** 2 * 0.02
                    pos = s[:2]
                    gap = None
                c = cost[-1][j] + step
                if c < best:
                    best, arg, lp = c, j, (pos, gap)
            unary = UNSEEN if s is None else -s[2]
            c_i.append(best + unary + (PRIOR_COST(s, prior[i]) if s else 0))
            b_i.append(arg)
            lp_i.append(lp or (None, None))
        cost.append(np.array(c_i))
        back.append(b_i)
        last_pos.append(lp_i)
    k = int(np.argmin(cost[-1]))
    path = [k]
    for b in reversed(back):
        k = b[k]
        path.append(k)
    path = path[::-1]
    return [states[i][k] for i, k in enumerate(path)]


def PRIOR_COST(s, p):
    return 0.0 if s is None or not np.isfinite(p).all() else (np.hypot(s[0] - p[0], s[1] - p[1]) / PRIOR) ** 2


def goal_json():
    """The window's goal without a feed: goal.json (by eye), else None."""
    path = f"{B}/goal.json"
    return json.load(open(path)) if os.path.exists(path) else None


def goal_feed_frame():
    """The shot frame of the window's goal (for the release search): the goal's estimate without a feed, else the feed's, else the
    window's last frame."""
    if g := goal_json():
        return g["release"]
    for e in json.load(open(GAME))["events"]:
        if e["kind"] == "goal" and 0 <= e.get("shotT", -1) - S["start"] < N / FPS:
            return int(round((e["shotT"] - S["start"]) * FPS))
    return N - 1


def goal_end():
    """(release frame, confirmed, blade xy, scorer key) of a goal whose shot is released inside the window, else None. The release is
    release.json's when there is one (confirmed in the video), else the feed's shot time; the blade is in front of the scorer at it.
    Without a feed the goal is goal_json()'s."""
    cast = json.load(open(f"{B}/cast.json"))
    found = [p for p in (f"{B}/release.json", f"{B}/release_auto.json") if os.path.exists(p)]
    confirmed = bool(found)

    def end(f, key=None, jersey=None, team=None):
        t = next(t for t in cast["tracks"] if t["key"] == key or key is None and t.get("jersey") == jersey and t["team"] == team)
        me = np.array(t["xy"][f])
        d = aim(me) - me
        return f, confirmed, me + BLADE * d / np.linalg.norm(d), t["key"]

    if g := goal_json():
        return end(
            json.load(open(found[0]))["frame"] if confirmed else g["release"], g.get("key"), g.get("jersey"), g.get("team")
        )
    for e in json.load(open(GAME))["events"]:
        if e["kind"] != "goal" or not 0 <= e.get("shotT", -1) - S["start"] < N / FPS:
            continue
        f = json.load(open(found[0]))["frame"] if confirmed else int(round((e["shotT"] - S["start"]) * FPS))
        return end(f, jersey=e["player"].split()[0].lstrip("#"), team=0 if e["team"] == "Home" else 1)
    return None


def release(xy, seen, scorer, feed):
    """The release in the tracked path: the last frame (near the feed's shot frame) the puck is seen within REACH of the scorer before
    it is next seen more than RELEASE_GONE m nearer the net, or None."""
    me = np.array(next(t for t in json.load(open(f"{B}/cast.json"))["tracks"] if t["key"] == scorer)["xy"])
    seen_at = np.flatnonzero(seen)
    for f in seen_at[(seen_at >= feed + RELEASE_SPAN[0]) & (seen_at <= feed + RELEASE_SPAN[1])]:
        if np.linalg.norm(xy[f] - me[f]) > REACH:
            continue
        nxt = seen_at[seen_at > f]
        if not len(nxt):
            return None
        g = nxt[0]
        net = aim(me[f])
        if np.linalg.norm(xy[g] - me[f]) > RELEASE_GONE and np.linalg.norm(net - xy[g]) < np.linalg.norm(net - xy[f]):
            return int(f)
    return None


def aim(xy):
    """Where a shot from xy at the middle of the back of the nearer net crosses its goal line, at least 0.2 m inside the posts."""
    g = np.sign(xy[0]) * GOAL_X
    back = np.sign(xy[0]) * (GOAL_X + NET_BACK)
    return np.array([g, np.clip(xy[1] * (back - g) / (back - xy[0]), -POST + 0.2, POST - 0.2)])


def shot_path(blade, n):
    """n frames of the puck from the release at the blade: straight over the goal line at SHOT_V, then inside the net off the mesh
    until it stops. Also the frames it crosses the goal line and comes to rest (relative to the release)."""
    side = np.sign(blade[0])
    sub = 20
    dt = 1 / (FPS * sub)
    a = aim(blade)
    p = blade.astype(np.float64)
    v = (a - blade) / np.linalg.norm(a - blade) * SHOT_V
    lo, hi = (
        np.array([PUCK_R, -POST + PUCK_R]),
        np.array([NET_BACK - PUCK_R, POST - PUCK_R]),
    )  # inside the net: depth behind the line, y
    out, crossed, rest, hit = [p], None, None, False
    for k in range(1, n):
        for _ in range(sub):
            p = p + v * dt
            q = np.array([side * p[0] - GOAL_X, p[1]])
            u = np.array([side * v[0], v[1]])
            if q[0] > 0:
                crossed = k if crossed is None else crossed
                if q[0] > hi[0]:
                    q[0] = hi[0]
                    u *= [-RESTITUTION, ALONG]
                    hit = True  # the back of the net
                if not lo[1] <= q[1] <= hi[1]:
                    q[1] = np.clip(q[1], lo[1], hi[1])
                    u *= [ALONG, -RESTITUTION]
                    hit = True  # a side
                if hit and q[0] < lo[0]:
                    q[0] = lo[0]
                    u *= [-RESTITUTION, ALONG]  # once in, it stays in
                if hit:
                    sp = np.linalg.norm(u)
                    u *= max(sp - NET_DRAG * dt, 0) / max(sp, 1e-9)
                p = np.array([side * (q[0] + GOAL_X), q[1]])
                v = np.array([side * u[0], u[1]])
        out.append(p)
        if hit and rest is None and np.linalg.norm(v) == 0:
            rest = k
    return np.array(out), crossed if crossed is not None else n - 1, rest if rest is not None else n - 1


def main():
    prior = Z["puck"]
    cands = [
        [c for c in candidates(i) if not np.isfinite(prior[i]).all() or np.hypot(c[0] - prior[i][0], c[1] - prior[i][1]) < GATE]
        for i in range(N)
    ]
    path = viterbi(cands, prior)
    seen = np.array([p is not None for p in path])
    xy = np.array([p[:2] if p else (np.nan, np.nan) for p in path])
    goal = goal_end()
    shot = None
    show = np.zeros(N)
    if goal:
        f, confirmed, blade, scorer = goal
        found = release(xy, seen, scorer, goal_feed_frame())
        print("release: confirmed", f if confirmed else None, "| found in the track", found)
        seen[f:] = False
        path, crossed, rest = shot_path(blade, N - f)
        xy[f:] = path
        known = np.r_[np.flatnonzero(seen), np.arange(f, N)]
        shot = dict(
            release=f,
            in_net=f + crossed,
            rest=f + rest,
            scorer=scorer,
            blade=blade.round(3).tolist(),
            confirmed=confirmed,
            found_release=found,
        )
        if confirmed:
            i = np.arange(N)
            end = f + rest + HOLD
            show = np.clip(np.minimum((i - f + 1) / FADE, (end - i) / FADE if end < N else 1), 0, 1) * (i >= f)
    else:
        known = np.flatnonzero(seen)
    for d in range(2):
        xy[:, d] = np.interp(np.arange(N), known, xy[known, d])
    json.dump(
        dict(
            xy=xy.round(3).tolist(),
            seen=seen.tolist(),
            show=show.round(3).tolist(),
            shot=shot,
            candidates=[[list(map(lambda v: round(v, 3), c)) for c in cs] for cs in cands],
        ),
        open(f"{B}/puck.json", "w"),
    )
    print(
        "seen",
        int(seen.sum()),
        "shown",
        int((show > 0).sum()),
        "of",
        N,
        "shot",
        shot,
        f"candidates/frame {np.mean([len(c) for c in cands]):.1f}",
    )
    tiles = []
    for i in list(range(40, 100, 10)) + list(range(100, 124, 2)) + list(range(130, N, 20)):
        img = cv2.imread(f"{B}/frames/{i + 1:03d}.jpg")
        cam = Z["cam"][i].astype(np.float64)
        u = project(cam, np.r_[xy[i], 0])[0][0].astype(int)
        x0, y0 = np.clip(u - [160, 100], 0, [1920 - 320, 1080 - 200])
        crop = img[y0 : y0 + 200, x0 : x0 + 320].copy()
        cv2.circle(crop, tuple(u - [x0, y0]), 10, (0, 200, 255) if show[i] > 0 else (0, 255, 0) if seen[i] else (0, 0, 255), 2)
        if np.isfinite(prior[i]).all():
            q = project(cam, np.r_[prior[i], 0])[0][0].astype(int) - [x0, y0]
            cv2.drawMarker(crop, tuple(q), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(crop, str(i), (5, 22), 0, 0.7, (0, 255, 255), 2)
        tiles.append(crop)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    os.makedirs("check", exist_ok=True)
    cv2.imwrite("check/puck.jpg", np.vstack([np.hstack(tiles[r : r + 6]) for r in range(0, len(tiles), 6)]))


if __name__ == "__main__":
    detect() if sys.argv[1:] == ["detect"] else main()
