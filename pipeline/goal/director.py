"""Plan the novel-view shots of a goal from the data: who has the puck, where it goes, and which camera sees that best.

Beats come from the puck track (puck.py) and the goal event: the carry up to the pass, the pass and the shot, the puck in the net.
A broadcast without a feed has no puck track to find the passer from: bundle/goal.json names the passer's jersey and the pass frame,
read off the replays.
  approach   the carrier with the puck, until the pass leaves the stick
  goal       pass, shot and the puck crossing the line
  replay     the same moment again from the next best angle, at least MIN_APART degrees around (or as far around as the cameras
             over the rink allow, near the end boards)
  bullettime the goal with time frozen twice, at the release and with the puck in the net, while the camera orbits the scorer and the
             puck (ease in, ease out) from the goal camera's side; the frame list repeats the frozen frame, so the renderer needs nothing new
  orbit_low, orbit_high   full turns around the release, frozen, at 15 and 40 degrees: the views a Gaussian splat of the moment is fit to
  goal_slow  the goal camera's view, slowing smoothly to quarter speed (less for a long play) through the shot and back (frame times are fractional: the
             renderer interpolates everyone's pose between source frames, so slow motion shows real in-between poses)
  pov        the scorer's helmet camera: the pass coming, the turn to the net and the shot, slowing to half speed through the shot; the
             render puts it at his eyes and draws his stick (the shot's `own`); the generator draws his gloves
  drone      a camera drone's flight through the play, flown smoothly on a gimbal: high behind the passer, down past the pass,
             beside the scorer at the release, ending low in front of the net looking in; wide lens, slowing to a third of real speed
             through the shot, lifting over anyone it would fly through and high enough to see over a skater between it and what it
             looks at or close in view; it never backs up, accelerates at most A_MAX, pans at most PAN_MAX and barely rolls
  puckcam    riding behind the puck at stick height, looking where it goes: the pass, the shot at quarter speed, stopping short of
             the crease as the puck flies on into the net
  goalcam    the in-net camera: fixed in the back of the net at ice level, wide, aimed at the release; slowing through the shot
  celebrate  the celebration after the goal, at real speed: the scorer and the teammates who join him, from the broadcast's side,
             easing in and arcing a little round them
  cuts       an edit of the shots above, cut on the action: the broadcast approach, the scorer's POV from the second before the pass, the puck cam from the release into the net, the
             goal cam as it goes in, the celebration (parts: shot and its frame range)
drone, pov, puckcam, goalcam and bullettime last SHORT frames (3 s); speed and start adapt to fit.
For each beat, candidate cameras on a hemisphere around the action (azimuth, elevation, distance fitted to the action's spread) are
scored over the beat's frames with a coarse z-buffer of everyone's body mesh: how much of the key people (carrier, passer, scorer) and
how much of the puck's path and the net mouth each camera sees unoccluded and in frame, how much of the frame is ice (the arena texture
only exists as the broadcast camera saw it, so near stands seen from elsewhere smear), how much of the key people it shows from the
broadcast's side when the broadcast saw them (their tracking, bodies and textures are best there), and a moderately elevated viewpoint. Cameras
hang over the rink like an arena's cable camera. A camera stands still on the chosen spot and pans along a cubic fit of the action with
at most a slow, linear zoom, so it moves like an operator, never jitters.

  bundle/shots.json        per shot: frames, the camera per frame (fx at `width`, R, C), the beat, its score and runner-ups
  check/director.jpg       the top candidates of each beat as point-cloud previews

usage (from pipeline/):  director.py [shot...]   (drone, puckcam, goalcam, celebrate, cuts: plan only those again, in the existing bundle/shots.json)"""

import json
import math
import os
import sys

import cv2
import numpy as np

B = os.environ.get("BUNDLE", "bundle")
BODIES = os.environ.get("BODIES", "bodies")  # per-frame fits (body.py) or bodies_fit (fit.py)
S = json.load(open(f"{B}/scene.json"))
Z = np.load(f"{B}/scene.npz")
N, FPS, HX, HY, CR = S["n"], S["fps"], S["hx"], S["hy"], S["corner"]
GOAL_X = HX - 4.0
W, H = 1280, 768
FOV = 50.0  # degrees, horizontal
FX = W / 2 / math.tan(math.radians(FOV) / 2)
MAX_FRAMES, MIN_APART = 121, 70.0
SHORT = 73  # frames of the drone, POV and bullet-time shots: 3 s at 25 fps, 8k+1
GRID = (160, 96)  # z-buffer resolution for visibility
PASS_BACK, CONTACT, HELD = 100, 2.0, 3  # frames before the shot the pass is looked for in, m a stick reaches, frames held
EVIDENCE = 2.0  # weight of the broadcast's evidence of the key people in a static shot's score


def look_at(C, target):
    fwd = target - C
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(fwd, right), fwd])


def rink_sd(xy):
    q = np.abs(xy) - [HX - CR, HY - CR]
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0) - CR


def inside(xy, margin):
    """xy moved onto the rink shrunk by margin (m) when it is outside it: the nearest such point."""
    core = np.clip(xy, [-(HX - CR), -(HY - CR)], [HX - CR, HY - CR])
    d = xy - core
    r = np.linalg.norm(d)
    return core + d * (CR - margin) / r if r > CR - margin else xy


def placeable(C):
    """Over the rink, a metre inside the boards, 3 to 12 m up: where an arena's cable camera flies."""
    return rink_sd(C[:2]) < -1.0 and 3.0 <= C[2] <= 12.0


def ice_share(fx, R, C):
    """Share of the frame that shows ice: rays on a coarse pixel grid hitting the ice plane inside the boards."""
    u, v = np.meshgrid(np.linspace(0, W, 32), np.linspace(0, H, 20))
    d = np.stack([(u - W / 2) / fx, (v - H / 2) / fx, np.ones_like(u)], -1).reshape(-1, 3) @ R
    t = -C[2] / np.where(d[:, 2] < -1e-6, d[:, 2], -1e-6)
    hit = C[:2] + t[:, None] * d[:, :2]
    return float(((d[:, 2] < -1e-6) & (rink_sd(hit) < -0.3)).mean())


class Scene:
    def __init__(self):
        cast = json.load(open(f"{B}/cast.json"))
        self.tracks = {t["key"]: t for t in cast["tracks"]}
        b = np.load(f"{B}/{BODIES}.npz")
        sub = np.arange(0, b["verts"].shape[1], 40)
        self.pts = {}  # (key, frame) -> body points
        for k, f, v in zip(b["key"], b["frame"], b["verts"][:, sub].astype(np.float32)):
            self.pts[(int(k), int(f))] = v
        self.xy = {k: np.array(t["xy"]) for k, t in self.tracks.items()}
        P = json.load(open(f"{B}/puck.json"))
        self.puck = np.array(P["xy"])
        self.shot = P["shot"]
        self.cam = Z["cam"][:, 11:14]  # the broadcast camera's centre per frame

    def body(self, k, f):
        """Body points of person k at frame f: the mesh when there is one, else a 1.8 m column at their position."""
        if (k, f) in self.pts:
            return self.pts[(k, f)]
        if not self.tracks[k]["seen"][f]:
            return None
        x, y = self.xy[k][f]
        z = np.linspace(0.05, 1.8, 24)
        return np.c_[np.full(24, x) + 0.2 * np.cos(z * 9), np.full(24, y) + 0.2 * np.sin(z * 9), z]


def beats(sc):
    """The goal's beats as frame ranges with their key people, from the puck and the goal event."""
    shot = sc.shot
    rel, net_f, scorer = shot["release"], shot["in_net"], shot["scorer"]
    team = sc.tracks[scorer]["team"]
    mates = [k for k, t in sc.tracks.items() if t["team"] == team and k != scorer and not t["goalie"]]
    near = lambda k, f: np.linalg.norm(sc.xy[k][f] - sc.puck[f])
    # the passer: goal.json's, else the teammate who last held the puck (nearest it, within CONTACT, for HELD frames) in the PASS_BACK frames
    # before the shot, the scorer often carrying it a second or two after the pass; the pass is their last touch. With no such
    # teammate, the one nearest the puck on average and a pass 8 frames before the shot
    G = json.load(open(f"{B}/goal.json")) if os.path.exists(f"{B}/goal.json") else {}
    if "passer" in G:
        passer, pass_f = next(k for k in mates if sc.tracks[k]["jersey"] == G["passer"]), G["pass_frame"]
    else:
        window = range(max(rel - PASS_BACK, 0), rel)
        skaters = [k for k, t in sc.tracks.items() if not t["goalie"] and t["team"] >= 0]
        holder = {f: min((k for k in skaters if sc.tracks[k]["seen"][f]), key=lambda k: near(k, f)) for f in window}
        held = {k: fs for k in mates if len(fs := [f for f in window if holder[f] == k and near(k, f) < CONTACT]) >= HELD}
        if held:
            passer = max(held, key=lambda k: held[k][-1])
            pass_f = held[passer][-1]
        else:
            passer, pass_f = (
                min(mates, key=lambda k: np.mean([near(k, f) for f in window if sc.tracks[k]["seen"][f]] or [np.inf])),
                rel - 8,
            )
    goalie = next(k for k, t in sc.tracks.items() if t["goalie"] and np.sign(t["xy"][rel][0]) == np.sign(sc.puck[net_f][0]))
    net = np.array([np.sign(sc.puck[net_f][0]) * GOAL_X, 0.0, 0.6])
    start = max(pass_f - 75, 0)
    return [
        dict(name="approach", frames=(start, pass_f + 6), keys=[passer], puck_weight=1.0, net=net),
        dict(
            name="goal",
            frames=(max(pass_f - 20, 0), min(net_f + 50, N - 1)),
            keys=[passer, scorer, goalie],
            puck_weight=3.0,
            net=net,
            focus=(pass_f, net_f),
        ),
        dict(
            name="replay",
            frames=(max(pass_f - 20, 0), min(net_f + 50, N - 1)),
            keys=[passer, scorer, goalie],
            puck_weight=3.0,
            net=net,
            focus=(pass_f, net_f),
        ),
    ], dict(passer=passer, scorer=scorer, goalie=goalie, pass_frame=pass_f, release=rel, in_net=net_f)


def action(sc, beat, f):
    """Points the camera should keep in view at frame f: the puck, the key people near it (feet and heads), the net when the puck is
    near it. People far from the puck do not widen the frame."""
    pts = [[*sc.puck[f], 0.0]]
    for k in beat["keys"]:
        if sc.tracks[k]["seen"][f] and np.linalg.norm(sc.xy[k][f] - sc.puck[f]) < 10:
            x, y = sc.xy[k][f]
            pts += [[x, y, 0.0], [x, y, 1.8]]
    if np.linalg.norm(sc.puck[f] - beat["net"][:2]) < 12:
        pts.append(beat["net"])
    return np.array(pts)


def project(fx, R, C, P):
    pc = (P - C) @ R.T
    z = pc[:, 2]
    return np.c_[fx * pc[:, 0] / np.maximum(z, 1e-3) + W / 2, fx * pc[:, 1] / np.maximum(z, 1e-3) + H / 2], z


OFFS = np.array([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])  # 3x3 splat, so sparse body points close up into a surface


def visible(fx, R, C, people, targets):
    """Share of each target point set that is in frame and not behind anyone else, by a coarse z-buffer of everyone's body points."""
    gw, gh = GRID
    keys = np.concatenate([np.full(len(P), k) for k, P in people.items()])
    P = np.concatenate(list(people.values()))
    uv, z = project(fx, R, C, P)
    ok = z > 0.3
    g = (uv[ok][:, None, :] * [gw / W, gh / H]).astype(int) + OFFS[None]
    zz = np.repeat(z[ok], 9)
    kk = np.repeat(keys[ok], 9)
    g = g.reshape(-1, 2)
    inb = (g[:, 0] >= 0) & (g[:, 0] < gw) & (g[:, 1] >= 0) & (g[:, 1] < gh)
    flat = g[inb, 1] * gw + g[inb, 0]
    zz, kk = zz[inb], kk[inb]
    zbuf = np.full(gw * gh, np.inf)
    np.minimum.at(zbuf, flat, zz)
    owner = np.full(gw * gh, -1)
    o = np.argsort(-zz)
    owner[flat[o]] = kk[o]  # nearest point written last wins
    out = {}
    for name, (k, T) in targets.items():
        uv, z = project(fx, R, C, T)
        inside = (z > 0.3) & (uv[:, 0] >= W * 0.04) & (uv[:, 0] < W * 0.96) & (uv[:, 1] >= H * 0.04) & (uv[:, 1] < H * 0.96)
        if not inside.any():
            out[name] = 0.0
            continue
        cell = np.clip((uv[:, 1] * gh / H).astype(int), 0, gh - 1) * gw + np.clip((uv[:, 0] * gw / W).astype(int), 0, gw - 1)
        front = (z <= zbuf[cell] + 0.35) | (owner[cell] == k)
        out[name] = float((inside & front).mean())
    return out


def path(sc, beat, C, fr):
    """A still camera at C that pans and zooms along a cubic fit of the action, keeping it within ~70% of the frame width."""
    t = np.arange(len(fr))
    cent, spread = [], []
    for f in fr:
        a = action(sc, beat, f)
        cent.append(a.mean(0))
        spread.append(np.ptp(a[:, :2], 0).max())
    cent = np.array(cent)
    cent[:, 2] = 0.8
    fit = np.stack([np.polyval(np.polyfit(t, cent[:, d], 3), t) for d in range(3)], 1)
    dist = np.linalg.norm(fit - C, axis=1)
    want = np.polyval(np.polyfit(t, np.clip(np.array(spread) + 2.5, 5.0, 40.0), 1), t)  # metres of action across the frame
    fx = np.clip(0.7 * W * dist / want, FX * 0.7, FX * 3.5)
    fx = np.clip(fx, np.median(fx) * 0.8, np.median(fx) * 1.2)  # a slow zoom at most
    return [(float(f_), look_at(C, c), C) for f_, c in zip(fx, fit)]


def score(sc, beat, cams, fr):
    keys, s, n = beat["keys"], 0.0, 0
    focus = beat.get("focus", beat["frames"])
    for (fx, R, C), f in list(zip(cams, fr))[::4]:
        people = {k: P for k in sc.tracks if (P := sc.body(k, f)) is not None}
        targets = {f"k{k}": (k, people[k]) for k in keys if k in people}
        targets["puck"] = (-2, np.array([[*sc.puck[f], 0.03]]))
        targets["net"] = (-3, beat["net"][None])
        v = visible(fx, R, C, people, targets)
        w_puck = beat["puck_weight"] * (2.0 if focus[0] - 3 <= f <= focus[1] + 3 else 1.0)
        s += (
            sum(v.get(f"k{k}", 0) for k in keys)
            + w_puck * v["puck"]
            + (1.0 if np.linalg.norm(sc.puck[f] - beat["net"][:2]) < 12 else 0.2) * v["net"]
        )
        n += 1
    fx, R, C = cams[len(cams) // 2]
    elev = math.degrees(math.asin(-R[2][2]))
    return (
        s / max(n, 1)
        + 1.5 * min(ice_share(fx, R, C) / 0.7, 1.0)
        - 0.3 * ((elev - 28) / 12) ** 2
        + EVIDENCE * evidence(sc, beat, cams, fr)
    )


def evidence(sc, beat, cams, fr):
    """How much of the key people the broadcast saw, as this camera shows them: per key person in frame, seen at that frame, times
    how close the view of them is to the broadcast camera's ((1 + cos) / 2 of the angle between the two directions to them). Bodies
    and their textures are tracked and fit from the broadcast's side; elsewhere they are held, borrowed or mirrored."""
    e, n = 0.0, 0
    for (fx, R, C), f in list(zip(cams, fr))[::4]:
        for k in beat["keys"]:
            X = np.r_[sc.xy[k][f], 1.0]
            uv, z = project(fx, R, C, X[None])
            if z[0] < 0.3 or not (0 <= uv[0, 0] < W and 0 <= uv[0, 1] < H):
                continue
            b, c = X - sc.cam[f], X - C
            e += sc.tracks[k]["seen"][f] * (1 + b @ c / np.linalg.norm(b) / np.linalg.norm(c)) / 2
            n += 1
    return e / max(n, 1)


def candidates(sc, beat, fr):
    cen = np.mean([action(sc, beat, f).mean(0) for f in fr], 0)
    spread = max(np.ptp(np.concatenate([action(sc, beat, f) for f in fr])[:, :2], 0).max(), 6.0)
    for az in range(0, 360, 15):
        for el in (15, 22, 30, 40):
            for dmul in (0.7, 1.0, 1.3):
                d = dmul * spread / 2 / math.tan(math.radians(FOV) / 2) / 0.6
                d = float(np.clip(d, 7.0, 35.0))
                C = cen + d * np.array(
                    [
                        math.cos(math.radians(el)) * math.cos(math.radians(az)),
                        math.cos(math.radians(el)) * math.sin(math.radians(az)),
                        math.sin(math.radians(el)),
                    ]
                )
                C[2] += 0.8
                if placeable(C):
                    yield az, el, d, C


def preview(sc, beat, cams, fr, label):
    """A point-cloud sketch of the shot's middle frame: people (key people highlighted), the puck path, the net and the rink outline."""
    f = fr[len(fr) // 2]
    fx, R, C = cams[len(fr) // 2]
    img = np.full((H // 2, W // 2, 3), 235, np.uint8)
    s = np.linspace(0, 1, 400)
    rim = np.c_[np.clip(np.cos(s * 2 * np.pi) * 1e3, -HX, HX), np.clip(np.sin(s * 2 * np.pi) * 1e3, -HY, HY), np.zeros(400)]
    for P, col, r in [
        (rim, (150, 150, 150), 1),
        (np.c_[sc.puck[fr], np.zeros(len(fr))], (0, 0, 0), 2),
        (beat["net"][None], (0, 0, 220), 6),
    ]:
        uv, z = project(fx, R, C, P)
        for (x, y), zz in zip(uv / 2, z):
            if zz > 0.3 and 0 <= x < W / 2 and 0 <= y < H / 2:
                cv2.circle(img, (int(x), int(y)), r, col, -1)
    for k in sc.tracks:
        P = sc.body(k, f)
        if P is None:
            continue
        uv, z = project(fx, R, C, P)
        col = (0, 140, 255) if k in beat["keys"] else ((200, 120, 40) if sc.tracks[k]["team"] == 0 else (40, 190, 230))
        for (x, y), zz in zip(uv / 2, z):
            if zz > 0.3 and 0 <= x < W / 2 and 0 <= y < H / 2:
                cv2.circle(img, (int(x), int(y)), 1, col, -1)
    cv2.putText(img, label, (8, 22), 0, 0.55, (0, 0, 0), 1)
    return img


def orbit_pos(c, az, el, r):
    """On the orbit around c, pulled in toward c where the orbit would leave the rink."""
    d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    while r > 3 and rink_sd((c + r * d)[:2]) > -1.5:
        r -= 0.25
    return c + r * d


def orbit_360(sc, info, el_deg, r):
    """121 frozen frames at the release, one full turn."""
    f, scorer = info["release"], info["scorer"]
    c = np.r_[(sc.xy[scorer][f] + sc.puck[f]) / 2, 0.9]
    cams = []
    for k in range(121):
        C = orbit_pos(c, 2 * math.pi * k / 121, math.radians(el_deg), r)
        cams.append((FX, look_at(C, c), C))
    return [f] * 121, cams


def bullet_time(sc, info, goal_cam, budget):
    """Frames and cameras: play from just before the pass reaches the scorer to the release, orbit 120 degrees frozen, play until the
    puck is in the net, orbit 45 degrees frozen; the two orbits share the frames the budget leaves, 3 to 2."""
    rel, net_f, scorer = info["release"], info["in_net"], info["scorer"]
    centre = lambda f: np.r_[(sc.xy[scorer][f] + sc.puck[f]) / 2, 0.9]
    az0 = math.radians(goal_cam["azimuth"])
    el, r = math.radians(25), 13.0
    pos = lambda az, c: orbit_pos(c, az, el, r)
    ease = lambda u: 0.5 - 0.5 * math.cos(math.pi * u)
    frames, cams = [], []

    def play(f0, f1, az):
        for f in range(f0, f1):
            c = centre(f)
            C = pos(az, c)
            frames.append(f)
            cams.append((FX, look_at(C, c), C))

    def orbit(f, az_a, az_b, n):
        c = centre(f)
        for k in range(n):
            az = az_a + (az_b - az_a) * ease(k / (n - 1))
            C = pos(az, c)
            frames.append(f)
            cams.append((FX, look_at(C, c), C))

    a1 = az0 + math.radians(120)
    a2 = a1 + math.radians(45)
    start = max(rel - min(rel - info["pass_frame"] + 4, 16), 0)
    free = budget - (rel - start) - (net_f + 4 - rel)
    play(start, rel, az0)
    orbit(rel, az0, a1, round(free * 0.6))
    play(rel, net_f + 4, a1)
    orbit(net_f + 4, a1, a2, budget - len(frames))
    # the play segments follow the action's centre; a short smoothing of the look-at point removes the steps at the freezes
    tgt = np.array([C + R[2] * np.linalg.norm(centre(f) - C) for (fx, R, C), f in zip(cams, frames)])
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2)
    k /= k.sum()
    tgt = np.stack([np.convolve(np.pad(tgt[:, d], 6, mode="edge"), k, "valid") for d in range(3)], 1)
    return frames, [(fx, look_at(C, t), C) for (fx, R, C), t in zip(cams, tgt)]


def retime(f0, f1, slow_a, slow_b, speed, ramp=8.0, max_frames=MAX_FRAMES):
    """Fractional source-frame times from f0 to f1: real time, easing to `speed` over `ramp` source frames before slow_a, held to
    slow_b, easing back. Cut to 8k+1 frames."""

    def v(t):
        e = lambda u: 0.5 - 0.5 * math.cos(math.pi * min(max(u, 0.0), 1.0))
        return 1 - (1 - speed) * (e((t - (slow_a - ramp)) / ramp) if t < slow_b else 1 - e((t - slow_b) / ramp))

    ts, t = [], float(f0)
    while t <= f1 and len(ts) < max_frames:
        ts.append(round(t, 3))
        t += v(t)
    return ts[: (len(ts) - 1) // 8 * 8 + 1]


def fit(f0, f1, slow_a, slow_b, speed, budget, ramp=8.0):
    """retime() from f0 to f1 within `budget` frames: slowed to `speed` if that fits, else as little faster as fits, else real time
    starting late enough to end at f1."""
    for s_ in np.arange(speed, 1.0001, 0.05):
        ts = retime(f0, f1, slow_a, slow_b, s_, ramp, budget)
        if ts[-1] >= f1 - 1:
            return ts
    return retime(max(f1 - budget + 1, f0), f1, slow_a, slow_b, 1.0, ramp, budget)


def hermite(keys, t):
    """Cubic Hermite (Catmull-Rom tangents) through keys [(time, vector)] at time t; ends held."""
    ts = [k[0] for k in keys]
    ps = [np.asarray(k[1], float) for k in keys]
    if t <= ts[0]:
        return ps[0]
    if t >= ts[-1]:
        return ps[-1]
    i = max(j for j in range(len(ts) - 1) if ts[j] <= t)
    h = ts[i + 1] - ts[i]
    u = (t - ts[i]) / h
    tan = lambda j: (
        (ps[min(j + 1, len(ps) - 1)] - ps[max(j - 1, 0)]) / max(ts[min(j + 1, len(ts) - 1)] - ts[max(j - 1, 0)], 1e-6) * h
    )
    m0, m1 = tan(i), tan(i + 1)
    return (
        (2 * u**3 - 3 * u**2 + 1) * ps[i] + (u**3 - 2 * u**2 + u) * m0 + (-2 * u**3 + 3 * u**2) * ps[i + 1] + (u**3 - u**2) * m1
    )


def lerp_xy(x, t):
    i0 = min(int(t), len(x) - 2)
    a = t - i0
    return x[i0] * (1 - a) + x[i0 + 1] * a


def goal_slow(sc, info, goal):
    """The goal shot's camera, re-timed: quarter speed from the pass (or the last 0.8 s before the release, after a long carry) until
    the puck is in the net; slowed less when that would not fit the frame budget."""
    fr0 = goal["frames"]
    cams0 = goal["cams"]
    net = info["in_net"] + 6
    slow_a = max(info["pass_frame"], info["release"] - 20)
    f0, f1 = max(slow_a - 16, fr0[0]), min(info["in_net"] + 30, fr0[-1])
    for speed in np.arange(0.25, 1.01, 0.05):
        ts = retime(f0, f1, slow_a, net, speed)
        if ts[-1] >= min(net, f1):
            break
    cams = []
    for t in ts:
        k = min(max(t - fr0[0], 0), len(cams0) - 1.001)
        i0 = int(k)
        a = k - i0
        c0, c1 = cams0[i0], cams0[min(i0 + 1, len(cams0) - 1)]
        C = (1 - a) * np.array(c0["C"]) + a * np.array(c1["C"])
        fwd = (1 - a) * np.array(c0["R"][2]) + a * np.array(c1["R"][2])
        cams.append(((1 - a) * c0["fx"] + a * c1["fx"], look_at(C, C + fwd), C))
    return ts, cams


HEAD, BLOCK_W, NEAR, MAX_LIFT = 2.1, 0.6, 3.0, 4.0  # m: sight line height over a blocker, blocker half-width, closer than this
# anywhere in view also blocks, highest the drone rises
FILL, MAX_FWD, MAX_SIDE = 0.10, 2.0, 2.5  # after the shot no skater may cover more of the frame than FILL: the drone flies
# on past him or sideways, by the least it can (m)
WALL = 3.0  # m the drone keeps inside the boards (its path is set from the scorer and the
# net, which can put it over the boards or the stands; nearer than this the stands,
# a flat texture, smear across the frame)
HALF_FOV_TAN = math.tan(math.radians(75) / 2)
JERK, TURN, LOOK, DIP = 1e-5, 1e-3, 4.0, 45.0  # weight of the drone's jerk ((m/s^3)^2, per shown frame) against its distance
# from the planned path (m^2), and of its view's against its turn from the planned
# view (rad^2); m it looks at least ahead, degrees it looks down at most (nearer
# or steeper, the look-at point swings the view round)
MAX_BANK, G = 4.0, 9.81  # degrees it rolls into a turn at most (the gimbal holds the horizon); m/s^2
A_MAX, PAN_MAX = 4.0, 45.0  # m/s^2 it accelerates, degrees/s its view turns, at most (95th percentile)
MOMENTUM = 0.6  # of its mean speed along its way it keeps at least, so it never stalls
CEILING = 7.4  # m it flies under at most: the arena's scoreboard cube hangs over centre ice from
# 8.46 m, and scoreboard.py keeps planned cameras 1 m below it


def glide(X, floor=None, wall=None, weight=JERK, onward=None, pace=0.0, ceiling=None):
    """X (shown frames, d) moved to the path of least jerk near it, as a drone flies: its acceleration changes gradually. floor: the
    lowest the last coordinate may go, frame by frame, and ceiling the highest; wall: m the first two keep inside the boards, frame by frame; onward: a
    horizontal direction it keeps flying along at pace m/s at least (momentum: it never stalls or backs up). All three are convex, so
    ADMM solves it exactly: the path's linear solve, then each point put back over the floor and inside the wall, and each step's
    progress along onward made at least pace."""
    n = len(X)
    D = np.diff(np.eye(n), 3, axis=0) * FPS**3
    A = np.eye(n) + weight * D.T @ D
    X = np.array(X, float)
    if floor is None and wall is None and onward is None:
        return np.linalg.solve(A, X)
    step = np.diff(np.eye(n), axis=0)
    way = np.zeros((X.shape[1], X.shape[1])) if onward is None else np.outer([*onward, 0], [*onward, 0])
    # (A + I) P + step' step P way = rhs, solved in the basis of way's eigenvectors, where it splits into one system per coordinate
    lam, E = np.linalg.eigh(way)
    solve = [np.linalg.inv(A + np.eye(n) + l * step.T @ step) for l in lam]

    def fit(P):
        if floor is not None:
            P[:, -1] = np.clip(P[:, -1], floor, ceiling)
        if wall is not None:
            P[:, :2] = [inside(p[:2], m + 0.05) for p, m in zip(P[:, :2], wall)]
        return P

    P = np.linalg.solve(A, X)
    Y, U = fit(P.copy()), np.zeros_like(P)
    S = step @ P @ way
    V = np.zeros_like(S)
    for _ in range(5000):
        rhs = (X + Y - U + step.T @ (S - V) @ way) @ E
        P = np.stack([M @ r for M, r in zip(solve, rhs.T)], 1) @ E.T
        Y = fit(P + U)
        SP = step @ P @ way
        S = SP + V
        if onward is not None:
            S = np.outer(np.maximum(S[:, :2] @ onward, pace / FPS), [*onward, 0])
        U += P - Y
        V += SP - S
        if max(np.abs(P - Y).max(), np.abs(SP - S).max()) < 1e-3:
            break
    return P


def steady(solve, weight, rate, most, keep=True):
    """solve(weight), with the weight raised fourfold at a time until the 95th percentile of rate(result) is at most `most`, or
    (keep) a raise lowers it by under 5% (what is left is the plan's own, which more weight would only flatten away)."""
    P, r = solve(weight), np.inf
    for _ in range(12):
        q = np.percentile(rate(P), 95)
        if q <= most or (keep and q > 0.95 * r):
            break
        weight *= 4
        r = q
        P = solve(weight)
    return P


def bank(R, Cs):
    """Camera rotations R rolled into the drone's turns: by the angle its sideways acceleration tilts its thrust from vertical,
    at most MAX_BANK."""
    a = np.gradient(np.gradient(Cs, axis=0), axis=0) * FPS**2
    out = []
    for r, acc in zip(R, a):
        side = r[0] - r[0] @ [0, 0, 1.0] * np.array([0, 0, 1.0])
        side /= max(np.linalg.norm(side), 1e-6)
        phi = math.radians(np.clip(math.degrees(math.atan2(acc @ side, G)), -MAX_BANK, MAX_BANK))
        out.append(np.stack([math.cos(phi) * r[0] + math.sin(phi) * r[1], math.cos(phi) * r[1] - math.sin(phi) * r[0], r[2]]))
    return out


def drone(sc, info, budget):
    """A camera drone flown smoothly through the play (see the module docstring)."""
    pf, rel, nf, sco, pas = info["pass_frame"], info["release"], info["in_net"], info["scorer"], info["passer"]
    net = np.array([np.sign(sc.puck[nf][0]) * GOAL_X, 0.0])
    u = net - sc.xy[sco][rel]
    u /= np.linalg.norm(u)
    if u[0] * net[0] < 0:
        u = np.array([np.sign(net[0]), 0.0])  # scored from behind the goal line: fly toward the end, ending in front
    perp = np.array([-u[1], u[0]])
    side = 1.0 if np.dot(sc.xy[pas][pf] - sc.xy[sco][rel], perp) < 0 else -1.0  # fly on the side away from the passer
    t0, t3 = max(pf - 30, 0), min(nf + 14, N - 1)
    keys_C = [
        (t0, [*(sc.xy[pas][t0] - 9 * u - 3 * side * perp), 7.0]),
        (pf, [*((sc.xy[pas][pf] + sc.xy[sco][pf]) / 2 - 4.5 * u + 1.5 * side * perp), 4.0]),
        (rel, [*(sc.xy[sco][rel] - 2.6 * u + 2.0 * side * perp), 1.7]),
        (t3, [*(net - 2.8 * u + 1.0 * side * perp), 1.1]),
    ]
    keys_T = [
        (t0, [*sc.xy[pas][t0], 0.6]),
        (pf, [*sc.puck[pf], 0.3]),
        (rel, [*(net - 0.3 * u), 0.5]),
        (t3, [*(net + 0.5 * u), 0.5]),
    ]
    ts = fit(t0, t3, rel - 4, nf + 4, 0.35, budget, ramp=10.0)
    Cs = np.array([hermite(keys_C, t) for t in ts])
    Ts = np.array([hermite(keys_T, t) for t in ts])
    Cs[:, :2] = [inside(C[:2], WALL) for C in Cs]
    floor = Cs[:, 2].copy()
    for j, t in enumerate(ts):  # lift over anyone the drone would fly through, a skater blocking its view of the
        # target or one close enough in view to fill the frame (pulling up over them)
        f = int(round(t))
        view = Ts[j, :2] - Cs[j, :2]
        D = np.linalg.norm(view)
        view /= D
        for k, tr in sc.tracks.items():  # everyone the render draws, seen in the broadcast at f or not
            v = sc.xy[k][f] - Cs[j, :2]
            along = v @ view
            if np.linalg.norm(v) < 1.1:
                floor[j] = max(floor[j], 2.3)
            elif (
                not tr["goalie"]
                and 0 < along < D - 0.5
                and np.linalg.norm(v - along * view) < BLOCK_W + (along < NEAR) * along * HALF_FOV_TAN
            ):
                a = along / D  # the sight line clears the head: C_z (1 - a) + T_z a >= HEAD
                floor[j] = max(floor[j], min((HEAD - Ts[j, 2] * a) / (1 - a), MAX_LIFT))
    fx = W / 2 / math.tan(math.radians(75) / 2)
    floor = np.minimum(floor, CEILING)
    Cs[:, 2] = floor
    tt = np.array(ts)
    after = np.flatnonzero(tt >= rel + 4)  # after the shot the net is the subject: no skater may fill the frame, so the drone
    if len(after):  # eases into one move on past or beside them, the least that does it
        ease = np.clip((tt - tt[after[0]]) / 12, 0, 1)
        ease = ease * ease * (3 - 2 * ease)
        view = Ts[after[-1], :2] - Cs[after[-1], :2]
        D = np.linalg.norm(view)
        view /= D
        perp_v = np.array([-view[1], view[0]])
        skaters = [
            [
                sc.xy[k][int(round(ts[j]))]
                for k, tr in sc.tracks.items()
                if not tr["goalie"] and np.isfinite(sc.xy[k][int(round(ts[j]))]).all()
            ]
            for j in after
        ]
        edge = [max(rink_sd(C[:2]), -0.5) for C in Cs[after]]  # no nearer the boards than 0.5 m, or than the path already is

        def excess(d):
            Cm = Cs[after] + ease[after, None] * [*d, 0]
            if any(rink_sd(C[:2]) > e for C, e in zip(Cm, edge)):
                return np.inf
            return sum(
                max(max((fill(fx, C, Ts[j], p) for p in ps), default=0.0) - FILL, 0) for C, j, ps in zip(Cm, after, skaters)
            )

        best = (excess(np.zeros(2)), np.zeros(2))  # frame share above FILL, summed over the frames after the shot
        if best[0] > 0:
            for m in sorted(
                ((a, s) for a in np.arange(0, MAX_FWD + 1e-6, 0.25) for s in np.arange(-MAX_SIDE, MAX_SIDE + 1e-6, 0.25)),
                key=lambda m: math.hypot(*m),
            )[1:]:
                if m[0] > D - 1.5:
                    continue
                d = m[0] * view + m[1] * perp_v
                w = excess(d)
                if w < best[0]:
                    best = (w, d)
                if w == 0:
                    break
        Cs[:, :2] += ease[:, None] * best[1]
    g = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
    floor = np.convolve(np.pad(floor, 8, mode="edge"), g / g.sum(), "valid")
    wall = np.clip(-rink_sd(Cs[:, :2]), 0.5, WALL)  # as far inside as the move past the skaters leaves it
    first = glide(Cs, floor, wall, ceiling=CEILING)
    travel = first[-1, :2] - first[0, :2]
    way = np.linalg.norm(travel)
    travel /= max(way, 1e-6)
    pace = MOMENTUM * way / (len(Cs) - 1) * FPS  # the flight's way, and the speed it keeps along it
    Cs = steady(
        lambda w: glide(Cs, floor, wall, w, travel, pace, CEILING),
        JERK,
        lambda P: np.linalg.norm(np.gradient(np.gradient(P, axis=0), axis=0), axis=1) * FPS**2,
        A_MAX,
    )
    ahead = Ts - Cs
    d = np.linalg.norm(ahead[:, :2], axis=1, keepdims=True)
    ahead[:, :2] *= np.maximum(d, np.maximum(LOOK, -ahead[:, 2:] / math.tan(math.radians(DIP)))) / d
    unit = ahead / np.linalg.norm(ahead, axis=1, keepdims=True)
    pan = lambda V: (
        np.degrees(
            np.arccos(np.clip(np.sum(V[1:] * V[:-1], 1) / np.linalg.norm(V[1:], axis=1) / np.linalg.norm(V[:-1], axis=1), -1, 1))
        )
        * FPS
    )
    view = steady(lambda w: glide(unit, weight=w), TURN, pan, PAN_MAX, keep=False)
    Ts = Cs + view
    return ts, list(zip([fx] * len(Cs), bank([look_at(C, T) for C, T in zip(Cs, Ts)], Cs), Cs))


POV_EYE, POV_AHEAD, POV_FOV, POV_DIP = 1.55, 0.3, 90.0, 0.25  # m above the ice, m ahead of the player's path, degrees horizontal
# (helmet camera), look-at drop per m of distance


def pov(sc, info, budget):
    """The scorer's helmet camera: at eye height, a little ahead of his (smoothed) path, looking at the puck as the pass comes and
    turning to the net for the shot; from 1 s before the pass until the puck has settled in the net, slowing to half speed through
    the shot. The render moves the camera to his fitted eyes and draws only his stick (the shot's `own`)."""
    pf, rel, nf, sco = info["pass_frame"], info["release"], info["in_net"], info["scorer"]
    net = np.array([np.sign(sc.puck[nf][0]) * GOAL_X, 0.0, 0.6])
    t0, t1 = max(pf - 25, 0), min(nf + 15, N - 1)
    ts = fit(t0, t1, rel - 4, nf + 4, 0.5, budget)
    g = np.exp(-0.5 * (np.arange(-6, 7) / 3.0) ** 2)
    g /= g.sum()
    xy = np.stack([np.convolve(np.pad(sc.xy[sco][:, d], 6, mode="edge"), g, "valid") for d in range(2)], 1)
    keys_T = [(t0, [*sc.puck[t0], 0.1]), (pf, [*sc.puck[pf], 0.1]), (rel - 2, net), (t1, net)]
    fx = W / 2 / math.tan(math.radians(POV_FOV) / 2)
    cams = []
    for t in ts:
        P = lerp_xy(xy, t)
        T = hermite(keys_T, t)
        d = T[:2] - P
        d /= max(np.linalg.norm(d), 1e-6)
        C = np.r_[P + POV_AHEAD * d, POV_EYE]
        T = T - [0, 0, POV_DIP * np.linalg.norm(T[:2] - C[:2])]  # a skater's gaze, a little down
        cams.append((fx, look_at(C, T), C))
    return ts, cams


PUCK_BACK, PUCK_UP, PUCK_AHEAD, PUCK_FOV, PUCK_STOP = 2.2, 0.45, 2.5, 75.0, 2.5  # m behind the puck, above the ice, looked ahead
# of it; degrees; m short of the goal line it stops (the crease's top)
PUCK_CLEAR = 0.8  # m it keeps from any skater


def puck_cam(sc, info, budget):
    """Riding PUCK_BACK behind the puck at PUCK_UP, looking PUCK_AHEAD ahead along where it goes (its smoothed heading, held while it
    barely moves), from 15 frames before the release until the puck has settled in the net, slowing to a quarter through the
    shot. It keeps inside the boards (nearer the puck where the boards are close), PUCK_CLEAR from skaters (the carrier is right
    behind the puck), and stops PUCK_STOP short of the goal line, so the puck flies on into the net."""
    rel, nf = info["release"], info["in_net"]
    side = np.sign(sc.puck[nf][0])
    t0, t1 = max(rel - 15, 0), min(nf + 10, N - 1)
    ts = fit(t0, t1, rel - 3, nf + 3, 0.25, budget)
    g = lambda s: (lambda k: k / k.sum())(np.exp(-0.5 * (np.arange(-3 * s, 3 * s + 1) / s) ** 2))
    smooth = lambda x, s: np.stack(
        [np.convolve(np.pad(x[:, d], 3 * s, mode="edge"), g(s), "valid") for d in range(x.shape[1])], 1
    )
    p = smooth(sc.puck, 2)
    d = np.zeros_like(p)
    last = np.array([side, 0.0])
    for f in range(N):
        v = p[min(f + 4, N - 1)] - p[max(f - 4, 0)]
        if np.linalg.norm(v) > 0.15:
            last = v / np.linalg.norm(v)
        d[f] = last
    d = smooth(d, 4)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    Cs, Ts = [], []
    for t in ts:
        P, D = lerp_xy(p, t), lerp_xy(d, t)
        back = PUCK_BACK
        while back > 0.3 and rink_sd(P - back * D) > -0.4:
            back *= 0.8
        Cs.append(P - back * D)
        Ts.append(P + PUCK_AHEAD * D)

    def clear(Cs):
        for C, t in zip(Cs, ts):
            f = int(round(t))
            C[0] = side * min(side * C[0], GOAL_X - PUCK_STOP)
            for k, t_ in sc.tracks.items():
                v = C - sc.xy[k][f]
                r = np.linalg.norm(v)
                if t_["seen"][f] and r < PUCK_CLEAR:
                    C[:] = sc.xy[k][f] + v / max(r, 1e-3) * PUCK_CLEAR
        return Cs

    Cs, Ts = clear(smooth(clear(np.array(Cs)), 2)), smooth(np.array(Ts), 2)
    fx = W / 2 / math.tan(math.radians(PUCK_FOV) / 2)
    return ts, [(fx, look_at(C, T), C) for C, T in zip(np.c_[Cs, np.full(len(Cs), PUCK_UP)], np.c_[Ts, np.full(len(Ts), 0.1)])]


NET_DEPTH, NET_EYE, NET_FOV = 0.9, 0.3, 95.0  # m behind the goal line, above the ice; degrees horizontal


def goal_cam(sc, info, budget):
    """The in-net camera: fixed NET_DEPTH behind the goal line at NET_EYE, wide, aimed at the puck at the release (a little above the
    ice), from 30 frames before the release until after the goal, slowing to a third through the shot."""
    rel, nf = info["release"], info["in_net"]
    side = np.sign(sc.puck[nf][0])
    C = np.array([side * (GOAL_X + NET_DEPTH), 0.0, NET_EYE])
    ts = fit(max(rel - 30, 0), min(nf + 20, N - 1), rel - 2, nf + 4, 0.33, budget)
    fx = W / 2 / math.tan(math.radians(NET_FOV) / 2)
    R = look_at(C, np.r_[sc.puck[rel], 0.6])
    return ts, [(fx, R, C)] * len(ts)


CELEB_AFTER, CELEB_NEAR, CELEB_DIST, CELEB_UP, CELEB_ARC, CELEB_FOV = 10, 4.0, 5.5, 2.0, 25.0, 50.0
CELEB_PUSH, CELEB_WALL = 0.75, 1.0  # of its distance it eases in to; m it keeps inside the boards
CELEB_CLEAR, CELEB_OFF = (
    1.5,
    (0.0, 30.0, -30.0, 60.0, -60.0),
)  # m from anyone it keeps if it can; degrees off the broadcast's side tried


def celebrate(sc, info):
    """The celebration: from CELEB_AFTER frames after the goal to the end of the clip, at real speed, the scorer and the teammates
    who come to him (within CELEB_NEAR m) framed from the broadcast's side, where their bodies are tracked best: CELEB_DIST m off
    at CELEB_UP, easing in to CELEB_PUSH of that and arcing CELEB_ARC degrees round them. Of the arcs either way, at 0.8 to 1.3
    times the distance and CELEB_OFF degrees off the broadcast's side, the nearest that side that stays CELEB_CLEAR from everyone
    (else the one that stays farthest). The group's centre is a quadratic fit, so the camera glides."""
    nf, sco = info["in_net"], info["scorer"]
    team = sc.tracks[sco]["team"]
    fr = list(range(min(nf + CELEB_AFTER, N - 9), N))
    group = [
        np.mean(
            [sc.xy[sco][f]]
            + [
                sc.xy[k][f]
                for k, t in sc.tracks.items()
                if k != sco
                and t["team"] == team
                and not t["goalie"]
                and t["seen"][f]
                and np.linalg.norm(sc.xy[k][f] - sc.xy[sco][f]) < CELEB_NEAR
            ],
            0,
        )
        for f in fr
    ]
    t = np.arange(len(fr))
    g = np.array(group)
    g = np.stack([np.polyval(np.polyfit(t, g[:, d], 2), t) for d in range(2)], 1)
    e = t / max(t[-1], 1)
    e = e * e * (3 - 2 * e)
    b = sc.cam[fr[0]][:2] - g[0]
    az0 = math.atan2(b[1], b[0])

    def way(turn, off, dmul):
        az = az0 + math.radians(off) + turn * math.radians(CELEB_ARC) * e
        d = dmul * CELEB_DIST * (1 - (1 - CELEB_PUSH) * e)
        return np.array([inside(P, CELEB_WALL) for P in g + d[:, None] * np.c_[np.cos(az), np.sin(az)]])

    def clear(P):
        return min(np.linalg.norm(P[i] - sc.xy[k][f]) for i, f in enumerate(fr) for k, t in sc.tracks.items() if t["seen"][f])

    ways = [
        (abs(off), -min(clear(P), CELEB_CLEAR), P)
        for off in CELEB_OFF
        for turn in (1, -1)
        for dmul in (1.0, 0.8, 1.3)
        for P in [way(turn, off, dmul)]
    ]
    xy = min(ways, key=lambda w: (w[1], w[0]))[2]
    fx = W / 2 / math.tan(math.radians(CELEB_FOV) / 2)
    cams = []
    for P, c in zip(xy, g):
        C = np.r_[P, CELEB_UP]
        cams.append((fx, look_at(C, np.r_[c, 1.2]), C))
    return [float(f) for f in fr], cams


def cuts(shots, info):
    """The cut list: (shot, at) pairs, each part running from the source time `at` to the next part's; a part whose shot does not
    reach its time is left out (the part before runs on). Each part as its shot's frame range [from, to)."""
    pf, rel, nf = info["pass_frame"], info["release"], info["in_net"]
    plan = [
        ("approach", -math.inf),
        ("pov", pf - 20),
        ("puckcam", rel - 6),
        ("goalcam", nf + 3),
        ("celebrate", nf + CELEB_AFTER + 4),
    ]
    index = lambda name, t: next((i for i, f in enumerate(shots[name]["frames"]) if f >= t), None)
    plan = [(name, at) for name, at in plan if name in shots and index(name, at) is not None]
    parts = []
    for (name, at), nxt in zip(plan, plan[1:] + [(None, math.inf)]):
        a = index(name, at)
        b = index(name, nxt[1])
        b = len(shots[name]["frames"]) if b is None else b
        if b > a:
            parts.append(dict(shot=name, frames=[a, b]))
    return parts


def fill(fx, C, T, xy):
    """Share of the frame covered by a skater standing at xy (a 0.6 x 1.9 m box), seen from C looking at T."""
    r = np.cross(T - C, [0, 0, 1.0])[:2]
    r /= max(np.linalg.norm(r), 1e-6)
    P = np.array([[xy[0] + dx * r[0], xy[1] + dx * r[1], z] for dx in (-0.3, 0.3) for z in (0.0, 1.9)])
    uv, z = project(fx, look_at(C, T), C, P)
    uv = uv[z > 0.3]
    if not len(uv):
        return 0.0
    x0, y0 = np.clip(uv.min(0), 0, [W, H])
    x1, y1 = np.clip(uv.max(0), 0, [W, H])
    return (x1 - x0) * (y1 - y0) / (W * H)


def moving(sc, info, shots, names):
    """Plan the shots `names` (goal_slow, drone, pov, puckcam, goalcam, celebrate) into `shots`."""
    plans = dict(
        goal_slow=lambda: (goal_slow(sc, info, shots["goal"]), shots["goal"]["camera"]),
        drone=lambda: (drone(sc, info, SHORT), dict(kind="drone")),
        pov=lambda: (pov(sc, info, SHORT), dict(kind="pov", of=info["scorer"])),
        puckcam=lambda: (puck_cam(sc, info, SHORT), dict(kind="puck")),
        goalcam=lambda: (goal_cam(sc, info, SHORT), dict(kind="net")),
        celebrate=lambda: (celebrate(sc, info), dict(kind="celebrate")),
    )
    for name in names:
        (fr, cams), camera = plans[name]()
        shots[name] = dict(
            frames=fr,
            width=W,
            beat=name,
            keys=[info["passer"], info["scorer"]],
            score=None,
            camera=camera,
            cams=[dict(fx=fx, R=R.round(6).tolist(), C=C.round(4).tolist()) for fx, R, C in cams],
        )
        print(name, len(fr), "frames, source", fr[0], "to", fr[-1], flush=True)
    if "pov" in names:
        shots["pov"]["own"] = info["scorer"]


def main():
    sc = Scene()
    plan, info = beats(sc)
    if sys.argv[1:]:
        shots = json.load(open(f"{B}/shots.json"))
        moving(sc, info, shots, [n for n in sys.argv[1:] if n != "cuts"])
        if "cuts" in sys.argv[1:]:
            shots["cuts"] = dict(kind="cuts", parts=cuts(shots, info))
        json.dump(shots, open(f"{B}/shots.json", "w"))
        return
    shots, sheets, chosen_az = {}, [], None
    for beat in plan:
        a, b = beat["frames"]
        fr = list(range(a, min(b + 1, a + MAX_FRAMES)))
        fr = fr[: (len(fr) - 1) // 8 * 8 + 1]  # the generators work in 8k+1 frames
        ranked = sorted(
            (
                (score(sc, beat, cams := path(sc, beat, C, fr), fr), az, el, d, C, cams)
                for az, el, d, C in candidates(sc, beat, fr)
            ),
            key=lambda r: -r[0],
        )
        if beat["name"] == "replay":
            apart = lambda r: min(abs(r[1] - chosen_az), 360 - abs(r[1] - chosen_az))
            ranked = [r for r in ranked if apart(r) >= MIN_APART] or sorted(ranked, key=lambda r: -apart(r))
        best = ranked[0]
        if beat["name"] == "goal":
            chosen_az = best[1]
        shots[beat["name"]] = dict(
            frames=fr,
            width=W,
            beat=beat["name"],
            keys=beat["keys"],
            score=round(best[0], 3),
            camera=dict(azimuth=best[1], elevation=best[2], distance=round(best[3], 1), C=best[4].round(3).tolist()),
            runner_ups=[dict(score=round(r[0], 3), azimuth=r[1], elevation=r[2]) for r in ranked[1:6]],
            cams=[dict(fx=fx, R=R.round(6).tolist(), C=C.round(4).tolist()) for fx, R, C in best[5]],
        )
        tiles = [
            preview(sc, beat, r[5], fr, f"{beat['name']} #{j + 1} az{r[1]} el{r[2]} s{r[0]:.2f}")
            for j, r in enumerate(ranked[:3])
        ]
        sheets.append(np.hstack(tiles + [np.full_like(tiles[0], 235)] * (3 - len(tiles))))
        print(
            beat["name"],
            fr[0],
            fr[-1],
            "best",
            shots[beat["name"]]["camera"],
            "score",
            shots[beat["name"]]["score"],
            "evidence",
            round(evidence(sc, beat, best[5], fr), 2),
            flush=True,
        )
    fr, cams = bullet_time(sc, info, shots["goal"]["camera"], SHORT)
    shots["bullettime"] = dict(
        frames=fr,
        width=W,
        beat="bullettime",
        keys=[info["scorer"]],
        score=None,
        camera=dict(azimuth=shots["goal"]["camera"]["azimuth"], elevation=25, distance=13.0),
        cams=[dict(fx=fx, R=R.round(6).tolist(), C=C.round(4).tolist()) for fx, R, C in cams],
    )
    print("bullettime", len(fr), "frames, frozen at", info["release"], "and", info["in_net"] + 4, flush=True)
    moving(sc, info, shots, ["goal_slow", "drone", "pov", "puckcam", "goalcam", "celebrate"])
    shots["cuts"] = dict(kind="cuts", parts=cuts(shots, info))
    for name, el, r in (("orbit_low", 15, 10.0), ("orbit_high", 40, 9.0)):
        fr, cams = orbit_360(sc, info, el, r)
        shots[name] = dict(
            frames=fr,
            width=W,
            beat=name,
            keys=[info["scorer"]],
            score=None,
            camera=dict(azimuth=0, elevation=el, distance=r),
            cams=[dict(fx=fx, R=R.round(6).tolist(), C=C.round(4).tolist()) for fx, R, C in cams],
        )
    json.dump(shots | {"_beats": info}, open(f"{B}/shots.json", "w"))
    os.makedirs("check", exist_ok=True)
    cv2.imwrite("check/director.jpg", np.vstack(sheets))


if __name__ == "__main__":
    main()
