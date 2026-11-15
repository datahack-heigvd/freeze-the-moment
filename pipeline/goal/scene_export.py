"""Export a goal's reconstructed scene for the three.js viewer (app/public/replay/): the same content render.py draws, as data.

  <out>/scene.json      rink dimensions, fps, frames exported, people (team, number, goalie, per-frame visibility: seen, and where
                        align.py ran, lined up with their outline: IoU at least MIN_IOU, median over RUN frames), puck path and
                        opacity, and per shot the source frame times and the final per-frame cameras (fx at the shot's width, R, C:
                        rink -> camera rows, OpenCV axes; a POV already sits at its player's eyes), heads (the scorer's, the
                        passer's and the goalie's head cameras at every exported frame), anatomy (the goal taken apart: anatomy),
                        and the broadcast
                        camera's centre (it only turns and zooms: the atlases are its view), and the cut list (director.py cuts:
                        parts, each a shot and its frame range [from, to))
  <out>/faces.bin       uint32 [F, 3] body triangles (MHR)
  <out>/uv.bin          MHR texture layout: float32 [U, 2] texcoords, uint32 [F, 3] texcoord triangles, uint32 [U] vertex of each texcoord
  <out>/body_<key>.gz   the body as render.py poses it (moved along a physical path: physical), with equipment as worn (gear: fitted
                        per goal, and the helmet over the cranium: helmet_gear), the front of the torso softened (soften) and the
                        head steadied (steady_heads: the person's own head, the same in every frame, its face eased in), for
                        every exported frame (scene.json body: first, frames) in BODY_STEP mm units (at most 1 mm off: invisible). One gzip stream, flushed every
                        BODY_CHUNK frames so each chunk decodes as it arrives: per chunk int16 [n, V, 3], its first frame as is,
                        the second as the change from the first, the rest as the change from a constant-velocity guess
                        (x - 2 x_prev + x_prev2), low bytes then high bytes. Half the size of all frames at 1 mm.
  <out>/color_<key>.bin uint8 [V, 3]: render.py's broadcast vertex colours, averaged over time
  <out>/sticks.bin      float32 [people, frames, 2, 3]: holding hand and blade heel (NaN for people without a stick): the rigid stick
                        best matching the outlines and both hands over the clip (stick_pose.py)
  <out>/atlas_ice.jpg, atlas_out.jpg   the static arena atlases (ice; boards and stands by arc length s and profile height h), the
                        unseen parts from the rink turned half a turn (rotated_fill)

Frames exported: the whole goal window the bodies were fitted for (frame0 .. frame0 + frames - 1), so the viewer can show any moment
of it, not only the planned shots.

usage (LTX venv, in a goal dir):  BUNDLE=bundle BODIES=bodies scene_export.py <out> [shot...]   (default: SHOTS, those planned)
                                      BUNDLE=bundle scene_export.py atlas <out>                    the atlases alone"""

import json
import math
import os
import sys
import zlib

import cv2
import numpy as np
import torch

os.environ.setdefault("STICKS", "1")
from goal import render as R
from goal import stick_pose
from kits.gear import CROWN, EARS, covered, head_frame, smoothstep
from paths import MODELS

UV = f"{MODELS}/mhr_uv.npz"
FEATHER = 4  # texels over which a rotated fill fades in from the seen atlas
FACE, HELMET, EDGE = 0.006, 0.028, 0.02  # m: equipment over the face, a helmet's shell and liner over the cranium; its edge
GAZE_DIP, GAZE_BOW, GAZE_SMOOTH = 18.0, 75.0, 2.0  # degrees below the horizon at most (a head camera's 62° tall view
# keeps the horizon 13° under its top edge); degrees down beyond which a head's
# heading means little; frames (a POV's heading)
SWAY, STRIDE, TILT, BUZZ = 1.0, 3.0, 4.0, 0.3  # m: lever from the eyes' sway to the camera's roll and pitch; frames: a stride's
# smoothing; degrees: their soft limit, the skates' vibration RMS (SHAKE scales all)
GEAR = (FACE, 0.015, 0.035, 0.032, 0.019, 0.026, 0.031, 0.0)  # m per render.py GEAR group where the goal has no gearfit.py fit
# (bundle gear_fit.json): the mean of zadina's, musa's and tambellini's
GEAR_SPREAD = 0.01  # m a goal's own fit may differ from GEAR
MIN_IOU, RUN = 0.35, 5  # a body off its outline in the footage counts as unseen (align.py)
VMAX, MARGIN, EDGE, FLIP_WIN = 12.0, 3, 0.3, 5  # m/s no skater exceeds; frames either side of a breach redrawn; m a path
# keeps inside the boards; frames either side a body's facing is held to
CUT, CUT_RUN = 0.5, 12  # an outline under this share of the recent whole ones' height is cut off; frames
GLIDE, TURN_SMOOTH = (0.25, 1.0), 3.0  # m/s over which a lent body turns from the puck to its path; frames
BORROW, BORROW_EDGE = (0.5, 1.5), 6  # m/s over which a held pose takes on a seen skater's; frames to blend in


def smooth(x, sigma=1.5):
    """x [n, ...] smoothed over time (Gaussian, edges held)."""
    r = int(3 * sigma)
    k = np.exp(-(np.arange(-r, r + 1) ** 2) / (2 * sigma**2))
    k /= k.sum()
    p = np.concatenate([x[:1].repeat(r, 0), x, x[-1:].repeat(r, 0)])
    return sum(w * p[i : i + len(x)] for i, w in enumerate(k))


def rotated_fill(img, known, rotate):
    """Texels the broadcast never saw (render.py fills them with mirror images, which reverse every sponsor's lettering) taken from
    the same place on the rink turned half a turn about its centre, where seen: lettering keeps reading the right way. A few texels
    of feathering hide the seam; texels unseen both ways keep render.py's fill."""
    src, ok = rotate(img), rotate(known) & ~known
    w = np.clip(cv2.distanceTransform((~known).astype(np.uint8), cv2.DIST_L2, 3) / FEATHER, 0, 1)[..., None] * ok[..., None]
    return (img * (1 - w) + src * w).round().astype(np.uint8)


def write_atlases(out):
    a = np.load(f"{R.B}/atlas.npz")
    ice = rotated_fill(a["ice"], a["known_ice"], lambda x: x[::-1, ::-1])
    half = a["out"].shape[1] // 2
    side = rotated_fill(a["out"], a["known_out"], lambda x: np.roll(x, half, axis=1))
    cv2.imwrite(f"{out}/atlas_ice.jpg", cv2.cvtColor(ice, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(f"{out}/atlas_out.jpg", cv2.cvtColor(side, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(
        out,
        "atlas unseen",
        round(1 - a["known_out"].mean(), 3),
        "filled from the turned rink",
        round((np.roll(a["known_out"], half, 1) & ~a["known_out"]).mean(), 3),
    )
    return a


def heading(cast, own, cams, frames):
    """A POV's views along where its player's head faces: the fitted head's frame (ears and crown, as viewer/helmet.js), smoothed
    over GAZE_SMOOTH frames, at each shot frame's time."""
    v = cast.verts[own]
    left, right = v[:, EARS[0]], v[:, EARS[1]]
    x = left - right
    x = x / x.norm(dim=1, keepdim=True)
    y = v[:, CROWN] - (left + right) / 2
    y = y - (y * x).sum(1, keepdim=True) * x
    f = R.time_smooth(torch.cross(x, y / y.norm(dim=1, keepdim=True), dim=1), GAZE_SMOOTH).cpu().numpy()
    out = []
    for (fx, _, C), t in zip(cams, frames):
        z = R.at(f, t)
        z = z / np.linalg.norm(z)
        r = np.cross(z, [0, 0, 1.0])
        r /= np.linalg.norm(r)
        out.append((fx, np.stack([r, np.cross(z, r), z]), C))
    return out


def gaze(cams):
    """A POV's views as a skater's gaze: at most GAZE_DIP below the horizon (a head bowed over the puck would film the ice at its
    feet). The heading is the head's own, except where it looks nearly straight down (over GAZE_BOW), where it means little and is
    carried over from the frames on either side."""
    fwd = np.array([Rm[2] for _, Rm, _ in cams])
    level = np.linalg.norm(fwd[:, :2], axis=1)
    yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
    dip = np.arctan2(-fwd[:, 2], level)
    k = np.arange(len(cams))
    ok = dip <= np.radians(GAZE_BOW)
    if ok.any():
        yaw = np.interp(k, k[ok], yaw[ok])
    dip = np.minimum(dip, np.radians(GAZE_DIP))
    out = []
    for (fx, _, C), a, b in zip(cams, yaw, dip):
        f = np.array([np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), -np.sin(b)])
        r = np.cross(f, [0, 0, 1.0])
        r /= np.linalg.norm(r)
        out.append((fx, np.stack([r, np.cross(f, r), f]), C))
    return out


def shake(cams, eyes, frames):
    """A head camera's shake on a POV's views: it rolls and nods with the skater's stride (the eyes' own sideways and vertical motion
    off their STRIDE-smoothed path, over a SWAY lever, softly limited to TILT) and buzzes with the skates (BUZZ, band-limited noise, fixed seed)."""
    k = float(os.environ.get("SHAKE", 1))
    if not k:
        return cams
    off = (eyes - R.time_smooth(eyes, STRIDE)).cpu().numpy()
    noise = np.random.default_rng(0).standard_normal((len(cams), 2))
    n = np.arange(len(cams))
    w = np.exp(-0.5 * (n[:, None] - n[None]) ** 2)
    noise = w @ noise
    noise *= np.radians(BUZZ) / noise.std(0)
    turn = lambda v, a, t: v * np.cos(t) + np.cross(a, v) * np.sin(t) + a * (a @ v) * (1 - np.cos(t))  # v about unit axis a by t
    out = []
    for (fx, Rm, C), t, (yaw, nod) in zip(cams, frames, noise):
        o = R.at(off, t)
        lim = lambda x: np.radians(TILT) * np.tanh(x / np.radians(TILT))
        up = np.array([0, 0, 1.0])
        roll, pitch = k * lim((o @ Rm[0]) / SWAY), k * (lim(o[2] / SWAY) + nod)
        rows = [turn(turn(v, Rm[2], roll), Rm[0], pitch) for v in Rm]
        rows = [turn(v, up, k * yaw) for v in rows]
        out.append((fx, np.stack(rows), C))
    return out


def shot_cameras(cast, name, shift):
    """Per-frame (fx, R, C) of a planned shot at R.W x R.H, a POV moved to its own player's eyes exactly as render_shot does (on
    the player's physical path: shift per person, m per frame), looking where their head faces (heading), held to a skater's gaze
    (gaze) and shaken as a head camera is (shake)."""
    plan = json.load(open(f"{R.B}/shots.json"))[name]
    frames = plan["frames"]
    cams = R.shot_cameras(name, frames)
    own = plan.get("own")
    if own is not None:
        cams = head_cameras(cast, own, shift, frames, cams[0][0])
    return plan, cams


def head_cameras(cast, own, shift, frames, fx):
    """Per-frame (fx, R, C) of player own's head camera at source times `frames`: at their eyes (on their physical path), looking
    where their head faces (heading), held to a skater's gaze (gaze) and shaken (shake)."""
    raw = cast.joints[own][:, list(R.EYES)].mean(1) + shift[own]
    eyes = R.time_smooth(raw, 2.0).cpu().numpy()
    cams = gaze(heading(cast, own, [(fx, None, None)] * len(frames), frames))
    cams = [
        (fx, Rm, R.at(eyes, t) + R.POV_FORWARD * np.r_[Rm[2, :2], 0] / max(np.linalg.norm(Rm[2, :2]), 1e-6))
        for (fx, Rm, _), t in zip(cams, frames)
    ]
    return shake(cams, raw, frames)


def gear():
    """Equipment per GEAR group as fitted to this goal's outlines (gearfit.py), held within GEAR_SPREAD of GEAR (a goal's fit can fail:
    corvi's, IoU 0.62 against 0.76 to 0.80, leaves the gloves at 7 mm); the face's always FACE (the helmet is its own)."""
    path = f"{R.B}/gear_fit.json"
    fit = json.load(open(path)).get(os.environ.get("BODIES", "bodies")) if os.path.exists(path) else None
    return (FACE, *(np.clip(fit, np.subtract(GEAR, GEAR_SPREAD), np.add(GEAR, GEAR_SPREAD)) if fit else GEAR)[1:])


def helmet_gear(cast):
    """The head's equipment as worn: render.py's (fitted to the blurred broadcast outlines) is 6.6 cm all round the head, which makes
    heads a third too wide and swells the face. FACE over the face, HELMET over the cranium (gear.py), blended in over EDGE below
    its rim. Sets the head's GEAR to FACE
    before the cast is built; call it on the cast to add the helmet."""
    rest = np.load(UV)["rest"] / 100
    cast.gear += torch.from_numpy((HELMET - FACE) * covered(rest, rest, EDGE)).to(cast.gear)


def physical(xy):
    """A path xy [n, 2] (m, per frame) with every stretch where it moves faster than VMAX (a tracker jump, smeared by the cast's
    smoothing) redrawn as a cubic from MARGIN frames before it to MARGIN after, keeping the positions and velocities there; and the
    frames of a redrawn stretch that still needs more than VMAX (the tracker moved to someone else). Kept on the ice, EDGE inside
    the boards: the tracker carries people it no longer sees on along their last motion, an official by the boards through them."""
    n = len(xy)
    v = np.gradient(xy, axis=0) * R.FPS
    p = xy.copy()
    bad = np.zeros(n, bool)
    breach = np.flatnonzero(np.linalg.norm(v, axis=1) > VMAX)
    runs = np.split(breach, np.flatnonzero(np.diff(breach) > 2 * MARGIN) + 1) if len(breach) else []
    for r in runs:
        i, j = max(r[0] - MARGIN, 0), min(r[-1] + MARGIN, n - 1)
        h = j - i
        if h < 2:
            continue
        u = np.arange(h + 1)[:, None] / h
        m0, m1 = (xy[i] - xy[max(i - 1, 0)]) * h, (xy[min(j + 1, n - 1)] - xy[j]) * h
        p[i : j + 1] = (
            (2 * u**3 - 3 * u**2 + 1) * xy[i] + (u**3 - 2 * u**2 + u) * m0 + (-2 * u**3 + 3 * u**2) * xy[j] + (u**3 - u**2) * m1
        )
        if (np.linalg.norm(np.diff(p[i : j + 1], axis=0), axis=1) * R.FPS > VMAX).any():
            bad[i : j + 1] = True
    return on_ice(p), bad


def on_ice(xy):
    """Points xy [n, 2] (m) outside the rink shrunk by EDGE moved onto it, to the nearest point."""
    a = np.array([R.HX - R.CR, R.HY - R.CR])
    core = np.clip(xy, -a, a)
    d = xy - core
    r = np.linalg.norm(d, axis=1, keepdims=True)
    return np.where(r > R.CR - EDGE, core + d * (R.CR - EDGE) / np.maximum(r, 1e-9), xy)


def cut_off(t):
    """Frames where track t's outline is cut off (behind a player or the boards): from an outline under CUT of the median height of
    the last CUT_RUN whole ones, through frames without one, to the next whole one. The fit follows the cut outline there, so the
    body squats and spins (asplund's referee: 1.6 m to 0.8 m, turned half round, in 4 frames)."""
    cut, whole, out = False, [], np.zeros(len(t["mask"]), bool)
    for i, m in enumerate(t["mask"]):
        if m >= 0:
            h = np.ptp(np.array(R.S["outlines"][i][m])[:, 1])
            cut = bool(whole) and h < CUT * np.median(whole[-CUT_RUN:])
            if not cut:
                whole.append(h)
        out[i] = cut
    return out


def mend(cast):
    """Bodies fitted wrong for a frame or a few, redrawn between the good frames around them, turned from the one's facing to the
    other's by the shorter way (or carried from the nearest one, moved along the path, at the ends): frames whose hips face more
    than 90° from the mean of FLIP_WIN either side (SAM 3D Body confuses front and back), and frames with a cut-off outline
    (cut_off)."""
    for t in cast.tracks:
        k = t["key"]
        if k not in cast.joints:
            continue
        h = R.heading(cast.joints[k])
        u = np.c_[np.cos(h.cpu().numpy()), np.sin(h.cpu().numpy())]
        bad = cut_off(t)
        for _ in range(2):  # the second pass without the first's flips in the mean
            w = np.where(bad, 0.0, 1.0)[:, None] * u
            c = np.cumsum(np.pad(w, ((FLIP_WIN + 1, FLIP_WIN), (0, 0))), axis=0)
            ref = c[2 * FLIP_WIN + 1 :] - c[: -2 * FLIP_WIN - 1] - w
            bad |= (ref * u).sum(1) < 0
        good = np.flatnonzero(~bad)
        if not bad.any() or not len(good):
            continue
        root = torch.cat([cast.root[k], torch.zeros_like(cast.root[k][:, :1])], 1)[:, None]
        pose = lambda X, i: R.rotate_z(X[i] - root[i], -h[i])
        for f in np.flatnonzero(bad):
            a, b = good[good < f], good[good > f]
            if len(a) and len(b):
                a, b = a[-1], b[0]
                t = (f - a) / (b - a)
                turn = h[a] + t * (torch.remainder(h[b] - h[a] + math.pi, 2 * math.pi) - math.pi)
                for X in (cast.verts[k], cast.joints[k]):
                    X[f] = R.rotate_z(torch.lerp(pose(X, a), pose(X, b), t), turn) + root[f]
            else:
                a = a[-1] if len(a) else b[0]
                for X in (cast.verts[k], cast.joints[k]):
                    X[f] = X[a] - root[a] + root[f]
        print("mend", k, "frames", np.flatnonzero(bad).tolist(), flush=True)


def turn_lent(cast):
    """Skaters without body fits wear another's body (render.py), facing their path, or the puck when slower than 0.5 m/s: at about
    that speed it flips between the two from frame to frame, swinging the body (a musa skater: 0.67 m in a frame). They turn from
    the puck to their path over GLIDE instead, by the shorter way, smoothed over TURN_SMOOTH frames."""
    fitted = set(np.load(f"{R.B}/{R.BODIES}.npz")["key"].tolist())
    for t in cast.tracks:
        k = t["key"]
        if k in fitted or t["goalie"]:
            continue
        xy = cast.root[k]
        puck = torch.tensor(R.PUCK, device=R.DEV, dtype=torch.float32) - xy
        toward = torch.atan2(puck[:, 1], puck[:, 0])
        heading = lambda p: (lambda v: (torch.atan2(v[:, 1], v[:, 0]), v.norm(dim=1) * R.FPS))(
            R.time_smooth(torch.diff(p, dim=0, prepend=p[:1]), 6.0)
        )
        path, speed = heading(xy)
        lent = torch.where(speed > 0.5, path, toward)  # as render.py turned it
        path, speed = heading(
            torch.tensor(on_ice(xy.cpu().numpy().astype(np.float64)), device=R.DEV, dtype=torch.float32)
        )  # as it is drawn
        w = ((speed - GLIDE[0]) / (GLIDE[1] - GLIDE[0])).clamp(0, 1)
        w = w * w * (3 - 2 * w)
        turn = toward + w * torch.remainder(path - toward + math.pi, 2 * math.pi) - w * math.pi
        turn = torch.from_numpy(np.unwrap(turn.cpu().numpy())).to(turn)
        d = (R.time_smooth(turn[:, None], TURN_SMOOTH)[:, 0] - lent)[:, None]
        base = torch.cat([xy, torch.zeros_like(xy[:, :1])], 1)[:, None]
        cast.verts[k] = R.rotate_z(cast.verts[k] - base, d) + base
        cast.joints[k] = R.rotate_z(cast.joints[k] - base, d) + base


def borrow_motion(cast):
    """Between a skater's body fits, render.py holds their pose (interpolated between the fits around), so a skater the broadcast
    lost slides over the ice as a statue. There they take the limbs' motion of the seen skater with the most fits over the same
    frames, turned to their own facing, blended in over BORROW_EDGE frames from each fit and by speed over BORROW (a skater who
    stands keeps the held pose)."""
    b = np.load(f"{R.B}/{R.BODIES}.npz")
    n = R.S["n"]
    fits = {k: np.isin(np.arange(n), b["frame"][b["key"] == k]) for k in set(b["key"].tolist())}
    skaters = [t["key"] for t in cast.tracks if not t["goalie"] and t["key"] in fits]
    base = {k: torch.cat([cast.root[k], torch.zeros_like(cast.root[k][:, :1])], 1)[:, None] for k in skaters}
    for k in skaters:
        held = ~fits[k]
        if not held.any() or held.all():
            continue
        gap = np.full(n, np.inf)  # frames to the nearest fit
        for i in np.flatnonzero(fits[k]):
            gap = np.minimum(gap, np.abs(np.arange(n) - i))
        xy = cast.root[k]
        speed = (R.time_smooth(torch.diff(xy, dim=0, prepend=xy[:1]), 3.0).norm(dim=1) * R.FPS).cpu().numpy()
        edge = np.clip(gap / BORROW_EDGE, 0, 1)
        go = np.clip((speed - BORROW[0]) / (BORROW[1] - BORROW[0]), 0, 1)
        w = torch.tensor(edge * edge * (3 - 2 * edge) * go * go * (3 - 2 * go), device=R.DEV, dtype=torch.float32)
        if not (w > 0).any():
            continue
        donors = [d for d in skaters if d != k]
        if not donors:
            continue
        d = max(donors, key=lambda d: (fits[d] & held).sum())
        turn = (R.heading(cast.joints[k]) - R.heading(cast.joints[d]))[:, None]
        pose = R.rotate_z(cast.verts[d] - base[d], turn)
        joints = R.rotate_z(cast.joints[d] - base[d], turn)
        cast.verts[k] = torch.lerp(cast.verts[k], pose + base[k], w[:, None, None])
        cast.joints[k] = torch.lerp(cast.joints[k], joints + base[k], w[:, None, None])


def aligned(t, align):
    """Per frame, whether track t was seen there and, where align.py scored it, lined up with its outline."""
    seen = np.array(t["seen"], bool)
    a = align.get(str(t["key"]))
    if a is None:
        return seen
    iou = np.array([1.0 if x is None else x[0] for x in a])
    pad = np.pad(iou, RUN // 2, mode="edge")
    return seen & (np.median(np.lib.stride_tricks.sliding_window_view(pad, RUN), axis=1) >= MIN_IOU)


HEAD_FX = 0.5  # head cameras' focal length, of the width (90° across, as the POV shot)
RECEIVE, NET_HALF, NET_TOP, NET_CELL, FLIGHT = (
    1.5,
    0.915,
    1.22,
    0.05,
    1.0,
)  # m from the scorer the pass is received; the net mouth's
# half width and height; the cells it is seen in; m in front of the line a release needs for a view of the mouth
SHOTS = [
    "approach",
    "goal",
    "replay",
    "goal_slow",
    "drone",
    "pov",
    "puckcam",
    "goalcam",
    "celebrate",
    "bullettime",
]  # exported where planned
SOFTEN, SOFTEN_PASSES, EASE = 0.5, 40, 0.015  # the front of the torso smoothed (SOFTEN_PASSES Laplacian passes of SOFTEN), eased
# in by at most EASE (m): within the padding (gear)
HEAD_R, NECK_BLEND = 0.2, (-0.17, -0.09)  # m on the rest mesh's head frame: the head steadied, and blended into the neck
EAR_SPAN, AXIS = 0.25, 0.03  # the viewer's head frame (app/public/replay/helmet.ts): ears' distance in
# its units, the face protection's axis
FLAT, FLAT_SOFT, FACE_PASSES = 0.85, 0.1, 30  # the face held within FLAT of the face protection's band (helmet.ts band()),
# eased in over FLAT_SOFT; its features smoothed by FACE_PASSES Laplacian passes
BODY_STEP, BODY_CHUNK = 2, 25  # mm per stored unit; frames per streamed chunk (1 s)


def torso_front(verts, joints, faces):
    """How much each vertex of a bare body (verts [V, 3], joints [70, 3], MHR70) is the front of its torso, 0 to 1: between the
    hips and the shoulders, inside the shoulders' width, facing forward; eased in from its edges."""
    pel = joints[[9, 10]].mean(0)
    up = joints[69] - pel
    height = np.linalg.norm(up)
    up /= height
    fwd = joints[0] - joints[69]
    fwd -= (fwd @ up) * up
    fwd /= np.linalg.norm(fwd)
    side = np.cross(up, fwd)
    half = np.linalg.norm(joints[5] - joints[6]) / 2
    p = verts - pel
    n = R.vertex_normals(torch.tensor(verts, dtype=torch.float32), torch.tensor(faces)).numpy()
    step = lambda x, a, b: np.clip((x - a) / (b - a), 0, 1) ** 2 * (3 - 2 * np.clip((x - a) / (b - a), 0, 1))
    t, s = p @ up / height, np.abs(p @ side) / half
    return (step(t, 0.05, 0.25) * (1 - step(t, 0.75, 0.92)) * (1 - step(s, 0.7, 0.95)) * step(n @ fwd, 0.2, 0.6)).astype(
        np.float32
    )


def soften(v, faces, w):
    """Bodies v [frames, V, 3] with the front of the torso (w [V], torso_front) moved along its normals onto the body smoothed:
    the hollows under the chest filled, the chest's outline and the belly eased in by at most EASE."""
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).long()
    e0, e1 = e[:, 0], e[:, 1]
    deg = (
        torch.zeros(v.shape[1], device=v.device)
        .index_add_(0, e0, torch.ones(len(e), device=v.device))
        .index_add_(0, e1, torch.ones(len(e), device=v.device))
    )
    on = (w > 0)[None, :, None].float()
    sm = v
    for _ in range(SOFTEN_PASSES):
        mean = (torch.zeros_like(sm).index_add_(1, e0, sm[:, e1]).index_add_(1, e1, sm[:, e0])) / deg[None, :, None]
        sm = sm + SOFTEN * on * (mean - sm)
    n = torch.stack([R.vertex_normals(x, faces) for x in v])
    return v + w[None, :, None] * ((sm - v) * n).sum(-1, keepdim=True).clamp(min=-EASE) * n


def head_weights(faces):
    """The MHR vertices of the head (within HEAD_R of the rest mesh's head frame's origin, above NECK_BLEND's bottom), how much
    each is the steadied head's (0 at the neck's bottom, 1 from NECK_BLEND's top up), and the edges between them (faces [F, 3],
    as indices into them, both ways)."""
    rest = np.load(UV)["rest"].astype(np.float64) / 100
    h = head_frame(rest)(rest)
    idx = np.flatnonzero((np.linalg.norm(h, axis=1) < HEAD_R) & (h[:, 1] > NECK_BLEND[0]))
    at = np.full(len(rest), -1)
    at[idx] = np.arange(len(idx))
    e = at[np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])]
    e = e[(e >= 0).all(1)]
    return idx, smoothstep(NECK_BLEND[0], NECK_BLEND[1], h[idx, 1]), np.concatenate([e, e[:, ::-1]])


def step(e0, e1, x):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def band(y):
    """helmet.ts band(): the face protection's half width and depth at height y (its head frame's units)."""
    t = ((y + 0.03) / -0.09).clamp(0, 1)
    t = t * t * (3 - 2 * t)
    return 0.098 - 0.025 * t, 0.132 - 0.022 * t


def steady_heads(v, idx, w, edges):
    """Bodies v [frames, V, 3] (m) with the head (idx, weights w, edges: head_weights) the person's own, the same in every frame: its shape
    in the viewer's head frame (origin between the ears, x to the left ear, y up through the crown; units of the ears' distance over
    EAR_SPAN) the median over the frames, so it neither bulges nor shrinks against the helmet built from it; the face's features
    smoothed away and the face eased in to within FLAT of the face protection's band (the fitted face holds its volume: the bodies
    were fitted to silhouettes with it on), both faded in over the face's edges; put on each frame's head frame (at the median
    scale) and blended into the fitted neck."""
    L, Rr = v[:, EARS[0]], v[:, EARS[1]]
    o = (L + Rr) / 2
    x = torch.nn.functional.normalize(L - Rr, dim=-1)
    y = v[:, CROWN] - o
    y = torch.nn.functional.normalize(y - x * (y * x).sum(-1, keepdim=True), dim=-1)
    M = torch.stack([x, y, torch.cross(x, y, dim=-1)], -1)
    s = (L - Rr).norm(dim=-1) / EAR_SPAN
    H = (torch.einsum("fvi,fij->fvj", v[:, idx] - o[:, None], M) / s[:, None, None]).median(0).values
    hx, hy, hz = H.unbind(-1)
    on = (step(-0.2, -0.13, hy) * (1 - step(0.04, 0.08, hy)) * step(AXIS - 0.03, AXIS + 0.02, hz))[:, None]
    e0, e1 = edges[:, 0], edges[:, 1]
    deg = torch.zeros(len(idx), device=v.device).index_add_(0, e0, torch.ones(len(e0), device=v.device)).clamp(min=1)[:, None]
    for _ in range(FACE_PASSES):
        H = H + 0.5 * on * (torch.zeros_like(H).index_add_(0, e0, H[e1]) / deg - H)
    hx, hy, hz = H.unbind(-1)
    A, B = band(hy)
    r = torch.hypot(hx / A, (hz - AXIS) / B)
    knee = FLAT - FLAT_SOFT
    t = torch.where(r > knee, (FLAT - FLAT_SOFT * torch.exp((knee - r) / FLAT_SOFT)) / r.clamp(min=1e-6), torch.ones_like(r))
    eased = torch.stack([hx * t, hy, torch.where(hz > AXIS, AXIS + (hz - AXIS) * t, hz)], -1)
    H = H + on * (eased - H)
    head = torch.einsum("vj,fij->fvi", H * s.median(), M) + o[:, None]
    w = w[None, :, None]
    out = v.clone()
    out[:, idx] = w * head + (1 - w) * v[:, idx]
    return out


def pack_body(mm):
    """A body file's bytes from mm [frames, V, 3] (see the module docstring)."""
    Y = np.round(mm / BODY_STEP).astype(np.int32)
    z = zlib.compressobj(6, zlib.DEFLATED, 31)
    out = []
    for c in range(0, len(Y), BODY_CHUNK):
        C = Y[c : c + BODY_CHUNK]
        d = np.concatenate([C[:1], C[1:2] - C[:1], C[2:] - 2 * C[1:-1] + C[:-2]]).astype(np.int16).view(np.uint16)
        out += [
            z.compress((d & 255).astype(np.uint8).tobytes() + (d >> 8).astype(np.uint8).tobytes()),
            z.flush(zlib.Z_SYNC_FLUSH),
        ]
    return b"".join(out + [z.flush()])


def cam_list(cams):
    return [
        dict(fx=round(float(c[0]), 3), R=np.round(np.asarray(c[1]), 6).tolist(), C=np.round(np.asarray(c[2]), 4).tolist())
        for c in cams
    ]


def anatomy(cast, shift, beats):
    """The goal taken apart (director.py's beats): the pass (the frame it left the passer's stick, the first frame after it the puck is
    within RECEIVE of the scorer), the shot's distance and angle off the centre line (from the scorer's blade at the release; the
    flight itself is puck.py's model, not measured, so no speed or crossing point), and the net mouth as the shooter saw it at the release: everyone but him projected
    from the puck onto the goal plane, a grid of NET_CELL cells (rows from the crossbar down, 1 open, 0 covered). A scramble at
    the crease has no view of the mouth (released less than FLIGHT in front of the line): None."""
    rel, nf, pf, sco = beats["release"], beats["in_net"], beats["pass_frame"], beats["scorer"]
    P = np.asarray(R.PUCK, float)
    side = np.sign(P[nf][0])
    gx = side * R.GOAL_X
    src = P[rel]
    at = lambda k, f: (cast.root[k][f] + shift[k][f][:2]).cpu().numpy()
    received = next((f for f in range(pf, rel + 1) if np.linalg.norm(P[f] - at(sco, f)) < RECEIVE), rel)
    front = R.GOAL_X - side * src[0]
    E = np.r_[src, 0.05]
    w, h = round(2 * NET_HALF / NET_CELL), round(NET_TOP / NET_CELL)
    covered = np.zeros((h, w), np.uint8)
    faces = cast.faces.cpu().numpy()
    for t in cast.tracks:
        if t["key"] == sco:
            continue
        v = (cast.verts[t["key"]][rel] + shift[t["key"]][rel]).cpu().numpy()
        dx = v[:, 0] - E[0]
        k = np.where(
            np.abs(dx) > 1e-6, (gx - E[0]) / np.where(np.abs(dx) > 1e-6, dx, 1), -1
        )  # E + k (v - E) is on the goal plane
        q = E + k[:, None] * (v - E)
        px = np.c_[(q[:, 1] + NET_HALF) / NET_CELL, (NET_TOP - q[:, 2]) / NET_CELL] * 16
        tri = np.round(px[faces[(k[faces] > 1).all(1)]]).astype(np.int32)  # wholly between the puck and the goal line
        tri = tri[((tri.max(1) >= 0) & (tri.min(1) < [w * 16, h * 16])).all(1)]  # over the mouth
        for p in tri:
            cv2.fillConvexPoly(covered, p, 1, shift=4)  # one by one: a union, not fillPoly's even-odd
    return dict(
        release=rel,
        in_net=nf,
        pass_frame=pf,
        received=int(received),
        passer=beats["passer"],
        scorer=sco,
        goalie=beats["goalie"],
        distance=round(float(np.hypot(gx - src[0], src[1])), 1),
        angle=round(float(np.degrees(np.arctan2(abs(src[1]), abs(gx - src[0])))), 1),
        **{"from": np.round(src, 2).tolist()},
        open=dict(w=w, h=h, cell=NET_CELL, grid=(1 - covered).tolist(), share=round(float(1 - covered.mean()), 3))
        if front >= FLIGHT
        else None,
    )


def main():
    planned = json.load(open(f"{R.B}/shots.json"))
    out, shots = sys.argv[1], sys.argv[2:] or [s for s in SHOTS if s in planned]
    os.makedirs(out, exist_ok=True)
    R.load_cast()
    R.GEAR[:] = [(idx, g) for (idx, _), g in zip(R.GEAR, gear())]
    with torch.no_grad():
        cast = R.Cast()
        mend(cast)
        turn_lent(cast)
        borrow_motion(cast)
    helmet_gear(cast)
    paths = {t["key"]: physical(cast.root[t["key"]].cpu().numpy().astype(np.float64)) for t in cast.tracks}
    shift = {
        k: torch.tensor(np.pad(p - cast.root[k].cpu().numpy(), ((0, 0), (0, 1))), device=R.DEV, dtype=torch.float32)
        for k, (p, _) in paths.items()
    }
    plans = {s: shot_cameras(cast, s, shift) for s in shots}
    lo, hi = 0, R.S["n"]
    fr = slice(lo, hi)
    root = {}
    faces = cast.faces
    front = torch.tensor(torso_front(*(x.cpu().numpy() for x in cast.gear_reference), faces.cpu().numpy()), device=R.DEV)
    idx, hw, edges = head_weights(faces.cpu().numpy())
    idx, hw, edges = (torch.tensor(x, device=R.DEV) for x in (idx, hw.astype(np.float32), edges))
    with torch.no_grad():
        for t in cast.tracks:
            k = t["key"]
            v = cast.verts[k][fr] + shift[k][fr][:, None]
            v = soften(torch.stack([x + R.vertex_normals(x, faces) * cast.gear[:, None] for x in v]), faces, front)
            v = steady_heads(v, idx, hw, edges)
            mm = v.cpu().numpy() * 1000
            root[k] = np.round(mm[:, :, :2].mean(1)).astype(int).tolist()
            open(f"{out}/body_{k}.gz.tmp", "wb").write(pack_body(mm))
            os.replace(f"{out}/body_{k}.gz.tmp", f"{out}/body_{k}.gz")  # never seen half-written
            (cast.cols[k].mean(0).clamp(0, 1) * 255).round().byte().cpu().numpy().tofile(f"{out}/color_{k}.bin")
    faces.cpu().numpy().astype(np.uint32).tofile(f"{out}/faces.bin")
    uv = np.load(UV)
    tc, tf, vf = uv["texcoords"], uv["texcoord_faces"].astype(np.int64), uv["faces"].astype(np.int64)
    vert_of = np.zeros(len(tc), np.int64)
    vert_of[tf.ravel()] = vf.ravel()
    with open(f"{out}/uv.bin", "wb") as f:
        tc.astype(np.float32).tofile(f)
        tf.astype(np.uint32).tofile(f)
        vert_of.astype(np.uint32).tofile(f)
    st = np.full((len(cast.tracks), hi - lo, 2, 3), np.nan, np.float32)
    sticks = stick_pose.fit(cast, [t["key"] for t in cast.tracks if t["key"] in cast.stick])
    for j, t in enumerate(cast.tracks):
        if t["key"] in sticks:
            st[j] = (sticks[t["key"]] + shift[t["key"]][:, None])[fr].cpu().numpy()
    st.tofile(f"{out}/sticks.bin")
    a = write_atlases(out)
    S = R.S
    align = json.load(open(f"{R.B}/align.json")) if os.path.exists(f"{R.B}/align.json") else {}
    scene = dict(
        fps=R.FPS,
        frame0=lo,
        frames=hi - lo,
        body=dict(first=0, frames=hi - lo, step=BODY_STEP, chunk=BODY_CHUNK),
        vertices=int(cast.verts[cast.tracks[0]["key"]].shape[1]),
        faces=int(len(faces)),
        width=R.W,
        height=R.H,
        rink=dict(
            hx=R.HX,
            hy=R.HY,
            corner=R.CR,
            wall=R.WALL,
            ledge=R.LEDGE,
            slope=R.SLOPE,
            stand_sd=R.STAND_SD,
            goal_x=R.GOAL_X,
            ice_res=R.ICE_RES,
            out_res=R.OUT_RES,
            perim=R.PERIM,
            ice_size=list(a["ice"].shape[:2]),
            out_size=list(a["out"].shape[:2]),
        ),
        people=[
            dict(
                key=t["key"],
                team=t["team"],
                jersey=t.get("jersey"),
                name=t.get("name"),
                goalie=bool(t.get("goalie")),
                seen=[bool(x) for x in (aligned(t, align) & ~paths[t["key"]][1])[lo:hi]],
                stick=t["key"] in cast.stick,
                root=root[t["key"]],
            )
            for t in cast.tracks
        ],
        puck=dict(xy=np.asarray(R.PUCK)[lo:hi].tolist(), show=np.asarray(R.PUCK_SHOW)[lo:hi].tolist(), shot=R.SHOT),
        shots={
            s: dict(frames=p["frames"], own=p.get("own"), hide=p.get("hide", []), cams=cam_list(cams))
            for s, (p, cams) in plans.items()
        },
        heads=[
            dict(key=k, role=role, cams=cam_list(head_cameras(cast, k, shift, list(range(lo, hi)), HEAD_FX * R.W)))
            for role in ("scorer", "passer", "goalie")
            if (k := planned["_beats"].get(role)) is not None and k in cast.joints
        ],
        anatomy=anatomy(cast, shift, planned["_beats"]),
        cuts=[c for c in planned.get("cuts", {}).get("parts", []) if c["shot"] in plans],
        source=dict(start=S["start"], camera=np.median(R.Z["cam"][:, 11:14], 0).tolist()),
    )
    json.dump(scene, open(f"{out}/scene.json", "w"))
    print(out, "frames", lo, hi, "people", len(cast.tracks), "shots", list(plans))


if __name__ == "__main__":
    if sys.argv[1] == "atlas":
        write_atlases(sys.argv[2])
    else:
        main()
