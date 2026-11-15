"""Render the reconstructed rink scene from virtual cameras (torch, GPU).

Scene: the rink heightfield (ice, boards, stands) textured from a static atlas back-projected from the broadcast frames with people
masked out, the two goals, the puck, and everyone on the ice as a textured 3D body mesh (SAM 3D Body on the stabilised cast, coloured
from the broadcast).

Each shot writes <out>/<shot>/: rgb.mp4 (draft render), depth.mp4 (inverse depth, fixed range per shot), canny.mp4, semantic.mp4,
unknown.mp4 (pixels whose texture was never observed), pose.mp4 (OpenPose-style skeletons), puck.mp4 (where the puck is in front),
glow.mp4 (for a window with a goal: the shot highlight as additive light, black outside the shot: added onto a generated video afterwards, never shown to the
generator), camera.json and preview.jpg (with SAVE_DEPTH set, also depth.npy: metres along the ray, float16, 300 where nothing is hit).
Shot cameras and frames come from director.py (bundle/shots.json).

usage: render.py atlas                build bundle/atlas.npz from the source frames
       render.py shot <name>...       render planned shots, or "check": the source camera (FIRST, COUNT), to compare against the footage"""

import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B = os.environ.get("BUNDLE", "bundle")
BODIES = os.environ.get("BODIES", "bodies")  # per-frame fits (body.py) or bodies_fit (fit.py)
S = json.load(open(f"{B}/scene.json"))
Z = np.load(f"{B}/scene.npz")


def puck_path():
    """The puck's path on the rink per frame, its opacity (puck.py's show: 0 unless it is the goal shot confirmed in the video) and that
    shot (release and in_net frames) if the window has one. Without puck.py: prep's event path, drawn where it exists."""
    if not os.path.exists(f"{B}/puck.json"):
        xy = Z["puck"].astype(np.float64)
        return np.nan_to_num(xy), np.isfinite(xy).all(1).astype(np.float64), None
    P = json.load(open(f"{B}/puck.json"))
    shot = P["shot"]
    return np.array(P["xy"], np.float64), np.array(P["show"], np.float64), shot if shot and shot["confirmed"] else None


PUCK, PUCK_SHOW, SHOT = puck_path()
SHUTTER = 1.0  # source frames the shutter spans per output frame (render_shot sets it)
HX, HY, CR = S["hx"], S["hy"], S["corner"]
A_, B_ = HX - CR, HY - CR  # straight half-lengths of the rounded rectangle
PERIM = 4 * A_ + 4 * B_ + 2 * math.pi * CR
WALL, LEDGE, SLOPE, STAND_SD = 1.07, 1.0, 0.6, 42.0  # boards height, flat ledge behind them, stand rise per metre, stand depth
GOAL_X = HX - 4.0  # goal line
ICE_RES, OUT_RES = 0.025, 0.05  # atlas metres per texel
W, H = map(int, os.environ.get("RES", "1280x768").split("x"))
TEX_HOLDOUT = int(os.environ.get("TEX_HOLDOUT", 0))  # frames around each frame whose own colours its bodies may not use
STICKS = os.environ.get("STICKS") == "1"  # STICKS=1: players carry sticks from sticks.py. Off: on 3 seeds the generators
# placed and kept players worse with them (experiments.md, C2)
FPS = S["fps"]
REF = -1  # team of referees and linesmen


# rink geometry ---------------------------------------------------------------------------------------------------------------------


def sdf(x, y):
    qx, qy = x.abs() - A_, y.abs() - B_
    return torch.hypot(qx.clamp(min=0), qy.clamp(min=0)) + torch.minimum(torch.maximum(qx, qy), torch.zeros_like(qx)) - CR


def surf_z(sd):
    return torch.where(
        sd < 0, torch.zeros_like(sd), torch.where(sd < LEDGE, torch.full_like(sd, WALL), WALL + SLOPE * (sd - LEDGE))
    )


def outline_s(x, y):
    """Arc-length parameter of the nearest outline point (counter-clockwise from the +x end centre)."""
    a, b, c, pi = A_, B_, CR, math.pi
    th = torch.atan2(y - torch.sign(y) * b, x - torch.sign(x) * a)
    th = torch.where(th < 0, th + 2 * pi, th)
    s = torch.where((x > a) & (y.abs() <= b), torch.remainder(y, PERIM), torch.zeros_like(x))
    s = torch.where((x > a) & (y > b), b + c * th, s)
    s = torch.where((x.abs() <= a) & (y > b), b + c * pi / 2 + (a - x), s)
    s = torch.where((x < -a) & (y > b), b + c * pi / 2 + 2 * a + c * (th - pi / 2), s)
    s = torch.where((x < -a) & (y.abs() <= b), b + c * pi + 2 * a + (b - y), s)
    s = torch.where((x < -a) & (y < -b), 3 * b + c * pi + 2 * a + c * (th - pi), s)
    s = torch.where((x.abs() <= a) & (y < -b), 3 * b + 3 * c * pi / 2 + 2 * a + (x + a), s)
    s = torch.where((x > a) & (y < -b), 3 * b + 3 * c * pi / 2 + 4 * a + c * (th - 3 * pi / 2), s)
    return s


def outline_point(s):
    """Outline point and outward normal at arc length s."""
    a, b, c, pi = A_, B_, CR, math.pi
    s = torch.remainder(s, PERIM)
    ends = np.cumsum([b, c * pi / 2, 2 * a, c * pi / 2, 2 * b, c * pi / 2, 2 * a, c * pi / 2, b])
    x, y, nx, ny = [torch.zeros_like(s) for _ in range(4)]

    def put(m, px, py, qx, qy):
        nonlocal x, y, nx, ny
        x, y, nx, ny = torch.where(m, px, x), torch.where(m, py, y), torch.where(m, qx, nx), torch.where(m, qy, ny)

    one, zero = torch.ones_like(s), torch.zeros_like(s)
    seg = torch.bucketize(s, torch.tensor(ends, device=s.device, dtype=s.dtype), right=True)
    start = torch.tensor(np.r_[0, ends], device=s.device, dtype=s.dtype)[seg.clamp(max=8)]
    u = s - start
    for k, (cx_, cy_, a0) in {1: (a, b, 0), 3: (-a, b, pi / 2), 5: (-a, -b, pi), 7: (a, -b, 3 * pi / 2)}.items():
        t = a0 + u / c
        put(seg == k, cx_ + c * torch.cos(t), cy_ + c * torch.sin(t), torch.cos(t), torch.sin(t))
    put(seg == 0, HX * one, u, one, zero)
    put(seg == 2, a - u, HY * one, zero, one)
    put(seg == 4, -HX * one, b - u, -one, zero)
    put(seg == 6, -a + u, -HY * one, zero, -one)
    put(seg >= 8, HX * one, -b + u, one, zero)
    return x, y, nx, ny


def out_world(s, h):
    """Atlas (s, h) -> world point on the boards/ledge/stands; h runs along the profile surface."""
    bx, by, nx, ny = outline_point(s)
    k = math.sqrt(1 + SLOPE**2)
    sd = torch.where(h < WALL, torch.zeros_like(h), torch.where(h < WALL + LEDGE, h - WALL, LEDGE + (h - WALL - LEDGE) / k))
    z = torch.where(h < WALL, h, surf_z(sd + 1e-6))
    return bx + nx * sd, by + ny * sd, z


def out_h(sd, z):
    k = math.sqrt(1 + SLOPE**2)
    return torch.where(sd < 0.02, z, torch.where(sd < LEDGE, WALL + sd, WALL + LEDGE + (sd - LEDGE) * k))


# cameras ---------------------------------------------------------------------------------------------------------------------------


def src_project(cam, P):
    """World points [...,3] -> source pixels [...,2] and camera depth."""
    f, k = float(cam[0]), float(cam[1])
    R = torch.tensor(cam[2:11].reshape(3, 3), device=DEV, dtype=P.dtype)
    C = torch.tensor(cam[11:14], device=DEV, dtype=P.dtype)
    pc = (P - C) @ R.T
    uv = f * pc[..., :2] / pc[..., 2:3]
    uv = uv * (1 + k * (uv**2).sum(-1, keepdim=True) / S["fref"] ** 2)
    return uv + torch.tensor([S["cx"], S["cy"]], device=DEV, dtype=P.dtype), pc[..., 2]


def shot_cameras(name, frames):
    """Camera path of a shot: None for the source camera ("check"), else the per-frame (fx, R, C) that director.py planned."""
    if name == "check":
        return None
    plan = json.load(open(f"{B}/shots.json"))[name]
    return [(c["fx"] * W / plan["width"], np.array(c["R"]), np.array(c["C"])) for c in plan["cams"]]


def rays(fx, R, C, w=W, h=H):
    v, u = torch.meshgrid(
        torch.arange(h, device=DEV, dtype=torch.float32) + 0.5,
        torch.arange(w, device=DEV, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    d = torch.stack([(u - w / 2) / fx, (v - h / 2) / fx, torch.ones_like(u)], -1) @ torch.tensor(
        R, device=DEV, dtype=torch.float32
    )
    return torch.tensor(C, device=DEV, dtype=torch.float32).expand_as(d), d / d.norm(dim=-1, keepdim=True)


def source_rays(cam):
    """Rays of the real broadcast camera (lens distortion included), at the render resolution."""
    v, u = torch.meshgrid(torch.linspace(0.5, 1079.5, H, device=DEV), torch.linspace(0.5, 1919.5, W, device=DEV), indexing="ij")
    d = torch.stack([u - S["cx"], v - S["cy"]], -1)
    q = d.clone()
    for _ in range(8):
        q = d / (1 + float(cam[1]) * (q**2).sum(-1, keepdim=True) / S["fref"] ** 2)
    R = torch.tensor(cam[2:11].reshape(3, 3), device=DEV, dtype=torch.float32)
    r = torch.cat([q / float(cam[0]), torch.ones_like(q[..., :1])], -1) @ R
    return torch.tensor(cam[11:14], device=DEV, dtype=torch.float32).expand_as(r), r / r.norm(dim=-1, keepdim=True)


# atlas -----------------------------------------------------------------------------------------------------------------------------


def load_frame(i, folder="frames"):
    name = f"{i + 1:03d}.jpg" if folder == "frames" else f"{i:03d}.jpg"
    return (
        torch.from_numpy(cv2.cvtColor(cv2.imread(f"{B}/{folder}/{name}"), cv2.COLOR_BGR2RGB)).to(DEV).permute(2, 0, 1).float()
        / 255
    )


def person_mask(outlines, dilate=11):
    m = np.zeros((1080, 1920), np.uint8)
    for p in outlines:
        cv2.fillPoly(m, [np.round(np.array(p)).astype(np.int32)], 1)
    return cv2.dilate(m, np.ones((dilate, dilate), np.uint8))


def sample(img, px):
    """Bilinear sample a [3,h,w] image at pixel coordinates [...,2]."""
    h, w = img.shape[1:]
    g = torch.stack([px[..., 0] / w * 2 - 1, px[..., 1] / h * 2 - 1], -1)
    return F.grid_sample(img[None], g.reshape(1, -1, 1, 2), align_corners=False)[0, :, :, 0].T.reshape(
        *px.shape[:-1], img.shape[0]
    )


def occluded(P, C, steps=64):
    """Ice points hidden from C by the boards and glass or by a goal."""
    hid = torch.zeros(P.shape[:-1], dtype=torch.bool, device=DEV)
    for f in np.geomspace(0.0015, 0.8, steps):
        Q = P + (C - P) * float(f)
        sd = sdf(Q[..., 0], Q[..., 1])
        hid |= (sd > 0) & (Q[..., 2] < surf_z(sd) + 1.85 * (sd < 0.4).float())  # the glass above the boards: frames, reflections
        hid |= (Q[..., 0].abs() > GOAL_X - 0.1) & (Q[..., 0].abs() < GOAL_X + 1.2) & (Q[..., 1].abs() < 1.0) & (Q[..., 2] < 1.3)
    return hid


def build_atlas():
    gx = torch.arange(-HX, HX, ICE_RES, device=DEV)
    gy = torch.arange(-HY, HY, ICE_RES, device=DEV)
    IY, IX = torch.meshgrid(gy, gx, indexing="ij")
    ice = torch.stack([IX, IY, torch.zeros_like(IX)], -1)
    gs = torch.arange(0, PERIM, OUT_RES, device=DEV)
    gh = torch.arange(0, 34, OUT_RES, device=DEV)
    OH, OS = torch.meshgrid(gh, gs, indexing="ij")
    ox, oy, oz = out_world(OS, OH)
    out = torch.stack([ox, oy, oz], -1)
    _, _, nx, ny = outline_point(OS)
    # facing direction of each outside texel: the boards face inward, the stands face up and inward
    k = math.sqrt(1 + SLOPE**2)
    on_wall = OH < WALL
    normal = torch.stack(
        [
            torch.where(on_wall, -nx, -SLOPE * nx / k),
            torch.where(on_wall, -ny, -SLOPE * ny / k),
            torch.where(on_wall, torch.zeros_like(nx), torch.full_like(nx, 1 / k)),
        ],
        -1,
    )
    normal = torch.where(
        ((OH >= WALL) & (OH < WALL + LEDGE))[..., None], torch.tensor([0, 0, 1.0], device=DEV).expand_as(normal), normal
    )
    inside = sdf(IX, IY) < -0.05
    views = [("atlas_frames", j, Z["atlas_cam"][j], S["atlas_outlines"][j]) for j in range(len(Z["atlas_cam"]))]
    views += [("frames", i, Z["cam"][i], S["outlines"][i]) for i in range(0, S["n"], 10)]
    stacks = {"ice": [], "out": []}
    for folder, j, cam, outl in views:
        img = load_frame(j, folder)
        m = torch.from_numpy(person_mask(outl)).to(DEV).float()[None]
        C = torch.tensor(cam[11:14], device=DEV, dtype=torch.float32)
        for key, P in (("ice", ice), ("out", out)):
            px, depth = src_project(cam, P)
            ok = (depth > 1) & (px[..., 0] > 2) & (px[..., 0] < 1918) & (px[..., 1] > 2) & (px[..., 1] < 1078)
            ok &= sample(m, px)[..., 0] < 0.5
            if key == "ice":
                ok &= inside & ~occluded(P, C)
            else:
                ok &= ((C - P) * normal).sum(-1) / (C - P).norm(dim=-1) > 0.3  # reject grazing views: they smear
            col = sample(img, px).half()
            col[~ok] = float("nan")
            stacks[key].append(col.cpu())
    res = {}
    for key, st in stacks.items():
        st = torch.stack(st)
        med = torch.empty(st.shape[1:], device=DEV, dtype=torch.float16)
        for r in range(0, st.shape[1], 32):
            med[r : r + 32] = torch.nanmedian(st[:, r : r + 32].to(DEV).float(), dim=0).values.half()
        res[key] = med
    known_ice, known_out = ~torch.isnan(res["ice"][..., 0]), ~torch.isnan(res["out"][..., 0])
    # unseen stands and boards: the mirror image across the long axis, then across the short axis
    for flip in ((1, -1), (-1, 1), (-1, -1)):
        miss = torch.isnan(res["out"][..., 0])
        if not miss.any():
            break
        mx, my = ox * flip[0], oy * flip[1]
        sd_ = sdf(mx, my).clamp(min=0)
        ms, mh = outline_s(mx, my), out_h(sd_, oz)
        src = res["out"][(mh / OUT_RES).long().clamp(0, len(gh) - 1), (ms / OUT_RES).long().clamp(0, len(gs) - 1)]
        res["out"][miss] = src[miss]
    # unseen ice: the mirror image across the long axis (markings are symmetric), then inpainting
    miss = torch.isnan(res["ice"][..., 0]) & inside
    src = torch.flip(res["ice"], [0])
    res["ice"][miss] = src[miss]
    ice_np = res["ice"].float().cpu().numpy()
    hole = np.isnan(ice_np[..., 0])
    ice8 = (np.nan_to_num(ice_np, nan=0.9) * 255).astype(np.uint8)
    ice8 = cv2.inpaint(ice8, (hole & inside.cpu().numpy()).astype(np.uint8), 7, cv2.INPAINT_TELEA)
    out_np = res["out"].float().cpu().numpy()
    hole = np.isnan(out_np[..., 0])
    with np.errstate(all="ignore"):
        rowmean = np.nanmean(out_np, axis=1, keepdims=True)
    out_np[hole] = np.nan_to_num(rowmean, nan=0.1).repeat(out_np.shape[1], 1)[hole]
    out8 = (np.nan_to_num(out_np, nan=0.1) * 255).astype(np.uint8)
    np.savez_compressed(
        f"{B}/atlas.npz", ice=ice8, out=out8, known_ice=known_ice.cpu().numpy(), known_out=known_out.cpu().numpy()
    )
    cv2.imwrite(f"{B}/atlas_ice.jpg", cv2.cvtColor(ice8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{B}/atlas_out.jpg", cv2.cvtColor(out8[::-1], cv2.COLOR_RGB2BGR))
    print(
        "views %d, ice observed %.1f%%, outside observed %.1f%%"
        % (len(views), 100 * known_ice[inside].float().mean(), 100 * known_out.float().mean())
    )


# people ----------------------------------------------------------------------------------------------------------------------------


def load_cast():
    """Per frame, the stabilised cast from cast.py: one entry per person on the ice with smoothed position, outline and team."""
    C = json.load(open(f"{B}/cast.json"))
    S["people"] = [
        [dict(key=t["key"], id=t["id"], team=t["team"], sxy=t["xy"][i], mask=t["mask"][i]) for t in C["tracks"] if t["seen"][i]]
        for i in range(S["n"])
    ]


OPENPOSE = [0, 69, 6, 8, 41, 5, 7, 62, 10, 12, 14, 9, 11, 13, 2, 1, 4, 3]  # MHR70 joints in OpenPose-18 order
LIMBS = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]
POSE_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
]


def at(x, i):
    """x[i] for a fractional frame time i: linear between the neighbouring frames (slow motion renders real in-between poses)."""
    i0 = min(int(math.floor(i)), len(x) - 1)
    a = i - i0
    return x[i0] if a < 1e-6 or i0 + 1 >= len(x) else x[i0] * (1 - a) + x[i0 + 1] * a


def seen_at(t, i):
    return t["seen"][min(int(round(i)), len(t["seen"]) - 1)]


def puck_at(i, path=PUCK):
    """A per-frame puck quantity (the path, or its opacity) at a fractional frame time."""
    i0 = min(max(int(math.floor(i)), 0), len(path) - 1)
    a = i - i0
    if a < 1e-6 or i0 + 1 >= len(path):
        return path[i0]
    return path[i0] * (1 - a) + path[i0 + 1] * a


def time_smooth(x, sigma, hole=0):
    """Gaussian smoothing along the first (time) axis, edges held; `hole` frames on each side of the centre get no weight."""
    r = int(3 * sigma) + hole
    w = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=DEV, dtype=torch.float32) / sigma) ** 2)
    if hole:
        w[r - hole : r + hole + 1] = 0
    flat = F.pad(x.reshape(x.shape[0], -1).T[:, None], (r, r), mode="replicate")
    return (
        F.conv1d(flat, w[None, None]).squeeze(1).T.reshape(x.shape)
        if hole
        else F.conv1d(flat, (w / w.sum())[None, None]).squeeze(1).T.reshape(x.shape)
    )


def fill_time(frames, x, n):
    """Values at sorted frames -> every frame: linear between neighbours, held at the ends."""
    if len(frames) == 1:
        return x[:1].expand(n, *x.shape[1:]).clone()
    t = torch.arange(n, device=DEV, dtype=torch.float32)
    f = torch.as_tensor(frames, device=DEV, dtype=torch.float32)
    hi = torch.searchsorted(f, t).clamp(1, len(f) - 1)
    lo = hi - 1
    a = ((t - f[lo]) / (f[hi] - f[lo])).clamp(0, 1).view(-1, *(1,) * (x.dim() - 1))
    return x[lo] * (1 - a) + x[hi] * a


BONES = [(5, 6), (5, 7), (6, 8), (9, 11), (10, 12), (11, 13), (12, 14)]  # shoulders, upper arms, thighs, shins (MHR70)


def body_size(j):
    """Summed bone lengths of bodies' joints [..., 70, 3]: a person's size, which their pose does not change."""
    return sum((j[..., a, :] - j[..., b, :]).norm(dim=-1) for a, b in BONES)


# equipment over the bare MHR body, by nearest joint (m), learned from the real outlines by gearfit.py on bodies_fit (source-camera
# silhouette IoU 0.55 bare -> 0.66): helmet, shoulder pads, elbow pads, gloves, pants, knee pads, shin pads, skates
GEAR = [
    ((0, 1, 2, 3, 4), 0.066),
    ((5, 6, 67, 68, 69), 0.045),
    ((7, 8, 63, 64, 65, 66), 0.066),
    (tuple(range(21, 63)), 0.090),
    ((9, 10), 0.028),
    ((11, 12), 0.083),
    ((13, 14), 0.067),
    ((15, 16, 17, 18, 19, 20), 0.028),
]
if os.path.exists(f"{B}/gear.json"):
    with open(f"{B}/gear.json") as f:
        offsets = json.load(f).get(BODIES)
    if offsets is not None:
        if len(offsets) != len(GEAR) or not all(0 <= o <= 0.2 for o in offsets):
            raise ValueError(f"{B}/gear.json: expected {len(GEAR)} offsets in [0, 0.2] for {BODIES}")
        GEAR = [(indices, float(o)) for (indices, _), o in zip(GEAR, offsets)]


def gear_offsets(v, j, faces):
    """Per-vertex outward offset for one body (vertices [V, 3], joints [70, 3]): each vertex takes the gear of its nearest joint,
    smoothed over the mesh so the pads blend into each other."""
    off_j = torch.zeros(j.shape[0], device=DEV)
    for idx, o in GEAR:
        off_j[list(idx)] = o
    off = off_j[torch.cdist(v, j).argmin(1)]
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).long()
    e = torch.cat([e, e.flip(1)])
    deg = torch.zeros_like(off).index_add_(0, e[:, 0], torch.ones(len(e), device=DEV))
    for _ in range(10):
        off = 0.5 * off + 0.5 * torch.zeros_like(off).index_add_(0, e[:, 0], off[e[:, 1]]) / deg
    return off


STICK_LEN, STICK_TOP, STICK_R = 1.5, 0.2, 0.015  # m: gripping hand to blade heel, shaft above that hand, shaft half-width
BLADE = (0.3, 0.07, 0.015)  # blade length, height, thickness (m)
WRISTS = (41, 62)  # right, left (MHR70)
EYES, POV_FORWARD = (1, 2), 0.1  # a POV camera sits this far (m) ahead of the own player's eyes, along its gaze
NUMBERS = os.environ.get("NUMBERS", "1") == "1"  # NUMBERS=0: the backs as sampled from the broadcast (Cast._numbers)
NUMBER_BOX = (-0.40, -0.13, 0.15)  # back number: bottom and top below the shoulder line, half-width (m)
NUMBER_FONT, NUMBER_PX = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 400  # its font and drawing resolution (px per m)
BOX_FACES = [
    (0, 1, 3),
    (0, 3, 2),
    (4, 6, 7),
    (4, 7, 5),
    (0, 4, 5),
    (0, 5, 1),
    (2, 3, 7),
    (2, 7, 6),
    (0, 2, 6),
    (0, 6, 4),
    (1, 5, 7),
    (1, 7, 3),
]


def src_ray(cam, px):
    """Origin and unit direction of the real broadcast camera's ray through one source pixel (lens distortion included)."""
    d = np.asarray(px, np.float64) - [S["cx"], S["cy"]]
    q = d.copy()
    for _ in range(8):
        q = d / (1 + cam[1] * (q**2).sum() / S["fref"] ** 2)
    r = np.r_[q / cam[0], 1.0] @ cam[2:11].reshape(3, 3)
    return cam[11:14], r / np.linalg.norm(r)


def lift_stick(cam, hand, far):
    """Blade heel of a stick held at `hand` (rink frame) whose far end shows at source pixel `far`, on that pixel's ray. One view leaves
    the depth along the ray open: the heel is the point that best keeps the stick STICK_LEN long (0.2 m spread) with the blade near the
    ice (0.5 m spread). An outline ending under 0.9 m from the hand on the ice is a stick partly hidden: it is extended along the ice
    to full length. None if the ray misses the ice."""
    C, d = src_ray(cam, far)
    t_ice = (BLADE[2] - C[2]) / d[2]
    if t_ice <= 0:
        return None
    if np.linalg.norm(C + t_ice * d - hand) <= 0.9:
        u = (C + t_ice * d - hand)[:2]
        u /= np.linalg.norm(u) + 1e-9
        return np.r_[hand[:2] + u * math.sqrt(max(STICK_LEN**2 - (hand[2] - BLADE[2]) ** 2, 0.25)), BLADE[2]]
    p = C + np.linspace(t_ice - 4, t_ice, 400)[:, None] * d
    cost = ((np.linalg.norm(p - hand, axis=1) - STICK_LEN) / 0.2) ** 2 + ((p[:, 2] - BLADE[2]) / 0.5) ** 2
    return p[cost.argmin()]


def boxes(o, ex, ey, ez):
    """Corners [K, 8, 3] of boxes with corner o and edges ex, ey, ez [K, 3] (vertex index 4i + 2j + l for o + i ex + j ey + l ez)."""
    return torch.stack([o + i * ex + j * ey + l * ez for i in (0, 1) for j in (0, 1) for l in (0, 1)], 1)


def stick_mesh(g, h):
    """Shaft (from STICK_TOP above the grip g to the heel h) and blade (from the heel along the shaft's ground direction) of sticks g, h [K, 3]:
    vertices [K, 16, 3]."""
    z = torch.tensor([0.0, 0.0, 1.0], device=DEV).expand_as(g)
    a = F.normalize(h - g, dim=-1)
    u = F.normalize(torch.cross(a, z, dim=-1) + 1e-6, dim=-1)
    v = torch.cross(a, u, dim=-1)
    top = g - a * STICK_TOP
    shaft = boxes(top - STICK_R * (u + v), 2 * STICK_R * u, 2 * STICK_R * v, h - top)
    b = F.normalize(a * torch.tensor([1.0, 1.0, 0.0], device=DEV) + 1e-6, dim=-1)
    side = torch.cross(z, b, dim=-1)
    base = torch.cat([h[:, :2], (h[:, 2:] - BLADE[2]).clamp(min=0)], 1)
    blade = boxes(base - BLADE[2] / 2 * side, BLADE[0] * b, BLADE[2] * side, BLADE[1] * z)
    return torch.cat([shaft, blade], 1)


def heading(j):
    """Facing angle of bodies from their joints (forward is the hip axis turned a quarter left)."""
    d = j[..., 9, :2] - j[..., 10, :2]
    return torch.atan2(-d[..., 0], d[..., 1])


def rotate_z(p, th):
    c, s = torch.cos(th)[..., None], torch.sin(th)[..., None]
    return torch.stack(
        [c[..., 0] * p[..., 0] - s[..., 0] * p[..., 1], s[..., 0] * p[..., 0] + c[..., 0] * p[..., 1], p[..., 2]], -1
    )


def vertex_normals(v, faces):
    a, b, c = v[faces[:, 0].long()], v[faces[:, 1].long()], v[faces[:, 2].long()]
    fn = torch.cross(b - a, c - a, dim=-1)
    n = torch.zeros_like(v)
    for k in range(3):
        n.index_add_(0, faces[:, k].long(), fn)
    return n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)


class Cast:
    """Everyone on the ice as a textured MHR mesh per frame, in the rink frame (SAM 3D Body on the stabilised cast).

    Pose and shape come from the model wherever the person was seen and are interpolated across gaps; each fit is scaled about the
    feet to the person's median size (fits to hidden feet or motion blur come out too small); the position is the smooth cast path
    plus the model's smoothed offset from it. People never seen get a teammate's (or the other goalie's) motion turned to
    their own heading. Equipment (GEAR) is added over the bare body along the vertex normals. Vertex colours are sampled from the
    broadcast where a vertex faced the camera unoccluded inside the person's own outline, blended over nearby frames so the texture is
    stable, with the person's (or their team's) average texture everywhere else. Each player carries a stick from one hand, placed from
    the stick outlines (sticks.py) where seen (_sticks)."""

    def __init__(self):
        b = np.load(f"{B}/{BODIES}.npz")
        tracks = json.load(open(f"{B}/cast.json"))["tracks"]
        n = S["n"]
        self.faces = torch.from_numpy(b["faces"].astype(np.int32)).to(DEV)
        self.ctx = dr.RasterizeCudaContext()
        self.tracks = tracks
        self.verts, self.joints, self.root, self.cols = {}, {}, {}, {}
        self.hide = set()  # people a shot leaves out (render_shot sets it)
        self.own = None  # a POV shot's own player: not drawn, his eyes are the camera
        rel, rj, seen = {}, {}, {}
        for t in tracks:
            k, sel = t["key"], np.flatnonzero(b["key"] == t["key"])
            xy = torch.tensor(t["xy"], device=DEV, dtype=torch.float32)
            if not len(sel):
                continue
            v = torch.from_numpy(b["verts"][sel].astype(np.float32)).to(DEV)
            j = torch.from_numpy(b["joints"][sel].astype(np.float32)).to(DEV)
            fr = b["frame"][sel]
            pel = torch.cat([(j[:, 9, :2] + j[:, 10, :2]) / 2, torch.zeros_like(j[:, 0, :1])], 1)
            off = fill_time(fr, time_smooth(pel[:, :2] - xy[fr], 4.0), n)
            self.root[k] = xy + off
            size = body_size(j)
            g = (size.median() / size)[:, None, None]
            rel[k] = time_smooth(fill_time(fr, (v - pel[:, None]) * g, n), 1.2)
            rj[k] = time_smooth(fill_time(fr, (j - pel[:, None]) * g, n), 1.2)
            seen[k] = fr
        for t in tracks:
            k = t["key"]
            if k in rel:
                continue
            pool = [q for q in tracks if q["key"] in seen and q["goalie"] == t["goalie"]] or [
                q for q in tracks if q["key"] in seen
            ]
            src = max(pool, key=lambda q: len(seen[q["key"]]))["key"]
            xy = torch.tensor(t["xy"], device=DEV, dtype=torch.float32)
            self.root[k] = xy
            if t["goalie"]:
                want = torch.atan2(torch.zeros(n, device=DEV), -torch.sign(xy[:, 0]))
            else:
                vel = time_smooth(torch.diff(xy, dim=0, prepend=xy[:1]), 6.0)
                puck = torch.tensor(PUCK, device=DEV, dtype=torch.float32) - xy
                want = torch.where(
                    vel.norm(dim=1) * FPS > 0.5, torch.atan2(vel[:, 1], vel[:, 0]), torch.atan2(puck[:, 1], puck[:, 0])
                )
            th = want - heading(rj[src])
            rel[k] = rotate_z(rel[src], th[:, None])
            rj[k] = rotate_z(rj[src], th[:, None])
        for t in tracks:
            k = t["key"]
            base = torch.cat([self.root[k], torch.zeros_like(self.root[k][:, :1])], 1)[:, None]
            self.verts[k] = rel[k] + base
            self.joints[k] = rj[k] + base
        src = max(seen, key=lambda k: len(seen[k]))
        mid = n // 2
        self.gear_reference = self.verts[src][mid], self.joints[src][mid]
        self.gear = gear_offsets(*self.gear_reference, self.faces)
        self._texture(seen)
        if NUMBERS:
            self._numbers(self.verts[src][mid], self.joints[src][mid])
        self._sticks()

    def _sticks(self):
        """Grip and blade heel of every player's stick per frame [n, 2, 3] in the rink frame (referees carry none). Each stick outline
        (sticks.py) belongs to the player whose wrist projects nearest it (within 30 px); the far end of its long axis is lifted from that wrist (lift_stick).
        Between sightings the heel's offset from the hand is interpolated (held at the ends); someone never seen with a stick carries it
        on the ice in front of them."""
        self.stick = {}
        n = S["n"]
        obs = {}
        sticks = json.load(open(f"{B}/sticks.json")) if os.path.exists(f"{B}/sticks.json") else [[] for _ in range(n)]
        for i, fr in enumerate(sticks):
            ks = [t["key"] for t in self.tracks if t["mask"][i] >= 0 and t["key"] in self.joints and t["team"] != REF]
            if not fr or not ks:
                continue
            cam = Z["cam"][i].astype(np.float64)
            W3 = torch.stack([self.joints[k][i][list(WRISTS)] for k in ks])
            wpx = src_project(cam, W3)[0].cpu().numpy()
            W3 = W3.cpu().numpy()
            for st in fr:
                poly = np.array(st["poly"], np.float32)
                d = np.array([[max(-cv2.pointPolygonTest(poly, (float(x), float(y)), True), 0) for x, y in w] for w in wpx])
                k, w = np.unravel_index(d.argmin(), d.shape)
                if d[k, w] > 30:
                    continue
                c = poly.mean(0)
                ax = np.linalg.svd(poly - c, full_matrices=False)[2][0]
                pr = (poly - c) @ ax
                ends = (poly[pr.argmin()], poly[pr.argmax()])
                far = max(ends, key=lambda e: np.linalg.norm(e - wpx[k, w]))
                heel = lift_stick(cam, W3[k, w], far)
                if heel is not None and (i not in obs.get(ks[k], {}) or d[k, w] < obs[ks[k]][i][0]):
                    obs.setdefault(ks[k], {})[i] = (d[k, w], w, heel - W3[k, w])
        for t in self.tracks:
            if t["team"] == REF:
                continue
            k = t["key"]
            j = self.joints[k]
            if obs.get(k):
                fr = sorted(obs[k])
                w = int(np.median([obs[k][i][1] for i in fr]))
                off = torch.tensor(np.array([obs[k][i][2] for i in fr]), device=DEV, dtype=torch.float32)
                grip = j[:, WRISTS[w]]
                heel = grip + time_smooth(fill_time(fr, off, n), 2.0)
            else:
                grip = j[:, WRISTS[0]]
                th = heading(j)
                pel = (j[:, 9] + j[:, 10]) / 2
                heel = torch.stack([pel[:, 0] + 0.9 * torch.cos(th), pel[:, 1] + 0.9 * torch.sin(th), torch.zeros_like(th)], 1)
            heel = torch.cat([heel[:, :2], heel[:, 2:].clamp(min=BLADE[2])], 1)
            self.stick[k] = torch.stack([grip, heel], 1)

    def _texture(self, seen):
        n, V = S["n"], self.faces.max().item() + 1
        acc = {k: torch.zeros(n, V, 3, device=DEV) for k in seen}
        wsum = {k: torch.zeros(n, V, device=DEV) for k in seen}
        track = {t["key"]: t for t in self.tracks}
        for i in range(n):
            ks = [t["key"] for t in self.tracks if t["key"] in seen and t["mask"][i] >= 0]
            if not ks:
                continue
            cam = Z["cam"][i]
            C = torch.tensor(cam[11:14], device=DEV, dtype=torch.float32)
            img = load_frame(i)
            hs, ws = 540, 960
            P = torch.cat([self.verts[k][i] + vertex_normals(self.verts[k][i], self.faces) * self.gear[:, None] for k in ks])
            uv, z = src_project(cam, P)
            faces = torch.cat([self.faces + j * V for j in range(len(ks))])
            clip = torch.stack([(uv[:, 0] / 960 - 1) * z, (uv[:, 1] / 540 - 1) * z, ((P - C).norm(dim=1) / 100 - 1) * z, z], 1)
            rast, _ = dr.rasterize(self.ctx, clip[None].contiguous(), faces, (hs, ws))
            zbuf, _ = dr.interpolate((P - C).norm(dim=1, keepdim=True)[None].contiguous(), rast, faces)
            zbuf = torch.where(rast[..., 3:] > 0, zbuf, torch.full_like(zbuf, 1e9))[0, ..., 0]
            px = (uv * torch.tensor([ws / 1920, hs / 1080], device=DEV)).long()
            inside = (px[:, 0] >= 0) & (px[:, 0] < ws) & (px[:, 1] >= 0) & (px[:, 1] < hs)
            pc = px.clamp(min=0)
            pc[:, 0] = pc[:, 0].clamp(max=ws - 1)
            pc[:, 1] = pc[:, 1].clamp(max=hs - 1)
            label = np.zeros((hs, ws), np.uint8)  # each person's own outline, 2 px inside its edge
            for j, k in enumerate(ks):
                cv2.fillPoly(
                    label,
                    [np.round(np.array(S["outlines"][i][track[k]["mask"][i]]) * [ws / 1920, hs / 1080]).astype(np.int32)],
                    j + 1,
                )
            ring = np.ones((5, 5), np.uint8)
            label = torch.from_numpy(np.where(cv2.erode(label, ring) == cv2.dilate(label, ring), label, 0)).to(DEV)
            owner = torch.arange(len(ks), device=DEV).repeat_interleave(V) + 1
            visible = inside & ((P - C).norm(dim=1) <= zbuf[pc[:, 1], pc[:, 0]] + 0.04) & (label[pc[:, 1], pc[:, 0]] == owner)
            col = sample(img, uv)
            for j, k in enumerate(ks):
                sl = slice(j * V, (j + 1) * V)
                nrm = vertex_normals(P[sl], self.faces)
                d = P[sl] - C
                cos = -(nrm * d / d.norm(dim=1, keepdim=True)).sum(1)
                w = ((cos - 0.25) / 0.75).clamp(0, 1) ** 2 * visible[sl].float()
                acc[k][i] = col[sl] * w[:, None]
                wsum[k][i] = w
        edges = torch.cat([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]]).long()
        edges = torch.cat([edges, edges.flip(1)])
        mean = {}
        for k in seen:
            tot = wsum[k].sum(0)
            m = acc[k].sum(0) / tot.clamp(min=1e-6)[:, None]
            known = tot > 1e-3
            for _ in range(200):  # never-seen vertices take their neighbours' colour
                if known.all():
                    break
                s_ = torch.zeros_like(m).index_add_(0, edges[:, 0], m[edges[:, 1]] * known[edges[:, 1], None].float())
                c_ = torch.zeros(V, device=DEV).index_add_(0, edges[:, 0], known[edges[:, 1]].float())
                new = ~known & (c_ > 0)
                m[new] = s_[new] / c_[new, None]
                known = known | new
            mean[k] = m
            a, w = time_smooth(acc[k], 4.0, TEX_HOLDOUT), time_smooth(wsum[k], 4.0, TEX_HOLDOUT)
            blend = (w / 0.15).clamp(max=1)[..., None]
            self.cols[k] = (a / w.clamp(min=1e-6)[..., None]) * blend + m * (1 - blend)
        for t in self.tracks:
            if t["key"] in self.cols:
                continue
            pool = [q for q in self.tracks if q["key"] in mean and q["team"] == t["team"] and q["goalie"] == t["goalie"]] or [
                q for q in self.tracks if q["key"] in mean and q["team"] == t["team"]
            ]
            self.cols[t["key"]] = torch.stack([mean[q["key"]] for q in pool]).mean(0).expand(S["n"], -1, -1)

    def _numbers(self, v, j):
        """The jersey number on every player's back, drawn instead of the one sampled from the broadcast (too small to read, the
        generators misread it): within NUMBER_BOX on the back, the team's jersey colour, with the number from the feed drawn per pixel
        (layers) in the team's number colour, or plain when the number is unknown. The box is placed once on a reference body
        (vertices v [V, 3], joints j [70, 3]); every fit shares its vertex layout. Sets num_uv [V, 3] (the box's grid coordinates,
        inside where both are within ±1, and 1 on the back), glyph {key: [h, w] coverage} and ink {key: (jersey, number colour)}.
        Jersey colour: the median over the team's backs; number colour: the mean of the tenth farthest from it."""
        from PIL import Image, ImageDraw, ImageFont

        sh = (j[5] + j[6]) / 2
        up = torch.tensor([0.0, 0.0, 1.0], device=DEV)
        left = j[5] - j[6]
        left = F.normalize(left - (left @ up) * up, dim=0)
        fwd = torch.linalg.cross(left, up)
        a, h = (v - sh) @ -left, (v - sh) @ up  # across (the player's right positive, as seen from behind) and up
        lo, hi, half = NUMBER_BOX
        back = (vertex_normals(v, self.faces) @ fwd < -0.3) & ((v - sh) @ fwd < 0)
        self.num_uv = torch.stack([a / half, (hi + lo - 2 * h) / (hi - lo), back.float()], 1)
        box = back & (self.num_uv[:, :2].abs() < 1).all(1)
        font = ImageFont.truetype(NUMBER_FONT, int((hi - lo) * NUMBER_PX))
        cw, ch = int(2 * half * NUMBER_PX), int((hi - lo) * NUMBER_PX)
        colours = {}
        for team in (0, 1):
            c = torch.cat(
                [self.cols[t["key"]][:, box].reshape(-1, 3) for t in self.tracks if t["team"] == team and t["key"] in self.cols]
            )
            base = c.median(0).values
            d = (c - base).norm(dim=1)
            colours[team] = (base, c[d >= d.quantile(0.9)].mean(0))
        self.glyph, self.ink = {}, {}
        for t in self.tracks:
            if t["team"] not in colours:
                continue
            k = t["key"]
            self.ink[k] = colours[t["team"]]
            c = self.cols[k].clone()
            c[:, box] = self.ink[k][0]
            self.cols[k] = c
            if not t["jersey"]:
                continue
            l, tp, r, bt = font.getbbox(t["jersey"])
            im = Image.new("L", (r - l, bt - tp))
            ImageDraw.Draw(im).text((-l, -tp), t["jersey"], font=font, fill=255)
            im = im.resize((min(r - l, cw), ch))
            pad = Image.new("L", (cw, ch))
            pad.paste(im, ((cw - im.width) // 2, 0))
            self.glyph[k] = torch.from_numpy(np.asarray(pad, np.float32) / 255).to(DEV)

    def layers(self, i, proj):
        """Everyone at frame i, one (t, rgb, alpha, semantic) layer per team, sticks marked as their own class in the semantic map.
        proj: rink points -> render x, y, camera z, ray distance."""
        out = []
        for team, sid in ((0, SEM["home"]), (1, SEM["away"]), (REF, SEM["other"])):
            ks = [t["key"] for t in self.tracks if t["team"] == team and self.drawn(t, i)]
            if not ks:
                continue
            V = self.verts[ks[0]].shape[1]
            P = torch.cat([(lambda v: v + vertex_normals(v, self.faces) * self.gear[:, None])(at(self.verts[k], i)) for k in ks])
            col = torch.cat([at(self.cols[k], i) for k in ks])
            faces = torch.cat([self.faces + j * V for j, k in enumerate(ks) if k != self.own])
            body_faces = len(faces)
            sk = [k for k in ks if k in self.stick and k != self.own] if STICKS else []
            if sk:
                g = torch.stack([at(self.stick[k], i) for k in sk])
                sv = stick_mesh(g[:, 0], g[:, 1]).reshape(-1, 3)
                bf = torch.tensor(BOX_FACES, device=DEV, dtype=faces.dtype)
                faces = torch.cat([faces] + [bf + len(P) + 8 * q for q in range(2 * len(sk))])
                P = torch.cat([P, sv])
                col = torch.cat([col, torch.full_like(sv, 0.06)])
            num = (
                torch.cat(
                    [
                        self.num_uv.repeat(len(ks), 1),
                        torch.arange(len(ks), device=DEV, dtype=torch.float32).repeat_interleave(V)[:, None],
                    ],
                    1,
                )
                if NUMBERS
                else torch.zeros(0, 4, device=DEV)
            )
            num = torch.cat([num, torch.zeros(len(P) - len(num), 4, device=DEV)])
            x, y, z, t = proj(P)
            clip = torch.stack([(2 * x / W - 1) * z, (2 * y / H - 1) * z, (t / 100 - 1) * z, z], 1)[None].contiguous()
            rast, _ = dr.rasterize(self.ctx, clip, faces, (H, W))
            attr, _ = dr.interpolate(
                torch.cat([col, torch.ones_like(t[:, None]), t[:, None], num], 1)[None].contiguous(), rast, faces
            )
            if NUMBERS:  # the back numbers, per pixel
                gx, gy, back, who = attr[0, ..., 5], attr[0, ..., 6], attr[0, ..., 7], attr[0, ..., 8].round().long()
                inbox = (rast[0, ..., 3] > 0) & (rast[0, ..., 3] <= body_faces) & (back > 0.5) & (gx.abs() < 1) & (gy.abs() < 1)
                rgb = attr[0, ..., :3]
                for q, k in enumerate(ks):
                    m = inbox & (who == q)
                    if k not in self.glyph or not m.any():
                        continue
                    g = F.grid_sample(self.glyph[k][None, None], torch.stack([gx, gy], -1)[None], align_corners=False)[
                        0, 0, ..., None
                    ]
                    base, ink = self.ink[k]
                    rgb = torch.where(m[..., None], base + g * (ink - base), rgb)
                attr = torch.cat([rgb[None], attr[..., 3:]], -1)
            aa = dr.antialias(attr[..., :4].contiguous(), rast, clip, faces)[0]
            cov = rast[0, ..., 3] > 0
            sem = torch.where(rast[0, ..., 3] > body_faces, SEM["stick"], sid)
            out.append(
                (
                    torch.where(cov, attr[0, ..., 4], torch.full_like(attr[0, ..., 4], float("inf"))),
                    aa[..., :3] / aa[..., 3:].clamp(min=1e-3),
                    aa[..., 3].clamp(0, 1),
                    sem,
                )
            )
        return out

    def drawn(self, t, i):
        """Person t is drawn at frame i: in view of the source at some point and not hidden by the shot (a POV shot's own player)."""
        return seen_at(t, i) and t["key"] not in self.hide

    def roots(self, i):
        return [at(self.root[t["key"]], i) for t in self.tracks if self.drawn(t, i)]

    def pose(self, i, proj):
        """OpenPose-style skeletons of everyone at frame i, far to near."""
        img = np.zeros((H, W, 3), np.uint8)
        people = []
        for t in self.tracks:
            if not self.drawn(t, i) or t["key"] == self.own:
                continue
            x, y, z, _ = proj(at(self.joints[t["key"]], i)[OPENPOSE])
            people.append((float(z.mean()), torch.stack([x, y], 1).cpu().numpy(), (z > 0.3).cpu().numpy()))
        for _, p, ok in sorted(people, key=lambda q: -q[0]):
            for c, (a, b_) in enumerate(LIMBS):
                if not (ok[a] and ok[b_]):
                    continue
                m = (p[a] + p[b_]) / 2
                d = p[b_] - p[a]
                L = float(np.hypot(*d))
                if L > W:
                    continue
                poly = cv2.ellipse2Poly(
                    (int(m[0]), int(m[1])), (int(L / 2), 3), int(math.degrees(math.atan2(d[1], d[0]))), 0, 360, 1
                )
                cv2.fillConvexPoly(img, poly, tuple(int(0.6 * v) for v in POSE_COLORS[c]))
            for c in range(18):
                if ok[c]:
                    cv2.circle(img, (int(p[c][0]), int(p[c][1])), 4, POSE_COLORS[c], -1)
        return img


# render ----------------------------------------------------------------------------------------------------------------------------


class Atlas:
    def __init__(self):
        a = np.load(f"{B}/atlas.npz")
        self.ice = torch.from_numpy(a["ice"]).to(DEV).permute(2, 0, 1).float() / 255
        self.out = torch.from_numpy(a["out"]).to(DEV).permute(2, 0, 1).float() / 255
        self.kice = torch.from_numpy(a["known_ice"]).to(DEV).float()[None]
        self.kout = torch.from_numpy(a["known_out"]).to(DEV).float()[None]


SEM = {"roof": 0, "ice": 1, "boards": 2, "stands": 3, "goal": 4, "home": 5, "away": 6, "other": 7, "puck": 8, "stick": 9}
SEM_COLORS = np.array(
    [
        [20, 20, 20],
        [235, 240, 245],
        [40, 60, 160],
        [150, 60, 60],
        [230, 30, 30],
        [60, 110, 255],
        [255, 210, 40],
        [40, 220, 40],
        [0, 0, 0],
        [220, 40, 220],
    ],
    np.uint8,
)


def march(O, D):
    """First hit of the heightfield along each ray: distance t (inf when nothing is hit)."""
    t = torch.full(O.shape[:-1], 0.05, device=DEV)
    hit = torch.zeros_like(t, dtype=torch.bool)
    lo = t.clone()
    for _ in range(520):
        P = O + D * t[..., None]
        sd = sdf(P[..., 0], P[..., 1])
        below = (P[..., 2] < surf_z(sd)) | (sd > STAND_SD)
        new = below & ~hit
        hit |= new
        lo = torch.where(hit, lo, t)
        t = torch.where(hit, t, t + 0.03 + 0.008 * t)
        if hit.all() or (t > 160).all():
            break
    hi = torch.where(hit, t, torch.full_like(t, float("inf")))
    a, b = lo.clone(), hi.clone()
    for _ in range(14):
        mid = torch.where(hit, (a + b) / 2, a)
        P = O + D * mid[..., None]
        sd = sdf(P[..., 0], P[..., 1])
        below = (P[..., 2] < surf_z(sd)) | (sd > STAND_SD)
        b = torch.where(below, mid, b)
        a = torch.where(below, a, mid)
    return torch.where(hit, b, torch.full_like(t, float("inf")))


def net_threads(coord):
    """Average periodic 2 mm cord coverage over a pixel's footprint, including distant subpixel mesh."""
    pitch, duty = 0.04, 0.05
    dy, dx = torch.gradient(coord)
    width = ((dy.abs() + dx.abs()) / pitch).clamp(min=1e-4)
    center = coord / pitch + duty / 2

    def integral(x):
        cell = torch.floor(x)
        return cell * duty + (x - cell).clamp(max=duty)

    return ((integral(center + width / 2) - integral(center - width / 2)) / width).clamp(0, 1)


def goal_layers(O, D):
    """Ray hits on both goals: list of (t, rgb, alpha) layers (posts and crossbar opaque red, the net translucent white mesh)."""
    layers = []
    for sx in (-1, 1):
        g = sx * GOAL_X
        # Net coordinates must lie on the surface: a fixed plane coordinate would turn its whole face into a thread.
        planes = [  # (point, normal, bounds fn, kind)
            (
                (g, 0, 1.22),
                (0, 0, 1),
                lambda P: (sx * (P[..., 0] - g) >= 0) & (sx * (P[..., 0] - g) <= 0.45) & (P[..., 1].abs() <= 0.915),
                "roof",
            ),
            (
                (g + sx * 0.45, 0, 1.22),
                (sx * 1.22, 0, 0.67),
                lambda P: (
                    (sx * (P[..., 0] - g) >= 0.45)
                    & (sx * (P[..., 0] - g) <= 1.12)
                    & (P[..., 1].abs() <= 0.915)
                    & (P[..., 2] >= 0)
                    & (P[..., 2] <= 1.22)
                ),
                "back",
            ),
            (
                (g, 0.915, 0),
                (0, 1, 0),
                lambda P: (
                    (sx * (P[..., 0] - g) >= 0)
                    & (sx * (P[..., 0] - g) <= 1.12 - 0.55 * P[..., 2])
                    & (P[..., 2] >= 0)
                    & (P[..., 2] <= 1.22)
                ),
                "side",
            ),
            (
                (g, -0.915, 0),
                (0, 1, 0),
                lambda P: (
                    (sx * (P[..., 0] - g) >= 0)
                    & (sx * (P[..., 0] - g) <= 1.12 - 0.55 * P[..., 2])
                    & (P[..., 2] >= 0)
                    & (P[..., 2] <= 1.22)
                ),
                "side",
            ),
        ]
        for p0, n, inside, kind in planes:
            n = torch.tensor(n, device=DEV, dtype=torch.float32)
            n = n / n.norm()
            p0 = torch.tensor(p0, device=DEV, dtype=torch.float32)
            den = (D * n).sum(-1)
            t = ((p0 - O) * n).sum(-1) / torch.where(den.abs() < 1e-6, torch.full_like(den, 1e-6), den)
            P = O + D * t[..., None]
            ok = (t > 0) & inside(P)
            if kind == "roof":
                u, v = P[..., 0], P[..., 1]
            elif kind == "side":
                u, v = P[..., 0], P[..., 2]
            else:
                u = P[..., 1]
                v = ((P[..., 0] - g) * 0.67 - sx * P[..., 2] * 1.22) / math.hypot(0.67, 1.22)
            a, b = net_threads(u), net_threads(v)
            alpha = 0.48 * (a + b - a * b)
            layers.append(
                (
                    torch.where(ok, t, torch.full_like(t, float("inf"))),
                    torch.tensor([0.95, 0.95, 0.97], device=DEV).expand(*t.shape, 3),
                    alpha,
                    SEM["goal"],
                )
            )  # translucent: never solid in depth
        # red frame: two posts and the crossbar as cylinders of radius 3 cm
        for a_, b_ in (
            ((g, -0.915, 0), (g, -0.915, 1.22)),
            ((g, 0.915, 0), (g, 0.915, 1.22)),
            ((g, -0.915, 1.22), (g, 0.915, 1.22)),
        ):
            t = ray_cylinder(O, D, torch.tensor(a_, device=DEV), torch.tensor(b_, device=DEV), 0.035)
            layers.append((t, torch.tensor([0.85, 0.08, 0.08], device=DEV).expand(*t.shape, 3), torch.ones_like(t), SEM["goal"]))
    return layers


def ray_cylinder(O, D, a, b, r):
    ax = (b - a) / (b - a).norm()
    L = (b - a).norm()
    oc = O - a
    d_ = D - (D * ax).sum(-1, keepdim=True) * ax
    o_ = oc - (oc * ax).sum(-1, keepdim=True) * ax
    A = (d_ * d_).sum(-1)
    Bq = 2 * (d_ * o_).sum(-1)
    Cq = (o_ * o_).sum(-1) - r * r
    disc = Bq * Bq - 4 * A * Cq
    t = (-Bq - disc.clamp(min=0).sqrt()) / (2 * A + 1e-9)
    s = ((O + D * t[..., None] - a) * ax).sum(-1)
    return torch.where((disc > 0) & (t > 0) & (s >= 0) & (s <= L), t, torch.full_like(t, float("inf")))


def render_frame(i, cam, atlas, cast, proj, O=None, D=None):
    if O is None:
        O, D = rays(*cam)
    t_bg = march(O, D)
    P = O + D * t_bg.clamp(max=1e4)[..., None]
    sd = sdf(P[..., 0], P[..., 1])
    is_ice = (sd < 0) & torch.isfinite(t_bg)
    roof = ~torch.isfinite(t_bg) | (sd > STAND_SD - 0.5)
    # ice texture with soft contact shadows under every person
    iu = (P[..., 0] + HX) / ICE_RES
    iv = (P[..., 1] + HY) / ICE_RES
    col_ice = sample(atlas.ice, torch.stack([iu, iv], -1))
    k_ice = sample(atlas.kice, torch.stack([iu, iv], -1))[..., 0]
    shade = torch.ones_like(t_bg)
    for r in cast.roots(i):
        d2 = (P[..., 0] - r[0]) ** 2 + (P[..., 1] - r[1]) ** 2
        shade = shade * (1 - 0.35 * torch.exp(-d2 / 0.18))
    col_ice = col_ice * shade[..., None]
    s, h = outline_s(P[..., 0], P[..., 1]), out_h(sd.clamp(min=0), P[..., 2])
    st = torch.stack([s / OUT_RES, h / OUT_RES], -1)
    col_out = sample(atlas.out, st)
    k_out = sample(atlas.kout, st)[..., 0]
    rgb = torch.where(is_ice[..., None], col_ice, col_out)
    rgb = torch.where(roof[..., None], torch.tensor([0.08, 0.08, 0.09], device=DEV).expand_as(rgb), rgb)
    known = torch.where(is_ice, k_ice, k_out) * (~roof).float()
    sem = torch.where(is_ice, SEM["ice"], torch.where(h < WALL + 0.02, SEM["boards"], SEM["stands"]))
    sem = torch.where(roof, SEM["roof"], sem)
    depth = torch.where(torch.isfinite(t_bg), t_bg, torch.full_like(t_bg, 300.0))
    layers = goal_layers(O, D)
    # puck: a black disc 7.6 cm across, 2.5 cm high, smeared over the shutter interval (half an output frame either side, so slow
    # motion sharpens it and a frozen moment shows it still) like a broadcast camera
    show = puck_at(i, PUCK_SHOW)
    if show > 0:
        a = torch.tensor(puck_at(max(i - SHUTTER / 2, 0)), device=DEV, dtype=torch.float32)
        b = torch.tensor(puck_at(min(i + SHUTTER / 2, len(PUCK) - 1)), device=DEV, dtype=torch.float32)
        t = (0.0254 - O[..., 2]) / D[..., 2]
        Pp = O + D * t[..., None]
        ab = b - a
        u = (((Pp[..., :2] - a) * ab).sum(-1) / (ab @ ab).clamp(min=1e-9)).clamp(0, 1)
        near = (Pp[..., :2] - (a + u[..., None] * ab)).norm(dim=-1) < 0.038
        t = torch.where((t > 0) & near, t, torch.full_like(t, float("inf")))
        smear = float(min(1.0, 0.076 / (0.076 + ab.norm())))  # exposure spread along the streak
        layers.append(
            (
                t,
                torch.tensor([0.02, 0.02, 0.025], device=DEV).expand(*t.shape, 3),
                torch.full_like(t, max(smear, 0.35) * show),
                SEM["puck"],
            )
        )
    layers += cast.layers(i, proj)
    # composite front-to-back over the background, by per-pixel depth
    ts = torch.stack([l[0] for l in layers])
    order = torch.argsort(ts, dim=0, descending=True)
    cols = torch.stack([l[1] for l in layers])
    alphas = torch.stack([torch.as_tensor(l[2], device=DEV).expand_as(t_bg) for l in layers])
    sems = torch.stack([torch.as_tensor(l[3], device=DEV).expand_as(t_bg) for l in layers])
    for k in range(len(layers)):
        idx = order[k]
        tk = ts.gather(0, idx[None])[0]
        ak = alphas.gather(0, idx[None])[0]
        ck = cols.gather(0, idx[None, ..., None].expand(1, *idx.shape, 3))[0]
        vis = (tk < depth) & torch.isfinite(tk)
        a = (ak * vis.float())[..., None]
        rgb = rgb * (1 - a) + ck * a
        solid = vis & (ak > 0.5)
        sem = torch.where(solid, sems.gather(0, idx[None])[0], sem)
        depth = torch.where(solid, tk, depth)
        known = torch.where(solid, torch.ones_like(known), known)
    return rgb.clamp(0, 1), depth, sem, known


# output ----------------------------------------------------------------------------------------------------------------------------


class Writer:
    def __init__(self, path, fps=FPS):
        self.p = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{W}x{H}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-crf",
                "12",
                "-pix_fmt",
                "yuv420p",
                path,
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, a):
        self.p.stdin.write(np.ascontiguousarray(a).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


GLOW_TRAIL, GLOW_HOLD = (
    6.0,
    12,
)  # source frames of comet tail behind the puck; source frames the highlight stays once it is in the net


def glow(i, proj, depth):
    """The shot highlight at source frame i, as additive light [H, W, 3]: a halo on the puck and a comet tail along its last GLOW_TRAIL
    source frames, from the release until GLOW_HOLD frames after it is in the net (fading out), hidden behind anything in front of it."""
    out = torch.zeros(H, W, 3, device=DEV)
    if not SHOT["release"] <= i <= SHOT["in_net"] + GLOW_HOLD:
        return out
    ts = np.linspace(max(i - GLOW_TRAIL, SHOT["release"]), i, 13)
    P = torch.tensor([[*puck_at(t), 0.0127] for t in ts], device=DEV, dtype=torch.float32)
    x, y, z, dist = proj(P)
    x2, y2, _, _ = proj(P + torch.tensor([0, 0, 0.1], device=DEV))
    if z[-1] <= 0.1:
        return out
    sig = ((x2 - x) ** 2 + (y2 - y) ** 2).sqrt().clamp(min=1.5) * 0.5  # 5 cm in pixels at each point
    yy, xx = torch.meshgrid(
        torch.arange(H, device=DEV, dtype=torch.float32), torch.arange(W, device=DEV, dtype=torch.float32), indexing="ij"
    )
    q = torch.stack([xx, yy], -1)
    pts = torch.stack([x, y], -1)
    tail = torch.zeros(H, W, device=DEV)
    for k in range(len(ts) - 1):
        a, ab = pts[k], pts[k + 1] - pts[k]
        u = (((q - a) * ab).sum(-1) / (ab @ ab).clamp(min=1e-6)).clamp(0, 1)
        d = (q - a - u[..., None] * ab).norm(dim=-1)
        s_ = sig[k] + u * (sig[k + 1] - sig[k])
        tail = torch.maximum(tail, (k + 1) / (len(ts) - 1) * torch.exp(-0.5 * (d / s_) ** 2))
    halo = torch.exp(-0.5 * ((q - pts[-1]).norm(dim=-1) / (3 * sig[-1])) ** 2)
    core = torch.exp(-0.5 * ((q - pts[-1]).norm(dim=-1) / sig[-1]) ** 2)
    shown = (depth > dist[-1] - 0.4).float() * min(1.0, (SHOT["in_net"] + GLOW_HOLD - i) / 6)
    col = lambda r, g, b: torch.tensor([r, g, b], device=DEV)
    return (
        (
            0.8 * tail[..., None] * col(0.25, 0.55, 1.0)
            + 0.6 * halo[..., None] * col(0.3, 0.75, 1.0)
            + core[..., None] * col(0.9, 0.97, 1.0)
        )
        * shown[..., None]
    ).clamp(0, 1)


def projector(i, cam):
    """Rink points -> render-pixel x, y, camera z and ray distance, for a virtual camera or (cam None) the source camera at frame i."""
    if cam is None:
        src = Z["cam"][i]
        C = torch.tensor(src[11:14], device=DEV, dtype=torch.float32)

        def proj(P):
            uv, z = src_project(src, P)
            return uv[:, 0] * W / 1920, uv[:, 1] * H / 1080, z, (P - C).norm(dim=1)

        return proj
    fx = cam[0]
    R = torch.tensor(np.asarray(cam[1]), device=DEV, dtype=torch.float32)
    C = torch.tensor(np.asarray(cam[2]), device=DEV, dtype=torch.float32)

    def proj(P):
        pc = (P - C) @ R.T
        return fx * pc[:, 0] / pc[:, 2] + W / 2, fx * pc[:, 1] / pc[:, 2] + H / 2, pc[:, 2], pc.norm(dim=1)

    return proj


def render_shot(name, frames, out_root, cast):
    global SHUTTER
    SHUTTER = (
        float(min(np.median(np.abs(np.diff(frames))), 1.0)) if len(frames) > 1 else 1.0
    )  # source frames per output frame, at most 1
    atlas = Atlas()
    cams = shot_cameras(name, frames)
    if cast.own is not None:  # a POV: the planned gaze, from the own player's (smoothed) eyes
        eyes = time_smooth(cast.joints[cast.own][:, list(EYES)].mean(1), 2.0).cpu().numpy()
        cams = [
            (fx, R, at(eyes, t) + POV_FORWARD * np.r_[R[2, :2], 0] / max(np.linalg.norm(R[2, :2]), 1e-6))
            for (fx, R, _), t in zip(cams, frames)
        ]
    os.makedirs(f"{out_root}/{name}", exist_ok=True)
    res, poses, glows = [], [], []
    for k, i in enumerate(frames):
        cam = None if cams is None else cams[k]
        proj = projector(i, cam)
        if cam is None:
            O, D = source_rays(Z["cam"][i])
            r = render_frame(i, None, atlas, cast, proj, O, D)
        else:
            r = render_frame(i, cam, atlas, cast, proj)
        res.append([x.cpu() for x in r])
        poses.append(cast.pose(i, proj))
        if SHOT:
            glows.append((glow(i, proj, r[1]) * 255).byte().cpu().numpy())
        if k % 20 == 0:
            print(name, k, flush=True)
    # inverse depth with one fixed range for the whole shot, so the control signal cannot flicker
    inv = [1 / d.clamp(min=0.5) for _, d, _, _ in res]
    samp = torch.stack(inv)[::4].flatten()[::97]
    lo, hi = float(torch.quantile(samp, 0.02)), float(torch.quantile(samp, 0.995))
    ws = {
        k: Writer(f"{out_root}/{name}/{k}.mp4")
        for k in ("rgb", "depth", "canny", "semantic", "unknown", "pose", "puck") + (("glow",) if SHOT else ())
    }
    for k, ((rgb, d, sem, known), pose) in enumerate(zip(res, poses)):
        ws["pose"].write(pose)
        if SHOT:
            ws["glow"].write(glows[k])
        rgb8 = (rgb.numpy() * 255).astype(np.uint8)
        ws["rgb"].write(rgb8)
        dv = ((1 / d.clamp(min=0.5) - lo) / (hi - lo)).clamp(0, 1).numpy()
        ws["depth"].write((np.repeat(dv[..., None], 3, -1) * 255).astype(np.uint8))
        e = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY), (3, 3), 0), 60, 140)
        ws["canny"].write(np.repeat(e[..., None], 3, -1))
        ws["puck"].write(np.repeat(((sem.numpy() == SEM["puck"]) * 255).astype(np.uint8)[..., None], 3, -1))
        ws["semantic"].write(SEM_COLORS[sem.numpy()])
        ws["unknown"].write(np.repeat(((known.numpy() < 0.5) * 255).astype(np.uint8)[..., None], 3, -1))
    for w in ws.values():
        w.close()
    if os.environ.get("SAVE_DEPTH"):
        np.save(f"{out_root}/{name}/depth.npy", torch.stack([d for _, d, _, _ in res]).half().numpy())
    json.dump(
        dict(
            shot=name,
            source_start=S["start"] + frames[0] / FPS,
            fps=FPS,
            frames=len(frames),
            width=W,
            height=H,
            cameras=None
            if cams is None
            else [dict(fx=c[0], R=np.asarray(c[1]).tolist(), C=np.asarray(c[2]).tolist()) for c in cams],
        ),
        open(f"{out_root}/{name}/camera.json", "w"),
    )
    sheet = np.concatenate([(res[j][0].numpy() * 255).astype(np.uint8) for j in (0, len(res) // 2, len(res) - 1)], 0)
    cv2.imwrite(f"{out_root}/{name}/preview.jpg", cv2.cvtColor(cv2.resize(sheet, (W // 2, H * 3 // 2)), cv2.COLOR_RGB2BGR))


def main():
    cmd = sys.argv[1]
    if cmd == "atlas":
        return build_atlas()
    load_cast()
    with torch.no_grad():
        cast = Cast()
    for name in sys.argv[2:]:
        if name == "check":
            frames, cast.hide = (
                list(range(int(os.environ.get("FIRST", 0)), int(os.environ.get("FIRST", 0)) + int(os.environ.get("COUNT", 121)))),
                set(),
            )
        else:
            shot = json.load(open(f"{B}/shots.json"))[name]
            frames, cast.hide, cast.own = shot["frames"], set(shot.get("hide", [])), shot.get("own")
        with torch.no_grad():
            render_shot(name, frames, os.environ.get("OUT", "renders"), cast)


if __name__ == "__main__":
    main()
