"""Each player's stick per frame, from the SAM 3.1 stick outlines (bundle/sticks.json) and both hands: a rigid stick LENGTH long, held
by either hand anywhere along its shaft, pointing anywhere above the ice. Every frame scores each held stick (hand x direction x hand to
heel distance) against what is seen, and the path through the clip with the least cost overall is kept (Viterbi):
- the outline lies along the projected stick, the blade heel on the outline, and the rest of the stick along it (weighted less:
  outlines often miss the upper shaft); distances in m at the hand's depth, so the broadcast's zoom does not matter;
- the holding hand is on the outline; a second hand on the outline in the picture is on the shaft in 3D (two hands fix the depth);
- neither blade nor knob is under the ice; weakly, the blade near the ice and the hand near the knob;
- from frame to frame the stick turns, slides in the hand and changes hands little.
An outline belongs to the player with a wrist nearest it (within NEAR_HAND m). With STICK_EVERY=2 only even frames' outlines are used
(held-out check on the odd ones)."""

import json
import math
import os

import cv2
import numpy as np
import torch
from goal import render as R

LENGTH, DIRS, HOLD = (
    1.58,
    2000,
    (0.5, 0.65, 0.8, 0.95, 1.1, 1.25, 1.4),
)  # blade heel to knob (props.ts); directions; hand to heel (m)
SIG, CLIP, NEAR_HAND, TWO = (
    0.05,
    0.3,
    0.25,
    0.08,
)  # m: outline accuracy, clipped mismatch, outline to its player's wrist, a second hand on it
PTS, SAMPLES = 40, 24  # outline points, shaft samples
W_IN, W_HEEL, W_TWO, W_LOW, W_HOLD = (
    0.3,
    0.5,
    1.0,
    0.5,
    0.3,
)  # weights: stick along the outline, its heel on it, second hand, blade near the ice, hand near the knob
TURN, SLIDE, SWAP, CAP = 0.3, 0.15, 4.0, 10.0  # rad and m a frame at cost 0.5; changing hands; most a step can cost
BAN = 1e4


def sphere(n):
    """n unit directions spread evenly (Fibonacci)."""
    i = torch.arange(n, device=R.DEV, dtype=torch.float32) + 0.5
    z = 1 - 2 * i / n
    r = torch.sqrt(1 - z * z)
    a = math.pi * (1 + 5**0.5) * i
    return torch.stack([r * torch.cos(a), r * torch.sin(a), z], 1)


def perimeter(poly, n=PTS):
    """n points evenly along a closed polygon."""
    p = np.vstack([poly, poly[:1]])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def to_segment(q, a, b):
    """Distance of points q [Q, k] to segments a, b [..., k]: [..., Q]."""
    ab = b - a
    t = (((q - a[..., None, :]) * ab[..., None, :]).sum(-1) / ((ab * ab).sum(-1, keepdim=True) + 1e-9)).clamp(0, 1)
    return (a[..., None, :] + t[..., None] * ab[..., None, :] - q).norm(dim=-1)


def held(W, U, H):
    """Heel and knob [2, D, H, 3] of each held stick: hand h (wrists W [2, 3]), towards the heel U [D, 3], H m from hand to heel."""
    heel = W[:, None, None] + H[None, None, :, None] * U[None, :, None]
    return heel, heel - LENGTH * U[None, :, None]


def prior(W, U, H):
    """Cost [2, D, H] that needs nothing seen: the ice (banned below it), the blade near it, the hand near the knob."""
    heel, knob = held(W, U, H)
    ice = R.BLADE[2]
    c = W_LOW * (heel[..., 2] - ice).clamp(min=0) / 0.6 + W_HOLD * (H - 1.35).abs()[None, None] / 0.5
    return c + BAN * ((heel[..., 2] < ice - 0.03) | (knob[..., 2] < 0)).float()


def seen(cam, W, U, H, polys):
    """Cost [2, D, H] of each held stick against this player's outlines in one frame."""
    heel, knob = held(W, U, H)
    uvw, zw = R.src_project(cam, W)
    ppm = float(cam[0]) / zw  # px per m at each hand
    t = torch.linspace(0, 1, SAMPLES, device=R.DEV)
    uv, z = R.src_project(cam, heel[..., None, :] + t[:, None] * (knob - heel)[..., None, :])  # [2, D, H, S, 2]
    m = ppm[:, None, None, None]
    # outline along the stick
    q = torch.tensor(np.concatenate([perimeter(p) for p in polys]), device=R.DEV, dtype=torch.float32)
    cover = (to_segment(q, uv[..., 0, :], uv[..., -1, :]) / m).clamp(max=CLIP).mean(-1) / SIG
    # stick along the outline: distance to it in a crop around it, in the picture only
    x0, y0 = np.floor(np.concatenate(polys).min(0)).astype(int) - 60
    x1, y1 = np.ceil(np.concatenate(polys).max(0)).astype(int) + 60
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, 1919), min(y1, 1079)
    mask = np.ones((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [np.round(p - [x0, y0]).astype(np.int32)], 0)
    D = torch.tensor(cv2.distanceTransform(mask, cv2.DIST_L2, 5), device=R.DEV)
    px = (uv - torch.tensor([x0, y0], device=R.DEV)).round().long()
    inc = (px[..., 0] >= 0) & (px[..., 0] <= x1 - x0) & (px[..., 1] >= 0) & (px[..., 1] <= y1 - y0) & (z > 0)
    ins = (uv[..., 0] >= 0) & (uv[..., 0] < 1920) & (uv[..., 1] >= 0) & (uv[..., 1] < 1080) & (z > 0)
    d = torch.where(inc, D[px[..., 1].clamp(0, y1 - y0), px[..., 0].clamp(0, x1 - x0)], torch.full_like(z, 1e3))
    d = (d / m).clamp(max=CLIP) * ins
    inside = W_IN * d.sum(-1) / ins.sum(-1).clamp(min=1) / SIG + W_HEEL * d[..., 0] / SIG
    # hands: the holding one on the outline, a second one on it in the picture is on the shaft
    uvw = uvw.cpu().numpy()
    dh = (
        np.array([min(max(-cv2.pointPolygonTest(p, (float(x), float(y)), True), 0) for p in polys) for x, y in uvw])
        / ppm.cpu().numpy()
    )
    c = cover + inside + torch.tensor(np.minimum(dh, CLIP) / SIG, device=R.DEV, dtype=torch.float32)[:, None, None]
    for h in (0, 1):
        if dh[1 - h] < TWO:
            c[h] += W_TWO * (to_segment(W[1 - h][None], heel[h], knob[h])[..., 0] / 0.1).clamp(max=3)
    return c


def viterbi(cost, U, H):
    """Least-cost path through cost [n, 2, D, H] with the turn, slide and change of hand costs between frames: (hand, dir, hold) [n]."""
    turn = (0.5 * (torch.arccos((U @ U.T).clamp(-1, 1)) / TURN) ** 2).clamp(max=CAP)
    slide = (0.5 * ((H[:, None] - H[None]) / SLIDE) ** 2).clamp(max=CAP)
    swap = torch.tensor([[0, SWAP], [SWAP, 0]], device=R.DEV)
    v, back = cost[0], []
    for c in cost[1:]:
        a, ad = (v[:, :, :, None] + slide).min(2)  # [h', u', d]
        b, bu = (a[:, :, None, :] + turn[None, :, :, None]).min(1)  # [h', u, d]
        e, bh = (b[:, None] + swap[:, :, None, None]).min(0)  # [h, u, d]
        v = e + c
        back.append((ad, bu, bh))
    h, u, d = np.unravel_index(int(v.argmin()), v.shape)
    path = [(h, u, d)]
    for ad, bu, bh in reversed(back):
        h = int(bh[h, u, d])
        u = int(bu[h, u, d])
        d = int(ad[h, u, d])
        path.append((h, u, d))
    return np.array(path[::-1])


def fit(cast, keys):
    """{key: [n, 2, 3] holding hand and blade heel} in the cast's frame for the players with these keys."""
    n, every = R.S["n"], int(os.environ.get("STICK_EVERY", "1"))
    U, H = sphere(DIRS), torch.tensor(HOLD, device=R.DEV)
    track = {t["key"]: t for t in cast.tracks}
    outlines = json.load(open(f"{R.B}/sticks.json")) if os.path.exists(f"{R.B}/sticks.json") else [[] for _ in range(n)]
    mine = {k: {} for k in keys}
    for i in range(0, min(n, len(outlines)), every):
        ks = [k for k in keys if track[k]["mask"][i] >= 0]
        if not outlines[i] or not ks:
            continue
        cam = R.Z["cam"][i]
        uv, z = R.src_project(cam, torch.stack([cast.joints[k][i][list(R.WRISTS)] for k in ks]))
        uv, ppm = uv.cpu().numpy(), float(cam[0]) / z.cpu().numpy()
        for o in outlines[i]:
            poly = np.array(o["poly"], np.float32)
            if len(poly) < 3:
                continue
            d = np.array([[max(-cv2.pointPolygonTest(poly, (float(x), float(y)), True), 0) for x, y in w] for w in uv]) / ppm
            if d.min() < NEAR_HAND:
                mine[ks[np.unravel_index(d.argmin(), d.shape)[0]]].setdefault(i, []).append(poly)
    out = {}
    with torch.no_grad():
        for k in keys:
            W = cast.joints[k][:, list(R.WRISTS)].float()
            cost = torch.stack(
                [prior(W[i], U, H) + (seen(R.Z["cam"][i], W[i], U, H, mine[k][i]) if i in mine[k] else 0) for i in range(n)]
            )
            p = viterbi(cost, U, H)
            u = R.time_smooth(U[p[:, 1]], 1.0)
            u = u / u.norm(dim=1, keepdim=True)
            grip = W[torch.arange(n), torch.tensor(p[:, 0], device=R.DEV)]
            out[k] = torch.stack([grip, grip + R.time_smooth(H[p[:, 2]][:, None], 1.0) * u], 1)
    return out
