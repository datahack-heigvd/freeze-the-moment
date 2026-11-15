"""The arena atlas at high resolution, shared by every goal: fused from every calibrated broadcast view of the match.

The broadcast camera only turns and zooms about one fixed centre, so each atlas texel's colour is what that camera saw along the ray
to it, in whichever views it was in frame. Views of the 2025 goals (each goal's atlas frames and its own window, people masked out)
are fused per texel in two passes:
  1. a reference: a weighted mean with every view counted about equally (resolution capped at one source pixel per texel), which
     averages occluders and noise away but is soft;
  2. the atlas: views weighted by their resolution there cubed (source pixels per texel, from zoom, distance and obliquity) and by
     motion-blur sharpness, so the most zoomed-in sharp views decide; a view counts only where its low-pass agrees with the reference
     (people the outlines missed, sticks, the puck drop out without losing the sharp edges of letters).
Before both, every view's calibration is corrected against a first reference (register: a small rotation and zoom per view), twice,
and in both every view is warped locally onto the reference (local_flow: block matching), so views a few pixels apart do not fuse into
doubled edges and steps. Where the chosen view group changes, the exposure and colour still step: the fused atlas is integrated
again in the gradient domain with the reference's gradients across those boundaries (seamless).
Texels no view saw are filled from the arena's symmetry (rotated half a turn about the centre, so the boards' sponsor text stays
readable), and what is still empty by copying observed texture at the same height (square patches), never smeared rows. Every filled texel is marked in the masks.

Grids: ice 1 cm per texel; the boards band (profile height h < BAND_TOP: boards, ledge, first rows) 1 cm, folded into STRIPS strips
stacked vertically (each texture side stays <= 8192); the stands (all h) 2.5 cm.

  <out>/ice.jpg, band.jpg, stands.jpg   the atlases (sRGB)
  <out>/ice_fill.png, band_fill.png, stands_fill.png   255 where a texel was filled rather than seen
  <out>/atlas.json      grid layout (viewer/atlas.js)

usage (LTX venv):  arena_atlas.py build <out>          (BUNDLE: any goal bundle, for the rink geometry; LIMIT=n: n views only, a quick look;
                                                           REG_ITERS=0: no view registration)
                       arena_atlas.py link <out> <scenes>  write <scenes>/<goal>/atlas.json pointing at <out> (as ../<out name>)
                       arena_atlas.py repair <out>         refill the stands seen only side-on (desmear), the filled ice from the
                                                           half turn (refill_ice), the band's fill, and level the band (repair_band)"""

import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from goal import render as R

DEV = R.DEV
from paths import GOALS

ICE_RES, BAND_RES, STAND_RES = 0.01, 0.01, 0.025
BAND_TOP, STAND_TOP, STRIPS, PAD = 2.4, 34.0, 3, 16  # m, m, strips of the band, padding texels around each strip
WINDOW_STEP, POWER, LP_SIGMA, AGREE = (
    2,
    3.0,
    4.0,
    0.07,
)  # every n-th window frame; resolution weight exponent; low-pass (texels); colour tolerance
GROUP_SIGMA = 15.0  # texels: the group choice follows its smoothed score, so it switches only
# where coverage changes, not texel by texel (that would patch together views cm apart)
ATLAS_WEIGHT = 0.001  # atlas frames (single-frame calibrations, seconds apart) in the sharp pass:
# they fill what the goal windows (smoothed calibrations) do not show
NCC_MIN = 0.6  # views correlating less with the reference are left out
MIN_COS = dict(band=0.3, stands=0.45)  # boards and stands seen more obliquely count as unseen: they smear (the
# stands on the camera's side and its corners are only ever seen so)
BOARDS_TOP, LEVEL_TOP, LEVEL_BLOCK, LEVEL_SPAN, LEVEL_SIGMA = 1.15, 2.3, 0.25, 0.2, 1.0
# m: the board profile matched along the band, the height the levelling
# fades out by, block width, largest shift, smoothing along s
FLOW = os.environ.get("FLOW", "1") == "1"  # local alignment of every view to the reference (local_flow) in passes 2 and 3
FLOW_BLOCK, FLOW_SEARCH, FLOW_MIN_NCC, FLOW_THIN = 64, 12, 0.5, 3  # px, px, block correlation kept, every n-th texel drawn
REG_GRIDS = os.environ.get("REG_GRIDS", "ice,band").split(",")  # surfaces views are registered on: where the geometry is right
REG_ITERS, REG_POINTS, REG_STEPS, REG_SIGMAS, REG_HP = (
    int(os.environ.get("REG_ITERS", 2)),
    200_000,
    30,
    (3.0, 1.0),
    8.0,
)  # view registration: rounds, texels, steps
# per blur level (px), detail high-pass (px)
SEAMLESS = os.environ.get("SEAMLESS", "1") == "1"  # gradient-domain removal of the steps where the chosen view group changes
GROUP_SEAM, SEAM_PULL, SEAM_ITERS = 2, 1e-4, 600  # texels either side taken from the reference, pull to its levels, CG steps
SEAM_SIGMA, SEAM_SPREAD = 0.08, 0.6  # m: ice fill shading, seam sampling and its spread
SMEAR_SPAN, SMEAR_MIN, SMEAR_OFF_AXIS = 1.0, 0.4, 25.0  # smear test window (m), direction consistency, degrees off the grid axes
ROOF = (0.08, 0.08, 0.09)  # above the highest stand row the camera ever saw: dark structure, not crowd


def views():
    """Every calibrated view: (image path, camera [14], person outlines) over the goals' atlas frames and windows."""
    out = []
    for b in sorted(glob.glob(f"{GOALS}/*/bundle")):
        S = json.load(open(f"{b}/scene.json"))
        Z = np.load(f"{b}/scene.npz")
        out += [
            (f"{b}/atlas_frames/{j:03d}.jpg", Z["atlas_cam"][j], S["atlas_outlines"][j], ("a", b, j))
            for j in range(len(Z["atlas_cam"]))
        ]
        out += [
            (f"{b}/frames/{i + 1:03d}.jpg", Z["cam"][i], S["outlines"][i], ("w", b, i)) for i in range(0, S["n"], WINDOW_STEP)
        ]
    return out


def ray_dir(c, px):
    """Unit ray direction of source pixel px [2] for camera c (lens distortion included)."""
    d = np.asarray(px, np.float64) - [R.S["cx"], R.S["cy"]]
    q = d.copy()
    for _ in range(8):
        q = d / (1 + c[1] * (q**2).sum() / R.S["fref"] ** 2)
    r = np.r_[q / c[0], 1.0] @ c[2:11].reshape(3, 3)
    return r / np.linalg.norm(r)


def motion_blur(vs):
    """Per window view, how fast it pans: the image motion of its centre per frame against its neighbours (px). Pans blur the frame
    and the smoothed calibration lags behind them, so these views are trusted less."""
    blur = np.zeros(len(vs))
    key = {v[3]: n for n, v in enumerate(vs)}
    for n, (_, c, _, (kind, b, j)) in enumerate(vs):
        if kind == "a":
            continue  # atlas frames are seconds apart: no neighbour to measure against
        step = WINDOW_STEP
        m = []
        for nb in (key.get((kind, b, j - step)), key.get((kind, b, j + step))):
            if nb is None:
                continue
            c2 = vs[nb][1]
            P = c[11:14] + 50 * ray_dir(c, (R.S["cx"], R.S["cy"]))
            uv, _ = R.src_project(c2.astype(np.float64), torch.tensor(P, device=DEV, dtype=torch.float64)[None])
            m.append(float(np.abs(uv[0].cpu().numpy() - [R.S["cx"], R.S["cy"]]).max()) / step)
        blur[n] = min(m) if m else 0.0
    return blur


class Grid:
    """Texels of one atlas: world points of the texels a view can ever see from the fixed camera centre, their facing (cos) and
    distance to it, and the fusion accumulators."""

    def __init__(self, name, shape, P, normal, ok, res, C0, min_cos):
        self.name, self.shape, self.res = name, shape, res
        self.idx = torch.nonzero(ok.flatten()).squeeze(1)
        self.P = P.reshape(-1, 3)[self.idx].contiguous()
        d = C0 - self.P
        dist = d.norm(dim=1)
        cos = (d * normal.reshape(-1, 3)[self.idx]).sum(1) / dist
        keep = cos > min_cos
        self.idx, self.P, self.k = (
            self.idx[keep],
            self.P[keep].contiguous(),
            (res * cos[keep] / dist[keep]).contiguous(),
        )  # q = fx * k
        N = len(self.idx)
        self.S1, self.W1 = torch.zeros(N, 3, device=DEV), torch.zeros(N, device=DEV)
        self.S2, self.W2 = torch.zeros(N, 3, device=DEV), torch.zeros(N, device=DEV)
        self.score = None
        self.group = None  # per texel and view group: summed score (pass 2); the chosen group
        print(name, shape, "visible texels", N, flush=True)

    def full(self, x, fill=0.0):
        out = torch.full((self.shape[0] * self.shape[1], *x.shape[1:]), fill, device=DEV, dtype=x.dtype)
        out[self.idx] = x
        return out.reshape(*self.shape, *x.shape[1:])


def gauss_blur(x, m, sigma):
    """Normalised Gaussian blur of a [h, w, c] field x with weights m [h, w] (texels where m is 0 do not contribute)."""
    r = int(3 * sigma)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=DEV, dtype=torch.float32) / sigma) ** 2)
    k /= k.sum()
    a = torch.cat([x * m[..., None], m[..., None]], -1).permute(2, 0, 1)[:, None]
    a = F.conv2d(F.pad(a, (r, r, 0, 0)), k.view(1, 1, 1, -1))
    a = F.conv2d(F.pad(a, (0, 0, r, r)), k.view(1, 1, -1, 1))
    a = a[:, 0].permute(1, 2, 0)
    return a[..., :-1] / a[..., -1:].clamp(min=1e-6), a[..., -1]


def build_grids():
    HX, HY = R.HX, R.HY
    gx = -HX + torch.arange(int(2 * HX / ICE_RES) + 1, device=DEV) * ICE_RES
    gy = -HY + torch.arange(int(2 * HY / ICE_RES) + 1, device=DEV) * ICE_RES
    IY, IX = torch.meshgrid(gy, gx, indexing="ij")
    P = torch.stack([IX, IY, torch.zeros_like(IX)], -1)
    inside = R.sdf(IX, IY) < -0.03
    occ = torch.zeros_like(inside)
    for r0 in range(0, P.shape[0], 256):
        occ[r0 : r0 + 256] = R.occluded(P[r0 : r0 + 256], C0)
    grids = [
        Grid("ice", tuple(IX.shape), P, torch.tensor([0, 0, 1.0], device=DEV).expand_as(P), inside & ~occ, ICE_RES, C0, 0.12)
    ]
    del P, IX, IY, occ
    for name, res, top in (("band", BAND_RES, BAND_TOP), ("stands", STAND_RES, STAND_TOP)):
        gs = torch.arange(int(math.ceil(R.PERIM / res)), device=DEV) * res
        gh = torch.arange(int(round(top / res)), device=DEV) * res
        OH, OS = torch.meshgrid(gh, gs, indexing="ij")
        ox, oy, oz = R.out_world(OS, OH)
        P = torch.stack([ox, oy, oz], -1)
        _, _, nx, ny = R.outline_point(OS)
        k = math.sqrt(1 + R.SLOPE**2)
        wall = OH < R.WALL
        n = torch.stack(
            [
                torch.where(wall, -nx, -R.SLOPE * nx / k),
                torch.where(wall, -ny, -R.SLOPE * ny / k),
                torch.where(wall, torch.zeros_like(nx), torch.full_like(nx, 1 / k)),
            ],
            -1,
        )
        n = torch.where(
            ((OH >= R.WALL) & (OH < R.WALL + R.LEDGE))[..., None], torch.tensor([0, 0, 1.0], device=DEV).expand_as(n), n
        )
        grids.append(Grid(name, tuple(OS.shape), P, n, torch.ones_like(OS, dtype=torch.bool), res, C0, MIN_COS[name]))
    return grids


def load(v):
    """A view's image (pinned, RGB uint8) and how clear of people each pixel is (0 on a person's dilated outline, 1 well away)."""
    path, cam, outl, _ = v
    img = cv2.imread(path)
    if img is None:
        return None
    free = 1 - np.clip(2 * cv2.GaussianBlur(R.person_mask(outl, 15).astype(np.float32), (0, 0), 6), 0, 1)
    return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), free


def splat(g, view, cam, blur, pas, ref=None, rejected=None, kind="w", grp=0, flow=None):
    """Add one view to grid g. Pass 1: the reference. Passes 2 and 3 (ref: the reference's low-pass and detail energy per texel):
    each view scores every texel it sees (resolution, pan speed, agreement with the reference, clear of frame edges and people);
    pass 2 sums the scores per view group (a goal's window, or all atlas frames), pass 3 averages the views of the group chosen for
    the texel, with the view's colours brought to the reference's exposure. A view whose low-pass does not correlate with the reference's (a wrong calibration, a cut to another camera)
    is left out and its correlation added to `rejected`."""
    (img, soft), free_img = view
    q = float(cam[0]) * g.k
    px, depth = R.src_project(cam.astype(np.float64), g.P)
    border = torch.minimum(torch.minimum(px[:, 0], 1919 - px[:, 0]), torch.minimum(px[:, 1], 1079 - px[:, 1]))
    sel = torch.nonzero((depth > 1) & (border > 2)).squeeze(1)
    if not len(sel):
        return
    px, q, border = px[sel], q[sel], border[sel]
    if flow is not None:
        px = px + R.sample(flow, px)
    free = R.sample(free_img, px)[..., 0]
    keep = free > 0.05
    sel, px, q, border, free = sel[keep], px[keep], q[keep], border[keep], free[keep]
    if len(sel) < 100:
        return
    col = torch.where((q < 0.6)[:, None], R.sample(soft, px), R.sample(img, px))
    w = (border / 30).clamp(max=1) * free / (1 + blur**2)
    if pas == 1:
        w = w * q.clamp(max=1)
        g.S1.index_add_(0, sel, col * w[:, None])
        g.W1.index_add_(0, sel, w)
        return
    ref_lp = ref
    flat = g.idx[sel]
    rr, cc = flat // g.shape[1], flat % g.shape[1]
    r0, r1, c0, c1 = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
    crop = torch.zeros(r1 - r0, c1 - c0, 3, device=DEV)
    m = torch.zeros(r1 - r0, c1 - c0, device=DEV)
    crop[rr - r0, cc - c0] = col
    m[rr - r0, cc - c0] = 1
    lp, _ = gauss_blur(crop, m, LP_SIGMA)
    a, b = lp[rr - r0, cc - c0], ref_lp[rr, cc]
    gain = (b.median(0).values / a.median(0).values.clamp(min=0.02)).clamp(0.7, 1.4)
    a = a * gain
    if b.std() > 0.03:
        ncc = float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-6))
        if ncc < NCC_MIN:
            rejected.append(ncc)
            return
    d = (a - b).norm(dim=1)
    w = w * q.clamp(max=4) ** POWER * torch.exp(-0.5 * (d / AGREE) ** 2) * (ATLAS_WEIGHT if kind == "a" else 1.0)
    if pas == 2:
        g.score[:, grp].index_add_(0, sel, w)
        return
    w = w * (g.group[sel] == grp)  # only the group chosen for the texel: views of one goal window
    # agree with each other, across windows they are cm apart
    g.S2.index_add_(0, sel, col * gain * w[:, None])
    g.W2.index_add_(0, sel, w)


def block_shifts(cam, gray, free, grids, sharp):
    """A view's misalignment against the reference per block: the reference's detail (sharp: per grid, high-passed per texel) is drawn
    into the view's image at half resolution, both are matched in blocks of FLOW_BLOCK px over shifts up to FLOW_SEARCH px (correlation
    of the high-passed images, soft-argmax for sub-pixel shifts). Returns the shift per block [2, hb, wb] (full-resolution px: where in
    the view the reference really is) and which blocks are confident [hb, wb]."""
    h2, w2 = 540, 960
    acc = torch.zeros(h2 * w2, device=DEV)
    cnt = torch.zeros(h2 * w2, device=DEV)
    for g in grids:
        P, val = g.P[::FLOW_THIN], sharp[g.name][::FLOW_THIN]
        px, depth = R.src_project(cam.astype(np.float64), P)
        q = (px / 2).round().long()
        ok = (depth > 1) & (q[:, 0] >= 0) & (q[:, 0] < w2) & (q[:, 1] >= 0) & (q[:, 1] < h2)
        flat = q[ok, 1] * w2 + q[ok, 0]
        acc.index_add_(0, flat, val[ok])
        cnt.index_add_(0, flat, torch.ones_like(val[ok]))
    have = (cnt > 0).float().view(1, 1, h2, w2)
    ref = (acc / cnt.clamp(min=1)).view(1, 1, h2, w2)
    img = F.avg_pool2d(gray[None, None], 2)
    fr = F.avg_pool2d(free[None], 2)
    blur = lambda x, m: gauss_blur(x[0].permute(1, 2, 0), m[0, 0], 4.0)[0].permute(2, 0, 1)[None]
    img = img - blur(img, torch.ones_like(have))
    m = have * (fr > 0.5).float()
    S, B = FLOW_SEARCH // 2, FLOW_BLOCK // 2
    pool = lambda x: F.avg_pool2d(x, B)
    mw = pool(m)
    r_ = ref * m
    mr = pool(r_) / mw.clamp(min=1e-6)
    vr = pool(r_ * ref) / mw.clamp(min=1e-6) - mr**2
    pad = F.pad(img, (S, S, S, S), mode="replicate")
    scores, shifts = [], []
    for dy in range(-S, S + 1):
        for dx in range(-S, S + 1):
            v = pad[..., S + dy : S + dy + h2, S + dx : S + dx + w2]
            mv = pool(v * m) / mw.clamp(min=1e-6)
            vv = pool(v * v * m) / mw.clamp(min=1e-6) - mv**2
            cov = pool(v * r_) / mw.clamp(min=1e-6) - mv * mr
            scores.append(cov / torch.sqrt(vv * vr + 1e-8))
            shifts.append((dx, dy))
    sc = torch.cat(scores, 1)[0]
    sh = torch.tensor(shifts, device=DEV, dtype=torch.float32)
    best = sc.max(0).values
    wts = torch.softmax(sc / 0.02, 0)
    d = torch.einsum("khw,kc->chw", wts, sh) * 2  # full-resolution px
    return d, ((best > FLOW_MIN_NCC) & (mw[0, 0] > 0.5)).float()


def local_flow(cam, gray, free, grids, sharp):
    """A view's local misalignment against the reference (block_shifts), confident blocks smoothed into a displacement field
    [2, 1080, 1920] (px) to add to every projected position. Rotation and zoom (register) cannot absorb what depends on the surface:
    the boards' and stands' depth errors, lens distortion away from the centre."""
    d, conf = block_shifts(cam, gray, free, grids, sharp)
    num, den = gauss_blur(d.permute(1, 2, 0), conf, 1.5)
    d = (num * (den > 0.05)[..., None]).permute(2, 0, 1)
    flow = F.interpolate(d[None], size=(1080, 1920), mode="bilinear", align_corners=False)[0]
    return flow, float(conf.mean()), float(d.norm(dim=0)[den > 0.05].median()) if (den > 0.05).any() else 0.0


def prefetch(ex, fn, items, ahead=12):
    """fn over items in order, at most `ahead` running or waiting at a time (the host has little memory for decoded frames)."""
    pending = [ex.submit(fn, x) for x in items[:ahead]]
    for k in range(len(items)):
        r = pending.pop(0).result()
        if k + ahead < len(items):
            pending.append(ex.submit(fn, items[k + ahead]))
        yield r


def choose_groups(g):
    """Each texel's view group: the one with the highest score density around it."""
    best = torch.full((g.shape[0] * g.shape[1],), -1.0, device=DEV)
    pick = torch.zeros_like(best, dtype=torch.long)
    for k in range(g.score.shape[1]):
        dens, _ = gauss_blur(g.full(g.score[:, k : k + 1]), torch.ones(g.shape, device=DEV), GROUP_SIGMA)
        dens = dens[..., 0].flatten()
        better = dens > best
        best = torch.where(better, dens, best)
        pick = torch.where(better, k, pick)
    g.group = pick[g.idx]
    g.score = None


def fuse(grids, vs, blur, passes=(1, 2, 3)):
    bundles = sorted({v[3][1] for v in vs})
    groups = [len(bundles) if v[3][0] == "a" else bundles.index(v[3][1]) for v in vs]
    for g in grids:
        g.score = torch.zeros(len(g.idx), len(bundles) + 1, device=DEV)
    for pas in passes:
        refs, rejected = {}, []
        if pas == 3:
            for g in grids:
                choose_groups(g)
        sharp, flows = {}, []
        if pas >= 2:
            for g in grids:
                mu = g.full(g.S1 / g.W1.clamp(min=1e-6)[:, None])
                m = g.full((g.W1 > 1e-3).float())
                refs[g.name], _ = gauss_blur(mu, m, LP_SIGMA)
                if FLOW:
                    sharp[g.name] = (mu.mean(-1) - gauss_blur(mu, m, 3.0)[0].mean(-1)).flatten()[g.idx] * (g.W1 > 1e-3)
        t0 = time.time()
        with ThreadPoolExecutor(6) as ex:
            for n, (v, view) in enumerate(zip(vs, prefetch(ex, load, vs))):
                if view is None:
                    continue
                img, free = view
                a = img.to(DEV).permute(2, 0, 1).float() / 255
                soft = torch.from_numpy(cv2.GaussianBlur(img.numpy(), (0, 0), 1.2)).to(DEV).permute(2, 0, 1).float() / 255
                fr = torch.from_numpy(free).to(DEV)[None]
                flow = None
                if FLOW and pas >= 2:
                    flow, cover, mag = local_flow(v[1], a.mean(0), fr, grids, sharp)
                    flows.append((cover, mag))
                for g in grids:
                    splat(g, ((a, soft), fr), v[1], blur[n], pas, refs.get(g.name), rejected, v[3][0], groups[n], flow)
                if n % 400 == 0:
                    print(f"pass {pas} view {n}/{len(vs)} {time.time() - t0:.0f} s", flush=True)
        if flows:
            print(
                f"pass {pas} local flow: confident blocks median {np.median([c for c, _ in flows]):.0%}, shift median {np.median([m for _, m in flows]):.2f} px",
                flush=True,
            )
        if pas == 3:
            print(
                "view-grid pairs rejected (low correlation with the reference):",
                len(rejected),
                np.round(sorted(rejected)[:12], 2),
                flush=True,
            )


def rodrigues(w):
    th = w.norm() + 1e-12
    k = w / th
    K = torch.stack([torch.stack([0 * th, -k[2], k[1]]), torch.stack([k[2], 0 * th, -k[0]]), torch.stack([-k[1], k[0], 0 * th])])
    return torch.eye(3, device=w.device, dtype=w.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


def detail_points(grids):
    """The reference's textured texels on REG_GRIDS (world points) and their detail (high-passed grey): what alignment is judged on."""
    pts, refv = [], []
    for g in grids:
        if g.name not in REG_GRIDS:
            continue
        mu = g.full(g.S1 / g.W1.clamp(min=1e-6)[:, None])
        m = g.full((g.W1 > 1e-3).float())
        lp, _ = gauss_blur(mu, m, 1.0)
        gray = lp.mean(-1).flatten()[g.idx]
        hp = gray - gauss_blur(lp, m, 6.0)[0].mean(-1).flatten()[g.idx]
        ok = (g.W1 > 1e-3) & (hp.abs() > torch.quantile(hp.abs()[g.W1 > 1e-3][::97], 0.5))
        pts.append(g.P[ok])
        refv.append(hp[ok])
    return torch.cat(pts), torch.cat(refv)


def register(grids, vs):
    """Each view's calibration corrected against the reference (pass 1 of fuse): a small rotation about the camera centre and a zoom
        factor, fitted by maximising the correlation between the view's colours at the atlas texels and the reference's there (coarse to fine
        over REG_SIGMAS px of image blur, on up to REG_POINTS textured texels of REG_GRIDS clear of people; the stands' assumed profile is too
    far off to anchor a view). Views calibrated a pixel or two apart
        otherwise fuse into doubled edges and steps where the chosen views change. Returns the views with corrected cameras."""
    pts, refv = detail_points(grids)
    print("register: reference texels", len(pts), flush=True)
    out, moved, gains = [], [], []
    with ThreadPoolExecutor(6) as ex:
        for n, (v, view) in enumerate(zip(vs, prefetch(ex, load, vs))):
            if view is None:
                out.append(v)
                continue
            img, free = view
            cam = v[1].astype(np.float64)
            px, depth = R.src_project(cam, pts)
            inside = (depth > 1) & (px[:, 0] > 20) & (px[:, 0] < 1900) & (px[:, 1] > 20) & (px[:, 1] < 1060)
            sel = torch.nonzero(inside).squeeze(1)
            if len(sel) < 2000:
                out.append(v)
                continue
            sel = sel[torch.randperm(len(sel), device=DEV)[:REG_POINTS]]
            P, ref = pts[sel], refv[sel]
            fr = torch.from_numpy(free).to(DEV)[None, None]
            Rm = torch.tensor(cam[2:11].reshape(3, 3), device=DEV, dtype=torch.float32)
            C = torch.tensor(cam[11:14], device=DEV, dtype=torch.float32)
            f0, k1 = float(cam[0]), float(cam[1])
            cxy = torch.tensor([R.S["cx"], R.S["cy"]], device=DEV)
            prm = torch.zeros(4, device=DEV, requires_grad=True)
            opt = torch.optim.Adam([prm], lr=0.02)

            def project(prm):
                pc = (P - C) @ (rodrigues(prm[:3] * 1e-3) @ Rm).T
                uv = f0 * torch.exp(prm[3] * 1e-3) * pc[:, :2] / pc[:, 2:3]
                return uv * (1 + k1 * (uv**2).sum(-1, keepdim=True) / R.S["fref"] ** 2) + cxy

            def sample(im, uv):
                grid = torch.stack([uv[:, 0] / 1919 * 2 - 1, uv[:, 1] / 1079 * 2 - 1], -1)[None, None]
                return F.grid_sample(im, grid, align_corners=True)[0, :, 0]

            def ncc(a, b, w):
                a = a - (a * w).sum() / w.sum()
                b = b - (b * w).sum() / w.sum()
                return (a * b * w).sum() / torch.sqrt((a * a * w).sum() * (b * b * w).sum() + 1e-9)

            gray0 = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
            detail = lambda sig: torch.from_numpy(
                cv2.GaussianBlur(gray0, (0, 0), sig) - cv2.GaussianBlur(gray0, (0, 0), REG_HP)
            ).to(DEV)[None, None]
            score = lambda im: float(ncc(sample(im, project(prm))[0], ref, sample(fr, project(prm))[0]))
            fine = detail(REG_SIGMAS[-1])
            first = score(fine)
            with torch.enable_grad():
                for sig in REG_SIGMAS:
                    im = detail(sig)
                    for _ in range(REG_STEPS):
                        uv = project(prm)
                        loss = -ncc(sample(im, uv)[0], ref, sample(fr, uv)[0])
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
            last = score(fine)
            if last < first:  # no better: keep the calibration
                with torch.no_grad():
                    prm.zero_()
                last = first
            with torch.no_grad():
                Rn = (rodrigues(prm[:3] * 1e-3) @ Rm).cpu().numpy()
                fn = f0 * math.exp(float(prm[3]) * 1e-3)
                shift = float(prm[:3].norm()) * 1e-3 * f0
            c2 = cam.copy()
            c2[0] = fn
            c2[2:11] = Rn.reshape(-1)
            out.append((v[0], c2.astype(v[1].dtype), v[2], v[3]))
            moved.append(shift)
            gains.append(last - first)
            if n % 400 == 0:
                print(
                    f"register view {n}/{len(vs)}: moved {shift:.2f} px, detail correlation {first:.3f} -> {last:.3f}", flush=True
                )
    moved, gains = np.array(moved), np.array(gains)
    print(
        f"register: views moved px median {np.median(moved):.2f}, p90 {np.quantile(moved, 0.9):.2f}, max {moved.max():.1f}; correlation gain median {np.median(gains):.3f}",
        flush=True,
    )
    return out


def finish(g):
    """Fused colours [h, w, 3] in [0, 1] (NaN where nothing was seen), without group steps if SEAMLESS."""
    mu2 = g.S2 / g.W2.clamp(min=1e-9)[:, None]
    mu1 = g.S1 / g.W1.clamp(min=1e-9)[:, None]
    col = torch.where((g.W2 > 1e-6)[:, None], mu2, torch.where((g.W1 > 1e-4)[:, None], mu1, torch.full_like(mu1, float("nan"))))
    col = g.full(col, float("nan"))
    return (
        seamless(g, col, g.full(torch.where((g.W1 > 1e-4)[:, None], mu1, torch.full_like(mu1, float("nan"))), float("nan")))
        if SEAMLESS
        else col
    )


def seamless(g, col, ref):
    """col [h, w, 3] with the steps where the chosen view group (choose_groups) changes removed, in the gradient domain: the target
    gradient between neighbouring texels is col's, except within GROUP_SEAM texels of a group boundary, where it is the reference's
    (pass 1: every view counted alike, soft but without steps); u minimises the squared misfit to those gradients plus SEAM_PULL times
    its squared distance to the reference (levels far from any boundary follow it), by conjugate gradients from col."""
    valid = torch.isfinite(col).all(-1) & torch.isfinite(ref).all(-1)
    grp = g.full(g.group.float(), -1.0)
    u0, r0 = torch.nan_to_num(col), torch.nan_to_num(ref)
    ex, ey = valid[:, :-1] & valid[:, 1:], valid[:-1] & valid[1:]
    edge = torch.zeros_like(valid)
    edge[:, :-1] |= ex & (grp[:, :-1] != grp[:, 1:])
    edge[:-1] |= ey & (grp[:-1] != grp[1:])
    near = F.max_pool2d(edge[None, None].float(), 2 * GROUP_SEAM + 1, 1, GROUP_SEAM)[0, 0] > 0
    nx, ny = near[:, :-1] | near[:, 1:], near[:-1] | near[1:]
    gx = torch.where(nx[..., None], r0[:, 1:] - r0[:, :-1], u0[:, 1:] - u0[:, :-1]) * ex[..., None]
    gy = torch.where(ny[..., None], r0[1:] - r0[:-1], u0[1:] - u0[:-1]) * ey[..., None]
    v = valid[..., None].float()
    exf, eyf = ex[..., None].float(), ey[..., None].float()

    def apply(u):  # (graph Laplacian + SEAM_PULL) u on the valid texels
        out = SEAM_PULL * u * v
        dx = (u[:, 1:] - u[:, :-1]) * exf
        dy = (u[1:] - u[:-1]) * eyf
        out[:, :-1] -= dx
        out[:, 1:] += dx
        out[:-1] -= dy
        out[1:] += dy
        return out

    b = SEAM_PULL * r0 * v
    b[:, :-1] -= gx
    b[:, 1:] += gx
    b[:-1] -= gy
    b[1:] += gy
    contrast = lambda u: float(
        ((u[:, 1:] - u[:, :-1]).abs().mean(-1)[ex & (grp[:, :-1] != grp[:, 1:])].mean())
        / (u[:, 1:] - u[:, :-1]).abs().mean(-1)[ex & nx & (grp[:, :-1] == grp[:, 1:])].mean()
    )
    x = u0 * v
    r = b - apply(x)
    p = r.clone()
    rr = (r * r).sum((0, 1))
    for _ in range(SEAM_ITERS):
        Ap = apply(p)
        a = rr / (p * Ap).sum((0, 1)).clamp(min=1e-20)
        x += a * p
        r -= a * Ap
        rr2 = (r * r).sum((0, 1))
        p = r + rr2 / rr.clamp(min=1e-20) * p
        rr = rr2
    out = torch.where(valid[..., None], x.clamp(0, 1), col)
    print(
        f"{g.name} seamless: boundary texels {float(edge.float().mean()):.1%}, step across a boundary / beside it {contrast(u0):.2f} -> {contrast(out):.2f}",
        flush=True,
    )
    return out


def fill_out(name, col, res):
    """Fill unseen texels of a boards/stands grid [h, w, 3] (NaN): from the arena's symmetry (the half turn about the centre, which on
    this outline is the shift s + PERIM / 2, so sponsor text reads the right way), rows above the highest one seen anywhere as dark structure, and
    what is still empty by copying observed stretches of the same rows in square patches. Returns colours, the filled mask and the profile
    height up to which the camera saw this grid."""
    h_, w_ = col.shape[:2]
    filled = torch.isnan(col[..., 0])
    seen_rows = (~filled).float().mean(1)
    top = int(torch.nonzero(seen_rows > 0.1).max()) + 1 if (seen_rows > 0.1).any() else 0
    gs = torch.arange(w_, device=DEV) * res
    gh = torch.arange(h_, device=DEV) * res
    OH, OS = torch.meshgrid(gh, gs, indexing="ij")
    ox, oy, oz = R.out_world(OS, OH)
    miss = torch.isnan(col[..., 0])  # the half turn (a mirror would reverse every banner's lettering)
    sd_ = R.sdf(-ox, -oy).clamp(min=0)
    ms, mh = R.outline_s(-ox, -oy), R.out_h(sd_, oz)
    src = col[(mh / res).round().long().clamp(0, h_ - 1), (ms / res).round().long().clamp(0, w_ - 1)]
    col[miss] = src[miss]
    col[top:] = torch.tensor(ROOF, device=DEV)
    miss = torch.isnan(col[..., 0])
    print(
        name,
        f"seen up to h {top * res:.2f} m, unseen below it after symmetry {100 * miss[:top].float().mean():.2f}%",
        flush=True,
    )
    c = col.cpu().numpy()
    miss = miss.cpu().numpy()
    pw = max(int(1.5 / res), 8)
    for r in range(top):  # rows: copy observed stretches of the same row (or the nearest seen row),
        if not miss[r].any():
            continue  # from the same columns over a block of pw rows: square patches of crowd
        src_r = r
        while miss[src_r].all() and src_r > 0:
            src_r -= 1
        if miss[src_r].all():
            continue
        ok = np.flatnonzero(~miss[src_r])
        starts = np.random.default_rng(r // pw).integers(w_, size=w_ // pw + 1)
        for k, c0 in enumerate(range(0, w_, pw)):
            seg = np.flatnonzero(miss[r, c0 : c0 + pw]) + c0
            if not len(seg):
                continue
            start = ok[np.searchsorted(ok, starts[k]) % len(ok)]
            src_cols = np.clip(start + (seg - c0), 0, w_ - 1)
            take = ~miss[src_r, src_cols]
            c[r, seg[take]] = c[src_r, src_cols[take]]
    left = np.isnan(c[..., 0])
    if left.any():  # isolated leftovers: nearest seen texel
        from scipy import ndimage

        _, (ii, jj) = ndimage.distance_transform_edt(left, return_indices=True)
        c = c[ii, jj]
    return np.clip(c, 0, 1), filled.cpu().numpy(), top * res


def fill_ice(col):
    miss = torch.isnan(col[..., 0])
    filled = miss.clone()
    src = torch.flip(col, [0, 1])
    col[miss] = src[miss]  # the half turn: ice logos keep reading the right way
    c = col.cpu().numpy()
    hole = np.isnan(c[..., 0])
    c8 = (np.nan_to_num(c, nan=0.9) * 255).astype(np.uint8)
    if hole.any():
        c8 = cv2.inpaint(c8, hole.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return c8, filled.cpu().numpy()


def smear_field(rgb8, res):
    """Per texel: how consistently its surroundings (SMEAR_SPAN m) run in one direction (resultant length of the doubled gradient
    angles, energy weighted: crowd is random, near 0; a smear near 1) and how far that direction is from the grid axes (deg; rows,
    railings and stairs run along them)."""
    g = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    sm = lambda x: cv2.GaussianBlur(x, (0, 0), SMEAR_SPAN / res)
    jxx, jyy, jxy = sm(gx * gx), sm(gy * gy), sm(gx * gy)
    consistency = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (jxx + jyy + 1e-3)
    ang = np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy)) % 90
    return consistency, np.minimum(ang, 90 - ang)


def refill_ice(out):
    """An existing build's filled ice texels again from the half turn about the centre where that was seen, then shaded to meet the
    seen ice: the colour difference along the seam (both sides averaged over SEAM_SIGMA), spread smoothly into the fill (SEAM_SPREAD)."""
    ice = cv2.imread(f"{out}/ice.jpg")
    filled = cv2.imread(f"{out}/ice_fill.png", 0) > 127
    take = filled & ~filled[::-1, ::-1]
    ice[take] = ice[::-1, ::-1][take]
    k = 8
    small = cv2.resize(ice, None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA).astype(np.float32)
    f = cv2.resize(filled.astype(np.uint8), small.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(np.float32)[..., None]
    blur = lambda x, sig: cv2.GaussianBlur(x, (0, 0), sig / ICE_RES / k).reshape(x.shape)
    side = lambda m: blur(small * m, SEAM_SIGMA) / (blur(m, SEAM_SIGMA) + 1e-3)
    seam = (cv2.dilate(f[..., 0], np.ones((3, 3))) - cv2.erode(f[..., 0], np.ones((3, 3))))[..., None]
    diff = (side(1 - f) - side(f)) * seam
    field = blur(diff, SEAM_SPREAD) / (blur(seam, SEAM_SPREAD) + 1e-3) * f
    field = cv2.resize(field, ice.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    ice = np.where(filled[..., None], np.clip(ice + field, 0, 255), ice).astype(np.uint8)
    cv2.imwrite(f"{out}/ice.jpg", ice, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(
        f"ice filled {100 * filled.mean():.1f}%, from the half turn {100 * take.mean():.1f}%, seam shading mean {np.abs(field[filled]).mean():.1f} levels",
        flush=True,
    )


def desmear(out):
    """Stands texels the camera saw only side-on (the seats of its own-side corners): their colour runs in streaks along one oblique
    direction. Marked filled and filled like unseen ones (fill_out)."""
    rgb8 = cv2.cvtColor(cv2.imread(f"{out}/stands.jpg"), cv2.COLOR_BGR2RGB)
    filled = cv2.imread(f"{out}/stands_fill.png", 0) > 127
    cons, off = smear_field(rgb8, STAND_RES)
    smear = cv2.dilate(((cons > SMEAR_MIN) & (off > SMEAR_OFF_AXIS) & ~filled).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    col = torch.from_numpy(rgb8.astype(np.float32) / 255).to(DEV)
    col[torch.from_numpy(smear | filled).to(DEV)] = float("nan")
    c, _, _ = fill_out("stands", col, STAND_RES)
    write(out, "stands", (c * 255).round().astype(np.uint8), smear | filled)
    print(
        f"stands smeared {100 * smear[~filled].mean():.1f}% of seen, refilled; filled now {100 * (smear | filled).mean():.1f}%",
        flush=True,
    )


def write(out, name, rgb8, filled):
    cv2.imwrite(f"{out}/{name}.jpg", cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
    cv2.imwrite(f"{out}/{name}_fill.png", filled.astype(np.uint8) * 255)


def fold(band8, filled):
    """The band [h, W] as STRIPS strips of L columns stacked vertically, each with PAD texels of neighbouring columns on both sides
    and PAD rows of edge repeat above and below."""
    h_, w_ = band8.shape[:2]
    L = int(math.ceil(w_ / STRIPS))
    rows8, rowsf = [], []
    for k in range(STRIPS):
        cols = np.arange(k * L - PAD, (k + 1) * L + PAD) % w_
        for src, dst in ((band8, rows8), (filled, rowsf)):
            s_ = src[:, cols]
            dst.append(np.concatenate([np.repeat(s_[:1], PAD, 0), s_, np.repeat(s_[-1:], PAD, 0)], 0))
    return np.concatenate(rows8, 0), np.concatenate(rowsf, 0), L


def unfold(folded, meta):
    """The band [h, W] back from its folded strips (fold)."""
    P, L, sr = meta["pad"], meta["strip_cols"], meta["strip_rows"]
    return np.concatenate([folded[k * sr + P : (k + 1) * sr - P, P : P + L] for k in range(meta["strips"])], 1)


def level_offsets(gray):
    """Per band column: how many texels the boards' profile (kick plate, sponsor panels, cap) sits below its median height there,
    from the vertical gradient profile of LEVEL_BLOCK blocks matched against the median block, smoothed along s by match quality."""
    w, top, span = int(LEVEL_BLOCK / BAND_RES), int(BOARDS_TOP / BAND_RES), int(LEVEL_SPAN / BAND_RES)
    nb = gray.shape[1] // w
    g = np.diff(gray[:top, : nb * w].reshape(top, nb, w).mean(2), axis=0).T
    ref = np.median(g, 0)[span:-span]
    score = np.stack(
        [
            [
                np.dot(np.roll(x, d)[span:-span], ref) / (np.linalg.norm(np.roll(x, d)[span:-span]) * np.linalg.norm(ref) + 1e-6)
                for d in range(-span, span + 1)
            ]
            for x in g
        ]
    )
    off = (score.argmax(1) - span).astype(np.float32)
    conf = np.clip(score.max(1), 0, 1) ** 2
    sm = lambda x: cv2.GaussianBlur(np.tile(x, 3)[None], (0, 0), LEVEL_SIGMA / LEVEL_BLOCK)[0, nb : 2 * nb]  # periodic in s
    off = sm(off * conf) / (sm(conf) + 1e-6)
    return np.interp(np.arange(gray.shape[1]), (np.arange(nb) + 0.5) * w, off, period=nb * w)


def repair_band(out):
    """The band's filled texels again (fill_out: the half turn, no mirrors), then levelled: each column shifted so the boards' profile
    runs at one height all round (the end boards are seen about 12 cm low, ramping through the corners), the shift fading out by
    LEVEL_TOP so the band still meets the stands."""
    meta = json.load(open(f"{out}/atlas.json"))["band"]
    rgb8 = unfold(cv2.cvtColor(cv2.imread(f"{out}/band.jpg"), cv2.COLOR_BGR2RGB), meta)
    filled = unfold(cv2.imread(f"{out}/band_fill.png", 0) > 127, meta)
    col = torch.from_numpy(rgb8.astype(np.float32) / 255).to(DEV)
    col[torch.from_numpy(filled).to(DEV)] = float("nan")
    c, _, _ = fill_out("band", col, BAND_RES)
    rgb8 = (c * 255).round().astype(np.uint8)
    off = level_offsets(cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float32))
    h_, w_ = filled.shape
    r = np.arange(h_, dtype=np.float32)[:, None]
    fade = np.clip((LEVEL_TOP / BAND_RES - r) / ((LEVEL_TOP - BOARDS_TOP) / BAND_RES), 0, 1)
    my = (r - off[None].astype(np.float32) * fade).astype(np.float32)
    mx = np.broadcast_to(np.arange(w_, dtype=np.float32), (h_, w_)).copy()
    rgb8 = cv2.remap(rgb8, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    filled = cv2.remap(filled.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0
    write(out, "band", *fold(rgb8, filled)[:2])
    print(
        "band levelled: shift cm p5 {:.1f}, median {:.1f}, p95 {:.1f}".format(*tuple(np.percentile(off, [5, 50, 95]))), flush=True
    )


def build(out):
    global C0
    os.makedirs(out, exist_ok=True)
    vs = views()
    lim = int(os.environ.get("LIMIT", 0))
    vs = vs[:: max(len(vs) // lim, 1)] if lim else vs
    C0 = torch.tensor(np.median([v[1][11:14] for v in vs], 0), device=DEV, dtype=torch.float32)
    print("views", len(vs), "camera", C0.tolist(), flush=True)
    blur = motion_blur(vs)
    print(f"motion blur px/frame: median {np.median(blur):.2f}, 90% {np.quantile(blur, 0.9):.2f}", flush=True)
    with torch.no_grad():
        grids = build_grids()
        for _ in range(REG_ITERS):  # reference, views registered to it, again
            fuse(grids, vs, blur, (1,))
            vs = register(grids, vs)
            for g in grids:
                g.S1.zero_()
                g.W1.zero_()
        fuse(grids, vs, blur)
        meta = dict(camera=C0.tolist(), views=len(vs))
        for g in grids:
            col = finish(g)
            if g.name == "ice":
                c8, filled = fill_ice(col)
                meta["ice"] = dict(file="ice.jpg", res=ICE_RES, rows=c8.shape[0], cols=c8.shape[1], x0=-R.HX, y0=-R.HY)
            else:
                c, filled, top = fill_out(g.name, col, g.res)
                c8 = (c * 255).round().astype(np.uint8)
                if g.name == "band":
                    c8, filled, L = fold(c8, filled)
                    meta["band"] = dict(
                        file="band.jpg",
                        res=BAND_RES,
                        top=BAND_TOP,
                        strips=STRIPS,
                        strip_cols=L,
                        pad=PAD,
                        rows=c8.shape[0],
                        cols=c8.shape[1],
                        strip_rows=c8.shape[0] // STRIPS,
                    )
                else:
                    meta["stands"] = dict(file="stands.jpg", res=STAND_RES, rows=c8.shape[0], cols=c8.shape[1], seen_top=top)
            write(out, g.name, c8, filled)
            print(g.name, c8.shape, "filled %.1f%%" % (100 * filled.mean()), flush=True)
    json.dump(meta, open(f"{out}/atlas.json", "w"), indent=1)


def link(out, scenes):
    name = os.path.basename(os.path.normpath(out))
    meta = json.load(open(f"{out}/atlas.json"))
    for part in meta.values():  # every image of every part (glare: ice_glare.py's roof and matte ice)
        if isinstance(part, dict):
            for k in ("file", "plain", "matte"):
                if k in part:
                    part[k] = f"../{name}/{part[k]}"
    for d in sorted(glob.glob(f"{scenes}/*/scene.json")):
        json.dump(meta, open(os.path.join(os.path.dirname(d), "atlas.json"), "w"), indent=1)
        print(os.path.dirname(d))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
    elif sys.argv[1] == "link":
        link(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "repair":
        desmear(sys.argv[2])
        refill_ice(sys.argv[2])
        repair_band(sys.argv[2])
