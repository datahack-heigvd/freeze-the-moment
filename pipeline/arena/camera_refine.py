"""Per-frame broadcast cameras registered to the arena atlas, so what is drawn over the footage stays on it.

The atlas holds the static arena (ice markings, boards) fused from every calibrated view. Each goal frame's camera is corrected
against it with arena_atlas.register (a small rotation about the fixed camera centre and a zoom, maximising the correlation of the
frame's detail with the atlas's on the ice and the boards, people masked out), then rotation and zoom are smoothed over time with a
Gaussian of SMOOTH frames (the camera head turns smoothly; the registration's own noise does not). The lens is shared by every
frame: lens fits one correction to the distortion (which the calibration ties to the zoom, k = k(f)) and the principal point, jointly
with every sampled frame's rotation and zoom, and register applies it (<out>/lens.json) before registering.

The registration alone is noisy where the camera pans fast, and smoothing it lags behind the pan. Tracks instead fuses it with the
camera's motion between frames: corners of the static background (clear of people, forward-backward checked) are followed from each
frame to the next (KLT) and give the small rotation and zoom of each frame relative to the previous one, to a few hundredths of a
pixel. Every frame's correction x_i (rotation about the camera axes, log zoom) then minimises the sum of its distance to the
registration (sigma ANCHOR px) and of the misfit of the frame-to-frame motion (sigma MOTION px), a linear least squares along the
whole window. The motion carries a small bias (corners on moving crowd, blur: hundredths of a pixel per frame, pixels of drift over a
window) solved for with it as one constant per goal; frame pairs count less the worse their corners fit; registrations further
than 3 ANCHOR from the solution are down-weighted (twice). Against registration alone (9 goals, means): overlays slide 0.19 px
between frames instead of 0.31, at about the same distance from the footage (4.2 px against 4.1); a bias growing with the motion fits no better, and
down-weighting by the median distance lets the motion drift.

Measured per frame with arena_atlas.block_shifts (the atlas's detail drawn into the frame, matched in 64 px blocks):
  residual    median shift of the confident blocks (px): how far an overlay sits from the footage
  jitter      median change of a block's shift from one frame to the next (px/frame): how much an overlay wobbles on the footage
  roughness   median second difference of a fixed world point's image position (px/frame^2): how unsteady the camera path is
  misfit      median distance (px) between where a static background corner was followed to in the next frame and where the cameras
              put it (half the corners, never used by tracks): how much an overlay slides on the footage between frames

usage (LTX venv, from pipeline/):  camera_refine.py register <atlas> <out> [goal...]   writes <out>/<goal>.npz (cam: the refined
                                                                                         cameras [n, 14], cam0: the original ones, cxy:
                                                                                         the principal point) and appends the measures
                                                                                         to <out>/measures.jsonl
                                        camera_refine.py fuse <atlas> <out> [goal...]   tracks again from the saved registrations
                                                                                (<out>/<goal>.npz reg), measured and saved as cam
                                        camera_refine.py steady <atlas> <out> [goal...]   tracks anchored to the bundle's calibrated
                                                                                cameras instead of registrations (keeps the calibration's
                                                                                placement in the rink; the atlas carries its small
                                                                                offsets): <out>/<goal>.npz cam, cam0, reg
                                        camera_refine.py adopt - <out> [goal...]   <out>'s cameras into the bundles, bodies moved with
                                                                                them (see adopt)
                                        camera_refine.py lens <atlas> <out>     fits <out>/lens.json on every LENS_STEP-th frame of the
                                                                                registered goals
                                        camera_refine.py video <atlas> <out> <goal> <mp4>   the footage with the atlas's ice markings
                                                                                (magenta) and the boards' foot and top (cyan) drawn
                                                                                through the original cameras (left) and the refined
                                                                                ones (right)"""

import glob
import json
import math
import os
import shutil
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from arena import arena_atlas as A
from goal import render as R
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

DEV = R.DEV
SMOOTH = [float(s) for s in os.environ.get("SMOOTH", "0,1").split(",")]  # frames: Gaussian sigmas tried for rotation and zoom
LENS_STEP, LENS_POINTS, LENS_STEPS = 25, 50_000, 60  # frames, texels per frame, Adam steps per blur level
ANCHOR, MOTION, MIN_TRACKS = 0.5, 0.05, 30  # px, px, corners for a frame pair to count


def reference(atlas, grids):
    """The atlas's colours as each grid's pass-1 reference (S1, W1: seen texels only, as register reads them) and its detail per texel
    (as block_shifts reads it)."""
    meta = json.load(open(f"{atlas}/atlas.json"))
    sharp = {}
    for g in grids:
        if g.name == "stands":
            continue
        rgb = cv2.cvtColor(cv2.imread(f"{atlas}/{g.name}.jpg"), cv2.COLOR_BGR2RGB)
        filled = cv2.imread(f"{atlas}/{g.name}_fill.png", 0) > 127
        if g.name == "band":
            rgb, filled = A.unfold(rgb, meta["band"]), A.unfold(filled, meta["band"])
        rgb, filled = rgb[: g.shape[0], : g.shape[1]], filled[: g.shape[0], : g.shape[1]]
        col = torch.from_numpy(rgb.reshape(-1, 3)).to(DEV).float()[g.idx] / 255
        seen = torch.from_numpy(~filled.reshape(-1)).to(DEV)[g.idx].float()
        g.S1, g.W1 = col * seen[:, None], seen
        mu = g.full(col)
        m = g.full(seen)
        sharp[g.name] = (mu.mean(-1) - A.gauss_blur(mu, m, 3.0)[0].mean(-1)).flatten()[g.idx] * seen
    return sharp


def frames(b):
    S = json.load(open(f"{b}/scene.json"))
    Z = np.load(f"{b}/scene.npz")
    return [(f"{b}/frames/{i + 1:03d}.jpg", Z["cam"][i], S["outlines"][i], ("w", b, i)) for i in range(S["n"])]


def smooth(cams, sigma):
    """Rotation (as rotation vectors about the mean orientation) and log focal length smoothed over time."""
    if sigma == 0:
        return cams.copy()
    Rs = Rotation.from_matrix(cams[:, 2:11].reshape(-1, 3, 3))
    mean = Rs.mean()
    w = gaussian_filter1d((Rs * mean.inv()).as_rotvec(), sigma, axis=0, mode="nearest")
    out = cams.copy()
    out[:, 2:11] = (Rotation.from_rotvec(w) * mean).as_matrix().reshape(-1, 9)
    out[:, 0] = np.exp(gaussian_filter1d(np.log(cams[:, 0]), sigma, mode="nearest"))
    return out


def follow(vs):
    """Per frame pair (i, i + 1): background corners in i and where KLT followed them in i + 1 ([k, 2] each, forward-backward within
    0.3 px, both frames clear of people there), alternately split into a fitting and a measuring half."""
    pairs, prev = [], None
    for v in vs:
        img = cv2.imread(v[0], cv2.IMREAD_GRAYSCALE)
        free = R.person_mask(v[2], 31) == 0
        if prev is not None:
            (ga, fa), gb = prev, img
            pts = cv2.goodFeaturesToTrack(ga, 2000, 0.01, 10, mask=(fa & free).astype(np.uint8) * 255, blockSize=7)
            if pts is None:
                pairs.append(None)
            else:
                lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
                p1, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None, **lk)
                p0, st2, _ = cv2.calcOpticalFlowPyrLK(gb, ga, p1, None, **lk)
                ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(p0 - pts, axis=2)[:, 0] < 0.3)
                a, b = pts[ok, 0].astype(np.float64), p1[ok, 0].astype(np.float64)
                pairs.append(((a[0::2], b[0::2]), (a[1::2], b[1::2])))
        prev = (img, free)
    return pairs


def carry(cam_a, cam_b, ua, prm=None):
    """Pixels ua [k, 2] of camera a seen by camera b, with b corrected by prm (rotation about b's axes [3] in rad, log zoom) (torch)."""
    cxy = torch.tensor([R.S["cx"], R.S["cy"]], device=DEV, dtype=torch.float64)
    fref = R.S["fref"]
    d = torch.tensor(ua, device=DEV) - cxy
    q = d.clone()
    for _ in range(8):
        q = d / (1 + float(cam_a[1]) * (q**2).sum(-1, keepdim=True) / fref**2)
    ray = torch.cat([q / float(cam_a[0]), torch.ones_like(q[:, :1])], 1) @ torch.tensor(cam_a[2:11].reshape(3, 3), device=DEV)
    P = torch.tensor(cam_a[11:14], device=DEV) + 40 * ray / ray.norm(dim=1, keepdim=True)
    Rb = torch.tensor(cam_b[2:11].reshape(3, 3), device=DEV)
    fb = float(cam_b[0])
    if prm is not None:
        Rb = A.rodrigues(prm[:3]) @ Rb
        fb = fb * torch.exp(prm[3])
    pc = (P - torch.tensor(cam_b[11:14], device=DEV)) @ Rb.T
    uv = fb * pc[:, :2] / pc[:, 2:3]
    return uv * (1 + float(cam_b[1]) * (uv**2).sum(-1, keepdim=True) / fref**2) + cxy


def scales(cam):
    """px per unit of each correction component near the image centre: tilt, pan (f), roll and log zoom (a typical radius)."""
    return np.array([cam[0], cam[0], 600.0, 600.0])


def motion(reg, pairs):
    """Per frame pair: the correction of frame i + 1 that makes its camera agree with frame i's on the followed corners ([n - 1, 4]),
    whether the pair counts, and the corners' remaining misfit (px, median)."""
    n = len(reg)
    e, wm, rest = np.zeros((n - 1, 4)), np.zeros(n - 1), np.full(n - 1, np.nan)
    for i, pr in enumerate(pairs):
        if pr is None or len(pr[0][0]) < MIN_TRACKS:
            continue
        (ua, ub), _ = pr
        target = torch.tensor(ub, device=DEV)
        prm = torch.zeros(4, device=DEV, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([prm], max_iter=60, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            r = (carry(reg[i], reg[i + 1], ua, prm) - target).norm(dim=1)
            loss = torch.where(r < 1, 0.5 * r**2, r - 0.5).mean()
            loss.backward()
            return loss

        with torch.enable_grad():
            opt.step(closure)
        e[i], wm[i] = prm.detach().cpu().numpy(), 1.0
        with torch.no_grad():
            rest[i] = float((carry(reg[i], reg[i + 1], ua, prm) - target).norm(dim=1).median())
    return e, wm, rest


def tracks(reg, pairs):
    """The registered cameras fused with the frame-to-frame motion (see above)."""
    n = len(reg)
    e, wm, rest = motion(reg, pairs)
    sig_m = MOTION * np.maximum(1, np.nan_to_num(rest, nan=1.0) / np.nanmedian(rest))  # pairs whose corners fit worse count less
    s = np.stack([scales(c) for c in reg])
    wa = np.ones(n)
    keep = np.flatnonzero(wm > 0)
    for _ in range(3):
        x, bias = np.zeros((n, 4)), np.zeros(4)
        for c in range(4):
            Ma = np.diag(s[:, c] / ANCHOR * np.sqrt(wa))
            Mm = np.zeros((len(keep), n + 1))  # unknowns: x_i, the motion's bias
            w = s[keep + 1, c] / sig_m[keep]
            Mm[np.arange(len(keep)), keep + 1], Mm[np.arange(len(keep)), keep], Mm[:, n] = w, -w, -w
            M = np.r_[np.c_[Ma, np.zeros(n)], Mm]
            b = np.r_[np.zeros(n), w * e[keep, c]]
            sol = np.linalg.lstsq(M, b, rcond=None)[0]
            x[:, c], bias[c] = sol[:n], sol[n]
        off = np.linalg.norm(x * s, axis=1)
        wa = np.where(off > 3 * ANCHOR, (3 * ANCHOR / np.maximum(off, 1e-9)) ** 2, 1.0)
    out = reg.copy()
    for i in range(n):
        out[i, 2:11] = (A.rodrigues(torch.tensor(x[i, :3])).numpy() @ reg[i, 2:11].reshape(3, 3)).reshape(-1)
        out[i, 0] = reg[i, 0] * math.exp(x[i, 3])
    print(
        f"tracks: pairs used {len(keep)}/{n - 1}, motion bias px/frame {np.round(bias * s.mean(0), 3).tolist()}, registrations down-weighted "
        f"{int((wa < 1).sum())}, correction px median {np.median(off):.2f}",
        flush=True,
    )
    return out


def misfit(cams, pairs):
    m = [
        float(np.median((carry(cams[i], cams[i + 1], pr[1][0]) - torch.tensor(pr[1][1], device=DEV)).norm(dim=1).cpu().numpy()))
        for i, pr in enumerate(pairs)
        if pr is not None and len(pr[1][0]) >= MIN_TRACKS
    ]
    return dict(misfit=float(np.median(m)), misfit_p90=float(np.percentile(m, 90)))


def measure(cams, imgs, grids, sharp):
    res, conf_share, jit, prev = [], [], [], None
    for cam, (gray, free) in zip(cams, imgs):
        d, conf = A.block_shifts(cam, gray.float(), free.float(), grids, sharp)
        ok = conf > 0
        conf_share.append(float(conf.mean()))
        res.append(float(d.norm(dim=0)[ok].median()) if ok.any() else math.nan)
        if prev is not None:
            both = ok & prev[1]
            if both.sum() >= 3:
                jit.append(float((d - prev[0]).norm(dim=0)[both].median()))
        prev = (d, ok)
    rough = []
    for i in range(1, len(cams) - 1):
        P = cams[i][11:14] + 50 * A.ray_dir(cams[i], (R.S["cx"], R.S["cy"]))
        uv = np.stack(
            [
                R.src_project(cams[j].astype(np.float64), torch.tensor(P, device=DEV, dtype=torch.float64)[None])[0][0]
                .cpu()
                .numpy()
                for j in (i - 1, i, i + 1)
            ]
        )
        rough.append(float(np.linalg.norm(uv[0] - 2 * uv[1] + uv[2])))
    q = lambda x, p: float(np.nanpercentile(x, p)) if len(x) else math.nan
    return dict(
        residual=q(res, 50),
        residual_p90=q(res, 90),
        jitter=q(jit, 50),
        jitter_p90=q(jit, 90),
        roughness=q(rough, 50),
        roughness_p90=q(rough, 90),
        confident=float(np.median(conf_share)),
    )


def with_lens(cam, ln):
    """Cameras [n, 14] with the lens correction's distortion."""
    out = cam.copy()
    out[:, 1] = cam[:, 1] * math.exp(ln["k_scale"]) + ln["k_add"]
    return out


def lens(goals, out, grids):
    """One lens for every frame: the distortion k -> k exp(a) + b and the principal point moved by (dx, dy) px, fitted with every
    LENS_STEP-th frame's own small rotation and zoom by maximising the mean detail correlation with the atlas (as register does per
    frame). Fitted twice, with the lens held fixed and with it free, so the gain is the lens's own."""
    pts, refv = A.detail_points(grids)
    items = []
    for goal in goals:
        cams = np.load(f"{out}/{goal}.npz")["cam"].astype(np.float64)
        vs = frames(f"{A.GOALS}/{goal}/bundle")
        for i in range(0, len(vs), LENS_STEP):
            img, free = A.load(vs[i])
            cam = cams[i]
            px, depth = R.src_project(cam, pts)
            sel = torch.nonzero((depth > 1) & (px[:, 0] > 20) & (px[:, 0] < 1900) & (px[:, 1] > 20) & (px[:, 1] < 1060)).squeeze(
                1
            )
            if len(sel) < 2000:
                continue
            sel = sel[torch.randperm(len(sel), device=DEV)[:LENS_POINTS]]
            gray = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
            ims = [
                torch.from_numpy(cv2.GaussianBlur(gray, (0, 0), s) - cv2.GaussianBlur(gray, (0, 0), A.REG_HP)).to(DEV)[None, None]
                for s in A.REG_SIGMAS
            ]
            items.append((pts[sel], refv[sel], cam, torch.from_numpy(free).to(DEV)[None, None], ims))
    print("lens: frames", len(items), flush=True)
    cxy0 = torch.tensor([R.S["cx"], R.S["cy"]], device=DEV)

    def ncc(a, b, w):
        a = a - (a * w).sum() / w.sum()
        b = b - (b * w).sum() / w.sum()
        return (a * b * w).sum() / torch.sqrt((a * a * w).sum() * (b * b * w).sum() + 1e-9)

    def score(prm, shared, level):
        total = 0
        for n, (P, ref, cam, fr, ims) in enumerate(items):
            Rm = torch.tensor(cam[2:11].reshape(3, 3), device=DEV, dtype=torch.float32)
            C = torch.tensor(cam[11:14], device=DEV, dtype=torch.float32)
            pc = (P - C) @ (A.rodrigues(prm[n, :3] * 1e-3) @ Rm).T
            uv = float(cam[0]) * torch.exp(prm[n, 3] * 1e-3) * pc[:, :2] / pc[:, 2:3]
            k = float(cam[1]) * torch.exp(shared[0] * 1e-2) + shared[1] * 1e-2
            uv = uv * (1 + k * (uv**2).sum(-1, keepdim=True) / R.S["fref"] ** 2) + cxy0 + shared[2:4]
            grid = torch.stack([uv[:, 0] / 1919 * 2 - 1, uv[:, 1] / 1079 * 2 - 1], -1)[None, None]
            s = lambda im: F.grid_sample(im, grid, align_corners=True)[0, 0, 0]
            total = total + ncc(s(ims[level]), ref, s(fr))
        return total / len(items)

    fits = {}
    for free_lens in (False, True):
        prm = torch.zeros(len(items), 4, device=DEV, requires_grad=True)
        shared = torch.zeros(4, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([prm, shared] if free_lens else [prm], lr=0.02)
        with torch.enable_grad():
            first = float(score(prm, shared, -1))
            for level in range(len(A.REG_SIGMAS)):
                for _ in range(LENS_STEPS):
                    loss = -score(prm, shared, level)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            last = float(score(prm, shared, -1))
        fits[free_lens] = (first, last, shared.detach().cpu().numpy())
        print(f"lens {'free' if free_lens else 'fixed'}: mean detail correlation {first:.4f} -> {last:.4f}", flush=True)
    a, b, dx, dy = fits[True][2]
    ln = dict(
        k_scale=float(a) * 1e-2,
        k_add=float(b) * 1e-2,
        dx=float(dx),
        dy=float(dy),
        frames=len(items),
        correlation=dict(lens_fixed=fits[False][1], lens_free=fits[True][1]),
    )
    print("lens:", json.dumps(ln), flush=True)
    json.dump(ln, open(f"{out}/lens.json", "w"), indent=1)


def register(goals, out, grids, sharp):
    """Every goal frame registered to the atlas (with the lens correction if fitted), smoothed over time, measured; the original
    cameras are measured only without a lens correction (it moves the principal point every projection uses)."""
    ln = json.load(open(f"{out}/lens.json")) if os.path.exists(f"{out}/lens.json") else None
    if ln:
        R.S["cx"] += ln["dx"]
        R.S["cy"] += ln["dy"]
    for goal in goals:
        b = f"{A.GOALS}/{goal}/bundle"
        vs = frames(b)
        cam0 = np.stack([v[1] for v in vs]).astype(np.float64)
        if ln:
            vs = [(p, c, o, key) for (p, _, o, key), c in zip(vs, with_lens(cam0, ln))]
        imgs = []
        for v in vs:
            img, free = A.load(v)
            imgs.append(
                (
                    torch.from_numpy(cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2GRAY)).to(DEV).half() / 255,
                    torch.from_numpy(free).to(DEV)[None].half(),
                )
            )
        reg = np.stack([v[1] for v in A.register(grids, vs)]).astype(np.float64)
        pairs = follow(vs)
        pre = "lens, " if ln else ""
        rows = ([] if ln else [("original", cam0)]) + [(f"{pre}registered, smoothed {s:g}", smooth(reg, s)) for s in SMOOTH]
        rows.append((f"{pre}registered + tracks", tracks(reg, pairs)))
        best = None
        with open(f"{out}/measures.jsonl", "a") as f:
            for name, cams in rows:
                m = dict(goal=goal, cameras=name, **measure(cams, imgs, grids, sharp), **misfit(cams, pairs))
                f.write(json.dumps(m) + "\n")
                print(json.dumps(m), flush=True)
                if name != "original" and (best is None or m["jitter"] + m["residual"] < best[0]):
                    best = (m["jitter"] + m["residual"], cams)
        np.savez(
            f"{out}/{goal}.npz",
            cam=best[1].astype(np.float32),
            cam0=cam0.astype(np.float32),
            reg=reg.astype(np.float32),
            cxy=np.array([R.S["cx"], R.S["cy"]]),
        )
        del imgs
        torch.cuda.empty_cache()


def overlay(img, cam, cxy, ice, meta):
    """The frame with the atlas's ice markings (edges of the atlas's grey and colour saturation drawn onto the ice through cam) and the boards' foot and top."""
    v, u = torch.meshgrid(
        torch.arange(1080, device=DEV, dtype=torch.float32), torch.arange(1920, device=DEV, dtype=torch.float32), indexing="ij"
    )
    d = torch.stack([u - cxy[0], v - cxy[1]], -1)
    q = d.clone()
    for _ in range(8):
        q = d / (1 + float(cam[1]) * (q**2).sum(-1, keepdim=True) / R.S["fref"] ** 2)
    r = torch.cat([q / float(cam[0]), torch.ones_like(q[..., :1])], -1) @ torch.tensor(
        cam[2:11].reshape(3, 3), device=DEV, dtype=torch.float32
    )
    C = torch.tensor(cam[11:14], device=DEV, dtype=torch.float32)
    t = -C[2] / r[..., 2]
    x, y = C[0] + t * r[..., 0], C[1] + t * r[..., 1]
    on = (t > 0) & (R.sdf(x, y) < -0.05)
    gx = (x - meta["x0"]) / (meta["res"] * 2) / (ice.shape[-1] - 1) * 2 - 1
    gy = (y - meta["y0"]) / (meta["res"] * 2) / (ice.shape[-2] - 1) * 2 - 1
    warped = (F.grid_sample(ice[None], torch.stack([gx, gy], -1)[None], align_corners=True)[0] * on).cpu().numpy()
    edges = np.any([cv2.Canny((x * 255).astype(np.uint8), 40, 100) > 0 for x in warped], 0)
    edges &= cv2.erode(on.cpu().numpy().astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out = img.copy()
    out[edges] = (255, 0, 255)
    s = torch.arange(0, R.PERIM, 0.05, device=DEV, dtype=torch.float64)
    for h in (0.0, R.WALL - 0.01):
        P = torch.stack(R.out_world(s, torch.full_like(s, h)), -1)
        px, depth = R.src_project(cam.astype(np.float64), P)
        px = px.cpu().numpy() - [R.S["cx"], R.S["cy"]] + cxy
        ok = (depth.cpu().numpy() > 1) & (np.abs(px[:, 0] - 960) < 1100) & (np.abs(px[:, 1] - 540) < 700)
        for run in np.split(np.arange(len(s)), np.flatnonzero(np.diff(ok.astype(int)) != 0) + 1):
            if ok[run[0]] and len(run) > 1:
                cv2.polylines(out, [np.round(px[run] * 4).astype(np.int32)], False, (0, 255, 255), 1, cv2.LINE_AA, 2)
    return out


def video(atlas, out, goal, mp4):
    import subprocess

    meta = json.load(open(f"{atlas}/atlas.json"))["ice"]
    rgb = cv2.imread(f"{atlas}/ice.jpg")
    half = (rgb.shape[1] // 2, rgb.shape[0] // 2)
    planes = [
        cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY),
        3 * (rgb.max(2).astype(np.int16) - rgb.min(2)).clip(0, 85).astype(np.uint8),
    ]  # logos; painted lines
    ice = torch.from_numpy(
        np.stack([cv2.resize(x, half, interpolation=cv2.INTER_AREA) for x in planes]).astype(np.float32) / 255
    ).to(DEV)
    Z = np.load(f"{out}/{goal}.npz")
    vs = frames(f"{A.GOALS}/{goal}/bundle")
    cxy0 = np.array([R.S["cx"], R.S["cy"]])
    enc = subprocess.Popen(
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
            "3840x1080",
            "-r",
            "25",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            mp4,
        ],
        stdin=subprocess.PIPE,
    )
    for i, v in enumerate(vs):
        img = cv2.cvtColor(cv2.imread(v[0]), cv2.COLOR_BGR2RGB)
        enc.stdin.write(
            np.concatenate(
                [
                    overlay(img, Z["cam0"][i], cxy0, ice, meta),
                    overlay(img, Z["cam"][i], Z["cxy"] if "cxy" in Z else cxy0, ice, meta),
                ],
                1,
            ).tobytes()
        )
    enc.stdin.close()
    enc.wait()


def fuse(goals, out, grids, sharp, steady=False):
    """Tracks from the saved registrations, or (steady) from the bundle's calibrated cameras as the anchors."""
    ln = json.load(open(f"{out}/lens.json")) if os.path.exists(f"{out}/lens.json") else None
    if ln:
        R.S["cx"] += ln["dx"]
        R.S["cy"] += ln["dy"]
    for goal in goals:
        b = f"{A.GOALS}/{goal}/bundle"
        vs = frames(b)
        if steady:
            cam0 = np.load(f"{b}/scene.npz")["cam"]
            Z = dict(cam0=cam0, reg=cam0)
        else:
            Z = dict(np.load(f"{out}/{goal}.npz"))
        reg = Z["reg"].astype(np.float64)
        imgs = [
            (
                torch.from_numpy(cv2.imread(v[0], cv2.IMREAD_GRAYSCALE)).to(DEV).half() / 255,
                torch.from_numpy(A.load(v)[1]).to(DEV)[None].half(),
            )
            for v in vs
        ]
        pairs = follow(vs)
        cams = tracks(reg, pairs)
        m = dict(
            goal=goal,
            cameras="calibrated + tracks" if steady else f"{'lens, ' if ln else ''}registered + tracks",
            **measure(cams, imgs, grids, sharp),
            **misfit(cams, pairs),
        )
        with open(f"{out}/measures.jsonl", "a") as f:
            f.write(json.dumps(m) + "\n")
        print(json.dumps(m), flush=True)
        Z["cam"] = cams.astype(np.float32)
        np.savez(f"{out}/{goal}.npz", **Z)
        del imgs
        torch.cuda.empty_cache()


def adopt(out, goals):
    """Writes <out>'s cameras into the bundles (scene.npz cam) and moves the bodies with them. The cameras differ from the bundle's only
    by a rotation about the same centre and a zoom, so each SAM 3D Body mesh is taken back into its camera's frame, its image-plane
    coordinates scaled by the zoom (its projection stays where the model put it), turned into the rink frame with the new camera and
    scaled about the camera centre again to stand on the ice, as body.py does. The bundle's scene.npz and bodies.npz before adoption are
    kept as <out>/<goal>_scene.npz and <out>/<goal>_bodies.npz; a bundle whose cameras are not <out>'s cam0 is left alone."""
    for goal in goals:
        b = f"{A.GOALS}/{goal}/bundle"
        Z = dict(np.load(f"{b}/scene.npz"))
        C = dict(np.load(f"{out}/{goal}.npz"))
        if not np.array_equal(Z["cam"], C["cam0"]):
            print(goal, "skipped: the bundle's cameras are not this run's originals", flush=True)
            continue
        shutil.copy(f"{b}/scene.npz", f"{out}/{goal}_scene.npz")
        shutil.copy(f"{b}/bodies.npz", f"{out}/{goal}_bodies.npz")
        Bd = dict(np.load(f"{b}/bodies.npz"))
        old, new = Z["cam"].astype(np.float64), C["cam"].astype(np.float64)
        verts, joints, scale = Bd["verts"].astype(np.float64), Bd["joints"].astype(np.float64), Bd["scale"].astype(np.float64)
        moved = []
        for k, i in enumerate(Bd["frame"]):
            c0, c1 = old[i], new[i]
            centre = c0[11:14]
            zoom = np.array([c0[0] / c1[0], c0[0] / c1[0], 1.0])
            back = lambda P: (P - centre) / scale[k] @ c0[2:11].reshape(3, 3).T * zoom @ c1[2:11].reshape(3, 3) + centre
            v, j = back(verts[k]), back(joints[k])
            s = centre[2] / (centre[2] - np.percentile(v[:, 2], 0.2))
            verts[k], joints[k], scale[k] = centre + s * (v - centre), centre + s * (j - centre), s
            moved.append(np.linalg.norm(joints[k] - Bd["joints"][k], axis=1).mean())
        Bd.update(verts=verts.astype(np.float16), joints=joints.astype(np.float32), scale=scale.astype(np.float32))
        np.savez_compressed(f"{b}/bodies.npz", **Bd)
        Z["cam"] = C["cam"]
        np.savez(f"{b}/scene.npz", **Z)
        print(
            goal,
            f"adopted; bodies moved {np.median(moved) * 100:.1f} cm median, {np.percentile(moved, 90) * 100:.1f} cm p90",
            flush=True,
        )


def main(cmd, atlas, out, goals):
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    if cmd == "video":
        with torch.no_grad():
            return video(atlas, out, *goals)
    goals = goals or sorted(os.path.basename(os.path.dirname(b)) for b in glob.glob(f"{A.GOALS}/*/bundle"))
    if cmd == "adopt":
        return adopt(out, goals)
    A.C0 = torch.tensor(json.load(open(f"{atlas}/atlas.json"))["camera"], device=DEV, dtype=torch.float32)
    with torch.no_grad():
        grids = [g for g in A.build_grids() if g.name != "stands"]
        sharp = reference(atlas, grids)
        if cmd == "lens":
            lens(goals, out, grids)
        elif cmd in ("fuse", "steady"):
            fuse(goals, out, grids, sharp, steady=cmd == "steady")
        else:
            register(goals, out, grids, sharp)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
