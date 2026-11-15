"""Team kits and player textures for the body mesh (MHR UV layout), baked from the 2025 training windows (the goal clip is held out).

For each window and frame, everyone's fitted mesh is rasterised in the source camera (for occlusion), then each person's mesh in UV
space: every texel is lifted to its point on the posed mesh and projected into the broadcast frame, and its colour is kept where that
point is the visible surface, inside the person's own outline (2 px in) and facing the camera (weight ((cos - 0.25) / 0.75)², as in
render.py). Weighted means are accumulated per kit (team and goalie, referees) and per player (team and jersey number from the feed);
texels never seen are filled from their surroundings (pull-push).

A match without a feed has no training windows: its kits are baked from its goal's own bundle into KITS (a directory; default
models/), which player_textures.py and gear_colour.py read and write too.

  <KITS>/kits.npz       kit_<team>_<goalie>, kit_ref, player_<team>_<jersey>: [T, T, 3] uint8, and <name>_seen [T, T] float16 (weight)
  check/kit_<name>.jpg  the texture next to the textured rest mesh from the front and the back

  <KITS>/kit_votes.npz  <kit>_palette [K, 3] float, <kit>_votes [T, T, K] float16: each kit's distinct colours from its raw
                        samples, and per texel the seen weight of samples of each colour (bake_votes)

usage (LTX venv):  [KITS=dir] kit.py bake [window...]   (default: every train/<id> with bodies)
                       kit.py votes <window>...   the kit votes
                       kit.py check              the check images"""

import glob
import json
import os
import sys

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

DEV = "cuda"
T = 1024
from paths import MODELS

KITS = os.environ.get("KITS", MODELS)
OUT, VOTES = f"{KITS}/kits.npz", f"{KITS}/kit_votes.npz"
KITCOLOURS = 7  # palette size before blends are dropped
UV = np.load(f"{MODELS}/mhr_uv.npz")
TC = torch.from_numpy(UV["texcoords"]).to(DEV)
TF = torch.from_numpy(UV["texcoord_faces"]).to(DEV)
FACES = torch.from_numpy(UV["faces"]).to(DEV)
ctx = dr.RasterizeCudaContext()


def vertex_normals(v):
    a, b, c = v[FACES[:, 0].long()], v[FACES[:, 1].long()], v[FACES[:, 2].long()]
    n = torch.cross(b - a, c - a, dim=1)
    out = (
        torch.zeros_like(v)
        .index_add_(0, FACES[:, 0].long(), n)
        .index_add_(0, FACES[:, 1].long(), n)
        .index_add_(0, FACES[:, 2].long(), n)
    )
    return F.normalize(out, dim=1)


def uv_raster():
    clip = torch.cat([TC * 2 - 1, torch.zeros_like(TC[:, :1]), torch.ones_like(TC[:, :1])], 1)[None].contiguous()
    return dr.rasterize(ctx, clip, TF, (T, T))[0]


def project(c, S, P):
    """Rink points [..., 3] -> source pixels [..., 2] (with the lens distortion) and camera depth."""
    R = torch.tensor(c[2:11].reshape(3, 3), device=DEV, dtype=torch.float32)
    C = torch.tensor(c[11:14], device=DEV, dtype=torch.float32)
    pc = (P - C) @ R.T
    uv = float(c[0]) * pc[..., :2] / pc[..., 2:3]
    uv = uv * (1 + float(c[1]) * (uv**2).sum(-1, keepdim=True) / S["fref"] ** 2)
    return uv + torch.tensor([S["cx"], S["cy"]], device=DEV), pc[..., 2]


def sample(img, px):
    h, w = img.shape[1:]
    g = torch.stack([px[..., 0] / w * 2 - 1, px[..., 1] / h * 2 - 1], -1)
    return F.grid_sample(img[None], g.reshape(1, -1, 1, 2), align_corners=False)[0, :, :, 0].T.reshape(*px.shape[:-1], 3)


def kits_of(t):
    """The textures a person's samples go to: their kit (team and goalie, or referees) and, with a known number, their own."""
    return (
        ["kit_ref"]
        if t["team"] < 0
        else [f"kit_{t['team']}_{int(t['goalie'])}"] + ([f"player_{t['team']}_{t['jersey']}"] if t.get("jersey") else [])
    )


def bake_window(w, uvr, add):
    """Calls add(track, colour [T, T, 3], weight [T, T]) for everyone seen in every frame of window w."""
    S = json.load(open(f"{w}/scene.json"))
    cams = np.load(f"{w}/scene.npz")["cam"].astype(np.float64)
    tracks = {t["key"]: t for t in json.load(open(f"{w}/cast.json"))["tracks"]}
    b = np.load(f"{w}/bodies.npz")
    fr, keys, verts = b["frame"], b["key"], b["verts"]
    cover = uvr[0, ..., 3] > 0
    for i in np.unique(fr):
        rows = [r for r in np.flatnonzero(fr == i) if tracks[int(keys[r])]["mask"][i] >= 0]
        if not rows:
            continue
        c = cams[i]
        C = torch.tensor(c[11:14], device=DEV, dtype=torch.float32)
        N = verts.shape[1]
        V = torch.from_numpy(verts[rows].astype(np.float32)).to(DEV)
        P = V.reshape(-1, 3)
        px, z = project(c, S, P)
        d = (P - C).norm(dim=1)
        clip = torch.stack([(px[:, 0] / 960 - 1) * z, (px[:, 1] / 540 - 1) * z, (d / 100 - 1) * z, z], 1)[None].contiguous()
        faces = torch.cat([FACES + j * N for j in range(len(rows))])
        rast, _ = dr.rasterize(ctx, clip, faces, (1080, 1920))
        zbuf = torch.where(
            rast[0, ..., 3] > 0,
            dr.interpolate(d[None, :, None].contiguous(), rast, faces)[0][0, ..., 0],
            torch.full_like(rast[0, ..., 0], 1e9),
        )
        label = np.zeros((1080, 1920), np.uint8)
        ring = np.ones((5, 5), np.uint8)
        for j, r in enumerate(rows):
            t = tracks[int(keys[r])]
            cv2.fillPoly(label, [np.round(np.array(S["outlines"][i][t["mask"][i]])).astype(np.int32)], j + 1)
        label = torch.from_numpy(np.where(cv2.erode(label, ring) == cv2.dilate(label, ring), label, 0)).to(DEV)
        img = (
            torch.from_numpy(cv2.cvtColor(cv2.imread(f"{w}/frames/{i + 1:03d}.jpg"), cv2.COLOR_BGR2RGB))
            .to(DEV)
            .permute(2, 0, 1)
            .float()
            / 255
        )
        for j, r in enumerate(rows):
            t = tracks[int(keys[r])]
            X = dr.interpolate(V[j][None].contiguous(), uvr, FACES)[0][0]
            nrm = F.normalize(dr.interpolate(vertex_normals(V[j])[None].contiguous(), uvr, FACES)[0][0], dim=-1)
            q, _ = project(c, S, X)
            dist = (X - C).norm(dim=-1)
            qi = q.round().long()
            inside = cover & (qi[..., 0] >= 0) & (qi[..., 0] < 1920) & (qi[..., 1] >= 0) & (qi[..., 1] < 1080)
            qx, qy = qi[..., 0].clamp(0, 1919), qi[..., 1].clamp(0, 1079)
            vis = inside & (dist <= zbuf[qy, qx] + 0.04) & (label[qy, qx] == j + 1)
            cos = -(nrm * (X - C) / dist[..., None]).sum(-1)
            wt = ((cos - 0.25) / 0.75).clamp(0, 1) ** 2 * vis.float()
            if wt.sum() < 1:
                continue
            add(t, sample(img, q), wt)


def pull_push(col, known):
    """Fill texels with no weight from coarser averages of their surroundings."""
    if known.all() or col.shape[0] <= 1:
        return col
    c, k = col * known[..., None], known.float()
    c2 = F.avg_pool2d(c.permute(2, 0, 1)[None], 2)[0].permute(1, 2, 0)
    k2 = F.avg_pool2d(k[None, None], 2)[0, 0]
    coarse = pull_push(c2 / k2.clamp(min=1e-6)[..., None], k2 > 0)
    up = F.interpolate(coarse.permute(2, 0, 1)[None], scale_factor=2, mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    return torch.where(known[..., None], col, up)


def bake(windows):
    uvr = uv_raster()
    acc = {}

    def add(t, col, wt):
        for nm in kits_of(t):
            if nm not in acc:
                acc[nm] = [torch.zeros(T, T, 3, device=DEV), torch.zeros(T, T, device=DEV)]
            acc[nm][0] += col * wt[..., None]
            acc[nm][1] += wt

    for n, w in enumerate(windows):
        with torch.no_grad():
            bake_window(w, uvr, add)
        print(
            f"{n + 1}/{len(windows)} {w} kits {len([a for a in acc if a.startswith('kit')])} players {len([a for a in acc if a.startswith('player')])}",
            flush=True,
        )
    out = {}
    for nm, (s, wt) in acc.items():
        col = pull_push(s / wt.clamp(min=1e-6)[..., None], wt > 0.05)
        out[nm] = (col.clamp(0, 1) * 255).round().byte().cpu().numpy()
        out[f"{nm}_seen"] = wt.half().cpu().numpy()
    np.savez_compressed(OUT, **out)


def palette(c, w, k=KITCOLOURS, iters=30):
    """k colours of samples c [n, 3] (weighted k-means, seeded from luminance quantiles), the blends between two others dropped:
    pixels straddling an edge (white over navy reads grey), not colours of the kit."""
    L = c.sum(1)
    mu = c[torch.argsort(L)[torch.linspace(0, len(c) - 1, k, device=DEV).long()]].clone()
    for _ in range(iters):
        a = torch.cdist(c, mu).argmin(1)
        mu = torch.stack([(c[a == j] * w[a == j, None]).sum(0) / w[a == j].sum() if (a == j).any() else mu[j] for j in range(k)])
    share = torch.stack([w[a == j].sum() for j in range(k)]) / w.sum()
    keep = []
    for j in range(k):
        o = [i for i in range(k) if i != j]
        blend = share[j] < 0.01
        for x in range(len(o)):
            for y in range(x + 1, len(o)):
                p, d = mu[o[x]], mu[o[y]] - mu[o[x]]
                s = ((mu[j] - p) @ d) / (d @ d)
                blend |= bool(0.1 < s < 0.9 and torch.norm(p + s * d - mu[j]) < 0.05)
        if not blend:
            keep.append(j)
    return mu[keep]


def bake_votes(windows):
    """Kit colours that never blend: each kit's palette from its raw broadcast samples, then per texel the seen weight of samples
    nearest each palette colour (a misaligned fit splits a texel's votes instead of averaging navy and white into grey)."""
    uvr = uv_raster()
    g = torch.Generator(device=DEV).manual_seed(0)
    pool = {}

    def collect(t, col, wt):
        nm = kits_of(t)[0]
        m = wt > 0.3
        c, w = col[m], wt[m]
        i = torch.randperm(len(c), device=DEV, generator=g)[:400]
        pool.setdefault(nm, []).append((c[i], w[i]))

    for w in windows:
        with torch.no_grad():
            bake_window(w, uvr, collect)
    pal = {nm: palette(torch.cat([c for c, _ in v]), torch.cat([w for _, w in v])) for nm, v in pool.items()}
    for nm, mu in pal.items():
        print(nm, (mu * 255).round().int().tolist(), flush=True)
    votes = {nm: torch.zeros(T, T, len(mu), device=DEV) for nm, mu in pal.items()}

    def vote(t, col, wt):
        nm = kits_of(t)[0]
        lab = torch.cdist(col.reshape(-1, 3), pal[nm]).argmin(1).reshape(T, T)
        votes[nm].scatter_add_(2, lab[..., None], wt[..., None])

    for n, w in enumerate(windows):
        with torch.no_grad():
            bake_window(w, uvr, vote)
        print(f"{n + 1}/{len(windows)} {w}", flush=True)
    np.savez_compressed(
        VOTES,
        **{f"{nm}_palette": pal[nm].cpu().numpy() for nm in pal},
        **{f"{nm}_votes": v.half().cpu().numpy() for nm, v in votes.items()},
    )


def check():
    """Each texture, its coverage, and the rest mesh textured with it from the front and the back."""
    K = np.load(OUT)
    rest = torch.from_numpy(UV["rest"]).to(DEV) / 100
    rest = rest - torch.tensor([0, 0.86, 0], device=DEV)
    os.makedirs("check", exist_ok=True)
    for nm in [k for k in K.files if not k.endswith("_seen")]:
        tex = torch.from_numpy(K[nm]).to(DEV).float()[None] / 255
        views = []
        for s in (1, -1):  # front, back (turned 180 degrees about the vertical)
            P = rest * torch.tensor([s, -1, -s], device=DEV)
            clip = torch.stack([P[:, 0], P[:, 1], P[:, 2] * 0.5, torch.ones_like(P[:, 0])], 1)[None].contiguous()
            rast, _ = dr.rasterize(ctx, clip, FACES, (T, T))
            uv, _ = dr.interpolate(TC[None].contiguous(), rast, TF)
            img = torch.where(
                rast[..., 3:] > 0,
                dr.texture(tex.contiguous(), uv, filter_mode="linear"),
                torch.ones(1, T, T, 3, device=DEV) * 0.5,
            )
            views.append((img[0].cpu().numpy() * 255).astype(np.uint8))
        seen = np.clip(K[f"{nm}_seen"].astype(np.float32) / 20, 0, 1)
        row = np.hstack([K[nm], (np.stack([seen] * 3, -1) * 255).astype(np.uint8)] + views)
        cv2.imwrite(
            f"check/kit_{nm}.jpg",
            cv2.cvtColor(
                cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR
            ),
        )
    print("\n".join(f"{k}: seen texels {(K[k + '_seen'] > 0.05).mean():.2f}" for k in K.files if not k.endswith("_seen")))


if __name__ == "__main__":
    if sys.argv[1] == "bake":
        bake(sys.argv[2:] or sorted(os.path.dirname(p) for p in glob.glob("train/*/bodies.npz")))
    elif sys.argv[1] == "votes":
        bake_votes(sys.argv[2:])
    elif sys.argv[1] == "check":
        check()
