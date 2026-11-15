"""Cloth wrinkle maps for the viewer's bodies (app/public/replay/people.ts), in the MHR texture layout, the same for everyone.

Each garment's folds are a height field over the MHR rest mesh (m, A pose), built from where folds form on hockey kit:
  jersey  relaxed: pipe folds hanging from the shoulders and armpits to the hem, wider and fading toward it, a ripple and a lip at
          the hem, long twisting folds down the sleeves, gathered rings at the cuffs; compressed: diamond buckles round the sleeves
          (Yoshimura's pattern of a squashed tube) and wavy bands across the body
  pants   relaxed: broad vertical creases on the padded shell and a quilting seam; compressed: bands round hips and thighs
  socks   relaxed: a ripple under the pants and bunching above the skates; compressed: fine rings
plus a faint crinkle everywhere. Height is defined in 3D on the mesh, so both sides of a seam in the layout get the same folds.

Per texel the surface gradient of the height (autograd, in the tangent plane) is written as its derivatives along the layout's u and
v (m per unit of u and v: the layout's own Jacobian per face), which the viewer turns back into a surface gradient on the posed body
from screen derivatives of the texture coordinates, so the folds stretch and squash with the body and need no tangents. Islands are
grown into the gaps so filtering never pulls in empty texels.

  <out>/folds.png     relaxed: R, G du and dv of the height (0.5 + d / (2 K)), B the fold's shadow (0 open, 1 deepest valley)
  <out>/squash.png    compressed, as folds.png
  <out>/garment.bin   uint8 per texture-coordinate vertex: 0 skin, 1 jersey, 2 pants, 3 socks
  <out>/joints.json   where the body bends, for how squashed the cloth is (people.ts bodyFrame): rings, lists of MHR vertices round a
                      limb or the body (their mean is a point on its centre line), and joints, ring triples [toward the body, at the
                      joint, away], whose angle is the bend
  <out>/joints.bin    uint8 [V, 2] per MHR vertex: 1 + the joint whose bend squashes the cloth there (0: none), and how much (255 full)
  <out>/preview.jpg   the rest mesh lit by each set, front, side and back (relaxed above, compressed below)

usage (LTX venv):  cloth_maps.py <out>"""

import math
import os
import sys

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from PIL import Image

DEV = "cuda"
T = 1024
K = 0.8  # m of height per unit of u or v at full scale of the encoding
from paths import MODELS

UV = np.load(f"{MODELS}/mhr_uv.npz")
HEM, SOCK, CROTCH, ARMPIT = 0.88, 0.57, 0.82, 0.24  # rest mesh (m): jersey hem, top of the socks, legs below, arms beyond |x|
WRIST = 0.51  # |x| where the hand begins
SHADOW = 0.004  # m of valley depth that is fully shadowed
GROW = 24  # texels the islands are grown by
CHUNK = 1 << 16  # texels evaluated at once

rest = torch.from_numpy(UV["rest"] / 100).float().to(DEV)
FACES = torch.from_numpy(UV["faces"]).int().to(DEV)
TC = torch.from_numpy(UV["texcoords"]).float().to(DEV)
TF = torch.from_numpy(UV["texcoord_faces"]).int().to(DEV)
ctx = dr.RasterizeCudaContext()


def smooth(e0, e1, x):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


_PERM = torch.randperm(256, generator=torch.Generator().manual_seed(1)).to(DEV).repeat(2)
_GRAD = torch.tensor(
    [
        [1, 1, 0],
        [-1, 1, 0],
        [1, -1, 0],
        [-1, -1, 0],
        [1, 0, 1],
        [-1, 0, 1],
        [1, 0, -1],
        [-1, 0, -1],
        [0, 1, 1],
        [0, -1, 1],
        [0, 1, -1],
        [0, -1, -1],
    ],
    dtype=torch.float32,
    device=DEV,
)


def perlin(p):
    """Perlin's gradient noise (about -0.7..0.7) over points p [..., 3], features about 1 across."""
    i = p.floor()
    f = p - i
    i = i.long() & 255
    u = f * f * f * (f * (f * 6 - 15) + 10)

    def corner(dx, dy, dz):
        h = _PERM[_PERM[_PERM[(i[..., 0] + dx) & 255] + ((i[..., 1] + dy) & 255)] + ((i[..., 2] + dz) & 255)] % 12
        return (_GRAD[h] * (f - torch.tensor([dx, dy, dz], dtype=torch.float32, device=DEV))).sum(-1)

    lx = lambda a, b: a + (b - a) * u[..., 0]
    y0 = (
        lx(corner(0, 0, 0), corner(1, 0, 0))
        + (lx(corner(0, 1, 0), corner(1, 1, 0)) - lx(corner(0, 0, 0), corner(1, 0, 0))) * u[..., 1]
    )
    y1 = (
        lx(corner(0, 0, 1), corner(1, 0, 1))
        + (lx(corner(0, 1, 1), corner(1, 1, 1)) - lx(corner(0, 0, 1), corner(1, 0, 1))) * u[..., 1]
    )
    return y0 + (y1 - y0) * u[..., 2]


def noise(p, scale, seed):
    """Perlin noise with features about `scale` m across, a different field per seed."""
    o = torch.tensor([37.1, 11.7, 23.3], device=DEV) * seed
    return perlin(p / scale + o)


def slices(sel, along, lo, hi, step=0.01):
    """Centroids of the rest vertices `sel` in slices of coordinate `along` (a function of points) from lo to hi."""
    c = along(sel)
    ts = torch.arange(lo, hi + 1e-6, step, device=DEV)
    out = []
    for t in ts:
        m = (c - t).abs() < step
        out.append(sel[m].mean(0))
    return ts, torch.stack(out)


def interp(ts, ys, t):
    """Piecewise linear ys [n, ...] at t (clamped), differentiable in t."""
    x = ((t - ts[0]) / (ts[1] - ts[0])).clamp(0, len(ts) - 1 - 1e-4)
    i = x.floor().long()
    f = (x - i)[..., None]
    return ys[i] * (1 - f) + ys[i + 1] * f


class Tube:
    """A limb as a tube: a centre line through slice centroids, each point's distance along it (a, m), its angle round it and the
    arc length round it (s, m, at the tube's mean radius there)."""

    def __init__(self, sel, along, lo, hi, ref):
        self.along, self.ref = along, ref
        self.ts, cen = slices(sel, along, lo, hi)
        k = torch.exp(-0.5 * (torch.arange(-9, 10, device=DEV) / 3.0) ** 2)
        k = k / k.sum()  # centroids smoothed over ~5 cm
        cen = F.conv1d(F.pad(cen.T[:, None], (9, 9), mode="replicate"), k[None, None])[:, 0].T
        seg = (cen[1:] - cen[:-1]).norm(dim=1)
        self.arc = torch.cat([torch.zeros(1, device=DEV), seg.cumsum(0)])
        tan = torch.cat([cen[1:2] - cen[:1], cen[2:] - cen[:-2], cen[-1:] - cen[-2:-1]])
        self.cen, self.tan = cen, F.normalize(tan, dim=1)
        r = []
        for k, t in enumerate(self.ts):
            m = (along(sel) - t).abs() < 0.01
            d = sel[m] - cen[k]
            d = d - (d @ self.tan[k])[:, None] * self.tan[k]
            r.append(d.norm(dim=1).mean())
        self.radius = torch.stack(r)

    def axis(self, p):
        """Each point's distance along the tube (m) and the tube's direction there."""
        t = self.along(p)
        return interp(self.ts, self.arc[:, None], t)[..., 0], F.normalize(interp(self.ts, self.tan, t), dim=-1)

    def coords(self, p):
        t = self.along(p)
        c, tn, rad = (
            interp(self.ts, self.cen, t),
            F.normalize(interp(self.ts, self.tan, t), dim=-1),
            interp(self.ts, self.radius[:, None], t)[..., 0],
        )
        d = p - c
        d = d - (d * tn).sum(-1, keepdim=True) * tn
        e1 = F.normalize(self.ref - (self.ref * tn).sum(-1, keepdim=True) * tn, dim=-1)
        e2 = torch.cross(tn, e1, dim=-1)
        ang = torch.atan2((d * e2).sum(-1), (d * e1).sum(-1))
        a = interp(self.ts, self.arc[:, None], t)[..., 0]
        return a, ang, ang * rad


def side(p):
    """Points with x made positive (the body is symmetric; folds mirror)."""
    return torch.stack([p[..., 0].abs(), p[..., 1], p[..., 2]], -1)


R = side(rest)
FWD = torch.tensor([0.0, 0.0, 1.0], device=DEV)
UP = torch.tensor([0.0, 1.0, 0.0], device=DEV)
ARM = Tube(R[(R[:, 0] > ARMPIT) & (R[:, 1] > 0.95)], lambda p: p[..., 0], ARMPIT + 0.01, WRIST + 0.06, UP)
LEG = Tube(R[R[:, 1] < CROTCH + 0.04], lambda p: p[..., 1], 0.04, CROTCH + 0.02, FWD)
TORSO = Tube(rest[(rest[:, 0].abs() < 0.17) & (rest[:, 1] > CROTCH - 0.02)], lambda p: p[..., 1], CROTCH - 0.02, 1.47, FWD)
A_CUFF = float(interp(ARM.ts, ARM.arc[:, None], torch.tensor(WRIST, device=DEV))[0])
TOP = float(rest[:, 1].max())


def parts(p):
    """Soft weights of the regions at points p: arms, legs, torso (each 0..1), and the garment ids' weights."""
    q = side(p)
    x, y = q[..., 0], q[..., 1]
    arm = smooth(ARMPIT - 0.01, ARMPIT + 0.03, x) * smooth(0.93, 1.0, y)
    leg = (1 - smooth(CROTCH - 0.01, CROTCH + 0.02, y)) * (1 - arm)
    torso = (1 - arm) * (1 - leg)
    head = smooth(TOP - 0.27, TOP - 0.24, y)
    hand = smooth(WRIST - 0.01, WRIST + 0.02, x) * arm
    cloth = (1 - head) * (1 - hand)
    jersey = (arm + torso * smooth(HEM - 0.004, HEM + 0.004, y)) * cloth
    socks = leg * (1 - smooth(SOCK - 0.004, SOCK + 0.004, y)) * cloth
    pants = (cloth - jersey - socks).clamp(min=0)
    return arm, leg, torso, jersey, pants, socks


def tube_space(tube, q, stretch):
    """Points q in a tube's own space: their direction round its centre line on a circle of its median radius (so a tube that swells
    does not cut through the folds), and the distance along it divided by `stretch` (above 1: features drawn out along the tube,
    below 1: squeezed along it)."""
    t = tube.along(q)
    c, tn = interp(tube.ts, tube.cen, t), F.normalize(interp(tube.ts, tube.tan, t), dim=-1)
    a = interp(tube.ts, tube.arc[:, None], t)
    off = q - c
    off = off - (off * tn).sum(-1, keepdim=True) * tn
    e1 = F.normalize(tube.ref - (tube.ref * tn).sum(-1, keepdim=True) * tn, dim=-1)
    e2 = torch.cross(tn, e1, dim=-1)
    ang = torch.atan2((off * e2).sum(-1, keepdim=True), (off * e1).sum(-1, keepdim=True))
    r0 = tube.radius.median()
    return torch.cat([r0 * torch.cos(ang), r0 * torch.sin(ang), a / stretch], -1)


def folds(space, width, seed):
    """Folds in a tube's space (tube_space): noise about `width` m between crests, with a finer octave that splits and joins them."""
    return noise(space, width, seed) + 0.3 * noise(space, width / 2.1, seed + 50)


def pipe(n):
    """Hanging folds' cross-section from noise n: crests and troughs alike rounded, a little flattened between."""
    return torch.sign(n) * n.abs().clamp(min=1e-6) ** 0.8


def buckle(n):
    """Squashed folds' cross-section from noise n: sharp valleys where n crosses zero, broad rounded crests, about zero mean."""
    return (2.6 * n.abs()).clamp(max=1) - 0.55


def fields(p):
    """Relaxed and compressed heights (m) at rest-mesh points p [..., 3]."""
    arm, leg, torso, jersey, pants, socks = parts(p)
    q = side(p)
    y = q[..., 1]
    left = torch.where(p[..., :1] < 0, 1.0, 0.0) * torch.tensor([0.0, 0.0, 7.3], device=DEV)  # the left side folds its own way
    sp = lambda tube, stretch: tube_space(tube, p, stretch) if tube is TORSO else tube_space(tube, q, stretch) + left
    patch = lambda sc, sd: 0.3 + 0.7 * smooth(-0.25, 0.35, noise(q + left, sc, sd))  # folds come and go in patches
    # each region's folds in its own space: hanging folds drawn out along it (5x), squashed ones squeezed along it k times (bands
    # round it, width / k apart along it)
    hang = lambda w, sd: (
        arm * pipe(folds(sp(ARM, 5.0), w[0], sd))
        + leg * pipe(folds(sp(LEG, 4.0), w[1], sd))
        + torso * pipe(folds(sp(TORSO, 5.0), w[2], sd))
    )
    squash = lambda w, k, sd: (
        arm * buckle(folds(sp(ARM, 1 / k), w[0], sd))
        + leg * buckle(folds(sp(LEG, 1 / k), w[1], sd))
        + torso * buckle(folds(sp(TORSO, 1 / k), w[2], sd))
    )
    aa = ARM.axis(q)[0]

    # jersey: hanging folds, deeper toward the hem, a lip at the hem, gathered at the cuffs; squashed, bands round body and sleeves
    sag = 1 + 0.8 * (1 - smooth(HEM, 1.2, y)) * torso
    jr = (torso * 0.0035 + arm * 0.004) * sag * hang((0.03, 0.05, 0.04), 1) * patch(0.22, 2)
    jr = jr + torso * (0.0015 * smooth(HEM - 0.003, HEM + 0.003, y) - 0.0015)
    jr = jr + arm * 0.002 * buckle(folds(sp(ARM, 1 / 3), 0.06, 3)) * smooth(A_CUFF - 0.11, A_CUFF - 0.04, aa)
    jc = (torso * 0.003 + arm * 0.0035) * squash((0.1, 0.15, 0.15), 2.8, 4) * patch(0.18, 5)

    # pants: broad creases on the padded shell and a quilting seam; squashed, bands at the hips and over the thighs
    pr = 0.003 * hang((0.04, 0.04, 0.04), 6) * patch(0.25, 7) - 0.0012 * torch.exp(
        -0.5 * ((y - 0.71 - 0.003 * noise(q, 0.2, 8)) / 0.003) ** 2
    )
    pc = 0.0035 * squash((0.15, 0.15, 0.15), 2.5, 9) * patch(0.2, 10)

    # socks: knit over the shin pads, bunched above the skates and under the pants
    near = (1 - smooth(0.14, 0.2, y)) + smooth(SOCK - 0.06, SOCK - 0.01, y)
    sr = 0.0012 * squash((0.075, 0.075, 0.075), 3.0, 11) * near
    sc = 0.0018 * squash((0.09, 0.09, 0.09), 3.0, 12)

    crinkle = 0.00012 * noise(q + left, 0.015, 13)
    cloth = jersey + pants + socks
    relaxed = jersey * jr + pants * pr + socks * sr + cloth * crinkle
    compressed = jersey * jc + pants * pc + socks * sc + cloth * crinkle
    return relaxed, compressed


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


def grow(img, valid, n):
    """img [H, W, C] with texels outside `valid` filled from the nearest valid ones, n texels out."""
    img = img * valid[..., None]
    w = valid.float()
    x, m = img.permute(2, 0, 1)[None], w[None, None]
    for _ in range(n):
        s = F.avg_pool2d(x * m, 3, 1, 1)
        c = F.avg_pool2d(m, 3, 1, 1)
        fill = (m == 0) & (c > 0)
        x = torch.where(fill, s / c.clamp(min=1e-6), x)
        m = torch.where(fill, torch.ones_like(m), m)
    return x[0].permute(1, 2, 0)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    clip = torch.cat([TC * 2 - 1, torch.zeros_like(TC[:, :1]), torch.ones_like(TC[:, :1])], 1)[None].contiguous()
    rast = dr.rasterize(ctx, clip, TF, (T, T))[0]
    cover = rast[0, ..., 3] > 0
    tri = (rast[0, ..., 3] - 1).long().clamp(min=0)
    X = dr.interpolate(rest[None].contiguous(), rast, FACES)[0][0]
    N = F.normalize(dr.interpolate(vertex_normals(rest)[None].contiguous(), rast, FACES)[0][0], dim=-1)
    # the layout's Jacobian per face: [dX/du, dX/dv] (m per unit)
    P, Q = rest[FACES.long()], TC[TF.long()]
    E = torch.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], -1)  # [F, 3, 2]
    D = torch.stack([Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0]], -1)  # [F, 2, 2]
    J = E @ torch.linalg.inv(D + 1e-12 * torch.eye(2, device=DEV))  # [F, 3, 2]
    Ju, Jv = J[tri][..., 0], J[tri][..., 1]
    xs = X[cover].detach()
    n = N[cover]
    sets = []
    for k in range(2):
        hs, gs = [], []
        for x in xs.split(CHUNK):  # the graph of a chunk at a time (the GPU is shared)
            x = x.clone().requires_grad_(True)
            h = fields(x)[k]
            gs.append(torch.autograd.grad(h.sum(), x)[0])
            hs.append(h.detach())
        h, g = torch.cat(hs), torch.cat(gs)
        g = g - (g * n).sum(-1, keepdim=True) * n
        hu, hv = (g * Ju[cover]).sum(-1), (g * Jv[cover]).sum(-1)
        shade = (-h.detach() / SHADOW).clamp(0, 1)
        print(
            ["relaxed", "compressed"][k],
            "height mm p1/p99",
            *(torch.quantile(h.detach()[::7], torch.tensor([0.01, 0.99], device=DEV)) * 1000).tolist(),
            "du,dv / K p99.5",
            float(torch.quantile(torch.cat([hu, hv]).abs()[::7], 0.995) / K),
            "clipped",
            float((torch.cat([hu, hv]).abs() > K).float().mean()),
        )
        img = torch.zeros(T, T, 3, device=DEV)
        img[cover] = torch.stack([0.5 + (hu / (2 * K)).clamp(-0.5, 0.5), 0.5 + (hv / (2 * K)).clamp(-0.5, 0.5), shade], -1)
        img = grow(img, cover, GROW)
        img[~cover & (img.sum(-1) == 0)] = torch.tensor([0.5, 0.5, 0.0], device=DEV)
        Image.fromarray((img.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()).save(
            f"{out}/{['folds', 'squash'][k]}.png", optimize=True
        )
        sets.append(img)
    # garment per texture-coordinate vertex
    vert = torch.zeros(len(TC), dtype=torch.long, device=DEV)
    vert[TF.long().flatten()] = FACES.long().flatten()
    _, _, _, jersey, pants, socks = parts(rest[vert])
    w = torch.stack([1 - jersey - pants - socks, jersey, pants, socks], -1)
    w.argmax(-1).byte().cpu().numpy().tofile(f"{out}/garment.bin")
    bends(out)
    density = (Ju[cover].norm(dim=-1) + Jv[cover].norm(dim=-1)) / 2
    print("m per uv unit median", float(density.median()))
    preview(out, sets)


def bends(out):
    """The joints whose bending squashes the cloth: elbows, knees, hips (thigh against body) and the waist (chest against pelvis),
    each as three rings of the rest mesh, and per vertex the joint it is near and how near (1 within JOINT_IN m of the joint's
    ring along the limb or body, 0 past JOINT_OUT)."""
    import json

    q = side(rest)
    y = q[:, 1]
    arm, leg, torso, *_ = parts(rest)
    aa = ARM.axis(q)[0]
    sgn = torch.where(rest[:, 0] < 0, -1, 1)
    elbow = 0.5 * A_CUFF

    def ring(m, at, near=0.008):
        idx = torch.nonzero(m & ((at[0] - at[1]).abs() < near)).flatten()
        return idx.tolist()

    rings, joints, region = [], [], []

    def add(r):
        rings.append(r)
        return len(rings) - 1

    zone = lambda d, a, b: 1 - smooth(a, b, d)
    for s in (1, -1):
        mine = sgn == s
        A, L = mine & (arm > 0.99), mine & (leg > 0.99)
        sh, el, wr = add(ring(A, (aa, 0.03))), add(ring(A, (aa, elbow))), add(ring(A, (aa, A_CUFF - 0.01)))
        th, kn, an, top = add(ring(L, (y, 0.70))), add(ring(L, (y, 0.49))), add(ring(L, (y, 0.14))), add(ring(L, (y, 0.78)))
        joints += [[sh, el, wr], [th, kn, an]]
        region += [mine * arm * zone((aa - elbow).abs(), 0.05, 0.13), mine * leg * zone((y - 0.49).abs(), 0.06, 0.14)]
        joints.append([None, top, kn])
        region.append(mine * zone((y - 0.82).abs(), 0.06, 0.15) * (1 - arm))
    T = torso > 0.99
    tor = lambda h: ring(T & (rest[:, 0].abs() < 0.16), (y, h))
    add(tor(1.42))  # neck
    ch, wa, pe = add(tor(1.25)), add(tor(1.08)), add(tor(0.93))
    for j in joints:
        if j[0] is None:
            j[0] = ch  # a hip: the thigh against the chest
    joints.append([ch, wa, pe])
    region.append(torso * zone((y - 1.08).abs(), 0.07, 0.17) * (1 - arm))
    W = torch.stack(region, 1)
    best = W.argmax(1)
    w = W.gather(1, best[:, None])[:, 0]
    tab = torch.stack([torch.where(w > 0.01, best + 1, 0), (w * 255).round()], 1).byte()
    tab.cpu().numpy().tofile(f"{out}/joints.bin")
    json.dump(dict(rings=rings, joints=joints), open(f"{out}/joints.json", "w"), separators=(",", ":"))
    print("rings", [len(r) for r in rings], "joints", joints, "vertices in a joint's zone", int((w > 0.01).sum()))


def preview(out, sets):
    """The rest mesh from the front, the side and the back, lit: each set from its true 3D gradient (top rows), then from its maps
    decoded as the viewer does (bottom rows), so the two can be compared."""
    S = 560
    mean = rest.mean(0)
    vtx = rest - mean
    rows = [[], [], [], []]
    light = F.normalize(torch.tensor([0.3, 0.6, 0.75], device=DEV), dim=0)
    for yaw in (0.0, math.pi / 2, math.pi):
        c, s = math.cos(yaw), math.sin(yaw)
        rot = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=DEV)
        v = vtx @ rot.T
        clipv = torch.cat([v[:, :1] / 0.95, v[:, 1:2] / 0.95, -v[:, 2:3] / 2 + 0.5, torch.ones_like(v[:, :1])], 1)[
            None
        ].contiguous()
        r, _ = dr.rasterize(ctx, clipv, FACES, (S, S))
        on = r[0, ..., 3] > 0
        uv = dr.interpolate(TC[None].contiguous(), r, TF)[0][0]
        Pp = dr.interpolate(v[None].contiguous(), r, FACES)[0][0]
        Nn = F.normalize(dr.interpolate(vertex_normals(v)[None].contiguous(), r, FACES)[0][0], dim=-1)
        x0 = (Pp[on] @ rot + mean).detach()
        n0 = Nn[on]
        for k in range(2):
            gs, hs = [], []
            for x in x0.split(CHUNK):
                x = x.clone().requires_grad_(True)
                h = fields(x)[k]
                gs.append(torch.autograd.grad(h.sum(), x)[0] @ rot.T)
                hs.append(h.detach())
            g, h = torch.cat(gs), torch.cat(hs)
            g = g - (g * n0).sum(-1, keepdim=True) * n0
            img = torch.zeros(S, S, device=DEV)
            img[on] = (0.25 + 0.75 * (F.normalize(n0 - g, dim=-1) @ light).clamp(min=0)) * (1 - 0.35 * (-h / SHADOW).clamp(0, 1))
            rows[k].append(img.flip(0))
        # decoded: the surface gradient from the derivatives along u and v and the screen derivatives of u, v and the position
        H = S - 1
        dux, duy = (uv[:, 1:] - uv[:, :-1])[:H, :H], (uv[1:] - uv[:-1])[:H, :H]
        sx, sy, n = (Pp[:, 1:] - Pp[:, :-1])[:H, :H], (Pp[1:] - Pp[:-1])[:H, :H], Nn[:H, :H]
        same = on[:H, :H] & (r[0, :H, 1:, 3] == r[0, :H, :H, 3]) & (r[0, 1:, :H, 3] == r[0, :H, :H, 3])
        r1, r2 = torch.cross(sy, n, dim=-1), torch.cross(n, sx, dim=-1)
        det = (sx * r1).sum(-1, keepdim=True)
        for k, tex in enumerate(sets):
            t = dr.texture(tex[None].contiguous(), uv[None].contiguous(), filter_mode="linear")[0][:H, :H]
            gu, gv = (t[..., 0] - 0.5) * 2 * K, (t[..., 1] - 0.5) * 2 * K
            hx, hy = gu * dux[..., 0] + gv * dux[..., 1], gu * duy[..., 0] + gv * duy[..., 1]
            b = det.abs() * n - torch.sign(det) * (hx[..., None] * r1 + hy[..., None] * r2)
            nb = torch.where(same[..., None] & (det.abs() > 1e-14), F.normalize(b, dim=-1), n)
            img = torch.zeros(S, S, device=DEV)
            img[:H, :H] = torch.where(
                on[:H, :H], (0.25 + 0.75 * (nb @ light).clamp(min=0)) * (1 - 0.35 * t[..., 2]), torch.zeros_like(t[..., 2])
            )
            rows[2 + k].append(img.flip(0))
    grid = torch.cat([torch.cat(r, 1) for r in rows], 0)
    Image.fromarray((grid.clamp(0, 1) * 255).byte().cpu().numpy()).save(f"{out}/preview.jpg", quality=90)


if __name__ == "__main__":
    main()
