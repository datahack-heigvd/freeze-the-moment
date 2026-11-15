"""One body texture per person of an exported goal (MHR UV layout), for the viewer. Players, goalies too (a goalie wears the
team's jersey): their team's kit design (kit_design: the skaters' kit colours from raw broadcast pixels, laid out by where the
broadcast showed each, left and right alike, cleared of what no garment has, every band edge drawn hard and smooth; with the
team's logo (team_logos.py) as its crest where the broadcast shows one on the chest: crest_place), with their own head from their
baked texture. Officials, whose narrow stripes the design cannot hold: their baked broadcast texture (kit.py: kit_ref). The
helmet (a goalie's mask down over the face) and a goalie's leg pads and gloves are each one colour, as they look in the footage
(gear_colour.py: <KITS>/gear_colours.json), and the roster's surname and number are drawn on the back, light on the kit's dark
bands and dark on its light ones. Numbers and names come only from the feed (cast.json); a person whose number is unknown gets
a plain back.

Where the back is: on the most upright fitted body of the goal, each texel's offset from the shoulder midpoint across the back
(as seen from behind) and along the torso axis, and whether it faces backwards. A team without a kit design (no votes) keeps its
cleaned bake, its blurred lettering painted over, the number in the team's contrast colour: the mean of the darkest 1 % of its
player textures' back texels on a light jersey, of the brightest 1 % on a dark one.

  <scene>/texture_<key>_2k.jpg [T, T] RGB, row 0 at v = 0 (the viewer's top quality setting)
  <scene>/texture_<key>.jpg    [KIT, KIT], the same averaged down (the viewer's default: a quarter of the memory)
  <KITS>/teams.json            optional, the teams' names and logos (team_logos.py)

usage (LTX venv, in a goal dir):  player_textures.py <scene_dir>
                                       player_textures.py check <scene_dir> <sheet.jpg> <key>...   old vertex colours | new texture,
                                                                                                  front and back of the rest mesh"""

import json
import os
import sys

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from kits.gear import covered, gloved, padded
from kits.kit_vote_model import fill_vote_bins
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

DEV = "cuda"
T, KIT, SS = 2048, 1024, 2  # texture size, the kits' (kit.py), the kit design's supersampling
CHUNK = 1 << 21  # texels evaluated at once
from paths import MODELS, PIPELINE

KITS = os.environ.get("KITS", MODELS)  # the match's kits (kit.py)
UV = np.load(f"{MODELS}/mhr_uv.npz")
TC = torch.from_numpy(UV["texcoords"]).to(DEV)
TF = torch.from_numpy(UV["texcoord_faces"]).to(DEV)
FACES = torch.from_numpy(UV["faces"]).to(DEV)
FONT = f"{PIPELINE}/fonts/Oswald.ttf"
NUMBER = (-0.115, -0.355, 0.32)  # back number: top, bottom below the shoulder line, widest (m)
NAME = (-0.052, -0.098, 0.34)  # surname: top, bottom, widest (m)
CLEAR = (0.19, -0.03, -0.42)  # area cleared to the jersey colour: half-width, top, bottom (m)
HEAD = 0.21  # rest mesh: the head (crown to chin, m); the neck below is the torso's
ARMPIT, CROTCH = 0.24, 0.82  # rest mesh: arms beyond this |x|, legs below this height (m)
AXIAL, ANGLES = 0.01, 24  # kit design bins: 1 cm along each part, 15 degrees around it
THIN, ISLAND = 3, 12  # kit design: the narrowest region round a part (bins), the smallest piece
SMOOTH = (1.5, 0.8)  # kit design: the bins' shares smoothed by these (bins: along, around)
VOTE = (1, 1)  # each bin's vote pooled over +-VOTE bins (along, around)
RING = 0.75  # kit design: how much a sleeve's or leg's colours at a height are those all round it (kit_design)
DESIGNED = ("kit_0_0", "kit_1_0")  # kits drawn as designs (kit_design): the skaters'. A goalie wears their
# team's (the same jersey; their own votes are mostly their gear's), the
# officials their bake (their narrow upright stripes are finer than ANGLES)
CREST_AREA = ((-0.25, 0.25), (-0.08, -0.6))  # the chest, below the collar, searched for the crest: across, up (m)
CREST_PX, CREST_DIFF, CREST_MID = (
    250,
    0.2,
    0.05,
)  # its resolution (px per m), the colour change a crest makes, how central it is (m)
CREST_SIZE = (0.08, 0.35)  # the crest's sides (m)
PX = 6000  # lettering drawing resolution (px per m), averaged down to the texels
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


def torso_field(n=T):
    """Per texel: across (m, positive to the right seen from behind), up (m, along the torso from the shoulder midpoint), and
    on-back and on-front flags, from the goal's most upright fitted body."""
    b = np.load("bundle/bodies.npz")
    j = b["joints"].astype(np.float32)
    axis = (j[:, 5] + j[:, 6]) / 2 - (j[:, 9] + j[:, 10]) / 2
    up = axis / np.linalg.norm(axis, axis=1, keepdims=True)
    k = int(up[:, 2].argmax())
    v = torch.from_numpy(b["verts"][k].astype(np.float32)).to(DEV)
    jt = torch.from_numpy(j[k]).to(DEV)
    sh = (jt[5] + jt[6]) / 2
    u = torch.from_numpy(up[k]).to(DEV)
    left = jt[5] - jt[6]
    left = F.normalize(left - (left @ u) * u, dim=0)
    fwd = torch.linalg.cross(left, u)
    clip = torch.cat([TC * 2 - 1, torch.zeros_like(TC[:, :1]), torch.ones_like(TC[:, :1])], 1)[None].contiguous()
    rast = dr.rasterize(ctx, clip, TF, (n, n))[0]
    X = dr.interpolate(v[None].contiguous(), rast, FACES)[0][0]
    N = F.normalize(dr.interpolate(vertex_normals(v)[None].contiguous(), rast, FACES)[0][0], dim=-1)
    d = X - sh
    cover = rast[0, ..., 3] > 0
    return d @ -left, d @ u, cover & (N @ fwd < -0.3) & (d @ fwd < 0), cover & (N @ fwd > 0.3) & (d @ fwd > 0)


def gear_fields(n=T):
    """Per texel, from the MHR rest mesh (gear.py): on the helmet, on a goalie's mask, leg pads, gloves."""
    rest = UV["rest"] / 100
    clip = torch.cat([TC * 2 - 1, torch.zeros_like(TC[:, :1]), torch.ones_like(TC[:, :1])], 1)[None].contiguous()
    rast = dr.rasterize(ctx, clip, TF, (n, n))[0]
    X = dr.interpolate(torch.from_numpy(rest).float().to(DEV)[None].contiguous(), rast, FACES)[0][0].cpu().numpy()
    on = lambda m: torch.from_numpy(m).to(DEV) & (rast[0, ..., 3] > 0)
    return on(covered(rest, X, 0.01) > 0.5), on(covered(rest, X, 0.01, True) > 0.5), on(padded(X)), on(gloved(X, rest))


def part_field(n=T):
    """Per texel of the rest mesh (T pose): body part (0 torso, 1 arms, 2 legs, 3 head; -1 off the mesh), the coordinate along the
    part's axis (m: up the torso and legs, out along the arms) and the angle around it, with the shoulder-line height."""
    rest = torch.from_numpy(UV["rest"]).to(DEV) / 100
    clip = torch.cat([TC * 2 - 1, torch.zeros_like(TC[:, :1]), torch.ones_like(TC[:, :1])], 1)[None].contiguous()
    rast = dr.rasterize(ctx, clip, TF, (n, n))[0]
    X = dr.interpolate(rest[None].contiguous(), rast, FACES)[0][0]
    cover = rast[0, ..., 3] > 0
    x, y, z = X.unbind(-1)
    top = rest[:, 1].max()
    arm = rest[rest[:, 0].abs() > 0.4]
    ay, az = arm[:, 1].median(), arm[:, 2].median()
    leg = rest[rest[:, 1] < 0.5]
    lx, lz = leg[:, 0].abs().median(), leg[:, 2].median()
    tor = rest[(rest[:, 0].abs() < ARMPIT) & (rest[:, 1] >= CROTCH)]
    hb = ((tor[:, 1] - CROTCH) / AXIAL).long()
    nb = int(hb.max()) + 1
    lo_z = torch.full((nb,), np.inf, device=DEV).scatter_reduce(0, hb, tor[:, 2], "amin")
    hi_z = torch.full((nb,), -np.inf, device=DEV).scatter_reduce(0, hb, tor[:, 2], "amax")
    mid = ((lo_z + hi_z) / 2).cpu().numpy()
    ok = np.isfinite(mid)  # the torso's middle at each height, front to back
    mid = torch.from_numpy(np.interp(np.arange(nb), np.flatnonzero(ok), mid[ok])).to(DEV).float()
    tz = mid[((y - CROTCH) / AXIAL).long().clamp(0, nb - 1)]
    head, arms, legs = y > top - HEAD, (x.abs() > ARMPIT) & (y > ay - 0.2), y < CROTCH
    part = torch.where(head, 3, torch.where(arms, 1, torch.where(legs, 2, 0)))
    axial = torch.where(part == 1, x.abs(), y)
    ang = torch.where(
        part == 1, torch.atan2(z - az, y - ay), torch.where(part == 2, torch.atan2(z - lz, x.abs() - lx), torch.atan2(z - tz, x))
    )
    return torch.where(cover, part, -1), axial, ang


def tidy(share):
    """A part's bins (axial x angle, round the part) with each colour's region cleared of what no garment has: bits narrower around
    the part than THIN bins, and pieces of fewer than ISLAND bins, which take the colours of the nearest bins that stay."""
    lab = share.argmax(-1).cpu().numpy()
    A = lab.shape[1]
    keep = np.zeros(lab.shape, bool)
    for c in np.unique(lab):
        wide = ndimage.binary_opening(np.tile(lab == c, (1, 3)), np.ones((1, THIN), bool))  # tiled: round the part
        pieces, _ = ndimage.label(wide)
        mid = pieces[:, A : 2 * A]
        size = np.bincount(mid.ravel())
        keep |= (mid > 0) & (size[mid] >= ISLAND)
    if keep.all() or not keep.any():
        return share
    s = share.cpu().numpy() * keep[..., None]
    return torch.from_numpy(fill_vote_bins(s)[0]).to(DEV).float()


def kit_design(votes, mu, coarse, fine, base):
    """The kit as a clean garment: each part's colour as a function of (axial position, angle around it), from AXIAL x ANGLES bins of
    the kit colours' seen samples (`mu`, from the raw broadcast pixels: kit.py votes, binned by `coarse`, the part field at the
    kits' size) pooled with the neighbouring bins (VOTE), tidied (tidy) and smoothed (SMOOTH), bins without samples from the nearest
    that has some.
    At each texel of `fine` (the part field at T * SS) the bins' shares are interpolated and the leading colour taken outright, so
    every band ends in a hard, smooth curve that stays sharp at any distance once averaged down to T. Colours are observed; pooling,
    smoothing and fill infer the garment's layout; on the sleeves and legs, rings (RING). A part with no samples, and texels outside
    the modelled parts, keep `base`."""
    out = base.clone()
    va, vb = VOTE
    pool = lambda v: F.avg_pool2d(F.pad(v, (vb, vb, 0, 0), mode="circular"), (2 * va + 1, 2 * vb + 1), 1, (va, 0))

    def smooth(v):
        for axis, s in enumerate(SMOOTH):
            r = int(np.ceil(3 * s))
            k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=DEV) / s) ** 2)
            k = k / k.sum()
            v = F.pad(v, (0, 0, r, r), mode="replicate") if axis == 0 else F.pad(v, (r, r, 0, 0), mode="circular")
            c = v.shape[1]
            w = (k[:, None] if axis == 0 else k[None, :]).expand(c, 1, *((2 * r + 1, 1) if axis == 0 else (1, 2 * r + 1)))
            v = F.conv2d(v, w.contiguous(), groups=c)
        return v

    for p in (0, 1, 2):
        m, mf = coarse[0] == p, fine[0] == p
        if not m.any() or not mf.any():
            continue
        a = coarse[1][m]
        lo = a.min()
        fi = (a - lo) / AXIAL
        fj = (coarse[2][m] + np.pi) / (2 * np.pi) * ANGLES - 0.5
        i, j = fi.long(), fj.floor().long() % ANGLES
        n = int(i.max()) + 1
        v = torch.zeros(n * ANGLES, len(mu), device=DEV).index_add_(0, i * ANGLES + j, votes[m])
        v = pool(v.reshape(n, ANGLES, len(mu)).permute(2, 0, 1)[None])[0].permute(1, 2, 0).cpu().numpy()
        filled, supported = fill_vote_bins(v)
        if not supported.any():
            continue
        share = torch.from_numpy(filled / filled.sum(-1, keepdims=True).clip(1e-9)).to(DEV).float()
        if p == 0:  # the torso's left and right halves alike (its angle
            th = (np.arange(ANGLES) + 0.5) / ANGLES * 2 * np.pi - np.pi  # is from the side: x to -x mirrors it to pi - angle)
            share = (share + share[:, np.round((2 * np.pi - th) / (2 * np.pi) * ANGLES - 0.5).astype(int) % ANGLES]) / 2
        share = smooth(tidy(share).permute(2, 0, 1)[None])[0].permute(1, 2, 0)
        if p in (
            1,
            2,
        ):  # sleeves and legs: rings round the limb, as kits are; a colour held only round part of it keeps its place
            share = RING * share.mean(1, keepdim=True) + (1 - RING) * share  # only where it leads by far
        ax, an = fine[1][mf], fine[2][mf]
        lead = torch.empty(len(ax), dtype=torch.long, device=DEV)
        for k in range(0, len(ax), CHUNK):  # bilinear over (axial, periodic angle), in chunks
            x = ((ax[k : k + CHUNK] - lo) / AXIAL - 0.5).clamp(0, n - 1)
            y = (an[k : k + CHUNK] + np.pi) / (2 * np.pi) * ANGLES - 0.5
            i0 = x.floor().long().clamp(max=max(n - 2, 0))
            j0 = y.floor().long()
            tx, ty = (x - i0)[:, None], (y - y.floor())[:, None]
            i1, j1 = (i0 + 1).clamp(max=n - 1), (j0 + 1) % ANGLES
            j0 = j0 % ANGLES
            q = (share[i0, j0] * (1 - ty) + share[i0, j1] * ty) * (1 - tx) + (share[i1, j0] * (1 - ty) + share[i1, j1] * ty) * tx
            lead[k : k + CHUNK] = q.argmax(-1)
        out[mf] = mu[lead]
    return out


def lettering(text, top, bottom, widest):
    """Coverage [h, w] of text set between top and bottom (m below the shoulder line), centred, at most `widest` wide, on a canvas
    spanning across +-0.25 m and up 0 .. -0.45 m at PX px per m."""
    W, H = int(0.5 * PX), int(0.45 * PX)
    canvas = Image.new("L", (W, H))
    font = ImageFont.truetype(FONT, 400)
    font.set_variation_by_name("Bold")
    l, t, r, bt = font.getbbox(text)
    im = Image.new("L", (r - l, bt - t))
    ImageDraw.Draw(im).text((-l, -t), text, font=font, fill=255)
    h = int((top - bottom) * PX)
    w = min(int(im.width * h / im.height), int(widest * PX))
    im = im.resize((w, h), Image.LANCZOS)
    canvas.paste(im, ((W - w) // 2, int(-top * PX)))
    return torch.from_numpy(np.asarray(canvas, np.float32) / 255).to(DEV)


def sample_canvas(canvas, across, up):
    """Canvas coverage at each texel's (across, up), the canvas first averaged down near the texel spacing so it does not alias."""
    c = F.avg_pool2d(canvas[None, None], 4)[0, 0]
    g = torch.stack([across / 0.25, -up / 0.45 * 2 - 1], -1)[None]
    return F.grid_sample(c[None, None], g, align_corners=False, padding_mode="zeros")[0, 0]


def pull_push(col, known):
    """Fill texels outside `known` from coarser averages of their surroundings (kit.py)."""
    if known.all() or col.shape[0] <= 1:
        return col
    c, k = col * known[..., None], known.float()
    c2 = F.avg_pool2d(c.permute(2, 0, 1)[None], 2, ceil_mode=True)[0].permute(1, 2, 0)
    k2 = F.avg_pool2d(k[None, None], 2, ceil_mode=True)[0, 0]
    coarse = pull_push(c2 / k2.clamp(min=1e-6)[..., None], k2 > 0)
    up = F.interpolate(coarse.permute(2, 0, 1)[None], size=col.shape[:2], mode="bilinear", align_corners=False)[0].permute(
        1, 2, 0
    )
    return torch.where(known[..., None], col, up)


def palette(img, k=6, iters=25):
    """k colours of a texture (k-means, seeded from its luminance quantiles)."""
    c = img.reshape(-1, 3)[::7]
    L = c.sum(1)
    mu = c[torch.argsort(L)[torch.linspace(0, len(c) - 1, k, device=DEV).long()]].clone()
    for _ in range(iters):
        a = torch.cdist(c, mu).argmin(1)
        mu = torch.stack([c[a == j].mean(0) if (a == j).any() else mu[j] for j in range(k)])
    return mu


def clean(img, mu, sigma=0.07, amount=0.6):
    """Pull every texel toward the palette colour it is nearest to (soft assignment): the broadcast noise and blotches averaged into
    the bake give way to the kit's flat colours while its layout (stripes, pants, sponsor bands) stays where it was observed."""
    d2 = torch.cdist(img.reshape(-1, 3), mu) ** 2
    w = torch.softmax(-d2 / (2 * sigma**2), 1)
    return img * (1 - amount) + (w @ mu).reshape(img.shape) * amount


def resize(img, n):
    """An [h, w, 3] image resized to [n, n] (bilinear)."""
    return (
        img
        if img.shape[0] == n
        else F.interpolate(img.permute(2, 0, 1)[None], size=(n, n), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    )


def down(img):
    """An image at T * SS averaged down to T."""
    return F.avg_pool2d(img.permute(2, 0, 1)[None], SS)[0].permute(1, 2, 0)


def to_plane(img, across, up, on):
    """The texels `on` the front of the torso laid flat on the CREST_AREA plane (seen from the front) at CREST_PX px per m, holes
    filled from their surroundings: [h, w, 3], and where it had texels."""
    (a0, a1), (u0, u1) = CREST_AREA
    w, h = round((a1 - a0) * CREST_PX), round((u0 - u1) * CREST_PX)
    x = ((across[on] - a0) * CREST_PX).long()
    y = ((u0 - up[on]) * CREST_PX).long()
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    i = y[ok] * w + x[ok]
    acc = torch.zeros(h * w, 3, device=DEV).index_add_(0, i, img[on][ok])
    n = torch.zeros(h * w, device=DEV).index_add_(0, i, torch.ones_like(i, dtype=torch.float))
    known = (n > 0).reshape(h, w)
    return pull_push((acc / n.clamp(min=1)[:, None]).reshape(h, w, 3), known), known


def crest_place(bake, design, across, up, on):
    """Where the kit's crest is, from the broadcast: on the front of the torso, the pieces where the kit's bake (`bake`, every
    view of it averaged) departs from its design by CREST_DIFF; the largest whose centre is within CREST_MID of the middle and
    whose sides are CREST_SIZE long. Its centre and size (across, up; m), or None."""
    seen, known = to_plane(bake, across, up, on)
    drawn, _ = to_plane(design, across, up, on)
    m = (((seen - drawn).norm(dim=-1) > CREST_DIFF) & known).cpu().numpy().astype(np.uint8)
    m = cv2.morphologyEx(
        cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    count, _, stats, centres = cv2.connectedComponentsWithStats(m)
    (a0, _), (u0, _) = CREST_AREA
    best = None
    for c in range(1, count):
        w, h, area = stats[c, 2] / CREST_PX, stats[c, 3] / CREST_PX, stats[c, 4]
        a, u = a0 + centres[c][0] / CREST_PX, u0 - centres[c][1] / CREST_PX
        if abs(a) < CREST_MID and all(CREST_SIZE[0] <= x <= CREST_SIZE[1] for x in (w, h)) and (best is None or area > best[0]):
            best = (area, (a, u), (w, h))
    return best and best[1:]


def put_logo(img, logo, centre, size, across, up, on):
    """`img` with RGBA `logo` printed on the front of the torso (`on`), as large as fits in `size` about `centre` (m)."""
    (a, u), (w, h) = centre, size
    k = min(w / logo.shape[1], h / logo.shape[0])
    w, h = logo.shape[1] * k, logo.shape[0] * k
    g = torch.stack([(across - a) / (w / 2), (u - up) / (h / 2)], -1)[None]
    c = F.grid_sample(logo.permute(2, 0, 1)[None], g, align_corners=False, padding_mode="zeros")[0].permute(1, 2, 0)
    a = c[..., 3:] * on[..., None].float()
    return img * (1 - a) + c[..., :3] * a


def surname(name):
    last = name.split()[-1]
    return "Mc" + last[2:].upper() if last.startswith("Mc") else last.upper()


def build(scene):
    K = np.load(f"{KITS}/kits.npz")
    tracks = json.load(open("bundle/cast.json"))["tracks"]
    across, up, back, front = torso_field()
    clear = back & (across.abs() < CLEAR[0]) & (up < CLEAR[1]) & (up > CLEAR[2])
    soft = F.avg_pool2d(clear.float()[None, None], 9, 1, 4)[0, 0]  # feathered edge
    island = F.max_pool2d(back.float()[None, None], 31, 1, 15)[0, 0] > 0  # the back's UV island, with margin
    region = back & (across.abs() < 0.2) & (up < 0) & (up > -0.6)
    tex = lambda nm: resize(torch.from_numpy(K[nm]).to(DEV).float() / 255, T)
    lum = lambda c: c @ torch.tensor([0.2126, 0.7152, 0.0722], device=DEV)
    ink = {}
    for team in (0, 1):
        base = tex(f"kit_{team}_0")[clear].median(0).values
        named = [nm for nm in K.files if nm.startswith(f"player_{team}_") and not nm.endswith("_seen")] or [f"kit_{team}_0"]
        c = torch.cat([tex(nm)[region] for nm in named])
        L = lum(c)
        ink[team] = c[L <= L.quantile(0.01)].mean(0) if lum(base) > 0.42 else c[L >= L.quantile(0.99)].mean(0)
        print("team", team, "jersey", (base * 255).round().tolist(), "number", (ink[team] * 255).round().tolist())
    feather = lambda m: F.avg_pool2d(m.float()[None, None], 7, 1, 3)[0, 0]
    helmet, mask, pads, gloves = (feather(f) for f in gear_fields())
    shell = {
        k: torch.tensor(v, device=DEV) / 255
        for k, v in json.load(open(f"{KITS}/gear_colours.json")).items()
        if not k.endswith("_seen")
    }
    kit_of = lambda t: "kit_ref" if t["team"] < 0 else f"kit_{t['team']}_{int(t['goalie'])}"
    palettes = {nm: palette(tex(nm)) for nm in K.files if nm.startswith("kit") and not nm.endswith("_seen")}
    field = part_field()
    face = F.avg_pool2d((field[0] == 3).float()[None, None], 9, 1, 4)[0, 0][..., None]
    KV = np.load(f"{KITS}/kit_votes.npz")
    kv = lambda nm, x: torch.from_numpy(KV[f"{nm}_{x}"].astype(np.float32)).to(DEV)
    coarse, fine = part_field(KIT), part_field(T * SS)
    big = lambda img: resize(img, T * SS)
    models = {
        nm: down(kit_design(kv(nm, "votes"), kv(nm, "palette"), coarse, fine, big(clean(tex(nm), palettes[nm]))))
        for nm in DESIGNED
        if f"{nm}_votes" in KV.files
    }
    teams = json.load(open(f"{KITS}/teams.json")) if os.path.exists(f"{KITS}/teams.json") else {}
    chest = front & (field[0] == 0)
    for team, known in teams.items():  # the crest: the team's logo, where its kit shows one
        if not known.get("logo"):
            continue
        logo = torch.from_numpy(np.asarray(Image.open(f"{KITS}/{known['logo']}").convert("RGBA"), np.float32) / 255).to(DEV)
        nm = f"kit_{team}_0"
        place = crest_place(tex(nm), models[nm], -across, up, chest) if nm in models else None
        print(nm, known["name"], "crest at", place and [round(x, 2) for x in place[0] + place[1]], flush=True)
        if place:
            models[nm] = put_logo(models[nm], logo, *place, -across, up, chest)
    for t in tracks:
        nm = f"player_{t['team']}_{t['jersey']}" if t["team"] >= 0 and t.get("jersey") else None
        nm = nm if nm in K.files else kit_of(t)
        img = clean(tex(nm), palettes[kit_of(t)])
        kit = models.get(f"kit_{t['team']}_0")  # players: their team's kit design, their own head;
        if kit is not None:
            img = kit * (1 - face) + img * face  # officials: their cleaned bake
        g = "g" if t["goalie"] else ""  # the helmet, or the goalie's mask, pads, gloves
        for c, m in (
            (shell.get(f"{t['team']}{g}"), mask if g else helmet),
            (shell.get(f"{t['team']}{g}_pads"), pads),
            (shell.get(f"{t['team']}{g}_gloves"), gloves),
        ):
            if c is None:
                continue
            img = img * (1 - m[..., None]) + c * m[..., None]
        if t["team"] >= 0:
            if kit is None:  # the bake's blurred lettering, painted over
                filled = pull_push(img, ~clear & island)
                img = img * (1 - soft[..., None]) + filled * soft[..., None]
            if t.get("jersey"):
                cov = sample_canvas(lettering(t["jersey"], *NUMBER), across, up)
                if t.get("name"):
                    cov = torch.maximum(cov, sample_canvas(lettering(surname(t["name"]), *NAME), across, up))
                cov = cov * back.float()
                colour = ink[t["team"]]
                if kit is not None:  # dark on the kit's light bands, light on its dark
                    mu = kv(kit_of(t), "palette")
                    under = lum(F.avg_pool2d(kit.permute(2, 0, 1)[None], 15, 1, 7)[0].permute(1, 2, 0))
                    colour = torch.where((under > 0.42)[..., None], mu[lum(mu).argmin()], mu[lum(mu).argmax()])
                img = img * (1 - cov[..., None]) + colour * cov[..., None]
        for name, im in (
            (f"texture_{t['key']}_2k", img),
            (f"texture_{t['key']}", F.avg_pool2d(img.permute(2, 0, 1)[None], T // KIT)[0].permute(1, 2, 0)),
        ):
            out = (im.clamp(0, 1) * 255).round().byte().cpu().numpy()
            cv2.imwrite(f"{scene}/{name}.jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(t["key"], t["team"], t.get("jersey"), t.get("name"), "kit design" if kit is not None else "from " + nm, flush=True)


def check(scene, sheet, keys):
    """Each person's old vertex colours and new texture on the rest mesh, front and back."""
    rest = torch.from_numpy(UV["rest"]).to(DEV) / 100
    rest = rest - torch.tensor([0, 0.86, 0], device=DEV)
    V = len(rest)
    rows = []
    for key in keys:
        vc = torch.from_numpy(np.fromfile(f"{scene}/color_{key}.bin", np.uint8).reshape(V, 3).astype(np.float32) / 255).to(DEV)
        tx = (
            torch.from_numpy(cv2.cvtColor(cv2.imread(f"{scene}/texture_{key}.jpg"), cv2.COLOR_BGR2RGB)).to(DEV).float()[None]
            / 255
        )
        views = []
        for mode in ("vertex", "texture"):
            for s in (1, -1):
                P = rest * torch.tensor([s, -1, -s], device=DEV)
                clip = torch.stack([P[:, 0], P[:, 1], P[:, 2] * 0.5, torch.ones_like(P[:, 0])], 1)[None].contiguous()
                if mode == "vertex":
                    rast, _ = dr.rasterize(ctx, clip, FACES, (512, 512))
                    col = dr.interpolate(vc[None].contiguous(), rast, FACES)[0]
                else:
                    rast, _ = dr.rasterize(ctx, clip, FACES, (512, 512))
                    uv, _ = dr.interpolate(TC[None].contiguous(), rast, TF)
                    col = dr.texture(tx.contiguous(), uv, filter_mode="linear")
                img = torch.where(rast[..., 3:] > 0, col, torch.full_like(col, 0.5))[0]
                views.append((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[:, 96:416])
        rows.append(np.hstack(views))
    cv2.imwrite(sheet, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    with torch.no_grad():
        if sys.argv[1] == "check":
            check(sys.argv[2], sys.argv[3], sys.argv[4:])
        else:
            build(sys.argv[1])
