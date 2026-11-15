"""Refine the arena atlas with Qwen-Image-2.1 in one pass: fill its holes and clean everything else (blur, smudges, seams).

Every texel gets a change level m in [0, 1] (differential diffusion): denoising starts from noise, and after every step the latents
whose m is below the step's noise level are reset to the atlas noised to that level. So holes (m = 1) are generated from scratch with
the whole canvas as context, the rest (m = BASE) may change only in the fine detail the last steps decide (sharpness, smudges), and texels
with m = 0 come out exactly. m is blurred across the hole edges, so generated and kept texels blend without a seam. The model also gets
the canvas as its condition image, holes greyed, with an instruction to fill the grey and restore the rest.

The atlases hold much less detail than their texels (the ice's content survives 4x down and up almost unchanged, the band's and
stands' 2x): at full resolution the model sees a soft picture and keeps it soft. With SCALE_<KIND>=k (a power of 2) a canvas is first
refined at 1/k of its size, where it looks like an ordinary photo and seams, smudges and noise are what the model fixes, then doubled
back to full size, each doubling an image-to-image pass from bicubic upsampling at change level UP (condition image only with
COND_UP=1: conditioned on the soft upsample, the model keeps it soft), so the model draws the fine detail.
The guidance (CFG) pushes away from NEGATIVE (blur, noise, smudges, patches, seams, uneven lighting) instead of an empty prompt
(NEG=0: empty): without it the model reproduces the atlas's softness at any scale.

The fill is the same for every atlas. Texels a view missed hold the other side's texels turned half a turn (arena_atlas: the copy),
and where the half turn missed them too, the builder's own fill (copied stretches of the same row of seats, which keep the spectators'
size; plain ice). First the whole atlas is refined as one canvas at 1/k (WHOLE), so the model sees the entire surface while it fills:
every filled texel is redrawn at COPY from what the builder put there, keeping its own fine detail (on the ice from scratch: the copy
is the far side seen from far away, soft and smeared, and brings the far goal's net along). The result replaces the fill, shifted back by the median colour the
model drifted from the copies (it whitens generated ice by 27 grey levels); the seen texels stay. Seen texels within MARGIN of a copy
are generated from scratch: whatever hid the copied texels from this side (boards, glass, the bench's frame) bleeds into the edge of
what it did not hide (on the ice the first 30 cm deviate up to 3 times as much from their row as the ice further in; the band's and
stands' edges deviate no more than their content). Then the canvases redraw every texel at BASE at full size; on the stands, texels
neither side saw (the builder's mosaic of copied stretches) are generated there from scratch, beside the reference columns that keep
the spectators' size.

  stands  (h up, s across, 2.5 cm): canvases of the full seen height, left to right, each with the best-seen REF_COLS columns of the
          atlas beside it (m = 0), aligned row for row, so the model sees the true size of spectators at every height. Later canvases
          keep what earlier ones made where they overlap (m ramps up).
  band    (h up, 1 cm): the boards band cut into stretches of BAND_SEG columns stacked into one canvas, upright so sponsor text reads,
          overlapping stretches blended.
  ice     (1 cm): overlapping tiles, blended.

Last (sharpen, part of all): every atlas through UltraSharpV2, a deterministic restoration model, back at the atlas resolution.

usage (image venv, uv --with spandrel):  atlas_refine.py <src_atlas_dir> <out_atlas_dir> <stands|band|ice|refine|sharpen|all>
  refine: the three atlases, all: refine then sharpen. BASE_<KIND>=m and COPY_<KIND>=m override a kind's change level for seen and
  copied texels; SCALE_<KIND>=k, UP, COND_UP: see above; ONLY=n processes only canvas n and writes <kind>_refine_check.jpg
  (before above, after below) instead of the atlas"""

import glob
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import QwenImage21Pipeline
from PIL import Image

HUB = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
PROMPTS = dict(
    stands="Fill the flat grey areas of this photo of arena seating so they continue the stands around them: the same rows, seats, "
    "stairs and railings, spectators of the same size, density, clothing colours and lighting, seen straight on: every row of "
    "seats runs horizontally across the image, stairs run vertically. Restore the rest: sharp and clean, "
    "no blur, noise, smears, patches, seams or uneven lighting, layout and scale unchanged. No new text.",
    band="Restore this photo of ice hockey dasher boards with sponsor panels, stacked in strips: sharp and clean, no blur, noise, smears, "
    "patches, seams or uneven lighting. Keep "
    "every logo, letter and colour exactly as it is. Fill any flat grey areas so they continue the boards around them.",
    ice="Restore this top-down photo of an ice hockey rink's ice: sharp and clean, no blur, noise, smears, patches, "
    "seams or uneven lighting. Keep every line, circle, logo and "
    "letter exactly as it is. Fill any flat grey areas so they continue the ice around them.",
)
NEGATIVE = (
    "blurry, soft focus, out of focus, low resolution, upscaled, noise, grain, jpeg artifacts, smudges, smears, blotchy patches, "
    "visible seams, uneven lighting, washed out"
)
NEG = os.environ.get("NEG", "1") == "1"  # guide away from NEGATIVE (else an empty negative prompt)
CFG = float(os.environ.get("CFG", 4.0))
BASE = dict(stands=0.85, band=0.85, ice=0.85)  # change level of seen texels: how much detail may be redone
EDGE = dict(stands=0.5, band=0.1, ice=0.2)  # m blur across hole edges (m)
SCALE = {
    k: int(os.environ.get(f"SCALE_{k.upper()}", 1)) for k in ("stands", "band", "ice")
}  # refine at 1/SCALE first, then double up
STANDS_COLS, REF_COLS, OVERLAP = 1280, 512, 320
BAND_SEG, BAND_STACK, BAND_RAMP = 2048, 4, 128  # band: stretch length, stretches per canvas, crossfade (texels)
ICE_TILE, ICE_RAMP = (1344, 2304), 256
STEPS, GREY = 30, 128
UP = float(os.environ.get("UP", 0.4))  # change level of the doubling passes (SCALE_<KIND> > 1)
COND_UP = os.environ.get("COND_UP", "0") == "1"
COPY = {
    k: float(os.environ.get(f"COPY_{k.upper()}", v)) for k, v in dict(stands=0.9, band=0.9, ice=1.0).items()
}  # change level of
# copied texels (band, stands: at 1 the model invents text and rows)
MARGIN = dict(stands=0, band=0, ice=0.3)  # m of seen texels beside a copy generated from scratch
WHOLE = dict(stands=(3, 3), band=(2, 4), ice=(4, 1))  # whole-atlas canvas: 1/k of the size, cut into n stacked lengths
WRAP = 32  # texels each stacked length takes of its neighbours either side
SHARP_TILE, SHARP_PAD = 512, 32
ONLY = int(os.environ["ONLY"]) if "ONLY" in os.environ else None


def load():
    return QwenImage21Pipeline.from_pretrained(
        sorted(glob.glob(f"{HUB}/models--Qwen--Qwen-Image-2.1/snapshots/*"))[0], dtype=torch.bfloat16
    ).to("cuda")


def encode(pipe, rgb8):
    """[h, w, 3] uint8 -> packed latents [1, h/f * w/f, C] (f: pipe.vae_scale_factor; the VAE takes RGBA)."""
    rgba = np.concatenate([rgb8, np.full_like(rgb8[..., :1], 255)], 2)
    x = torch.from_numpy(rgba).to("cuda", torch.bfloat16).permute(2, 0, 1)[None, :, None] / 127.5 - 1
    z = pipe._encode_vae_image(x, None)
    return pipe._pack_latents(z, 1, z.shape[1], z.shape[3], z.shape[4])


def resize(x, h, w, mode):
    """[H, W] or [H, W, 3] float array resized (torch: "area" down, "bicubic" up)."""
    t = torch.from_numpy(np.ascontiguousarray(x, np.float32)).cuda()
    t = t[None, None] if t.ndim == 2 else t.permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(h, w), mode=mode, **({} if mode == "area" else dict(align_corners=False)))[0]
    return (t[0] if x.ndim == 2 else t.permute(1, 2, 0)).cpu().numpy()


def refine(pipe, canvas, m, prompt, seed, scale=1):
    """canvas [h, w, 3] uint8 (h, w multiples of 32: the vision encoder's patches) with change levels m [h, w]; returns the refined canvas
    [h, w, 3] float. scale > 1: refined at 1/scale first, then doubled back to full size (sizes rounded to multiples of 32)."""
    h, w = m.shape
    sizes = [(h, w)]
    while len(sizes) < int(np.log2(scale)) + 1:
        sizes.append((max(32, round(sizes[-1][0] / 64) * 32), max(32, round(sizes[-1][1] / 64) * 32)))
    (h0, w0), up = sizes[-1], sizes[-2::-1]
    if not up:
        return diffuse(pipe, canvas, m, prompt, seed, True)
    cur = diffuse(pipe, resize(canvas, h0, w0, "area").round().astype(np.uint8), resize(m, h0, w0, "area"), prompt, seed, True)
    for hh, ww in up:
        mu = np.where(resize(m, hh, ww, "area") > 0, UP, 0.0).astype(np.float32)
        cur = diffuse(pipe, resize(cur, hh, ww, "bicubic").clip(0, 255).round().astype(np.uint8), mu, prompt, seed, COND_UP)
    return cur


def whole(pipe, kind, img, m, a, copy, seed):
    """img [H, W, 3] uint8 refined as one canvas at 1/k (WHOLE[kind] = k, n) with change levels m: cut into n lengths (n > 1: the atlas
    is a loop, each length takes WRAP texels of its neighbours either side, blended back by weight) stacked top to bottom. Texels it
    redrew (m < 1) take only its low frequencies (Gaussian of k texels) and keep their own detail, at the scale the builder's fill has it
    (a canvas at 1/k draws spectators too large); generated ones take it whole. The result, shifted back by the median colour it
    drifted from the copied texels (copy [H, W] bool), replaces img by the fill's weight a [H, W]."""
    (k, n), (H, W) = WHOLE[kind], m.shape
    o = WRAP if n > 1 else 0
    h, L = max(32, round(H / k / 32) * 32), max(32, round(W / k / n / 32) * 32)
    small, ms = resize(img, h, n * L, "area"), resize(m, h, n * L, "area")
    cols = [np.arange(i * L - o, (i + 1) * L + o) % (n * L) for i in range(n)]
    g = diffuse(
        pipe,
        np.concatenate([small[:, c] for c in cols]).round().astype(np.uint8),
        np.concatenate([ms[:, c] for c in cols]),
        PROMPTS[kind],
        seed,
        True,
    )
    wt = trapezoid(L + 2 * o, 2 * o) if o else np.ones(L, np.float32)
    num, den = np.zeros_like(small), np.zeros(n * L, np.float32)
    for i, c in enumerate(cols):
        np.add.at(num, (slice(None), c), g[i * h : (i + 1) * h] * wt[:, None])
        np.add.at(den, c, wt)
    print(f"{kind}: whole-atlas pass at {L + 2 * o}x{n * h}", flush=True)
    g, f = resize(num / den[:, None], H, W, "bicubic"), img.astype(np.float32)
    low = lambda x: np.stack([blur(x[..., c], k) for c in range(3)], -1)
    g = np.where((m < 0.999)[..., None], f + low(g) - low(f), g)
    shift = np.median(img[copy], 0) - np.median(g[copy], 0)
    print(f"{kind}: fill colour shifted by {np.round(shift, 1)}", flush=True)
    g = g + shift
    return (img + (g - img) * a[..., None]).clip(0, 255).round().astype(np.uint8)


def diffuse(pipe, canvas, m, prompt, seed, cond_image):
    """One differential-diffusion pass over canvas with change levels m; the canvas (holes greyed) is the condition image if cond_image."""
    h, w = m.shape
    f = pipe.vae_scale_factor
    cond = canvas.copy()
    cond[m > 0.95] = GREY
    x0 = encode(pipe, canvas)
    lat = torch.from_numpy(m).float().cuda().reshape(h // f, f, w // f, f).mean((1, 3)).flatten()[None, :, None]
    g = torch.Generator("cuda").manual_seed(seed)
    noise = torch.randn(x0.shape, generator=g, device="cuda", dtype=x0.dtype)

    def differential(p, i, t, kw):
        s = float(p.scheduler.sigmas[i + 1])
        keep = ((lat < s) | (lat == 0)).to(x0.dtype)
        return {"latents": keep * ((1 - s) * x0 + s * noise) + (1 - keep) * kw["latents"]}

    out = pipe(
        prompt=prompt,
        image=Image.fromarray(cond) if cond_image else None,
        height=h,
        width=w,
        output_resolution=int((h * w) ** 0.5),
        num_inference_steps=STEPS,
        true_cfg_scale=CFG,
        negative_prompt=NEGATIVE if NEG else " ",
        latents=noise.clone(),
        generator=g,
        callback_on_step_end=differential,
    ).images[0]
    return np.asarray(out.convert("RGB"), np.float32)


def blur(x, sigma):
    """Gaussian blur of a float map [h, w] (separable, edges replicated)."""
    r = int(3 * sigma) + 1
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device="cuda") / sigma) ** 2)
    k = (k / k.sum()).float()
    t = torch.from_numpy(np.ascontiguousarray(x, np.float32)).cuda()[None, None]
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return t[0, 0].cpu().numpy()


def grow(x, r):
    """Bool map [h, w] dilated by r texels (a square)."""
    t = torch.from_numpy(x.astype(np.float32)).cuda()[None, None]
    t = F.max_pool2d(F.max_pool2d(t, (1, 2 * r + 1), 1, (0, r)), (2 * r + 1, 1), 1, (r, 0))
    return t[0, 0].cpu().numpy() > 0


def change_map(kind, filled, copy, base, res):
    """Change levels of the whole-atlas pass and the fill's weight: 1 in the seen texels within MARGIN of a copy (copy: filled from the
    half turn), COPY in the filled texels, base elsewhere, each raised smoothly over EDGE beyond its edge so the fill blends in."""
    r = round(MARGIN[kind] / res)
    bled = grow(copy, r) & ~filled if r else np.zeros_like(filled)
    soft = lambda x: np.clip(blur(x.astype(np.float32), EDGE[kind] / res) * 2, 0, 1)
    a = soft(filled | bled)
    print(f"{kind}: filled {filled.mean():.1%} (copied {copy.mean():.1%}), generated beside copies {bled.mean():.1%}", flush=True)
    return np.maximum(base + (COPY[kind] - base) * a, soft(bled)), a


def check(out, kind, before, after):
    Image.fromarray(np.concatenate([before, after.round().astype(np.uint8)], 0)).save(
        f"{out}/{kind}_refine_check.jpg", quality=92
    )


def stands(pipe, out, meta):
    T = meta["stands"]
    res = T["res"]
    base = float(os.environ.get("BASE_STANDS", BASE["stands"]))
    img = np.asarray(Image.open(f"{out}/stands.jpg").convert("RGB"))[::-1].copy()  # upright: h up
    filled = (np.asarray(Image.open(f"{out}/stands_fill.png").convert("L")) > 127)[::-1].copy()
    H, W = filled.shape
    top = int(T["seen_top"] / res) // 16 * 16
    y0 = H - top
    f = filled[y0:]
    copy = f & ~np.roll(f, W // 2, 1)
    m, a = change_map("stands", f, copy, base, res)
    orig = img.copy()
    img[y0:] = whole(pipe, "stands", img[y0:], m, a, copy, 50)
    m = np.maximum(
        base, np.clip(blur((f & np.roll(f, W // 2, 1)).astype(np.float32), EDGE["stands"] / res) * 2, 0, 1)
    )  # neither side saw: from scratch
    ref_x = int(np.argmax(np.convolve((~filled[y0:]).mean(0), np.ones(REF_COLS), "valid")))
    ref = img[y0:, ref_x : ref_x + REF_COLS].copy()
    print(f"stands: reference at s {ref_x * res:.1f} m, base {base}", flush=True)
    xs = list(range(0, W - STANDS_COLS + 1, STANDS_COLS - OVERLAP)) + [W - STANDS_COLS]
    ramp = np.clip(np.arange(STANDS_COLS) / OVERLAP, 0, 1)[None]  # m over what the last canvas made
    for n, x in enumerate(xs):
        if ONLY is not None and n != ONLY:
            continue
        mc = m[:, x : x + STANDS_COLS] * (ramp if n else 1)
        canvas = np.concatenate([ref, img[y0:, x : x + STANDS_COLS]], 1)
        gen = refine(
            pipe, canvas, np.concatenate([np.zeros_like(ref[..., 0], np.float32), mc], 1), PROMPTS["stands"], n, SCALE["stands"]
        )[:, REF_COLS:]
        a = np.clip(mc / 0.05, 0, 1)[..., None]
        before = orig[y0:, x : x + STANDS_COLS].copy()
        img[y0:, x : x + STANDS_COLS] = (img[y0:, x : x + STANDS_COLS] * (1 - a) + gen * a).round().astype(np.uint8)
        print(f"stands canvas {n + 1}/{len(xs)} at s {x * res:.1f} m", flush=True)
        if ONLY is not None:
            return check(out, "stands", before, img[y0:, x : x + STANDS_COLS].astype(np.float32))
    Image.fromarray(np.ascontiguousarray(img[::-1])).save(f"{out}/stands.jpg", quality=93)


def blended(pipe, kind, img, m, windows, layout, seed0):
    """Refine img [h, w, 3] through canvases; windows: lists of (rows, cols) index arrays stacked by layout into one canvas each, with
    weights [len(rows), len(cols)]; overlapping results are averaged by weight. Returns the refined image (float) and the total weight."""
    num = np.zeros(img.shape, np.float32)
    den = np.zeros(img.shape[:2], np.float32)
    for n, parts in enumerate(windows):
        if ONLY is not None and n != ONLY:
            continue
        canvas = layout([img[np.ix_(r, c)] for r, c, _ in parts])
        mc = layout([m[np.ix_(r, c)] for r, c, _ in parts])
        gen = refine(pipe, canvas, mc, PROMPTS[kind], seed0 + n, SCALE[kind])
        y = 0
        for r, c, wt in parts:
            np.add.at(num, np.ix_(r, c), gen[y : y + len(r)] * wt[..., None])
            np.add.at(den, np.ix_(r, c), wt)
            y += len(r)
        print(f"{kind} canvas {n + 1}/{len(windows)}", flush=True)
    return num / np.maximum(den, 1e-6)[..., None], den


def trapezoid(n, ramp, pad=0):
    w = np.minimum(np.arange(n) - pad + 1, n - pad - np.arange(n)).astype(np.float32) / ramp
    return np.clip(w, 0, 1)


def band(pipe, out, meta):
    B = meta["band"]
    P, L, sr, k = B["pad"], B["strip_cols"], B["strip_rows"], B["strips"]
    base = float(os.environ.get("BASE_BAND", BASE["band"]))
    folded = np.asarray(Image.open(f"{out}/band.jpg").convert("RGB"))
    ff = np.asarray(Image.open(f"{out}/band_fill.png").convert("L")) > 127
    unfold = lambda x: np.concatenate([x[j * sr + P : (j + 1) * sr - P, P : P + L] for j in range(k)], 1)[
        ::-1, ::-1
    ]  # h up, s leftwards: text reads
    img, filled = unfold(folded).copy(), unfold(ff).copy()
    H, W = filled.shape
    copy = filled & ~np.roll(filled, int(round(W / 2)), 1)
    m, a = change_map("band", filled, copy, base, B["res"])
    img = whole(pipe, "band", img, m, a, copy, 250)
    m = np.full_like(m, base)
    rows = np.arange(H)
    Hc = H // 16 * 16
    rows = rows[H - Hc :]  # the top rows (h 2.4 m) are the stands' anyway
    step = BAND_SEG - BAND_RAMP
    starts = np.arange(0, W, step)
    segs = [
        (rows, np.arange(s0, s0 + BAND_SEG) % W, np.broadcast_to(trapezoid(BAND_SEG, BAND_RAMP), (Hc, BAND_SEG))) for s0 in starts
    ]
    windows = [segs[i : i + BAND_STACK] for i in range(0, len(segs), BAND_STACK)]
    windows[-1] = windows[-1] + segs[: BAND_STACK - len(windows[-1])]  # fill the last canvas round the loop
    print(f"band: {len(segs)} stretches in {len(windows)} canvases, base {base}", flush=True)
    gen, den = blended(pipe, "band", img, m, windows, lambda xs: np.concatenate(xs, 0), 0)
    a = (np.clip(m / 0.05, 0, 1) * (den > 0))[..., None]
    new = (img * (1 - a) + gen * a).round().astype(np.uint8)
    if ONLY is not None:
        c = windows[ONLY][0][1]
        return check(out, "band", img[:, c], new[:, c].astype(np.float32))
    newf = new[::-1, ::-1]
    rows8 = []
    for j in range(k):  # fold again (arena_atlas.fold)
        cols = np.arange(j * L - P, (j + 1) * L + P) % W
        s_ = newf[:, cols]
        rows8.append(np.concatenate([np.repeat(s_[:1], P, 0), s_, np.repeat(s_[-1:], P, 0)], 0))
    Image.fromarray(np.concatenate(rows8, 0)).save(f"{out}/band.jpg", quality=93)


def ice(pipe, out, meta):
    base = float(os.environ.get("BASE_ICE", BASE["ice"]))
    img = np.asarray(Image.open(f"{out}/ice.jpg").convert("RGB"))[::-1].copy()  # y up: seen from above
    filled = (np.asarray(Image.open(f"{out}/ice_fill.png").convert("L")) > 127)[::-1].copy()
    copy = filled & ~filled[::-1, ::-1]
    m, a = change_map("ice", filled, copy, base, meta["ice"]["res"])
    orig = img
    img = whole(pipe, "ice", img, m, a, copy, 150)
    m = np.full_like(m, base)
    (th, tw), H, W = ICE_TILE, *img.shape[:2]
    grid = lambda n, t: sorted(set(list(range(0, n - t + 1, t - ICE_RAMP)) + [n - t]))
    wt = np.outer(trapezoid(th, ICE_RAMP), trapezoid(tw, ICE_RAMP))
    windows = [[(np.arange(y, y + th), np.arange(x, x + tw), wt)] for y in grid(H, th) for x in grid(W, tw)]
    print(f"ice: {len(windows)} tiles, base {base}", flush=True)
    gen, den = blended(pipe, "ice", img, m, windows, lambda xs: xs[0], 100)
    a = (np.clip(m / 0.05, 0, 1) * (den > 0))[..., None]
    new = (img * (1 - a) + gen * a).round().astype(np.uint8)
    if ONLY is not None:
        r, c, _ = windows[ONLY][0]
        return check(out, "ice", orig[np.ix_(r, c)], new[np.ix_(r, c)].astype(np.float32))
    Image.fromarray(np.ascontiguousarray(new[::-1])).save(f"{out}/ice.jpg", quality=93)


def sharpen(out):
    """Every atlas file through UltraSharpV2 (4x, deterministic: it sharpens what is there rather than inventing) and back down to the
    atlas resolution by area, in SHARP_TILE tiles with SHARP_PAD context."""
    from spandrel import ModelLoader

    model = (
        ModelLoader()
        .load_from_file(glob.glob(f"{HUB}/models--Kim2091--UltraSharpV2/snapshots/*/4x-UltraSharpV2.safetensors")[0])
        .cuda()
        .eval()
    )
    for name in ("stands", "band", "ice"):
        img = (
            torch.from_numpy(np.asarray(Image.open(f"{out}/{name}.jpg").convert("RGB"))).cuda().permute(2, 0, 1)[None].float()
            / 255
        )
        H, W = img.shape[2:]
        res = torch.empty_like(img)
        for y in range(0, H, SHARP_TILE):
            for x in range(0, W, SHARP_TILE):
                y0, x0 = max(y - SHARP_PAD, 0), max(x - SHARP_PAD, 0)
                y1, x1 = min(y + SHARP_TILE + SHARP_PAD, H), min(x + SHARP_TILE + SHARP_PAD, W)
                up = F.interpolate(model(img[:, :, y0:y1, x0:x1]).float(), size=(y1 - y0, x1 - x0), mode="area")
                res[:, :, y : y + SHARP_TILE, x : x + SHARP_TILE] = up[
                    :, :, y - y0 : y - y0 + SHARP_TILE, x - x0 : x - x0 + SHARP_TILE
                ]
        Image.fromarray((res[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)).save(
            f"{out}/{name}.jpg", quality=93
        )
        print(f"sharpened {name}", flush=True)


def main():
    src, out, kind = sys.argv[1:4]
    if os.path.abspath(src) != os.path.abspath(out):
        shutil.copytree(src, out, dirs_exist_ok=True)
    meta = json.load(open(f"{out}/atlas.json"))
    pipe = load() if kind != "sharpen" else None
    for k, fn in (("stands", stands), ("band", band), ("ice", ice)):
        if kind in (k, "refine", "all"):
            fn(pipe, out, meta)
    if kind in ("sharpen", "all"):
        del pipe
        torch.cuda.empty_cache()
        sharpen(out)


if __name__ == "__main__":
    with torch.no_grad():
        main()
