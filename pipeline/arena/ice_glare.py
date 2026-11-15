"""The ice's glare, taken out of the arena atlas and turned into the roof it reflects, so the viewer can light the ice from any view.

The atlas's ice (arena_atlas.py) is what the broadcast camera saw from its one fixed centre C: the ice's own colour plus the roof's
lamps mirrored in it, soft streaks wherever C's reflected rays meet them. Seen from anywhere else those streaks lie elsewhere. So:
  1. the glare: where the ice is plain white (no lines or logos), its brightness over the white ice's own (a low percentile of
     BLOCK medians over BASE m), smoothed; carried across the markings from the white ice round them;
  2. the matte ice: the atlas with the glare taken off (sRGB, as the atlas);
  3. the roof: the ice point x reflects, seen from C, the point q = x + (x - C) Z / C.z of a roof plane at height Z (C mirrored in the
     ice, through x, on up). That map is affine, so the glare divided by the ice's reflectance seen from C (Fresnel's, ice's f0, at
     the angle C sees x, weighted by GGX's energy: envBrdf, as light.ts) is the roof's radiance, resampled onto a grid over the roof.
     What C never saw by reflection is filled from the arena's symmetry (mirrored across the rink's long axis), then along the
     trusses' direction from the nearest row seen.
The viewer reads the roof along each view's reflected ray (q = x + r.xy (Z - x.z) / r.z) times the reflectance there, over the matte
ice: from C it is the atlas again; from anywhere else the lamps' streaks move as they would.

  <out>/ice_matte.jpg     the ice without its glare (the ice grid)
  <out>/roof.png          the roof's radiance, sqrt-encoded: value = sqrt(radiance / max) (8 bit)
  <out>/atlas.json        "glare": file (roof.png), matte, z, res, x0, y0, cols, rows, max; "ice" unchanged

usage (LTX venv):  ice_glare.py <arena atlas dir> <bundle scene.json>  (any goal's: the rink)    (the arena's last step: just arena)"""

import json
import sys

import cv2
import numpy as np
from scipy import ndimage

Z = 16.0  # m: the lamps' height over the ice (assumed: the footage never shows the roof); only the parallax depends on it
F0, ROUGH = 0.018, 0.1  # ice: reflectance head on (n = 1.31), roughness of its glare's lobe
K = 5  # the glare is estimated on a grid K times coarser than the ice's (5 cm): it is soft
BLOCK, BASE, LOW = 0.5, 6.0, 25  # m blocks of white ice (each its median), over a BASE m window the LOW percentile of those: the
# ice's own brightness under the glare (streaks a lamp throws are narrower than the window)
WHITE_CHROMA, WHITE_MIN = 0.1, 0.45  # white ice: chroma at most (linear (max - min) / max), luminance at least (linear)
SMOOTH, ACROSS = 0.25, 0.35  # m: the glare smoothed over, and carried across the markings from (doubling until met)
FLOOR = 0.02  # glare below this (linear) is the ice's own unevenness, not a lamp
ROOF_RES = 0.1  # m per roof texel
CLEAR = 0.6  # m: the ice this close to the boards (their shade, their reflection) holds no glare


def to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def env_brdf(rough, nv):
    """GGX's split-sum (scale, bias) for roughness and n.v: Karis' analytic fit, as light.ts."""
    c0, c1 = np.array([-1.0, -0.0275, -0.572, 0.022]), np.array([1.0, 0.0425, 1.04, -0.04])
    r = rough * c0 + c1
    a004 = np.minimum(r[0] * r[0], np.exp2(-9.28 * nv)) * r[0] + r[1]
    return -1.04 * a004 + r[2], 1.04 * a004 + r[3]


def reflectance(nv):
    a, b = env_brdf(ROUGH, nv)
    return F0 * a + b


def glare(lin, res, inside):
    """The glare (linear luminance) on a grid of res m, from the ice's linear colour there; inside: the ice clear of the boards."""
    L = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    mx, mn = lin.max(-1), lin.min(-1)
    white = ((mx - mn) / np.maximum(mx, 1e-3) < WHITE_CHROMA) & (L > WHITE_MIN) & inside
    blur = lambda x, m: cv2.GaussianBlur(x, (0, 0), m / res)
    # the ice's own brightness: per BLOCK the white ice's median, then a low percentile of those over BASE m
    b = int(round(BLOCK / res))
    h, w = L.shape[0] // b, L.shape[1] // b
    cells = np.where(white, L, np.nan)[: h * b, : w * b].reshape(h, b, w, b).transpose(0, 2, 1, 3).reshape(h, w, -1)
    enough = np.isfinite(cells).mean(-1) > 0.3
    med = np.where(enough, np.nanmedian(np.where(enough[..., None], cells, 0), -1), np.inf)
    base = ndimage.percentile_filter(med, LOW, size=int(BASE / BLOCK) | 1, mode="nearest")
    for _ in range(8):  # windows with too few white blocks: from their neighbours
        hole = ~np.isfinite(base)
        if not hole.any():
            break
        grown = ndimage.grey_erosion(np.where(hole, np.inf, -base), size=3)
        base = np.where(hole & np.isfinite(grown), -grown, base)
    base = cv2.resize(
        np.where(np.isfinite(base), base, np.nanmedian(med[enough])).astype(np.float32),
        L.shape[::-1],
        interpolation=cv2.INTER_LINEAR,
    )
    base = blur(base, BLOCK)
    # the glare over it, on the white ice, carried across the markings at growing scales until it meets white ice
    m = white.astype(np.float32)
    g = np.where(white, L - base, 0).astype(np.float32)
    out, wsum = blur(g, SMOOTH), blur(m, SMOOTH)
    known = wsum > 0.5
    est = np.where(known, out / np.maximum(wsum, 1e-3), 0)
    sigma = ACROSS
    while not known.all() and sigma < 8:
        ws = blur(m, sigma)
        fill = blur(g, sigma) / np.maximum(ws, 1e-3)
        new = ~known & (ws > 0.05)
        est[new], known = fill[new], known | new
        sigma *= 2
    return np.clip(est - FLOOR, 0, None) * inside


def sd_rink(x, y, rink):
    """Signed distance to the rink's outline (m, negative inside), as the viewer's sdRink."""
    qx, qy = np.abs(x) - (rink["hx"] - rink["corner"]), np.abs(y) - (rink["hy"] - rink["corner"])
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rink["corner"]


def main(out, scene):
    meta, rink = json.load(open(f"{out}/atlas.json")), json.load(open(scene))
    I, C = meta["ice"], np.array(meta["camera"], np.float64)
    srgb = cv2.cvtColor(cv2.imread(f"{out}/{I['file']}"), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    rows, cols = srgb.shape[:2]
    small = cv2.resize(srgb, (cols // K, rows // K), interpolation=cv2.INTER_AREA)
    res = I["res"] * cols / (cols // K)
    ys, xs = (np.arange(small.shape[0]) + 0.5) * res + I["y0"], (np.arange(small.shape[1]) + 0.5) * res + I["x0"]
    X, Y = np.meshgrid(xs, ys)
    G = glare(to_linear(small), res, sd_rink(X, Y, rink) < -CLEAR)
    print(f"glare: mean {G.mean():.4f}, 99% {np.quantile(G, 0.99):.3f}, max {G.max():.3f}", flush=True)

    # 2. the matte ice, at full resolution
    Gf = cv2.resize(G, (cols, rows), interpolation=cv2.INTER_LINEAR)[..., None]
    matte = to_srgb(to_linear(srgb) - Gf)
    cv2.imwrite(
        f"{out}/ice_matte.jpg",
        cv2.cvtColor((matte * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 93],
    )

    # 3. the roof: glare over the reflectance seen from C, resampled through q = x + (x - C) Z / C.z
    nv = C[2] / np.sqrt((C[0] - X) ** 2 + (C[1] - Y) ** 2 + C[2] ** 2)
    R = (G / reflectance(nv)).astype(np.float32)
    s = Z / C[2]
    qx = lambda x: x + (x - C[0]) * s
    qy = lambda y: y + (y - C[1]) * s
    x0, x1 = qx(xs[0]), qx(xs[-1])
    hi = max(abs(qy(ys[0])), abs(qy(ys[-1])))
    rc, rr = int((x1 - x0) / ROOF_RES), int(2 * hi / ROOF_RES)
    gx, gy = x0 + (np.arange(rc) + 0.5) * ROOF_RES, -hi + (np.arange(rr) + 0.5) * ROOF_RES
    # roof point back to the ice point C reflected it from, in ice texels
    ix = ((gx + C[0] * s) / (1 + s) - xs[0]) / res
    iy = ((gy + C[1] * s) / (1 + s) - ys[0]) / res
    seen = (iy >= 0) & (iy <= len(ys) - 1)
    mx, my = np.meshgrid(ix.astype(np.float32), iy.astype(np.float32))
    roof = cv2.remap(R, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # unseen rows: the mirror across the long axis where that was seen, then the nearest row seen
    mirror = seen[::-1] & ~seen
    roof[mirror] = roof[::-1][mirror]
    have = seen | mirror
    idx = np.where(have)[0]
    near = idx[np.abs(np.arange(rr)[:, None] - idx[None]).argmin(1)]
    roof = roof[near]
    top = float(np.quantile(roof, 0.999))
    cv2.imwrite(f"{out}/roof.png", (np.sqrt(np.clip(roof / top, 0, 1)) * 255).round().astype(np.uint8))
    meta["glare"] = dict(
        file="roof.png",
        matte="ice_matte.jpg",
        z=Z,
        res=ROOF_RES,
        x0=float(gx[0] - ROOF_RES / 2),
        y0=float(-hi),
        cols=rc,
        rows=rr,
        max=top,
    )
    json.dump(meta, open(f"{out}/atlas.json", "w"), indent=1)
    print(
        f"roof {rc}x{rr} over x {x0:.1f}..{x1:.1f}, y +-{hi:.1f} m; seen rows {seen.mean():.2f}, max radiance {top:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
