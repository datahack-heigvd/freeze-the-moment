"""The stands the broadcast camera saw badly, quilted from the crowd it saw well: real spectators instead of generated ones.

Three kinds of stands texels hold no usable crowd. The stands behind the goals and round the corners (outside SOURCE and its half
turn, which the camera's own side copies), seen at a glancing angle through a profile far from the real one: up to ENDS_TOP (above it
hang the banners) they are a swirl of smeared people and beams. The builder's mosaic of copied stretches where neither the texel nor its half turn
was seen (arena_atlas.fill_out: the blocky patchwork behind the goals), and the crowd's tiers (below CROWD_TOP; above them roof beams
and rails run in lines of their own) where the corners, seen through the wrong profile, turned the crowd into streaks: direction
consistency of the structure tensor over STREAK_SPAN above STREAK_MIN (crowd is near 0.4), or over SMEAR_SPAN above SMEAR_MIN in a
direction SMEAR_OFF_AXIS off the grid axes (arena_atlas's smear test). All of them are
rebuilt by image quilting (Efros and Freeman 2001) from SOURCE, the far side the camera faced, at the same height (so tiers, rails and
the roof line continue round): patches of PATCH m overlapping by OVERLAP m, each chosen at random among the candidates that agree
best with what is already there, a source already used at that height counting as REUSE times worse (or the same spectators and
boxes come back side by side), joined along minimum-error cuts, and feathered into the seen texels around.

The quilted texels hold crowd laid out straight on, not what the broadcast camera's rays met: stands_plain.png (atlas.json
stands.plain, 1/PLAIN_DOWN size) marks them, so the viewer reads them by height rather than through the camera; stands.crowd_top
(CROWD_TOP) tells it where the tiers end (above it builds no rows: beams and walls).

usage:  stands_quilt.py <atlas_dir> [seed]     (rewrites stands.jpg; prints the masks' shares and the detail before and after)"""

import json
import sys

import cv2
import numpy as np

PATCH, OVERLAP, FEATHER = 2.4, 0.6, 0.15  # m
STREAK_SPAN, STREAK_MIN, CROWD_TOP = (
    3.0,
    0.55,
    15.0,
)  # m window, consistency above which a corner is streaks; top of the tiers (m)
SMEAR_SPAN, SMEAR_MIN, SMEAR_OFF_AXIS = 1.0, 0.4, 25.0
SOURCE = (18.0, 64.0)  # s (m): the far side, seen straight on
ENDS_TOP = 22.0  # m: the ends are rebuilt up to here
SCOREBOARD = (31.0, 51.5, 5.5, 22.5)  # s, h (m): its painted copy, generated over (scoreboard.py): not a source
PLAIN_DOWN = 4
CANDIDATES, TOLERANCE, REUSE = 300, 0.15, 1.0  # random sources per patch; accepted error above the best; reuse penalty


def consistency(g, span):
    """Per texel: how consistently its surroundings (span texels) run in one direction, and how far that is off the grid axes (deg)."""
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    sm = lambda x: cv2.GaussianBlur(x, (0, 0), span)
    jxx, jyy, jxy = sm(gx * gx), sm(gy * gy), sm(gx * gy)
    ang = np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy)) % 90
    return np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (jxx + jyy + 1e-3), np.minimum(ang, 90 - ang)


def detail(g, m):
    """Fine over coarse detail inside mask m (as in the atlas reviews)."""
    b1, b2 = cv2.GaussianBlur(g, (0, 0), 1.5), cv2.GaussianBlur(g, (0, 0), 6)
    return float(np.abs(g - b1)[m].mean() / np.abs(b1 - b2)[m].mean())


def cut(err):
    """Minimum-error path down an overlap (rows x cols); True right of it."""
    c = err.copy()
    for i in range(1, len(c)):
        p = np.pad(c[i - 1], 1, constant_values=np.inf)
        c[i] += np.minimum(np.minimum(p[:-2], p[1:-1]), p[2:])
    path = np.empty(len(c), int)
    path[-1] = int(np.argmin(c[-1]))
    for i in range(len(c) - 2, -1, -1):
        j = path[i + 1]
        lo = max(j - 1, 0)
        path[i] = lo + int(np.argmin(c[i, lo : j + 2]))
    return np.arange(err.shape[1])[None] >= path[:, None]


def quilt(atlas, seed=0):
    meta = json.load(open(f"{atlas}/atlas.json"))
    T = meta["stands"]
    res = T["res"]
    img = cv2.imread(f"{atlas}/stands.jpg").astype(np.float32)
    H, W = img.shape[:2]
    filled = cv2.imread(f"{atlas}/stands_fill.png", 0) > 127
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows = np.arange(H)[:, None] * res
    cols = np.arange(W)[None] * res
    live = np.broadcast_to((rows > meta["band"]["top"]) & (rows < T["seen_top"]), (H, W))
    (wide, _), (cons, off) = consistency(g, STREAK_SPAN / res), consistency(g, SMEAR_SPAN / res)
    streak = (rows < CROWD_TOP) & (
        (wide > STREAK_MIN) | cv2.dilate(((cons > SMEAR_MIN) & (off > SMEAR_OFF_AXIS)).astype(np.uint8), np.ones((9, 9))) > 0
    )
    good = lambda c: (c > SOURCE[0]) & (c < SOURCE[1])
    ends = (rows < ENDS_TOP) & ~good(cols) & ~good((cols - W * res / 2) % (W * res))
    target = live & ((filled & np.roll(filled, W // 2, 1)) | streak | ends)
    sb = (cols > SCOREBOARD[0]) & (cols < SCOREBOARD[1]) & (rows > SCOREBOARD[2]) & (rows < SCOREBOARD[3])
    source = (cols > SOURCE[0]) & (cols < SOURCE[1]) & ~filled & ~streak & ~sb
    print(
        f"target {target[live].mean():.1%} of the stands below the seen top (ends {(live & ends).mean() / live.mean():.1%},"
        f" mosaic {(live & filled & np.roll(filled, W // 2, 1)).mean() / live.mean():.1%}, streaks {(live & streak).mean() / live.mean():.1%}); detail there {detail(g, target):.2f}, source {detail(g, source):.2f}",
        flush=True,
    )

    P, O = round(PATCH / res), round(OVERLAP / res)
    step = P - O
    rng = np.random.default_rng(seed)
    out, done = img.copy(), np.zeros((H, W), bool)
    r0, r1 = int(np.argmax(live.any(1))), H - int(np.argmax(live.any(1)[::-1]))
    for r in range(r0, r1, step):
        rr = slice(r, min(r + P, H))
        h = rr.stop - r
        ok = source[rr].mean(0) > 0.97  # columns whose patch is all source at these rows
        okc = np.flatnonzero(np.convolve(ok, np.ones(P), "valid") >= P)
        used = np.zeros(W)
        for c in range(0, W, step):
            cc = np.arange(c, c + P) % W
            if not target[rr][:, cc].any() or not len(okc):
                continue
            known = (~target[rr][:, cc]) | done[rr][:, cc]
            cur = out[rr][:, cc]
            cand = rng.choice(okc, min(CANDIDATES, len(okc)), replace=False)
            errs = np.array([(((img[rr, s : s + P] - cur) ** 2).sum(-1) * known).sum() / max(known.sum(), 1) for s in cand])
            score = errs / (errs.min() + 1e-6) + REUSE * np.array([used[s : s + P].mean() for s in cand])
            good = cand[score <= score.min() + TOLERANCE]
            s = int(rng.choice(good))
            patch = img[rr, s : s + P]
            used[s : s + P] += 1
            take = target[rr][:, cc].copy()
            e = ((patch - cur) ** 2).sum(-1)
            if done[rr][:, cc[:O]].any():
                take[:, :O] &= cut(e[:, :O]) | ~done[rr][:, cc[:O]]
            if h > O and done[rr][:O][:, cc].any():
                take[:O] &= cut(e[:O].T).T | ~done[rr][:O][:, cc]
            blk = out[rr][:, cc]
            blk[take] = patch[take]
            out[rr, cc] = blk
            d = done[rr][:, cc]
            d |= take
            done[rr, cc] = d
    m = cv2.GaussianBlur(target.astype(np.float32), (0, 0), FEATHER / res)[..., None] * target[..., None]
    m = np.maximum(m, cv2.erode(target.astype(np.uint8), np.ones((3, 3)))[..., None].astype(np.float32))
    res_img = (img * (1 - m) + out * m).round().clip(0, 255).astype(np.uint8)
    print(
        f"quilted {done.sum() / target.sum():.1%} of the target; detail there {detail(cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY).astype(np.float32), target):.2f}"
    )
    cv2.imwrite(f"{atlas}/stands.jpg", res_img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    plain = cv2.GaussianBlur(target.astype(np.float32), (0, 0), 0.5 / res)
    cv2.imwrite(
        f"{atlas}/stands_plain.png",
        (cv2.resize(plain, (W // PLAIN_DOWN, H // PLAIN_DOWN), interpolation=cv2.INTER_AREA) * 255).round().astype(np.uint8),
    )
    meta["stands"]["plain"] = "stands_plain.png"
    meta["stands"]["crowd_top"] = CROWD_TOP
    json.dump(meta, open(f"{atlas}/atlas.json", "w"), indent=1)


if __name__ == "__main__":
    quilt(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
