"""The centre-hung scoreboard: a cube over centre ice instead of a picture painted on the stands.

The broadcast camera sees the scoreboard's face against the far stands, so the atlas holds it there (stands atlas, s 32.9 to 49.4 m,
h 6.4 to 21.3 m), and the fill copied it half a turn round onto the camera's own stands. Every row of the face spans the same width
in 3D on a vertical plane, which fixes its depth: rays from the camera centre through the face's edges in the atlas meet a vertical
plane y = -W/2 at a constant width W (a cube): 7.8 m, from z 9.8 to 13.3 m, with a ring tapering below it to z 8.5 m.

  face:    the face rectified from the atlas onto that plane (scoreboard.jpg, 1 cm per texel, columns left to right as seen from
           in front, rows top down) and the cube's size (atlas.json "scoreboard"); the viewer puts the face on all four sides. The
           video screen showed the live feed, which the atlas averaged into smears and the refine into swirls: it is drawn dark.
  inpaint: both painted copies removed from the stands atlas: Qwen-Image-2.1 generates them from scratch with the stands around
           them as context (atlas_refine.diffuse), on a canvas MARGIN either side.
  clear:   planned cameras (scene.json shots, planned before the cube existed) that pass over centre ice within CLEAR of the cube are
           lowered to CLEAR below it, the drop eased in and out over EASE frames, and turned (keeping their roll) to look at the same
           ice point as before, so the framing holds.

usage:  scoreboard.py face <atlas_dir>       (reads stands.jpg before inpaint)
        scoreboard.py inpaint <atlas_dir>    (image venv, GPU; rewrites stands.jpg)
        scoreboard.py clear <scene_dir>...   (rewrites scene.json's planned cameras)"""

import json
import sys

import numpy as np
from paths import SCENES
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
EDGES = [
    (10.0, 33.85, 48.65),
    (14.0, 33.35, 49.03),
    (18.0, 32.88, 49.48),
    (21.0, 32.45, 49.88),
]  # (h, s left, s right) of the body
RING = (7.0, 35.43, 47.00)  # the ring's lowest row
H_BOTTOM, H_BODY, H_TOP = 6.4, 9.8, 21.3  # atlas h of the ring's bottom, the body's bottom, the top
HALF_TURN = 82.35  # s offset of the near copy (half the perimeter, by correlation)
CUT = (32.2, 50.3, 6.0, 22.0)  # s and h of the far copy removed (frame plus 0.3 to 0.7 m)
TEX = 0.01  # m per face texel
SCREEN = (1.32, 6.56, 0.66, 3.52)  # m on the face, from its top left: the video screen's left, right, top, bottom
MARGIN = 6.0  # m of stands either side of a removed copy on its canvas
CLEAR, EASE = 1.0, 6  # m kept between a planned camera and the cube; frames of easing


def geometry(atlas):
    """Rink and camera of an atlas dir; maps from atlas (s, h) on the far side to 3D and from 3D back through the camera."""
    meta = json.load(open(f"{atlas}/atlas.json"))
    C = np.array(meta["camera"])
    r = json.load(open(f"{SCENES}/musa/scene.json"))["rink"]
    RA, RB, RC, K = r["hx"] - r["corner"], r["hy"] - r["corner"], r["corner"], np.hypot(1, r["slope"])
    s_mid = RA + RB + RC * np.pi / 2

    def point(s, h):
        sd = r["ledge"] + (h - r["wall"] - r["ledge"]) / K
        return np.stack([s_mid - s, r["hy"] + sd, r["wall"] + r["slope"] * (sd - r["ledge"])], -1)

    def atlas_sh(P):  # 3D points -> the atlas (s, h) their camera ray meets on the slope
        d = P - C
        c0 = r["wall"] + r["slope"] * (C[1] - r["hy"] - r["ledge"]) - C[2]
        t = c0 / (d[..., 2] - r["slope"] * d[..., 1])
        Q = C + d * t[..., None]
        sd = Q[..., 1] - r["hy"]
        return s_mid - Q[..., 0], r["wall"] + r["ledge"] + (sd - r["ledge"]) * K

    def on_plane(s, h, y):  # the camera ray through atlas (s, h) at the plane y
        P = point(s, h)
        return C + (P - C) * ((y - C[1]) / (P[..., 1] - C[1]))[..., None]

    return meta, on_plane, atlas_sh


def face(atlas):
    meta, on_plane, atlas_sh = geometry(atlas)
    width = lambda y: np.mean([on_plane(a, h, y)[0] - on_plane(b, h, y)[0] for h, a, b in EDGES])
    y = -4.0
    for _ in range(20):
        y = -width(y) / 2  # a cube: the face's width is the depth
    W = width(y)
    x0 = np.mean([(on_plane(a, h, y)[0] + on_plane(b, h, y)[0]) / 2 for h, a, b in EDGES])
    z = [float(on_plane(41.0, h, y)[2]) for h in (H_BOTTOM, H_BODY, H_TOP)]
    ring = float(on_plane(RING[1], RING[0], y)[0] - on_plane(RING[2], RING[0], y)[0])
    img = np.asarray(Image.open(f"{atlas}/stands.jpg").convert("RGB")).astype(np.float32)
    res = meta["stands"]["res"]
    xs = x0 + np.arange(-W / 2, W / 2, TEX) + TEX / 2
    zs = np.arange(z[2], z[0], -TEX) - TEX / 2
    X, Z = np.meshgrid(xs, zs)
    s, h = atlas_sh(np.stack([X, np.full_like(X, y), Z], -1))
    u, v = s / res - 0.5, h / res - 0.5
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = (u - u0)[..., None], (v - v0)[..., None]
    at = lambda a, b: img[np.clip(b, 0, img.shape[0] - 1), np.clip(a, 0, img.shape[1] - 1)]
    tex = (
        at(u0, v0) * (1 - fu) * (1 - fv)
        + at(u0 + 1, v0) * fu * (1 - fv)
        + at(u0, v0 + 1) * (1 - fu) * fv
        + at(u0 + 1, v0 + 1) * fu * fv
    )
    u, v = np.meshgrid(np.arange(tex.shape[1]) * TEX, np.arange(tex.shape[0]) * TEX)
    l, r, t, b = SCREEN
    inside = np.clip(np.minimum.reduce([u - l, r - u, v - t, b - v]) / 0.03, 0, 1)[..., None]
    tex = tex * (1 - inside) + (np.array([10.0, 13, 20]) + 10 * (1 - (v - t) / (b - t))[..., None]) * inside
    Image.fromarray(tex.round().clip(0, 255).astype(np.uint8)).save(f"{atlas}/scoreboard.jpg", quality=93)
    meta["scoreboard"] = dict(
        file="scoreboard.jpg", x=round(float(x0), 3), width=round(float(W), 3), ring=round(ring, 3), z=[round(v, 3) for v in z]
    )
    json.dump(meta, open(f"{atlas}/atlas.json", "w"), indent=1)
    print("scoreboard:", meta["scoreboard"], f"face plane y {y:.2f}, texture {tex.shape[1]}x{tex.shape[0]}")


def inpaint(atlas):
    from arena import atlas_refine as AR

    meta = json.load(open(f"{atlas}/atlas.json"))
    res = meta["stands"]["res"]
    img = np.asarray(Image.open(f"{atlas}/stands.jpg").convert("RGB")).copy()
    H, W = img.shape[:2]  # stored h down: row = h / res
    top = int(meta["stands"]["seen_top"] / res) // 32 * 32
    pipe = AR.load()
    for n, ds in enumerate((0.0, HALF_TURN)):
        s0, s1, h0, h1 = CUT[0] + ds, CUT[1] + ds, CUT[2], CUT[3]
        c0 = int((s0 - MARGIN) / res)
        cols = (c0 + np.arange(round((s1 - s0 + 2 * MARGIN) / res / 32) * 32)) % W
        hole = np.zeros((top, len(cols)), np.float32)
        ss = cols * res
        hole[int(h0 / res) : int(h1 / res)] = ((ss >= s0) & (ss < s1))[None]
        m = np.clip(AR.blur(hole, 0.3 / res) * 2, 0, 1)
        m[hole > 0] = 1
        canvas = img[:top, cols][::-1].copy()  # upright for the model
        gen = AR.diffuse(pipe, canvas, m[::-1].copy(), AR.PROMPTS["stands"], 70 + n, True)[::-1]
        img[:top, cols] = (img[:top, cols] * (1 - m[..., None]) + gen * m[..., None]).round().clip(0, 255).astype(np.uint8)
        print(f"removed the copy at s {s0:.1f} to {s1:.1f} m on a {len(cols)}x{top} canvas", flush=True)
    Image.fromarray(img).save(f"{atlas}/stands.jpg", quality=93)


def clear(scene):
    S = json.load(open(f"{scene}/scene.json"))
    sb = json.load(open(f"{scene}/atlas.json"))["scoreboard"]
    for name, sh in S["shots"].items():
        C = np.array([c["C"] for c in sh["cams"]], float)
        R = np.array([c["R"] for c in sh["cams"]], float)
        over = np.maximum(np.abs(C[:, 0] - sb["x"]), np.abs(C[:, 1])) < sb["width"] / 2 + CLEAR
        need = np.where(over, np.maximum(C[:, 2] - (sb["z"][0] - CLEAR), 0), 0)
        if not need.any():
            continue
        drop = np.array([need[max(k - EASE, 0) : k + EASE + 1].max() for k in range(len(need))])  # held EASE frames each side,
        w = np.hanning(2 * EASE + 3)[1:-1]
        drop = np.maximum(np.convolve(drop, w / w.sum(), "same"), need)  # then eased
        for k in np.flatnonzero(drop > 0):
            f = R[k, 2]
            t = -C[k, 2] / f[2] if f[2] < -0.05 else 20.0
            T = C[k] + t * f  # the ice point it looks at
            C[k, 2] -= drop[k]
            g = (T - C[k]) / np.linalg.norm(T - C[k])
            v = np.cross(f, g)
            c = f @ g
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            turn = np.eye(3) + K + K @ K / (1 + c)
            R[k] = R[k] @ turn.T  # rows (camera axes) turned
            sh["cams"][k]["C"], sh["cams"][k]["R"] = C[k].tolist(), R[k].tolist()
        print(scene, name, f"lowered {int((drop > 0).sum())} frames, by up to {drop.max():.2f} m")
    json.dump(S, open(f"{scene}/scene.json", "w"))


if __name__ == "__main__":
    {
        "face": lambda: face(sys.argv[2]),
        "inpaint": lambda: inpaint(sys.argv[2]),
        "clear": lambda: [clear(d) for d in sys.argv[2:]],
    }[sys.argv[1]]()
