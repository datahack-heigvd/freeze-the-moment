"""A goal's broadcast camera re-solved against the arena, its moving parts moved to stay where the broadcast shows them.

The scene's broadcast cameras (reel.json, one per exported frame) come from the calibration, smoothed over time: in a fast pan they
trail the picture by up to a frame, so the static arena drawn through them slides off the footage (up to ~40 px) while the players,
fitted through the same cameras, stay on it. Here, per frame:

  measure   the arena's own ice (atlas.json's ice atlas, made from thousands of views of the match) drawn through the frame's camera
            against the broadcast frame, the players masked out (reel masks.bin, grown by MASK_GROW px): the shift (px) of the edges
            of one onto the other, searched over SEARCH px (at a quarter size, then refined); frames matching worse than MIN_CORR
            or at the search's limit are left out and filled in from their neighbours; the shifts smoothed (a median over MEDIAN frames, then a Gaussian of
            SIGMA frames), keeping the lag's swings (a few frames long) and losing the search's pixel noise
  re-solve  each camera turned about its own centre by the shift (yaw dx / fx, pitch dy / fx), so the arena lands on the picture
  re-anchor everything placed through the old camera moved across the ice so it projects where it did: each person (their body,
            stick and eye cameras) by the move of their root at BODY_Z, the puck and the shooter's blade by theirs at the ice; the
            move is along the ice (heights kept), so skates stay on it

Writes reel.json, scene.json, body_<key>.gz and sticks.bin in place, recording what it did in scene.json's "resolved" (run once:
a scene already resolved is left alone). `resolve.py <scene dir> [--measure]` (--measure: only print the shifts).
"""

import json
import struct
import sys
import zlib

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

MASK_GROW, SEARCH, MIN_CORR = 15, 48, 0.25  # px, px, normalised correlation
MEDIAN, SIGMA = 5, 1.5  # frames
BODY_Z = 0.9  # m: the height a body's move is taken at (its middle, what the picture aligns)
BODY_STEP, BODY_CHUNK = 2, 25  # scene_export.py's body packing


def K(fx, W, H):
    return np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])


def project(cam, W, H, X):
    fx, R, C = cam
    p = (np.asarray(X, float) - C) @ R.T
    return p[..., :2] / p[..., 2:3] * fx + (W / 2, H / 2)


def on_plane(cam, W, H, u, z):
    """The point at height z the camera sees at pixel u."""
    fx, R, C = cam
    d = np.c_[(np.asarray(u, float) - (W / 2, H / 2)) / fx, np.ones(len(u))] @ R
    t = (z - C[2]) / d[:, 2]
    return C + d * t[:, None]


def broadcast(d):
    cap = cv2.VideoCapture(f"{d}/reel/broadcast.mp4")
    out = []
    while True:
        ok, im = cap.read()
        if not ok:
            return out
        out.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))


def outlines(d, W, H):
    b = open(f"{d}/reel/masks.bin", "rb").read()
    n = struct.unpack_from("<I", b, 0)[0]
    off = struct.unpack_from(f"<{n + 1}I", b, 4)
    data = np.frombuffer(b, "<i2", offset=4 + 4 * (n + 1))
    for f in range(n):
        m = np.zeros((H, W), np.uint8)
        v = data[off[f] : off[f + 1]]
        i = 0
        while i < len(v):
            k = int(v[i])
            cv2.fillPoly(m, [np.round(v[i + 1 : i + 1 + 2 * k].reshape(-1, 2) / 8).astype(np.int32)], 255)
            i += 1 + 2 * k
        yield cv2.dilate(m, np.ones((MASK_GROW, MASK_GROW), np.uint8))


def edges(g):
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 2)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def measure(d, cams, W, H):
    """Per frame (dx, dy, corr): where the broadcast shows the ice the atlas drawn through the camera puts at the origin."""
    A = json.load(open(f"{d}/atlas.json"))["ice"]
    ice = cv2.imread(f"{d}/{A['file']}", cv2.IMREAD_GRAYSCALE)
    T = np.array([[A["res"], 0, A["x0"]], [0, A["res"], A["y0"]], [0, 0, 0], [0, 0, 1.0]])
    r, s = SEARCH, 4
    out = []
    for f, (g, mask) in enumerate(zip(broadcast(d), outlines(d, W, H), strict=False)):
        fx, R, C = cams[f]
        Hm = K(fx, W, H) @ np.c_[R, -R @ C] @ T
        drawn = cv2.warpPerspective(ice, Hm, (W, H), flags=cv2.INTER_AREA)
        seen = cv2.warpPerspective(np.full(ice.shape, 255, np.uint8), Hm, (W, H), flags=cv2.INTER_NEAREST)
        m = cv2.erode(((seen > 0) & (mask == 0)).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
        a, b = edges(drawn) * m, edges(g) * m
        small = [cv2.resize(x, (W // s, H // s), interpolation=cv2.INTER_AREA) for x in (a, b)]
        res = cv2.matchTemplate(small[1], small[0][r // s : -r // s, r // s : -r // s], cv2.TM_CCOEFF_NORMED)
        _, _, _, (x, y) = cv2.minMaxLoc(res)
        cx, cy = (x - r // s) * s, (y - r // s) * s
        t = a[r:-r, r:-r]
        y0, x0 = max(r + cy - 8, 0), max(r + cx - 8, 0)
        res = cv2.matchTemplate(b[y0 : y0 + t.shape[0] + 16, x0 : x0 + t.shape[1] + 16], t, cv2.TM_CCOEFF_NORMED)
        _, v, _, (x, y) = cv2.minMaxLoc(res)
        out.append((x0 + x - r, y0 + y - r, v))
    return np.array(out, float)


def smoothed(m):
    """The shifts with the weak matches (or at the search's limit: not found) filled in from their neighbours, median filtered,
    then smoothed."""
    ok = (m[:, 2] >= MIN_CORR) & (np.abs(m[:, :2]).max(1) < SEARCH)
    f = np.arange(len(m))
    xy = np.stack([np.interp(f, f[ok], m[ok, k]) for k in (0, 1)], 1)
    return gaussian_filter1d(median_filter(xy, size=(MEDIAN, 1), mode="nearest"), SIGMA, axis=0, mode="nearest")


def turned(cam, dx, dy):
    """The camera turned about its centre so what it saw at a pixel it sees (dx, dy) px further on."""
    fx, R, C = cam
    a, b = np.arctan(dx / fx), np.arctan(dy / fx)
    yaw = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    pitch = np.array([[1, 0, 0], [0, np.cos(b), np.sin(b)], [0, -np.sin(b), np.cos(b)]])
    return fx, pitch @ yaw @ R, C


def moves(old, new, W, H, X):
    """Per frame, the move along the ice that keeps points X [frames, 3] where the old cameras saw them through the new."""
    out = np.zeros_like(X)
    for f in range(len(X)):
        u = project(old[f], W, H, X[f : f + 1])
        out[f] = on_plane(new[f], W, H, u, X[f, 2])[0] - X[f]
    out[:, 2] = 0
    return out


def unpack_body(b, frames, V):
    raw = zlib.decompress(b, 31)
    Y, o = [], 0
    for c in range(0, frames, BODY_CHUNK):
        n = min(BODY_CHUNK, frames - c) * V * 3
        d = np.frombuffer(raw, np.uint8, n, o).astype(np.uint16) | (np.frombuffer(raw, np.uint8, n, o + n).astype(np.uint16) << 8)
        o += 2 * n
        d = d.view(np.int16).astype(np.int32).reshape(-1, V, 3)
        C = np.empty_like(d)
        for k in range(len(d)):
            C[k] = d[k] + (C[k - 1] if k >= 1 else 0) * (2 if k >= 2 else 1) - (C[k - 2] if k >= 2 else 0)
        Y.append(C)
    return np.concatenate(Y) * BODY_STEP


def pack_body(mm):
    Y = np.round(mm / BODY_STEP).astype(np.int32)
    z = zlib.compressobj(6, zlib.DEFLATED, 31)
    out = []
    for c in range(0, len(Y), BODY_CHUNK):
        C = Y[c : c + BODY_CHUNK]
        d = np.concatenate([C[:1], C[1:2] - C[:1], C[2:] - 2 * C[1:-1] + C[:-2]]).astype(np.int16).view(np.uint16)
        out += [
            z.compress((d & 255).astype(np.uint8).tobytes() + (d >> 8).astype(np.uint8).tobytes()),
            z.flush(zlib.Z_SYNC_FLUSH),
        ]
    return b"".join(out + [z.flush()])


def main():
    d = sys.argv[1]
    reel = json.load(open(f"{d}/reel/reel.json"))
    W, H = reel["width"], reel["height"]
    old = [(c["fx"], np.array(c["R"], float), np.array(c["C"], float)) for c in reel["cams"]]
    m = measure(d, old, W, H)
    shift = smoothed(m)
    print(
        f"{d}: shift px mean |dx| {np.abs(shift[:, 0]).mean():.1f} |dy| {np.abs(shift[:, 1]).mean():.1f}, max "
        f"{np.abs(shift).max(0).round(1).tolist()}, left out {((m[:, 2] < MIN_CORR) | (np.abs(m[:, :2]).max(1) >= SEARCH)).sum()}"
    )
    if "--measure" in sys.argv:
        for f in range(0, len(m), 10):
            print(
                f"  f{f:3d} measured {m[f, 0]:+4.0f} {m[f, 1]:+4.0f} ({m[f, 2]:.2f})  smoothed {shift[f, 0]:+5.1f} {shift[f, 1]:+5.1f}"
            )
        return
    S = json.load(open(f"{d}/scene.json"))
    if S.get("resolved"):
        print(f"{d}: resolved already")
        return
    n = S["frames"]
    new = [turned(old[f], *shift[f]) for f in range(len(old))] + old[len(shift) :]
    cams = new[:n] + [new[-1]] * max(0, n - len(new))
    olds = old[:n] + [old[-1]] * max(0, n - len(old))

    def at(xy, z):
        return np.c_[np.asarray(xy, float), np.full(len(xy), z)]

    keys = {p["key"]: k for k, p in enumerate(S["people"])}
    sticks = np.fromfile(f"{d}/sticks.bin", np.float32).reshape(len(S["people"]), n, 2, 3)
    B = S["body"]
    for k, p in enumerate(S["people"]):
        root = np.asarray(p["root"], float) / 1000
        mv = moves(olds, cams, W, H, at(root, BODY_Z))
        p["root"] = np.round((root + mv[:, :2]) * 1000).astype(int).tolist()
        sticks[k] += mv[:, None, :].astype(np.float32)
        path = f"{d}/body_{p['key']}.gz"
        body = unpack_body(open(path, "rb").read(), B["frames"], S["vertices"])
        body = body + np.round(mv[B["first"] : B["first"] + B["frames"], None, :] * 1000).astype(body.dtype)
        open(path, "wb").write(pack_body(body))
        p["_move"] = mv
    for h in S["heads"]:
        mv = S["people"][keys[h["key"]]]["_move"]
        for f, c in enumerate(h["cams"]):
            c["C"] = np.round(np.asarray(c["C"]) + mv[min(f, n - 1)], 4).tolist()
    for p in S["people"]:
        del p["_move"]
    P = S["puck"]
    P["xy"] = np.round(np.asarray(P["xy"]) + moves(olds, cams, W, H, at(P["xy"], 0))[:, :2], 3).tolist()
    shot = P.get("shot")
    if shot and shot.get("blade"):
        f = shot["release"]
        shot["blade"] = np.round(
            np.asarray(shot["blade"]) + moves(olds[f : f + 1], cams[f : f + 1], W, H, at([shot["blade"]], 0))[0, :2], 3
        ).tolist()
    sticks.tofile(f"{d}/sticks.bin")
    reel["cams"] = [dict(fx=round(float(c[0]), 3), R=np.round(c[1], 6).tolist(), C=np.round(c[2], 4).tolist()) for c in new]
    json.dump(reel, open(f"{d}/reel/reel.json", "w"))
    S["resolved"] = {"shift_px": np.round(shift, 2).tolist()}
    json.dump(S, open(f"{d}/scene.json", "w"))
    print(f"{d}: resolved")


if __name__ == "__main__":
    main()
