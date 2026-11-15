"""How good each planned shot of an exported scene looks, per frame, from the data alone (no rendering):

  crowd     share of the frame that is not the ice inside the boards: rays through a GRID of the frame, those that reach the ice
            outside the rink or never come down (stands, roof). The arena behind the broadcast camera exists only as the far side
            copied round, and close stands smear, so a frame full of crowd looks worst.
  framed    the scorer (chest) and the puck inside the frame's middle FRAMED share.
  evidence  how much of what is in view the broadcast actually saw: each person in frame weighted by their size on screen, by
            whether they were seen (and lined up with their outline) at that frame, and by how close the view of them is to the
            broadcast camera's ((1 + cos) / 2 of the angle between the two directions to them: from behind it, 0). Bodies,
            textures and the arena are only observed from the broadcast's side; elsewhere they are held, borrowed or mirrored.

Prints per shot the mean, the worst frame and the share of frames past the limits (CROWD, EVIDENCE).

usage:  shotcheck.py <scene dir>...   (e.g. scenes/musa)"""

import json
import sys

import numpy as np

GRID = (32, 18)
FRAMED = 0.9
CROWD, EVIDENCE = 0.5, 0.5  # a frame more than half crowd, or with under half of what it shows seen, fails


def rink_sd(S, xy):
    r = S["rink"]
    q = np.abs(xy) - [r["hx"] - r["corner"], r["hy"] - r["corner"]]
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0) - r["corner"]


def crowd(S, cam):
    """Share of GRID rays through the frame of camera cam (fx, R rows rink -> camera, C) not ending on the ice inside the rink."""
    W, H = S["width"], S["height"]
    gw, gh = GRID
    u, v = np.meshgrid((np.arange(gw) + 0.5) * W / gw - W / 2, (np.arange(gh) + 0.5) * H / gh - H / 2)
    d = np.stack([u.ravel() / cam["fx"], v.ravel() / cam["fx"], np.ones(gw * gh)], 1) @ np.asarray(cam["R"])
    C = np.asarray(cam["C"])
    down = d[:, 2] < -1e-6
    t = np.where(down, -C[2] / np.where(down, d[:, 2], -1), np.inf)
    hit = C[:2] + t[:, None] * d[:, :2]
    return float((~down | (rink_sd(S, np.where(down[:, None], hit, 0)) > 0)).mean())


def project(S, cam, X):
    p = (np.asarray(X) - cam["C"]) @ np.asarray(cam["R"]).T
    return np.c_[cam["fx"] * p[:, 0] / p[:, 2] + S["width"] / 2, cam["fx"] * p[:, 1] / p[:, 2] + S["height"] / 2], p[:, 2]


def check(d):
    S = json.load(open(f"{d}/scene.json"))
    f0, n = S["frame0"], S["frames"]
    W, H = S["width"], S["height"]
    root = np.array([p["root"] for p in S["people"]], float) / 1000
    seen = np.array([p["seen"] for p in S["people"]], bool)
    Cb = np.asarray(S["source"]["camera"])
    scorer = S["anatomy"]["scorer"]
    sco = [p["key"] for p in S["people"]].index(scorer)
    puck = np.asarray(S["puck"]["xy"], float)
    m = (1 - FRAMED) / 2
    inner = lambda uv, z: (
        (z > 0.3) & (uv[:, 0] > W * m) & (uv[:, 0] < W * (1 - m)) & (uv[:, 1] > H * m) & (uv[:, 1] < H * (1 - m))
    )
    print(d)
    for name, sh in S["shots"].items():
        rows = []
        for cam, t in zip(sh["cams"], sh["frames"]):
            f = min(max(int(round(t - f0)), 0), n - 1)
            X = np.c_[root[:, f], np.full(len(root), 1.0)]
            uv, z = project(S, cam, X)
            inf = (z > 0.3) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            size = np.where(inf, cam["fx"] * 1.8 / np.maximum(z, 0.3), 0)
            b, c = X - Cb, X - np.asarray(cam["C"])
            cos = (b * c).sum(1) / np.linalg.norm(b, axis=1) / np.linalg.norm(c, axis=1)
            own = sh.get("own")
            keep = np.array([p["key"] != own for p in S["people"]])  # a POV's own body is not drawn
            w = size * keep
            ev = float((w * seen[:, f] * (1 + cos) / 2).sum() / w.sum()) if w.sum() > 0 else 1.0
            fr = bool(inner(uv[sco : sco + 1], z[sco : sco + 1])[0]) if own != scorer else True
            fp = bool(inner(*project(S, cam, [[*puck[f], 0.05]]))[0])
            rows.append((crowd(S, cam), ev, fr, fp))
        a = np.array(rows, float)
        print(
            f"  {name:11s} crowd {a[:, 0].mean():.2f} (worst {a[:, 0].max():.2f}, {(a[:, 0] > CROWD).mean():4.0%} over)   "
            f"evidence {a[:, 1].mean():.2f} (worst {a[:, 1].min():.2f}, {(a[:, 1] < EVIDENCE).mean():4.0%} under)   "
            f"scorer framed {a[:, 2].mean():4.0%}  puck framed {a[:, 3].mean():4.0%}"
        )


if __name__ == "__main__":
    for d in sys.argv[1:]:
        check(d.rstrip("/"))
