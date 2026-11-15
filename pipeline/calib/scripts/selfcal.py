"""Label-free self-calibration: shared lens / camera-position / rink parameters fitted so that the projected rink matches
image evidence (kickplate lower edge, red and blue paint) over many frames. Frames alternate fit / held-out for honest metrics.

Evidence and metrics use half-resolution distance transforms. Per-frame poses come from the panorama bundle adjustment."""

import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
import rink as RK
from masks import ridge_masks
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from pipeline import VIDEO, G

SC = 0.5
F_REF = 3050.0
CUT = 30.0
K1_BASE = float(G["cams"][0][7])
C_BASE = np.array(G["cams"][0][4:7], float)

GNAMES = [
    "s",
    "rg0",
    "rg1",
    "rg2",
    "dC0",
    "dC1",
    "dC2",
    "dCf0",
    "dCf1",
    "dCf2",
    "kb",
    "pw",
    "cxo",
    "cyo",
    "hx",
    "hy",
    "corner",
    "bl",
    "fox",
    "foy",
    "rcirc",
]


def g_init(w2name="pano300_w2", pano="pano300"):
    """Start from the aligned panorama solution (align_pano2 output)."""
    Z = np.load(f"cache/cam_{pano}.npz")
    W = np.load(f"cache/cam_{w2name}.npz")
    s = float(W["f"][0] / Z["f"][0])
    Rg = cv2.Rodrigues(cv2.Rodrigues(Z["rv"][0])[0].T @ cv2.Rodrigues(W["rv"][0])[0])[0].ravel()
    lens = Z["lens"]
    g = dict(
        s=s,
        rg0=Rg[0],
        rg1=Rg[1],
        rg2=Rg[2],
        dC0=W["C"][0] - C_BASE[0],
        dC1=W["C"][1] - C_BASE[1],
        dC2=W["C"][2] - C_BASE[2],
        dCf0=0,
        dCf1=0,
        dCf2=0,
        kb=float(W["k1"]) * lens[0],
        pw=float(lens[1]),
        cxo=float(lens[2] - 960),
        cyo=float(lens[3] - 540),
        hx=RK.LENGTH / 2,
        hy=RK.WIDTH / 2,
        corner=RK.CORNER,
        bl=RK.BLUE_LINE,
        fox=20.0,
        foy=7.0,
        rcirc=4.5,
    )
    return np.array([g[k] for k in GNAMES], float)


def project(g, f_pano, rv_pano, P):
    d = dict(zip(GNAMES, g))
    f = d["s"] * f_pano
    R = cv2.Rodrigues(rv_pano)[0] @ cv2.Rodrigues(np.array([d["rg0"], d["rg1"], d["rg2"]]))[0]
    C = C_BASE + np.array([d["dC0"], d["dC1"], d["dC2"]]) + np.array([d["dCf0"], d["dCf1"], d["dCf2"]]) * (f / F_REF - 1)
    pc = (P - C) @ R.T
    z = np.maximum(pc[:, 2], 1e-3)
    u = f * pc[:, :2] / z[:, None]
    k = d["kb"] * (F_REF / f) ** d["pw"]
    out = u * (1 + k * (u**2).sum(1, keepdims=True) / F_REF**2) + [960 + d["cxo"], 540 + d["cyo"]]
    out[pc[:, 2] < 0.5] = 1e6
    return out


def dense(poly, step=2):
    return np.concatenate([np.linspace(poly[i], poly[i + 1], step, endpoint=False) for i in range(len(poly) - 1)])


def rink_points(g):
    d = dict(zip(GNAMES, g))
    hx, hy, c = d["hx"], d["hy"], d["corner"]
    kick = dense(RK.boards(length=2 * hx, width=2 * hy, corner=c), 3)
    gl = hx - 4.0

    def vl(x):
        h = float(RK.half_width(x, length=2 * hx, width=2 * hy, corner=c))
        return np.stack([np.full(40, x), np.linspace(-h + 0.3, h - 0.3, 40)], 1)

    red = [vl(0.0), vl(gl), vl(-gl), RK.circle(0, 0, 4.07, 90)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            red.append(RK.circle(sx * d["fox"], sy * d["foy"], d["rcirc"], 90))
    blue = [vl(d["bl"]), vl(-d["bl"])]
    return kick, np.concatenate(red), np.concatenate(blue)


def to3(P2, z=0.0):
    return np.c_[P2, np.full(len(P2), z)]


# ---------------------------------------------------------------- evidence
def read_frames(start, dur, step):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        str(start),
        "-t",
        str(dur),
        "-i",
        VIDEO,
        "-vf",
        f"fps={1 / step},scale=960:540",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    n = 960 * 540 * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(540, 960, 3).copy()
    p.wait()


def evidence_of(img, ice):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (15, 110, 140), (35, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    below = np.zeros_like(m)
    below[:-3] = m[3:]
    kick = ((m > 0) & (below == 0)).astype(np.uint8)
    red, blue = ridge_masks(img, 8, 8)
    red = red * ice
    blue = blue * ice
    dt = lambda e: np.minimum(cv2.distanceTransform((1 - e).astype(np.uint8), cv2.DIST_L2, 3), CUT).astype(np.float32)
    return dt(kick), dt(red), dt(blue)


def build_evidence(name, start, dur, step, pano, g0, out):
    Z = np.load(f"cache/cam_{pano}.npz")
    times, K, R_, B_ = [], [], [], []
    kick0, _, _ = rink_points(g0)
    for k, img in enumerate(read_frames(start, dur, step)):
        t = start + k * step
        i = int(np.argmin(abs(Z["t"] - t)))
        q = project(g0, Z["f"][i], Z["rv"][i], to3(kick0)) * SC
        ice = np.zeros((540, 960), np.uint8)
        if np.isfinite(q).all() and np.abs(q).max() < 1e5:
            cv2.fillPoly(ice, [q.astype(np.int32)], 1)
        ice = cv2.erode(ice, np.ones((15, 15), np.uint8))
        a, b, c = evidence_of(img, ice)
        times.append(t)
        K.append(a.astype(np.uint8))
        R_.append(b.astype(np.uint8))
        B_.append(c.astype(np.uint8))
    np.savez_compressed(out, t=np.array(times), kick=np.array(K), red=np.array(R_), blue=np.array(B_))
    print("evidence", len(times), "frames ->", out)


# ---------------------------------------------------------------- cost
def sample(dt, q):
    ok = (q[:, 0] > 2) & (q[:, 0] < 957) & (q[:, 1] > 2) & (q[:, 1] < 537) & np.isfinite(q).all(1)
    v = np.zeros(len(q), np.float32)
    if ok.any():
        v[ok] = map_coordinates(dt, [q[ok, 1], q[ok, 0]], order=1, mode="nearest", output=np.float32)
    return v, ok


def frame_residuals(g, f, rv, E, idx, free_rink=True, pts=None):
    kick, red, blue = pts if pts is not None else rink_points(g)
    out = []
    for P, key, w in ((kick, "kick", 1.0), (red, "red", 0.7), (blue, "blue", 0.2)):
        q = project(g, f, rv, to3(P)) * SC
        v, ok = sample(E[key][idx], q)
        out.append(np.where(ok, v, 0.0) * w)
    return out


def residuals(g, F, RV, E, idxs):
    pts = rink_points(g)
    return np.concatenate([np.concatenate(frame_residuals(g, F[i], RV[i], E, i, pts=pts)) for i in idxs])


def metrics(g, F, RV, E, idxs):
    ks, rs, bs = [], [], []
    pts = rink_points(g)
    for i in idxs:
        k, r, b = frame_residuals(g, F[i], RV[i], E, i, False, pts)
        ks.append(k[k > 0])
        rs.append(r[r > 0] / 0.7)
        bs.append(b[b > 0] / 0.2)
    cat = lambda a: np.concatenate(a) if len(a) else np.zeros(1)
    k, r, b = cat(ks), cat(rs), cat(bs)
    return dict(
        kick_med=np.median(k),
        kick_p90=np.percentile(k, 90),
        red_med=np.median(r),
        red_within3=(r < 3).mean(),
        blue_med=np.median(b),
        blue_within3=(b < 3).mean(),
    )


def fmt(m):
    return " ".join(f"{k}={v:.2f}" for k, v in m.items())


def fit(g0, F, RV, E, idxs, free, loss_scale=3.0, nfev=60):
    free = np.array([GNAMES.index(n) for n in free])

    def res(x):
        g = g0.copy()
        g[free] = x
        return residuals(g, F, RV, E, idxs)

    scale = np.array(
        [
            {
                "s": 0.01,
                "cxo": 5,
                "cyo": 5,
                "hx": 0.2,
                "hy": 0.2,
                "corner": 0.5,
                "bl": 0.2,
                "fox": 0.2,
                "foy": 0.2,
                "rcirc": 0.1,
                "kb": 0.05,
                "pw": 0.3,
            }.get(GNAMES[j], 0.01 if GNAMES[j].startswith("rg") else 0.3)
            for j in free
        ]
    )
    sol = least_squares(res, g0[free], x_scale=scale, loss="soft_l1", f_scale=loss_scale, max_nfev=nfev, diff_step=1e-4)
    g = g0.copy()
    g[free] = sol.x
    return g


def frame_at_time(t):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        str(t),
        "-i",
        VIDEO,
        "-frames:v",
        "1",
        "-vf",
        "scale=960:540",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    b = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(b, np.uint8).reshape(540, 960, 3).copy()


def still_times(pano, n=80, max_px_per_s=120.0):
    """Pano frame times where the camera is nearly stationary (pixel speed of the view centre), spread evenly."""
    Z = np.load(f"cache/cam_{pano}.npz")
    t, f, rv = Z["t"], Z["f"], Z["rv"]
    Rm = [cv2.Rodrigues(r)[0] for r in rv]
    sp = np.zeros(len(t))
    for i in range(1, len(t) - 1):
        dR = Rm[i + 1] @ Rm[i - 1].T
        ang = np.linalg.norm(cv2.Rodrigues(dR)[0])
        sp[i] = (ang * f[i] / (t[i + 1] - t[i - 1])) + abs(f[i + 1] - f[i - 1]) / (t[i + 1] - t[i - 1]) * 0.3
    ok = np.flatnonzero(sp < max_px_per_s)
    pick = ok[np.linspace(0, len(ok) - 1, min(n, len(ok))).astype(int)]
    print(f"{len(ok)} of {len(t)} pano frames are near-still; using {len(pick)}")
    return t[pick], pick


def build_evidence_at(pano, g0, out, n=80):
    Z = np.load(f"cache/cam_{pano}.npz")
    times, pick = still_times(pano, n)
    kick0, _, _ = rink_points(g0)
    K, R_, B_ = [], [], []
    for t, i in zip(times, pick):
        img = frame_at_time(t)
        q = project(g0, Z["f"][i], Z["rv"][i], to3(kick0)) * SC
        ice = np.zeros((540, 960), np.uint8)
        if np.isfinite(q).all() and np.abs(q).max() < 1e5:
            cv2.fillPoly(ice, [q.astype(np.int32)], 1)
        ice = cv2.erode(ice, np.ones((15, 15), np.uint8))
        a, b, c = evidence_of(img, ice)
        K.append(a.astype(np.uint8))
        R_.append(b.astype(np.uint8))
        B_.append(c.astype(np.uint8))
    np.savez_compressed(out, t=times, idx=pick, kick=np.array(K), red=np.array(R_), blue=np.array(B_))
    print("saved", out)


def build_evidence_loc(loc, g0, out, n=100, seed=0):
    """Evidence at exact source frames, spread evenly in time (pans included); poses from per-frame localization."""
    Z = np.load(f"cache/cam_{loc}.npz")
    pick = np.linspace(40, len(Z["N"]) - 40, n).astype(int)
    kick0, _, _ = rink_points(g0)
    K, R_, B_ = [], [], []
    for i in pick:
        img = frame_at_time((Z["N"][i] - 0.5) / 25.0)
        q = project(g0, Z["f"][i], Z["rv"][i], to3(kick0)) * SC
        ice = np.zeros((540, 960), np.uint8)
        if np.isfinite(q).all() and np.abs(q).max() < 1e5:
            cv2.fillPoly(ice, [q.astype(np.int32)], 1)
        ice = cv2.erode(ice, np.ones((15, 15), np.uint8))
        a, b, c = evidence_of(img, ice)
        K.append(a.astype(np.uint8))
        R_.append(b.astype(np.uint8))
        B_.append(c.astype(np.uint8))
    np.savez_compressed(out, t=Z["t"][pick], idx=pick, kick=np.array(K), red=np.array(R_), blue=np.array(B_))
    print("saved", out)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "loc":
        build_evidence_loc(sys.argv[2], g_init(), sys.argv[3])
    if mode == "still":
        build_evidence_at("pano300", g_init(), "cache/evidence_still.npz")
    if mode == "evidence":
        g0 = g_init()
        build_evidence("t300", 925.0, 290.0, 3.7, "pano300", g0, "cache/evidence_t300.npz")
