"""Smooth, physically constrained camera trajectories from the per-frame localization, scored against the rink markings.

The broadcast camera sits on a fixed head: its centre is constant, the lens distortion follows the zoom, and the head pans, tilts and
zooms smoothly. Per-frame solves add independent noise and occasional spikes on top. Here every rotation is split into pan, tilt and
roll about the mean view; roll, which a head cannot produce, is replaced by a smooth function of pan and tilt (a slightly unlevel
head); pan, tilt and log focal length are each fitted with a robust Whittaker smoother (penalised third differences, so acceleration
stays smooth through fast pans), iteratively reweighted so spike frames lose their pull. Frames without a pose are interpolated.

The score: the rink's red and blue markings projected with a camera, against thin red and blue line pixels detected in the frame
(logos removed by a morphological opening), as the median distance in pixels over marking points near a detection.

export_ar.py smooths every exported camera with LAM. This script scores the per-frame solves against smoothing strengths.
usage (from pipeline/calib/):  camsmooth.py <segment> [lam,...]     e.g. camsmooth.py c130944 3,20,100   (on store/ar/<segment>/camera_raw.bin)
output: cache/camsmooth_<segment>.json, the scores of the per-frame cameras and of each smoothing strength"""

import json
import subprocess
import sys

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "scripts")
import rink

VIDEO = "../../store/data/20251231_115035_SpenglerCup_Kam1 31.12.25 Game 11.mp4"
AR = "../../store/ar"
FPS, STEP = 25, 10  # scored frames: every 10th
LAM = 100.0  # passes head motion below ~1.9 Hz; stronger starts rejecting real fast moves


def whittaker(y, w, lam, order=3):
    """argmin_z sum w (y - z)^2 + lam |D^order z|^2."""
    n = len(y)
    D = sparse.eye(n, format="csc")
    for _ in range(order):
        D = D[1:] - D[:-1]
    return spsolve((sparse.diags(w) + lam * D.T @ D).tocsc(), w * y)


def robust_smooth(y, w0, lam, px, iters=6, c=4.685, floor=0.5):
    """Whittaker smoothing with Tukey reweighting of residuals. Residuals are measured in image pixels (`px` per unit of y) and
    their scale, from the MAD, never drops below `floor` px, so sub-pixel noise does not get genuine motion rejected as spikes."""
    w = w0.copy()
    z = whittaker(y, w, lam)
    for _ in range(iters):
        r = (y - z) * px
        s = max(1.4826 * np.median(np.abs(r[w0 > 0])), floor)
        u = r / (c * s)
        w = w0 * np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0)
        z = whittaker(y, w, lam)
    return z, w


def level_axes(d):
    """The level camera axes for viewing directions d: right is horizontal, down completes the frame."""
    right0 = np.cross(d, [0, 0, 1.0])
    right0 /= np.maximum(np.linalg.norm(right0, axis=1, keepdims=True), 1e-9)
    return right0, np.cross(d, right0)


def angles(R):
    """World->camera rotations (rows: camera right, down, forward) -> pan and tilt of the viewing direction, and roll: the turn of
    the camera's right axis about the viewing direction away from the level right axis."""
    d = R[:, 2]
    right0, down0 = level_axes(d)
    roll = np.arctan2((R[:, 0] * down0).sum(1), (R[:, 0] * right0).sum(1))
    return np.arctan2(d[:, 1], d[:, 0]), np.arcsin(np.clip(d[:, 2], -1, 1)), roll


def rotation(pan, tilt, roll):
    d = np.stack([np.cos(tilt) * np.cos(pan), np.cos(tilt) * np.sin(pan), np.sin(tilt)], 1)
    right0, down0 = level_axes(d)
    right = np.cos(roll)[:, None] * right0 + np.sin(roll)[:, None] * down0
    return np.stack([right, np.cross(d, right), d], 1)


def smooth_cameras(cam, lam):
    ok = np.abs(cam).sum(1) > 0
    R = cam[:, 2:11].reshape(-1, 3, 3).copy()
    R[~ok] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]  # a level view for frames without a pose
    pan, tilt, roll = angles(R)
    w0 = ok.astype(float)
    lf = np.log(np.where(ok, cam[:, 0], 1.0))
    pan = np.unwrap(pan)
    pan_s, tilt_s, lf_s = pan.copy(), tilt.copy(), lf.copy()
    wp, wt, wf = w0.copy(), w0.copy(), w0.copy()
    # each continuous shot of this camera separately: the smoother must not bridge a cut to another camera
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < 8:
            continue
        sl = slice(a, b)
        f = np.median(cam[sl, 0])
        pan_s[sl], wp[sl] = robust_smooth(pan[sl], w0[sl], lam, f * np.cos(np.median(tilt[sl])))
        tilt_s[sl], wt[sl] = robust_smooth(tilt[sl], w0[sl], lam, f)
        lf_s[sl], wf[sl] = robust_smooth(lf[sl], w0[sl], lam, 1100.0)  # a zoom step moves the frame corners by ~1100 px dlog f
    # roll: a head that is not level rolls the view as it pans and tilts; a low-order model of pan and tilt, fitted robustly
    A = np.c_[np.ones_like(pan_s), np.sin(pan_s), np.cos(pan_s) - 1, tilt_s, pan_s * tilt_s]
    wr = w0.copy()
    for _ in range(6):
        coef = np.linalg.lstsq(A * wr[:, None], roll * wr, rcond=None)[0]
        r = roll - A @ coef
        s = 1.4826 * np.median(np.abs(r[w0 > 0])) + 1e-9
        wr = w0 * (np.abs(r) < 3 * s)
    roll_s = A @ coef
    out = cam.copy()
    out[:, 0] = np.exp(lf_s)
    out[:, 2:11] = rotation(pan_s, tilt_s, roll_s).reshape(-1, 9)
    # lens distortion follows the zoom: k = kb (F/f)^pw, refitted on the frames with a pose
    lk = np.polyfit(np.log(cam[ok, 0]), np.log(cam[ok, 1]), 1)
    out[:, 1] = np.exp(np.polyval(lk, np.log(out[:, 0])))
    # the centre moves along the optical axis as the lens zooms: C = a + b f, refitted on the frames with a pose
    Cf = np.linalg.lstsq(np.c_[np.ones(ok.sum()), cam[ok, 0]], cam[ok, 11:14], rcond=None)[0]
    out[:, 11:14] = np.c_[np.ones(len(out)), out[:, 0]] @ Cf
    out[~ok] = 0
    spikes = (wp < 0.05) | (wt < 0.05) | (wf < 0.05)
    return out, dict(
        roll_model_resid_deg=float(np.degrees(np.std((roll - roll_s)[ok]))),
        spike_frames=int((spikes & ok).sum()),
        no_pose_frames=int((~ok).sum()),
    )


def project(cam, P, cx, cy, fref):
    f, k, R, C = cam[0], cam[1], cam[2:11].reshape(3, 3), cam[11:14]
    pc = (P - C) @ R.T
    front = pc[:, 2] > 1
    uv = f * pc[:, :2] / np.where(front, pc[:, 2], 1)[:, None]
    return uv * (1 + k * (uv**2).sum(1, keepdims=True) / fref**2) + [cx, cy], front


def markings():
    """Red and blue marking points (rink frame, z=0), every 20 cm."""
    L = rink.lines()
    pts = {"red": [], "blue": []}
    for name, poly in L.items():
        if name == "boards":
            continue
        seg = np.diff(poly, axis=0)
        n = np.maximum((np.linalg.norm(seg, axis=1) / 0.2).astype(int), 1)
        dense = np.concatenate([poly[i] + seg[i] * np.linspace(0, 1, n[i], endpoint=False)[:, None] for i in range(len(seg))])
        pts["blue" if name.startswith("blue") else "red"].append(dense)
    return {k: np.c_[np.concatenate(v), np.zeros(len(np.concatenate(v)))] for k, v in pts.items()}


def colour_maps(img):
    """Redness and blueness of every pixel against white ice, lightly blurred."""
    b, g, r = [img[..., i].astype(np.float32) for i in range(3)]
    return {"red": cv2.GaussianBlur(r - (g + b) / 2, (5, 5), 0), "blue": cv2.GaussianBlur(b - r, (5, 5), 0)}


def score(cam, img, marks, T, reach=24, contrast=12.0):
    """Median distance (px) between projected markings and the marking actually in the frame: along each projected point's normal,
    the sub-pixel peak of the marking's colour within `reach`, kept only where the peak stands out (players, logos, glare drop out)."""
    maps = colour_maps(img)
    off = []
    steps = np.arange(-reach, reach + 1, dtype=np.float32)
    for name, P in marks.items():
        uv, front = project(cam, P, T["cx"], T["cy"], T["fref"])
        tang = np.gradient(uv, axis=0)
        nrm = np.c_[-tang[:, 1], tang[:, 0]]
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        ok = front & (uv[:, 0] > reach + 2) & (uv[:, 0] < 1918 - reach) & (uv[:, 1] > reach + 2) & (uv[:, 1] < 1078 - reach)
        ok &= np.linalg.norm(tang, axis=1) > 0.5  # projected spacing: skip degenerate points
        if ok.sum() < 20:
            continue
        q = uv[ok, None, :] + steps[None, :, None] * nrm[ok, None, :]
        prof = cv2.remap(maps[name], q[..., 0].astype(np.float32), q[..., 1].astype(np.float32), cv2.INTER_LINEAR)
        k = prof.argmax(1)
        peak = prof[np.arange(len(k)), k]
        good = (peak - np.median(prof, 1) > contrast) & (k > 0) & (k < len(steps) - 1)
        kk = np.clip(k, 1, len(steps) - 2)
        y0, y1, y2 = (prof[np.arange(len(k)), kk + d] for d in (-1, 0, 1))
        sub = kk + 0.5 * (y0 - y2) / np.where(np.abs(y0 - 2 * y1 + y2) > 1e-6, y0 - 2 * y1 + y2, 1e-6)
        off.append(np.abs(steps[0] + sub[good]))
    off = np.concatenate(off) if off else np.zeros(0)
    return (float(np.median(off)), len(off)) if len(off) > 40 else (np.nan, 0)


def frames(n0, count):
    """Every STEP-th source frame of the segment, decoded in one pass."""
    p = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{(n0 - 0.5) / FPS:.4f}",
            "-i",
            VIDEO,
            "-t",
            f"{count / FPS:.3f}",
            "-vf",
            f"select=not(mod(n\\,{STEP}))",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    while True:
        buf = p.stdout.read(1920 * 1080 * 3)
        if len(buf) < 1920 * 1080 * 3:
            break
        yield np.frombuffer(buf, np.uint8).reshape(1080, 1920, 3)


def jitter(cam):
    ok = np.abs(cam).sum(1) > 0
    R = cam[:, 2:11].reshape(-1, 3, 3).copy()
    R[~ok] = np.eye(3)
    rv = Rot.from_matrix(R).as_rotvec()
    lf = np.log(np.maximum(cam[:, 0], 1))
    dev = np.maximum(
        np.linalg.norm(rv[1:-1] - (rv[:-2] + rv[2:]) / 2, axis=1) * cam[1:-1, 0], np.abs(lf[1:-1] - (lf[:-2] + lf[2:]) / 2) * 1000
    )
    dev = dev[ok[1:-1] & ok[:-2] & ok[2:]]
    return dict(
        median_px=round(float(np.median(dev)), 3),
        p99_px=round(float(np.percentile(dev, 99)), 2),
        frames_over_3px=int((dev > 3).sum()),
    )


def main(seg, lams="3,20,100,400"):
    T = json.load(open(f"{AR}/{seg}/tracks.json"))
    cam = np.fromfile(f"{AR}/{seg}/camera_raw.bin", np.float32).reshape(-1, 14).astype(np.float64)
    runs = {"raw": (cam, {})} | {lam: smooth_cameras(cam, float(lam)) for lam in lams.split(",")}
    marks = markings()
    scores = {k: [] for k in runs}
    for j, img in enumerate(frames(T["n0"], len(cam))):
        i = j * STEP
        if i >= len(cam):
            break
        if np.abs(cam[i]).sum() == 0:
            continue
        s = {k: score(c[i], img, marks, T) for k, (c, _) in runs.items()}
        if all(v[1] for v in s.values()):
            for k in runs:
                scores[k].append(s[k][0])
    base = np.array(scores["raw"])
    report = dict(segment=seg, frames_scored=len(base))
    for k, (c, info) in runs.items():
        v = np.array(scores[k])
        report[k] = dict(
            **info,
            jitter=jitter(c),
            lines_median=round(float(np.median(v)), 3),
            lines_p90=round(float(np.percentile(v, 90)), 3),
            better=int((v < base - 0.1).sum()),
            worse=int((v > base + 0.1).sum()),
        )
    print(json.dumps(report, indent=1))
    json.dump(report, open(f"cache/camsmooth_{seg}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
