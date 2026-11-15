"""Per-period ice-plane tracking: chain of plane homographies with anchor correction, then (f, R) extraction."""

import subprocess
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, "scripts")
from rink import boards

VIDEO = "../../store/data/20251231_115035_SpenglerCup_Kam1 31.12.25 Game 11.mp4"
G = np.load("cache/calib_global.npz")
K1, CXO, CYO, HB = float(G["k1"]), float(G["cxo"]), float(G["cyo"]), float(G["hb"])
CX, CY = 960 + CXO, 540 + CYO
F_REF = 3050.0
C_CAM = np.array([G["Cx"], G["Cy"], G["Cz"]], float)
BOARDS = boards(width=2 * HB).astype(np.float32)
GRID = np.stack(np.meshgrid(np.linspace(-27, 27, 19), np.linspace(-13, 13, 9)), -1).reshape(-1, 2)
GRID_H = np.c_[GRID, np.ones(len(GRID))]

_sift = None
_flann = None


def _init_worker():
    global _sift, _flann
    cv2.setNumThreads(1)
    _sift = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.02)


def features_of(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = _sift.detectAndCompute(g, None)
    if des is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.array([k.pt for k in kp], np.float32) * 2.0, des


def undistort(p):
    d = p - [CX, CY]
    u = d.copy()
    for _ in range(6):
        u = d / (1 + K1 * (u**2).sum(1, keepdims=True) / F_REF**2)
    return u + [CX, CY]


def K_of(f):
    return np.array([[f, 0, CX], [0, f, CY], [0, 0, 1.0]])


def H_of(f, rv, C=C_CAM):
    R = cv2.Rodrigues(np.asarray(rv, float))[0]
    return K_of(f) @ np.column_stack([R[:, 0], R[:, 1], -R @ C])


def proj(H, pts):
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    return q[:, :2] / q[:, 2:3]


def ice_mask(H):
    q = proj(H, BOARDS)
    m = np.zeros((1080, 1920), np.uint8)
    if np.isfinite(q).all() and np.abs(q).max() < 1e5:
        cv2.fillPoly(m, [q.astype(np.int32)], 1)
    return m


def in_ice(H, pts_u):
    m = ice_mask(H)
    ix = np.clip(pts_u[:, 0].astype(int), 0, 1919)
    iy = np.clip(pts_u[:, 1].astype(int), 0, 1079)
    return m[iy, ix] > 0


def view_overlap(H1, H2):
    a, b = proj(H1, GRID), proj(H2, GRID)
    ok = lambda q: (q[:, 0] > 0) & (q[:, 0] < 1920) & (q[:, 1] > 0) & (q[:, 1] < 1080)
    both = ok(a) & ok(b)
    return int(both.sum()), (np.linalg.norm(a[both] - b[both], axis=1).mean() if both.sum() else np.inf)


def knn_pairs(src_des, dst_des, ratio=0.8):
    global _flann
    if _flann is None:
        _flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))
    if len(src_des) < 15 or len(dst_des) < 15:
        return None
    m = _flann.knnMatch(src_des, dst_des, k=2)
    good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < ratio * b.distance]
    return good if len(good) >= 12 else None


def solve_pose(world, dst, pose0, thr=2.5, min_inl=15):
    """world (N,2) ice points -> dst (N,2) undistorted px. RANSAC then LM on (f, rotvec) with the camera position fixed."""
    from scipy.optimize import least_squares

    H, inl = cv2.findHomography(
        world.astype(np.float32), dst.astype(np.float32), cv2.RANSAC, thr, maxIters=3000, confidence=0.999
    )
    if H is None or inl.sum() < min_inl:
        return None
    k = inl.ravel().astype(bool)
    w, d = world[k], dst[k]
    f0, rv0 = pose0
    res = lambda x: (proj(H_of(x[0], x[1:4]), w) - d).ravel()
    sol = least_squares(
        res, np.concatenate([[f0], rv0]), x_scale=[50, 0.01, 0.01, 0.01], loss="soft_l1", f_scale=1.5, max_nfev=30
    )
    e = np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)
    return sol.x[0], sol.x[1:4], int((e < 2.5).sum()), float(np.sqrt((e[e < 5] ** 2).mean())) if (e < 5).any() else 99.0


class Anchor:
    def __init__(self, f, rv, pts_u, des):
        self.f, self.rv = f, np.array(rv, float)
        self.H = H_of(f, rv)
        keep = in_ice(self.H, pts_u)
        pu = pts_u[keep]
        self.des = des[keep]
        self.world = proj(np.linalg.inv(self.H), pu)


def read_frames(start, dur, fps):
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
        f"fps={fps},scale=960:540",
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
        yield np.frombuffer(b, np.uint8).reshape(540, 960, 3)
    p.wait()


def seed_anchors():
    out = []
    for i, cam_i in ((0, 0), (200, 2), (359, 1)):
        img = cv2.imread(f"cache/frames/seg0/{i:06d}.jpg")
        cam = G["cams"][cam_i]
        _init_worker()
        pts, des = features_of(img)
        out.append(Anchor(cam[0], cam[1:4], undistort(pts), des))
    return out


def run_period(name, start, dur, fps, workers, max_anchors=3):
    t0 = time.time()
    anchors = seed_anchors()
    pool = Pool(workers, initializer=_init_worker)
    frames = read_frames(start, dur, fps)
    poses, inl_chain, inl_abs, rms_l, times = [], [], [], [], []
    state = {"prev": None}
    n_total = int(dur * fps)

    def process(feat_pts, feat_des, i):
        pu = undistort(feat_pts)
        prev = state["prev"]
        pose = None
        ic = 0
        ia = 0
        rms = 99.0
        if prev is not None:
            ppts, pdes, pf, prv = prev
            Hp = H_of(pf, prv)
            keep = in_ice(Hp, ppts)
            good = knn_pairs(pdes[keep], feat_des)
            if good is not None:
                src = ppts[keep][[g.queryIdx for g in good]]
                dst = pu[[g.trainIdx for g in good]]
                r = solve_pose(proj(np.linalg.inv(Hp), src), dst, (pf, prv))
                if r is not None:
                    pose = (r[0], r[1])
                    ic = r[2]
                    rms = r[3]
        if pose is None or i % 4 == 0 or ic < 60:
            ref = pose if pose is not None else ((prev[2], prev[3]) if prev is not None else None)
            Hr = H_of(*ref) if ref is not None else None
            cands = []
            for a in anchors:
                if Hr is not None:
                    n_ov, _ = view_overlap(Hr, a.H)
                    if n_ov >= 15:
                        cands.append((n_ov, a))
                else:
                    cands.append((0, a))
            cands = [a for _, a in sorted(cands, key=lambda t: -t[0])[: 3 if Hr is not None else len(anchors)]]
            best = None
            for a in cands:
                good = knn_pairs(a.des, feat_des)
                if good is None:
                    continue
                w = a.world[[g.queryIdx for g in good]]
                dst = pu[[g.trainIdx for g in good]]
                r = solve_pose(w, dst, ref if ref is not None else (a.f, a.rv), min_inl=25)
                if r is not None and (best is None or r[2] > best[2]):
                    best = r
            if best is not None and (pose is None or best[2] >= 40):
                disagree = np.inf
                if pose is not None:
                    disagree = view_overlap(H_of(*pose), H_of(best[0], best[1]))[1]
                if not (pose is not None and ic > 80 and disagree > 60):
                    pose = (best[0], best[1])
                    ia = best[2]
                    rms = best[3]
        if pose is None and prev is not None:
            pose = (prev[2], prev[3])
        if pose is not None:
            H = H_of(*pose)
            if (
                max(ic, ia) >= 150
                and rms < 2.0
                and len(anchors) < max_anchors
                and all((lambda ov: ov[0] < 25 or ov[1] > 140)(view_overlap(H, a.H)) for a in anchors)
            ):
                anchors.append(Anchor(pose[0], pose[1], pu, feat_des))
            state["prev"] = (pu, feat_des, pose[0], pose[1])
            poses.append((pose[0], *pose[1]))
        else:
            poses.append((np.nan,) * 4)
        inl_chain.append(ic)
        inl_abs.append(ia)
        rms_l.append(rms)
        times.append(start + i / fps)

    i = 0
    for feats in pool.imap(_feat_job, frames, chunksize=2):
        process(feats[0], feats[1], i)
        i += 1
        if i % 200 == 0:
            print(
                f"[{name}] {i}/{n_total} anchors={len(anchors)} inl chain={int(np.mean(inl_chain[-200:]))} abs={int(np.mean(inl_abs[-200:]))} rms={np.median(rms_l[-200:]):.2f} {time.time() - t0:.0f}s",
                flush=True,
            )
    pool.close()
    P = np.array(poses)
    np.savez(
        f"cache/cam_{name}.npz",
        t=np.array(times),
        f=P[:, 0],
        rv=P[:, 1:4],
        rms=np.array(rms_l),
        inl_chain=np.array(inl_chain),
        inl_abs=np.array(inl_abs),
    )
    print(
        f"[{name}] done {len(times)} frames, {np.isfinite(P[:, 0]).mean() * 100:.1f}% with pose, anchors {len(anchors)}, median fit rms {np.median(rms_l):.2f}px, {time.time() - t0:.0f}s",
        flush=True,
    )


def _feat_job(img):
    return features_of(img)


def ic_mean(a):
    return int(np.mean(a[-200:]))


def ia_mean(a):
    return int(np.mean(a[-200:]))


if __name__ == "__main__":
    name, start, dur, fps, workers = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
    run_period(name, start, dur, fps, workers)
