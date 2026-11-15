"""Identity and puck carrier where the metadata is missing, for the whole game (the method the gap benchmark chose: C + D + B + split, carrier K2).
Per chunk (WIN=c<N0>_<n>):
  index   person detections on every STEP-th frame, labelled from the full-metadata tracker (identity) and the carrier rule (has the puck)
  crops   decodes the chunk and cuts person crops (identity) and wide crops (stick area), stored as JPEG bytes
  solve   after the GPU models ran (id_gpu.py <task> game): the solver's identities fill the numbers the production tracker leaves blank,
          and a forward-backward pass over the rings gives each frame's carrier with its probability
usage: WIN=<chunk> gapid.py index | crops | solve       (solve -> cache/full/id/result_<N0>.pkl, read by export_ar.py)"""

import os
import pickle
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
import appear
import carrier as K
import idbench as B
from idsolve import STEP
from opt2 import Detections
from window import W as WIN

OUT = "cache/full/id"
N0 = WIN["frame0"]
CUES = ["D", "B"]
STAY = 0.99  # carrier: probability per frame that the puck stays with the ring that has it
GAP_S = 2.0  # a frame is in a metadata gap when no snapshot lies within this many seconds; only there the video carrier is used
CAR_MARGIN = 6.0  # nats the carrier ring must beat the runner-up by (benchmark: shown ~75% of the time, right ~71% of those)


def index():
    DT = Detections()
    rows, Ns, xy = [], [], []
    for N in range(DT.N0 - DT.N0 % STEP + STEP, DT.N0 + WIN["n"], STEP):
        r = DT.evaluate(N)
        if r is None:
            continue
        D, p, o, on, fits, _ = r
        k = np.flatnonzero(on & fits)
        a = DT.start[N - DT.F0]
        rows += list(a + k)
        Ns += [N] * len(k)
        xy += list(p[k])
    rows, Ns, xy = np.array(rows), np.array(Ns), np.array(xy)
    R = pickle.load(open(WIN["gtrack"], "rb"))
    lab = appear.labels_from(R, DT, rows, Ns, xy)
    g, snaps = B.load_snaps()
    Z = dict(row=rows, N=Ns, img=Ns - DT.F0, box=DT.D[rows, 1:5], xy=xy, lab=lab)
    Z["car"] = K.carrier_lab(Z, R, snaps, B.carriers())
    np.savez(f"{OUT}/dets_{N0}.npz", **Z)
    print(
        f"{WIN['frame0']}: {len(rows)} detections, identity labels {(lab >= 0).mean() * 100:.0f}%, carrier positives {(Z['car'] == 1).sum()}"
    )


def crops():
    """Decode the chunk (image i = source frame N0 + i, the same seek as full_run.prepare) and cut the crops of the indexed frames."""
    import crops_gpu as C
    import localize as L
    from full_run import FPS, VIDEO

    Z = np.load(f"{OUT}/dets_{N0}.npz")
    want = {}
    for k, i in enumerate(Z["img"]):
        want.setdefault(int(i), []).append(k)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{(N0 - 0.5) / FPS:.4f}",
        "-i",
        VIDEO,
        "-frames:v",
        str(WIN["n"]),
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    n = 1920 * 1080 * 3
    person, wide = [None] * len(Z["img"]), [None] * len(Z["img"])
    enc = lambda a: cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    for i in range(WIN["n"]):
        b = p.stdout.read(n)
        if len(b) < n:
            break
        if i == 0:
            ref = next(iter(L.read_range(N0, 1))).astype(np.float32)
            got = cv2.resize(np.frombuffer(b, np.uint8).reshape(1080, 1920, 3), (960, 540), interpolation=cv2.INTER_AREA).astype(
                np.float32
            )
            if np.abs(ref - got).mean() > 6:
                raise SystemExit(f"frame alignment check failed for chunk {N0}")
        if i not in want:
            continue
        img = np.frombuffer(b, np.uint8).reshape(1080, 1920, 3)
        for k in want[i]:
            person[k] = enc(C.person(img, Z["box"][k]))
            wide[k] = enc(C.wide(img, Z["box"][k]))
    p.kill()
    p.wait()
    missing = sum(x is None for x in person)
    if missing:
        raise SystemExit(f"{missing} crops not cut (video ended early)")
    pickle.dump(dict(person=person, wide=wide), open(f"{OUT}/crops_{N0}.pkl", "wb"))
    print(f"{N0}: {len(person)} crops, {sum(map(len, person)) / 2**20:.0f} + {sum(map(len, wide)) / 2**20:.0f} MB")


def solve():
    """Merged rings (production numbers + solver numbers where it had none) and the video carrier inside metadata gaps."""
    import idsolve as X
    import ptrack as PT

    R = pickle.load(open(WIN["gtrack"], "rb"))
    g, snaps = B.load_snaps()
    Z = np.load(f"{OUT}/dets_{N0}.npz")
    ctx = X.Ctx(R, snaps, Z, {t: np.load(f"{OUT}/out_{t}_{N0}.npz")[t] for t in CUES})
    cuts = ctx.split(CUES)
    ident = X.ilp(ctx, X.evidence(ctx, CUES))
    W = PT.World()
    show = PT.display(pickle.load(open(PT.CACHE.format(""), "rb")), W)
    base = {N: PT.frame_rings(show, W, N) for N in range(W.n0, W.n1)}
    rings = X.merge(base, ctx, ident)
    snapN = np.array(sorted(int(round(t * B.FPS)) for t, _ in snaps))
    gap = lambda N: np.abs(snapN - N).min() > GAP_S * B.FPS
    logit = np.load(f"{OUT}/out_K_{N0}.npz")["K"].astype(np.float32)
    byN = {}
    for i, N in enumerate(Z["N"]):
        byN.setdefault(int(N), []).append(i)
    frames = [N for N in range(W.n0 + (-W.n0) % STEP, W.n1, STEP) if gap(N) and rings.get(N)]
    post = K.posterior(rings, frames, {N: K.k2_scores(rings, N, Z, logit, byN) for N in frames}, stay=STAY**STEP)
    car = {}
    for N in range(
        W.n0, W.n1
    ):  # each frame takes the carrier of its sampled frame, if that player is numbered here (the app marks tags)
        M = N - N % STEP
        if M not in post or not gap(N):
            continue
        k, margin = post[M]
        p = rings[M][k][0]
        if margin >= CAR_MARGIN and p >= 0 and any(e[0] == p and e[2] for e in rings.get(N, [])):
            car[N] = p
    pickle.dump(dict(rings=rings, car=car), open(f"{OUT}/result_{N0}.pkl", "wb"))
    old = sum(e[2] for rs in base.values() for e in rs)
    new = sum(e[2] for rs in rings.values() for e in rs)
    ng = sum(map(gap, range(W.n0, W.n1)))
    print(
        f"{N0}: appearance split {cuts} cuts; numbered rings {old} -> {new} (+{100 * (new - old) / max(old, 1):.0f}%); metadata-gap frames {ng}, video carrier on {len(car)}"
    )


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"index": index, "crops": crops, "solve": solve}[sys.argv[1]]()
