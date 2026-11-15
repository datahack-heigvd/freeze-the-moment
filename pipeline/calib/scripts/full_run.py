"""Whole-game pipeline: camera, detections, tracking and export for every period, in 300 s chunks (intermissions are skipped).
Each stage skips chunks it has already finished, so the run can be stopped and restarted. Exports go to cache/full/ar and replace store/ar only in `finish`.

usage (from pipeline/calib/):  full_run.py all        run the camera, detection and tracking lanes in parallel, then finish
                      full_run.py masks      run the person-outline lanes (GPU segmentation, export), then attach masks.bin to every segment in store/ar
                      full_run.py gaps       identity and puck carrier where the metadata is missing (gapid.py): crops of every chunk, the appearance
                                             models trained on the whole game on the GPU machine (those not trained yet), then solve and re-export every segment in store/ar
                      full_run.py lane cam|det|track|seg|mask
                      full_run.py finish     index + swap the staged segments into store/ar (only when every chunk is exported)
                      full_run.py attach     copy the exported masks.bin files into the segments of store/ar
                      full_run.py plan       print the chunks"""

import concurrent.futures as cf
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

VIDEO = "../../store/data/20251231_115035_SpenglerCup_Kam1 31.12.25 Game 11.mp4"
FULL, STAGE, LIVE_AR, MASKS = "cache/full", "cache/full/ar", "../../store/ar", "cache/full/masks"
GPU, REMOTE = os.environ.get("GPU_HOST", "gpu"), "hockey"  # the GPU machine (ssh) and the working dir there
CHUNK_N, HEAD_S, TAIL_S, MIN_LAST_N = (
    7500,
    5.0,
    20.0,
    250,
)  # frames per chunk; seconds before the first and after the last snapshot of a period
MAP_PANO, MAP_START, WORKERS = "pano300", "920.35", 6
FPS = 25


def chunks():
    """[{period, N0, n}]: every period from HEAD_S before its first snapshot to TAIL_S after its last."""
    g = json.load(open("../../store/site/game.json"))
    ps = g["periods"]
    ts = sorted(f[0] for f in g["frames"])
    out = []
    for i, p in enumerate(ps):
        nxt = ps[i + 1]["t"] if i + 1 < len(ps) else 1e9
        te = max(t for t in ts if p["t"] <= t < nxt)
        a, b = math.ceil((p["t"] - HEAD_S) * FPS), math.ceil((te + TAIL_S) * FPS)
        cs = [dict(period=p["period"], N0=s, n=min(s + CHUNK_N, b) - s) for s in range(a, b, CHUNK_N)]
        if len(cs) > 1 and cs[-1]["n"] < MIN_LAST_N:
            cs.pop()
            cs[-1]["n"] = b - cs[-1]["N0"]
        out += cs
    return out


def order():
    """The first chunk of the 2nd and 3rd period go first (they test the camera map and the metadata sync far from the map window), then time order."""
    cs = chunks()
    first = [c for c in cs if c["period"] > 1 and c is next(d for d in cs if d["period"] == c["period"])]
    return first + [c for c in cs if c not in first]


name = lambda c: f"c{c['N0']}_{c['n']}"
cam_file = lambda c: f"cache/cam_full_{c['N0']}.npz"
det_file = lambda c: f"{FULL}/det_{c['N0']}.npy"
seg_file = lambda c: f"{FULL}/seg_{c['N0']}.npz"
mask_file = lambda c: f"{MASKS}/c{c['N0']}/masks.bin"
staged = lambda c: os.path.exists(f"{STAGE}/c{c['N0']}/meta.json")


def log(lane, msg):
    print(f"[{time.strftime('%H:%M:%S')}] {lane}: {msg}", flush=True)


def run(cmd, lane, env=None, nice=False, out=None):
    e = dict(os.environ, **(env or {}))
    p = subprocess.run((["nice", "-n", "10"] if nice else []) + cmd, env=e, capture_output=True, text=True)
    if out:
        open(out, "w").write(p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError(f"{' '.join(cmd)[:120]} -> {p.returncode}\n{(p.stdout + p.stderr)[-600:]}")
    return p.stdout


def lane_cam():
    for c in order():
        if os.path.exists(cam_file(c)):
            continue
        t = time.time()
        log("cam", f"{name(c)} start")
        run(
            [
                "uv",
                "run",
                "python",
                "-u",
                "scripts/localize_map.py",
                f"full_{c['N0']}",
                MAP_PANO,
                MAP_START,
                str(c["N0"] / FPS),
                str((c["n"] + 0.5) / FPS),
                str(WORKERS),
            ],
            "cam",
            out=f"{FULL}/cam_{c['N0']}.log",
        )
        log("cam", f"{name(c)} done {time.time() - t:.0f}s: " + open(f"{FULL}/cam_{c['N0']}.log").read().strip().splitlines()[-1])


def prepare(c, tag="f"):
    """Extract the chunk's frames (image i = source frame N0 + i) and copy them to the GPU machine (the detection and segmentation lanes use separate folders)."""
    d = f"{FULL}/{tag}{c['N0']}"
    if not (os.path.isdir(d) and len(os.listdir(d)) == c["n"]):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                f"{(c['N0'] - 0.5) / FPS:.4f}",
                "-i",
                VIDEO,
                "-frames:v",
                str(c["n"]),
                "-fps_mode",
                "passthrough",
                "-start_number",
                "0",
                "-q:v",
                "2",
                f"{d}/%05d.jpg",
            ],
            "det",
            nice=True,
        )
    sys.path.insert(0, "scripts")
    import cv2
    import localize as L

    refs = list(L.read_range(c["N0"], 2))
    got = cv2.resize(cv2.imread(f"{d}/00000.jpg"), (960, 540), interpolation=cv2.INTER_AREA).astype(np.float32)
    diff, diff_next = [float(np.abs(r.astype(np.float32) - got).mean()) for r in refs]
    if diff > 6 or diff > diff_next + 0.5:
        raise RuntimeError(f"frame alignment check failed for {name(c)}: diff to frame N0 {diff:.1f}, to N0+1 {diff_next:.1f}")
    run(["ssh", GPU, f"mkdir -p {REMOTE}/full && rm -rf {REMOTE}/full/{tag}{c['N0']}"], "det")
    run(["rsync", "-a", "--info=stats0", f"{d}/", f"{GPU}:{REMOTE}/full/{tag}{c['N0']}/"], "det", nice=True)
    return diff


def lane_det():
    todo = [c for c in order() if not os.path.exists(det_file(c))]
    if not todo:
        return
    run(["rsync", "-a", "gpu/detect_gpu.py", f"{GPU}:{REMOTE}/"], "det")
    ex = cf.ThreadPoolExecutor(1)
    nxt = ex.submit(prepare, todo[0])
    for i, c in enumerate(todo):
        t = time.time()
        diff = nxt.result()
        nxt = ex.submit(prepare, todo[i + 1]) if i + 1 < len(todo) else None
        log("det", f"{name(c)} frames on the GPU machine (align diff {diff:.2f}), detecting")
        run(
            [
                "ssh",
                GPU,
                f"cd {REMOTE} && .venv/bin/python -u detect_gpu.py full/f{c['N0']} full/det_{c['N0']}.npz yolo11x.pt 1920 > full/det_{c['N0']}.log 2>&1",
            ],
            "det",
        )
        run(["scp", "-q", f"{GPU}:{REMOTE}/full/det_{c['N0']}.npy", det_file(c)], "det")
        n = len(np.load(det_file(c)))
        if n < 3 * c["n"]:
            raise RuntimeError(f"{name(c)}: only {n} detections")
        run(["ssh", GPU, f"rm -rf {REMOTE}/full/f{c['N0']}"], "det")
        shutil.rmtree(f"{FULL}/f{c['N0']}", ignore_errors=True)
        log("det", f"{name(c)} done {time.time() - t:.0f}s, {n} detections")


def lane_track():
    for c in order():
        while not (os.path.exists(cam_file(c)) and os.path.exists(det_file(c))):
            time.sleep(30)
        if staged(c):
            continue
        t = time.time()
        env = dict(WIN=name(c), AR_ROOT=STAGE)
        log("track", f"{name(c)} start")
        for step in (["gtrack.py", "build"], ["ptrack.py", "run"]):
            run(
                ["uv", "run", "python", "-u", f"scripts/{step[0]}", step[1]],
                "track",
                env,
                nice=True,
                out=f"{FULL}/{step[0][:-3]}_{c['N0']}.log",
            )
        ev = [
            l
            for l in run(["uv", "run", "python", "-u", "scripts/ptrack.py", "eval"], "track", env, nice=True).splitlines()
            if l.startswith("all windows")
        ]
        open(f"{FULL}/eval_{c['N0']}.txt", "w").write("\n".join(ev))
        run(["uv", "run", "python", "-u", "scripts/export_ar.py"], "track", env, nice=True, out=f"{FULL}/export_{c['N0']}.log")
        log("track", f"{name(c)} done {time.time() - t:.0f}s | {ev[0] if ev else 'no eval'}")


def lane_seg():
    """Person outlines (players, referees, staff) of every frame, from a segmentation model on the GPU machine."""
    todo = [c for c in order() if not os.path.exists(seg_file(c))]
    if not todo:
        return
    run(["rsync", "-a", "gpu/seg_poly_gpu.py", f"{GPU}:{REMOTE}/"], "seg")
    ex = cf.ThreadPoolExecutor(1)
    nxt = ex.submit(prepare, todo[0], "s")
    for i, c in enumerate(todo):
        t = time.time()
        nxt.result()
        nxt = ex.submit(prepare, todo[i + 1], "s") if i + 1 < len(todo) else None
        log("seg", f"{name(c)} frames on the GPU machine, segmenting")
        run(
            [
                "ssh",
                GPU,
                f"cd {REMOTE} && .venv/bin/python -u seg_poly_gpu.py full/s{c['N0']} full/seg_{c['N0']}.npz > full/seg_{c['N0']}.log 2>&1",
            ],
            "seg",
        )
        run(["scp", "-q", f"{GPU}:{REMOTE}/full/seg_{c['N0']}.npz", seg_file(c) + ".part"], "seg")
        os.replace(seg_file(c) + ".part", seg_file(c))
        run(["ssh", GPU, f"rm -rf {REMOTE}/full/s{c['N0']}"], "seg")
        shutil.rmtree(f"{FULL}/s{c['N0']}", ignore_errors=True)
        log("seg", f"{name(c)} done {time.time() - t:.0f}s")


def lane_mask():
    """Keep the outlines of the people on the ice and pack them for the app (needs the chunk's camera and detections for the geometry)."""
    for c in order():
        while not all(os.path.exists(f) for f in (seg_file(c), cam_file(c), det_file(c))):
            time.sleep(30)
        if os.path.exists(mask_file(c)):
            continue
        t = time.time()
        out = run(
            ["uv", "run", "python", "-u", "scripts/export_masks.py", seg_file(c)],
            "mask",
            dict(WIN=name(c), AR_ROOT=MASKS),
            nice=True,
        )
        log("mask", f"{name(c)} done {time.time() - t:.0f}s: {out.strip().splitlines()[-1]}")


def attach():
    """Copy the exported masks.bin files into the segments of store/ar (once the full run has swapped those in)."""
    n = 0
    for c in chunks():
        dst = f"{LIVE_AR}/c{c['N0']}"
        if not os.path.isdir(dst) or not os.path.exists(mask_file(c)):
            continue
        shutil.copyfile(mask_file(c), dst + "/masks.bin.part")
        os.replace(dst + "/masks.bin.part", dst + "/masks.bin")
        n += 1
    log("attach", f"masks.bin in {n} of {len(chunks())} segments")
    return n


def gaps():
    """Runs after the full run: needs every chunk's tracker output. The models learn from the whole game's metadata-labelled crops."""
    ID = "cache/full/id"
    os.makedirs(ID, exist_ok=True)
    for c in chunks():
        env = dict(WIN=name(c), OMP_NUM_THREADS="2")
        if not os.path.exists(f"{ID}/dets_{c['N0']}.npz"):
            run(["uv", "run", "python", "-u", "scripts/gapid.py", "index"], "gaps", env)
        if not os.path.exists(f"{ID}/crops_{c['N0']}.pkl"):
            run(["uv", "run", "python", "-u", "scripts/gapid.py", "crops"], "gaps", env, nice=True)
            log("gaps", f"{name(c)} crops cut")
    tasks = [t for t in ("D", "B", "K") if not all(os.path.exists(f"{ID}/out_{t}_{c['N0']}.npz") for c in chunks())]
    if tasks:
        run(["ssh", GPU, f"mkdir -p {REMOTE}/id/game"], "gaps")
        run(
            ["rsync", "-a", "--info=stats0"]
            + sorted(glob.glob(f"{ID}/dets_*.npz") + glob.glob(f"{ID}/crops_*.pkl"))
            + [f"{GPU}:{REMOTE}/id/game/"],
            "gaps",
        )
        run(["scp", "-q", "gpu/id_gpu.py", f"{GPU}:{REMOTE}/id/"], "gaps")
    for task in tasks:
        t = time.time()
        run(["ssh", GPU, f"cd {REMOTE}/id && ../.venv-id/bin/python -u id_gpu.py {task} game > log_{task}_game.txt 2>&1"], "gaps")
        log("gaps", f"model {task} trained and applied {time.time() - t:.0f}s")
    if tasks:
        run(["rsync", "-a", f"{GPU}:{REMOTE}/id/game/out_*.npz", f"{ID}/"], "gaps")
    for c in chunks():
        env = dict(WIN=name(c), OMP_NUM_THREADS="2", AR_ROOT=LIVE_AR)
        log("gaps", run(["uv", "run", "python", "-u", "scripts/gapid.py", "solve"], "gaps", env).strip().splitlines()[-1])
        run(["uv", "run", "python", "-u", "scripts/export_ar.py"], "gaps", env)
    log("gaps", f"{len(chunks())} segments re-exported")


def finish():
    cs = chunks()
    missing = [name(c) for c in cs if not staged(c)]
    if missing:
        raise SystemExit(f"not finished, missing: {missing}")
    os.environ["AR_ROOT"] = STAGE
    sys.path.insert(0, "scripts")
    import export_ar

    export_ar.write_index()
    if os.path.exists(f"{LIVE_AR}.old"):
        shutil.rmtree(f"{LIVE_AR}.old")
    if os.path.exists(LIVE_AR):
        os.rename(LIVE_AR, f"{LIVE_AR}.old")
    os.rename(STAGE, LIVE_AR)
    shutil.rmtree(f"{LIVE_AR}.old", ignore_errors=True)
    log("finish", f"store/ar now holds {len(cs)} segments")
    attach()


LANES = dict(cam=lane_cam, det=lane_det, track=lane_track, seg=lane_seg, mask=lane_mask)


def lane(which):
    os.makedirs(FULL, exist_ok=True)
    for attempt in range(3):
        try:
            LANES[which]()
            return
        except Exception as e:
            log(which, f"FAILED (attempt {attempt + 1}): {e}")
            time.sleep(60)
    raise SystemExit(f"lane {which} gave up")


def run_lanes(which):
    """The lanes as parallel processes; if one gives up, the others (which would wait for its output forever) are stopped. Returns their exit codes."""
    os.makedirs(FULL, exist_ok=True)
    ps = [
        subprocess.Popen(
            ["uv", "run", "python", "-u", "scripts/full_run.py", "lane", w],
            stdout=open(f"{FULL}/lane_{w}.log", "a"),
            stderr=subprocess.STDOUT,
        )
        for w in which
    ]
    while any(p.poll() is None for p in ps):
        if any(p.poll() for p in ps):
            [p.terminate() for p in ps if p.poll() is None]
        time.sleep(20)
    return [p.returncode for p in ps]


def all_():
    codes = run_lanes(["cam", "det", "track"])
    log("all", f"lanes finished {codes}")
    if any(codes):
        raise SystemExit("a lane failed, nothing was swapped in")
    finish()


def masks():
    codes = run_lanes(["seg", "mask"])
    log("masks", f"lanes finished {codes}")
    if any(codes):
        raise SystemExit("a lane failed")
    while not all(os.path.exists(f"{LIVE_AR}/c{c['N0']}/meta.json") for c in chunks()):
        time.sleep(60)  # the full run swaps its segments in when it is done
    attach()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        for c in order():
            print(c["period"], c["N0"], f"{c['N0'] / FPS:.1f}s", c["n"], f"{c['n'] / FPS:.0f}s")
        print(len(chunks()), "chunks,", sum(c["n"] for c in chunks()) / FPS, "s")
    elif cmd == "lane":
        lane(sys.argv[2])
    elif cmd == "finish":
        finish()
    elif cmd == "attach":
        attach()
    elif cmd == "masks":
        masks()
    elif cmd == "gaps":
        gaps()
    else:
        all_()
