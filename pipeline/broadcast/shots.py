"""Sort a broadcast into its camera shots and find the stretches filmed by the wide game camera.

The produced stream cuts between the wide game camera, close-ups, replays, the bench, intermission shows and graphics. Frames are
sampled at RATE fps at low resolution. Per sample: the share of the lower frame that is ice (bright, unsaturated), and whether the
score bug is on screen (its box matches the match's median bug appearance; the bug is hidden during replays and breaks). Cuts are
large jumps in a coarse colour histogram. A shot counts as wide game camera when most of its samples show lots of ice and the bug.

  <out>/samples.npz     t, ice share, bug score, cut flag per sample
  <out>/shots.json      [{t0, t1, wide, ice, bug}] per shot
  <out>/sheet.jpg       a sample of wide and other shots, labelled, for checking

usage (from pipeline/broadcast/):  shots.py <video> <out_dir>"""

import json
import os
import subprocess
import sys

import cv2
import numpy as np

RATE, W, H = 2, 384, 216
BUG = (slice(0, 22), slice(150, 234))  # score bug box at 384x216 (top centre)
ICE_MIN, BUG_MIN, CUT = 0.35, 0.45, 0.35  # wide shot thresholds; histogram distance that marks a cut
MIN_SHOT = 2.0  # s


def samples(video):
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", video, "-vf", f"fps={RATE},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    while True:
        b = p.stdout.read(W * H * 3)
        if len(b) < W * H * 3:
            return
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)


def ice_share(img):
    hsv = cv2.cvtColor(img[H // 3 :], cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 2] > 170) & (hsv[..., 1] < 40)).mean())


def measure(video, out):
    """Ice share, bug score and cut flag of every sample, saved to samples.npz."""
    ice, bugs, hists = [], [], []
    for i, img in enumerate(samples(video)):
        ice.append(ice_share(img))
        bugs.append(cv2.cvtColor(img[BUG], cv2.COLOR_BGR2GRAY))
        h = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        hists.append(cv2.normalize(h, h).ravel())
        if i % 2000 == 0:
            print(i / RATE, flush=True)
    ice = np.array(ice)
    bugs = np.array(bugs, np.float32)
    ref = np.median(bugs[ice > ICE_MIN], 0)  # the bug as it looks during play
    e_ref = cv2.Canny(ref.astype(np.uint8), 50, 150) > 0
    bug = np.array([((cv2.Canny(b.astype(np.uint8), 50, 150) > 0) & e_ref).sum() / max(e_ref.sum(), 1) for b in bugs])
    cut = np.r_[True, [cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA) > CUT for a, b in zip(hists, hists[1:])]]
    np.savez(f"{out}/samples.npz", t=np.arange(len(ice)) / RATE, ice=ice, bug=bug, cut=cut)


def frame(video, t, w=192, h=108):
    b = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{t:.2f}",
            "-i",
            video,
            "-frames:v",
            "1",
            "-vf",
            f"scale={w}:{h}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ],
        capture_output=True,
    ).stdout
    return np.frombuffer(b, np.uint8).reshape(h, w, 3).copy()


def main(video, out):
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(f"{out}/samples.npz"):
        measure(video, out)
    z = np.load(f"{out}/samples.npz")
    t, ice, bug, cut = z["t"], z["ice"], z["bug"], z["cut"]
    edges = np.r_[np.flatnonzero(cut), len(ice)]
    shots = []
    for a, b in zip(edges, edges[1:]):
        wide = bool(np.mean(ice[a:b] > ICE_MIN) > 0.7 and np.mean(bug[a:b] > BUG_MIN) > 0.7)
        if shots and shots[-1]["wide"] == wide and (b - a) / RATE < MIN_SHOT:
            shots[-1]["t1"] = b / RATE
            continue
        shots.append(
            dict(t0=a / RATE, t1=b / RATE, wide=wide, ice=round(float(ice[a:b].mean()), 2), bug=round(float(bug[a:b].mean()), 2))
        )
    json.dump(shots, open(f"{out}/shots.json", "w"), indent=0)
    wide_s = sum(s["t1"] - s["t0"] for s in shots if s["wide"])
    print(f"{len(shots)} shots, wide {wide_s / 60:.1f} min of {t[-1] / 60:.1f} min, wide shots {sum(s['wide'] for s in shots)}")
    rng = np.random.default_rng(0)
    tiles = []
    for wide in (True, False):
        pick = [s for s in shots if s["wide"] == wide and s["t1"] - s["t0"] >= MIN_SHOT]
        for s in rng.choice(pick, min(24, len(pick)), replace=False):
            im = frame(video, (s["t0"] + s["t1"]) / 2)
            cv2.rectangle(im, (0, 0), (191, 107), (0, 255, 0) if wide else (0, 0, 255), 3)
            cv2.putText(im, f"{s['t0']:.0f}s i{s['ice']} b{s['bug']}", (4, 14), 0, 0.4, (0, 255, 255), 1)
            tiles.append(im)
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(f"{out}/sheet.jpg", np.vstack([np.hstack(tiles[r : r + 8]) for r in range(0, len(tiles), 8)]))


if __name__ == "__main__":
    main(*sys.argv[1:])
