"""Goals in any broadcast from its score bug, with nothing set by hand: where the bug is, what it reads, when the score rises.

Pass 1 (every STEP s, W px wide): the bug is the graphic that stays put while the picture behind it changes: pixels that hold their
median value (within TOL grey levels) in at least ON of the samples, grouped into blobs; of those, the one with the most clock-like
change round it (pixels changing in over 20% of the samples it shows). Pass 2 (every STEP s, the bug's box at full resolution): a
sample shows the bug where its steady pixels match their median. The samples showing it are read by a vision-language model (an
OpenAI-style server, VLM_URL, serving VLM_MODEL) every COARSE samples; a score counts once read on two reads running (a misread is
rarely repeated) and scores only go up; each rise is narrowed to one sample by bisection (reading the samples in between). Goals
scored before the bug is first shown (it can show up mid-game already reading 1-0) cannot be timed and are left out. The bug
updates after the goal, when the replays start (a minute or more after a video review), but the game clock stops at the goal: the
clock is the bug's busiest pixels (changing in over TICK of the samples), running where they change from one sample to the next, and
a goal's clock_stop is the last sample it ran in before the score rose.

Reading the score replaced tracking which pixels change: on the 2023 bugs a statistics bar pops up beside each score and the score's
own pixels were not found apart from it (TNMZOdxz4Lk: 18 changes, none a goal, the four goals missed). Read, HC Davos - Kärpät 2024
gives its seven goals (final 4-3) and Ambrì - Frölunda 2023 its five (0-5, each rise checked by eye).

  <out>/score_bug.json    {"bug": [x0, y0, x1, y1] (px of the source), "overlays": [every steady graphic's box],
                           "reads": {sample: [left, right] or null}, "goals": [{t, side (left or right), state (that side's score),
                           clock_stop, before, after}]}
  <out>/pass1.npy, pass2.npy   the samples (rerunning reads them; without pass1.npy the bug's box is taken from score_bug.json)
  <out>/score_bug.jpg     the bug before and at each goal

usage:  score_bug.py <video> <out_dir>   (the VLM server reachable at VLM_URL)"""

import base64
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

import cv2
import numpy as np

STEP, W = 2.0, 640  # s between samples; px width of a pass-1 sample
TOL, ON, MIN_BLOB = 12, 0.35, 40  # grey levels; share of samples a bug pixel holds its median; px (pass 1)
DILATE, CHANGE, TICK = 6, 40, 0.2  # px (pass 1) round the bug; grey-level jump; share of samples the clock's seconds change in
COARSE = 10  # samples showing the bug between reads
VLM_URL, VLM_MODEL = (
    os.environ.get("VLM_URL", "http://127.0.0.1:9031/v1/chat/completions"),
    os.environ.get("VLM_MODEL", "qwen-vl"),
)
PROMPT = (
    "This is an ice hockey broadcast's score bug. What are the two teams' scores, left team first? Answer as JSON "
    '{"left": <int>, "right": <int>}, or {"left": null, "right": null} if the scores are not shown.'
)


def probe(video):
    return tuple(
        map(
            int,
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=p=0",
                    video,
                ]
            )
            .decode()
            .split(",")[:2],
        )
    )


def sample(video, vf, w, h, cache=None):
    """Grey samples every STEP s through the ffmpeg filter vf, w x h px (kept in cache, an .npy, when given)."""
    if cache and os.path.exists(cache):
        return np.load(cache)
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", video, "-vf", f"fps={1 / STEP},{vf},format=gray", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE,
    )
    frames = []
    while len(b := p.stdout.read(w * h)) == w * h:
        frames.append(np.frombuffer(b, np.uint8).reshape(h, w))
    if cache:
        np.save(cache, np.stack(frames))
    return np.stack(frames)


def steady(F):
    """Pixels that hold their median in at least ON of the samples, and the median."""
    med = np.median(F[:: max(len(F) // 400, 1)], axis=0)
    return np.mean([np.abs(f.astype(np.int16) - med) <= TOL for f in F], axis=0) >= ON, med


def find_bug(video, out):
    """The bug's box and every steady graphic's (the channel logo too), px of the source: pass 1."""
    if not os.path.exists(f"{out}/pass1.npy") and os.path.exists(f"{out}/score_bug.json"):
        d = json.load(open(f"{out}/score_bug.json"))
        return tuple(d["bug"]), d["overlays"]
    w0, h0 = probe(video)
    H = round(W * h0 / w0 / 2) * 2
    F = sample(video, f"scale={W}:{H}", W, H, f"{out}/pass1.npy")
    scale = w0 / W
    stable, med = steady(F)
    n, lab, st, _ = cv2.connectedComponentsWithStats(
        cv2.morphologyEx(stable.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    )
    best, overlays = None, []
    for k in range(1, n):
        if st[k, 4] < MIN_BLOB or st[k, 2] > W * 0.6:
            continue
        x, y, w, h = st[k, :4]
        overlays.append([round(v * scale) for v in (x, y, x + w, y + h)])
        m = lab == k
        on = np.array([(np.abs(f[m].astype(np.int16) - med[m]) <= TOL).mean() > 0.8 for f in F])
        if on.sum() < 10:
            continue
        x, y, w, h = st[k, :4]
        box = (max(x - DILATE, 0), max(y - DILATE, 0), min(x + w + DILATE, W), min(y + h + DILATE, H))
        sub = F[on][:, box[1] : box[3], box[0] : box[2]].astype(np.int16)
        busy = float(((np.abs(np.diff(sub, axis=0)) > CHANGE).mean(0) > 0.2).mean())
        if best is None or busy > best[0]:
            best = (busy, box)
    x0, y0, x1, y1 = (round(v * scale) // 2 * 2 for v in best[1])
    return (x0, y0, x1, y1), overlays


def read(img):
    """[left, right] the model reads off a bug sample, or None."""
    ok, jpg = cv2.imencode(
        ".jpg", cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    msg = [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()}},
        {"type": "text", "text": PROMPT},
    ]
    body = json.dumps(
        dict(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": msg}],
            max_tokens=30,
            temperature=0,
            chat_template_kwargs={"enable_thinking": False},
        )
    ).encode()
    with urlopen(Request(VLM_URL, body, {"Content-Type": "application/json"}), timeout=300) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    try:
        a = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return None
    return [a["left"], a["right"]] if isinstance(a.get("left"), int) and isinstance(a.get("right"), int) else None


def main(video, out):
    os.makedirs(out, exist_ok=True)
    (x0, y0, x1, y1), overlays = find_bug(video, out)
    w, h = x1 - x0, y1 - y0
    print("bug", (x0, y0, x1, y1), flush=True)
    F = sample(video, f"crop={w}:{h}:{x0}:{y0}", w, h, f"{out}/pass2.npy")
    stable, med = steady(F)
    on = np.flatnonzero([(np.abs(f[stable].astype(np.int16) - med[stable]) <= TOL).mean() > 0.8 for f in F])
    rate = np.zeros((h, w))
    for a, b in zip(on[:-1], on[1:]):
        rate += np.abs(F[b].astype(np.int16) - F[a]) > CHANGE
    tick = rate / max(len(on) - 1, 1) > TICK  # the clock's seconds: running where they change between samples showing the bug
    running = np.r_[
        False, [(np.abs(F[b][tick].astype(np.int16) - F[a][tick]) > CHANGE).mean() > 0.1 for a, b in zip(on[:-1], on[1:])]
    ]

    cache = {}

    def reads(idx):  # idx: positions in on
        new = [k for k in idx if k not in cache]
        with ThreadPoolExecutor(8) as pool:
            for k, r in zip(new, pool.map(lambda k: read(np.asarray(F[on[k]])), new)):
                cache[k] = r
        return [cache[k] for k in idx]

    coarse = list(range(0, len(on), COARSE))
    seq = reads(coarse)
    states = [(0, [0, 0])]  # (position in on where first read, score)
    for j in range(len(coarse) - 1):
        s, nxt = seq[j], seq[j + 1]
        if s and s == nxt and s != states[-1][1] and all(a >= b for a, b in zip(s, states[-1][1])):
            states.append((coarse[j], s))
    goals = []
    for (k0, s0), (k1, s1) in zip(states[:-1], states[1:]):
        for side in (0, 1):
            for v in range(s0[side] + 1, s1[side] + 1):  # the first sample reading at least v on this side, by bisection
                a, b = k0, k1
                while b - a > 1:
                    m = (a + b) // 2
                    r = reads([m])[0]
                    if r and r[side] >= v:
                        b = m
                    else:
                        a = m
                if b == 0:
                    continue  # shown from the first bug sample on: scored before it
                k = b - 1
                while k > 0 and not running[k]:
                    k -= 1
                goals.append(
                    dict(
                        t=float(on[b] * STEP),
                        side=("left", "right")[side],
                        state=v,
                        clock_stop=float(on[k] * STEP),
                        before=cache.get(a),
                        after=cache.get(b),
                        at=int(on[b]),
                        prev=int(on[a]),
                    )
                )
    goals.sort(key=lambda g: g["t"])
    rows = [np.hstack([F[g.pop("prev")], np.full((h, 4), 255, np.uint8), F[g.pop("at")]]) for g in goals]
    json.dump(
        dict(bug=[x0, y0, x1, y1], overlays=overlays, reads={int(on[k]): r for k, r in sorted(cache.items())}, goals=goals),
        open(f"{out}/score_bug.json", "w"),
        indent=1,
    )
    if rows:
        cv2.imwrite(f"{out}/score_bug.jpg", np.vstack(rows))
    print(
        f"{len(on)} of {len(F)} samples show the bug; {len(cache)} reads; scores {[s for _, s in states]}:",
        [(round(g["t"]), g["side"][0], round(g["clock_stop"])) for g in goals],
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
