"""How the live overlay's player tags behave, per exported segment, against the tracking feed (store/ar/<segment>/).

Every frame with a camera: the feed's on-ice players (ctl) whose feet project inside the picture are the players a viewer can see.
A tag (ring) within MATCH m of such a player covers them; with the player's roster index and the number shown it is right. A ring with
no feed player within MATCH m is on nobody (spectators, bench, staff, a stale track). Per in-view player: the frames inside their visible
stretches where they have no tag are gaps (count and mean length), and on/off changes are flicker. A named ring that moves more than
JUMP m between consecutive frames has changed person; a roster index with two rings in one frame is a duplicate tag.

  in_view    feed players in view per frame          tagged     share of them with a ring
  numbered   share with the right number shown
  wrong_place  named rings more than MATCH m from that player's feed position (of those the feed has)
  unverified   share of named rings whose player is not in the feed that frame (feed gap or bench)
  duplicates   share of frames with a player tagged twice
  gaps/min   tag gaps per visible player minute      gap_len    mean gap length (s)
  flicker    tag on/off changes per visible player minute       jumps/min  named rings that jumped, per minute
  outside    share of rings whose spot is off the ice (on or behind the boards)

usage (from pipeline/calib/):  tag_stats.py [segment ...]      (default: every segment in store/ar)"""

import glob
import json
import os
import sys

import numpy as np

AR = "../../store/ar"
MATCH, JUMP, MARGIN = 2.5, 1.5, 40  # m; m per frame (37 m/s); px kept clear of the picture edge


def project(cam, P, T):
    f, k, R, C = cam[0], cam[1], cam[2:11].reshape(3, 3), cam[11:14]
    pc = (np.c_[P, np.zeros(len(P))] - C) @ R.T
    uv = f * pc[:, :2] / pc[:, 2:3]
    uv = uv * (1 + k * (uv**2).sum(1, keepdims=True) / T["fref"] ** 2)
    return uv + [T["cx"], T["cy"]], pc[:, 2]


def off_ice(xy, T):
    q = np.abs(xy) - [T["hx"] - T["corner"], T["hy"] - T["corner"]]
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0) - T["corner"] > -0.3


def runs(mask):
    """(start, length) of the True runs of a boolean array."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    s = np.flatnonzero(d == 1)
    return list(zip(s, np.flatnonzero(d == -1) - s))


def segment(seg):
    T = json.load(open(f"{AR}/{seg}/tracks.json"))
    cams = np.fromfile(f"{AR}/{seg}/camera.bin", np.float32).reshape(-1, 14)
    n = len(T["rings"])
    fps = T["fps"]
    vis, tagged, right = {}, {}, {}
    wrong = checked = unverified = named_total = outside = rings_total = jumps = dup = frames = 0
    last = {}
    for i in range(n):
        if not cams[i].any():
            last = {}
            continue
        frames += 1
        ctl = np.array(T["ctl"][i], float).reshape(-1, 4)
        rg = np.array(T["rings"][i], float).reshape(-1, 4)
        ctl_xy, rg_xy = ctl[:, 1:3] / 100, rg[:, 1:3] / 100
        rings_total += len(rg)
        outside += off_ice(rg_xy, T).sum() if len(rg) else 0
        named = rg[:, 0] >= 0
        named_total += named.sum()
        feed = {int(p): xy for p, xy in zip(ctl[:, 0], ctl_xy) if p >= 0}
        for p, xy in zip(rg[named, 0].astype(int), rg_xy[named]):
            if p in feed:
                checked += 1
                wrong += np.linalg.norm(xy - feed[p]) > MATCH
            else:
                unverified += 1
        dup += len(set(rg[named, 0])) < named.sum()
        now = {int(p): (x / 100, y / 100) for p, x, y, _ in rg[named]}
        jumps += sum(p in last and np.hypot(xy[0] - last[p][0], xy[1] - last[p][1]) > JUMP for p, xy in now.items())
        last = now
        if not len(ctl):
            continue
        uv, z = project(cams[i], ctl_xy, T)
        inview = (z > 0) & (uv[:, 0] > MARGIN) & (uv[:, 0] < 1920 - MARGIN) & (uv[:, 1] > MARGIN) & (uv[:, 1] < 1080 - MARGIN)
        for (p, *_), xy, v in zip(ctl, ctl_xy, inview):
            if p < 0:
                continue
            key = int(p)
            vis.setdefault(key, np.zeros(n, bool))[i] = v
            if not v:
                continue
            d = np.linalg.norm(rg_xy - xy, axis=1) if len(rg) else np.array([])
            near = d < MATCH
            tagged.setdefault(key, np.zeros(n, bool))[i] = near.any()
            right.setdefault(key, np.zeros(n, bool))[i] = (
                bool(((rg[:, 0] == p) & (rg[:, 3].astype(int) & 1 == 1) & near).any()) if len(rg) else False
            )
    vis_frames = sum(v.sum() for v in vis.values())
    gaps = flicker = gap_frames = 0
    for p, v in vis.items():
        t = tagged.get(p, np.zeros(n, bool))
        for s, l in runs(v):
            seg_t = t[s : s + l]
            g = [r for r in runs(~seg_t) if 0 < r[0] and r[0] + r[1] < l]  # gaps strictly inside the visible stretch
            gaps += len(g)
            gap_frames += sum(r[1] for r in g)
            flicker += int(np.abs(np.diff(seg_t.astype(int))).sum())
    vis_min = vis_frames / fps / 60
    return dict(
        segment=seg,
        minutes=round(frames / fps / 60, 1),
        in_view=round(vis_frames / max(frames, 1), 1),
        tagged=round(sum(t.sum() for t in tagged.values()) / max(vis_frames, 1), 3),
        numbered=round(sum(r.sum() for r in right.values()) / max(vis_frames, 1), 3),
        wrong_place=round(wrong / max(checked, 1), 3),
        unverified=round(unverified / max(named_total, 1), 3),
        duplicates=round(dup / max(frames, 1), 3),
        outside=round(outside / max(rings_total, 1), 3),
        gaps_min=round(gaps / max(vis_min, 1e-9), 2),
        gap_len=round(gap_frames / max(gaps, 1) / fps, 2),
        flicker_min=round(flicker / max(vis_min, 1e-9), 2),
        jumps_min=round(jumps / max(frames / fps / 60, 1e-9), 2),
    )


def main(*segs):
    segs = segs or sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{AR}/*/tracks.json"))
    rows = [segment(s) for s in segs]
    keys = list(rows[0])
    print("\t".join(keys))
    for r in rows:
        print("\t".join(str(r[k]) for k in keys), flush=True)
    tot = sum(r["minutes"] for r in rows)
    print("all", round(tot, 1), *(round(sum(r[k] * r["minutes"] for r in rows) / tot, 3) for k in keys[2:]), sep="\t")
    json.dump(rows, open("cache/tag_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
