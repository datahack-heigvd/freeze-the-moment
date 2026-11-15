"""The goal shot's release from the video: the frame the scorer's stick blade moves fastest (the fitted sticks, stick_pose.py), from
SPAN[0] to SPAN[1] frames around the feed's shot time (which runs 0.2 to 0.5 s early), if that peak is a strike: at least STRIKE m/s and
PEAK times the median blade speed of the span. Against the six releases read by eye: within 2 frames on five (asplund -1, musa -2,
stransky +1, tambellini -1, zadina -1); corvi +10, on the follow-through (by eye "probable", the puck not visible). Weighting by the
tracked puck's distance to the blade is worse (tambellini -13: before the release the track is mostly wrong).

Writes bundle/release_auto.json {"frame", "how", "speed"} (puck.py takes release.json, read by eye, before it).

usage (from a goal dir, after scene_export.py):                  BUNDLE=bundle release.py <scene_dir> [check]   (check: prints
                                                                      the found release against release.json's, writes nothing)"""

import json
import os
import sys

import numpy as np
from goal import puck

SPAN, STRIKE, PEAK = (-2, 16), 8.0, 1.8  # frames around the feed's shot; m/s; times the median


def blade_speed(scene_dir, key):
    """Exported frame numbers and the blade heel's speed (m/s) of the person with this cast key from each frame to the next."""
    s = json.load(open(f"{scene_dir}/scene.json"))
    P, F = len(s["people"]), s["frames"]
    st = np.fromfile(f"{scene_dir}/sticks.bin", np.float32).reshape(P, F, 2, 3)
    blade = st[next(i for i, p in enumerate(s["people"]) if p["key"] == key), :, 1]
    return np.arange(F - 1) + s["frame0"], np.linalg.norm(np.diff(blade, axis=0), axis=1) * s["fps"]


def main():
    scene_dir, check = sys.argv[1], sys.argv[2:] == ["check"]
    goal = puck.goal_end()
    feed = puck.goal_feed_frame()
    if goal is None:
        return print("no goal shot in the window")
    fr, v = blade_speed(scene_dir, goal[3])
    span = (fr >= feed + SPAN[0]) & (fr <= feed + SPAN[1]) & np.isfinite(v)
    if not span.any():
        return print("no stick in the span")
    k = np.flatnonzero(span)[np.argmax(v[span])]
    f, top, med = int(fr[k]), float(v[k]), float(np.median(v[span]))
    strike = top >= STRIKE and top >= PEAK * med
    if check:
        eye = json.load(open(f"{puck.B}/release.json"))["frame"] if os.path.exists(f"{puck.B}/release.json") else None
        return print(
            json.dumps(
                dict(
                    found=f,
                    eye=eye,
                    off=None if eye is None else f - eye,
                    speed=round(top, 1),
                    median=round(med, 1),
                    strike=strike,
                )
            )
        )
    if not strike:
        return print(f"no strike: blade peak {top:.1f} m/s, median {med:.1f}")
    json.dump(
        dict(frame=f, speed=round(top, 1), how=f"blade speed peak {top:.1f} m/s (median {med:.1f}) at {f}, feed shot {feed}"),
        open(f"{puck.B}/release_auto.json", "w"),
    )
    print("release", f, f"{top:.1f} m/s")


if __name__ == "__main__":
    main()
