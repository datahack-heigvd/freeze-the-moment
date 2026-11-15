"""Which stretch of the game the tracking scripts work on (env WIN). Frames, detections and poses are per window.
WIN="300" / "2160": the two demo windows. WIN="c<N0>_<n>": a chunk of n frames starting at source frame N0 (full-game run, see full_run.py)."""

import os

NAME = os.environ.get("WIN", "300")
WINDOWS = {
    "300": dict(
        det="cache/det300.npy",
        loc="loc300",
        frames="cache/f300",
        frame0=23008,
        n=7500,
        gtrack="cache/gtrack.pkl",
        ptrack="cache/ptrack{}.pkl",
        t0=920.5,
        dur=298.5,
        out="ar/w920",
    ),
    "2160": dict(
        det="cache/det2160.npy",
        loc="loc2160",
        frames="cache/f2160",
        frame0=54000,
        n=7500,
        gtrack="cache/gtrack2160.pkl",
        ptrack="cache/ptrack2160{}.pkl",
        t0=2160.5,
        dur=298.5,
        out="ar/w2160",
    ),
}


def chunk(N0, n):
    return dict(
        det=f"cache/full/det_{N0}.npy",
        loc=f"full_{N0}",
        frames=f"cache/full/f{N0}",
        frame0=N0,
        n=n,
        gtrack=f"cache/full/gtrack_{N0}.pkl",
        ptrack=f"cache/full/ptrack_{N0}" + "{}.pkl",
        t0=N0 / 25 + 0.5,
        dur=n / 25 - 1.5,
        out=f"ar/c{N0}",
    )


if NAME.startswith("c"):
    _N0, _n = NAME[1:].split("_")
    W = chunk(int(_N0), int(_n))
else:
    W = WINDOWS[NAME]
