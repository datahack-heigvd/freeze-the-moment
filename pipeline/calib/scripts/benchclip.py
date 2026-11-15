"""Comparison clips for the gap benchmark: up to four saved methods side by side (two in a row, three or four in a 2x2 grid) on the same stretch of video.
Each panel draws the method's rings and numbers; a number is outlined green when a metadata sample (within SNAP_TOL frames) puts that
player under the ring, red when it does not; the verdict is held HOLD frames, white when no sample is near (samples are 0.3-3 s apart and
players skate too fast to interpolate between them). While the metadata is visible every panel marks the metadata carrier (as the live app
does, held CARRY_HOLD frames); inside a gap each method's own carrier is drawn, if it has one. The puck carrier gets a white triangle. The banner says whether
the metadata is visible or hidden at that moment.
usage (WIN=300|2160):  benchclip.py <t0> <dur> <out.mp4> <method>[=label] ..."""

import pickle
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
import idbench as B
import selfcal as S
from opt2 import Detections
from window import W as WIN

TEAM = {0: (255, 120, 0), 1: (0, 215, 255), -1: (230, 230, 230)}
GREEN, RED, WHITE = (60, 200, 60), (40, 40, 230), (255, 255, 255)
SNAP_TOL, HOLD, CARRY_HOLD = 4, 25, 15


def truth_track(corr):
    """{player: (frames, corrected xy)} from every snapshot of the window."""
    g, snaps = B.load_snaps()
    out = {}
    for t, d in snaps:
        N = int(round(t * B.FPS))
        for p, m in d.items():
            out.setdefault(p, ([], []))[0].append(N)
            out[p][1].append(corr(m, p))
    return {p: (np.array(a), np.array(b)) for p, (a, b) in out.items()}


def truth_at(tt, p, N):
    """Player p's metadata position if a sample lies within SNAP_TOL frames of N."""
    if p not in tt:
        return None
    Ns, xy = tt[p]
    i = np.searchsorted(Ns, N)
    for j in (i - 1, i):
        if 0 <= j < len(Ns) and abs(Ns[j] - N) <= SNAP_TOL:
            return xy[j]
    return None


def gap_clock(snapN):
    """(missing(N), seconds into the gap): the benchmark's hidden blocks on its windows; on a full-game chunk the real metadata gaps,
    frames with no snapshot within GAP_S."""
    from window import NAME

    if not NAME.startswith("c"):
        return lambda N: B.hidden_at(N / B.FPS), lambda N: B.gap_age(N / B.FPS)
    from gapid import GAP_S

    near = lambda N: np.abs(snapN - N).min() <= GAP_S * B.FPS
    return (lambda N: not near(N)), (lambda N: (N - snapN[max(np.searchsorted(snapN, N) - 1, 0)]) / B.FPS)


def panel(img, g, f, rv, rings, carrier, team_p, roster, tt, N, label, verdict, hidden, age):
    """verdict: {player: (frame, right?)} carried between calls of the same method."""
    out = img.copy()
    cxy = carrier[1] if carrier is not None else None
    for p, xy, num, *_ in rings:
        team = int(team_p[p]) if p >= 0 else -1
        ring = np.array([[xy[0] + 0.7 * np.cos(a), xy[1] + 0.7 * np.sin(a), 0.0] for a in np.linspace(0, 2 * np.pi, 28)])
        q = S.project(g, f, rv, ring)
        if not np.isfinite(q).all() or np.abs(q).max() > 4000:
            continue
        cv2.polylines(out, [q.astype(np.int32)], True, TEAM[team], 5, cv2.LINE_AA)
        head = S.project(g, f, rv, np.array([[xy[0], xy[1], 2.2]]))[0]
        if not (0 < head[0] < 1920 and 0 < head[1] < 1080):
            continue
        is_car = cxy is not None and np.linalg.norm(cxy - xy) < 0.3
        if num:
            text = ("G" if roster[p].get("goalie") else "") + str(roster[p]["jersey"])
            (tw, th), _ = cv2.getTextSize(text, 0, 1.6, 4)
            x0, y0 = int(head[0] - tw / 2), int(head[1])
            tru = truth_at(tt, p, N)
            if tru is not None:
                verdict[p] = (N, np.linalg.norm(tru - xy) < B.NEAR)
            v = verdict.get(p)
            edge = WHITE if v is None or N - v[0] > HOLD else (GREEN if v[1] else RED)
            cv2.rectangle(out, (x0 - 10, y0 - th - 10), (x0 + tw + 10, y0 + 10), TEAM[team], -1)
            cv2.rectangle(out, (x0 - 10, y0 - th - 10), (x0 + tw + 10, y0 + 10), edge, 6)
            cv2.putText(out, text, (x0, y0), 0, 1.6, (0, 0, 0), 4, cv2.LINE_AA)
            top = y0 - th - 18
        else:
            top = int(head[1])
        if is_car:
            cx = int(head[0])
            tri = np.array([[cx - 18, top - 34], [cx + 18, top - 34], [cx, top - 4]], np.int32)
            cv2.fillPoly(out, [tri], (255, 255, 255))
            cv2.polylines(out, [tri], True, (0, 0, 0), 3, cv2.LINE_AA)
    banner = f"{label}   |   metadata {'MISSING' if hidden else 'present'}" + (f" ({age:.0f}s into gap)" if hidden else "")
    cv2.rectangle(out, (0, 0), (1920, 90), (0, 0, 0) if hidden else (60, 60, 60), -1)
    cv2.putText(out, banner, (30, 64), 0, 1.9, (255, 255, 255), 4, cv2.LINE_AA)
    return out


def frames(f0, f1, F0):
    """BGR frames f0..f1-1: from the window's extracted images, or decoded from the video when the chunk's images are gone."""
    import os

    if os.path.isdir(WIN["frames"]):
        for N in range(f0, f1):
            yield N, cv2.imread(f"{WIN['frames']}/{N - F0:05d}.jpg")
        return
    from full_run import VIDEO

    p = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{(f0 - 0.5) / B.FPS:.4f}",
            "-i",
            VIDEO,
            "-frames:v",
            str(f1 - f0),
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ],
        stdout=subprocess.PIPE,
        bufsize=10**8,
    )
    for N in range(f0, f1):
        b = p.stdout.read(1920 * 1080 * 3)
        if len(b) < 1920 * 1080 * 3:
            break
        yield N, np.frombuffer(b, np.uint8).reshape(1080, 1920, 3)
    p.kill()
    p.wait()


def main(t0, dur, path, specs):
    DT = Detections()
    corr = B.truth_corr()
    tt = truth_track(corr)
    g = B.load_snaps()[0]
    roster = g["roster"]
    team_p = np.array([0 if r["team"] == "Home" else 1 for r in roster])
    ms = []
    for s in specs:
        name, label = (s.split("=", 1) + [s])[:2]
        R = pickle.load(open(f"{B.OUT}/{name}.pkl", "rb"))
        ms.append((R["rings"], R["carrier"] or {}, label))
    n = len(ms)
    W_, H_ = (1920, 540) if n == 2 else (1920, 1080)
    w = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{W_}x{H_}",
            "-r",
            "25",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            path,
        ],
        stdin=subprocess.PIPE,
    )
    f0, f1 = int(round(t0 * B.FPS)), int(round((t0 + dur) * B.FPS))
    last = None
    verdicts = [{} for _ in ms]
    car = B.carriers()
    carN = np.array(sorted(car))
    missing, age = gap_clock(np.array(sorted(int(round(t * B.FPS)) for t, _ in B.load_snaps()[1])))
    for N, img in frames(f0, f1, DT.F0):
        k = DT.rowN.get(N, last)
        last = k
        f, rv = DT.Z["f"][k], DT.Z["rv"][k]
        meta_car = None
        hid = missing(N)
        if not hid:
            j = np.searchsorted(carN, N, "right") - 1
            if j >= 0 and N - carN[j] <= CARRY_HOLD:
                meta_car = car[carN[j]]
        ps = []
        for (r, c, lab), v in zip(ms, verdicts):
            rs = r.get(N, [])
            cc = c.get(N)
            if meta_car is not None:
                pos = truth_at(tt, meta_car, carN[j])
                near = min(rs, key=lambda e: np.linalg.norm(e[1] - pos)) if rs and pos is not None else None
                cc = (meta_car, near[1]) if near is not None and np.linalg.norm(near[1] - pos) < B.NEAR else None
            ps.append(panel(img, DT.g, f, rv, rs, cc, team_p, roster, tt, N, lab, v, hid, age(N) if hid else 0))
        if n == 1:
            frame = ps[0]
        else:
            small = [cv2.resize(p, (960, 540), interpolation=cv2.INTER_AREA) for p in ps] + [
                np.zeros((540, 960, 3), np.uint8)
            ] * (4 - n)
            frame = np.hstack(small[:2]) if n == 2 else np.vstack([np.hstack(small[:2]), np.hstack(small[2:4])])
        w.stdin.write(frame.tobytes())
    w.stdin.close()
    w.wait()
    print("wrote", path)


if __name__ == "__main__":
    main(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4:])
