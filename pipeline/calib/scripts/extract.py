"""Extract frames at a fixed rate (960x540 jpg) for a time range into cache/frames/<name>/"""

import pathlib
import subprocess
import sys

VIDEO = "../../store/data/20251231_115035_SpenglerCup_Kam1 31.12.25 Game 11.mp4"


def extract(start, dur, fps, out):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            str(start),
            "-t",
            str(dur),
            "-i",
            VIDEO,
            "-vf",
            f"fps={fps},scale=960:540",
            "-q:v",
            "3",
            "-start_number",
            "0",
            str(out / "%06d.jpg"),
        ],
        check=True,
    )


if __name__ == "__main__":
    start, dur, fps, out = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    extract(start, dur, fps, out)
