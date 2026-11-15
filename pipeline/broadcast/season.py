"""Every downloaded game of a Spengler Cup season through the shot sort (shots.py, three games at a time) or the per-frame cameras
(cams.py, 60 windows each, one game at a time, waiting for each game's shot sort), into out/<prefix><N>/ with a log beside it.

usage (calib venv):  season.py sort <season> <prefix>      e.g. season.py sort 2024 g, season.py sort 2023 y23g
                     season.py cams <season> <prefix>"""

import glob
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from paths import PIPELINE, STORE

HERE = f"{PIPELINE}/broadcast"


def games(season):
    """{game number: video} of the season's downloaded broadcasts."""
    files = glob.glob(f"{STORE}/data/thirdparty/spengler-cup-{season}/*Game [0-9]*.mkv") + glob.glob(
        f"{STORE}/data/thirdparty/spengler-cup-{season}/*Game [0-9]*.mp4"
    )
    return {int(re.search(r"Game (\d+) ", f).group(1)): f for f in sorted(files)}


def run(script, args, name):
    """Run a broadcast script into its log; print the log's last line."""
    log = f"{HERE}/{script.removesuffix('.py')}_{name}.log"
    with open(log, "w") as out:
        subprocess.run([sys.executable, "-u", f"{HERE}/{script}", *args], stdout=out, stderr=subprocess.STDOUT, cwd=HERE)
    print(name, open(log).read().strip().splitlines()[-1:])


def sort(season, prefix):
    todo = [(n, f) for n, f in games(season).items() if not os.path.exists(f"{HERE}/out/{prefix}{n}/shots.json")]
    for n, _ in todo:
        os.makedirs(f"{HERE}/out/{prefix}{n}", exist_ok=True)
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(lambda g: run("shots.py", [g[1], f"{HERE}/out/{prefix}{g[0]}"], f"{prefix}{g[0]}"), todo))


def cams(season, prefix):
    for n, f in games(season).items():
        out = f"{HERE}/out/{prefix}{n}"
        while not os.path.exists(f"{out}/shots.json"):
            time.sleep(30)
        if not os.path.exists(f"{out}/windows.json"):
            run("cams.py", [f, out, "60", "4"], f"{prefix}{n}")


if __name__ == "__main__":
    {"sort": sort, "cams": cams}[sys.argv[1]](*sys.argv[2:4])
