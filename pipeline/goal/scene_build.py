"""A goal's viewer scene from its bundle (prep.py, cast.py, puck.py detect, body.py and sticks.py done), every step a script:
  1. the puck from its detections, and the planned shots (director.py);
  2. how well each body lines up with the footage (align.py: people off their outline are exported as unseen);
  3. the export (equipment fitted to the goal's outlines when gearfit.py has run, sticks fitted to their outlines);
     the release found from the scorer's stick when nobody read it by eye, then the puck and the export again;
  4. the match's kits (KITS, kit.py), baked there from this goal's own clip when it has none yet (a match without a feed);
  5. the arena (the atlas dir's atlas.json, its images made relative to the scene) and the planned cameras cleared of its scoreboard;
  6. the teams' names and logos, found once per match (team_logos.py: from the feed's game.json, else the YouTube video's title,
     the goal dir's prefix), and the body textures in the match's kits;
  7. the reel, the broadcast's clip with its camera per frame for the submission film (reel_export.py, into <out>/reel);
  8. the broadcast camera re-solved against the arena and everything placed through it moved to stay on the picture (resolve.py).
Forecast arcs come from research/forecast/forecast.py and are kept.

usage:  scene_build.py <goal dir> <out dir> <atlas dir>   e.g. scene_build.py goals/zadina ../store/scenes/zadina ../store/scenes/arena
  LTX_PY, YOLO_PY: the two pythons (torch and SAM 3D Body's stack; Ultralytics); BODIES (bodies), RES (1280x768), KITS"""

import json
import os
import subprocess
import sys

from paths import GAME, MODELS, PIPELINE

LTX = os.environ.get("LTX_PY", f"{PIPELINE}/LTX-2/.venv/bin/python")
YOLO = os.environ.get("YOLO_PY", f"{PIPELINE}/.venv-yolo/bin/python")


def run(py, script, *args, quiet=True, cwd=None, **env):
    """One step; its output is dropped unless the step fails."""
    p = subprocess.run([py, f"{PIPELINE}/{script}", *args], capture_output=quiet, text=True, cwd=cwd, env={**os.environ, **env})
    if p.returncode:
        sys.exit(f"{script} failed:\n{p.stdout or ''}{p.stderr or ''}")
    return p.stdout


def read(path):
    return open(path).read() if os.path.exists(path) else None


def build(goal, out, atlas):
    goal, out, atlas = map(os.path.abspath, (goal, out, atlas))
    os.environ.update(
        BUNDLE="bundle",
        BODIES=os.environ.get("BODIES", "bodies"),
        RES=os.environ.get("RES", "1280x768"),
        NUMBERS="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        PYTHONPATH=PIPELINE,
    )
    os.makedirs(out, exist_ok=True)
    os.chdir(goal)
    run(YOLO, "goal/puck.py")
    run(YOLO, "goal/director.py")
    print(run(LTX, "goal/align.py").strip().splitlines()[-1])
    run(LTX, "goal/scene_export.py", out)
    if not os.path.exists("bundle/release.json"):
        before = read("bundle/release_auto.json")
        run(LTX, "goal/release.py", out, quiet=False)
        if read("bundle/release_auto.json") not in (None, before):
            run(YOLO, "goal/puck.py")
            run(LTX, "goal/scene_export.py", out)
    kits = os.environ.get("KITS")
    if kits and not os.path.exists(f"{kits}/kits.npz"):
        os.makedirs(kits, exist_ok=True)
        run(LTX, "kits/kit.py", "bake", f"{goal}/bundle")
        run(LTX, "kits/kit.py", "votes", f"{goal}/bundle")
        run(LTX, "kits/gear_colour.py", f"{goal}/bundle", cwd=PIPELINE)
    meta = json.load(open(f"{atlas}/atlas.json"))
    for v in meta.values():  # every image of every part
        if isinstance(v, dict):
            for k in ("file", "plain", "matte"):
                if k in v:
                    v[k] = os.path.relpath(f"{atlas}/{v[k]}", out)
    json.dump(meta, open(f"{out}/atlas.json", "w"), indent=1)
    run(LTX, "arena/scoreboard.py", "clear", out)
    kits = kits or MODELS
    if not os.path.exists(f"{kits}/teams.json"):
        source = (
            ["game", GAME]
            if json.load(open("bundle/scene.json"))["roster"]
            else ["youtube", os.path.basename(goal).rsplit("_", 1)[0]]
        )
        if subprocess.run([LTX, f"{PIPELINE}/kits/team_logos.py", kits, *source]).returncode:
            print("team logos: none this time")
    run(LTX, "kits/player_textures.py", out)
    run(LTX, "export/reel_export.py", f"{out}/reel", CUDA_VISIBLE_DEVICES="")
    print(run(LTX, "goal/resolve.py", out).strip().splitlines()[-2])
    print(f"{goal} -> {out}")


if __name__ == "__main__":
    build(*sys.argv[1:4])
