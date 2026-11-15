set dotenv-load

export PYTHONPATH := justfile_directory() / "pipeline"
calib_py := justfile_directory() / "pipeline/calib/.venv/bin/python"
ltx_py := env("LTX_PY", justfile_directory() / "pipeline/LTX-2/.venv/bin/python")
yolo_py := env("YOLO_PY", justfile_directory() / "pipeline/.venv-yolo/bin/python")
sam_py := env("SAM_PY", "python")
image_py := env("IMAGE_PY", "python")
export GOALS := env("GOALS", justfile_directory() / "pipeline/goals")
store := justfile_directory() / "store"
p := justfile_directory() / "pipeline"

# the deck, the video and publishing, kept out of the public repository
import? 'submission.just'

# list recipes
default:
    @just --list

# the site's dependencies and dvc
install:
    cd app && bun install
    uv sync

# the pipeline's own environment, on top of install (the GPU environments come from .env)
install-pipeline: install
    cd pipeline/calib && uv sync

# the site with live reload
dev port="8000":
    cd app && PORT={{port}} bun run dev

# the site, built for production
start port="8000":
    cd app && PORT={{port}} bun run start

# everything the site needs, then the site: the first run with the access kit's .env
demo: install pull-site start

# type-check and lint the site and the pipeline
check:
    cd app && bun run typecheck && bunx biome check .
    uv run ruff check
    uv run ruff format --check

# format and fix what can be fixed safely
fmt:
    cd app && bunx biome check --write .
    uv run ruff check --fix
    uv run ruff format

# the pipeline's unit tests
test:
    {{calib_py}} -m unittest discover -s pipeline/checks -p 'test_*.py'

# rebuild whatever changed, as dvc.yaml lays it out (`just repro scene@zadina` for one stage)
repro *stages:
    uv run dvc repro {{stages}}

# upload the pipeline's outputs to the bucket
push *stages:
    uv run dvc push {{stages}}

# fetch the pipeline's outputs from the bucket
pull *stages:
    uv run dvc pull {{stages}}

# fetch only what the site serves: the match video, the AR export, the arena, the goal scenes and the site's data
pull-site:
    uv run dvc pull store.dvc ar arena scene game moments audio how

# the match feed (store/site/game.json) and the players' stats from the Wisehockey exports
game:
    cd app && bun scripts/build-data.ts && bun scripts/build-players.ts

# the feed's goal moments and hover strips, from the video and the built scenes
moments:
    cd app && bun scripts/build-moments.ts && bun scripts/build-orbits.ts

# each goal's broadcast sound and its slowed copies, from the video (needs ffmpeg with rubberband)
audio:
    cd app && bun scripts/build-audio.ts

# the "How it works" page's frames, cameras and outlines for one goal into store/site/how (CPU)
how goal="musa":
    cd {{GOALS}}/{{goal}} && CUDA_VISIBLE_DEVICES= BUNDLE=bundle {{ltx_py}} {{p}}/export/how_export.py {{store}}/site/how/{{goal}}

# the goal windows (pipeline/goals.json) from the feed's goal events
windows:
    uv run {{p}}/goal/windows.py

# whole-game calibration, tracking and AR export into store/ar (`just calib plan` lists the chunks)
calib *args="all":
    cd pipeline/calib && uv run scripts/full_run.py {{args}}

# a goal's bundle: the window's frames, cameras and outlines, then the cast (start: s into the video)
bundle goal start dur="10":
    mkdir -p {{GOALS}}/{{goal}}
    cd {{GOALS}}/{{goal}} && BUNDLE=bundle {{calib_py}} {{p}}/goal/prep.py {{start}} {{dur}} && BUNDLE=bundle {{calib_py}} {{p}}/goal/cast.py

# puck candidates in every frame of a goal (GPU)
detect goal:
    cd {{GOALS}}/{{goal}} && BUNDLE=bundle {{yolo_py}} {{p}}/goal/puck.py detect

# 3D bodies for a goal's cast (GPU)
bodies goal:
    cd {{GOALS}}/{{goal}} && {{sam_py}} {{p}}/goal/body.py bundle

# stick outlines in a goal's frames (GPU)
sticks goal:
    cd {{GOALS}}/{{goal}} && {{sam_py}} {{p}}/goal/sticks.py bundle

# the arena atlas fused from every goal's calibrated views (GPU; any goal's bundle gives the rink)
atlas goal="zadina":
    BUNDLE={{GOALS}}/{{goal}}/bundle {{ltx_py}} {{p}}/arena/arena_atlas.py build {{store}}/atlas

# the arena the viewer shows: the atlas refined, its scoreboard made a cube, the stands quilted, the ice's glare put on the roof (GPU)
arena goal="zadina":
    {{image_py}} {{p}}/arena/atlas_refine.py {{store}}/atlas {{store}}/scenes/arena all
    {{ltx_py}} {{p}}/arena/scoreboard.py face {{store}}/scenes/arena
    {{image_py}} {{p}}/arena/scoreboard.py inpaint {{store}}/scenes/arena
    {{ltx_py}} {{p}}/arena/stands_quilt.py {{store}}/scenes/arena
    {{ltx_py}} {{p}}/arena/ice_glare.py {{store}}/scenes/arena {{GOALS}}/{{goal}}/bundle/scene.json

# a goal's 3D scene into store/scenes/<goal>
scene goal:
    uv run {{p}}/goal/scene_build.py {{GOALS}}/{{goal}} {{store}}/scenes/{{goal}} {{store}}/scenes/arena

# a year's Spengler Cup broadcasts into store/data/thirdparty
fetch-broadcasts year:
    mkdir -p {{store}}/data/thirdparty/spengler-cup-{{year}}
    cd {{store}}/data/thirdparty/spengler-cup-{{year}} && uv tool run --from 'yt-dlp[default]' yt-dlp --ignore-config \
        --batch-file {{p}}/broadcast/urls/{{year}}.txt --format 'bestvideo+bestaudio/best' \
        --merge-output-format mkv --download-archive downloaded.txt --write-info-json --continue --no-overwrites \
        --ignore-errors --retries 10 --fragment-retries 10 --output '%(title)s [%(id)s].%(ext)s'
