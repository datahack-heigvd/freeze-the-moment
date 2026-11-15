# Freeze the moment

The goals of the Spengler Cup 2025 final, rebuilt in 3D from the TV broadcast alone. Stop time inside a goal, walk around it, or
see it through a player's eyes, in the browser. Made for the Data-hack 2026 challenge.

The match video and data are confidential: they and everything made from them are in a private bucket, not in this repository.
The access kit sent with the submission has the keys.

## Run the site

Needs [Bun](https://bun.sh), [uv](https://docs.astral.sh/uv/) and [just](https://just.systems).

```sh
cp <access kit>/.env .
just demo
```

This downloads the data (about 15 GB) and opens the site on http://localhost:8000.

## Run the pipeline

Needs a CUDA machine, the GPU environments in `.env` (`LTX_PY`, `YOLO_PY`, `SAM_PY`, `IMAGE_PY`, `GPU_HOST`) and `ffmpeg` with rubberband.

```sh
just install-pipeline
just pull     # every output
just repro    # rebuild what changed (stages in dvc.yaml)
```

## Structure

```
app/        the site: Bun server, React, three.js
pipeline/   broadcast to 3D: camera, players, bodies, sticks, puck, arena
store/      data and outputs, pulled from the bucket (not in git)
```

`just` lists every recipe. Credits: [CREDITS.md](CREDITS.md). Licence: [MIT](LICENSE), for the code only (not the match data).
