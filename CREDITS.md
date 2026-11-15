# Credits

## Data

- Spengler Cup 2025 final: TV broadcast (camera 1) and Wisehockey match data, provided by the organisers for the Data-hack 2026
  challenge only. Not in this repository.
- Tournament results: [Wikipedia](https://en.wikipedia.org/wiki/2025_Spengler_Cup) (CC BY-SA 4.0) and
  [spenglercup.ch](https://www.spenglercup.ch/en/tournament/games-standing).
- Team and tournament logos: their owners.

## Models

- HockeyAI (Simula, ACM MMSys 2025, YOLOv8): player, referee and puck detection.
- [SAM 3](https://github.com/facebookresearch/sam3) and [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) with
  the MHR body model (Meta): outlines, sticks and 3D bodies.
- [YOLO11](https://github.com/ultralytics/ultralytics) (Ultralytics, AGPL-3.0): detection, segmentation and pose; the puck
  detector is fine-tuned by us.
- [Qwen-Image](https://github.com/QwenLM/Qwen-Image) (Alibaba): clean-up of the arena textures.
- Qwen3 vision-language model (Alibaba): reading the score bug of a broadcast.
- [UltraSharpV2](https://huggingface.co/Kim2091/UltraSharpV2) (Kim2091): sharpening of the arena textures.
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) small: depth for the moment cards.
- SIFT ([OpenCV](https://opencv.org)): camera calibration.

## Software

[three.js](https://threejs.org), [React](https://react.dev), [React Router](https://reactrouter.com),
[Tailwind CSS](https://tailwindcss.com), [d3](https://d3js.org), [Bun](https://bun.sh), [PyTorch](https://pytorch.org),
[DVC](https://dvc.org), [FFmpeg](https://ffmpeg.org) with Rubber Band.

## Fonts

[Inter](https://rsms.me/inter/) (Rasmus Andersson) and [Oswald](https://fonts.google.com/specimen/Oswald) (Vernon Adams), both
SIL Open Font License 1.1.
