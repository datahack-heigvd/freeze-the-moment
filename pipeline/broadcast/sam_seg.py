"""SAM 3.1 outlines of everyone on the ice for many short windows (GPU, SAM venv).

Each window's frames are one video session; "hockey player" and "referee" are tracked through it with persistent object ids (the
largest contour of each mask). SAM 3.1 fits people's edges much more tightly than YOLO11x-seg and flickers less
(seg_compare.py on the 2025 goal clip: edge fit 1.98 vs 1.21, 0.50 vs 0.67 flips per second).

  <win>/sam3.json   per frame [{obj, prompt, poly}] in source pixels

usage:  sam_seg.py <win_dir>...     (skips windows that already have sam3.json)"""

import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from sam3.model_builder import build_sam3_multiplex_video_predictor

PROMPTS = ("hockey player", "referee")


def main(dirs):
    pred = build_sam3_multiplex_video_predictor(
        checkpoint_path=os.path.expanduser(f"{os.environ.get('MODELS', '~/store/models')}/sam3.1/sam3.1_multiplex.pt"),
        max_num_objects=32,
        use_fa3=False,
    )
    init = pred.model.init_state  # the multiplex model does not take the base predictor's offload_state_to_cpu
    pred.model.init_state = lambda offload_state_to_cpu=False, **kw: init(**kw)
    t0 = time.time()
    for k, d in enumerate(dirs):
        if os.path.exists(f"{d}/sam3.json"):
            continue
        n = len(os.listdir(f"{d}/frames"))
        frames = [[] for _ in range(n)]
        sid = pred.handle_request(dict(type="start_session", resource_path=os.path.abspath(f"{d}/frames")))["session_id"]
        for p, text in enumerate(PROMPTS):
            pred.handle_request(dict(type="reset_session", session_id=sid))
            pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=text))
            try:
                with torch.no_grad():
                    for r in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
                        o = r["outputs"]
                        for oid, m in zip(o["out_obj_ids"].tolist(), o["out_binary_masks"]):
                            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if not cs:
                                continue
                            c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.0, True)[:, 0]
                            if len(c) >= 3:
                                frames[r["frame_index"]].append(dict(obj=p * 1000 + oid, prompt=text, poly=c.tolist()))
            except RuntimeError as e:  # the prompt found nobody in the first frame (e.g. no referee in view)
                if "No points are provided" not in str(e):
                    raise
        pred.handle_request(dict(type="close_session", session_id=sid))
        json.dump(frames, open(f"{d}/sam3.json", "w"))
        print(
            f"{k + 1}/{len(dirs)} {os.path.basename(d)} masks/frame {np.mean([len(f) for f in frames]):.1f} {time.time() - t0:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
