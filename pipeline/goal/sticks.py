"""Hockey sticks in the source frames with SAM 3.1: the prompt "hockey stick" tracked through the bundle's frames (GPU, SAM venv,
with python -I so the sam3 checkout next to this file does not shadow the package). render.py lifts each outline into 3D from the
body it belongs to.

  bundle/sticks.json   per frame [{obj, poly}] in source pixels

usage (from pipeline/):  sticks.py [bundle...]   (default: $BUNDLE or bundle; the model loads once for all of them;
          processes sharing a QUEUE directory take the bundles one at a time)"""

import json
import os
import sys

import cv2
import numpy as np
import torch
from paths import MODELS
from sam3.model_builder import build_sam3_multiplex_video_predictor


def sticks(pred, B):
    n = len(os.listdir(f"{B}/frames"))
    frames = [[] for _ in range(n)]
    sid = pred.handle_request(dict(type="start_session", resource_path=os.path.abspath(f"{B}/frames")))["session_id"]
    try:
        pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text="hockey stick"))
        with torch.no_grad():
            for r in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
                o = r["outputs"]
                for oid, m in zip(o["out_obj_ids"].tolist(), o["out_binary_masks"]):
                    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not cs:
                        continue
                    c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.0, True)[:, 0]
                    if len(c) >= 3:
                        frames[r["frame_index"]].append(dict(obj=oid, poly=c.tolist()))
    except RuntimeError as e:  # no stick in the first frame
        if "No points are provided" not in str(e):
            raise
    finally:
        pred.handle_request(dict(type="close_session", session_id=sid))
    json.dump(frames, open(f"{B}/sticks.json", "w"))
    print(B, f"sticks/frame {np.mean([len(f) for f in frames]):.1f}", flush=True)


def mine(B):
    """Claims a bundle in the shared $QUEUE directory, so processes given the same bundles split them as they go; without one, all are mine."""
    if "QUEUE" not in os.environ:
        return True
    try:
        os.mkdir(os.path.join(os.environ["QUEUE"], B.strip("/").replace("/", "_")))
        return True
    except FileExistsError:
        return False


def main(bundles):
    pred = build_sam3_multiplex_video_predictor(
        checkpoint_path=f"{MODELS}/sam3.1/sam3.1_multiplex.pt", max_num_objects=32, use_fa3=False
    )
    init = pred.model.init_state  # the multiplex model does not take the base predictor's offload_state_to_cpu
    pred.model.init_state = lambda offload_state_to_cpu=False, **kw: init(**kw)
    failed = 0
    for B in filter(mine, bundles):
        try:
            sticks(pred, B)
        except Exception as e:
            failed += 1
            print(B, "sticks FAILED:", repr(e)[:300], flush=True)
    sys.exit(failed > 0)


if __name__ == "__main__":
    main(sys.argv[1:] or [os.environ.get("BUNDLE", "bundle")])
