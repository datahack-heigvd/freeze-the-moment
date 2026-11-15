"""Where the pipeline's inputs and outputs live. STORE, MODELS and GOALS can be moved with environment variables of the same name."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = f"{ROOT}/pipeline"
STORE = os.environ.get("STORE", f"{ROOT}/store")
MODELS = os.environ.get("MODELS", f"{STORE}/models")  # model weights and the match's kits
GOALS = os.environ.get("GOALS", f"{PIPELINE}/goals")  # one dir per goal, each with its bundle/
AR, SCENES = f"{STORE}/ar", f"{STORE}/scenes"  # the AR export (pipeline/calib) and the built goal scenes
VIDEO = f"{STORE}/data/20251231_115035_SpenglerCup_Kam1 31.12.25 Game 11.mp4"
GAME = f"{STORE}/site/game.json"  # the match's feed: roster, events, tracking frames
SAM3D = f"{PIPELINE}/sam-3d-body"  # github.com/facebookresearch/sam-3d-body, cloned here
