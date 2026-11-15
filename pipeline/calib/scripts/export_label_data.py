import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from rink import crease, goal_frame, lines

G = np.load("cache/calib_global.npz")
HB = float(G["hb"])
cam = G["cams"][0]
L = {k: v.tolist() for k, v in lines().items()}
out = {
    "lines": L,
    "goals": {str(s): [p.tolist() for p in goal_frame(s)] for s in (1, -1)},
    "creases": {str(s): crease(s).tolist() for s in (1, -1)},
    "cam": {"C": cam[4:7].tolist(), "k1": float(cam[7]), "cxo": float(cam[8]), "cyo": float(cam[9]), "fref": 3050.0},
}
# initial poses (time, f, rotvec) from the bundle adjustment of the first 300 s
B = np.load("cache/cam_ba300.npz")
out["init"] = {"t": B["t"][::4].tolist(), "f": B["f"][::4].tolist(), "rv": B["rv"][::4].tolist()}
json.dump(out, open("label/rink.json", "w"))
print("ok", len(L), "lines", len(out["init"]["t"]), "init poses")
