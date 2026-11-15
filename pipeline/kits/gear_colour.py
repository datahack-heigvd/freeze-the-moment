"""The equipment's colour in the footage: helmets per team for skaters, goalies (masks) and the officials, and goalies' leg pads and
gloves. The
part's vertices on everyone fitted (MHR rest mesh: the cranium, within CRANIUM of the top; the pads, gear.py), projected into the
broadcast frames (every STEP-th) where they are the visible surface inside the person's own outline and face the camera squarely (cos
above CORE: the middle of the part, away from its blurred edge); per person and frame the median of those pixels (MIN_PX or more),
and per team the median over the people the camera shows at least WIDE px from ear to ear. Smaller ones are blurred with the ice
around them into a lighter, greyer colour (US skaters' helmets (85, 85, 93) over everyone; (68, 69, 78) on the wide ones, whose
crops show black helmets).

  <KITS>/gear_colours.json    {"<team>" (skaters' helmets, officials -1), "<team>g" (goalies' masks), "<team>g_pads", "<team>g_gloves": [r, g, b]
                              (sRGB 0-255); "<key>_seen": count}

usage (LTX venv, from pipeline/):  gear_colour.py <bundle>...   e.g. goals/*/bundle"""

import json
import sys

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
from kits.gear import EARS, gloved, padded
from kits.kit import DEV, FACES, KITS, UV, ctx, project, vertex_normals

CRANIUM, STEP, CORE, MIN_PX, WIDE = 0.07, 2, 0.8, 6, 16

rest = UV["rest"] / 100
top = rest[:, 1].max()
PARTS = {
    "": torch.from_numpy(np.flatnonzero(rest[:, 1] > top - CRANIUM)).to(DEV),
    "_pads": torch.from_numpy(np.flatnonzero(padded(rest))).to(DEV),
    "_gloves": torch.from_numpy(np.flatnonzero(gloved(rest, rest))).to(DEV),
}
out = {}
for w in sys.argv[1:]:
    S = json.load(open(f"{w}/scene.json"))
    cams = np.load(f"{w}/scene.npz")["cam"].astype(np.float64)
    tracks = {t["key"]: t for t in json.load(open(f"{w}/cast.json"))["tracks"]}
    b = np.load(f"{w}/bodies.npz")
    fr, keys, verts = b["frame"], b["key"], b["verts"]
    for i in np.unique(fr)[::STEP]:
        rows = [r for r in np.flatnonzero(fr == i) if tracks[int(keys[r])]["mask"][i] >= 0]
        if not rows:
            continue
        c = cams[i]
        C = torch.tensor(c[11:14], device=DEV, dtype=torch.float32)
        N = verts.shape[1]
        V = torch.from_numpy(verts[rows].astype(np.float32)).to(DEV)
        P = V.reshape(-1, 3)
        px, z = project(c, S, P)
        d = (P - C).norm(dim=1)
        clip = torch.stack([(px[:, 0] / 960 - 1) * z, (px[:, 1] / 540 - 1) * z, (d / 100 - 1) * z, z], 1)[None].contiguous()
        faces = torch.cat([FACES + j * N for j in range(len(rows))])
        rast, _ = dr.rasterize(ctx, clip, faces, (1080, 1920))
        zbuf = torch.where(
            rast[0, ..., 3] > 0,
            dr.interpolate(d[None, :, None].contiguous(), rast, faces)[0][0, ..., 0],
            torch.full_like(rast[0, ..., 0], 1e9),
        )
        label = np.zeros((1080, 1920), np.uint8)
        ring = np.ones((3, 3), np.uint8)
        for j, r in enumerate(rows):
            t = tracks[int(keys[r])]
            cv2.fillPoly(label, [np.round(np.array(S["outlines"][i][t["mask"][i]])).astype(np.int32)], j + 1)
        label = torch.from_numpy(cv2.erode(label, ring)).to(DEV)
        img = cv2.cvtColor(cv2.imread(f"{w}/frames/{i + 1:03d}.jpg"), cv2.COLOR_BGR2RGB)
        for j, r in enumerate(rows):
            t = tracks[int(keys[r])]
            if float((px[j * N + EARS[0]] - px[j * N + EARS[1]]).norm()) < WIDE:
                continue
            for part, CR in PARTS.items():
                if part and not t["goalie"]:
                    continue
                X = V[j][CR]
                n = vertex_normals(V[j])[CR]
                q = px[j * N + CR]
                dist = (X - C).norm(dim=-1)
                qi = q.round().long()
                ok = (qi[:, 0] >= 0) & (qi[:, 0] < 1920) & (qi[:, 1] >= 0) & (qi[:, 1] < 1080)
                qx, qy = qi[:, 0].clamp(0, 1919), qi[:, 1].clamp(0, 1079)
                cos = -(n * (X - C) / dist[:, None]).sum(-1)
                vis = ok & (dist <= zbuf[qy, qx] + 0.03) & (label[qy, qx] == j + 1) & (cos > CORE)
                if vis.sum() < MIN_PX:
                    continue
                qv = qi[vis].cpu().numpy()
                pix = img[qv[:, 1], qv[:, 0]]
                out.setdefault(f"{t['team']}{'g' if t['goalie'] else ''}{part}", []).append(np.median(pix, 0))
res = {}
for team, v in out.items():
    v = np.array(v)
    res[team] = np.median(v, 0).round().tolist()
    res[f"{team}_seen"] = len(v)
    print(team, "seen", len(v), "median", res[team], "quartiles", np.quantile(v, [0.25, 0.75], 0).round().tolist())
json.dump(res, open(f"{KITS}/gear_colours.json", "w"), indent=1)
