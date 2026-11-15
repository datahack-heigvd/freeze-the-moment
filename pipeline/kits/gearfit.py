"""Equipment thickness per body part, learned from the footage: the outward offset (m, along the vertex normals) of each GEAR group that
makes everyone, as render.py places them (Cast, BODIES), best match the real outlines in the source camera (soft silhouette IoU through
nvdiffrast, stick outlines ignored). Prints the offsets for render.py's GEAR, with the IoU of the bare bodies, the current GEAR and the
learned one, and writes them (at least 0) to <bundle>/gear_fit.json under BODIES for scene_export.py. ERODE=n (default 1) shrinks the
real outlines by n px (of the 960 x 540 silhouettes) first: the broadcast's blur and the segmentation widen them, which the equipment
would otherwise absorb (at 1 the helmet and glove come out at their real thickness, 1.5 and 3.2 cm, on three goals).

usage (LTX venv, BODIES as for render.py):  [ERODE=n] gearfit.py [iters]"""

import json
import os
import sys

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
from goal import render
from goal.render import DEV, GEAR, B, Cast, S, Z, at, gear_offsets, src_project, vertex_normals

W, H, STEP = 960, 540, 2  # silhouette size, every STEP-th frame
ERODE = int(os.environ.get("ERODE", 1))


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    T = json.load(open(f"{B}/cast.json"))["tracks"]
    K = json.load(open(f"{B}/sticks.json"))
    sc = [W / 1920, H / 1080]
    cast = Cast()
    faces = cast.faces
    ctx = dr.RasterizeCudaContext()

    # per-vertex weight of each group: gear_offsets with that group alone at 1 m, so the blend over the mesh is render.py's own
    v0, j0 = cast.gear_reference
    A = []
    for idx, _ in GEAR:
        render.GEAR = [(idx, 1.0)]
        A.append(gear_offsets(v0, j0, faces))
    render.GEAR = GEAR
    A = torch.stack(A, 1)

    frames = []
    for i in range(0, S["n"], STEP):
        ks = [t["key"] for t in T if cast.drawn(t, i)]
        if not ks:
            continue
        real, care = np.zeros((H, W), np.uint8), np.ones((H, W), np.uint8)
        for t in T:
            if t["mask"][i] >= 0:
                cv2.fillPoly(real, [np.round(np.array(S["outlines"][i][t["mask"][i]]) * sc).astype(np.int32)], 1)
        for k in K[i]:
            cv2.fillPoly(care, [np.round(np.array(k["poly"]) * sc).astype(np.int32)], 0)
        if ERODE:
            real = cv2.erode(real, np.ones((2 * ERODE + 1, 2 * ERODE + 1), np.uint8))
        v = torch.stack([at(cast.verts[k], i) for k in ks])
        vn = torch.stack([vertex_normals(x, faces) for x in v])
        frames.append(
            (
                v,
                vn,
                Z["cam"][i].astype(np.float64),
                torch.from_numpy(real).float().to(DEV),
                torch.from_numpy(care).float().to(DEV),
            )
        )
    tri = {n: torch.cat([faces + j * v0.shape[0] for j in range(n)]).contiguous() for n in {f[0].shape[0] for f in frames}}

    def cover(f, off):
        v, vn, cam, real, care = f
        uv, z = src_project(cam, (v + vn * (A @ off)[None, :, None]).reshape(-1, 3))
        clip = torch.stack([(uv[:, 0] / 960 - 1) * z, (uv[:, 1] / 540 - 1) * z, z * 0.5, z], 1)[None].contiguous()
        t = tri[v.shape[0]]
        rast, _ = dr.rasterize(ctx, clip, t, (H, W))
        return dr.antialias((rast[..., 3:] > 0).float(), rast, clip, t)[0, ..., 0], real, care

    def iou(off, hard=False):
        inter = union = 0.0
        for f in frames:
            p, r, m = cover(f, off)
            if hard:
                p = (p > 0.5).float()
            inter = inter + (p * r * m).sum()
            union = union + ((p + r - p * r) * m).sum()
        return inter / union

    with torch.no_grad():
        print(
            f"frames {len(frames)}; IoU bare {float(iou(torch.zeros(len(GEAR), device=DEV), True)):.3f}, "
            f"current GEAR {float(iou(torch.tensor([o for _, o in GEAR], device=DEV), True)):.3f}",
            flush=True,
        )
    off = torch.tensor([o for _, o in GEAR], device=DEV, requires_grad=True)
    opt = torch.optim.Adam([off], lr=3e-3)
    for it in range(iters):
        opt.zero_grad()
        loss = 1 - iou(off)
        loss.backward()
        opt.step()
        if it % 25 == 0:
            print(
                f"  iter {it}: soft IoU {1 - float(loss):.3f}  cm " + " ".join(f"{o * 100:.1f}" for o in off.tolist()), flush=True
            )
    with torch.no_grad():
        print(f"IoU learned {float(iou(off, True)):.3f}; GEAR offsets (m): " + ", ".join(f"{o:.3f}" for o in off.tolist()))
    path = f"{B}/gear_fit.json"
    fits = json.load(open(path)) if os.path.exists(path) else {}
    fits[os.environ.get("BODIES", "bodies")] = [round(max(o, 0.0), 4) for o in off.tolist()]
    json.dump(fits, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
