"""3D bodies for the cast with SAM 3D Body: one mesh per person and frame, placed in the rink frame (GPU).

Each frame is undistorted to an ideal pinhole camera with the calibrated focal length, so the model gets the exact intrinsics.
The outline of each cast member is its box and mask prompt. The camera-frame mesh is moved into the rink frame with the calibrated
extrinsics, then scaled about the camera centre (which keeps its projection) so its lowest point stands on the ice.

  bundle/bodies.npz  frame [M], key [M], verts [M,V,3] float16 (rink frame), joints [M,J,3], scale [M], faces [F,3], and the model's
                     parameters per fit for fit.py: global_rot [M,3], pose_raw [M,6+body], hand [M,H], shape [M,S], scale_params [M,K],
                     cam_t [M,3] (undistorted camera frame, before the ice scaling), kp2d [M,70,2] (undistorted source pixels)
  check/bodies_NNN.jpg  meshes projected back onto the source frame

usage (from pipeline/, with the SAM venv):  body.py [bundle...]   (default: $BUNDLE or bundle; the model loads once for all of them;
          processes sharing a QUEUE directory take the bundles one at a time)"""

import json
import os
import sys

import cv2
import numpy as np
import torch
from paths import MODELS, SAM3D

MODEL = f"{MODELS}/sam-3d-body-dinov3"
sys.path.insert(0, SAM3D)
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body


def undistort_points(S, cam, px):
    """Source pixels -> ideal pinhole pixels with the same focal length."""
    c = np.array([S["cx"], S["cy"]])
    d = np.asarray(px, np.float64) - c
    u = d.copy()
    for _ in range(8):
        u = d / (1 + cam[1] * (u**2).sum(-1, keepdims=True) / S["fref"] ** 2)
    return u + c


def undistort_image(S, cam, img):
    v, u = np.mgrid[0 : img.shape[0], 0 : img.shape[1]].astype(np.float32)
    du, dv = u - S["cx"], v - S["cy"]
    s = 1 + cam[1] * (du**2 + dv**2) / S["fref"] ** 2
    return cv2.remap(img, (S["cx"] + du * s).astype(np.float32), (S["cy"] + dv * s).astype(np.float32), cv2.INTER_LINEAR)


def project(S, cam, P):
    """Rink points -> source pixels (with lens distortion)."""
    R, C = cam[2:11].reshape(3, 3), cam[11:14]
    pc = (P - C) @ R.T
    uv = cam[0] * pc[:, :2] / pc[:, 2:3]
    return uv * (1 + cam[1] * (uv**2).sum(1, keepdims=True) / S["fref"] ** 2) + [S["cx"], S["cy"]]


def to_rink(cam, pts_cam):
    """Camera-frame points (x right, y down, z forward) -> rink frame."""
    return pts_cam @ cam[2:11].reshape(3, 3) + cam[11:14]


def bodies(est, B):
    S = json.load(open(f"{B}/scene.json"))
    Z = np.load(f"{B}/scene.npz")
    CAST = json.load(open(f"{B}/cast.json"))
    CX, CY = S["cx"], S["cy"]
    out = dict(
        frame=[],
        key=[],
        verts=[],
        joints=[],
        scale=[],
        global_rot=[],
        pose_raw=[],
        hand=[],
        shape=[],
        scale_params=[],
        cam_t=[],
        kp2d=[],
    )
    os.makedirs("check", exist_ok=True)
    for i in range(S["n"]):
        cam = Z["cam"][i].astype(np.float64)
        img = cv2.cvtColor(undistort_image(S, cam, cv2.imread(f"{B}/frames/{i + 1:03d}.jpg")), cv2.COLOR_BGR2RGB)
        keys, boxes, masks = [], [], []
        for t in CAST["tracks"]:
            m = t["mask"][i]
            if m < 0:
                continue
            poly = undistort_points(S, cam, S["outlines"][i][m])
            (x0, y0), (x1, y1) = poly.min(0), poly.max(0)
            pad = 0.08 * max(x1 - x0, y1 - y0)
            mk = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(mk, [np.round(poly).astype(np.int32)], 1)
            keys.append(t["key"])
            boxes.append([x0 - pad, y0 - pad, x1 + pad, y1 + pad])
            masks.append(mk)
        if not keys:
            continue
        K = torch.tensor([[[cam[0], 0, CX], [0, cam[0], CY], [0, 0, 1]]], dtype=torch.float32)
        with torch.no_grad():
            res = est.process_one_image(
                img, bboxes=np.array(boxes, np.float32), masks=np.array(masks), cam_int=K, inference_type="body"
            )
        for key, r in zip(keys, res):
            v = to_rink(cam, r["pred_vertices"] + r["pred_cam_t"])
            j = to_rink(cam, r["pred_keypoints_3d"] + r["pred_cam_t"])
            C = cam[11:14]
            zmin = np.percentile(v[:, 2], 0.2)
            s = C[2] / (C[2] - zmin)
            out["frame"].append(i)
            out["key"].append(key)
            out["scale"].append(s)
            out["verts"].append((C + s * (v - C)).astype(np.float16))
            out["joints"].append((C + s * (j - C)).astype(np.float32))
            for a, b in (
                ("global_rot", "global_rot"),
                ("pose_raw", "pred_pose_raw"),
                ("hand", "hand_pose_params"),
                ("shape", "shape_params"),
                ("scale_params", "scale_params"),
                ("cam_t", "pred_cam_t"),
                ("kp2d", "pred_keypoints_2d"),
            ):
                out[a].append(np.asarray(r[b], np.float32))
        print(i, len(keys), "scale", np.round([out["scale"][-len(keys) :]], 2).tolist(), flush=True)
        if i % 60 == 0:
            vis = cv2.imread(f"{B}/frames/{i + 1:03d}.jpg")
            for k in range(len(out["frame"]) - len(keys), len(out["frame"])):
                for p in project(S, cam, out["verts"][k][::7].astype(np.float64)).astype(int):
                    cv2.circle(vis, tuple(p), 1, (60, 255, 60), -1)
            cv2.imwrite(f"check/bodies_{i:03d}.jpg", vis)
    np.savez_compressed(f"{B}/bodies.npz", **{k: np.array(v) for k, v in out.items()}, faces=est.faces)


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
    with torch.device("cuda"):  # random init of the weights the checkpoint replaces costs ~30 s on the CPU
        model, cfg = load_sam_3d_body(f"{MODEL}/model.ckpt", device=torch.device("cuda"), mhr_path=f"{MODEL}/assets/mhr_model.pt")
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    failed = 0
    for B in filter(mine, bundles):
        try:
            bodies(est, B)
            print(B, "bodies ok", flush=True)
        except Exception as e:
            failed += 1
            print(B, "bodies FAILED:", repr(e)[:300], flush=True)
    sys.exit(failed > 0)


if __name__ == "__main__":
    main(sys.argv[1:] or [os.environ.get("BUNDLE", "bundle")])
