"""Consistent bodies over the clip: every person refitted with one shape and size for the whole clip and a smooth pose, against all
their frames at once (queue 16 D4). SAM 3D Body fits each frame alone, so shape, size and pose jump from frame to frame and textures
slide over the real players; this keeps its per-frame fits as the starting point and as a weak prior.

Per person: shared MHR shape, scale components and overall size; per frame: global rotation, body pose (SAM 3D Body's continuous
parameters) and position in the rink. Hands keep the per-frame estimate. Losses, over every frame the person was fitted in:
  keypoints   the model's 2D keypoints (body, wrists, arms) and the independent YOLO11x-pose ones (pose2d.py, weighted by confidence),
              robust (Geman-McClure), in source pixels with the lens distortion
  outline     body points projected outside every person's outline (theirs or an occluder's), by distance
  coverage    optional (FIT_COVER=0.3): points inside the person's own outline (stick outlines left out) farther than the equipment from the projected body
  ice         the lowest point of the body on the ice
  smooth      acceleration of the 3D joints between consecutive frames
  prior       distance from the per-frame pose

  bundle/bodies_fit.npz   as bodies.npz (frame, key, verts float16, joints, scale = size, faces) plus the fitted parameters
  check/fit_<frame>.jpg   per-frame fits (red) and refitted bodies (green) over the source frame

usage (from pipeline/, SAM venv):  fit.py [bundle]   (render.py and director.py read it with BODIES=bodies_fit)
  FIT_ITERS (400; 0 keeps the per-frame poses), FIT_ACC (the smooth weight, 400), FIT_OUT (bodies_fit): the variant and where it goes"""

import json
import os
import sys

import cv2
import numpy as np
import roma
import torch
import torch.nn.functional as F
from paths import MODELS, SAM3D

MODEL = f"{MODELS}/sam-3d-body-dinov3"
sys.path.insert(0, SAM3D)
from sam_3d_body import load_sam_3d_body
from sam_3d_body.models.modules.geometry_utils import rot6d_to_rotmat
from sam_3d_body.models.modules.mhr_utils import compact_cont_to_model_params_body, mhr_param_hand_mask

B = sys.argv[1] if len(sys.argv) > 1 else "bundle"
DEV = "cuda"
ITERS = int(os.environ.get("FIT_ITERS", 400))  # 0: the shared shape and size on the per-frame poses as they are
OUT = os.environ.get("FIT_OUT", "bodies_fit")
KP = [*range(21), 41, 62, *range(63, 70)]  # MHR70 body, wrists, elbows/arms (no fingers)
COCO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]  # YOLO's COCO-17 order -> MHR70
W_SAM, W_YOLO, W_OUT, W_ICE, W_PRIOR = 1.0, 1.0, 0.3, 30.0, 0.02
W_ACC = float(os.environ.get("FIT_ACC", 400.0))
W_COVER = float(os.environ.get("FIT_COVER", "0"))  # opt-in until the corrected-frame ablation wins
NC, GEAR_TOL = 64, 0.08  # outline points per frame; m a bare-body point may lie inside them
FLIP = torch.tensor([1.0, -1.0, -1.0], device=DEV)  # SAM 3D Body's model frame -> camera frame


def gm(r2, s):
    """Geman-McClure on squared residuals, scale s (px)."""
    return r2 / (r2 + s * s)


class Body:
    def __init__(self):
        with torch.device(DEV):
            model, _ = load_sam_3d_body(f"{MODEL}/model.ckpt", device=torch.device(DEV), mhr_path=f"{MODEL}/assets/mhr_model.pt")
        self.head = model.head_pose.eval()
        for p in self.head.parameters():
            p.requires_grad_(False)

    def __call__(self, rot6, cont, hand, shape, scale):
        """Vertices [b, V, 3] and keypoints [b, 70, 3] in the camera-oriented model frame (metres), from SAM 3D Body's parameters."""
        euler = roma.rotmat_to_euler("ZYX", rot6d_to_rotmat(rot6))
        pose = compact_cont_to_model_params_body(cont)
        pose = pose.masked_fill(mhr_param_hand_mask.to(DEV)[None], 0)
        pose = torch.cat([pose[:, :-3], torch.zeros_like(pose[:, -3:])], 1)
        n = len(rot6)
        v, kp = self.head.mhr_forward(
            global_trans=torch.zeros(n, 3, device=DEV),
            global_rot=euler,
            body_pose_params=pose,
            hand_pose_params=hand,
            scale_params=scale.expand(n, -1),
            shape_params=shape.expand(n, -1),
            expr_params=torch.zeros(n, self.head.num_face_comps, device=DEV),
            return_keypoints=True,
        )
        return v * FLIP, kp[:, :70] * FLIP


def project(cam, S, X):
    """Rink points [b, P, 3] with per-frame cameras [b, 14] -> distorted source pixels [b, P, 2] and depth."""
    R = cam[:, 2:11].reshape(-1, 3, 3)
    pc = torch.einsum("bpi,bji->bpj", X - cam[:, None, 11:14], R)
    uv = cam[:, None, :1] * pc[..., :2] / pc[..., 2:3].clamp(min=0.1)
    uv = uv * (1 + cam[:, None, 1:2] * (uv**2).sum(-1, keepdim=True) / S["fref"] ** 2)
    return uv + torch.tensor([S["cx"], S["cy"]], device=DEV), pc[..., 2]


def distort(cam, S, px):
    """Undistorted source pixels (SAM 3D Body's input) -> distorted source pixels."""
    c = torch.tensor([S["cx"], S["cy"]], device=DEV)
    u = px - c
    return c + u * (1 + cam[:, None, 1:2] * (u**2).sum(-1, keepdim=True) / S["fref"] ** 2)


def outside_fields(S, tracks, frames):
    """Per frame: distance (px, at quarter resolution) from each pixel to the nearest person outline (0 inside any outline)."""
    out = []
    for i in frames:
        m = np.zeros((270, 480), np.uint8)
        for t in tracks:
            if t["mask"][i] >= 0:
                cv2.fillPoly(m, [np.round(np.array(S["outlines"][i][t["mask"][i]]) / 4).astype(np.int32)], 1)
        out.append(cv2.distanceTransform((1 - cv2.dilate(m, np.ones((3, 3), np.uint8))).astype(np.uint8), cv2.DIST_L2, 3) * 4)
    return torch.from_numpy(np.stack(out)).to(DEV)


def sample_field(field, px):
    g = torch.stack([px[..., 0] / 1920 * 2 - 1, px[..., 1] / 1080 * 2 - 1], -1)[:, :, None]
    return F.grid_sample(field[:, None], g, align_corners=False, padding_mode="border")[:, 0, :, 0]


def cover_points(S, sticks, t, frames, rng):
    """Per frame: NC pixels inside the person's own outline (visible there, so the body must reach them), stick outlines left out;
    [n, NC, 2] and which frames have them."""
    pts, ok = np.zeros((len(frames), NC, 2), np.float32), np.zeros(len(frames), bool)
    for j, i in enumerate(frames):
        if t["mask"][i] < 0:
            continue
        poly = np.round(np.array(S["outlines"][i][t["mask"][i]])).astype(np.int32)
        x0, y0 = poly.min(0)
        x1, y1 = poly.max(0)
        m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.fillPoly(m, [poly - [x0, y0]], 1)
        for k in sticks[i]:
            cv2.fillPoly(m, [np.round(np.array(k["poly"])).astype(np.int32) - [x0, y0]], 0)
        ys, xs = np.nonzero(m)
        if not len(ys):
            continue
        sel = rng.choice(len(ys), NC, replace=len(ys) < NC)
        pts[j] = np.stack([xs[sel] + x0, ys[sel] + y0], 1)
        ok[j] = True
    return torch.from_numpy(pts).to(DEV), torch.from_numpy(ok).to(DEV)


def fit_person(body, S, cams, b, rows, yolo, fields, cover, cover_ok):
    fr = b["frame"][rows]
    cam = torch.from_numpy(cams[fr]).float().to(DEV)
    raw = torch.from_numpy(b["pose_raw"][rows]).to(DEV)
    hand = torch.from_numpy(b["hand"][rows]).to(DEV)
    shape0 = torch.from_numpy(np.median(b["shape"][rows], 0)).to(DEV)[None]
    scale0 = torch.from_numpy(np.median(b["scale_params"][rows], 0)).to(DEV)[None]
    R = cam[:, 2:11].reshape(-1, 3, 3)
    with torch.no_grad():
        _, kp0 = body(raw[:, :6], raw[:, 6:], hand, shape0, scale0)
    root0 = torch.from_numpy(b["joints"][rows][:, [9, 10]].mean(1)).to(DEV)  # where the per-frame fits put the pelvis
    size = torch.tensor(float(np.median(b["scale"][rows])), device=DEV)
    trans = (root0 - size * torch.einsum("bi,bij->bj", kp0[:, [9, 10]].mean(1), R)).clone()
    p = dict(
        rot6=raw[:, :6].clone(),
        cont=raw[:, 6:].clone(),
        shape=shape0.clone(),
        scale=scale0.clone(),
        log_size=size.log().clone(),
        trans=trans,
    )
    for v in p.values():
        v.requires_grad_(True)
    opt = torch.optim.Adam(
        [
            {"params": [p["rot6"], p["cont"]], "lr": 0.01},
            {"params": [p["shape"], p["scale"]], "lr": 0.005},
            {"params": [p["log_size"]], "lr": 0.002},
            {"params": [p["trans"]], "lr": 0.01},
        ]
    )
    sam_kp = distort(cam, S, torch.from_numpy(b["kp2d"][rows][:, KP]).to(DEV))
    yk = torch.zeros(len(rows), 17, 3, device=DEV)
    for j, f in enumerate(fr):
        y = yolo[f].get(str(int(b["key"][rows[0]])))
        if y:
            yk[j] = torch.tensor(y, device=DEV)
    consec = torch.from_numpy((np.diff(fr) == 1)[1:] & (np.diff(fr) == 1)[:-1]).to(DEV)
    sub = torch.arange(0, 18439, 25, device=DEV)
    for _it in range(ITERS):
        v, kp = body(p["rot6"], p["cont"], hand, p["shape"], p["scale"])
        s = p["log_size"].exp()
        X = s * torch.einsum("bpi,bij->bpj", v[:, sub], R) + p["trans"][:, None]
        J = s * torch.einsum("bpi,bij->bpj", kp, R) + p["trans"][:, None]
        pj, _ = project(cam, S, J)
        l_sam = gm(((pj[:, KP] - sam_kp) ** 2).sum(-1), 8.0).mean()
        l_yolo = (gm(((pj[:, COCO] - yk[..., :2]) ** 2).sum(-1), 8.0) * yk[..., 2]).sum() / yk[..., 2].sum().clamp(min=1)
        px, depth = project(cam, S, X)
        l_out = (sample_field(fields[fr], px) / 10).pow(2).mean()
        l_cover = torch.zeros((), device=DEV)
        if W_COVER and cover_ok.any():
            gap = torch.cdist(cover, px).min(-1).values - GEAR_TOL * cam[:, :1] / depth.mean(1, keepdim=True)
            l_cover = (gap.clamp(min=0) / 10).pow(2)[cover_ok].mean()
        zmin = (s * torch.einsum("bpi,bi->bp", v, R[:, :, 2]) + p["trans"][:, 2:3]).min(1).values
        l_ice = (zmin**2).mean()
        acc = J[2:] - 2 * J[1:-1] + J[:-2]
        l_acc = (
            (acc.pow(2).sum(-1).mean(-1) * consec).sum() / consec.sum().clamp(min=1)
            if len(J) > 2
            else torch.zeros((), device=DEV)
        )
        l_prior = (p["cont"] - raw[:, 6:]).pow(2).mean() + (p["rot6"] - raw[:, :6]).pow(2).mean()
        loss = (
            W_SAM * l_sam
            + W_YOLO * l_yolo
            + W_OUT * l_out
            + W_COVER * l_cover
            + W_ICE * l_ice
            + W_ACC * l_acc
            + W_PRIOR * l_prior
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        v, kp = body(p["rot6"], p["cont"], hand, p["shape"], p["scale"])
        s = p["log_size"].exp()
        V = s * torch.einsum("bpi,bij->bpj", v, R) + p["trans"][:, None]
        J = s * torch.einsum("bpi,bij->bpj", kp, R) + p["trans"][:, None]
    if not ITERS:
        return V.cpu().numpy(), J.cpu().numpy(), s.item(), {k: x.detach().cpu().numpy() for k, x in p.items()}, {}
    terms = dict(
        sam=l_sam.item(),
        yolo=l_yolo.item(),
        out=l_out.item(),
        cover=l_cover.item(),
        ice=l_ice.item(),
        acc=l_acc.item(),
        size=s.item(),
    )
    return V.cpu().numpy(), J.cpu().numpy(), s.item(), {k: x.detach().cpu().numpy() for k, x in p.items()}, terms


def main():
    S = json.load(open(f"{B}/scene.json"))
    cams = np.load(f"{B}/scene.npz")["cam"].astype(np.float64)
    tracks = json.load(open(f"{B}/cast.json"))["tracks"]
    yolo = json.load(open(f"{B}/pose2d.json"))
    b = {k: v for k, v in np.load(f"{B}/bodies.npz").items()}
    body = Body()
    fields = outside_fields(S, tracks, range(S["n"]))
    sticks = json.load(open(f"{B}/sticks.json"))
    rng = np.random.default_rng(0)
    out = {k: np.zeros_like(b[k]) for k in ("verts", "joints")} | dict(scale=np.zeros(len(b["frame"])))
    params = {}
    for key in np.unique(b["key"]):
        rows = np.flatnonzero(b["key"] == key)
        rows = rows[np.argsort(b["frame"][rows])]
        t = next(t for t in tracks if t["key"] == key)
        V, J, s, p, terms = fit_person(body, S, cams, b, rows, yolo, fields, *cover_points(S, sticks, t, b["frame"][rows], rng))
        out["verts"][rows] = V.astype(np.float16)
        out["joints"][rows] = J
        out["scale"][rows] = s
        for k, x in p.items():
            params.setdefault(k, {})[int(key)] = x
        print(key, len(rows), "frames", {k: round(x, 4) for k, x in terms.items()}, flush=True)
    np.savez_compressed(
        f"{B}/{OUT}.npz",
        frame=b["frame"],
        key=b["key"],
        faces=b["faces"],
        **out,
        **{f"fit_{k}_{key}": x for k, d in params.items() for key, x in d.items()},
    )
    os.makedirs("check", exist_ok=True)
    for i in (60, 110, 160):
        img = cv2.imread(f"{B}/frames/{i + 1:03d}.jpg")
        rows = np.flatnonzero(b["frame"] == i)
        cam = torch.from_numpy(cams[[i]]).float().to(DEV)
        for src, col in ((b["verts"], (60, 60, 255)), (out["verts"], (60, 255, 60))):
            for r in rows:
                px, _ = project(cam, S, torch.from_numpy(src[r][::9].astype(np.float32)).to(DEV)[None])
                for q in px[0].cpu().numpy().astype(int):
                    cv2.circle(img, tuple(q), 1, col, -1)
        cv2.imwrite(f"check/fit_{i:03d}.jpg", img)


if __name__ == "__main__":
    main()
