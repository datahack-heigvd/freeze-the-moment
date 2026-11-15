"""Runs on the GPU machine (hockey/id). Appearance models for the gap benchmark, trained on metadata labels only:
the other window (full metadata) + the visible blocks of the test window; applied to every detection of the test window.
  D  player classifier: crop -> roster player (closed set)
  B  jersey number reader: torso crop -> tens digit (or none) and units digit (any number, not tied to the roster)
  A  re-identification embedding: OSNet (MSMT17) as is (A0) and fine-tuned with softmax + triplet on the same labels (A)
  K  puck carrier: wide crop around the feet and stick -> 'has the puck' log-odds (labels from carrier.py label)
Whole game ('game'): trained on every labelled crop of every chunk (game/dets_<N0>.npz + crops_<N0>.pkl from gapid.py), applied to all of them.
usage: id_gpu.py <D|B|A|K> <test window: 300|2160 | game>   -> out_<task>_<win>.npz, or game/out_<task>_<N0>.npz"""

import glob
import json
import pickle
import sys
import time

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

TASK, WIN = sys.argv[1], sys.argv[2]
OTHER = {"300": "2160", "2160": "300"}.get(WIN)
DEV = "cuda:0"
EPOCHS, BS = (3 if WIN == "game" else 6), 128
T0 = {"300": 920.5, "2160": 2160.5}  # window start (s): the benchmark hides every second 30 s block from here
JERSEY = [r["jersey"] for r in json.load(open("roster.json"))]
TENS = np.array([int(j) // 10 if len(j) == 2 else 10 for j in JERSEY])
UNITS = np.array([int(j) % 10 for j in JERSEY])
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def load(win):
    Z = dict(np.load(f"dets{win}.npz"))
    if TASK == "K":
        Z.update(np.load(f"carrier_lab{win}.npz"))
        Z["lab_full"], Z["lab_vis"] = Z["y_full"], Z["y_vis"]
    return np.load(f"c{win}_{'wide' if TASK == 'K' else 'person'}.npy", mmap_mode="r"), Z


class Jpegs:
    """JPEG-encoded crops, decoded on access like the memmapped arrays of the window mode."""

    def __init__(self, blobs):
        self.b = blobs

    def __len__(self):
        return len(self.b)

    def __getitem__(self, i):
        return cv2.imdecode(np.frombuffer(self.b[i], np.uint8), cv2.IMREAD_COLOR)


def game_sources():
    """[(N0, crops, labels)] per chunk: identity labels for D/B/A, carrier labels for K."""
    out = []
    for f in sorted(glob.glob("game/dets_*.npz")):
        N0 = int(f.split("_")[-1][:-4])
        Z = np.load(f)
        C = pickle.load(open(f"game/crops_{N0}.pkl", "rb"))
        out.append((N0, Jpegs(C["wide" if TASK == "K" else "person"]), Z["car" if TASK == "K" else "lab"]))
    return out


def batch(X, idx, train):
    """uint8 BGR crops -> normalised RGB tensor at the model's input size, with geometric and photometric jitter when training."""
    x = torch.from_numpy(np.stack([X[i] for i in np.sort(idx)])).to(DEV).permute(0, 3, 1, 2).flip(1).float() / 255
    if TASK == "B":
        x = x[:, :, 8:96, :]  # torso: where the back number sits
    n = len(x)
    if train:
        s = 1 + (torch.rand(n, device=DEV) - 0.5) * 0.16
        t = (torch.rand(n, 2, device=DEV) - 0.5) * 0.12
        th = torch.zeros(n, 2, 3, device=DEV)
        th[:, 0, 0] = s
        th[:, 1, 1] = s
        th[:, :, 2] = t
        x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode="border", align_corners=False)
        x = (
            (x - 0.5) * (1 + (torch.rand(n, 1, 1, 1, device=DEV) - 0.5) * 0.4)
            + 0.5
            + (torch.rand(n, 1, 1, 1, device=DEV) - 0.5) * 0.2
        )
    x = F.interpolate(x, size=(192, 192) if TASK in "BK" else (256, 128), mode="bilinear", align_corners=False)
    return ((x.clamp(0, 1) - MEAN) / STD), np.sort(idx)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        if TASK == "A":
            from boxmot.reid.core.runtime import ReID

            self.body = ReID(weights="osnet_x1_0_msmt17.pt", device=DEV).model.model.float()
            dim = 512
            self.body.classifier = nn.Identity()
        else:
            self.body = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=0)
            dim = self.body.num_features
        self.bn = nn.BatchNorm1d(dim)
        self.head = nn.Linear(dim, {"B": 21, "K": 1}.get(TASK, len(JERSEY)))

    def embed(self, x):
        f = self.body(x)
        return f[0] if isinstance(f, (tuple, list)) else f

    def forward(self, x):
        f = self.embed(x)
        return f, self.head(self.bn(f))


def loss_of(f, z, y):
    if TASK == "K":
        return F.binary_cross_entropy_with_logits(
            z[:, 0], torch.as_tensor(y, device=DEV).float(), pos_weight=torch.tensor(4.0, device=DEV)
        )
    if TASK == "B":
        return F.cross_entropy(z[:, :11], torch.as_tensor(TENS[y], device=DEV), label_smoothing=0.1) + F.cross_entropy(
            z[:, 11:], torch.as_tensor(UNITS[y], device=DEV), label_smoothing=0.1
        )
    l = F.cross_entropy(z, torch.as_tensor(y, device=DEV), label_smoothing=0.1)
    if TASK == "A":  # batch-hard triplet on normalised embeddings
        e = F.normalize(f, dim=1)
        d = torch.cdist(e, e)
        yy = torch.as_tensor(y, device=DEV)
        same = yy[:, None] == yy[None]
        l = l + F.relu((d * same).max(1).values - (d + 4 * same).min(1).values + 0.3).mean()
    return l


def pk_batches(y, rng, P=16, K=8):
    """P players x K crops per batch (triplets need positives); falls back to plain shuffling for D and B."""
    if TASK != "A":
        idx = rng.permutation(len(y))
        return [idx[i : i + BS] for i in range(0, len(idx), BS)]
    by = {p: rng.permutation(np.flatnonzero(y == p)) for p in set(y.tolist())}
    out = []
    for _ in range(len(y) // (P * K)):
        ps = rng.choice(list(by), min(P, len(by)), replace=False)
        out.append(np.concatenate([rng.choice(by[p], K) for p in ps]))
    return out


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    if WIN == "game":
        chunks = game_sources()
        src = [(X, np.flatnonzero(y >= 0), y) for _, X, y in chunks]
        print(f"{TASK} whole game: train crops {sum(len(s[1]) for s in src)} from {len(src)} chunks", flush=True)
    else:
        Xo, Zo = load(OTHER)
        Xt, Zt = load(WIN)
        src = [(Xo, np.flatnonzero(Zo["lab_full"] >= 0), Zo["lab_full"]), (Xt, np.flatnonzero(Zt["lab_vis"] >= 0), Zt["lab_vis"])]
        print(
            f"{TASK} test {WIN}: train crops {sum(len(s[1]) for s in src)} (other window {len(src[0][1])}, visible blocks {len(src[1][1])})",
            flush=True,
        )
    net = Net().to(DEV)
    out = {}
    if TASK == "A" and WIN != "game":
        out["A0"] = infer(net, Xt)
    opt = torch.optim.AdamW(net.parameters(), lr=3.5e-4 if TASK == "A" else 2e-4, weight_decay=0.05)
    pairs = [(k, i) for k, s in enumerate(src) for i in s[1]]
    ys = np.array([src[k][2][i] for k, i in pairs])
    steps = EPOCHS * len(pk_batches(ys, rng))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, opt.param_groups[0]["lr"], total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for ep in range(EPOCHS):
        net.train()
        tot = 0.0
        bs = pk_batches(ys, rng)
        for b in bs:
            xs, ys_ = [], []
            for k in range(len(src)):
                sel = [pairs[j][1] for j in b if pairs[j][0] == k]
                if sel:
                    x, order = batch(src[k][0], np.array(sel), True)
                    xs.append(x)
                    ys_.append(src[k][2][order])
            x = torch.cat(xs)
            y = np.concatenate(ys_)
            with torch.autocast("cuda", torch.bfloat16):
                f, z = net(x)
            l = loss_of(f.float(), z.float(), y)
            opt.zero_grad()
            l.backward()
            opt.step()
            sched.step()
            tot += l.item()
        print(f"  epoch {ep}: loss {tot / len(bs):.3f}  {time.time() - t0:.0f}s", flush=True)
    if WIN == "game":
        for N0, X, _ in chunks:
            np.savez(f"game/out_{TASK}_{N0}.npz", **{TASK: infer(net, X, head=TASK != "A")})
        return
    res = infer(net, Xt, head=TASK != "A")
    if TASK == "A":
        out["A"] = res
    else:
        out[TASK] = res
    hid = (Zt["lab_vis"] < 0) & (Zt["lab_full"] >= 0) & np.isin(Zt["N"], hidden_frames(Zt))
    y = Zt["lab_full"][hid]
    if TASK == "D":
        print(
            f"  per-crop accuracy on hidden-block crops (full-metadata labels): {100 * (res[hid].argmax(1) == y).mean():.1f}% of {hid.sum()}"
        )
    if TASK == "K":
        hk = np.isin(Zt["N"], hidden_frames(Zt)) & (Zt["lab_full"] >= 0)
        Ns = Zt["N"][hk]
        pos = Zt["lab_full"][hk] == 1
        sc = res[hk].astype(float)
        top = []
        for N in np.unique(Ns[pos]):
            m = Ns == N
            top.append(pos[m][np.argmax(sc[m])])
        print(
            f"  hidden-block frames with a carrier: top-scored detection is the carrier in {100 * np.mean(top):.1f}% of {len(top)}"
        )
    if TASK == "B":
        lp = res[hid]
        t_ = lp[:, :11].argmax(1)
        u_ = lp[:, 11:].argmax(1)
        print(
            f"  per-crop number accuracy on hidden-block crops: {100 * ((t_ == TENS[y]) & (u_ == UNITS[y])).mean():.1f}% of {hid.sum()}"
        )
    np.savez(f"out_{TASK}_{WIN}.npz", **out)


def hidden_frames(Z):
    t = Z["N"] / 25.0
    return Z["N"][((t - T0[WIN]) // 30).astype(int) % 2 == 1]


@torch.no_grad()
def infer(net, X, head=False):
    net.eval()
    out = []
    for i in range(0, len(X), 512):
        x, _ = batch(X, np.arange(i, min(i + 512, len(X))), False)
        with torch.autocast("cuda", torch.bfloat16):
            f, z = net(x)
        if not head:
            out.append(F.normalize(f.float(), dim=1).cpu().numpy().astype(np.float16))
        elif TASK == "K":
            out.append(z[:, 0].float().cpu().numpy().astype(np.float16))
        elif TASK == "B":
            out.append(
                torch.cat([F.log_softmax(z[:, :11].float(), 1), F.log_softmax(z[:, 11:].float(), 1)], 1)
                .cpu()
                .numpy()
                .astype(np.float16)
            )
        else:
            out.append(F.log_softmax(z.float(), 1).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


if __name__ == "__main__":
    main()
