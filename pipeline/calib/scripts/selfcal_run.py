import sys

sys.path.insert(0, "scripts")
import time

import numpy as np
import selfcal as S

E0 = np.load("cache/evidence_loc.npz")
E = {k: E0[k] for k in E0.files}
Z = np.load("cache/cam_loc300.npz")
idx = E["idx"]
F = Z["f"][idx]
RV = Z["rv"][idx]
n = len(idx)
tr = np.arange(0, n, 2)
te = np.arange(1, n, 2)
g0 = S.g_init()


def report(name, g):
    print(
        f"{name:28s} train: {S.fmt(S.metrics(g, F, RV, E, tr))}\n{'':28s} test : {S.fmt(S.metrics(g, F, RV, E, te))}", flush=True
    )


report("start", g0)
base = ["s", "rg0", "rg1", "rg2", "dC0", "dC1", "dC2", "kb", "pw", "cxo", "cyo"]
variants = {
    "V1 lens+C": base,
    "V2 +C(f)": base + ["dCf0", "dCf1", "dCf2"],
    "V3 +rink dims": base + ["dCf0", "dCf1", "dCf2", "hx", "hy", "corner", "bl", "fox", "foy", "rcirc"],
}
res = {}
for k, free in variants.items():
    t0 = time.time()
    g = S.fit(g0, F, RV, E, tr, free)
    res[k] = g
    report(k + f" ({time.time() - t0:.0f}s)", g)
    print({n_: round(float(v), 3) for n_, v in zip(S.GNAMES, g) if n_ in free}, flush=True)
np.savez("cache/selfcal_v1.npz", g0=g0, **{k.split()[0]: v for k, v in res.items()})
