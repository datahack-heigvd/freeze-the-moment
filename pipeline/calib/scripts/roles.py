"""Who is on the ice? Per-tracklet classifier: home skater / away skater / not a player (referee, staff).
Labels come from the metadata, no manual work: a tracklet identified with certainty is a player of that roster team; a tracklet that spends
its whole life far from every snapshot player is not a player. Features: torso colour (yellow, blue, dark, white, saturation, value)."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from vtrack import FPS

NONE_DIST, NONE_FRAC, NONE_MIN_SNAPS, NONE_MIN_S = 2.5, 0.9, 4, 2.0
P_NON, P_TEAM = 0.5, 0.7


def feature_matrix(f):
    """f: (n, 6) torso colour [yellow, blue, dark, white, saturation, value] of tracklets or single detections."""
    f = np.nan_to_num(np.asarray(f, float), nan=0.0)
    return np.c_[f, f[:, 2] + f[:, 3], np.abs(f[:, 0] - f[:, 1])]


def features(tracks):
    return feature_matrix([t["feat"] for t in tracks])


def labels(tracks, ident, sl, corr, team_p):
    """0 home, 1 away, 2 not a player, -1 unlabeled."""
    y = np.full(len(tracks), -1)
    for ti, tr in enumerate(tracks):
        if ident[ti] >= 0:
            y[ti] = team_p[ident[ti]]
            continue
        if (tr["b"] - tr["a"] + 1) / FPS < NONE_MIN_S:
            continue
        far = []
        for N, d in sl:
            if tr["a"] <= N <= tr["b"] and len(d) >= 11:
                xy = tr["pos"][N - tr["a"]]
                far.append(min(np.linalg.norm(xy - corr(m, p)) for p, m in d.items()) > NONE_DIST)
        if len(far) >= NONE_MIN_SNAPS and np.mean(far) >= NONE_FRAC:
            y[ti] = 2
    return y


def fit_predict(tracks, y, verbose=True):
    """Returns class probabilities (n, 3) for every tracklet (cross-validated for labeled ones) and the final classifier."""
    X = features(tracks)
    w = np.sqrt([t["b"] - t["a"] + 1 for t in tracks])
    lab = y >= 0
    groups = np.array([int(t["a"] / FPS / 60) for t in tracks])
    mk = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=6)
    P = np.zeros((len(tracks), 3))
    idx = np.flatnonzero(lab)
    for tr_i, te_i in GroupKFold(n_splits=min(5, len(set(groups[idx])))).split(idx, groups=groups[idx]):
        m = mk().fit(X[idx[tr_i]], y[idx[tr_i]], sample_weight=w[idx[tr_i]])
        P[idx[te_i]] = m.predict_proba(X[idx[te_i]])
    final = mk().fit(X[idx], y[idx], sample_weight=w[idx])
    P[~lab] = final.predict_proba(X[~lab])
    if verbose:
        pred = P[idx].argmax(1)
        print(
            f"  labels: home {int((y == 0).sum())}, away {int((y == 1).sum())}, not-a-player {int((y == 2).sum())}, unlabeled {int((~lab).sum())}"
        )
        for c, name in enumerate(("home", "away", "not a player")):
            m = y[idx] == c
            print(
                f"  cross-validated by 60 s block: {name:13s} recall {100 * np.mean(pred[m] == c):.1f}% (n={m.sum()}), precision {100 * np.mean(y[idx][pred == c] == c):.1f}%"
            )
        ln = np.array([t["b"] - t["a"] + 1 for t in tracks])
        fp = (y[idx] < 2) & (pred == 2)
        print(
            f"  players wrongly called not-a-player: {fp.sum()} tracklets ({ln[idx][fp].sum() / FPS:.0f} s of {ln[idx][y[idx] < 2].sum() / FPS:.0f} s)"
        )
    return P, final


def assign_roles(tracks, P):
    """-2 not a player, 0 home, 1 away, -1 a player whose team the classifier cannot tell."""
    for t, p in zip(tracks, P):
        if p[2] >= P_NON:
            t["team"] = -2
        elif max(p[0], p[1]) / (p[0] + p[1]) >= P_TEAM:
            t["team"] = int(p[1] > p[0])
        else:
            t["team"] = -1
