"""Teams by jersey colour, with nothing labelled by hand: every on-ice person's upper body becomes a colour histogram, the histograms
fall into K groups (motion blur and shadow split a team, so more groups than teams), and the groups are labelled from their own
statistics: the two teams are the pair of groups, each holding at least MIN_SHARE of the people, that covers the most people among
pairs whose colours differ (Hellinger distance over APART); every other group joins the team it is nearer to when it is nearer than
JOIN of the teams' distance, else it is junk (empty ice, spectators, mixed crops). "home" is the team of the larger group.

usage:  kit_groups.py check <bundle>...   (any env with cv2: the 2025 goal bundles) the groups against the feed's teams (cast.json):
                                          share of people put in their own team (teams matched either way round), and how many junk"""

import json
import sys

import cv2
import numpy as np

K, MIN_SHARE, APART, JOIN = 6, 0.1, 0.35, 0.5


def upper_px(img, poly):
    """Lab pixels of the upper body (top half of the outline below the helmet, eroded), at most 400."""
    y0, y1 = poly[:, 1].min(), poly[:, 1].max()
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    m[int(y0 + 0.55 * (y1 - y0)) :] = 0
    m[: int(y0 + 0.12 * (y1 - y0))] = 0
    px = img[cv2.erode(m, np.ones((5, 5), np.uint8)) > 0].astype(np.float32)
    return px[np.linspace(0, len(px) - 1, min(len(px), 400)).astype(int)] if len(px) > 30 else None


def colour_hist(px):
    """12-bin jersey histogram of Lab pixels (OpenCV 8-bit): coloured pixels (chroma > 12) by hue in 8 bins, neutral ones by lightness in
    4 (black, dark, light, white)."""
    a, b = px[:, 1] - 128, px[:, 2] - 128
    col = np.hypot(a, b) > 12
    bins = np.where(
        col, (np.arctan2(b, a) % (2 * np.pi) / (np.pi / 4)).astype(int) % 8, 8 + np.digitize(px[:, 0], [60, 120, 190])
    )
    return np.bincount(bins, minlength=12) / len(px)


def team_groups(pxs, k=K):
    """Jersey groups (mean colour histogram, largest first) from every sampled outline's pixels, and each outline's group."""
    h = np.array([colour_hist(p) for p in pxs], np.float32)
    cv2.setRNGSeed(0)
    _, lab, _ = cv2.kmeans(
        np.sqrt(h), k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 8, cv2.KMEANS_PP_CENTERS
    )
    lab = lab.ravel()
    order = np.argsort(-np.bincount(lab, minlength=k))
    rank = np.argsort(order)[lab]
    groups = [h[rank == g].mean(0) for g in range(k)]
    print("group sizes", np.bincount(rank, minlength=k).tolist())
    return dict(groups=[g.round(4).tolist() for g in groups], sizes=np.bincount(rank, minlength=k).tolist()), rank


def hellinger(p, q):
    return float(np.sqrt(max(0.0, 1 - np.sqrt(np.asarray(p) * np.asarray(q)).sum())))


def labels(teams):
    """home, away or junk per group, from the groups' colours and sizes alone."""
    G, n = teams["groups"], np.array(teams["sizes"], float)
    share = n / n.sum()
    k = len(G)
    pairs = [
        (n[a] + n[b], a, b)
        for a in range(k)
        for b in range(a + 1, k)
        if share[a] >= MIN_SHARE and share[b] >= MIN_SHARE and hellinger(G[a], G[b]) > APART
    ]
    if not pairs:
        return ["junk"] * k
    _, a, b = max(pairs)
    apart = hellinger(G[a], G[b])
    out = []
    for g in range(k):
        da, db = hellinger(G[g], G[a]), hellinger(G[g], G[b])
        out.append(
            "home" if g == a else "away" if g == b else ("home" if da < db else "away") if min(da, db) < JOIN * apart else "junk"
        )
    return out


def check(bundles):
    pxs, truth = [], []
    for b in bundles:
        S = json.load(open(f"{b}/scene.json"))
        C = json.load(open(f"{b}/cast.json"))
        for f in range(0, S["n"], 8):
            img = cv2.cvtColor(cv2.imread(f"{b}/frames/{f + 1:03d}.jpg"), cv2.COLOR_BGR2LAB)
            for t in C["tracks"]:
                m = t["mask"][f]
                if m < 0 or t["team"] not in (0, 1):
                    continue
                px = upper_px(img, np.array(S["outlines"][f][m], np.float32))
                if px is not None:
                    pxs.append(px)
                    truth.append(t["team"])
    teams, rank = team_groups(pxs)
    lab = labels(teams)
    truth = np.array(truth)
    pred = np.array([{"home": 0, "away": 1, "junk": -1}[lab[g]] for g in rank])
    kept = pred >= 0
    acc = max((pred[kept] == truth[kept]).mean(), (pred[kept] == 1 - truth[kept]).mean())
    print(f"{len(pxs)} people; groups {lab}; junk {100 * (1 - kept.mean()):.1f}%; own team {100 * acc:.1f}% of the rest")


if __name__ == "__main__":
    {"check": lambda: check(sys.argv[2:])}[sys.argv[1]]()
