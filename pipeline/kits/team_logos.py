"""Each team of a match: its name and, where a public machine-readable source has it, its logo, for player_textures.py's crest.

Names: a match with a feed takes them from live/game.json (`teams`, the feed's Home and Away: team 0 and 1, as prep.py numbers
them); a broadcast without one from its video's title ("A vs. B", YouTube's oEmbed), whose order says nothing of the kits, so
each name goes to the kit whose colours its logo's are nearest (both logos needed and the pairing clear by MARGIN, else no
names). Logos: Wikidata's logo image
(P154) of the ice hockey team the name finds (its label or an alias the name, folded), else TheSportsDB's badge of the ice hockey
team that name names (one only); a name neither has gets no logo, and nothing is put in its place.

  <KITS>/teams.json        {"0": {name, logo (file in KITS, or null), source}, "1": ...}
  <KITS>/logo_<team>.png   RGBA, about LOGO px wide

A source that cannot be asked (offline, refused) fails the run and nothing is written, so the next build asks again.

usage (LTX venv):  team_logos.py <KITS> game <game.json>
                       team_logos.py <KITS> youtube <video id>"""

import io
import json
import re
import sys
import unicodedata
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

LOGO = 1024
MARGIN = 0.3  # the kit-to-logo pairing taken only when it is this much better than the other
AGENT = {"User-Agent": "hockey-replays-team-logos/1.0 (research prototype)"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=30) as r:
        return r.read()


def fold(s):
    return re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()).strip()


def wikidata(name):
    """The logo of the ice hockey team whose label or alias is `name`, from Wikidata (P154 on Commons), or None."""
    q = urllib.parse.urlencode(dict(action="wbsearchentities", search=name, language="en", type="item", format="json", limit=10))
    for hit in json.loads(get(f"https://www.wikidata.org/w/api.php?{q}"))["search"]:
        names = [hit.get("label", "")] + hit.get("aliases", []) + [hit.get("match", {}).get("text", "")]
        if "hockey" not in hit.get("description", "").lower() or fold(name) not in {fold(n) for n in names}:
            continue
        claims = json.loads(get(f"https://www.wikidata.org/wiki/Special:EntityData/{hit['id']}.json"))["entities"][hit["id"]][
            "claims"
        ]
        for c in claims.get("P154", []):
            f = c["mainsnak"].get("datavalue", {}).get("value")
            if f:
                return get(
                    f"https://commons.wikimedia.org/w/index.php?title=Special:FilePath/{urllib.parse.quote(f)}&width={LOGO}"
                ), f"wikidata {hit['id']}"
    return None


def sportsdb(name):
    """The badge of the one ice hockey team on TheSportsDB (its free key) whose name, or an alternate name, is made of words of
    `name` (it names "HC Davos" "Davos"), searched by the name and by each of its longer words; None unless exactly one is."""
    words = set(fold(name).split())
    found = {}
    for q in [name] + [w for w in fold(name).split() if len(w) >= 4]:
        for t in (
            json.loads(
                get(f"https://www.thesportsdb.com/api/v1/json/123/searchteams.php?{urllib.parse.urlencode(dict(t=q))}")
            ).get("teams")
            or []
        ):
            names = [t.get("strTeam") or ""] + (t.get("strTeamAlternate") or "").split(",")
            if t.get("strSport") == "Ice Hockey" and any(n.strip() and set(fold(n).split()) <= words for n in names):
                found[t["idTeam"]] = t
    badge = [t.get("strBadge") or t.get("strTeamBadge") for t in found.values()]
    return (get(badge[0]), f"thesportsdb {next(iter(found))}") if len(found) == 1 and badge[0] else None


def logo(name):
    """Its logo (RGBA) and where from, or (None, None); raises when a source could not be asked, so no answer is kept for it."""
    for source in (wikidata, sportsdb):
        got = source(name)
        if got:
            return Image.open(io.BytesIO(got[0])).convert("RGBA"), got[1]
    return None, None


def colours(img):
    """The opaque pixels' colours of an RGBA image, [n, 3] in 0..1."""
    a = np.asarray(img.resize((128, round(128 * img.height / img.width))), np.float32) / 255
    return a[a[..., 3] > 0.5][:, :3]


def by_kit(kits, logos):
    """The order of `logos` (two RGBA images) that puts each on the kit whose colours its own are nearest: per pair, the mean
    distance of the kit's palette colours (KITS/kit_votes.npz, weighted by their votes) to the logo's nearest colour."""
    K = np.load(f"{kits}/kit_votes.npz")
    cost = np.zeros((2, 2))
    for t in (0, 1):
        mu = K[f"kit_{t}_0_palette"]
        w = K[f"kit_{t}_0_votes"].reshape(-1, len(mu)).astype(np.float64).sum(0)
        w = w / w.sum()
        for j, img in enumerate(logos):
            c = colours(img)
            cost[t, j] = (w * np.sqrt(((mu[:, None] - c[None]) ** 2).sum(-1)).min(1)).sum()
    print("kit to logo colour distance", cost.round(3).tolist(), flush=True)
    keep, swap = cost[0, 0] + cost[1, 1], cost[0, 1] + cost[1, 0]
    if abs(keep - swap) < MARGIN * max(keep, swap):
        return None  # too close to tell
    return [0, 1] if keep < swap else [1, 0]


def main():
    kits, mode, arg = sys.argv[1:4]
    if mode == "game":
        t = json.load(open(arg)).get("teams") or {}
        names = [t.get("Home"), t.get("Away")]
        ordered = True
    else:
        title = json.loads(get(f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={arg}&format=json"))["title"]
        m = next((m for part in title.split("|") if (m := re.match(r"\s*(.+?)\s+vs\.?\s+(.+?)\s*$", part))), None)
        names = [m.group(1), m.group(2)] if m else [None, None]
        ordered = False
        print("title", title, "->", names, flush=True)
    got = [logo(n) if n else (None, None) for n in names]
    if not ordered:
        order = by_kit(kits, [g[0] for g in got]) if all(g[0] for g in got) else None
        if order:
            names, got = [names[i] for i in order], [got[i] for i in order]
        else:
            names, got = [None, None], [(None, None), (None, None)]  # which kit is whose is unknown
    out = {}
    for team, (name, (img, source)) in enumerate(zip(names, got)):
        file = None
        if img is not None:
            file = f"logo_{team}.png"
            img.thumbnail((LOGO, LOGO))
            img.save(f"{kits}/{file}")
        out[str(team)] = dict(name=name, logo=file, source=source)
        print(team, name, source or "no logo", flush=True)
    json.dump(out, open(f"{kits}/teams.json", "w"), indent=1)


if __name__ == "__main__":
    main()
