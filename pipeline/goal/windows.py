"""The match's goal windows from its feed: every goal event of game.json, LEAD s before it, DUR s long. A goal is named after its
scorer; a scorer's earlier goals get the period too (tambellini_p1, then tambellini). dvc.yaml builds one scene per window.

usage:  windows.py   -> pipeline/goals.json {"goals": {name: {"start", "dur"}}}"""

import json
from collections import Counter

from paths import GAME, PIPELINE

LEAD, DUR = 5.96, 10.0  # s: the shot and the build-up before the goal event, then the celebration

goals = [e for e in json.load(open(GAME))["events"] if e["kind"] == "goal"]
surname = [e["player"].split()[-1].lower() for e in goals]
left = Counter(surname)
windows = {}
for e, s in zip(goals, surname):
    left[s] -= 1
    windows[f"{s}_p{e['period']}" if left[s] else s] = dict(start=round(e["t"] - LEAD, 2), dur=DUR)
json.dump(dict(goals=windows), open(f"{PIPELINE}/goals.json", "w"), indent=1)
print(windows)
