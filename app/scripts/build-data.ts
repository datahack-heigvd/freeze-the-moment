import { join } from "node:path";
import { csvParse } from "d3";

const SRC = join(import.meta.dir, "../../store/data/U.S. Collegiate Selects_HC Davos_Wisehockey_Match_Events_31_12_2025.csv");
const OUT = join(import.meta.dir, "../../store/site/game.json");
// The export is named <home>_<away>_Wisehockey_Match_Events_<date>.csv: the teams' names, as the feed's Home and Away.
const [home = "", away = ""] = SRC.split("/").at(-1)!.split("_Wisehockey_")[0]!.split("_");

// Locked by matching the Fink pass at 1767183821720 to video second 5314.24.
const ANCHOR_TS = 1767183821720;
const ANCHOR_VIDEO = 5314.24;
const videoTime = (ts: number) => (ts - ANCHOR_TS) / 1000 + ANCHOR_VIDEO;

// The teams change ends in period 2 and the source records its coordinates from the attacking side, so that period is rotated 180 degrees against the camera's rink frame.
const ROTATED_PERIODS = new Set(["2"]);

const xy = (v: string, period: string): [number, number] | null => {
	if (!v) return null;
	const [x, y] = v.split(",").map(Number);
	return ROTATED_PERIODS.has(period) ? [-x!, -y!] : [x!, y!];
};

const rows = csvParse((await Bun.file(SRC).text()).replace(/^﻿/, ""));

const goalieIds = new Set(
	rows.filter((r) => r.EventType === "Shot" && ["Saved", "Goal"].includes(r.ShotResult!)).map((r) => r.EventSecondaryPlayerId!),
);

const roster: { id: string; name: string; jersey: string; team: string; goalie: boolean }[] = [];
const rosterIndex = new Map<string, number>();
const nameById = new Map<string, string>();
const register = (id: string, label: string, team: string) => {
	nameById.set(id, label);
	if (rosterIndex.has(id)) return rosterIndex.get(id)!;
	const [jersey = "", ...name] = label.replace("#", "").split(" ");
	rosterIndex.set(id, roster.length);
	roster.push({ id, name: name.join(" "), jersey, team, goalie: goalieIds.has(id) });
	return roster.length - 1;
};

type Frame = (number | null)[];
const frames = new Map<number, Frame>();
for (const r of rows) {
	const ts = Number(r.Timestamp);
	const frame: Frame = [Math.round(videoTime(ts) * 100) / 100, null, null];
	const puck = xy(r.EventStartCoordinate!, r.Period!);
	if (puck && r.EventType !== "Faceoff") [frame[1], frame[2]] = [puck[0], puck[1]];
	for (let s = 1; s <= 12; s++) {
		const id = r[`StartPlayerId${s}`]!;
		const pos = xy(r[`StartPlayerCoordinates${s}`]!, r.Period!);
		if (!id || !pos) continue;
		frame.push(register(id, r[`StartPlayerName${s}`]!, r[`StartPlayerTeam${s}`]!), pos[0], pos[1]);
	}
	frames.set(ts, frame);
}

const goalShots = rows.filter((r) => r.EventType === "Shot" && r.ShotResult === "Goal");
const shotFor = (goal: (typeof rows)[number]) =>
	goalShots.findLast(
		(s) => Number(s.Timestamp) <= Number(goal.Timestamp) && s.EventPrimaryPlayerId === goal.EventPrimaryPlayerId,
	);

const kinds: Record<string, string> = { Goal: "goal", Penalty: "penalty", Shot: "shot" };
const events = rows
	.filter((r) => kinds[r.EventType!] && !(r.EventType === "Shot" && r.ShotResult === "Goal"))
	.map((r) => ({
		t: Math.round(videoTime(Number(r.Timestamp)) * 100) / 100,
		kind: kinds[r.EventType!],
		result: r.EventType === "Shot" ? r.ShotResult : (r.PenaltyReason ?? ""),
		team: r.EventPrimaryTeam,
		player: r.EventPrimaryPlayerName,
		assists: [r.GoalAssist1Id, r.GoalAssist2Id].filter(Boolean).map((id) => nameById.get(id!) ?? ""),
		period: Number(r.Period),
		clock: Number(r.MatchClock),
		xg: (r.EventType === "Goal" ? shotFor(r) : r)?.ShotXG ? Number((r.EventType === "Goal" ? shotFor(r) : r)!.ShotXG) : null,
		xy: xy((r.EventType === "Goal" ? shotFor(r) : r)?.EventStartCoordinate ?? "", r.Period!),
		strength: r.TeamStrength,
		shotT: r.EventType === "Goal" && shotFor(r) ? Math.round(videoTime(Number(shotFor(r)!.Timestamp)) * 100) / 100 : undefined,
	}));

const periods = [1, 2, 3].map((p) => {
	const first = rows.find((r) => r.Period === String(p))!;
	return { period: p, t: Math.round(videoTime(Number(first.Timestamp)) * 100) / 100 };
});

const sorted = [...frames.entries()].sort((a, b) => a[0] - b[0]).map(([, f]) => f.flat());
await Bun.write(OUT, JSON.stringify({ teams: { Home: home, Away: away }, roster, periods, events, frames: sorted }));
const last = sorted.at(-1)![0];
console.log(`${events.length} events, ${sorted.length} frames, ${roster.length} players, ends at video ${last}s`);
