export type Roster = { id: string; name: string; jersey: string; team: "Home" | "Away"; goalie: boolean };
export type GameEvent = {
	t: number;
	kind: "goal" | "penalty" | "shot";
	result: string;
	team: "Home" | "Away";
	player: string;
	assists: string[];
	period: number;
	clock: number;
	xg: number | null;
	xy: [number, number] | null;
	strength: string;
	/** video time of the shot itself, for goals (the goal event is stamped a moment later) */
	shotT?: number;
};
export type Game = {
	roster: Roster[];
	periods: { period: number; t: number }[];
	events: GameEvent[];
	/** [videoSec, puckX|null, puckY|null, rosterIdx, x, y, ...] */
	frames: (number | null)[][];
};

const token = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** Colours come from the CSS tokens in styles.css, so the DOM, SVG and canvas layers share one palette. */
export const COLOR = {
	ink: token("--ink"),
	paper: token("--paper"),
	dim: token("--dim"),
	live: token("--live"),
};
export const TEAMS = { Home: "U.S. Collegiate Selects", Away: "HC Davos" } as const;
export const TEAM_COLOR = { Home: token("--home"), Away: token("--away") } as const;

export type Player = { num: string; name: string; last: string };

/** "#21 Ryan Walsh" → number, full name and surname. */
export function player(label: string): Player {
	const m = /^#(\d+)\s+(.*)$/.exec(label);
	const name = m?.[2] ?? label;
	return { num: m?.[1] ?? "", name, last: name.split(" ").at(-1)! };
}

export const periodAt = (game: Game, t: number) => [...game.periods].reverse().find((q) => q.t <= t)?.period ?? 0;

const pad = (n: number) => String(Math.floor(n)).padStart(2, "0");
export const periodClock = (e: GameEvent) => `P${e.period} ${Math.floor(e.clock / 60)}:${pad(e.clock % 60)}`;
export const xgText = (xg: number) => (xg < 0.01 ? "<0.01" : xg.toFixed(2));
const infraction = (e: GameEvent) =>
	e.result
		.replace(/([a-z])([A-Z])/g, "$1 $2")
		.toLowerCase()
		.replace(/^./, (c) => c.toUpperCase());

export function eventTitle(e: GameEvent): string {
	if (e.kind === "goal") return "Goal";
	if (e.kind === "penalty") return "Penalty";
	return `Shot ${e.result.toLowerCase()}`;
}

/** Secondary facts, in reading order, shared by the tooltip and the banner. */
export function eventDetails(e: GameEvent): string[] {
	if (e.kind === "goal") {
		const assists = e.assists.map((a) => player(a).name).join(", ");
		return [assists ? `Assist ${assists}` : "Unassisted", e.strength, e.xg != null ? `xG ${xgText(e.xg)}` : ""].filter(Boolean);
	}
	if (e.kind === "penalty") return [infraction(e)];
	return e.xg != null ? [`xG ${xgText(e.xg)}`] : [];
}

/** Player name with the jersey number set apart. */
export const playerHTML = (label: string) => {
	const p = player(label);
	return `${p.num ? `<span class="text-dim">${p.num}</span> ` : ""}${p.name}`;
};

/** Timeline marker: colour says which team, shape says what happened. */
export function markerHTML(e: Pick<GameEvent, "kind" | "result">, color: string): string {
	if (e.kind === "goal") {
		return `<span class="flex flex-col items-center"><i class="size-[7px] rounded-full" style="background:${color}"></i><i class="h-[9px] w-[2px]" style="background:${color}"></i></span>`;
	}
	if (e.kind === "penalty") return `<i class="size-[7px] rounded-[1px] border-[1.5px]" style="border-color:${color}"></i>`;
	const saved = e.result === "Saved";
	return `<i class="w-[2px] ${saved ? "h-[7px]" : "h-[5px]"}" style="background:${color};opacity:${saved ? 0.9 : 0.4}"></i>`;
}
