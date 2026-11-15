// The ice graphics' one visual language, as a model: what is drawn on (and over) the ice in a frame, in rink metres, whatever
// draws it (shapes.ts lays it out through a Project, canvas.ts and gl.ts fill it in, labels.ts puts its text over it).
//   player    a ring on the ice under each player in their team's colour, and their jersey tag over the head (the puck
//             carrier's with a white triangle)
//   focus     a stronger ring under whoever the moment is about (scorer, passer, goalie; a goal's assists, a shooter), with
//             their name and role in a card below
//   trail     each skater's path over the last TRAIL s, a ribbon fading out behind them (the focus players' wider)
//   pass      the puck's pass, dashed; carry: its path while carried, solid; shot: its flight as an arc through the air with
//             its shadow on the ice and a ring opening where it ended (white for a goal); all in PUCK orange
//   zones     each team's territory (whose skater gets there first, as Voronoi cells), one shape a team and one line between
//   mark      a shot of the shot map, a ring sized by its xG
//   card      the glass card (labels.ts), the one style for numbers
// Everything but the tags, the cards and the shot's arc lies on the ice, under the players (cut out of the video, or behind
// them in depth). What a view shows is a preset, the same names everywhere (PRESETS).
import { SIDE, team as teamOf } from "../lib/league";
import { TEAM_COLOR } from "../player/events";

export type Pt = [number, number];
/** A frame's camera as a function: rink metres to CSS px over the view, null behind the camera. */
export type Project = (x: number, y: number, z?: number) => Pt | null;
/** 0 home, 1 away, -1 neither (officials) */
export type Team = 0 | 1 | -1;

/** A shot's path: from (sx, sy) at z0 to (tx, ty) at tz m, sagging `sag` m at its middle, over `dur` s. */
export type Flight = { sx: number; sy: number; tx: number; ty: number; z0: number; tz: number; sag: number; dur: number };

export type Card = {
	kind: "card";
	key: string;
	/** where it is pinned (m): on the ice, or z up from it */
	x: number;
	y: number;
	z?: number;
	/** leader: off to the side with a leader to the point, towards `toward`; above or below: right over or under it */
	anchor: "leader" | "above" | "below";
	toward?: Pt;
	team: Team;
	/** a line over it: what or who it is about */
	kicker: string;
	/** a name, beside the team's emblem */
	title?: string;
	/** the one big number, and its unit */
	value?: string;
	unit?: string;
	stats?: [string, string][];
	/** the net's mouth, rows from the crossbar down, 1 open */
	net?: number[][];
	/** how far it has come in, 0 to 1 */
	p: number;
};

export type Element =
	| { kind: "player"; key: string; x: number; y: number; team: Team; tag: string; carrier: boolean }
	| { kind: "focus"; key: string; x: number; y: number; team: Team; strong: boolean }
	| { kind: "trail"; key: string; team: Team; strong: boolean; pts: Pt[] }
	| { kind: "pass"; from: Pt; to: Pt }
	| { kind: "carry"; pts: Pt[] }
	| { kind: "shot"; flight: Flight; dt: number; goal: boolean }
	| { kind: "zones"; sites: { x: number; y: number; team: Team }[] }
	| { kind: "mark"; key: string; x: number; y: number; r: number; team: Team; fill: number; goal: boolean; fade: number }
	| Card;

export type Layer = "players" | "focus" | "movement" | "puck" | "zones" | "shots" | "numbers";
const LAYER: Record<Element["kind"], Layer> = {
	player: "players",
	focus: "focus",
	trail: "movement",
	pass: "puck",
	carry: "puck",
	shot: "puck",
	zones: "zones",
	mark: "shots",
	card: "numbers",
};

export type PresetName = "clean" | "players" | "tactics" | "shots";
/** What a view shows, as a few named looks: name, label, layers. */
export const PRESETS: readonly (readonly [PresetName, string, readonly Layer[]])[] = [
	["clean", "Clean", []],
	["players", "Players", ["players", "focus", "puck", "numbers"]],
	["tactics", "Tactics", ["players", "focus", "movement", "puck", "zones", "numbers"]],
	["shots", "Shots", ["focus", "puck", "shots", "numbers"]],
];

export const layersOf = (preset: PresetName) => PRESETS.find(([n]) => n === preset)![2];
/** The elements a set of layers shows. */
export const only = (elements: Element[], layers: readonly Layer[]) => elements.filter((e) => layers.includes(LAYER[e.kind]));

export const PUCK = "#ff7a1a";
export const GOAL = "#ffffff";
export const color = (team: number) => (team === 1 ? TEAM_COLOR.Away : team === 0 ? TEAM_COLOR.Home : "#c9c9c9");
/** The emblem of team 0 (home) or 1 (away) of the final, none for -1. */
export const logo = (team: number) => (team < 0 ? null : teamOf(SIDE[team === 1 ? "Away" : "Home"]).logo);
/** hex as rgba components 0 to 1, with alpha a */
export const rgba = (hex: string, a: number): [number, number, number, number] => {
	const n = Number.parseInt(hex.slice(1), 16);
	return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, a];
};

export const TRAIL = 2; // s of each skater's path
export const RING = { player: 0.55, focus: 1.05 }; // m
export const G = 9.81;
export const SHOT_SPEED = 30; // m/s a shot travels, where only its ends are known (wrist shots about 25, slap shots up to 45)

/** A flight from (sx, sy) to (tx, ty, tz) at SHOT_SPEED (or over `dur` s), sagging as gravity has it. */
export function flight(sx: number, sy: number, tx: number, ty: number, tz: number, dur?: number): Flight {
	const d = dur ?? Math.hypot(tx - sx, ty - sy) / SHOT_SPEED;
	return { sx, sy, tx, ty, z0: 0.05, tz, sag: 0.5 * G * d * d, dur: Math.max(0.4, d) };
}

export function circle(x: number, y: number, r: number, n = 40): Pt[] {
	return Array.from({ length: n }, (_, k) => [x + r * Math.cos((2 * Math.PI * k) / n), y + r * Math.sin((2 * Math.PI * k) / n)]);
}

/** The rink's outline (half length hx, half width hy, corner radius), counter-clockwise. */
export function rinkOutline(hx: number, hy: number, corner: number, per = 12): Pt[] {
	const out: Pt[] = [];
	for (const [sx, sy, a0] of [
		[1, 1, 0],
		[-1, 1, 90],
		[-1, -1, 180],
		[1, -1, 270],
	] as const)
		for (let k = 0; k <= per; k++) {
			const a = ((a0 + (90 * k) / per) * Math.PI) / 180;
			out.push([sx * (hx - corner) + corner * Math.cos(a), sy * (hy - corner) + corner * Math.sin(a)]);
		}
	return out;
}
