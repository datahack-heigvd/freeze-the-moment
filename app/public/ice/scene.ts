// The language's elements (model.ts) of a 3D replay's frame, from its scene.json and anatomy: who is where (S.people's roots), the
// puck (S.puck), and the goal's make-up (S.anatomy: the pass, the carry, the shot, its numbers). Fractional frames interpolate.
import type { SceneData } from "../replay/types";
import { type Card, type Element, flight, type Pt, TRAIL } from "./model";

const REACH = 1.5, // m from the puck within which a skater has it
	CLEAR = 0.5, // m nearer than anyone else that skater must be
	HOLD = 12, // frames the passer stays in focus after the pass is taken
	REVEAL = 0.6, // s the card takes to come in
	NET = { z: 0.45 }; // m up the shot's flight ends in the net's mouth

/** Index in S.people of whoever has meta.key `key` (-1 if none). */
const index = (S: SceneData, key: number | null) => (key === null ? -1 : S.people.findIndex((p) => p.key === key));

/** Rink point of track `xy` (at `scale` to m) at fractional frame f. */
function at(xy: [number, number][], f: number, scale = 1): Pt {
	const a = Math.min(Math.max(Math.floor(f), 0), xy.length - 1),
		b = Math.min(a + 1, xy.length - 1),
		w = Math.min(Math.max(f - a, 0), 1);
	return [(xy[a]![0] + (xy[b]![0] - xy[a]![0]) * w) / scale, (xy[a]![1] + (xy[b]![1] - xy[a]![1]) * w) / scale];
}

/**
 * Elements of exported frame f (fractional) of scene S, leaving out the people whose meta.key is in `hidden`; the one whose key is
 * `hover` (under the pointer) with a strong ring and trail.
 */
export function sceneElements(S: SceneData, f: number, hidden: ReadonlySet<number> = new Set(), hover?: number): Element[] {
	const a = S.anatomy,
		out: Element[] = [],
		puck = at(S.puck.xy, f),
		shown = (S.puck.show[Math.round(f)] ?? 0) > 0.5,
		on = S.people.map((p, j) => ({ p, j, xy: at(p.root, f, 1000) })).filter(({ p }) => p.team >= 0 && !hidden.has(p.key)),
		scorer = index(S, a?.scorer ?? null),
		passer = index(S, a?.passer ?? null),
		goalie = index(S, a?.goalie ?? null);
	// the carrier: the skater nearest the puck, within REACH and clearly nearest
	const near = on
			.filter(({ p }) => !p.goalie)
			.map((o) => ({ ...o, d: Math.hypot(o.xy[0] - puck[0], o.xy[1] - puck[1]) }))
			.sort((x, y) => x.d - y.d),
		carrier = shown && near[0] && near[0].d < REACH && (near[1]?.d ?? Infinity) - near[0].d > CLEAR ? near[0].j : -1;
	for (const { p, j, xy } of on) {
		const team = p.team as 0 | 1;
		out.push({
			kind: "player",
			key: `p${p.key}`,
			x: xy[0],
			y: xy[1],
			team,
			tag: (p.goalie ? "G" : "") + (p.jersey ?? ""),
			carrier: j === carrier,
		});
		const pts: Pt[] = [];
		for (let g = Math.max(f - TRAIL * S.fps, 0); g < f; g += 1) pts.push(at(p.root, g, 1000));
		pts.push(xy);
		const key = j === scorer || j === passer || p.key === hover;
		if (!p.goalie && pts.length > 1) out.push({ kind: "trail", key: `t${p.key}`, team, strong: key, pts });
		const passing = j === passer && a && f <= (a.received ?? a.release) + HOLD;
		if (j === scorer || passing || j === goalie || p.key === hover)
			out.push({ kind: "focus", key: `f${p.key}`, x: xy[0], y: xy[1], team, strong: j === scorer || p.key === hover });
	}
	out.push({
		kind: "zones",
		sites: on.filter(({ p }) => !p.goalie).map(({ p, xy }) => ({ x: xy[0], y: xy[1], team: p.team as 0 | 1 })),
	});
	if (!a) return out;
	const xy = (g: number) => at(S.puck.xy, g),
		took = a.received ?? a.release;
	if (a.pass_frame !== null && f >= a.pass_frame) {
		// between its ends the puck is often unseen, so the pass is the line from where it left to where it was taken
		const [from, to] = [xy(a.pass_frame), xy(took)],
			w = Math.min((f - a.pass_frame) / Math.max(took - a.pass_frame, 1), 1);
		out.push({ kind: "pass", from, to: [from[0] + (to[0] - from[0]) * w, from[1] + (to[1] - from[1]) * w] });
	}
	if (f >= took && a.release > took) {
		const pts: Pt[] = [];
		for (let g = took; g < Math.min(f, a.release); g++) pts.push(xy(g));
		pts.push(xy(Math.min(f, a.release)));
		out.push({ kind: "carry", pts });
	}
	if (f >= a.release) {
		const gx = (Math.sign(a.from[0]) || 1) * S.rink.goal_x,
			dt = (f - a.release) / S.fps;
		out.push({
			kind: "shot",
			flight: flight(a.from[0], a.from[1], gx, 0, NET.z, Math.max(a.in_net - a.release, 1) / S.fps),
			dt,
			goal: true,
		});
		const card: Card = {
			kind: "card",
			key: "shot",
			x: a.from[0],
			y: a.from[1],
			anchor: "leader",
			toward: [0, 0],
			team: -1,
			kicker: a.open ? "Shot" : "Scramble",
			value: a.distance.toFixed(1),
			unit: "m",
			stats: [
				[`${Math.round(a.angle)}°`, "off centre"],
				...(a.open ? ([[`${Math.round(a.open.share * 100)}%`, "net open"]] as [string, string][]) : []),
			],
			net: a.open?.grid,
			p: dt / REVEAL + 1e-3,
		};
		out.push(card);
	}
	return out;
}
