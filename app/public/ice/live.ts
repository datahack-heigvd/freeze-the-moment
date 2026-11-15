// The language's elements (model.ts) of a moment of the livestream, from the AR clip's tracking (player/ar.ts Scene: who is where
// in each video frame, with their roster index) and the game's metadata (player/events.ts Game: the puck's samples, the events):
//   players  a ring and a tag for everyone identified, the puck carrier's (the metadata's, else the video's) with its triangle
//   trails   each identified skater's path over the last TRAIL s, back to where the tracking last lost them or jumped
//   focus    the players an event is about (a goal's scorer and assists, a shooter, the penalised), with their card below
//   puck     the carrier's run with the puck (carry) and the pass that gave it to them, from the metadata's samples; each shot's
//            flight (shotFlight) for ARC_TAIL s after it
//   zones    each team's territory, goalies left out
//   shots    this period's shot map (a mark each, sized by xG, fading with age) and the latest shot's xG card
import type { Scene } from "../player/ar";
import { type Game, type GameEvent, periodAt, player, xgText } from "../player/events";
import { type Element, type Flight, flight, type Pt, type Team, TRAIL } from "./model";

const GOAL_X = 26,
	SHOOTER_NEAR = 4, // m: a shooter tracked further than this from the event's spot did not take it from there: no arc
	LANE = 0.8, // m from a blocked shot's line within which an opponent is its blocker
	ARC_TAIL = 4, // s a shot's flight is shown
	SHOT_FADE = [20, 60] as const, // s after the shot: fully shown until the first, faded to a trace by the second
	SHOT_CARD = 10, // s the latest shot keeps its xG card
	REVEAL = 0.6, // s a card takes to come in
	CARRY_RADIUS = 2, // m between the puck and a player for that player to be the carrier (samples are good to about 1 m)
	CARRY_MARGIN = 0.7, // m the carrier must be nearer than the next nearest player, else nobody is
	CARRY_HOLD = 0.6, // s a carrier stays marked after its sample, unless the next sample has the same carrier
	CARRY_SPAN = 3, // s, the longest gap bridged between two samples with the same carrier
	PASS_GAP = 1.5, // s at most between a teammate's last touch and the carrier's first for it to be a pass
	SPEED_SPAN = 0.4, // s over which a player's speed is measured: shorter shakes with the tracking
	TRAIL_SPEED = 15; // m/s: a tracked skater moving faster than this has been switched with someone else, so their trail ends there

const shotRadius = (xg: number | null) => 0.35 + Math.sqrt(xg ?? 0.02) * 1.8;
const teamOf = (e: GameEvent): Team => (e.team === "Away" ? 1 : 0);
const rand = (e: GameEvent, k: number) => {
	const v = Math.sin(e.t * 12.9898 + k * 78.233) * 43758.5453;
	return v - Math.floor(v);
};

/** The AR clip's frame at video time t, or -1 outside it (or where its camera is unknown). */
export const frameAt = (s: Scene, t: number) => {
	const i = Math.round(t * s.fps) - s.n0;
	return i >= 0 && i < s.cam.length / 14 && s.cam[i * 14] !== 0 ? i : -1;
};

type Tracked = { idx: number; x: number; y: number; team: number };
/** Everyone tracked in frame i. */
function tracked(s: Scene, i: number): Tracked[] {
	const r = s.ctl[i] ?? [],
		out: Tracked[] = [];
	for (let j = 0; j < r.length; j += 4) out.push({ idx: r[j]!, x: r[j + 1]! / 100, y: r[j + 2]! / 100, team: r[j + 3]! });
	return out;
}

const labels = new WeakMap<Game, Map<string, number>>();
/** Roster index by event label ("#<jersey> <name>"). */
function labelOf(game: Game): Map<string, number> {
	let m = labels.get(game);
	if (!m) labels.set(game, (m = new Map(game.roster.map((p, i): [string, number] => [`#${p.jersey} ${p.name}`, i]))));
	return m;
}

/**
 * Where a shot went, from its result and the players tracked when it was taken: from the shooter, a goal to the side of the net
 * away from the goalie, a save to the goalie, a block to the opponent standing in its lane, a miss wide of the post. Null when the
 * tracking puts the shooter elsewhere.
 */
function shotFlight(e: GameEvent, s: Scene, game: Game): Flight | null {
	let [sx, sy] = e.xy!;
	const team = teamOf(e),
		i = frameAt(s, e.shotT ?? e.t),
		at = i >= 0 ? tracked(s, i) : null,
		shooter = at?.find((p) => p.idx === labelOf(game).get(e.player));
	if (shooter) {
		if (Math.hypot(shooter.x - sx, shooter.y - sy) > SHOOTER_NEAR) return null;
		[sx, sy] = [shooter.x, shooter.y];
	}
	const dir = sx >= 0 ? 1 : -1,
		gx = dir * GOAL_X,
		goalie = at?.find((p) => p.team !== team && game.roster[p.idx]?.goalie && Math.sign(p.x) === dir),
		side = goalie && Math.abs(goalie.y) > 0.1 ? -Math.sign(goalie.y) : rand(e, 3) > 0.5 ? 1 : -1;
	let tx = gx,
		ty = side * (0.3 + rand(e, 1) * 0.5),
		tz = 0.15 + rand(e, 2) * 0.75;
	if (e.result === "Saved") {
		[tx, ty] = goalie ? [goalie.x - dir * 0.3, goalie.y] : [gx - dir * 1.0, (rand(e, 1) - 0.5) * 1.3];
		tz = 0.3 + rand(e, 2) * 0.5;
	}
	if (e.result === "Missed") [tx, ty, tz] = [gx + dir * 1.2, side * (1.3 + rand(e, 1) * 0.9), 0.5];
	if (e.result === "Blocked") {
		const len = Math.hypot(gx - sx, sy),
			[ux, uy] = [(gx - sx) / len, -sy / len],
			lane = (at ?? [])
				.filter((p) => p.team !== team && !game.roster[p.idx]?.goalie)
				.map((p) => ({ p, along: (p.x - sx) * ux + (p.y - sy) * uy, off: Math.abs((p.x - sx) * uy - (p.y - sy) * ux) }))
				.filter((q) => q.along > 1 && q.along < len - 1 && q.off < LANE)
				.sort((a, b) => a.along - b.along)[0],
			f = 0.3 + rand(e, 1) * 0.2;
		[tx, ty] = lane ? [lane.p.x - ux * 0.3, lane.p.y - uy * 0.3] : [sx + (gx - sx) * f, sy * (1 - f)];
		tz = 0.3;
	}
	return flight(sx, sy, tx, ty, tz);
}

/** A metadata sample: its time, the puck, and the roster player carrying it (-1 if nobody clearly is). */
type Sample = { t: number; puck: Pt; idx: number };
const samples = new WeakMap<Game, Sample[]>();
function samplesOf(game: Game): Sample[] {
	let c = samples.get(game);
	if (c) return c;
	c = [];
	for (const f of game.frames) {
		const [t, px, py] = f;
		if (t == null || px == null || py == null) continue;
		let best = -1,
			bd = Number.POSITIVE_INFINITY,
			second = Number.POSITIVE_INFINITY;
		for (let j = 3; j + 2 < f.length; j += 3) {
			const d = Math.hypot(f[j + 1]! - px, f[j + 2]! - py);
			if (d < bd) [best, bd, second] = [f[j]!, d, bd];
			else if (d < second) second = d;
		}
		c.push({ t, puck: [px, py], idx: bd <= CARRY_RADIUS && second - bd >= CARRY_MARGIN ? best : -1 });
	}
	samples.set(game, c);
	return c;
}

/** Index of the last sample at or before t (-1 if none). */
function sampleAt(c: Sample[], t: number) {
	let lo = 0,
		hi = c.length;
	while (lo < hi) {
		const m = (lo + hi) >> 1;
		if (c[m]!.t <= t) lo = m + 1;
		else hi = m;
	}
	return lo - 1;
}

/** Roster index of the puck carrier at video time t, or -1 when unknown (puck in flight, between distant samples). */
function carrierAt(game: Game, t: number): number {
	const c = samplesOf(game),
		k = sampleAt(c, t),
		prev = c[k],
		next = c[k + 1];
	if (!prev || prev.idx < 0) return -1;
	if (next && next.idx === prev.idx && next.t - prev.t <= CARRY_SPAN) return prev.idx;
	return t - prev.t <= CARRY_HOLD ? prev.idx : -1;
}

/** Where the puck is at t along samples q (sorted by time, t within them): between the two around t, in proportion. */
function puckAlong(q: Sample[], t: number): Pt {
	let j = 0;
	while (j < q.length - 2 && q[j + 1]!.t <= t) j++;
	const u = q[j]!,
		v = q[j + 1] ?? u,
		f = v.t > u.t ? Math.min(1, Math.max(0, (t - u.t) / (v.t - u.t))) : 1;
	return [u.puck[0] + (v.puck[0] - u.puck[0]) * f, u.puck[1] + (v.puck[1] - u.puck[1]) * f];
}

/** A pass: from a player's last sample with the puck, through the loose samples, to a teammate's first (sample index `at`). */
type Pass = { at: number; to: number; path: Sample[] };
const passes = new WeakMap<Game, Pass[]>();
function passesOf(game: Game): Pass[] {
	let out = passes.get(game);
	if (out) return out;
	const c = samplesOf(game);
	out = [];
	for (let a = 1; a < c.length; a++) {
		const to = c[a]!;
		if (to.idx < 0 || c[a - 1]!.idx === to.idx) continue;
		// back past the loose samples to whoever had it before
		let b = a - 1;
		while (b >= 0 && c[b]!.idx < 0 && to.t - c[b]!.t <= PASS_GAP) b--;
		const from = c[b];
		if (
			from &&
			from.idx >= 0 &&
			from.idx !== to.idx &&
			game.roster[from.idx]?.team === game.roster[to.idx]?.team &&
			to.t - from.t <= PASS_GAP
		)
			out.push({ at: a, to: to.idx, path: c.slice(b, a + 1) });
	}
	passes.set(game, out);
	return out;
}

/**
 * The puck's story at t: a pass on its way, drawn out to where the puck is (the samples are about 0.6 s apart, so it moves between
 * them); else the carrier's run with the puck over TRAIL s at most, up to where the puck is now, and the pass from a teammate that
 * started it.
 */
function puckStory(game: Game, t: number, carrier: number): Element[] {
	const c = samplesOf(game),
		k = sampleAt(c, t),
		all = passesOf(game),
		flying = all.find((p) => p.path[0]!.t < t && t < p.path[p.path.length - 1]!.t);
	if (flying) return [{ kind: "pass", from: flying.path[0]!.puck, to: puckAlong(flying.path, t) }];
	if (carrier < 0 || k < 0 || c[k]!.idx !== carrier) return [];
	let a = k;
	while (a > 0 && c[a - 1]!.idx === carrier && t - c[a - 1]!.t <= TRAIL) a--;
	const out: Element[] = [],
		run = c.slice(a, k + 1).map((q) => q.puck),
		next = c[k + 1];
	if (next && next.idx === carrier && next.t - c[k]!.t <= CARRY_SPAN) run.push(puckAlong([c[k]!, next], t));
	if (run.length > 1) out.push({ kind: "carry", pts: run });
	const pass = all.find((p) => p.at === a);
	if (pass && t - c[a]!.t <= TRAIL) out.push({ kind: "pass", from: pass.path[0]!.puck, to: c[a]!.puck });
	return out;
}

/** Where identified roster player idx is in frame g, or null. */
function ringAt(s: Scene, g: number, idx: number): Pt | null {
	const q = s.rings[g] ?? [];
	for (let k = 0; k < q.length; k += 4) if (q[k] === idx && q[k + 3]! & 1) return [q[k + 1]! / 100, q[k + 2]! / 100];
	return null;
}

/**
 * Player idx's path up to `here` (frame i), every `step` frames over `back` frames at most, oldest first. It goes back only while
 * each earlier frame has them, near where they were next: where they are missing or jump (another skater given their number),
 * the path ends rather than joining the two.
 */
function trailOf(s: Scene, idx: number, here: Pt, i: number, back: number, step: number): Pt[] {
	const pts = [here],
		reach = (TRAIL_SPEED * step) / s.fps;
	for (let g = i - step; g >= Math.max(i - back, 0); g -= step) {
		const q = ringAt(s, g, idx),
			last = pts[pts.length - 1]!;
		if (!q || Math.hypot(q[0] - last[0], q[1] - last[1]) > reach) break;
		pts.push(q);
	}
	return pts.reverse();
}

type Focus = { idx: number; role: string; strong: boolean; since: number };
/** Players the current event is about: scorer and assists of a goal, the shooter of a shot, the penalised player. */
function focusAt(game: Game, t: number): Focus[] {
	const ids = labelOf(game),
		out: Focus[] = [];
	const add = (label: string | undefined, role: string, strong: boolean, since: number) => {
		const idx = label ? ids.get(label) : undefined;
		if (idx !== undefined) out.push({ idx, role, strong, since });
	};
	for (const e of game.events) {
		const dt = t - e.t;
		if (e.kind === "goal" && dt > -1.8 && dt < 6) {
			add(e.player, "Goal", true, e.t - 1.8);
			for (const a of e.assists) add(a, "Assist", false, e.t - 1.8);
		} else if (e.kind === "shot" && dt > -0.6 && dt < 2.5) add(e.player, "Shot", true, e.t - 0.6);
		else if (e.kind === "penalty" && dt > 0 && dt < 5) add(e.player, "Penalty", true, e.t);
	}
	return out;
}

/** Identified roster player idx's speed at frame i (km/h), over SPEED_SPAN s round it; null where the tracking lost or swapped them. */
export function speedAt(s: Scene, idx: number, i: number): number | null {
	const w = Math.max(1, Math.round((SPEED_SPAN * s.fps) / 2)),
		a = ringAt(s, i - w, idx),
		b = ringAt(s, i + w, idx);
	if (!a || !b) return null;
	const v = Math.hypot(b[0] - a[0], b[1] - a[1]) / ((2 * w) / s.fps);
	return v > TRAIL_SPEED ? null : v * 3.6;
}

/**
 * Elements of the livestream at video time t, AR clip frame i (frameAt); the element with key `hover` (a player's, or a shot's
 * mark) marked out: a player with a strong ring and trail, a shot with its card.
 */
export function liveElements(s: Scene, game: Game, t: number, i: number, hover: string | null = null): Element[] {
	const out: Element[] = [],
		r = s.rings[i] ?? [],
		meta = carrierAt(game, t),
		carrier = meta >= 0 ? meta : (s.car?.[i] ?? -1),
		focus = focusAt(game, t),
		back = Math.round(TRAIL * s.fps),
		step = Math.max(1, Math.round(s.fps / 15));
	for (let j = 0; j < r.length; j += 4) {
		const idx = r[j]!,
			x = r[j + 1]! / 100,
			y = r[j + 2]! / 100,
			flags = r[j + 3]!,
			team = ((flags >> 1) & 1) as Team,
			p = game.roster[idx];
		if (!(flags & 1) || idx < 0 || !p) continue;
		out.push({ kind: "player", key: `p${idx}`, x, y, team, tag: (p.goalie ? "G" : "") + p.jersey, carrier: idx === carrier });
		const f = focus.find((q) => q.idx === idx),
			on = hover === `p${idx}`;
		if (!p.goalie) {
			const pts = trailOf(s, idx, [x, y], i, back, step);
			if (pts.length > 1) out.push({ kind: "trail", key: `t${idx}`, team, strong: !!f?.strong || on, pts });
		}
		if (on && !f) out.push({ kind: "focus", key: `f${idx}`, x, y, team, strong: true });
		if (f) {
			out.push({ kind: "focus", key: `f${idx}`, x, y, team, strong: f.strong || on });
			out.push({
				kind: "card",
				key: `f${idx}`,
				x,
				y: y - 1.05,
				anchor: "below",
				team,
				kicker: `${f.role} · #${p.jersey}`,
				title: player(`#${p.jersey} ${p.name}`).last,
				p: (t - f.since) / REVEAL,
			});
		}
	}
	const sites = tracked(s, i)
		.filter((q) => !(q.idx >= 0 && game.roster[q.idx]?.goalie))
		.map((q) => ({ x: q.x, y: q.y, team: q.team as Team }));
	out.push({ kind: "zones", sites });
	out.push(...puckStory(game, t, carrier));
	// only this period's shots: earlier periods were played with the teams at the other ends
	const period = periodAt(game, t),
		shots = game.events.filter((e) => e.xy && e.kind !== "penalty" && e.period === period);
	let latest: GameEvent | undefined;
	for (const e of shots) {
		const dt = t - (e.shotT ?? e.t);
		if (dt < 0) continue;
		latest = e;
		const [x, y] = e.xy!,
			fade = 1 - 0.65 * Math.min(1, Math.max(0, (dt - SHOT_FADE[0]) / (SHOT_FADE[1] - SHOT_FADE[0])));
		out.push({
			kind: "mark",
			key: `m${e.t}`,
			x,
			y,
			r: shotRadius(e.xg),
			team: teamOf(e),
			fill: e.result === "Missed" ? 0 : e.result === "Blocked" ? 0.2 : 0.45,
			goal: e.kind === "goal",
			fade,
		});
		const fl = dt <= ARC_TAIL ? shotFlight(e, s, game) : null;
		if (fl) out.push({ kind: "shot", flight: fl, dt, goal: e.kind === "goal" });
	}
	// the hovered shot's card; else the latest's, but while a player is in focus their card and the banner already tell its story
	const hovered = shots.find((e) => hover === `m${e.t}` && t >= (e.shotT ?? e.t)),
		age = latest ? t - (latest.shotT ?? latest.t) : Infinity,
		shot = hovered ?? (latest && age <= SHOT_CARD && !focus.length ? latest : undefined);
	if (shot && shot.xg != null)
		out.push({
			kind: "card",
			key: `xg${shot.t}`,
			x: shot.xy![0],
			y: shot.xy![1] - shotRadius(shot.xg),
			anchor: "above",
			team: teamOf(shot),
			kicker: shot.kind === "goal" ? "Goal" : shot.result,
			title: player(shot.player).last,
			value: xgText(shot.xg),
			unit: "xG",
			p: shot === hovered ? 1 : age / REVEAL,
		});
	return out;
}
