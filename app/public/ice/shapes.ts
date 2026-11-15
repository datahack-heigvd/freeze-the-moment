// The model's elements (model.ts) laid out through a frame's Project as flat shapes in CSS px, which canvas.ts and gl.ts both fill:
// so the two draw the same thing to the pixel but for how each smooths its edges. A line is a ribbon round its centre (half width
// h px, mitred joins), a ring a closed ribbon, an area a polygon fan; each shape one colour, with the rink point under each vertex
// for the occlusion (gl.ts) and whether it lies on the ice or in the air.
import { Delaunay } from "d3";
import { circle, color, type Element, type Flight, GOAL, type Project, type Pt, PUCK, RING, rgba, rinkOutline } from "./model";

/** a centre (px), its unit normal scaled for the mitre, the half width (px; 0 inside an area), which side (-1, 1), the rink point */
export type Vertex = { x: number; y: number; nx: number; ny: number; h: number; s: number; w: [number, number, number] };
export type Shape = { v: Vertex[]; tris: number[]; loops: number[][]; rgba: [number, number, number, number]; ice: boolean };
export type Rink = { hx: number; hy: number; corner: number };

type W = [number, number, number];
const MITRE = 4; // the longest mitre, in half widths
const DASH = { pass: [0.6, 0.45], zone: [0.8, 0.6], shadow: [0.5, 0.8] } as const; // m on, m off
const ARC = { steps: 28, tail: 4, ring: 0.8, hold: 1.5 }; // segments of a whole flight; s the arc is shown; s the end ring opens over; s held
const HEAD = 6; // px, the puck in flight
const SNAP = 1e4; // per m: the grid areas' corners are snapped to

/**
 * A ribbon along rink points `pts`, `width` px wide (or `{ m }` metres, as it lies on the ice), closed or not; null if any of it
 * is behind the camera.
 */
function ribbon(
	P: Project,
	pts: W[],
	width: number | { m: number },
	c: Shape["rgba"],
	ice: boolean,
	closed = false,
): Shape | null {
	const px: Pt[] = [],
		half: number[] = [],
		ws: W[] = [];
	for (let i = 0; i < pts.length; i++) {
		const [x, y, z] = pts[i]!,
			p = P(x, y, z);
		if (!p) return null;
		if (px.length && Math.hypot(p[0] - px.at(-1)![0], p[1] - px.at(-1)![1]) < 0.05) continue;
		let h = typeof width === "number" ? width / 2 : 0;
		if (typeof width !== "number") {
			// the width across the path on the ice, as seen
			const a = pts[Math.max(i - 1, 0)]!,
				b = pts[Math.min(i + 1, pts.length - 1)]!,
				l = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1,
				q = P(x - ((b[1] - a[1]) / l) * (width.m / 2), y + ((b[0] - a[0]) / l) * (width.m / 2), z);
			if (!q) return null;
			h = Math.hypot(q[0] - p[0], q[1] - p[1]);
		}
		px.push(p);
		half.push(h);
		ws.push(pts[i]!);
	}
	const n = px.length;
	if (n < 2 || (closed && n < 3)) return null;
	const seg = (i: number): Pt => {
		const a = px[(i + n) % n]!,
			b = px[(i + 1) % n]!,
			l = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
		return [-(b[1] - a[1]) / l, (b[0] - a[0]) / l];
	};
	const v: Vertex[] = [],
		tris: number[] = [];
	for (let i = 0; i < n; i++) {
		const a = closed || i > 0 ? seg(i - 1) : seg(i),
			b = closed || i < n - 1 ? seg(i) : seg(i - 1);
		let mx = a[0] + b[0],
			my = a[1] + b[1];
		const ml = Math.hypot(mx, my);
		[mx, my] = ml < 1e-6 ? b : [mx / ml, my / ml];
		const k = Math.min(1 / Math.max(mx * b[0] + my * b[1], 1e-3), MITRE);
		for (const s of [-1, 1]) v.push({ x: px[i]![0], y: px[i]![1], nx: mx * k, ny: my * k, h: half[i]!, s, w: ws[i]! });
	}
	const last = closed ? n : n - 1;
	for (let i = 0; i < last; i++) {
		const j = (i + 1) % n;
		tris.push(2 * i, 2 * i + 1, 2 * j, 2 * i + 1, 2 * j + 1, 2 * j);
	}
	const left = Array.from({ length: n }, (_, i) => 2 * i),
		right = Array.from({ length: n }, (_, i) => 2 * (n - 1 - i) + 1);
	return { v, tris, loops: closed ? [left, right] : [[...left, ...right]], rgba: c, ice };
}

/** Areas: each polygon of rink points (on the ice) as a fan, all one shape and colour; null if any of it is behind the camera. */
function area(P: Project, polys: Pt[][], c: Shape["rgba"]): Shape | null {
	const v: Vertex[] = [],
		tris: number[] = [],
		loops: number[][] = [];
	for (const poly of polys) {
		const at = v.length,
			loop: number[] = [];
		for (const [x0, y0] of poly) {
			// snapped, so the corners two polygons share (clipped from either side) are one point and their edges meet exactly
			const x = Math.round(x0 * SNAP) / SNAP,
				y = Math.round(y0 * SNAP) / SNAP,
				p = P(x, y, 0);
			if (!p) return null;
			loop.push(v.length);
			v.push({ x: p[0], y: p[1], nx: 0, ny: 0, h: 0, s: 0, w: [x, y, 0] });
		}
		for (let k = 1; k + 1 < poly.length; k++) tris.push(at, at + k, at + k + 1);
		loops.push(loop);
	}
	return v.length ? { v, tris, loops, rgba: c, ice: true } : null;
}

/** A disc `r` px round rink point w, as seen. */
function disc(P: Project, w: W, r: number, c: Shape["rgba"]): Shape | null {
	const p = P(...w);
	if (!p) return null;
	const v: Vertex[] = [{ x: p[0], y: p[1], nx: 0, ny: 0, h: 0, s: 0, w }],
		tris: number[] = [];
	for (let k = 0; k < 24; k++) {
		const a = (2 * Math.PI * k) / 24;
		v.push({ x: p[0] + r * Math.cos(a), y: p[1] + r * Math.sin(a), nx: 0, ny: 0, h: 0, s: 0, w });
		tris.push(0, 1 + k, 1 + ((k + 1) % 24));
	}
	return { v, tris, loops: [v.map((_, k) => k).slice(1)], rgba: c, ice: false };
}

/** Polyline `pts` (rink points) every `step` m. */
function densify(pts: W[], step: number): W[] {
	const out: W[] = [];
	for (let k = 0; k + 1 < pts.length; k++) {
		const [a, b] = [pts[k]!, pts[k + 1]!],
			n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / step));
		for (let j = 0; j < n; j++)
			out.push([a[0] + ((b[0] - a[0]) * j) / n, a[1] + ((b[1] - a[1]) * j) / n, a[2] + ((b[2] - a[2]) * j) / n]);
	}
	if (pts.length) out.push(pts.at(-1)!);
	return out;
}

/** Polyline `pts` cut into dashes `on` m long, `off` m apart, each densified. */
function dashes(pts: W[], on: number, off: number): W[][] {
	const out: W[][] = [];
	let cur: W[] = [],
		left = on,
		drawing = true;
	for (let k = 0; k + 1 < pts.length; k++) {
		let a = pts[k]!;
		const b = pts[k + 1]!;
		let l = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
		if (drawing && !cur.length) cur.push(a);
		while (l > left) {
			const f = left / l,
				m: W = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
			if (drawing) {
				cur.push(m);
				out.push(densify(cur, 0.5));
				cur = [];
			} else cur = [m];
			l -= left;
			a = m;
			drawing = !drawing;
			left = drawing ? on : off;
		}
		left -= l;
		if (drawing) cur.push(b);
	}
	if (drawing && cur.length > 1) out.push(densify(cur, 0.5));
	return out;
}

const flat = (pts: Pt[], z = 0): W[] => pts.map(([x, y]) => [x, y, z]);

/** Convex polygon `poly` clipped to convex polygon `clip` (both counter-clockwise). */
function clipConvex(poly: Pt[], clip: Pt[]): Pt[] {
	let out = poly;
	for (let k = 0; k < clip.length && out.length; k++) {
		const [a, b] = [clip[k]!, clip[(k + 1) % clip.length]!],
			inside = (p: Pt) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0,
			cut = (p: Pt, q: Pt): Pt => {
				const d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]),
					d2 = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]),
					t = d1 / (d1 - d2);
				return [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
			},
			src = out;
		out = [];
		for (let i = 0; i < src.length; i++) {
			const p = src[i]!,
				q = src[(i + 1) % src.length]!;
			if (inside(p)) out.push(p);
			if (inside(p) !== inside(q)) out.push(cut(p, q));
		}
	}
	return out;
}

/** Segment a-b clipped to convex polygon `clip`, or null. */
function clipSegment(a: Pt, b: Pt, clip: Pt[]): [Pt, Pt] | null {
	let t0 = 0,
		t1 = 1;
	for (let k = 0; k < clip.length; k++) {
		const [p, q] = [clip[k]!, clip[(k + 1) % clip.length]!],
			nx = -(q[1] - p[1]),
			ny = q[0] - p[0],
			den = nx * (b[0] - a[0]) + ny * (b[1] - a[1]),
			num = nx * (a[0] - p[0]) + ny * (a[1] - p[1]);
		if (Math.abs(den) < 1e-12) {
			if (num < 0) return null;
			continue;
		}
		const t = -num / den;
		if (den > 0) t0 = Math.max(t0, t);
		else t1 = Math.min(t1, t);
		if (t0 > t1) return null;
	}
	const at = (t: number): Pt => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
	return [at(t0), at(t1)];
}

const key = (p: Pt) => `${p[0].toFixed(5)},${p[1].toFixed(5)}`;

/** The zones: each team's cells (clipped to the rink), a shape a team, and the line between the teams, dashed. */
function zones(P: Project, sites: { x: number; y: number; team: number }[], rink: Rink, out: (s: Shape | null) => void) {
	if (sites.length < 3) return;
	const bounds = rinkOutline(rink.hx, rink.hy, rink.corner),
		voronoi = Delaunay.from(
			sites,
			(d) => d.x,
			(d) => d.y,
		).voronoi([-rink.hx - 2, -rink.hy - 2, rink.hx + 2, rink.hy + 2]),
		cells = sites.map((_, j) => (voronoi.cellPolygon(j)?.slice(0, -1) ?? null) as Pt[] | null);
	for (const team of [0, 1] as const)
		out(
			area(
				P,
				cells.flatMap((c, j) => (c && sites[j]!.team === team ? [clipConvex(ccw(c), bounds)] : [])).filter((c) => c.length > 2),
				rgba(color(team), 0.14),
			),
		);
	cells.forEach((cell, j) => {
		if (!cell) return;
		const mine = new Set(cell.map(key));
		for (const k of voronoi.neighbors(j)) {
			const other = cells[k];
			if (k < j || !other || sites[k]!.team === sites[j]!.team) continue;
			const shared = other.filter((p) => mine.has(key(p)));
			const seg = shared.length === 2 ? clipSegment(shared[0]!, shared[1]!, bounds) : null;
			if (seg) for (const d of dashes(flat(seg), ...DASH.zone)) out(ribbon(P, d, 2, [1, 1, 1, 0.7], true));
		}
	});
}

const ccw = (poly: Pt[]) => {
	let a = 0;
	for (let k = 0; k < poly.length; k++) {
		const [p, q] = [poly[k]!, poly[(k + 1) % poly.length]!];
		a += p[0] * q[1] - q[0] * p[1];
	}
	return a < 0 ? [...poly].reverse() : poly;
};

/** A shot's flight `dt` s after it was taken: its arc so far and its shadow, then the ring opening where it ended. */
function shot(P: Project, fl: Flight, dt: number, goal: boolean, out: (s: Shape | null) => void) {
	const prog = Math.min(1, dt / fl.dur),
		fade = dt <= fl.dur + ARC.hold ? 1 : Math.max(0, 1 - (dt - fl.dur - ARC.hold) / (ARC.tail - fl.dur - ARC.hold)),
		at = (u: number): W => [
			fl.sx + (fl.tx - fl.sx) * u,
			fl.sy + (fl.ty - fl.sy) * u,
			fl.z0 + (fl.tz - fl.z0) * u + fl.sag * u * (1 - u),
		],
		steps = Math.max(2, Math.ceil(prog * ARC.steps)),
		pts = Array.from({ length: steps + 1 }, (_, k) => at((prog * k) / steps));
	if (fade <= 0) return;
	for (const d of dashes(
		pts.map(([x, y]): W => [x, y, 0]),
		...DASH.shadow,
	))
		out(ribbon(P, d, 2.5, [0, 0, 0, 0.3 * fade], true));
	for (let k = 1; k < pts.length; k++) {
		const f = k / pts.length;
		out(ribbon(P, [pts[k - 1]!, pts[k]!], 2 + 5 * f, rgba(PUCK, (0.2 + 0.8 * f) * fade), false));
	}
	if (prog < 1) return void out(disc(P, pts.at(-1)!, HEAD, [1, 1, 1, 1]));
	const k = Math.min(1, (dt - fl.dur) / ARC.ring);
	if (k < 1)
		out(
			ribbon(
				P,
				flat(circle(fl.tx, fl.ty, 0.35 + 0.9 * k, 28)),
				goal ? 5 : 3,
				rgba(goal ? GOAL : PUCK, (1 - k) * fade),
				true,
				true,
			),
		);
}

/** A ring `r` m round (x, y): its fill and its edge `width` px. */
function ring(
	P: Project,
	x: number,
	y: number,
	r: number,
	edge: string,
	width: number,
	alpha: number,
	fill: number,
	out: (s: Shape | null) => void,
) {
	const c = circle(x, y, r, 40);
	if (fill > 0) out(area(P, [c], rgba(edge, fill)));
	out(ribbon(P, flat(c), width, rgba(edge, alpha), true, true));
}

/** Every element's shapes through P, in drawing order: zones, marks, trails, rings, the puck's story on the ice, then the arcs. */
export function layout(elements: Element[], P: Project, rink: Rink): Shape[] {
	const out: Shape[] = [],
		add = (s: Shape | null) => void (s && out.push(s)),
		order: Element["kind"][] = ["zones", "mark", "trail", "player", "focus", "carry", "pass", "shot"],
		// one ring a player: whoever is in focus has the focus ring alone
		focused = new Set(elements.flatMap((e) => (e.kind === "focus" ? [e.key.slice(1)] : [])));
	for (const kind of order)
		for (const e of elements) {
			if (e.kind !== kind) continue;
			if (e.kind === "zones") zones(P, e.sites, rink, add);
			else if (e.kind === "mark")
				ring(P, e.x, e.y, e.r, e.goal ? GOAL : color(e.team), e.goal ? 4 : 2.5, e.fade, e.fill * e.fade, add);
			else if (e.kind === "trail")
				for (let k = 1; k < e.pts.length; k++) {
					const f = k / (e.pts.length - 1); // 0 oldest, 1 newest
					add(
						ribbon(
							P,
							flat([e.pts[k - 1]!, e.pts[k]!], 0.01),
							{ m: (e.strong ? 0.5 : 0.24) * (0.4 + 0.6 * f) },
							rgba(color(e.team), (e.strong ? 0.9 : 0.55) * f * f),
							true,
						),
					);
				}
			else if (e.kind === "player") {
				if (!focused.has(e.key.slice(1))) ring(P, e.x, e.y, RING.player, color(e.team), 2, 0.95, 0.1, add);
			} else if (e.kind === "focus")
				ring(P, e.x, e.y, RING.focus, color(e.team), e.strong ? 5 : 3, 1, e.strong ? 0.22 : 0.12, add);
			else if (e.kind === "carry") add(ribbon(P, densify(flat(e.pts, 0.02), 0.5), 3, rgba(PUCK, 0.95), true));
			else if (e.kind === "pass")
				for (const d of dashes(flat([e.from, e.to], 0.02), ...DASH.pass)) add(ribbon(P, d, 3, rgba(PUCK, 0.95), true));
			else if (e.kind === "shot") shot(P, e.flight, e.dt, e.goal, add);
		}
	return out;
}
