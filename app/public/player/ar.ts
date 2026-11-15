// The livestream's overlay (Live AR): the language of the ice graphics (ice/) drawn over the broadcast through the AR clip's camera,
// on a canvas laid over the object-cover video, redrawn for every presented video frame. What lies on the ice is drawn on a layer
// of its own with the people cut out of it (the clip's outlines), so it sits under them; the shots' arcs over everything; the tags
// and the cards are DOM (ice/labels.ts). A player or a shot's mark under the pointer is marked out, with its card (a player's:
// who they are, their speed and their numbers over the game); a click (a tap) keeps it so until the next one. Segments load as
// playback nears them.
import { drawShapes } from "../ice/canvas";
import { createLabels } from "../ice/labels";
import { frameAt, liveElements, speedAt } from "../ice/live";
import { type Card, type Element, layersOf, only, type PresetName, type Project, type Pt } from "../ice/model";
import { layout } from "../ice/shapes";
import { minutes, SIDE } from "../lib/league";
import { playerStats } from "../lib/site";
import { type Cutout, eraseOutlines, parseCutout } from "./cutout";
import { type Game, player } from "./events";

const TALL = 1.9, // m: a player is picked anywhere from their skates to their helmet
	REACH = 12, // px round a player's column, or a shot's mark, within which the pointer is on them
	BELOW = 1.05; // m towards the camera side of a player where their card hangs

/** Per-frame camera and tracker output of the AR clip (see pipeline/calib/scripts/export_ar.py). */
export type Scene = {
	n0: number;
	fps: number;
	fref: number;
	cx: number;
	cy: number;
	hx: number;
	hy: number;
	corner: number;
	/** per frame: [roster idx or -1, x cm, y cm, flags (bit0 number shown, bit1 away)] repeated */
	rings: number[][];
	/** per frame: [roster idx or -1, x cm, y cm, team (0 home, 1 away)] repeated, every on-ice player */
	ctl: number[][];
	/** per frame: roster idx of the puck carrier seen in the video while the metadata is missing, -1 otherwise */
	car?: number[];
	/** per frame: f, k, R (9, row major), C (3) */
	cam: Float32Array;
	/** outlines of the people on the ice, when the segment has them */
	cut?: Cutout;
};

/** One processed stretch of the game; its camera and tracker data live in /ar/<base>/. */
export type Segment = { id: string; base: string; t0: number; t1: number; bad?: number };
export type Index = { segments: Segment[]; defaultStart: number };

export const loadIndex = async (): Promise<Index> => (await fetch("/ar/index.json", { cache: "no-store" })).json();

export async function loadScene(seg: Segment): Promise<Scene> {
	const [meta, cam, cut] = await Promise.all([
		fetch(`/ar/${seg.base}/tracks.json`).then((r) => r.json()),
		fetch(`/ar/${seg.base}/camera.bin`).then((r) => r.arrayBuffer()),
		fetch(`/ar/${seg.base}/masks.bin`).then((r) => (r.ok ? r.arrayBuffer() : null)),
	]);
	return { ...meta, cam: new Float32Array(cam), cut: cut ? parseCutout(cut) : undefined };
}

/** Frame i's camera: rink metres to video px, through the lens (k). */
function projector(s: Scene, i: number): Project {
	const c = s.cam;
	const o = i * 14;
	const f = c[o]!;
	const k = c[o + 1]!;
	const R = Array.from(c.subarray(o + 2, o + 11));
	const C = Array.from(c.subarray(o + 11, o + 14));
	return (x, y, z = 0) => {
		const dx = x - C[0]!;
		const dy = y - C[1]!;
		const dz = z - C[2]!;
		const depth = R[6]! * dx + R[7]! * dy + R[8]! * dz;
		if (depth < 0.5) return null;
		const ux = (f * (R[0]! * dx + R[1]! * dy + R[2]! * dz)) / depth;
		const uy = (f * (R[3]! * dx + R[4]! * dy + R[5]! * dz)) / depth;
		const d = 1 + (k * (ux * ux + uy * uy)) / (s.fref * s.fref);
		return [ux * d + s.cx, uy * d + s.cy];
	};
}

let iceCanvas: HTMLCanvasElement | undefined;
/** A transparent layer with ctx's size, for what lies on the ice and gets the people cut out of it. */
function iceLayer(ctx: CanvasRenderingContext2D): CanvasRenderingContext2D {
	iceCanvas ??= document.createElement("canvas");
	const { width, height } = ctx.canvas;
	if (iceCanvas.width !== width || iceCanvas.height !== height) {
		iceCanvas.width = width;
		iceCanvas.height = height;
	}
	const l = iceCanvas.getContext("2d")!;
	l.setTransform(1, 0, 0, 1, 0, 0);
	l.clearRect(0, 0, width, height);
	return l;
}

/** The key of the element under view point `at` (CSS px) through P: a player's (nearest their column), else a shot's mark's. */
function pickAt(elements: Element[], P: Project, at: Pt): string | null {
	let best: string | null = null,
		near = Infinity;
	for (const e of elements) {
		if (e.kind !== "player") continue;
		const a = P(e.x, e.y, 0),
			b = P(e.x, e.y, TALL);
		if (!a || !b) continue;
		const dx = b[0] - a[0],
			dy = b[1] - a[1],
			u = Math.min(Math.max(((at[0] - a[0]) * dx + (at[1] - a[1]) * dy) / (dx * dx + dy * dy || 1), 0), 1),
			d = Math.hypot(at[0] - a[0] - u * dx, at[1] - a[1] - u * dy);
		if (d < Math.max(REACH, Math.hypot(dx, dy) * 0.25) && d < near) [best, near] = [e.key, d];
	}
	if (best) return best;
	for (const e of elements) {
		if (e.kind !== "mark") continue;
		const c = P(e.x, e.y),
			r = P(e.x + e.r, e.y);
		if (!c || !r) continue;
		const d = Math.hypot(at[0] - c[0], at[1] - c[1]);
		if (d < Math.hypot(r[0] - c[0], r[1] - c[1]) + REACH && d < near) [best, near] = [e.key, d];
	}
	return best;
}

/** Roster player idx's card, standing at (x, y) in frame i: who they are and, when `full`, their speed now, their top speed and ice
 * time in the game. */
function playerCard(s: Scene, game: Game, idx: number, x: number, y: number, i: number, full: boolean): Card {
	const r = game.roster[idx]!,
		stats = playerStats.find((q) => q.team === SIDE[r.team] && q.jersey === r.jersey),
		speed = r.goalie ? undefined : speedAt(s, idx, i);
	return {
		kind: "card",
		// a card keeps the rows it was made with: the short one and the full one are two cards
		key: `${full ? "h" : "n"}${idx}`,
		x,
		y: y - BELOW,
		anchor: "below",
		team: r.team === "Away" ? 1 : 0,
		kicker: `${r.goalie ? "Goalie" : "Skater"} · #${r.jersey}`,
		title: player(r.name).last,
		stats: !full
			? []
			: [
					...(speed === undefined ? [] : ([[speed === null ? "-" : `${Math.round(speed)}`, "km/h"]] as [string, string][])),
					...(stats
						? ([...(r.goalie ? [] : [[`${Math.round(stats.topSpeed)}`, "top km/h"]]), [minutes(stats.toi), "ice time"]] as [
								string,
								string,
							][])
						: []),
				],
		p: 1,
	};
}

/** Canvas laid over an object-cover video; redrawn for every presented video frame. Segments load when playback gets near them. */
export function createAR(canvas: HTMLCanvasElement, labelRoot: HTMLElement, video: HTMLVideoElement, index: Index, game: Game) {
	let layers = layersOf("players");
	const ctx = canvas.getContext("2d")!;
	const labels = createLabels(labelRoot);
	const scenes = new Map<string, Promise<Scene>>();
	const loaded = new Map<string, Scene>();
	let last = video.currentTime;
	// the pointer over the view (CSS px, a mouse or pen with no button down), what is under it, and what a click or tap kept
	let pointer: Pt | null = null,
		hovered: string | null = null,
		kept: string | null = null,
		seen: { elements: Element[]; P: Project } | null = null,
		queued = 0;

	const segmentAt = (t: number, margin = 0) => index.segments.find((g) => t >= g.t0 - margin && t < g.t1 + margin);
	function prefetch(t: number) {
		const seg = segmentAt(t, 30);
		if (!seg || scenes.has(seg.id)) return;
		const p = loadScene(seg);
		scenes.set(seg.id, p);
		p.then((sc) => {
			loaded.set(seg.id, sc);
			render(last);
		});
	}

	function render(t: number) {
		last = t;
		prefetch(t);
		const dpr = window.devicePixelRatio || 1;
		const cw = canvas.clientWidth;
		const ch = canvas.clientHeight;
		// detached or hidden: nothing to draw on
		if (!cw || !ch) return;
		if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
			canvas.width = Math.round(cw * dpr);
			canvas.height = Math.round(ch * dpr);
		}
		ctx.setTransform(1, 0, 0, 1, 0, 0);
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		const seg = segmentAt(t);
		const scene = seg && loaded.get(seg.id);
		const i = scene ? frameAt(scene, t) : -1;
		if (!scene || i < 0) {
			seen = null;
			return labels.clear();
		}
		const vw = video.videoWidth || 1920;
		const vh = video.videoHeight || 1080;
		const k = Math.max(cw / vw, ch / vh);
		const ox = (cw - vw * k) / 2;
		const oy = (ch - vh * k) / 2;
		const V = projector(scene, i);
		const P: Project = (x, y, z) => {
			const p = V(x, y, z);
			return p && [ox + p[0] * k, oy + p[1] * k];
		};
		let elements = only(liveElements(scene, game, t, i), layers);
		seen = { elements, P };
		const key = (pointer && pickAt(elements, P, pointer)) ?? (elements.some((e) => "key" in e && e.key === kept) ? kept : null);
		if (key !== hovered) {
			hovered = key;
			video.style.cursor = pointer && key ? "pointer" : "";
		}
		if (key) {
			elements = only(liveElements(scene, game, t, i, key), layers);
			const at = elements.find((e) => e.kind === "player" && e.key === key);
			if (at?.kind === "player" && layers.includes("numbers")) {
				const idx = Number(key.slice(1));
				elements = elements.filter((e) => !(e.kind === "card" && e.key === `f${idx}`));
				// passed over as the broadcast plays, only who they are; their numbers once it is paused, or they are kept
				elements.push(playerCard(scene, game, idx, at.x, at.y, i, video.paused || key === kept));
			}
		}
		const shapes = layout(elements, P, scene);
		const ice = scene.cut ? iceLayer(ctx) : ctx;
		ice.setTransform(dpr, 0, 0, dpr, 0, 0);
		drawShapes(ice, shapes, "ice");
		if (scene.cut) {
			ice.setTransform(dpr * k, 0, 0, dpr * k, dpr * ox, dpr * oy);
			eraseOutlines(ice, scene.cut, i);
			ctx.drawImage(iceCanvas!, 0, 0);
		}
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		drawShapes(ctx, shapes, "air");
		labels.place(elements, P);
	}

	let frameHandle = 0;
	const onFrame: VideoFrameRequestCallback = (_now, meta) => {
		render(meta.mediaTime);
		frameHandle = video.requestVideoFrameCallback(onFrame);
	};
	frameHandle = video.requestVideoFrameCallback(onFrame);
	const resize = new ResizeObserver(() => render(last));
	resize.observe(canvas);
	/** Drawn again on the next animation frame, as the pointer moves over a paused video. */
	const redraw = () => {
		queued ||= requestAnimationFrame(() => {
			queued = 0;
			render(last);
		});
	};
	const local = (e: MouseEvent): Pt => {
		const r = canvas.getBoundingClientRect();
		return [e.clientX - r.left, e.clientY - r.top];
	};
	const abort = new AbortController(),
		{ signal } = abort;
	video.addEventListener(
		"pointermove",
		(e) => {
			pointer = e.pointerType === "touch" || e.buttons ? null : local(e);
			redraw();
		},
		{ signal },
	);
	for (const turn of ["play", "pause"]) video.addEventListener(turn, redraw, { signal });
	video.addEventListener(
		"pointerleave",
		() => {
			pointer = null;
			redraw();
		},
		{ signal },
	);

	return {
		segments: index.segments,
		start: index.defaultStart,
		preset(name: PresetName) {
			layers = layersOf(name);
			kept = null;
			render(video.currentTime);
		},
		/**
		 * A click or a tap at client point (x, y): keeps what is there marked out (a second one on it lets it go). False if it did
		 * nothing: nothing there, and nothing kept to let go of.
		 */
		press(x: number, y: number): boolean {
			const r = canvas.getBoundingClientRect(),
				key = seen && pickAt(seen.elements, seen.P, [x - r.left, y - r.top]),
				had = kept;
			kept = key && key !== had ? key : null;
			render(last);
			return key !== null || had !== null;
		},
		destroy() {
			abort.abort();
			cancelAnimationFrame(queued);
			video.cancelVideoFrameCallback(frameHandle);
			resize.disconnect();
			labels.dispose();
		},
	};
}
