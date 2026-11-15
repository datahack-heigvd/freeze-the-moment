// Equipment the fitted body does not model (its limbs are the bare MHR limbs with equipment layered on): a goalie's leg pads, and
// every player's gloves, modelled on real ones and fitted to their legs and hands, coloured from their own texture where the
// broadcast saw the gear (Swatch), in synthetic leather (light.ts FINISH.pad).
//   A pad (PAD, m): WIDTH wide (the IIHF maximum), from its toe over the skate (forward of the ankle, down near the ice) up past the
// knee into the thigh rise, which narrows; its face just clear of the fitted shin (CLEAR in front of the leg's foremost vertices,
// 97th percentile), its knee stack swelling round the knee, flex breaks across it at the boot, below the knee and above it, and the
// outer roll, a round ridge along its outer edge; painted two-tone as real pads are, a base with a trim (Swatch.tones, PAINT). The
// leg's frame follows the ankle and knee rings of the MHR mesh: y up the shin, z forward, x across.
import * as THREE from "three";
import { lens, sweep } from "./forms";
import { type Gear, texel } from "./helmet";
import { FINISH, litMaterial } from "./light";
import type { Person } from "./people";
import type { Vec3 } from "./types";
import { srgbToLinear } from "./util";

const LEGS = [
	// MHR vertices: knee ring, ankle ring (front, back, out, in), shin front
	{ knee: [11521, 11634, 11575, 11562], ankle: [12230, 12193, 12222, 12185], shin: 11869 },
	{ knee: [16705, 16783, 16778, 16727], ankle: [17413, 17351, 17394, 17358], shin: 17036 },
] as const;
const PAD = {
	WIDTH: 0.28,
	THICK: 0.1, // through the face, at its thickest off the knee stack
	BELOW: 0.13, // the toe's bottom below the ankle ring's centre
	ABOVE: 0.22, // the thigh rise's top above the knee ring's centre
	TOE: 0.07, // how far the toe reaches forward over the skate
	STACK: 0.025, // the knee stack's swell
	BREAK: 0.014, // a flex break's depth
	ROLL: 0.034, // the outer roll's radius
};
const PAINT = { BAND: 0.016, STRIPE: 0.03 }; // m: a flex break's band, half its height; the inner edge's stripe, its width
const CLEAR = 0.01,
	REACH = 0.15; // m: the face off the leg; the leg's vertices within REACH of its axis (not a glove resting on it)

const smooth = (e0: number, e1: number, x: number) => {
	const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
	return t * t * (3 - 2 * t);
};
/** 1 inside a form, rounding to 0 over its last `r` toward either end (a quarter ellipse): a form's rounded ends. */
const ends = (y: number, y0: number, y1: number, r: number) =>
	Math.sqrt(Math.max(1 - (Math.max(y0 + r - y, y - (y1 - r), 0) / r) ** 2, 0.04));

/**
 * A pad in its leg's frame: its face at z = front, the knee ring's centre `len` up, its outer roll on side `outer` (+1 or -1 in x),
 * painted (vertex colours, linear) `base` with its trim in `trim`: the outer roll, a band over each flex break, a stripe down the
 * inner edge of the face (PAINT, m).
 */
function padGeometry(front: number, len: number, outer: number, base: Vec3, trim: Vec3) {
	const y0 = -PAD.BELOW,
		y1 = len + PAD.ABOVE,
		n = 48,
		V = THREE.Vector3,
		X = new V(1, 0, 0),
		Z = new V(0, 0, 1),
		breaks = [0.05, len - 0.15, len + 0.08],
		P: THREE.Vector3[] = [],
		a: number[] = [],
		b: number[] = [],
		roll: THREE.Vector3[] = [],
		rr: number[] = [];
	for (let k = 0; k < n; k++) {
		const y = y0 + ((y1 - y0) * k) / (n - 1),
			round = ends(y, y0, y1, 0.05),
			toe = PAD.TOE * smooth(0.03, -0.12, y),
			half = (PAD.WIDTH / 2) * (1 - 0.25 * smooth(0.0, -0.12, y) - 0.16 * smooth(len + 0.05, y1, y)) * round,
			depth =
				(PAD.THICK / 2 - 0.012) * round +
				PAD.STACK * Math.exp(-(((y - len) / 0.08) ** 2)) -
				breaks.reduce((s, yb) => s + PAD.BREAK * Math.exp(-(((y - yb) / 0.014) ** 2)), 0);
		P.push(new V(0, y, front - PAD.THICK / 2 + toe));
		a.push(half);
		b.push(Math.max(depth, 0.004));
		if (y > y0 + 0.06 && y < y1 - 0.05) {
			roll.push(new V(outer * (half - PAD.ROLL * 0.6), y, front - PAD.THICK / 2 + toe + 0.006));
			rr.push(PAD.ROLL * ends(y, y0 + 0.06, y1 - 0.05, 0.04));
		}
	}
	const body = sweep(
			P,
			P.map(() => X),
			P.map(() => Z),
			a,
			b,
		),
		ridge = sweep(
			roll,
			roll.map(() => X),
			roll.map(() => Z),
			rr,
			rr,
		);
	const g = merge([body, ridge]),
		pos = g.attributes.position!,
		rolled = body.attributes.position!.count,
		colour = new Float32Array(pos.count * 3);
	for (let i = 0; i < pos.count; i++) {
		const x = pos.getX(i),
			y = pos.getY(i),
			z = pos.getZ(i),
			face = z > front - PAD.THICK / 2,
			half = (PAD.WIDTH / 2) * (1 - 0.25 * smooth(0.0, -0.12, y) - 0.16 * smooth(len + 0.05, y1, y)),
			trimmed =
				i >= rolled ||
				(face && breaks.some((yb) => Math.abs(y - yb) < PAINT.BAND)) ||
				(face && -outer * x > half - PAINT.STRIPE && y > y0 + 0.05 && y < y1 - 0.04);
		colour.set(trimmed ? trim : base, 3 * i);
	}
	g.setAttribute("color", new THREE.Float32BufferAttribute(colour, 3));
	return g;
}

/** Geometries (position and normal, indexed) as one. */
function merge(parts: THREE.BufferGeometry[]) {
	const pos: number[] = [],
		nrm: number[] = [],
		idx: number[] = [];
	for (const g of parts) {
		const base = pos.length / 3;
		pos.push(...(g.attributes.position!.array as Float32Array));
		nrm.push(...(g.attributes.normal!.array as Float32Array));
		for (const i of g.index!.array) idx.push(base + i);
	}
	const g = new THREE.BufferGeometry();
	g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
	g.setAttribute("normal", new THREE.Float32BufferAttribute(nrm, 3));
	g.setIndex(idx);
	return g;
}

/**
 * A person's colours where gear covers them (their texture through the UV layout, or their vertex colours): over the body vertices
 * the gear covers, the colour at LIT of the way up by brightness, its side the arena's light falls on rather than its shade (the
 * texture holds the broadcast's colours, shading and all).
 */
const LIT = 0.8;
class Swatch {
	private uvOf = new Map<number, number>();
	constructor(
		private map: THREE.Texture | null,
		private tc: Float32Array | null,
		vertOf: Uint32Array | null,
		private colours: Uint8Array | null,
	) {
		vertOf?.forEach((v, k) => {
			if (!this.uvOf.has(v)) this.uvOf.set(v, k);
		});
	}
	/** The colour (linear) over body vertices ids. */
	of(ids: number[]): Vec3 {
		const found = this.texels(ids);
		if (!found.length) return GREY;
		found.sort((a, b) => a[0] + a[1] + a[2] - (b[0] + b[1] + b[2]));
		return found[Math.floor(LIT * (found.length - 1))]!;
	}
	/**
	 * A two-tone piece of gear's colours over body vertices ids: its base, the colour TRIM.BASE of the way up by brightness (its lit face), and its trim,
	 * the most saturated colour there (TRIM.AT of the way up by saturation) if it is a colour (TRIM.SAT), else its darks (TRIM.DARK
	 * of the way up by brightness).
	 */
	tones(ids: number[]): [Vec3, Vec3] {
		const found = this.texels(ids);
		if (!found.length) return [GREY, DARK];
		const sat = (c: Vec3) => 1 - Math.min(...c) / Math.max(...c, 1e-4),
			bright = (c: Vec3) => c[0] + c[1] + c[2],
			pick = (key: (c: Vec3) => number, at: number) =>
				[...found].sort((a, b) => key(a) - key(b))[Math.floor(at * (found.length - 1))]!,
			vivid = pick(sat, TRIM.AT);
		return [pick(bright, TRIM.BASE), sat(vivid) > TRIM.SAT ? vivid : pick(bright, TRIM.DARK)];
	}
	private texels(ids: number[]) {
		return ids.flatMap((i): Vec3[] => {
			if (this.colours) return [[0, 1, 2].map((c) => srgbToLinear(this.colours![3 * i + c]! / 255)) as Vec3];
			const u = this.uvOf.get(i);
			return u === undefined || !this.map || !this.tc ? [] : [texel(this.map, this.tc[2 * u]!, this.tc[2 * u + 1]!)];
		});
	}
}
const TRIM = { BASE: 0.92, AT: 0.9, SAT: 0.45, DARK: 0.1 },
	GREY: Vec3 = [0.6, 0.6, 0.6],
	DARK: Vec3 = [0.05, 0.05, 0.06];

export class Pads implements Gear {
	group = new THREE.Group();
	legs: ((typeof LEGS)[number] & { mesh: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial> })[];
	private swatch: Swatch;

	// a goalie's (people.ts) map/tc/vertOf: their texture and UV layout, or colours: their vertex colours (uint8, sRGB).
	constructor(map: THREE.Texture | null, tc: Float32Array | null, vertOf: Uint32Array | null, colours: Uint8Array | null) {
		this.swatch = new Swatch(map, tc, vertOf, colours);
		this.legs = LEGS.map((leg) => {
			const mesh = new THREE.Mesh(new THREE.BufferGeometry(), litMaterial(FINISH.pad, { painted: true }));
			mesh.matrixAutoUpdate = false;
			mesh.frustumCulled = false;
			this.group.add(mesh);
			return { ...leg, mesh };
		});
	}

	// Each leg's frame in the posed vertices p (m): origin at the ankle ring's centre, y up to the knee ring's, z forward (back to
	// front of the knee ring), x across. The first pose sizes the pad: its length, and its face in front of the leg's foremost vertex.
	place(p: Float32Array) {
		const at = (i: number) => new THREE.Vector3(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
		const mean = (ids: readonly number[]) =>
			ids.reduce((s, i) => s.add(at(i)), new THREE.Vector3()).multiplyScalar(1 / ids.length);
		for (const [k, leg] of this.legs.entries()) {
			const A = mean(leg.ankle),
				K = mean(leg.knee),
				y = K.clone().sub(A),
				len = y.length();
			y.normalize();
			const z = at(leg.knee[0]).sub(at(leg.knee[1]));
			z.addScaledVector(y, -z.dot(y)).normalize();
			const x = new THREE.Vector3().crossVectors(y, z);
			leg.mesh.matrix.makeBasis(x, y, z).setPosition(A);
			leg.mesh.matrixWorldNeedsUpdate = true;
			if (!leg.mesh.geometry.attributes.position) {
				const inv = leg.mesh.matrix.clone().invert(),
					h = new THREE.Vector3(),
					zs: number[] = [],
					front: number[] = [];
				for (let i = 0; i < p.length / 3; i++) {
					h.set(p[3 * i]!, p[3 * i + 1]!, p[3 * i + 2]!).applyMatrix4(inv);
					if (h.y > -PAD.BELOW && h.y < len + PAD.ABOVE && Math.hypot(h.x, h.z) < REACH) {
						zs.push(h.z);
						if (h.z > 0) front.push(i);
					}
				}
				zs.sort((a, b) => a - b);
				// the outer roll on the side away from the other leg
				const other = mean(this.legs[1 - k]!.ankle).applyMatrix4(inv);
				leg.mesh.geometry = padGeometry(
					(zs.length ? zs[Math.floor(0.97 * (zs.length - 1))]! : 0.1) + CLEAR,
					len,
					other.x > 0 ? -1 : 1,
					...this.swatch.tones(front),
				);
			}
		}
	}
}

// Gloves. A skater's (a glove on each hand) and a goalie's: a blocker on the hand that holds the stick (the hand nearer the stick's top, scene_export.py's sticks): a padded board
// (BLOCKER) over the back of the hand, its corners rounded, on a glove round the hand; a catching glove on the other (TRAPPER): a
// broad pocket over the hand and past the fingers, its outline wider at the top, its back domed, its front hollow. Each ends at the
// wrist: the forearm is the fitted body's own. Coloured from the player's texture at the hand. The hand's frame follows the MHR
// wrist ring (top, bottom, front, back) and the middle fingertip: x to the fingertip, y across, z through the hand.
export const HANDS = [
	{ wrist: [8534, 8633, 8626, 8582], tip: 10069 },
	{ wrist: [13730, 13799, 13806, 13738], tip: 15459 },
] as const;
const BLOCKER = { from: -0.06, to: 0.32, half: 0.102, thick: 0.02, back: 0.045 },
	TRAPPER = { at: 0.11, long: [0.2, 0.13] as const, wide: 0.15, back: 0.05, hollow: 0.045, rim: 0.05 },
	HAND = 0.2; // m

/** Where the stick's top sits in hand h (m): between its wrist ring's centre and its middle fingertip, in body vertices v (mm). */
export function grip(v: ArrayLike<number>, h: (typeof HANDS)[number], into = new THREE.Vector3()) {
	into.set(0, 0, 0);
	for (const i of h.wrist) into.add(new THREE.Vector3(v[3 * i], v[3 * i + 1], v[3 * i + 2]));
	into.multiplyScalar(1 / h.wrist.length);
	const tip = new THREE.Vector3(v[3 * h.tip], v[3 * h.tip + 1], v[3 * h.tip + 2]);
	return into.lerp(tip, 0.55).multiplyScalar(0.001);
}

/** A form along x from x0 to x1 in the hand's frame, half sizes hy (across) and hz (through), at z = dz, its ends rounded r. */
function along(x0: number, x1: number, hy: number, hz: number, dz: number, r: number) {
	const V = THREE.Vector3,
		P: THREE.Vector3[] = [],
		a: number[] = [],
		b: number[] = [];
	for (let k = 0; k < 24; k++) {
		const x = x0 + ((x1 - x0) * k) / 23,
			e = ends(x, x0, x1, r);
		P.push(new V(x, 0, dz));
		a.push(hy * e);
		b.push(hz * Math.max(e, 0.5));
	}
	return sweep(
		P,
		P.map(() => new V(0, 1, 0)),
		P.map(() => new V(0, 0, 1)),
		a,
		b,
	);
}

function blockerGeometry() {
	return merge([
		along(BLOCKER.from, BLOCKER.to, BLOCKER.half, BLOCKER.thick, BLOCKER.back, 0.035),
		along(-0.03, 0.15, 0.05, 0.04, 0, 0.04),
	]);
}

/** A skater's glove: rolls of padding over the back of the hand, padded fingers. */
function gloveGeometry() {
	return merge([
		...[-0.02, 0.02, 0.06].map((x0) => along(x0, x0 + 0.045, 0.05, 0.042, 0.004, 0.018)),
		along(0.09, 0.19, 0.047, 0.03, -0.004, 0.035),
	]);
}

function trapperGeometry() {
	const { long, wide } = TRAPPER,
		edge = (t: number): [number, number] => {
			const c = Math.cos(t),
				s = Math.sin(t),
				top = 1 + 0.12 * Math.exp(-((Math.atan2(s, c) / 0.5) ** 2)); // the pocket's top, wider (its T)
			return [(c > 0 ? long[0] : long[1]) * c * top, wide * s * (c > 0 ? 1 + 0.15 * c : 1)];
		};
	return lens(edge, TRAPPER.back, TRAPPER.hollow, TRAPPER.rim).translate(TRAPPER.at, 0, 0);
}

export class Gloves implements Gear {
	group = new THREE.Group();
	hands: ((typeof HANDS)[number] & { mesh: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial> })[];
	/** the hand at the stick's top (HANDS' index) */
	readonly stickHand: number;
	private swatch: Swatch | null;

	// person: a player (people.ts), a goalie wearing a blocker and a catching glove, a skater a glove on each hand; map/tc/vertOf/colours
	// as for Pads; stickTop: the stick's top per exported frame (x, y, z; NaN without one).
	constructor(
		person: Person,
		map: THREE.Texture | null,
		tc: Float32Array | null,
		vertOf: Uint32Array | null,
		colours: Uint8Array | null,
		stickTop: Float32Array,
	) {
		const hand = (f: number, h: (typeof HANDS)[number]) => {
			const v = person.at(f),
				o = h.tip * 3;
			return new THREE.Vector3(v[o], v[o + 1], v[o + 2]).multiplyScalar(0.001);
		};
		let stick = 0;
		for (const f of person.fitting) {
			const top = new THREE.Vector3(stickTop[3 * f], stickTop[3 * f + 1], stickTop[3 * f + 2]);
			if (Number.isFinite(top.x)) {
				stick = hand(f, HANDS[0]).distanceTo(top) < hand(f, HANDS[1]).distanceTo(top) ? 0 : 1;
				break;
			}
		}
		this.stickHand = stick;
		this.swatch = new Swatch(map, tc, vertOf, colours);
		this.hands = HANDS.map((h, k) => {
			const form = !person.meta.goalie ? gloveGeometry() : k === stick ? blockerGeometry() : trapperGeometry(),
				mesh = new THREE.Mesh(form, litMaterial(FINISH.pad));
			mesh.matrixAutoUpdate = false;
			mesh.frustumCulled = false;
			this.group.add(mesh);
			return { ...h, mesh };
		});
	}

	// Each hand's frame in the posed vertices p (m): origin at the wrist ring's centre, x to the fingertip, y across (bottom to top of the
	// wrist ring), z through the hand. The first pose colours each glove: the hand's vertices, past the wrist and within HAND of it.
	place(p: Float32Array) {
		const at = (i: number) => new THREE.Vector3(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
		for (const h of this.hands) {
			const o = h.wrist.reduce((s, i) => s.add(at(i)), new THREE.Vector3()).multiplyScalar(1 / h.wrist.length);
			if (this.swatch) {
				const out = at(h.tip).sub(o),
					ids: number[] = [];
				for (let i = 0; i < p.length / 3; i++) {
					const d = at(i).sub(o);
					if (d.length() < HAND && d.dot(out) > 0) ids.push(i);
				}
				h.mesh.material.uniforms.colour!.value.setRGB(...this.swatch.of(ids));
			}
			const x = at(h.tip).sub(o).normalize(),
				y = at(h.wrist[0]).sub(at(h.wrist[1]));
			y.addScaledVector(x, -y.dot(x)).normalize();
			h.mesh.matrix.makeBasis(x, y, new THREE.Vector3().crossVectors(x, y)).setPosition(o);
			h.mesh.matrixWorldNeedsUpdate = true;
		}
		this.swatch = null;
	}
}
