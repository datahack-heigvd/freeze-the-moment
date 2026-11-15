// Skates on everyone's boots. The fitted feet stand on the ice (the bodies were fitted to broadcast outlines that include the
// skates), so the blade takes the bottom of the boot's place rather than adding to the skater's height: a holder and a steel runner
// under the sole, and the boot's bottom pressed up onto the holder (everything under LIFT over the runner's edge flattened onto the
// holder's top, the rise fading out over RAMP up the leg). The foot moves near rigidly with its joint, so each blade is set once,
// in the frame where that foot stands flattest on the ice (its lowest vertices, SOLE deep, spread furthest along the ice): along
// their long axis, toward the front of the ankle, its runner's edge on the sole. Every frame it is carried rigidly with the boot:
// the frame of three parts of the foot (the back and front END of the sole and the ankle ring), so it follows the shoe, tilted,
// turned or lifted. The foot is the leg's vertices below its ankle ring (MHR vertices, as gear.ts), the leg those reached over the
// mesh from the ankle ring, told apart in the seen frame where that shin stands most upright: a glove or the other skate near the
// foot is near in space but not on the leg, and a blade sized with it would run far past the boot.
import * as THREE from "three";
import type { Gear } from "./helmet";
import { FINISH, litMaterial } from "./light";
import type { Person } from "./people";

type FootRings = { knee: readonly number[]; ankle: readonly number[] };
export const FEET: readonly FootRings[] = [
	// MHR vertices: knee ring, ankle ring (front, back, out, in)
	{ knee: [11521, 11634, 11575, 11562], ankle: [12230, 12193, 12222, 12185] },
	{ knee: [16705, 16783, 16778, 16727], ankle: [17413, 17351, 17394, 17358] },
];
const LIFT = 0.05,
	RAMP = 0.2,
	REACH = 0.35,
	BELOW = 0.02; // m; the leg's vertices: within REACH of the ankle and nearer it
// than the other ankle, in the first pose; the foot's: those
// past BELOW above the ankle ring's centre, down the shin
const SOLE = 0.015,
	END = 0.25; // m; share of the sole's length
const RUNNER = { HIGH: 0.012, THICK: 0.004, LENGTH: 0.95 }; // m, and share of the sole's length
const HOLDER = { TOP: 0.05, BOTTOM: 0.012, LENGTH: 0.85, ARCH: 0.55 }; // m across at the boot and at the runner; shares of the length

// A holder: its side profile (x along the foot, y up, from the runner's top to the sole) with the open arch between its pillars,
// narrowing from TOP across under the boot to BOTTOM at the runner. The runner: a steel strip whose toe and heel curl up.
function holderGeometry(L: number) {
	const s = new THREE.Shape(),
		top = LIFT,
		low = RUNNER.HIGH;
	s.moveTo(-L / 2 + 0.01, low);
	s.lineTo(L / 2 - 0.01, low);
	s.lineTo(L / 2, top);
	s.lineTo(-L / 2, top);
	s.closePath();
	const a = (HOLDER.ARCH * L) / 2,
		arch = new THREE.Path();
	arch.moveTo(-a, low + 0.005);
	arch.lineTo(a, low + 0.005);
	arch.quadraticCurveTo(a, top - 0.014, 0, top - 0.014);
	arch.quadraticCurveTo(-a, top - 0.014, -a, low + 0.005);
	s.holes.push(arch);
	const g = new THREE.ExtrudeGeometry(s, { depth: 1, bevelEnabled: false, curveSegments: 8 }),
		P = g.attributes.position as THREE.BufferAttribute;
	for (let k = 0; k < P.count; k++) {
		const t = (P.getY(k) - low) / (top - low);
		P.setZ(k, (P.getZ(k) - 0.5) * (HOLDER.BOTTOM + (HOLDER.TOP - HOLDER.BOTTOM) * t * t));
	}
	g.computeVertexNormals();
	return g;
}
function runnerGeometry(L: number) {
	const s = new THREE.Shape(),
		h = RUNNER.HIGH,
		r = 0.025;
	s.moveTo(-L / 2 + r, 0);
	s.lineTo(L / 2 - r, 0);
	s.quadraticCurveTo(L / 2, 0, L / 2 + 0.004, h + 0.003);
	s.lineTo(-L / 2 - 0.002, h + 0.003);
	s.quadraticCurveTo(-L / 2, 0, -L / 2 + r, 0);
	const g = new THREE.ExtrudeGeometry(s, { depth: RUNNER.THICK, bevelEnabled: false, curveSegments: 6 });
	g.translate(0, 0, -RUNNER.THICK / 2);
	return g;
}
const holderMaterial = litMaterial(FINISH.holder, { colour: [0.06, 0.06, 0.07] }),
	runnerMaterial = litMaterial(FINISH.steel, { colour: [0.3, 0.31, 0.33] });

// Centroid of vertices idx in positions p (m).
const centroid = (p: ArrayLike<number>, idx: readonly number[]) => {
	const c = new THREE.Vector3();
	for (const i of idx) c.set(c.x + p[3 * i]!, c.y + p[3 * i + 1]!, c.z + p[3 * i + 2]!);
	return c.multiplyScalar(1 / idx.length);
};

type Parts = [number[], number[], readonly number[]];

/** Each vertex's neighbours over the mesh's edges: those of vertex i are at[start[i]] up to at[start[i + 1]]. */
function neighbours(faces: Uint32Array, V: number) {
	const start = new Uint32Array(V + 1),
		at = new Uint32Array(faces.length * 2);
	for (const v of faces) start[v + 1]! += 2;
	for (let i = 0; i < V; i++) start[i + 1]! += start[i]!;
	const next = start.slice(0, V);
	for (let t = 0; t < faces.length; t += 3)
		for (let e = 0; e < 3; e++) {
			const a = faces[t + e]!,
				b = faces[t + ((e + 1) % 3)]!;
			at[next[a]!++] = b;
			at[next[b]!++] = a;
		}
	return { start, at };
}
type Foot = { leg: number[]; parts: Parts; blade: THREE.Group; rel: THREE.Matrix4 };

export class Skates implements Gear {
	person: Person;
	group = new THREE.Group();
	feet: (Foot | null)[] | null = null;
	private mesh?: ReturnType<typeof neighbours>;

	// person: a Person (people.ts), all of whose exported frames are searched for each foot's most upright shin and flattest stance.
	constructor(person: Person) {
		this.person = person;
	}

	// Positions (m) of the seen frame (of those the gear is fitted on) in which foot's shin is nearest vertical, the knee ring
	// straight above the ankle ring (any frame when none is seen).
	upright(foot: FootRings) {
		const { meta } = this.person,
			frames = this.person.fitting,
			any = frames.some((f) => meta.seen[f]);
		let best = frames[0]!,
			most = -Infinity;
		for (const f of frames) {
			if (any && !meta.seen[f]) continue;
			const at = this.person.at(f),
				d = centroid(at, foot.knee).sub(centroid(at, foot.ankle)),
				up = d.z / (d.length() || 1);
			if (up > most) {
				most = up;
				best = f;
			}
		}
		return Float32Array.from(this.person.at(best), (v) => v / 1000);
	}

	// A foot's leg and sole vertices in pose p (other: the other foot), its blade set in its flattest stance, and the
	// blade's pose relative to the foot's frame there; null when no frame shows the sole on the ice.
	foot(p: Float32Array, foot: FootRings, other: FootRings): Foot | null {
		const A = centroid(p, foot.ankle),
			B = centroid(p, other.ankle),
			down = A.clone().sub(centroid(p, foot.knee)).normalize(),
			leg: number[] = [...foot.ankle],
			sole: number[] = [],
			q = new THREE.Vector3(),
			{ start, at } = (this.mesh ??= neighbours(this.person.faces, this.person.V)),
			reached = new Uint8Array(this.person.V);
		for (const i of leg) reached[i] = 1;
		for (let h = 0; h < leg.length; h++) {
			// out over the mesh from the ankle ring
			const i = leg[h]!;
			if (
				q
					.set(p[3 * i]!, p[3 * i + 1]!, p[3 * i + 2]!)
					.sub(A)
					.dot(down) > -BELOW
			)
				sole.push(i);
			for (let k = start[i]!; k < start[i + 1]!; k++) {
				const j = at[k]!;
				if (reached[j]) continue;
				reached[j] = 1;
				q.set(p[3 * j]!, p[3 * j + 1]!, p[3 * j + 2]!);
				const d = q.distanceTo(A);
				if (d < REACH && d < q.distanceTo(B)) leg.push(j);
			}
		}
		const V = this.person.V;
		let best: { s: Float32Array; low: number; heading: THREE.Vector3; spread: number } | null = null;
		for (const f of this.person.fitting) {
			// the stance: the sole's lowest vertices spread furthest along one line
			const s = new Float32Array(V * 3),
				v = this.person.at(f);
			for (const i of [...sole, ...foot.ankle]) for (let c = 0; c < 3; c++) s[3 * i + c] = v[i * 3 + c]! / 1000;
			let low = Infinity;
			for (const i of sole) low = Math.min(low, s[3 * i + 2]!);
			const bottom = sole.filter((i) => s[3 * i + 2]! < low + SOLE);
			if (bottom.length < 10) continue;
			const m = centroid(s, bottom);
			let xx = 0,
				xy = 0,
				yy = 0;
			for (const i of bottom) {
				const dx = s[3 * i]! - m.x,
					dy = s[3 * i + 1]! - m.y;
				xx += dx * dx;
				xy += dx * dy;
				yy += dy * dy;
			}
			const t = 0.5 * Math.atan2(2 * xy, xx - yy),
				u = [Math.cos(t), Math.sin(t)] as const;
			let lo = Infinity,
				hi = -Infinity;
			for (const i of bottom) {
				const a = (s[3 * i]! - m.x) * u[0] + (s[3 * i + 1]! - m.y) * u[1];
				lo = Math.min(lo, a);
				hi = Math.max(hi, a);
			}
			if (!best || hi - lo > best.spread) best = { s, low, heading: new THREE.Vector3(u[0], u[1], 0), spread: hi - lo };
		}
		if (!best) return null;
		const { s, low, heading: x } = best,
			front = centroid(s, [foot.ankle[0]!]).sub(centroid(s, [foot.ankle[1]!]));
		if (x.dot(front) < 0) x.negate();
		let back = Infinity,
			ahead = -Infinity;
		const m = centroid(s, sole);
		for (const i of sole) {
			const a = (s[3 * i]! - m.x) * x.x + (s[3 * i + 1]! - m.y) * x.y;
			back = Math.min(back, a);
			ahead = Math.max(ahead, a);
		}
		const L = ahead - back,
			along = (i: number) => (s[3 * i]! - m.x) * x.x + (s[3 * i + 1]! - m.y) * x.y;
		const parts: Parts = [
			sole.filter((i) => along(i) < back + END * L),
			sole.filter((i) => along(i) > ahead - END * L),
			foot.ankle,
		];
		const up = new THREE.Vector3(0, 0, 1),
			z = new THREE.Vector3().crossVectors(x, up);
		const set = new THREE.Matrix4()
			.makeBasis(x, up, z)
			.setPosition(m.x + (x.x * (back + ahead)) / 2, m.y + (x.y * (back + ahead)) / 2, low);
		const blade = new THREE.Group();
		blade.matrixAutoUpdate = false;
		blade.add(
			new THREE.Mesh(holderGeometry(HOLDER.LENGTH * L), holderMaterial),
			new THREE.Mesh(runnerGeometry(RUNNER.LENGTH * L), runnerMaterial),
		);
		blade.traverse((o) => {
			o.frustumCulled = false;
		});
		this.group.add(blade);
		return { leg, parts, blade, rel: this.frame(s, parts).invert().multiply(set) };
	}

	// The foot's frame in positions p (m): origin at the back of the sole, x to its front, y toward the ankle, rigid.
	frame(p: ArrayLike<number>, [heel, toe, ankle]: Parts) {
		const o = centroid(p, heel),
			x = centroid(p, toe).sub(o).normalize(),
			y = centroid(p, ankle).sub(o);
		y.addScaledVector(x, -y.dot(x)).normalize();
		return new THREE.Matrix4().makeBasis(x, y, new THREE.Vector3().crossVectors(x, y)).setPosition(o);
	}

	// The blades carried with the boots, and the boots' bottoms pressed onto them.
	place(p: Float32Array) {
		if (!this.feet) this.feet = FEET.map((foot, j) => this.foot(this.upright(foot), foot, FEET[1 - j]!));
		const step = (e0: number, e1: number, v: number) => {
			const t = Math.min(Math.max((v - e0) / (e1 - e0), 0), 1);
			return t * t * (3 - 2 * t);
		};
		const o = new THREE.Vector3(),
			y = new THREE.Vector3();
		for (const f of this.feet) {
			if (!f) continue;
			f.blade.matrix.multiplyMatrices(this.frame(p, f.parts), f.rel);
			f.blade.matrixWorldNeedsUpdate = true;
			o.setFromMatrixPosition(f.blade.matrix);
			y.setFromMatrixColumn(f.blade.matrix, 1);
			for (const i of f.leg) {
				// height over the runner's edge: up to the holder's top, eased above it
				const h = (p[3 * i]! - o.x) * y.x + (p[3 * i + 1]! - o.y) * y.y + (p[3 * i + 2]! - o.z) * y.z;
				const lift = Math.max(LIFT - h, LIFT * (1 - step(0, RAMP, h)));
				p[3 * i]! += y.x * lift;
				p[3 * i + 1]! += y.y * lift;
				p[3 * i + 2]! += y.z * lift;
			}
		}
	}
}
