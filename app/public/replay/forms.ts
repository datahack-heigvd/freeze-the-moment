// Forms the gear is built from: a rounded-rectangle section swept along a path (sticks, pads, a blocker's board), and a lens (a
// rounded outline with a domed back and a hollowed front: a catching glove).
import * as THREE from "three";

// Sweep a closed rounded-rectangle section along a path: centres P[k], section axes U[k], W[k] (unit), half sizes a[k], b[k].
// Returns positions for (path samples) x SECTION vertices; faces from sweepIndex, plus a fan cap at each end (two extra vertices).
export const SECTION = 16;
const squircle = Array.from({ length: SECTION }, (_, i) => {
	const t = (i / SECTION) * 2 * Math.PI,
		c = Math.cos(t),
		s = Math.sin(t),
		e = 0.45; // superellipse: rounded rectangle
	return [Math.sign(c) * Math.abs(c) ** e, Math.sign(s) * Math.abs(s) ** e] as const;
});
type V3 = THREE.Vector3;
export function sweepInto(out: THREE.TypedArray, off: number, P: V3[], U: V3[], W: V3[], a: number[], b: number[]) {
	let o = off;
	for (let k = 0; k < P.length; k++) {
		const p = P[k]!,
			u = U[k]!,
			w = W[k]!,
			ak = a[k]!,
			bk = b[k]!;
		for (const [x, y] of squircle) {
			out[o++] = p.x + u.x * ak * x + w.x * bk * y;
			out[o++] = p.y + u.y * ak * x + w.y * bk * y;
			out[o++] = p.z + u.z * ak * x + w.z * bk * y;
		}
	}
	for (const k of [0, P.length - 1]) {
		const p = P[k]!;
		out[o++] = p.x;
		out[o++] = p.y;
		out[o++] = p.z;
	}
	return o;
}
export function sweepIndex(base: number, n: number, flipStart: boolean) {
	const idx: number[] = [];
	for (let k = 0; k < n - 1; k++)
		for (let i = 0; i < SECTION; i++) {
			const a = base + k * SECTION + i,
				b = base + k * SECTION + ((i + 1) % SECTION);
			idx.push(a, b, a + SECTION, b, b + SECTION, a + SECTION);
		}
	const c0 = base + n * SECTION,
		c1 = c0 + 1;
	for (let i = 0; i < SECTION; i++) {
		const j = (i + 1) % SECTION;
		idx.push(...(flipStart ? [c0, base + j, base + i] : [c0, base + i, base + j]));
		idx.push(c1, base + (n - 1) * SECTION + j, base + (n - 1) * SECTION + i);
	}
	return idx;
}

/**
 * A swept form as a geometry of its own: sections at centres P along axes U and W (unit), half sizes a (along U) and b (along W),
 * closed by a fan at each end; smooth normals.
 */
export function sweep(P: V3[], U: V3[], W: V3[], a: number[], b: number[]) {
	const pos = new Float32Array((P.length * SECTION + 2) * 3);
	sweepInto(pos, 0, P, U, W, a, b);
	const g = new THREE.BufferGeometry();
	g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
	g.setIndex(sweepIndex(0, P.length, true));
	g.computeVertexNormals();
	return g;
}

/**
 * A lens in the x-y plane: its outline at angle t (rad round z) `edge(t)` = [x, y] from the origin; its back (+z) domed `back` high
 * at the middle, its front (-z) hollowed `hollow` deep at the middle (as a glove's pocket), the two meeting in a rounded rim `rim`
 * thick. Rings from the middle out (nr), round (nt).
 */
export function lens(edge: (t: number) => [number, number], back: number, hollow: number, rim: number, nr = 10, nt = 40) {
	const pos: number[] = [],
		idx: number[] = [],
		row = nt + 1;
	// a face: rings r = 0..1 of the outline, at height z(r)
	const face = (z: (r: number) => number) => {
		const start = pos.length / 3;
		for (let i = 0; i <= nr; i++) {
			const r = i / nr;
			for (let j = 0; j <= nt; j++) {
				const [x, y] = edge((2 * Math.PI * j) / nt);
				pos.push(x * r, y * r, z(r));
			}
		}
		return start;
	};
	const dome = (r: number) => rim / 2 + back * Math.sqrt(Math.max(1 - r * r, 0)),
		pocket = (r: number) => -rim / 2 + hollow * (1 - r * r) * (1 - r * r);
	const b0 = face(dome),
		f0 = face(pocket);
	for (let i = 0; i < nr; i++)
		for (let j = 0; j < nt; j++) {
			const a = i * row + j,
				c = a + row;
			idx.push(b0 + a, b0 + c, b0 + a + 1, b0 + a + 1, b0 + c, b0 + c + 1);
			idx.push(f0 + a, f0 + a + 1, f0 + c, f0 + a + 1, f0 + c + 1, f0 + c);
		}
	// the rim: the outer rings of both faces joined round a half circle
	const rim0 = pos.length / 3,
		steps = 6;
	for (let k = 0; k <= steps; k++) {
		const u = Math.PI * (k / steps - 0.5);
		for (let j = 0; j <= nt; j++) {
			const [x, y] = edge((2 * Math.PI * j) / nt),
				l = Math.hypot(x, y) || 1,
				grow = (l + 0.5 * rim * Math.cos(u)) / l;
			pos.push(x * grow, y * grow, (rim / 2) * Math.sin(u));
		}
	}
	for (let k = 0; k < steps; k++)
		for (let j = 0; j < nt; j++) {
			const a = rim0 + k * row + j,
				c = a + row;
			idx.push(a, a + 1, c, a + 1, c + 1, c);
		}
	const g = new THREE.BufferGeometry();
	g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
	g.setIndex(idx);
	g.computeVertexNormals();
	return g;
}
