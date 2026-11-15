// A helmet on each skater's and official's head. The fitted head already holds the helmet's volume (the bodies were fitted to the
// broadcast silhouettes, helmets on), so the shell is that head, made smooth: the head's vertices above the helmet's rim (the brow in
// front, below the ears at the sides, the nape at the back) in the head's frame, their distance from CENTRE fitted as a polynomial of
// the direction (to DEGREE; the fit's own surface is crumpled), lifted to clear nearly all of them (CLEAR) and closed by a lip of
// THICK at the open edge; a skater's shaped closer low round the sides and fuller at the back (SHAPE), the head drawn in under it
// wherever it would reach it (tuck). The head is the same in every frame (scene_export.py steady_heads: each person's own head,
// steadied and its face eased in behind the face protection), so the shell built from the first holds it throughout. Glossy, in the colour the
// person's texture holds at the crown (the team's helmet colour in the footage: gear_colour.py). Skaters wear face protection in
// front of it: a clear half visor, or with ?guard=cage a wire cage (whether the U.S. college players wore cages here is not in the
// footage). A goalie's shell is a mask: down over the face to under the chin (gear.py), standing MASK further off the face, with an
// opening for the eyes and nose (EYES), the face in shadow behind a wire cage standing off it; painted with the pattern its texture
// holds (paint), trimmed dark round its edges and backplate seam (trim), in satin (light.ts FINISH). Shell and face protection
// follow the head's frame, from the crown and both
// ears.
import * as THREE from "three";
import { FINISH, litMaterial } from "./light";
import type { Person } from "./people";
import type { Vec3 } from "./types";
import { param, srgbToLinear } from "./util";

const CROWN = 2810,
	LEFT = 256,
	RIGHT = 3142,
	EARS = 0.25; // MHR vertices; ear-to-ear over the helmet as exported (m)
const CENTRE: Vec3 = [0, 0.02, 0.01],
	DEGREE = [4, 4],
	REACH = 0.17,
	OUTLIER = 0.015; // m: head vertices within REACH of the frame's origin;
// refitted without those OUTLIER off the first fit (neck, shoulders)
const CLEAR = [0.9, 0.003] as const,
	THICK = 0.008;
const MASK = 0.006,
	EYES = { side: 0.6, top: 0.05, bottom: -0.06 }; // m; the mask's opening: rad either side of forward, m above the ears
const GUARD = param("guard") === "cage" ? "cage" : "visor";
// Head frame (m): x to the left ear, y up through the crown, z forward; origin between the ears. On the bare head (MHR rest mesh) the
// nose tip is at (0, 0, 0.144), the brow near y 0.04, the chin near y -0.11; the export adds 0.6 cm over the face and 2.8 cm of
// helmet over the cranium (scene_export.py helmet_gear).

// Lowest point of the shell at azimuth a (0 forward), as gear.py: the brow in front, below the ears at the sides, the nape at the
// back; a goalie's mask (mask) under the chin in front.
const smooth = (e0: number, e1: number, x: number) => {
	const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
	return t * t * (3 - 2 * t);
};
const rim = (a: number, mask = false) => {
	const back = smooth(2.0, 2.8, Math.abs(a));
	if (mask) {
		const front = smooth(1.3, 0.8, Math.abs(a));
		return (-0.07 * (1 - back) - 0.065 * back) * (1 - front) - 0.12 * front;
	}
	const front = smooth(1.1, 0.6, Math.abs(a));
	return (-0.035 * (1 - back) - 0.065 * back) * (1 - front) + 0.035 * front;
};

// A skater's shell's own shape over the head it is fitted to (m along the direction d from CENTRE): closer low round the sides, and
// standing off the back of the head. Over the brow it keeps the head's own fit: the face below the rim is not drawn in, so a shell
// brought closer there would show the brow through it.
// The head is drawn in (tucks) wherever it would reach the shell, so none of it shows through.
const SHAPE = { low: 0.006, back: 0.012, edge: 0.02 } as const, // m; edge: below the rim the head is drawn in over
	INSIDE = 0.002; // m: how far inside the shell's inner face the head is drawn in to
const shape = (d: Vec3) => SHAPE.back * smooth(0, -1, d[2]) - SHAPE.low * smooth(0.8, 0, d[1]) * smooth(-1, 0, d[2]);

type Radius = (d: Vec3) => number;

// The least-squares fit of r over the monomials of the direction d up to degree n (ridge-regularised normal equations).
function fit(dirs: Vec3[], r: number[], degree: number): Radius {
	const POWERS: Vec3[] = [];
	for (let a = 0; a <= degree; a++)
		for (let b = 0; a + b <= degree; b++) for (let c = 0; a + b + c <= degree; c++) POWERS.push([a, b, c]);
	const basis = (d: Vec3) => POWERS.map(([a, b, c]) => d[0] ** a * d[1] ** b * d[2] ** c);
	const n = POWERS.length,
		M = Array.from({ length: n }, () => new Float64Array(n + 1));
	dirs.forEach((d, k) => {
		const f = basis(d);
		for (let i = 0; i < n; i++) {
			const row = M[i]!;
			for (let j = 0; j < n; j++) row[j]! += f[i]! * f[j]!;
			row[n]! += f[i]! * r[k]!;
		}
	});
	for (let i = 0; i < n; i++) M[i]![i]! += 1e-6 * dirs.length;
	for (let i = 0; i < n; i++) {
		// Gauss-Jordan with partial pivoting
		let q = i;
		for (let k = i + 1; k < n; k++) if (Math.abs(M[k]![i]!) > Math.abs(M[q]![i]!)) q = k;
		[M[i], M[q]] = [M[q]!, M[i]!];
		const Mi = M[i]!;
		for (let k = 0; k < n; k++)
			if (k !== i) {
				const Mk = M[k]!,
					f = Mk[i]! / Mi[i]!;
				for (let j = i; j <= n; j++) Mk[j]! -= f * Mi[j]!;
			}
	}
	const w = M.map((row, i) => row[n]! / row[i]!),
		flat = Int8Array.from(POWERS.flat()),
		x = new Float64Array(degree + 1),
		y = new Float64Array(degree + 1),
		z = new Float64Array(degree + 1);
	x[0] = y[0] = z[0] = 1;
	return (d) => {
		// the powers of each component, then the weighted monomials
		for (let k = 1; k <= degree; k++) {
			x[k] = x[k - 1]! * d[0];
			y[k] = y[k - 1]! * d[1];
			z[k] = z[k - 1]! * d[2];
		}
		let s = 0;
		for (let i = 0; i < n; i++) s += w[i]! * x[flat[3 * i]!]! * y[flat[3 * i + 1]!]! * z[flat[3 * i + 2]!]!;
		return s;
	};
}

// Points of the surface at radius R(direction from CENTRE) + lift (and a skater's SHAPE, shaped): at azimuth a and polar angle t
// (from the top), and the polar angle where it comes down to height y.
type Point = (a: number, t: number) => Vec3;
const surface =
	(R: Radius, lift: number, shaped = false): Point =>
	(a, t) => {
		const d: Vec3 = [Math.sin(t) * Math.sin(a), Math.cos(t), Math.sin(t) * Math.cos(a)],
			r = R(d) + lift + (shaped ? shape(d) : 0);
		return d.map((x, i) => CENTRE[i]! + r * x) as Vec3;
	};
const down = (point: Point, a: number, y: number) => {
	let t = 0;
	while (t < Math.PI && point(a, t)[1] > y) t += 0.01;
	return t;
};
const inEyes = (p: ArrayLike<number>) =>
	Math.abs(Math.atan2(p[0]!, p[2]!)) < EYES.side && p[1]! < EYES.top && p[1]! > EYES.bottom && p[2]! > 0;

const OUT = 0.01; // m, under the cage's CAGE

// The shell: rows from the top down to the rim and a lip turned in at the edge; a mask without the triangles in front of the eyes.
function shellGeometry(R: Radius, lift: number, mask: boolean, na = 64, nv = 20) {
	const point = surface(R, lift, !mask),
		edge = new Map<number, number>(); // the rim's polar angle, by azimuth: once a column
	const g = grid(
		(i, j) => {
			const a = -Math.PI + 2 * Math.PI * i;
			if (!edge.has(a)) edge.set(a, down(point, a, rim(a, mask)));
			const p = point(a, edge.get(a)! * Math.min((j * (nv + 1)) / nv, 1));
			if (j < 1) return p;
			const d = p.map((x, k) => x - CENTRE[k]!),
				l = Math.hypot(...d);
			return p.map((x, k) => x - (d[k]! / l) * THICK) as Vec3;
		},
		na,
		nv + 1,
	);
	if (mask) {
		// the opening: closed by the face in shadow behind the cage (trim)
		const P = g.attributes.position!.array as Float32Array,
			eyes: number[] = [];
		g.userData.eyes = eyes;
		for (let v = 0; v < P.length / 3; v++) {
			// stood OUT off the shell, over the face's tip
			const q = P.subarray(3 * v, 3 * v + 3);
			if (!inEyes(q)) continue;
			const d = [...q].map((x, k) => x - CENTRE[k]!),
				l = Math.hypot(...d);
			for (let k = 0; k < 3; k++) q[k]! += (d[k]! / l) * OUT;
			eyes.push(v);
		}
		g.computeVertexNormals();
	}
	return g;
}

// A goalie mask's dark trim, darkening its per-vertex colours (head frame, m): the seam of the backplate behind the ears (SEAM rad
// from the front, SEAM_HALF wide), the rubber round the eye opening and along the bottom edge (TRIM wide).
const SEAM = 1.75,
	SEAM_HALF = 0.04,
	TRIM = 0.012,
	DARK = 0.15,
	SEAM_DARK = 0.5,
	SHADOW = 0.04;
function trim(g: THREE.BufferGeometry) {
	const P = g.attributes.position as THREE.BufferAttribute,
		C = (g.attributes.color as THREE.BufferAttribute).array as Float32Array,
		q = new THREE.Vector3();
	for (let k = 0; k < P.count; k++) {
		q.fromBufferAttribute(P, k);
		const a = Math.atan2(q.x, q.z),
			r = Math.hypot(q.x, q.z);
		const eye = Math.max(Math.abs(a) * r - EYES.side * r, q.y - EYES.top, EYES.bottom - q.y); // m outside the opening (< 0 inside)
		const dark =
			(q.z > 0 && eye < TRIM) || q.y < rim(a, true) + TRIM ? DARK : Math.abs(Math.abs(a) - SEAM) < SEAM_HALF ? SEAM_DARK : 1;
		for (let c = 0; c < 3; c++) C[3 * k + c]! *= dark;
	}
	for (const k of g.userData.eyes as number[]) for (let c = 0; c < 3; c++) C[3 * k + c] = SHADOW * C[3 * k + c]!;
}

// A goalie's cage over the mask's opening: vertical bars and horizontal ones, bulging out of the mask's surface up to CAGE in the
// middle of the opening (a goalie's cage stands off the face).
const CAGE = 0.045;
function maskCageGeometry(R: Radius, lift: number) {
	const point = surface(R, lift + 0.004),
		lines: Vec3[][] = [],
		n = 12;
	const bulge = (a: number, y: number) =>
		CAGE * Math.cos(((Math.PI / 2) * a) / EYES.side) * Math.sin((Math.PI * (y - EYES.bottom)) / (EYES.top - EYES.bottom));
	const out = (a: number, y: number) => {
		const p = point(a, down(point, a, y)),
			d = p.map((x, k) => x - CENTRE[k]!),
			l = Math.hypot(...d),
			b = bulge(a, y);
		return p.map((x, k) => x + (d[k]! / l) * b) as Vec3;
	};
	const along = (a0: number, a1: number, y0: number, y1: number) =>
		Array.from({ length: n + 1 }, (_, k) => out(a0 + ((a1 - a0) * k) / n, y0 + ((y1 - y0) * k) / n));
	for (let a = -EYES.side; a <= EYES.side + 1e-6; a += EYES.side / 4) lines.push(along(a, a, EYES.top, EYES.bottom));
	for (const y of [EYES.top, 0.02, -0.01, -0.035, EYES.bottom]) lines.push(along(-EYES.side, EYES.side, y, y));
	return wires(lines, 0.0022);
}

// The face protection's surface: an elliptic band round the face from ear to ear (about the axis z = AXIS), drawn in toward the chin.
// Each person's is widened to clear their own face by CLEARANCE (ellipse-relative) wherever the face comes nearer, and across or
// forward to stand VISOR_OFF outside the shell where they meet (mounted on it at the temples, its top over the brow).
const AXIS = 0.03,
	CLEARANCE = 0.08,
	VISOR_OFF = 0.03; // ellipse-relative: off the shell's outside
const band = (y: number) => {
	const t = smooth(-0.03, -0.12, y);
	return [0.098 - 0.025 * t, 0.132 - 0.022 * t] as const;
};
const face = (a: number, y: number): Vec3 => {
	const [A, B] = band(y);
	return [A * Math.sin(a), y, AXIS + B * Math.cos(a)];
};

function grid(f: (u: number, v: number) => Vec3, na: number, nv: number) {
	// f(i / na, j / nv) -> point: a surface of na x nv quads
	const pos: number[] = [],
		idx: number[] = [];
	for (let i = 0; i <= na; i++) for (let j = 0; j <= nv; j++) pos.push(...f(i / na, j / nv));
	for (let i = 0; i < na; i++)
		for (let j = 0; j < nv; j++) {
			const p = i * (nv + 1) + j,
				q = p + nv + 1;
			idx.push(p, p + 1, q, q, p + 1, q + 1);
		}
	const g = new THREE.BufferGeometry();
	g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
	g.setIndex(idx);
	g.computeVertexNormals();
	return g;
}

function visorGeometry() {
	return grid((u, v) => face(-1.3 + 2.6 * u, 0.04 - 0.075 * v), 24, 4);
}

// Wires along polylines, each a square tube of radius r, in one geometry.
function wires(lines: Vec3[][], r: number) {
	const pos: number[] = [],
		idx: number[] = [];
	for (const pts of lines) {
		const P = pts.map((p) => new THREE.Vector3(...p)),
			base = pos.length / 3;
		for (let k = 0; k < P.length; k++) {
			const d = P[Math.min(k + 1, P.length - 1)]!.clone()
				.sub(P[Math.max(k - 1, 0)]!)
				.normalize();
			const u = new THREE.Vector3()
				.crossVectors(d, Math.abs(d.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0))
				.normalize();
			const w = new THREE.Vector3().crossVectors(d, u);
			for (const [s, t] of [
				[1, 0],
				[0, 1],
				[-1, 0],
				[0, -1],
			] as const)
				pos.push(
					...P[k]!.clone()
						.addScaledVector(u, s * r)
						.addScaledVector(w, t * r)
						.toArray(),
				);
		}
		for (let k = 0; k + 1 < P.length; k++)
			for (let s = 0; s < 4; s++) {
				const a = base + k * 4 + s,
					b = base + k * 4 + ((s + 1) % 4);
				idx.push(a, b, a + 4, a + 4, b, b + 4);
			}
	}
	const g = new THREE.BufferGeometry();
	g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
	g.setIndex(idx);
	g.computeVertexNormals();
	return g;
}

// A cage: vertical bars across the face, horizontal ones from the brow to the chin, a frame round the side.
function cageGeometry() {
	const line = (n: number, f: (t: number) => Vec3) => Array.from({ length: n + 1 }, (_, k) => f(k / n)),
		lines: Vec3[][] = [],
		SIDE = 1.25;
	for (let a = -0.9; a <= 0.91; a += 0.18) lines.push(line(8, (t) => face(a, 0.035 - 0.15 * t)));
	for (const y of [0.035, 0.005, -0.03, -0.065, -0.1]) {
		const w = y < -0.08 ? 0.75 : SIDE;
		lines.push(line(16, (t) => face(-w + 2 * w * t, y)));
	}
	for (const s of [-1, 1]) lines.push(line(6, (t) => face(s * (SIDE - 0.5 * smooth(0.5, 1, t)), 0.035 - 0.135 * t)));
	lines.push(line(10, (t) => face(-0.75 + 1.5 * t, -0.115)));
	return wires(lines, 0.0022);
}

const SHIELD = { visor: visorGeometry, cage: cageGeometry }[GUARD]();
const vertex = /* glsl */ `
varying vec3 vN, vP, vC;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0); vP = w.xyz; vN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * viewMatrix * w;
#ifdef PAINTED
  vC = color;
#endif
}`;

// The colour (linear) of a texture at texcoord (u, v), read from its image once (row 0 at v = 0, as util.texture loads it).
const pixels = new WeakMap<THREE.Texture, { d: Uint8ClampedArray; w: number; h: number }>();
export function texel(map: THREE.Texture, u: number, v: number): Vec3 {
	let px = pixels.get(map);
	if (!px) {
		const im = map.image as ImageBitmap,
			c = document.createElement("canvas");
		c.width = im.width;
		c.height = im.height;
		const g = c.getContext("2d")!;
		g.drawImage(im, 0, 0);
		px = { d: g.getImageData(0, 0, c.width, c.height).data, w: c.width, h: c.height };
		pixels.set(map, px);
	}
	const i = (Math.min(px.h - 1, Math.floor(v * px.h)) * px.w + Math.min(px.w - 1, Math.floor(u * px.w))) * 4;
	return [0, 1, 2].map((k) => srgbToLinear(px.d[i + k]! / 255)) as Vec3;
}

// Per-vertex colours for geometry g (in the frame of matrix m) from the nearest of the person's posed vertices p (m), through their
// texture: a painted piece of equipment carries the pattern the broadcast saw there.
export function paint(
	g: THREE.BufferGeometry,
	m: THREE.Matrix4,
	p: Float32Array,
	near: number[],
	map: THREE.Texture,
	tc: Float32Array,
	vertOf: Uint32Array,
) {
	const uvOf = new Map<number, number>();
	vertOf.forEach((v, k) => {
		if (!uvOf.has(v)) uvOf.set(v, k);
	});
	const P = g.attributes.position as THREE.BufferAttribute,
		q = new THREE.Vector3(),
		col = new Float32Array(P.count * 3);
	const cand = near.map((i) => [i, new THREE.Vector3(p[3 * i], p[3 * i + 1], p[3 * i + 2])] as const);
	for (let k = 0; k < P.count; k++) {
		q.fromBufferAttribute(P, k).applyMatrix4(m);
		let best = cand[0]!,
			d = Infinity;
		for (const c of cand) {
			const e = c[1].distanceToSquared(q);
			if (e < d) {
				d = e;
				best = c;
			}
		}
		const u = uvOf.get(best[0])!;
		col.set(texel(map, tc[2 * u]!, tc[2 * u + 1]!), 3 * k);
	}
	g.setAttribute("color", new THREE.BufferAttribute(col, 3));
}

const visorMaterial = new THREE.ShaderMaterial({
	transparent: true,
	depthWrite: false,
	side: THREE.DoubleSide,
	vertexShader: vertex,
	fragmentShader: /* glsl */ `
varying vec3 vN, vP;
void main() {
  vec3 n = normalize(vN), v = normalize(cameraPosition - vP); float c = abs(dot(n, v)), fres = 0.04 + 0.96 * pow(1.0 - c, 5.0);
  gl_FragColor = vec4(mix(vec3(0.55, 0.6, 0.65), vec3(0.9, 0.93, 0.95), fres), 0.1 + 0.6 * fres);
  #include <colorspace_fragment>
}`,
});
const cageMaterial = litMaterial(FINISH.wire, { colour: [0.06, 0.06, 0.07] });

/** A piece of equipment that follows a person's posed vertices (people.ts gear). */
export interface Gear {
	group: THREE.Group;
	place(p: Float32Array): void;
}

export class Helmet implements Gear {
	person: Person;
	group = new THREE.Group();
	mask: boolean;
	painted: { map: THREE.Texture; tc: Float32Array; vertOf: Uint32Array } | null;
	colour: number[];
	shell: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>;
	guard?: THREE.Mesh;
	under?: number[];
	R?: Radius;
	lift = 0;

	// person: a Person (people.ts); map/tc/vertOf its texture and UV layout, or colours its vertex colours (uint8, sRGB).
	constructor(
		person: Person,
		map: THREE.Texture | null,
		tc: Float32Array | null,
		vertOf: Uint32Array | null,
		colours: Uint8Array | null,
	) {
		const u = vertOf ? vertOf.indexOf(CROWN) : -1;
		const uv = u >= 0 && tc ? [tc[2 * u]!, tc[2 * u + 1]!] : [0, 0],
			c = colours ? [...colours.subarray(CROWN * 3, CROWN * 3 + 3)].map((x) => x / 255) : [0.3, 0.3, 0.33];
		this.person = person;
		this.group.matrixAutoUpdate = false;
		this.mask = !!person.meta.goalie;
		this.painted = this.mask && u >= 0 && map && tc && vertOf ? { map, tc, vertOf } : null;
		this.colour = c;
		this.shell = new THREE.Mesh(
			undefined,
			litMaterial(this.mask ? FINISH.mask : FINISH.shell, { map: u >= 0 ? map : null, uv, colour: c, painted: this.mask }),
		);
		this.group.add(this.shell);
		if (person.meta.team >= 0 && !this.mask) {
			this.guard = new THREE.Mesh(SHIELD, GUARD === "cage" ? cageMaterial : visorMaterial);
			this.guard.matrixAutoUpdate = false;
			if (GUARD === "visor") this.guard.renderOrder = 3;
			this.group.add(this.guard);
		}
		this.group.traverse((o) => {
			o.frustumCulled = false;
		});
	}

	// The shell from the head of the first pose (frame: the head's frame), and the head vertices under it.
	build(p: Float32Array, frame: THREE.Matrix4) {
		const inv = frame.clone().invert(),
			h = new THREE.Vector3(),
			dirs: Vec3[] = [],
			r: number[] = [],
			under: number[] = [];
		this.under = under;
		for (let i = 0; i < p.length / 3; i++) {
			h.set(p[3 * i]!, p[3 * i + 1]!, p[3 * i + 2]!).applyMatrix4(inv);
			const a = Math.atan2(h.x, h.z),
				lo = rim(a, this.mask);
			if (h.length() > REACH || h.y < lo) continue;
			const d = h.toArray().map((x, k) => x - CENTRE[k]!),
				l = Math.hypot(...d);
			dirs.push(d.map((x) => x / l) as Vec3);
			r.push(l);
			under.push(i);
		}
		const degree = DEGREE[+this.mask]!,
			R0 = fit(dirs, r, degree),
			near = r.map((l, k) => Math.abs(l - R0(dirs[k]!)) < OUTLIER);
		const nearDirs = dirs.filter((_, k) => near[k]),
			nearR = r.filter((_, k) => near[k]);
		const R = fit(nearDirs, nearR, degree),
			out = nearR.map((l, k) => l - R(nearDirs[k]!)).sort((a, b) => a - b);
		this.lift = Math.max(out[Math.floor(CLEAR[0] * (out.length - 1))] ?? 0, 0) + CLEAR[1] + (this.mask ? MASK : 0);
		this.shell.geometry = shellGeometry(R, this.lift, this.mask);
		this.R = R;
		if (this.guard) {
			// widened (kx across, kz forward) to clear the face, at its widest, and to stand outside the shell where they meet
			const reach: number[] = [];
			for (let i = 0; i < p.length / 3; i++) {
				h.set(p[3 * i]!, p[3 * i + 1]!, p[3 * i + 2]!).applyMatrix4(inv);
				if (h.length() > REACH || h.y < -0.13 || h.y > rim(Math.atan2(h.x, h.z)) || h.z < AXIS) continue;
				const [A, B] = band(h.y);
				reach.push(Math.hypot(h.x / A, (h.z - AXIS) / B));
			}
			reach.sort((a, b) => a - b);
			let kx = Math.max(1, (reach[Math.floor(0.99 * (reach.length - 1))] ?? 0) + CLEARANCE),
				kz = kx;
			const point = surface(R, this.lift + THICK, true),
				shell: Vec3[] = [];
			for (let a = -1.3; a <= 1.3; a += 0.1)
				for (let y = -0.035; y <= 0.04; y += 0.015) if (y >= rim(a)) shell.push(point(a, down(point, a, y)));
			for (let pass = 0; pass < 20; pass++)
				for (const [x, y, z] of shell) {
					const [A, B] = band(y),
						u = Math.abs(x) / (A * kx),
						w = (z - AXIS) / (B * kz),
						e = Math.hypot(u, w) * (1 + VISOR_OFF);
					if (e <= 1) continue;
					if (u > w) kx *= e;
					else kz *= e;
				}
			this.guard.matrix
				.makeTranslation(0, 0, AXIS)
				.multiply(new THREE.Matrix4().makeScale(kx, 1, kz))
				.multiply(new THREE.Matrix4().makeTranslation(0, 0, -AXIS));
		}
		if (!this.mask) {
			// the head drawn in to INSIDE under the shell wherever it would reach it, fading out over SHAPE.edge below its rim
			const tuck = new Float32Array(p.length / 3);
			for (let i = 0; i < tuck.length; i++) {
				h.set(p[3 * i]!, p[3 * i + 1]!, p[3 * i + 2]!).applyMatrix4(inv);
				const lo = rim(Math.atan2(h.x, h.z));
				if (h.length() > REACH || h.y < lo - SHAPE.edge) continue;
				const d = h.toArray().map((x, k) => x - CENTRE[k]!),
					l = Math.hypot(...d),
					n = d.map((x) => x / l) as Vec3,
					inside = R(n) + this.lift + shape(n) - THICK - INSIDE;
				tuck[i] = Math.max(l - inside, 0) * smooth(lo - SHAPE.edge, lo, h.y);
			}
			this.person.tuck(tuck, (into) => into.set(...CENTRE).applyMatrix4(this.group.matrix));
		}
		if (this.mask) {
			const g = this.shell.geometry;
			if (this.painted) {
				const { map, tc, vertOf } = this.painted;
				paint(g, frame, p, under, map, tc, vertOf);
			} else
				g.setAttribute(
					"color",
					new THREE.BufferAttribute(
						new Float32Array(g.attributes.position!.count * 3).map((_, k) => srgbToLinear(this.colour[k % 3]!)),
						3,
					),
				);
			trim(g);
			this.guard = new THREE.Mesh(maskCageGeometry(R, this.lift), cageMaterial);
			this.guard.frustumCulled = false;
			this.group.add(this.guard);
		}
	}

	// Follows the head's frame in the posed vertices p (m): origin between the ears, x to the left ear, y up through the crown,
	// z forward, scaled by the ears' distance.
	place(p: Float32Array) {
		const at = (i: number) => new THREE.Vector3(p[3 * i], p[3 * i + 1], p[3 * i + 2]),
			L = at(LEFT),
			R = at(RIGHT);
		const o = L.clone().add(R).multiplyScalar(0.5),
			x = L.clone().sub(R),
			s = x.length() / EARS;
		x.normalize();
		const y = at(CROWN).sub(o);
		y.addScaledVector(x, -y.dot(x)).normalize();
		const z = new THREE.Vector3().crossVectors(x, y);
		this.group.matrix.makeBasis(x.multiplyScalar(s), y.multiplyScalar(s), z.multiplyScalar(s)).setPosition(o);
		this.group.matrixWorldNeedsUpdate = true;
		if (!this.under) this.build(p, this.group.matrix);
	}
}
