// Equipment: both goals and the sticks, modelled on the real IIHF goal and a senior stick, under the arena's light (light.ts): the
// goal's frame painted steel, its padding fabric; a stick all black: its shaft carbon fibre, its blade and knob in black tape.
import * as THREE from "three";
import { SECTION, sweepIndex, sweepInto } from "./forms";
import { FINISH, litMaterial } from "./light";
import type { Rink, Vec3 } from "./types";

type V3 = THREE.Vector3;
const srgb = (r: number, g: number, b: number) => new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);

// ---- goal ---------------------------------------------------------------------------------------------------------------------

const POST_R = 0.025,
	HALF = 0.915 + POST_R,
	BAR_Z = 1.22 + POST_R; // 5 cm tubes, 1.83 x 1.22 m opening
const DEPTH = 1.12,
	SHELF = 0.44,
	MESH = 0.045,
	BASE_R = 0.022; // base depth, top shelf depth, net mesh, base frame tube radius (m)

// Base frame and top shelf outlines as one parameter u in [-1, 1] from the left post round the back to the right post: in plan,
// arcs of a circle through both posts and `depth` behind the goal line at the middle (d behind the line, y across). The net hangs
// between them.
const arc = (u: number, depth: number): [number, number] => {
	const r = (HALF * HALF + depth * depth) / (2 * depth),
		c = depth - r, // the circle's centre, along d
		a = u * Math.acos(-c / r);
	return [c + r * Math.cos(a), r * Math.sin(a)];
};
const base = (u: number): Vec3 => [...arc(u, DEPTH), 0];
const shelf = (u: number): Vec3 => [...arc(u, SHELF), BAR_Z];

function twineTexture() {
	const n = 256,
		cv = Object.assign(document.createElement("canvas"), { width: n, height: n }),
		ctx = cv.getContext("2d")!,
		img = ctx.createImageData(n, n);
	const w = 0.07; // twine width / mesh (a 3 mm twine: from the broadcast's camera the net is barely there)
	for (let j = 0; j < n; j++)
		for (let i = 0; i < n; i++) {
			const x = (i + 0.5) / n,
				y = (j + 0.5) / n,
				d = Math.min(Math.abs(x - 0.5), Math.abs(y - 0.5)); // lines through the centre
			const knot = Math.hypot(x - 0.5, y - 0.5) < 0.07 ? 1 : 0;
			const a = Math.max(knot, 1 - THREE.MathUtils.smoothstep(d, w * 0.35, w * 0.65));
			img.data.set([248, 248, 250, Math.round(a * 245)], (j * n + i) * 4);
		}
	ctx.putImageData(img, 0, 0);
	const t = new THREE.CanvasTexture(cv);
	t.wrapS = t.wrapT = THREE.RepeatWrapping;
	t.colorSpace = THREE.SRGBColorSpace;
	t.anisotropy = 8;
	return t;
}

// The net's twine, unlit white: its texture's cover (mipmapped, so far off the share of a cell the twine covers) thickened toward
// grazing views, where the twines of a cell overlap: 1 - (1 - cover)^(1 / |n.v|) (at most GRAZE).
const GRAZE = 4;
function netMaterial() {
	return new THREE.ShaderMaterial({
		uniforms: { map: { value: twineTexture() }, colour: { value: srgb(0.88, 0.88, 0.9) } },
		transparent: true,
		side: THREE.DoubleSide,
		depthWrite: false,
		vertexShader: /* glsl */ `
varying vec2 vUv; varying vec3 vN, vP;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0); vUv = uv; vP = w.xyz; vN = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * viewMatrix * w;
}`,
		fragmentShader: /* glsl */ `
uniform sampler2D map; uniform vec3 colour; varying vec2 vUv; varying vec3 vN, vP;
void main() {
  vec4 t = texture2D(map, vUv);
  float facing = abs(dot(normalize(vN), normalize(cameraPosition - vP)));
  gl_FragColor = vec4(t.rgb * colour, 1.0 - pow(1.0 - t.a, min(1.0 / max(facing, 1e-3), ${GRAZE.toFixed(1)})));
  #include <colorspace_fragment>
}`,
	});
}

// The net: the skirt from the base frame up to the top shelf (bulging out a little between them), and the roof under the shelf.
// uv: twine cells, laid as diamonds.
function netGeometry() {
	const nu = 64,
		nv = 14,
		pos: number[] = [],
		uv: number[] = [],
		idx: number[] = [],
		len = [0];
	for (let i = 1; i <= nu; i++) {
		// base arc length, for uv
		const [d0, y0] = base(-1 + (2 * (i - 1)) / nu),
			[d1, y1] = base(-1 + (2 * i) / nu);
		len.push(len.at(-1)! + Math.hypot(d1 - d0, y1 - y0));
	}
	const diamond = (s: number, t: number) => [(s + t) / MESH / Math.SQRT2, (t - s) / MESH / Math.SQRT2];
	for (let i = 0; i <= nu; i++) {
		const u = -1 + (2 * i) / nu,
			b = base(u),
			s = shelf(u),
			out = Math.cos((u * Math.PI) / 2); // bulge is largest at the back
		for (let j = 0; j <= nv; j++) {
			const v = j / nv,
				sag = 0.05 * Math.sin(Math.PI * v) * out,
				n = Math.hypot(b[0], b[1]) || 1;
			pos.push(b[0] + (s[0] - b[0]) * v + (sag * b[0]) / n, b[1] + (s[1] - b[1]) * v + (sag * b[1]) / n, v * BAR_Z);
			uv.push(...diamond(len[i]!, v * Math.hypot(BAR_Z, DEPTH - SHELF)));
		}
	}
	for (let i = 0; i < nu; i++)
		for (let j = 0; j < nv; j++) {
			const a = i * (nv + 1) + j,
				b = a + nv + 1;
			idx.push(a, b, a + 1, a + 1, b, b + 1);
		}
	const roof0 = pos.length / 3;
	pos.push(SHELF * 0.4, 0, BAR_Z - 0.03);
	uv.push(...diamond(0, 0));
	for (let i = 0; i <= nu; i++) {
		const [d, y] = shelf(-1 + (2 * i) / nu);
		pos.push(d, y, BAR_Z);
		uv.push(...diamond(y, d));
	}
	for (let i = 0; i < nu; i++) idx.push(roof0, roof0 + 1 + i, roof0 + 2 + i);
	const g = new THREE.BufferGeometry();
	g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
	g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
	g.setIndex(idx);
	g.computeVertexNormals();
	return g;
}

// A frame curve's point (d, y) moved `by` toward the goal's middle, so what runs along it stays inside the net hung from it.
function insetBy(d: number, y: number, by: number): [number, number] {
	const k = Math.max(0, 1 - by / (Math.hypot(d, y) || 1));
	return [d * k, y * k];
}
const inset = ([d, y]: Vec3, by: number, z: number): Vec3 => [...insetBy(d, y, by), z];

// A band of fabric hanging along a frame curve: the white base pad inside the base frame, the skirt under the top shelf.
function bandGeometry(curve: (u: number) => Vec3, from: number, to: number, h0: number, h1: number, inset: number) {
	const n = 48,
		pos: number[] = [],
		idx: number[] = [];
	for (let i = 0; i <= n; i++) {
		const u = from + ((to - from) * i) / n,
			[d, y] = curve(u),
			[x, z] = insetBy(d, y, inset);
		pos.push(x, z, h0, x, z, h1);
	}
	for (let i = 0; i < n; i++) {
		const a = 2 * i;
		idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
	}
	const g = new THREE.BufferGeometry();
	g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
	g.setIndex(idx);
	g.computeVertexNormals();
	return g;
}

// A tube along points (local goal frame), smooth.
function tube(points: Vec3[], r: number, closed = false) {
	return new THREE.TubeGeometry(
		new THREE.CatmullRomCurve3(
			points.map((p) => new THREE.Vector3(...p)),
			closed,
		),
		Math.max(8, points.length * 3),
		r,
		12,
		closed,
	);
}
const curvePoints = (fn: (u: number) => Vec3, n = 32) => Array.from({ length: n + 1 }, (_, i) => fn(-1 + (2 * i) / n));

type GoalMaterials = Record<"red" | "white" | "pad" | "net", THREE.Material>;

function goal(sx: number, rink: Rink, mats: GoalMaterials) {
	const g = new THREE.Group();
	const posts: Vec3[] = [
		[0, -HALF, 0],
		[0, -HALF, BAR_Z],
		[0, HALF, BAR_Z],
		[0, HALF, 0],
	];
	g.add(
		new THREE.Mesh(
			new THREE.TubeGeometry(
				new THREE.CatmullRomCurve3(
					posts.map((p) => new THREE.Vector3(...p)),
					false,
					"catmullrom",
					0.02,
				),
				64,
				POST_R,
				16,
			),
			mats.red,
		),
	);
	g.add(
		new THREE.Mesh(
			tube(
				curvePoints((u) => inset(base(u), BASE_R + 0.006, BASE_R)),
				BASE_R,
			),
			mats.pad,
		),
	); // base frame, padded, inside the net
	g.add(new THREE.Mesh(tube(curvePoints(shelf), 0.024), mats.pad)); // top shelf frame, padded, from the crossbar corners
	g.add(
		new THREE.Mesh(
			tube(
				[
					[DEPTH - 0.03, 0, 0.03],
					[(DEPTH + SHELF) / 2 - 0.02, 0, BAR_Z / 2],
					[SHELF - 0.01, 0, BAR_Z - 0.02],
				],
				0.014,
			),
			mats.white,
		),
	); // back bar
	g.add(new THREE.Mesh(bandGeometry(base, -0.97, 0.97, 0.0, 0.22, 2 * BASE_R + 0.012), mats.pad)); // pad inside the base
	g.add(new THREE.Mesh(bandGeometry(shelf, -0.9, 0.9, BAR_Z - 0.045, BAR_Z, 0.01), mats.pad)); // skirt under the top shelf
	const net = new THREE.Mesh(netGeometry(), mats.net);
	net.renderOrder = 1;
	net.userData.ao = false; // see-through: ambient occlusion (post.ts) would take it for a wall
	g.add(net);
	g.position.set(sx * rink.goal_x, 0, 0);
	g.scale.set(sx, 1, 1); // depth runs away from the centre
	return g;
}

export function buildGoals(rink: Rink) {
	const mats: GoalMaterials = {
		// the deep crimson the broadcast shows the frame in, the padding an off-white
		red: litMaterial(FINISH.paint, { colour: [0.5, 0.06, 0.08] }),
		white: litMaterial(FINISH.paint, { colour: [0.88, 0.88, 0.89] }),
		pad: litMaterial(FINISH.fabric, { colour: [0.84, 0.84, 0.85] }),
		net: netMaterial(),
	};
	const grp = new THREE.Group();
	for (const sx of [-1, 1]) grp.add(goal(sx, rink, mats));
	return grp;
}

// ---- stick --------------------------------------------------------------------------------------------------------------------
// One piece, swept from the knob to the toe in the plane of the shaft and the blade: the shaft narrows into the hosel, turns
// through the heel (round a FILLET, a soft bend) into a thin neck along the ice, and the blade rises out of the neck (its top
// edge a concave curve up from the hosel, over RISE) to its full height, curving slightly (CURVE) and rounded at the toe.

const LENGTH = 1.58; // heel to knob along the shaft
const SHAFT = [0.0145, 0.011] as const; // half sizes: 2.9 x 2.2 cm (wide side in the blade's plane)
const HOSEL = 0.22; // the shaft narrows over its last 22 cm into the neck
const NECK = [0.0125, 0.0085] as const; // the neck's half sizes (in the plane, across it)
const FILLET = 0.07; // the heel's turn: radius of the centre line (a soft bend, shaft and blade one piece)
const RISE = 0.12; // the blade grows from the neck to its full height over this
const LIE = [0.25, 1.45] as const; // rad over the ice: the lies the stick is shaped for (the heel's turn), the rest turned to
const KNOB = 0.11; // tape at the top
const BLADE_L = 0.31,
	BLADE_H = 0.075,
	BLADE_T = 0.0055,
	CURVE = 0.012; // blade length from the heel, height, half thickness, curve depth
const SHAFT_N = 10,
	HOSEL_N = 5,
	TURN_N = 10,
	BLADE_N = 16;
const TAPE_AT = 2 + SHAFT_N + HOSEL_N + TURN_N + 1; // the rings: the knob's two, the shaft, the hosel, the turn, the blade (its first twice)
const RINGS = TAPE_AT + BLADE_N;
const BLACK = srgb(0.05, 0.05, 0.055),
	TAPE = srgb(0.03, 0.03, 0.034); // black tape, as the footage shows
const STICK = { shaft: litMaterial(FINISH.carbon, { painted: true }), blade: litMaterial(FINISH.tape, { painted: true }) };
const smooth01 = (x: number) => {
	const t = Math.min(Math.max(x, 0), 1);
	return t * t * (3 - 2 * t);
};

/** A ring along the stick: its centre (g along the blade from the heel, h up from the ice), its direction (dg, dh), its half sizes
 * (in the plane, across it) and how far along the blade (0 at the heel, 1 at the toe; -1 off it). */
type Ring = { g: number; h: number; dg: number; dh: number; w: number; t: number; along: number };

/** The rings, knob to toe, for a shaft `lie` rad over the ice. */
function rings(lie: number): Ring[] {
	const c = Math.cos(lie),
		s = Math.sin(lie),
		T = FILLET * Math.tan(lie / 2), // from the corner (0, NECK[0]) back up the shaft, and on along the ice, to the turn's ends
		out: Ring[] = [],
		shaft = (l: number, w: number, t: number) =>
			out.push({ g: -T * c - l * c, h: NECK[0] + T * s + l * s, dg: c, dh: -s, w, t, along: -1 }),
		top = LENGTH - T; // the knob, up the shaft from the turn
	shaft(top, SHAFT[0] * 1.12, SHAFT[1] * 1.12);
	shaft(top - KNOB, SHAFT[0] * 1.12, SHAFT[1] * 1.12);
	for (let k = 0; k < SHAFT_N; k++) shaft(top - KNOB - 0.003 - ((top - KNOB - HOSEL) * k) / SHAFT_N, SHAFT[0], SHAFT[1]);
	for (let k = 0; k < HOSEL_N; k++) {
		const l = HOSEL * (1 - k / HOSEL_N),
			e = smooth01(1 - l / HOSEL);
		shaft(l, SHAFT[0] + (NECK[0] - SHAFT[0]) * e, SHAFT[1] + (NECK[1] - SHAFT[1]) * e);
	}
	// the turn, round the centre (T, NECK[0] + FILLET), heading from down the shaft (-lie) to along the ice (0)
	for (let k = 0; k < TURN_N; k++) {
		const a = -lie * (1 - k / TURN_N);
		out.push({
			g: T + FILLET * Math.sin(a),
			h: NECK[0] + FILLET - FILLET * Math.cos(a),
			dg: Math.cos(a),
			dh: Math.sin(a),
			w: NECK[0],
			t: NECK[1],
			along: -1,
		});
	}
	for (let k = 0; k <= BLADE_N; k++) {
		const u = Math.max(k - 1, 0) / (BLADE_N - 1),
			g = T + u * (BLADE_L - T),
			along = g / BLADE_L,
			rise = smooth01((g - T) / RISE),
			full = BLADE_H * (1 - 0.12 * along) * (along > 0.85 ? Math.sqrt(Math.max(0.15, 1 - ((along - 0.85) / 0.15) ** 2)) : 1),
			w = NECK[0] + (full / 2 - NECK[0]) * rise,
			rocker = 0.01 * ((along - 0.4) ** 2 - 0.16); // on the ice at the heel, a little off it at the toe
		out.push({
			g,
			h: rocker + w,
			dg: 1,
			dh: 0,
			w,
			t: NECK[1] + (BLADE_T * (1 - 0.3 * along) - NECK[1]) * smooth01(u * 4),
			along,
		});
	}
	return out;
}

export class Stick {
	mesh: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial[]>;

	constructor() {
		const g = new THREE.BufferGeometry(),
			pos = new THREE.BufferAttribute(new Float32Array((RINGS * SECTION + 2) * 3), 3),
			col = new Float32Array((RINGS * SECTION + 2) * 3);
		pos.setUsage(THREE.DynamicDrawUsage);
		g.setAttribute("position", pos);
		g.setIndex(sweepIndex(0, RINGS, true));
		// carbon to the blade (its knob taped), the blade taped (with both ends' caps)
		const quads = (TAPE_AT - 1) * SECTION * 6;
		g.addGroup(0, quads, 0);
		g.addGroup(quads, Infinity, 1);
		for (let k = 0; k < RINGS; k++) {
			const c = k < 2 || k >= TAPE_AT ? TAPE : BLACK;
			for (let i = 0; i < SECTION; i++) col.set([c.r, c.g, c.b], (k * SECTION + i) * 3);
		}
		col.set([TAPE.r, TAPE.g, TAPE.b], RINGS * SECTION * 3);
		col.set([TAPE.r, TAPE.g, TAPE.b], (RINGS * SECTION + 1) * 3);
		g.setAttribute("color", new THREE.BufferAttribute(col, 3));
		this.mesh = new THREE.Mesh(g, [STICK.shaft, STICK.blade]);
		this.mesh.frustumCulled = false;
	}

	/**
	 * Held at its top in `hand`, its shaft toward `toward` (the tracked blade heel), LENGTH long: the blade on the ice where the
	 * shaft reaches it (turned about the hand toward it as far as it must), else in the air where the shaft points (a stick raised).
	 */
	hold(hand: THREE.Vector3, toward: THREE.Vector3) {
		const d = this.d.subVectors(toward, hand).normalize(),
			heel = this.heel.copy(hand).addScaledVector(d, LENGTH);
		if (heel.z < 0) {
			const flat = Math.sqrt(Math.max(LENGTH * LENGTH - hand.z * hand.z, 0)),
				h = Math.hypot(d.x, d.y) || 1;
			heel.set(hand.x + (d.x / h) * flat, hand.y + (d.y / h) * flat, 0);
		}
		this.set(hand, heel);
	}
	private d = new THREE.Vector3();
	private heel = new THREE.Vector3();

	/** grip: the tracked hand holding the stick, heel: the blade heel on (or near) the ice; both in the rink frame. The stick is
	 * shaped for its lie (within LIE), the blade flat on the ice; a shaft steeper or flatter than that (a stick raised, the
	 * blade up in a windup) turns the whole stick about the blade's face normal, round the heel, so its shaft runs to the hand. */
	set(grip: THREE.Vector3, heel: THREE.Vector3) {
		const V3 = THREE.Vector3,
			z = new V3(0, 0, 1);
		const axis = new V3().subVectors(grip, heel).normalize(); // up the shaft
		const ground = new V3(-axis.x, -axis.y, 0);
		if (ground.lengthSq() < 1e-8) ground.set(1, 0, 0);
		ground.normalize(); // along the blade
		const side = new V3().crossVectors(z, ground); // the blade's face normal (horizontal)
		const lie = Math.asin(Math.min(Math.max(axis.z, -1), 1)),
			shaped = Math.min(Math.max(lie, LIE[0]), LIE[1]),
			turn = new THREE.Quaternion().setFromAxisAngle(side, lie - shaped); // + raises the shaft toward the vertical
		const base = new V3(heel.x, heel.y, Math.max(heel.z - BLADE_H * 0.5, 0) + 0.003);
		const P: V3[] = [],
			U: V3[] = [],
			W: V3[] = [],
			a: number[] = [],
			b: number[] = [],
			U0 = side.clone().applyQuaternion(turn);
		for (const r of rings(shaped)) {
			const bend = r.along >= 0 ? CURVE * Math.sin(Math.PI * r.along * 0.8) : 0,
				d = new V3().addScaledVector(ground, r.dg).addScaledVector(z, r.dh);
			const off = new V3().addScaledVector(ground, r.g).addScaledVector(z, r.h).addScaledVector(side, bend).applyQuaternion(turn);
			P.push(off.add(base));
			U.push(U0);
			W.push(new V3().crossVectors(d, side).normalize().applyQuaternion(turn)); // in the plane, across the path (up, along the blade)
			a.push(r.t);
			b.push(r.w);
		}
		const pos = this.mesh.geometry.attributes.position as THREE.BufferAttribute;
		sweepInto(pos.array, 0, P, U, W, a, b);
		pos.needsUpdate = true;
		this.mesh.geometry.computeVertexNormals();
	}
}
