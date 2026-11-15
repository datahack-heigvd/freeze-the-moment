// Everyone on the ice: their fitted body mesh per exported frame, with their body texture (player_textures.py: the broadcast
// texture with the roster's surname and number on the back) on the MHR UV layout, or their broadcast vertex colours where a scene has
// no textures. Lit by the arena's light (light.ts), each garment with its own finish (FINISH); their jerseys, pants and socks folded
// (FOLD: the cloth maps, as relaxed or squashed as the body is there) with the fabric's weave close up. Everyone wears a helmet,
// goalies a mask (helmet.ts), and skates (skates.ts); players gloves, goalies leg pads (gear.ts): `gear`.
import * as THREE from "three";
import { Gloves, Pads } from "./gear";
import { type Gear, Helmet } from "./helmet";
import { arenaLight, LIGHT_GLSL } from "./light";
import { Skates } from "./skates";
import type { PersonMeta, SceneData } from "./types";
import { bin, body, srgbToLinear, texture } from "./util";

// Each garment's finish under the arena's light (light.ts), by its garment id (cloth_maps.py: 0 skin, which is also the gloves, 1
// jersey, 2 pants, 3 socks): roughness and sheen. A jersey's mesh knit is matte with a strong sheen, the pants' nylon shell smoother,
// the socks a knit, gloves and faces between.
const FINISH = /* glsl */ `
vec2 finish(float g) {                                    // roughness, sheen
  return g < 0.5 ? vec2(0.55, 0.12) : g < 1.5 ? vec2(0.72, 0.5) : g < 2.5 ? vec2(0.42, 0.15) : vec2(0.82, 0.42);
}`;

// Cloth folds (pipeline/cloth_maps.py, in the MHR texture layout, the same for everyone): folds.png the folds of relaxed cloth,
// squash.png those of squashed cloth, each texel the height's derivatives along u and v (K m per unit at full scale) and its valley's
// shadow; garment.bin each texture-coordinate vertex's garment; joints.json and joints.bin where the body bends (Bends); loaded
// with the first scene's people. Blended per pixel by how squashed the body is there (bodyFrame's squash: how far the nearest joint
// bends), turned into a surface gradient from screen derivatives (no tangents), DEPTH times as deep, valleys darkened by up to
// SHADOW. Close up, at high quality only (clothWeave), the weave: bumps WEAVE m apart and as deep, per garment (jersey mesh, pants
// shell, sock knit), fixed to the cloth (rest), faded out as a pixel grows past a fifth of their spacing.
const FOLD = { K: 0.8, DEPTH: 3.2, SHADOW: 0.45 },
	WEAVE = { JERSEY: [0.0025, 0.00012], PANTS: [0.0016, 0.00004], SOCKS: [0.003, 0.00015] },
	FOLDS = /* glsl */ `
uniform sampler2D folds, squashed; uniform float weaveOn;
float weave(vec3 m, float g) {                            // the fabric's bumps at material point m (m), garment g
  vec2 w = g < 1.5 ? vec2(${WEAVE.JERSEY.join(", ")}) : g < 2.5 ? vec2(${WEAVE.PANTS.join(", ")}) : vec2(${WEAVE.SOCKS.join(", ")});
  vec3 q = 6.2831853 * m / w.x;
  float far = length(fwidth(m)) / w.x;
  return w.y * sin(q.x) * sin(q.y) * sin(q.z) * (1.0 - smoothstep(0.2, 0.45, far));
}`;
function loadCloth() {
	const load = (name: string) =>
		new THREE.TextureLoader().loadAsync(`/assets/cloth/${name}.png`).then((t) => {
			t.colorSpace = THREE.NoColorSpace;
			t.flipY = false;
			t.minFilter = THREE.LinearMipmapLinearFilter;
			return t;
		});
	return Promise.all([
		load("folds"),
		load("squash"),
		bin("/assets/cloth/garment.bin", Uint8Array),
		fetch("/assets/cloth/joints.json").then((r) => r.json() as Promise<{ rings: number[][]; joints: number[][] }>),
		bin("/assets/cloth/joints.bin", Uint8Array),
	]).then(([folds, squash, garment, { rings, joints }, zone]) => {
		const bends: Bends = {
			rings: Uint32Array.from(rings.flat()),
			starts: Uint32Array.from(
				rings.reduce(
					(a, r) => {
						a.push(a.at(-1)! + r.length);
						return a;
					},
					[0],
				),
			),
			joints: Uint32Array.from(joints.flat()),
			zone,
		};
		return { folds, squash, garment, bends };
	});
}
type Cloth = Awaited<ReturnType<typeof loadCloth>>;
let cloth: Promise<Cloth> | null = null;

/** Whether bodies show the fabric's weave close up: 1 at high quality (viewer.ts sets it with the quality), 0 below. */
export const clothWeave = { value: 1 };

// A body comes with its equipment's bulk (scene_export.py: the pads, fitted to the broadcast's outlines); the cloth over them (a
// loose jersey, the pants' shell, the socks) stands PAD m further off wherever it is cloth (clothWeight), along its own normals
// (the drape's, bodyFrame).
const PAD = 0.024;

// ghost: 0 where the broadcast sees the person, up to 1 where it does not; fade: how much of them is left (scene.ts fades the long
// lost out). A ghost is see-through as games draw what stands between the camera and the player: their own colours, cooled and
// greyed, down to GHOST opaque (more at the rim), blended over what is behind; a depth-only copy drawn first (Person.seeThrough)
// leaves only the nearest surface to blend, so an arm behind the body does not show through it.
const GHOST = 0.35;
/** Cloth's colour saturation against its texture's: the kit design holds each band's colour as the broadcast showed it lit flat
 * on, purer than a woven fabric under the arena's light looks (matched to the broadcast's kits). */
const CLOTH_SATURATION = 0.72;

function material(tex: THREE.Texture | null, folds: Cloth) {
	return new THREE.ShaderMaterial({
		uniforms: {
			map: { value: tex },
			folds: { value: folds.folds },
			squashed: { value: folds.squash },
			weaveOn: clothWeave,
			ghost: { value: 0 },
			fade: { value: 1 },
			blend: { value: 0 },
			head: { value: new THREE.Vector3() },
			...arenaLight,
		},
		defines: tex ? { MAP: 1 } : {},
		vertexColors: !tex,
		vertexShader: /* glsl */ `
attribute float cloth, garment, squash, nextSquash, tuck; attribute vec3 rest; attribute vec3 next, nextNormal; uniform float blend; uniform vec3 head;
varying vec3 vN; varying vec2 vUv; varying vec3 vC; varying vec3 vP; varying float vCloth, vGarment, vSquash; varying vec3 vRest;
void main() {
  vec3 n = mix(normal, nextNormal, blend);
  vN = normalize(mat3(modelMatrix) * n);
  vec3 p = mix(position, next, blend) + normalize(n) * ${PAD.toFixed(3)} * cloth;   // the cloth, off the pads
  vP = (modelMatrix * vec4(p - tuck * normalize(p - head), 1.0)).xyz;
  vCloth = cloth; vGarment = garment; vSquash = mix(squash, nextSquash, blend); vRest = rest;
#ifdef MAP
  vUv = uv;
#else
  vC = color;
#endif
  gl_Position = projectionMatrix * viewMatrix * vec4(vP, 1.0);
}`,
		fragmentShader:
			LIGHT_GLSL +
			FINISH +
			FOLDS +
			/* glsl */ `
uniform sampler2D map; uniform float ghost, fade; varying vec3 vN; varying vec2 vUv; varying vec3 vC; varying vec3 vP; varying float vCloth, vGarment, vSquash; varying vec3 vRest;
void main() {
  vec3 n = normalize(vN); float valley = 0.0;
#ifdef MAP
  vec3 albedo = texture2D(map, vUv).rgb;
  albedo = mix(vec3(dot(albedo, vec3(0.2126, 0.7152, 0.0722))), albedo, mix(1.0, ${CLOTH_SATURATION.toFixed(2)}, vCloth));
  if (vCloth > 0.0) {                                    // bump: the folds' surface gradient, from the maps' derivatives along u and v
    vec2 ux = dFdx(vUv), uy = dFdy(vUv);
    float s = vSquash;                                    // relaxed folds pulled half flat where the cloth bunches into squashed ones
    vec3 a = texture2D(folds, vUv).rgb, b = s > 0.01 ? textureGrad(squashed, vUv, ux, uy).rgb : vec3(0.5, 0.5, 0.0);   // most cloth is not squashed
    vec2 g = ((a.rg - 0.5) * (1.0 - 0.5 * s) + (b.rg - 0.5) * s) * ${(2 * FOLD.K * FOLD.DEPTH).toFixed(4)} * vCloth;
    valley = (a.b * (1.0 - 0.5 * s) + b.b * s) * vCloth;
    float w = weaveOn > 0.5 ? weave(vRest, vGarment) * vCloth : 0.0;
    vec3 sx = dFdx(vP), sy = dFdy(vP), r1 = cross(sy, n), r2 = cross(n, sx); float det = dot(sx, r1);
    vec3 bump = abs(det) * n - sign(det) * ((dot(g, ux) + dFdx(w)) * r1 + (dot(g, uy) + dFdy(w)) * r2);
    if (dot(bump, bump) > 1e-20) n = normalize(bump);     // a pixel seeing the surface edge-on has no gradient: no bump
  }
#else
  vec3 albedo = vC;
#endif
  vec3 v = normalize(cameraPosition - vP); vec2 fin = finish(vGarment);
  vec3 c = arenaLit(Surface(albedo, n, fin.x, 0.04, fin.y, 0.0, 1.0 - ${FOLD.SHADOW.toFixed(4)} * min(valley, 1.0)), vP, v);
  float rim = pow(1.0 - abs(dot(normalize(vN), v)), 2.0);   // brightest where the surface turns away
  float keep = mix(1.0, ${GHOST.toFixed(2)} + 0.25 * rim, ghost) * fade;
  float l = dot(c, vec3(0.3, 0.59, 0.11));
  gl_FragColor = vec4(mix(c, mix(vec3(l) * vec3(0.92, 0.97, 1.06), c, 0.35) + 0.08 * rim, ghost), keep);
  #include <colorspace_fragment>
}`,
	});
}

/** The MHR UV layout: texcoords, faces over them, and the body vertex of each texcoord. */
export type UvLayout = { tc: Float32Array; tf: Uint32Array; vertOf: Uint32Array };

/**
 * One frame of a body (a row of its stream), built once: positions (m, gear placed) and normals split along the UV seams, ready to upload; the gear's
 * own matrices (local, to blend) and its meshes' world matrices (whose change is their motion); the ground position; and the box
 * the body fills, for culling it where no camera sees it.
 */
type Built = {
	pos: Float32Array;
	nrm: Float32Array;
	sq: Float32Array;
	local: Float32Array;
	world: Float32Array;
	root: [number, number];
	box: THREE.Box3;
};

/** Position, normal and squash attributes on the GPU, holding the body's row `at`. */
type Slot = { at: number; pos: THREE.BufferAttribute; nrm: THREE.BufferAttribute; sq: THREE.BufferAttribute };

const both = new THREE.Box3();

export class Person {
	/** frames built on the main thread so far, by every person, when no worker had (bench.ts tells a frame that built many) */
	static built = 0;
	meta: PersonMeta;
	body: Stream;
	V: number;
	/** exported frames (the scene's) */
	frames: number;
	faces: Uint32Array;
	vertOf: Uint32Array | null;
	/** where the body bends, for bodyFrame's squash */
	bends: Bends;
	/** how the cloth hangs over the body, for bodyFrame's normals */
	drape: Drape;
	mesh: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>;
	/** the body's depth alone, drawn just before it while it is see-through */
	private depth: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>;
	gear: Gear[];
	private built = new Map<number, Built>();
	/** frames being built by a worker */
	private coming = new Set<number>();
	/** the frames to keep built (want) */
	private wanted: number[] = [];
	private slots: [Slot, Slot];
	/** gear objects placed by matrix, and gear meshes, found once the gear is first placed */
	private moved: THREE.Object3D[] | null = null;
	private meshes: THREE.Mesh[] = [];
	private shown: [number, number] = [-1, -1];
	private a = 0;
	private posed: [number, number, number] = [-1, -1, NaN];
	private headAt: ((into: THREE.Vector3) => void) | null = null;

	// body: as it streams in, its first chunk in; faces: uint32 [F * 3]; uv: the MHR layout, or null with colors: uint8 [V * 3];
	// stickTop: float32 [frames * 3], the top of their stick at each exported frame (NaN without one); folds: the cloth maps.
	constructor(
		meta: PersonMeta,
		body: Stream,
		frames: number,
		faces: Uint32Array,
		V: number,
		tex: THREE.Texture | null,
		uv: UvLayout,
		colors: Uint8Array | null,
		stickTop: Float32Array,
		folds: Cloth,
	) {
		this.meta = meta;
		this.body = body;
		this.frames = frames;
		this.V = V;
		const verts = body.v;
		this.faces = faces;
		this.vertOf = tex ? uv.vertOf : null;
		const g = new THREE.BufferGeometry(),
			U = tex ? uv.vertOf.length : V,
			slot = (): Slot => ({
				at: -1,
				pos: new THREE.BufferAttribute(new Float32Array(U * 3), 3).setUsage(THREE.DynamicDrawUsage),
				nrm: new THREE.BufferAttribute(new Float32Array(U * 3), 3).setUsage(THREE.DynamicDrawUsage),
				sq: new THREE.BufferAttribute(new Float32Array(U), 1).setUsage(THREE.DynamicDrawUsage),
			});
		this.bends = folds.bends;
		this.slots = [slot(), slot()];
		const cloth = clothWeight(verts, V);
		this.drape = drapeOf(body, V, faces, folds.bends, cloth);
		const rest = Float32Array.from({ length: V * 3 }, (_, i) => verts[i]! / 1000),
			split = (a: Float32Array, k: number) =>
				tex ? Float32Array.from({ length: uv.vertOf.length * k }, (_, i) => a[uv.vertOf[Math.floor(i / k)]! * k + (i % k)]!) : a;
		g.setAttribute("cloth", new THREE.BufferAttribute(split(cloth, 1), 1));
		g.setAttribute("rest", new THREE.BufferAttribute(split(rest, 3), 3));
		g.setAttribute("garment", new THREE.BufferAttribute(tex ? Float32Array.from(folds.garment) : new Float32Array(V), 1));
		if (tex) {
			g.setAttribute("uv", new THREE.BufferAttribute(uv.tc, 2));
			g.setIndex(new THREE.BufferAttribute(uv.tf, 1));
		} else {
			const col = new Float32Array(V * 3);
			for (let i = 0; i < V * 3; i++) col[i] = srgbToLinear(colors![i]! / 255);
			g.setAttribute("color", new THREE.BufferAttribute(col, 3));
			g.setIndex(new THREE.BufferAttribute(faces, 1));
		}
		this.mesh = new THREE.Mesh(g, material(tex, folds));
		this.mesh.geometry.boundingSphere = new THREE.Sphere();
		this.mesh.userData.blend = 0; // read by the broadcast look's geometry pass (post.ts), as is the head's centre
		this.mesh.userData.head = this.mesh.material.uniforms.head!.value;
		const depth = this.mesh.material.clone();
		depth.uniforms = this.mesh.material.uniforms;
		Object.assign(depth, { colorWrite: false, transparent: true });
		this.depth = new THREE.Mesh(g, depth);
		Object.assign(this.depth, { userData: this.mesh.userData, renderOrder: 1, visible: false, frustumCulled: false });
		this.mesh.add(this.depth);
		this.attach(this.slots[0], this.slots[1]);
		const tc = tex ? uv.tc : null,
			vertOf = tex ? uv.vertOf : null,
			colours = tex ? null : colors;
		this.gear = [
			new Skates(this),
			new Helmet(this, tex, tc, vertOf, colours),
			...(meta.goalie ? [new Pads(tex, tc, vertOf, colours)] : []),
			...(meta.team >= 0 ? [new Gloves(this, tex, tc, vertOf, colours, stickTop)] : []),
		];
	}

	/**
	 * How far (m) each body vertex is drawn in toward the head's centre (head: put where it is as posed, in the body's frame): the
	 * head under a helmet that sits closer than it (helmet.ts).
	 */
	tuck(by: Float32Array, head: (into: THREE.Vector3) => void) {
		const v = this.vertOf;
		this.headAt = head;
		this.mesh.geometry.setAttribute("tuck", new THREE.BufferAttribute(v ? Float32Array.from(v, (i) => by[i]!) : by, 1));
	}

	/** Drawn see-through (a ghost, or fading out): blended after everything opaque, over its own depth drawn just before. */
	set seeThrough(on: boolean) {
		const m = this.mesh.material;
		if (m.transparent === on) return;
		Object.assign(m, { transparent: on, depthWrite: !on, needsUpdate: true });
		this.mesh.renderOrder = on ? 2 : 0;
		this.depth.visible = on;
	}

	/** The body's row at exported frame f: held at the last one decoded, past what has come in. */
	row(f: number) {
		const b = this.body;
		return Math.max(Math.min(f - b.first, b.frames - 1, b.rows - 1), 0);
	}

	/** The body's vertices (int16 mm) at exported frame f, as row(f). */
	at(f: number) {
		const n = this.V * 3,
			r = this.row(f);
		return this.body.v.subarray(r * n, (r + 1) * n);
	}

	/** The exported frames the gear is fitted on: the first chunk's, the same however fast the rest comes. */
	get fitting() {
		return Array.from({ length: this.body.fit }, (_, k) => this.body.first + k);
	}

	/**
	 * Pose a fraction a of the way from exported frame i0 to i1 (the two around a moment, or two far apart a glide goes between).
	 * Both are built once each and sit on the GPU (position and normal, next and nextNormal), blended there by `blend`; per display
	 * frame only the blend and the gear's matrices change.
	 */
	pose(f0: number, f1: number, a: number) {
		const i0 = this.row(f0),
			i1 = this.row(f1),
			posed = this.posed;
		if (posed[0] === i0 && posed[1] === i1 && posed[2] === a) return;
		this.posed = [i0, i1, a];
		this.a = a;
		const b0 = this.build(i0),
			b1 = this.build(i1);
		if (this.shown[0] !== i0 || this.shown[1] !== i1) {
			// a slot that already holds one of the two keeps it; the other is filled
			const [s, t] = this.slots,
				first = s.at === i0 || t.at === i1 ? s : t,
				second = first === s ? t : s;
			this.fill(first, i0, b0);
			this.fill(second, i1, b1);
			this.attach(first, second);
			this.shown = [i0, i1];
			// the body lies between the two frames: culled, in every pass that draws it, outside the box of both
			both.copy(b0.box).union(b1.box).getBoundingSphere(this.mesh.geometry.boundingSphere!);
			this.meshes.forEach((m, k) => {
				// how far each gear mesh moves in a frame: the change of its world matrix
				const motion = (m.userData.motion ??= new THREE.Matrix4()) as THREE.Matrix4;
				for (let e = 0; e < 16; e++) motion.elements[e] = b1.world[16 * k + e]! - b0.world[16 * k + e]!;
			});
		}
		this.mesh.material.uniforms.blend!.value = this.mesh.userData.blend = a;
		this.moved!.forEach((o, k) => {
			blendMatrix(o.matrix, b0.local, b1.local, k, a);
			o.matrixWorldNeedsUpdate = true;
		});
		this.headAt?.(this.mesh.userData.head as THREE.Vector3);
	}

	/**
	 * Keep built only exported frames `frames` (those shown and those soon to be), building those missing in workers, ahead of need:
	 * a frame posed that is not built yet is built there and then (build), holding up the display frame. Built frames are kept by
	 * the body's row, so one held while the rest streams in is built again once its own row is in.
	 */
	want(frames: number[]) {
		const rows = [...new Set(frames.map((f) => this.row(f)))];
		this.wanted = rows;
		for (const k of this.built.keys()) if (!rows.includes(k)) this.built.delete(k);
		for (const i of rows) {
			if (this.built.has(i) || this.coming.has(i) || this.coming.size >= COMING) continue;
			this.coming.add(i);
			void pool(this.meta.key, this.frame(i), this.faces, this.vertOf, this.bends, this.drape).then((f) => {
				this.coming.delete(i);
				if (this.wanted.includes(i) && !this.built.has(i)) this.built.set(i, this.gearAt(f));
			});
		}
	}

	/** The vertices of the body's row i, a copy (to go to a worker). */
	private frame(i: number) {
		const n = this.V * 3;
		return this.body.v.slice(i * n, (i + 1) * n);
	}

	/** The body's row i, built here if no worker has (when it is needed now). */
	private build(i: number): Built {
		const cached = this.built.get(i);
		if (cached) return cached;
		Person.built++;
		const n = this.V * 3,
			b = this.gearAt(bodyFrame(this.body.v.subarray(i * n, (i + 1) * n), this.faces, this.vertOf, this.bends, this.drape));
		this.built.set(i, b);
		return b;
	}

	/** Built from a frame of the body: the gear placed on it, as a frame built ahead must not show, so what is shown is put back. */
	private gearAt(f: BodyFrame): Built {
		const shown = this.moved?.map((x) => x.matrix.clone());
		for (const g of this.gear) g.place(f.p);
		if (!this.moved) {
			this.moved = [];
			for (const g of this.gear)
				g.group.traverse((x) => {
					if (!x.matrixAutoUpdate) this.moved!.push(x);
					if (x instanceof THREE.Mesh) this.meshes.push(x);
				});
			// gear is rigid, placed by matrix: three bounds each mesh from its geometry and culls it like the body
			for (const m of this.meshes) m.frustumCulled = true;
		}
		const local = new Float32Array(this.moved.length * 16),
			world = new Float32Array(this.meshes.length * 16);
		for (const [k, x] of this.moved.entries()) x.matrix.toArray(local, 16 * k);
		for (const g of this.gear) g.group.updateMatrixWorld(true);
		for (const [k, m] of this.meshes.entries()) m.matrixWorld.toArray(world, 16 * k);
		shown?.forEach((m, k) => {
			this.moved![k]!.matrix.copy(m);
			this.moved![k]!.matrixWorldNeedsUpdate = true;
		});
		const box = new THREE.Box3(new THREE.Vector3(...f.min), new THREE.Vector3(...f.max));
		return { pos: f.pos, nrm: f.nrm, sq: f.sq, local, world, root: f.root, box };
	}

	private fill(slot: Slot, i: number, b: Built) {
		if (slot.at === i) return;
		(slot.pos.array as Float32Array).set(b.pos);
		(slot.nrm.array as Float32Array).set(b.nrm);
		(slot.sq.array as Float32Array).set(b.sq);
		slot.pos.needsUpdate = slot.nrm.needsUpdate = slot.sq.needsUpdate = true;
		slot.at = i;
	}

	private attach(from: Slot, to: Slot) {
		const g = this.mesh.geometry;
		g.setAttribute("position", from.pos);
		g.setAttribute("normal", from.nrm);
		g.setAttribute("next", to.pos);
		g.setAttribute("nextNormal", to.nrm);
		g.setAttribute("squash", from.sq);
		g.setAttribute("nextSquash", to.sq);
	}

	/** Mean ground position as posed, for the contact shadow. */
	root(): [number, number] {
		const p = this.built.get(this.shown[0]),
			q = this.built.get(this.shown[1]);
		if (!p || !q) return [0, 0];
		return [p.root[0] + (q.root[0] - p.root[0]) * this.a, p.root[1] + (q.root[1] - p.root[1]) * this.a];
	}

	/** The mean ground position at fractional exported frame f, without posing. */
	rootAt(f: number): [number, number] {
		const r = this.meta.root,
			n = r.length,
			i = Math.min(Math.max(Math.floor(f), 0), n - 1),
			a = Math.min(Math.max(f - i, 0), 1),
			p = r[i]!,
			q = r[Math.min(i + 1, n - 1)]!;
		return [(p[0] + (q[0] - p[0]) * a) / 1000, (p[1] + (q[1] - p[1]) * a) / 1000];
	}
}

const m0 = new THREE.Matrix4(),
	m1 = new THREE.Matrix4(),
	t0 = new THREE.Vector3(),
	t1 = new THREE.Vector3(),
	q0 = new THREE.Quaternion(),
	q1 = new THREE.Quaternion(),
	s0 = new THREE.Vector3(),
	s1 = new THREE.Vector3();

/** out: the k-th matrices of the arrays from and to, a fraction a of the way between (positions and scales lerped, turns slerped). */
function blendMatrix(out: THREE.Matrix4, from: Float32Array, to: Float32Array, k: number, a: number) {
	m0.fromArray(from, 16 * k).decompose(t0, q0, s0);
	m1.fromArray(to, 16 * k).decompose(t1, q1, s1);
	out.compose(t0.lerp(t1, a), q0.slerp(q1, a), s0.lerp(s1, a));
}

/** A body's frame built from its vertices `v` (int16 mm, one frame): what Built holds but the gear's. */
/**
 * A body as it streams in (util.ts body): v int16 [frames, V, 3] mm, rows of exported frames first .. first + frames - 1, of which
 * `rows` are decoded so far; the gear is fitted on the first `fit` (the first chunk), in before the person is made.
 */
export type Stream = { v: Int16Array; first: number; frames: number; rows: number; fit: number };

/**
 * Where a body bends (pipeline/cloth_maps.py): rings of MHR vertices (rings, from starts[r] to starts[r + 1]), whose mean is a point
 * on a limb's or the body's centre line; joints as ring triples [toward the body, at the joint, away]; and per MHR vertex (zone,
 * [V, 2]) 1 + the joint whose bend squashes the cloth there (0: none) and how much (255: fully).
 */
type Bends = { rings: Uint32Array; starts: Uint32Array; joints: Uint32Array; zone: Uint8Array };

type BodyFrame = {
	p: Float32Array;
	pos: Float32Array;
	nrm: Float32Array;
	sq: Float32Array;
	root: [number, number];
	min: number[];
	max: number[];
};

/**
 * Exported frame v (int16 mm [V * 3]) of a body over faces F: positions in m (p), area-weighted vertex normals and how squashed the
 * cloth is at each vertex (sq, 0 to 1), all three split along the UV seams (vertOf, or as they are without), ground position and
 * bounds. Squash: each joint's bend (Bends: the angle at its middle ring's centre, from 15 degrees off straight, none, to 80,
 * fully), on the vertices near it, the most where they face into the bend; points from rings of many vertices, as the fitted
 * surface jitters a little from frame to frame. The normals are the cloth's (drape: each vertex's own, turned as drapeOf found).
 * Self-contained: the body workers run its source.
 */
function bodyFrame(v: Int16Array, F: Uint32Array, vertOf: Uint32Array | null, bends: Bends, drape: Drape): BodyFrame {
	const p = new Float32Array(v.length),
		n = new Float32Array(v.length),
		min = [Infinity, Infinity, Infinity],
		max = [-Infinity, -Infinity, -Infinity];
	for (let k = 0; k < v.length; k++) {
		const x = (p[k] = v[k]! / 1000),
			e = k % 3;
		if (x < min[e]!) min[e] = x;
		if (x > max[e]!) max[e] = x;
	}
	for (let t = 0; t < F.length; t += 3) {
		const a = F[t]! * 3,
			b = F[t + 1]! * 3,
			c = F[t + 2]! * 3;
		const ux = p[b]! - p[a]!,
			uy = p[b + 1]! - p[a + 1]!,
			uz = p[b + 2]! - p[a + 2]!,
			vx = p[c]! - p[a]!,
			vy = p[c + 1]! - p[a + 1]!,
			vz = p[c + 2]! - p[a + 2]!;
		const x = uy * vz - uz * vy,
			y = uz * vx - ux * vz,
			z = ux * vy - uy * vx;
		n[a]! += x;
		n[a + 1]! += y;
		n[a + 2]! += z;
		n[b]! += x;
		n[b + 1]! += y;
		n[b + 2]! += z;
		n[c]! += x;
		n[c + 1]! += y;
		n[c + 2]! += z;
	}
	let rx = 0,
		ry = 0;
	for (let i = 0; i < n.length; i += 3) {
		const l = Math.sqrt(n[i]! * n[i]! + n[i + 1]! * n[i + 1]! + n[i + 2]! * n[i + 2]!) || 1;
		n[i]! /= l;
		n[i + 1]! /= l;
		n[i + 2]! /= l;
		rx += p[i]!;
		ry += p[i + 1]!;
	}
	const { rings, starts, joints, zone } = bends,
		R = starts.length - 1,
		c = new Float32Array(R * 3);
	for (let r = 0; r < R; r++) {
		for (let k = starts[r]!; k < starts[r + 1]!; k++) for (let e = 0; e < 3; e++) c[3 * r + e]! += p[3 * rings[k]! + e]!;
		for (let e = 0; e < 3; e++) c[3 * r + e]! /= starts[r + 1]! - starts[r]!;
	}
	const J = joints.length / 3,
		bend = new Float32Array(J),
		into = new Float32Array(J * 3);
	for (let j = 0; j < J; j++) {
		const [a, m, b] = [joints[3 * j]!, joints[3 * j + 1]!, joints[3 * j + 2]!],
			u = [0, 1, 2].map((e) => c[3 * a + e]! - c[3 * m + e]!),
			w = [0, 1, 2].map((e) => c[3 * b + e]! - c[3 * m + e]!),
			cos =
				(u[0]! * w[0]! + u[1]! * w[1]! + u[2]! * w[2]!) /
				(Math.hypot(u[0]!, u[1]!, u[2]!) * Math.hypot(w[0]!, w[1]!, w[2]!) || 1),
			t = Math.min(Math.max((Math.PI - Math.acos(Math.min(Math.max(cos, -1), 1)) - 0.26) / 1.14, 0), 1),
			mid = [0, 1, 2].map((e) => (u[e]! + w[e]!) / 2),
			l = Math.hypot(mid[0]!, mid[1]!, mid[2]!) || 1;
		bend[j] = t * t * (3 - 2 * t);
		for (let e = 0; e < 3; e++) into[3 * j + e] = mid[e]! / l;
	}
	const V = p.length / 3,
		sq = new Float32Array(V),
		zoned = Math.min(V, zone.length / 2);
	for (let i = 0; i < zoned; i++) {
		const j = zone[2 * i]! - 1;
		if (j < 0 || !bend[j]) continue;
		const facing = n[3 * i]! * into[3 * j]! + n[3 * i + 1]! * into[3 * j + 1]! + n[3 * i + 2]! * into[3 * j + 2]!,
			t = Math.min(Math.max((facing + 0.3) / 0.9, 0), 1);
		sq[i] = (zone[2 * i + 1]! / 255) * bend[j]! * (0.35 + 0.65 * t * t * (3 - 2 * t));
	}
	const { toward, by } = drape;
	for (let i = 0; i < V; i++) {
		const a = 3 * i,
			b = 3 * toward[i]!,
			nx = n[a]!,
			ny = n[a + 1]!,
			nz = n[a + 2]!,
			ex = p[b]! - p[a]!,
			ey = p[b + 1]! - p[a + 1]!,
			ez = p[b + 2]! - p[a + 2]!,
			along = ex * nx + ey * ny + ez * nz,
			tx = ex - along * nx,
			ty = ey - along * ny,
			tz = ez - along * nz,
			l = Math.hypot(tx, ty, tz) || 1,
			u = by[a]!,
			v = by[a + 1]! / l,
			w = by[a + 2]! / l;
		n[a] = u * nx + v * tx + w * (ny * tz - nz * ty);
		n[a + 1] = u * ny + v * ty + w * (nz * tx - nx * tz);
		n[a + 2] = u * nz + v * tz + w * (nx * ty - ny * tx);
	}
	const split = (a: Float32Array, k: number) => {
		if (!vertOf) return a;
		const out = new Float32Array(vertOf.length * k);
		for (let u = 0; u < vertOf.length; u++) for (let c = 0; c < k; c++) out[k * u + c] = a[k * vertOf[u]! + c]!;
		return out;
	};
	return { p, pos: split(p, 3), nrm: split(n, 3), sq: split(sq, 1), root: [(3 * rx) / p.length, (3 * ry) / p.length], min, max };
}

/**
 * How cloth hangs over a body: a jersey over pads does not show the chest and arms under it. Per vertex, the cloth's normal in the
 * vertex's own frame (`by`: along the body's normal, along the edge to its neighbour `toward` as far as that lies in the surface,
 * and along the third axis), which turns with the body as it moves, so a frame's cloth normals cost no more than its own.
 */
type Drape = { toward: Uint32Array; by: Float32Array };

// The cloth's normals are the body's smoothed PASSES times over the mesh where it is cloth (clothWeight; each pass a vertex takes
// the mean of its faces' corners), on FRAMES frames of the first chunk: what the frames share is the body's own relief, which the
// cloth hides, while a pose's creases, different in each, mostly cancel.
const DRAPE = { PASSES: 16, FRAMES: 4 };
function drapeOf(body: Stream, V: number, F: Uint32Array, bends: Bends, cloth: Float32Array): Drape {
	const toward = new Uint32Array(V),
		worn: number[] = [];
	for (let t = 0; t < F.length; t += 3) {
		for (let e = 0; e < 3; e++) toward[F[t + e]!] = F[t + ((e + 1) % 3)]!;
		if (cloth[F[t]!]! + cloth[F[t + 1]!]! + cloth[F[t + 2]!]! > 0) worn.push(F[t]! * 3, F[t + 1]! * 3, F[t + 2]! * 3);
	}
	const plain = new Float32Array(V * 3),
		by = new Float32Array(V * 3),
		sum = new Float32Array(V * 3),
		n = V * 3;
	for (let i = 0; i < V; i++) plain[3 * i] = 1;
	for (let k = 0; k < DRAPE.FRAMES; k++) {
		const row = Math.floor(((k + 0.5) * body.fit) / DRAPE.FRAMES),
			{ p, nrm } = bodyFrame(body.v.subarray(row * n, (row + 1) * n), F, null, bends, { toward, by: plain }),
			hung = Float32Array.from(nrm);
		for (let pass = 0; pass < DRAPE.PASSES; pass++) {
			sum.fill(0);
			for (let t = 0; t < worn.length; t += 3) {
				const a = worn[t]!,
					b = worn[t + 1]!,
					c = worn[t + 2]!;
				for (let e = 0; e < 3; e++) {
					const s = hung[a + e]! + hung[b + e]! + hung[c + e]!;
					sum[a + e]! += s;
					sum[b + e]! += s;
					sum[c + e]! += s;
				}
			}
			for (let i = 0; i < V; i++) {
				const a = 3 * i,
					w = cloth[i]!,
					l = Math.hypot(sum[a]!, sum[a + 1]!, sum[a + 2]!);
				if (!w || !l) continue; // what is not cloth keeps the body's normal, and holds the cloth's edge to it
				for (let e = 0; e < 3; e++) hung[a + e] = nrm[a + e]! + w * (sum[a + e]! / l - nrm[a + e]!);
			}
		}
		for (let i = 0; i < V; i++) {
			const a = 3 * i,
				b = 3 * toward[i]!,
				N = [nrm[a]!, nrm[a + 1]!, nrm[a + 2]!],
				E = [p[b]! - p[a]!, p[b + 1]! - p[a + 1]!, p[b + 2]! - p[a + 2]!],
				along = E[0]! * N[0]! + E[1]! * N[1]! + E[2]! * N[2]!,
				T = E.map((x, e) => x - along * N[e]!),
				l = Math.hypot(T[0]!, T[1]!, T[2]!) || 1,
				B = [N[1]! * T[2]! - N[2]! * T[1]!, N[2]! * T[0]! - N[0]! * T[2]!, N[0]! * T[1]! - N[1]! * T[0]!];
			for (let e = 0; e < 3; e++) {
				by[a]! += hung[a + e]! * N[e]!;
				by[a + 1]! += (hung[a + e]! * T[e]!) / l;
				by[a + 2]! += (hung[a + e]! * B[e]!) / l;
			}
		}
	}
	for (let a = 0; a < n; a += 3) {
		const l = Math.hypot(by[a]!, by[a + 1]!, by[a + 2]!) || 1;
		for (let e = 0; e < 3; e++) by[a + e]! /= l;
	}
	return { toward, by };
}

/**
 * bodyFrame off the main thread, in parallel: a few workers (one fewer than the cores, at most WORKERS), each body on one of them,
 * which keeps its faces and seams; a frame goes over as its own vertices and comes back as bodyFrame's arrays.
 */
const WORKERS = 4,
	COMING = 4; // frames a body has in the workers at most: past that, behind as it is, it asks again on a later display frame
const pool = (() => {
	const n = Math.min(WORKERS, Math.max((navigator.hardwareConcurrency || 2) - 1, 1)),
		url = URL.createObjectURL(
			new Blob([
				`const bodyFrame = ${bodyFrame}, bodies = new Map();
onmessage = ({ data: d }) => {
  if (d.faces) return void bodies.set(d.key, d);
  const b = bodies.get(d.key), r = bodyFrame(d.v, b.faces, b.vertOf, b.bends, b.drape);
  postMessage({ id: d.id, r }, [r.p.buffer, r.pos.buffer, r.nrm.buffer, r.sq.buffer].filter((x, k, a) => a.indexOf(x) === k));
};`,
			]),
		),
		workers = Array.from({ length: n }, () => new Worker(url)),
		waiting = new Map<number, (r: BodyFrame) => void>(),
		known = new Set<number>();
	let id = 0;
	for (const w of workers)
		w.onmessage = ({ data }: MessageEvent<{ id: number; r: BodyFrame }>) => {
			waiting.get(data.id)?.(data.r);
			waiting.delete(data.id);
		};
	return (key: number, v: Int16Array, faces: Uint32Array, vertOf: Uint32Array | null, bends: Bends, drape: Drape) =>
		new Promise<BodyFrame>((done) => {
			const w = workers[key % n]!;
			if (!known.has(key)) {
				known.add(key);
				w.postMessage({ key, faces, vertOf, bends, drape });
			}
			waiting.set(++id, done);
			w.postMessage({ id, key, v }, [v.buffer]);
		});
})();

// How much each body vertex is cloth, from the first frame (m): none on the head (within HEAD of the crown), the hands (HAND of a
// fingertip) or the skates (SKATE above the lowest point), blended in over BLEND.
const CROWN = 2810,
	FINGERTIPS = [10069, 15459],
	HEAD = 0.28,
	HAND = 0.14,
	SKATE = 0.12,
	BLEND = 0.06;
function clothWeight(verts: Int16Array, V: number) {
	const at = (i: number) => [verts[3 * i]! / 1000, verts[3 * i + 1]! / 1000, verts[3 * i + 2]! / 1000] as const,
		d = (p: readonly number[], q: readonly number[]) => Math.hypot(p[0]! - q[0]!, p[1]! - q[1]!, p[2]! - q[2]!);
	const ramp = (x: number, e: number) => Math.min(Math.max((x - e) / BLEND, 0), 1),
		crown = at(CROWN),
		tips = FINGERTIPS.map(at);
	let low = Infinity;
	for (let i = 0; i < V; i++) low = Math.min(low, verts[3 * i + 2]! / 1000);
	return Float32Array.from({ length: V }, (_, i) => {
		const p = at(i);
		return ramp(d(p, crown), HEAD) * Math.min(...tips.map((t) => ramp(d(p, t), HAND))) * ramp(p[2] - low, SKATE);
	});
}

/**
 * Everyone of a scene: textured when the scene has texture_<key>.jpg, else with their broadcast vertex colours. Their bodies stream
 * in (`soon`: those wanted first, fetched ahead of the rest); `people` is had once everyone's first chunk is in (each person is made
 * then, their gear fitted on it), `streamed` once every body is; the rest of each body is asked for once `rest` is had (all that
 * starting takes is in), not to hold that up. `bytes(got, of)`: of the bodies' compressed bytes, those in so far, of those it takes
 * to have `people`.
 */
export function loadPeople(
	S: SceneData,
	base: string,
	faces: Uint32Array,
	sticks: Float32Array,
	soon: (m: PersonMeta) => boolean,
	bytes: (got: number, of: number) => void,
	rest: Promise<unknown>,
) {
	const B = S.body,
		fit = Math.min(B.chunk, B.frames),
		F = sticks.length / (S.people.length * 6),
		layout = bin(`${base}/uv.bin`, Uint8Array).then((raw): UvLayout => {
			const U = raw.byteLength / 12 - S.faces;
			return {
				tc: new Float32Array(raw.buffer, 0, U * 2),
				tf: new Uint32Array(raw.buffer, U * 8, S.faces * 3),
				vertOf: new Uint32Array(raw.buffer, U * 8 + S.faces * 12, U),
			};
		}),
		got = S.people.map(() => 0),
		need = S.people.map(() => NaN), // the compressed bytes up to the first chunk's end, as a share of the file
		tell = () => {
			const known = need.filter((x) => !Number.isNaN(x)),
				mean = known.length ? known.reduce((a, x) => a + x, 0) / known.length : 1,
				of = need.reduce((a, x) => a + (Number.isNaN(x) ? mean : x), 0);
			bytes(
				got.reduce((a, x, j) => a + Math.min(x, Number.isNaN(need[j]!) ? mean : need[j]!), 0),
				of,
			);
		};
	const streams: Promise<void>[] = [];
	const people = Promise.all(
		S.people.map(async (m, j) => {
			const stream: Stream = { v: new Int16Array(B.frames * S.vertices * 3), first: B.first, frames: B.frames, rows: 0, fit };
			let fitted = () => {};
			const first = new Promise<void>((r) => (fitted = r)),
				streaming = body(
					`${base}/body_${m.key}.gz`,
					S.vertices,
					B,
					stream.v,
					(rows) => {
						stream.rows = rows;
						if (rows >= fit) fitted();
					},
					(n, of) => {
						got[j] = n;
						need[j] = (of * fit) / B.frames;
						tell();
					},
					soon(m) ? "high" : "auto",
					rest,
				);
			streams.push(streaming.catch(() => {})); // logged by body; held at its last frame in
			const top = new Float32Array(F * 3);
			for (let f = 0; f < F; f++) top.set(sticks.subarray((j * F + f) * 6, (j * F + f) * 6 + 3), 3 * f);
			const [uv, tex, folds] = await Promise.all([
				layout,
				texture(`${base}/texture_${m.key}.jpg`).catch(() => null),
				(cloth ??= loadCloth()),
				Promise.race([first, streaming]),
			]);
			return new Person(
				m,
				stream,
				S.frames,
				faces,
				S.vertices,
				tex,
				uv,
				tex ? null : await bin(`${base}/color_${m.key}.bin`, Uint8Array),
				top,
				folds,
			);
		}),
	);
	return { people, streamed: Promise.all(streams).then(() => {}) };
}
