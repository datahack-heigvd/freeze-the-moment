// A goal's reconstructed scene (scene_export.py) in three.js: the arena (arena.ts), both goals and the sticks (props.ts), everyone
// on the ice (people.ts), the lines their skates cut (marks.ts), the snow off their skates (spray.ts), and the puck. setTime(t)
// poses everything at a fractional source frame t. Someone the broadcast does not see at t is a ghost (people.ts), more so the further
// t is from a frame it saw them in (fully over FADE frames), and fades out once that is more than HOLD frames, over OUT more: their
// pose there is the tracker's guess, held or carried on from where they were last seen. A fast puck draws a faint trail on the ice.
import * as THREE from "three";
import { type Arena, buildArena } from "./arena";
import { Gloves, grip, HANDS } from "./gear";
import { Marks } from "./marks";
import { loadPeople, type Person } from "./people";
import type { Perf } from "./perf";
import { buildGoals, Stick } from "./props";
import { Spray } from "./spray";
import type { Cam, PersonMeta, SceneData, Shot } from "./types";
import { bin, bodyHead, request, texture, warm } from "./util";

const FADE = 8,
	HOLD = 50,
	OUT = 25;
const smooth = (e0: number, e1: number, x: number) => {
	const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
	return t * t * (3 - 2 * t);
};

const prefetched = new Map<string, Promise<void>>();

/**
 * Scene `id`'s files as its load first asks for them (scene.json, faces, uv, sticks, the arena's and everyone's textures, each
 * body's head), fetched at low priority into the HTTP cache; its load is then served from there, waiting for any still landing.
 * Once per id (again once aborted); a file that fails is left to the load.
 */
export function prefetchScene(id: string, signal?: AbortSignal): Promise<void> {
	let job = prefetched.get(id);
	if (!job) {
		prefetched.set(id, (job = warmScene(`/scenes/${encodeURIComponent(id)}`, signal)));
		job.catch(() => prefetched.delete(id));
	}
	return job;
}

async function warmScene(base: string, signal?: AbortSignal) {
	const json = async <T>(file: string) => JSON.parse(new TextDecoder().decode(await warm(`${base}/${file}`, { signal }))) as T,
		S = await json<SceneData>("scene.json"),
		get = (file: string, init: RequestInit = {}) => warm(`${base}/${file}`, { ...init, signal });
	// the atlas's images, wherever it keeps them (each part's file, the stands' plain mask, the ice's matte), or a scene's from before atlas.json
	const atlas = json<Record<string, { file?: string; plain?: string; matte?: string } | null>>("atlas.json").then(
		(m) => Object.values(m).flatMap((a) => [a?.file, a?.plain, a?.matte].filter((f) => typeof f === "string")),
		() => ["atlas_ice.jpg", "atlas_out.jpg"],
	);
	await Promise.allSettled([
		...["faces.bin", "sticks.bin", "uv.bin"].map((f) => get(f)),
		...S.people.flatMap((m) => [
			get(`texture_${m.key}.jpg`),
			get(`body_${m.key}.gz`, { headers: { Range: `bytes=0-${bodyHead(S.vertices, S.body) - 1}` } }),
		]),
		atlas.then((files) => Promise.allSettled(files.map((f) => get(f)))),
	]);
	signal?.throwIfAborted();
}

/** Whether someone stands in the first shot's first view (the broadcast's approach), to be fetched first. */
function inView(S: SceneData) {
	const sh = S.shots.approach ?? Object.values(S.shots)[0]!,
		pose = cameraPose(sh.cams[0]!, S.height),
		camera = new THREE.PerspectiveCamera(pose.fov, S.width / S.height, 0.1, 600),
		f = Math.min(Math.max(sh.frames[0]! - S.frame0, 0), S.frames - 1),
		p = new THREE.Vector3();
	camera.position.copy(pose.position);
	camera.quaternion.copy(pose.quaternion);
	camera.updateMatrixWorld();
	return (m: PersonMeta) => {
		const [x, y] = m.root[f]!;
		p.set(x / 1000, y / 1000, 1).project(camera);
		return p.z < 1 && Math.abs(p.x) < 1.2 && Math.abs(p.y) < 1.2;
	};
}

/** Frames from each frame to the nearest one in which seen is true (Infinity if none is). */
function unseen(seen: ArrayLike<number | boolean>) {
	const n = seen.length,
		d = new Float32Array(n).fill(Infinity);
	for (let i = 0, last = -Infinity; i < n; i++) {
		if (seen[i]) last = i;
		d[i] = i - last;
	}
	for (let i = n - 1, next = Infinity; i >= 0; i--) {
		if (seen[i]) next = i;
		d[i] = Math.min(d[i]!, next - i);
	}
	return d;
}

const AHEAD = 2; // frames built ahead of the two shown

// The puck's trail: a thin streak on the ice along where it was over the last WINDOW source frames, at most LONGEST m of it (the
// tracker can snap metres in a frame), WIDE m at the puck narrowing to nothing behind. It is faint (PEAK opaque at most) so
// it does not double the motion blur's smear, shows only while the puck goes faster than SLOW..FAST m/s, and not while the
// moment stands still (pace near 0, bullet time's frozen frame).
const WINDOW = 4,
	LONGEST = 1.6,
	WIDE = 0.03,
	PEAK = 0.3,
	SLOW = 6,
	FAST = 14,
	STEPS = 16;

class Trail {
	mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshBasicMaterial>;
	private points: [number, number, number, number][] = []; // x, y, metres back from the puck, how shown

	constructor() {
		const g = new THREE.BufferGeometry(),
			index: number[] = [];
		for (let k = 0; k < STEPS; k++) index.push(2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 1, 2 * k + 3, 2 * k + 2);
		g.setIndex(index);
		g.setAttribute("position", new THREE.BufferAttribute(new Float32Array((STEPS + 1) * 6), 3).setUsage(THREE.DynamicDrawUsage));
		g.setAttribute(
			"color",
			new THREE.BufferAttribute(new Float32Array((STEPS + 1) * 8).fill(1), 4).setUsage(THREE.DynamicDrawUsage),
		);
		this.mesh = new THREE.Mesh(
			g,
			new THREE.MeshBasicMaterial({
				color: new THREE.Color().setRGB(0.05, 0.06, 0.08, THREE.SRGBColorSpace),
				vertexColors: true,
				transparent: true,
				depthWrite: false,
				side: THREE.DoubleSide,
			}),
		);
		this.mesh.frustumCulled = false;
		this.mesh.userData.ao = false; // no normals or motion of its own: the ice under it speaks for it
	}

	/** The trail behind the puck at fractional source frame f, while the moment moves `pace` source frames a timeline frame. */
	set(S: SceneData, f: number, pace: number) {
		const P = S.puck,
			n = P.xy.length,
			at = (x: number): [number, number, number] => {
				x = Math.min(Math.max(x, 0), n - 1);
				const i0 = Math.floor(x),
					i1 = Math.min(i0 + 1, n - 1),
					a = x - i0,
					p0 = P.xy[i0]!,
					p1 = P.xy[i1]!;
				return [p0[0] * (1 - a) + p1[0] * a, p0[1] * (1 - a) + p1[1] * a, P.show[i0]! * (1 - a) + P.show[i1]! * a];
			},
			head = at(f),
			back = at(f - 1),
			speed = Math.hypot(head[0] - back[0], head[1] - back[1]) * S.fps,
			opacity = PEAK * smooth(SLOW, FAST, speed) * smooth(0.05, 0.3, pace) * Math.min(head[2], 1);
		this.mesh.visible = opacity > 0.005;
		if (!this.mesh.visible) return;
		const pts = this.points;
		pts.length = 0;
		pts.push([head[0], head[1], 0, head[2]]);
		for (let k = 1; k <= STEPS; k++) {
			const [x0, y0, l0] = pts[k - 1]!,
				[x, y, show] = at(f - (WINDOW * k) / STEPS),
				d = Math.hypot(x - x0, y - y0);
			if (l0 + d < LONGEST) {
				pts.push([x, y, l0 + d, show]);
				continue;
			}
			const u = (LONGEST - l0) / d;
			pts.push([x0 + (x - x0) * u, y0 + (y - y0) * u, LONGEST, show]);
			break;
		}
		const total = pts[pts.length - 1]![2];
		if (pts.length < 2 || total < 0.05) {
			this.mesh.visible = false;
			return;
		}
		const pos = this.mesh.geometry.attributes.position as THREE.BufferAttribute,
			col = this.mesh.geometry.attributes.color as THREE.BufferAttribute;
		for (const [k, [x, y, l, show]] of pts.entries()) {
			const [xa, ya] = pts[Math.max(k - 1, 0)]!,
				[xb, yb] = pts[Math.min(k + 1, pts.length - 1)]!,
				len = Math.hypot(xb - xa, yb - ya) || 1,
				left = 1 - l / total,
				w = (WIDE / 2) * left,
				nx = (-(yb - ya) / len) * w,
				ny = ((xb - xa) / len) * w;
			pos.setXYZ(2 * k, x + nx, y + ny, 0.004);
			pos.setXYZ(2 * k + 1, x - nx, y - ny, 0.004);
			col.setW(2 * k, left * left * Math.min(show, 1));
			col.setW(2 * k + 1, left * left * Math.min(show, 1));
		}
		pos.needsUpdate = col.needsUpdate = true;
		this.mesh.geometry.setDrawRange(0, (pts.length - 1) * 6);
		this.mesh.material.opacity = opacity;
	}
}

export type SceneOptions = { spray?: boolean; progress?: (done: number, total: number) => void };

const held = new THREE.Vector3(); // the stick's grip at the later frame

export class GoalScene {
	S: SceneData;
	/** where the scene's files are */
	base = "";
	root = new THREE.Scene();
	shadowMaterial: THREE.ShaderMaterial;
	hidden = new Set<number>();
	/** the player whose eyes are the camera: their stick drawn, not their body */
	own: number | null = null;
	keepOut: THREE.Box3 | null = null;
	arena!: Arena;
	people: Person[] = [];
	/** had once every body has streamed in (each is held at its last frame in until then) */
	streamed: Promise<void> = Promise.resolve();
	/** per person, frames from each frame to the nearest one that saw them */
	unseen: Float32Array[] = [];
	sticks = new Float32Array(0);
	stickMeshes: Stick[] = [];
	/** each one's hand at the stick's top (gear.ts HANDS), when they wear gloves */
	stickHands: (number | undefined)[] = [];
	shadows: THREE.Mesh[] = [];
	/** who builds their next frame ahead first, in turn */
	spray: Spray | null = null;
	marks: Marks | null = null;
	puck!: THREE.Mesh<THREE.CylinderGeometry, THREE.MeshBasicMaterial>;
	trail = new Trail();
	private sharp = false;

	/**
	 * Everyone's body texture at twice the size (texture_<key>_2k.jpg, for the top quality setting: crisp up close, four times the
	 * memory), swapped in as each arrives; once.
	 */
	sharpen() {
		if (this.sharp) return;
		this.sharp = true;
		for (const p of this.people) {
			const map = p.mesh.material.uniforms.map!;
			if (!map.value) continue;
			void texture(`${this.base}/texture_${p.meta.key}_2k.jpg`)
				.then((t) => {
					(map.value as THREE.Texture).dispose();
					map.value = t;
				})
				.catch(() => {});
		}
	}

	static async load(base: string, { spray = true, progress = () => {} }: SceneOptions = {}) {
		const S = (await (await request(`${base}/scene.json`)).json()) as SceneData,
			s = new GoalScene(S);
		s.base = base;
		progress(0, 1);
		let started: (value?: unknown) => void = () => {};
		const [faces, sticks] = await Promise.all([bin(`${base}/faces.bin`, Uint32Array), bin(`${base}/sticks.bin`, Float32Array)]),
			arena = buildArena(S, base),
			{ people, streamed } = loadPeople(S, base, faces, sticks, inView(S), progress, new Promise((r) => (started = r)));
		s.arena = await arena;
		s.root.add(s.arena, buildGoals(S.rink));
		s.keepOut = s.arena.keepOut ?? null;
		s.people = await people;
		started();
		s.unseen = s.people.map((p) => unseen(p.meta.seen));
		// each one's gear fitted now (100 ms or so each), on a frame of the first chunk they are seen in, not when they first come on
		for (const p of s.people) {
			const fit = p.fitting;
			p.want([fit.find((f) => p.meta.seen[f]) ?? fit[0]!]);
			await new Promise((r) => setTimeout(r));
		}
		s.sticks = sticks;
		s.stickMeshes = s.people.map(() => new Stick());
		s.stickHands = s.people.map((p) => p.gear.find((g): g is Gloves => g instanceof Gloves)?.stickHand);
		s.shadows = s.people.map(() => new THREE.Mesh(new THREE.CircleGeometry(1.3, 32), s.shadowMaterial));
		for (const m of s.shadows) m.userData.wire = false; // a soft shade, not a surface: left out of the wireframe (wire.ts)
		for (const [j, p] of s.people.entries()) {
			const figure = [p.mesh, s.stickMeshes[j]!.mesh, ...p.gear.map((g) => g.group)];
			for (const o of figure) o.userData.wire = "figure"; // drawn as a figure in the wireframe (wire.ts)
			s.root.add(...figure, s.shadows[j]!);
		}
		// the skate marks and the spray once every body is in, as they are worked out over all their frames
		s.streamed = streamed.then(() => {
			s.marks = new Marks(s.people, S.fps);
			s.marks.mesh.userData.wire = false; // lines on the ice, not a surface: left out of the wireframe
			s.root.add(s.marks.mesh);
			if (!spray) return;
			s.spray = new Spray(S, s.people);
			s.root.add(s.spray.points);
		});
		s.puck = new THREE.Mesh(
			new THREE.CylinderGeometry(0.038, 0.038, 0.0254, 24),
			new THREE.MeshBasicMaterial({ color: 0x050506, transparent: true }),
		);
		s.puck.rotation.x = Math.PI / 2;
		s.trail.mesh.userData.wire = false;
		s.root.add(s.puck, s.trail.mesh);
		return s;
	}

	constructor(S: SceneData) {
		this.S = S;
		this.root.background = new THREE.Color().setRGB(0.08, 0.08, 0.09, THREE.SRGBColorSpace);
		// soft contact shadow: render.py's 1 - 0.35 exp(-d² / 0.18), as a multiplied radial falloff
		this.shadowMaterial = new THREE.ShaderMaterial({
			transparent: true,
			depthWrite: false,
			blending: THREE.MultiplyBlending,
			premultipliedAlpha: true,
			vertexShader:
				"varying vec2 q; void main() { q = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
			fragmentShader:
				"varying vec2 q; void main() { float s = 1.0 - 0.35 * exp(-dot(q, q) / 0.18); gl_FragColor = vec4(vec3(s), 1.0); }",
		});
	}

	/**
	 * Pose everything at fractional source frame t, the moment moving `pace` source frames a timeline frame there (0 frozen), and
	 * soon to be at source frames `soon`; t between source frames `keys` (built ahead, far apart in a glide), else the two around
	 * it. The bodies and the spray are CPU spans when root.userData.perf is a Perf.
	 */
	setTime(t: number, pace = 1, soon: number[] = [], keys?: [number, number]) {
		const S = this.S,
			n = S.frames,
			at = (x: number) => Math.min(Math.max(x - S.frame0, 0), n - 1),
			f = at(t);
		const i0 = keys ? Math.round(at(keys[0])) : Math.floor(f),
			i1 = keys ? Math.round(at(keys[1])) : Math.min(i0 + 1, n - 1),
			a = i1 === i0 ? 0 : (f - i0) / (i1 - i0);
		const looks = this.people.map((p, j) => {
				const u = this.unseen[j]!,
					d = a > 0 ? u[i0]! * (1 - a) + u[i1]! * a : u[i0]!, // Infinity (never seen) times 0 is NaN
					fade = 1 - smooth(HOLD, HOLD + OUT, d);
				return { ghost: smooth(0, FADE, d), fade, on: fade > 0 && !this.hidden.has(p.meta.key) };
			}),
			shown = looks.map((l) => l.on && l.ghost < 0.5),
			perf = this.root.userData.perf as Perf | undefined,
			span = (name: string, fn: () => void) => (perf ? perf.span(name, fn) : fn());
		span("bodies", () =>
			this.people.forEach((p, j) => {
				// a ghost's gear, stick and shadow only while it is still mostly solid
				const { ghost, fade, on } = looks[j]!,
					solid = shown[j]!,
					stick = this.stickMeshes[j]!,
					shadow = this.shadows[j]!,
					m = p.mesh.material,
					mine = p.meta.key === this.own;
				m.uniforms.ghost!.value = ghost;
				m.uniforms.fade!.value = fade;
				p.seeThrough = ghost > 0 || fade < 1;
				p.mesh.visible = on && !mine;
				shadow.visible = solid;
				stick.mesh.visible = solid && p.meta.stick;
				for (const g of p.gear) g.group.visible = solid && !mine;
				if (!on) return;
				p.pose(i0, i1, a);
				const [x, y] = p.root();
				shadow.position.set(x, y, 0.002);
				if (!p.meta.stick) return;
				const at = (fr: number, e: number) => {
					const o = ((j * n + fr) * 2 + e) * 3;
					return new THREE.Vector3(this.sticks[o], this.sticks[o + 1], this.sticks[o + 2]);
				};
				// held in the glove at its top (gear.ts), toward the tracked blade: the stick stays in the hand the body shows
				const heel = at(i0, 1).lerp(at(i1, 1), a),
					hand = this.stickHands[j];
				if (hand === undefined) stick.set(at(i0, 0).lerp(at(i1, 0), a), heel);
				else stick.hold(grip(p.at(i0), HANDS[hand]!).lerp(grip(p.at(i1), HANDS[hand]!, held), a), heel);
			}),
		);
		span("ahead", () => {
			// built in workers (people.ts), so crossing into them costs little: the frames shown, the AHEAD after them and those
			// around each time to be soon (across a cut, a loop or a glide)
			this.prepare(soon, [i0, i1, ...Array.from({ length: AHEAD }, (_, k) => i1 + 1 + k)]);
		});
		span("spray", () => this.spray?.set(f));
		this.marks?.set(f);
		this.arena.setTime(f); // the crowd runs on source time: slowed in slow motion, still in bullet time
		const P = S.puck,
			show = P.show[i0]! * (1 - a) + P.show[i1]! * a,
			p0 = P.xy[i0]!,
			p1 = P.xy[i1]!;
		this.puck.visible = show > 0;
		this.puck.material.opacity = Math.min(1, show);
		this.puck.position.set(p0[0] * (1 - a) + p1[0] * a, p0[1] * (1 - a) + p1[1] * a, 0.0127);
		(this.puck.userData.motion ??= new THREE.Matrix4()).set(
			0,
			0,
			0,
			p1[0] - p0[0],
			0,
			0,
			0,
			p1[1] - p0[1],
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
		); // its move in a frame
		this.trail.set(S, f, pace);
	}

	/** Have everyone's exported frames `frames` and the two around each source time in `times` built, in workers, and no others. */
	prepare(times: number[], frames: number[] = []) {
		const S = this.S,
			want = [...frames];
		for (const t of times) {
			const j = Math.min(Math.max(Math.floor(t - S.frame0), 0), S.frames - 1);
			want.push(j, j + 1);
		}
		for (const p of this.people) p.want(want);
	}

	/** Hide the people shot `sh` leaves out, and the body of its POV player (whose eyes are the camera), not their stick; none for no shot. */
	useShot(sh: Shot | undefined) {
		this.hidden = new Set(sh?.hide ?? []);
		this.own = sh?.own ?? null;
	}
}

/** Where a camera is and how it looks: three.js position, orientation and vertical field of view (degrees). */
export type Pose = { position: THREE.Vector3; quaternion: THREE.Quaternion; fov: number };

/** A planned camera's pose (fx at the scene's width, R rows: rink to OpenCV camera axes, C centre), for a frame `height` px high. */
export function cameraPose(c: Cam, height: number): Pose {
	const [r0, r1, r2] = c.R,
		m = new THREE.Matrix4().set(
			r0[0],
			-r1[0],
			-r2[0],
			c.C[0],
			r0[1],
			-r1[1],
			-r2[1],
			c.C[1],
			r0[2],
			-r1[2],
			-r2[2],
			c.C[2],
			0,
			0,
			0,
			1,
		);
	const pose: Pose = {
		position: new THREE.Vector3(),
		quaternion: new THREE.Quaternion(),
		fov: (2 * Math.atan(height / 2 / c.fx) * 180) / Math.PI,
	};
	m.decompose(pose.position, pose.quaternion, new THREE.Vector3());
	return pose;
}

/** The pose a fraction a of the way from p to q. */
export const blendPose = (p: Pose, q: Pose, a: number): Pose => ({
	position: p.position.clone().lerp(q.position, a),
	quaternion: p.quaternion.clone().slerp(q.quaternion, a),
	fov: p.fov + (q.fov - p.fov) * a,
});
