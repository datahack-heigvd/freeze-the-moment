// The "How it works" page's 3D: the goal's scene (replay/scene.ts) seen through the broadcast's own camera, frame by frame, either
// alone over the picture (bodies, sticks, the puck: transparent elsewhere) or whole with the replay's look; then flown out of the
// broadcast's view and orbited freely.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Perf } from "../replay/perf";
import { broadcastLook, LOOKS } from "../replay/post";
import { GoalScene, type Pose } from "../replay/scene";
import { blendPose, cameraAt, GOAL, H, type How, poseOf, W } from "./data";

/** What is drawn. whole: the scene with its arena and the replay's look; else only the people's layers, over the picture. */
export type Layers = { whole: boolean; people: boolean; sticks: boolean; puck: boolean };

const FLIGHT = 3.2; // s
const SLOW = 0.3; // the moment's pace while flying, of real time
const AROUND = { eye: new THREE.Vector3(-15.5, -8.5, 4.2), target: new THREE.Vector3(-24, 0.6, 0.7), fov: 42 };
const ease = (x: number) => (x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2);

export class World {
	scene: GoalScene;
	private how: How;
	private renderer: THREE.WebGLRenderer;
	private camera = new THREE.PerspectiveCamera(30, W / H, 0.1, 600);
	private controls: OrbitControls;
	private post: ReturnType<typeof broadcastLook>;
	private resize: ResizeObserver;
	private background: THREE.Color | THREE.Texture | null;
	/** the arena's own: the goals and anything else not a person's, shown with the arena */
	private props: THREE.Object3D[];
	private layers: Layers = { whole: false, people: true, sticks: true, puck: true };
	private time = 0;
	private pace = 0;
	private hover: number | null = null;
	private dirty = true;
	/** off the broadcast's camera: flown out or orbited */
	free = false;
	private flight: { from: Pose; to: Pose; t0: number; from_t: number; to_t: number; done: () => void } | null = null;
	onTime: (t: number) => void = () => {};

	static async open(container: HTMLElement, how: How, progress: (done: number, total: number) => void) {
		return new World(container, how, await GoalScene.load(`/scenes/${GOAL}`, { progress }));
	}

	private constructor(container: HTMLElement, how: How, scene: GoalScene) {
		this.how = how;
		this.scene = scene;
		this.background = scene.root.background;
		const s = scene,
			own = new Set<THREE.Object3D>([
				s.arena,
				s.puck,
				s.trail.mesh,
				...s.people.flatMap((p) => [p.mesh, ...p.gear.map((g) => g.group)]),
				...s.stickMeshes.map((k) => k.mesh),
				...s.shadows,
			]);
		this.props = s.root.children.filter((o) => !own.has(o));
		this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
		this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
		this.renderer.domElement.className = "absolute inset-0 block size-full";
		container.append(this.renderer.domElement);
		const { width, height } = this.size(container);
		this.renderer.setSize(width, height, false);
		const perf = new Perf(this.renderer.getContext() as WebGL2RenderingContext);
		this.post = broadcastLook(this.renderer, s.root, this.camera, LOOKS, width, height, perf, width < 640 ? "medium" : "high");
		this.camera.up.set(0, 0, 1);
		this.controls = new OrbitControls(this.camera, this.renderer.domElement);
		Object.assign(this.controls, {
			enabled: false,
			enableDamping: true,
			dampingFactor: 0.08,
			minDistance: 3,
			maxDistance: 70,
			maxPolarAngle: THREE.MathUtils.degToRad(84),
		});
		this.controls.addEventListener("change", () => (this.dirty = true));
		this.resize = new ResizeObserver(() => {
			const { width, height } = this.size(container);
			this.renderer.setSize(width, height, false);
			this.post.setSize(width, height);
			this.dirty = true;
		});
		this.resize.observe(container);
		void this.renderer
			.compileAsync(s.root, this.camera)
			.catch(() => {})
			.then(() => this.renderer.setAnimationLoop((now) => this.frame(now)));
	}

	private size(container: HTMLElement) {
		return { width: Math.max(container.clientWidth, 1), height: Math.max(container.clientHeight, 1) };
	}

	get canvas() {
		return this.renderer.domElement;
	}

	set(layers: Layers) {
		this.layers = layers;
		this.dirty = true;
	}

	/** Show scene frame t (fractional), the moment moving at pace (0 still). */
	setTime(t: number, pace = 0) {
		if (this.flight) return;
		this.time = t;
		this.pace = pace;
		this.dirty = true;
	}

	setHover(key: number | null) {
		this.hover = key;
		this.dirty = true;
	}

	/** Person key's stick at scene frame f: top hand and blade heel (m), or null without one. */
	stick(key: number, f: number): [THREE.Vector3, THREE.Vector3] | null {
		const s = this.scene,
			j = s.people.findIndex((p) => p.meta.key === key),
			n = s.S.frames,
			i = Math.min(Math.max(Math.round(f - s.S.frame0), 0), n - 1);
		if (j < 0 || !s.people[j]!.meta.stick) return null;
		const end = (e: number) => {
			const o = ((j * n + i) * 2 + e) * 3;
			return new THREE.Vector3(s.sticks[o], s.sticks[o + 1], s.sticks[o + 2]);
		};
		const top = end(0),
			heel = end(1);
		return Number.isFinite(top.x) && Number.isFinite(heel.x) ? [top, heel] : null;
	}

	/** The broadcast's camera at fractional scene frame t. */
	private broadcast(t: number): Pose {
		const [a, b] = this.how.frames,
			c = Math.min(Math.max(t, a), b),
			i = Math.floor(c),
			j = Math.min(i + 1, b),
			p = poseOf(this.how.cams[i - a]!);
		return j === i ? p : blendPose(p, poseOf(this.how.cams[j - a]!), c - i);
	}

	/** Fly out of the broadcast's view to one beside the net, the moment going on slowly to the puck in the net; then free to orbit. */
	fly(done: () => void = () => {}) {
		const to = cameraAt(this.broadcast(this.time), new THREE.PerspectiveCamera());
		to.position.copy(AROUND.eye);
		to.up.set(0, 0, 1);
		to.lookAt(AROUND.target);
		const shot = this.how.puck.shot;
		this.free = true;
		this.controls.enabled = false;
		this.flight = {
			from: this.pose(),
			to: { position: to.position.clone(), quaternion: to.quaternion.clone(), fov: AROUND.fov },
			t0: performance.now(),
			from_t: Math.min(this.time, shot.release - 6),
			to_t: shot.release - 6 + FLIGHT * this.how.fps * SLOW,
			done: () => {
				this.controls.target.copy(AROUND.target);
				this.controls.enabled = true;
				this.controls.update();
				done();
			},
		};
	}

	/** Fly back into the broadcast's camera at the time shown. */
	land(done: () => void = () => {}) {
		this.controls.enabled = false;
		this.flight = {
			from: this.pose(),
			to: this.broadcast(this.time),
			t0: performance.now(),
			from_t: this.time,
			to_t: this.time,
			done: () => {
				this.free = false;
				done();
			},
		};
	}

	private pose(): Pose {
		return { position: this.camera.position.clone(), quaternion: this.camera.quaternion.clone(), fov: this.camera.fov };
	}

	private frame(now: number) {
		const f = this.flight;
		if (f) {
			const x = Math.min((now - f.t0) / 1000 / FLIGHT, 1),
				e = ease(x);
			cameraAt(blendPose(f.from, f.to, e), this.camera);
			this.time = f.from_t + (f.to_t - f.from_t) * x;
			this.pace = f.to_t === f.from_t ? 0 : SLOW;
			this.onTime(this.time);
			this.dirty = true;
			if (x >= 1) {
				this.flight = null;
				f.done();
			}
		} else if (this.free) this.controls.update();
		else cameraAt(this.broadcast(this.time), this.camera);
		if (!this.dirty) return;
		this.dirty = false;
		this.draw();
	}

	private draw() {
		const s = this.scene,
			L = this.layers;
		s.setTime(this.time, this.pace);
		// what setTime shows, less what this step leaves out
		s.arena.visible = L.whole;
		for (const o of this.props) o.visible = L.whole;
		s.root.background = L.whole ? this.background : null;
		if (s.spray) s.spray.points.visible &&= L.whole && L.people;
		s.trail.mesh.visible &&= L.whole && L.puck;
		s.puck.visible &&= L.puck;
		for (const [j, p] of s.people.entries()) {
			// everyone but the one picked thinned out, as the replay draws those it does not see
			const lit = this.hover === null || this.hover === p.meta.key,
				u = p.mesh.material.uniforms;
			p.mesh.visible &&= L.people;
			if (!lit) u.ghost!.value = Math.max(u.ghost!.value as number, 0.92);
			for (const g of p.gear) g.group.visible &&= L.people && lit;
			s.stickMeshes[j]!.mesh.visible &&= L.people && L.sticks;
			s.shadows[j]!.visible &&= L.whole && L.people;
		}
		if (L.whole) this.post.render();
		else {
			this.renderer.setRenderTarget(null);
			this.renderer.setClearColor(0x000000, 0);
			this.renderer.clear();
			this.renderer.render(s.root, this.camera);
		}
	}

	dispose() {
		this.renderer.setAnimationLoop(null);
		this.resize.disconnect();
		this.controls.dispose();
		this.post.dispose();
		this.renderer.dispose();
		this.renderer.domElement.remove();
	}
}
