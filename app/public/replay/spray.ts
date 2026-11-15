// Snow spray off the skates when a skater brakes hard (a hockey stop), from their own motion: the body's mean ground position,
// smoothed over SIGMA frames, braking (deceleration along the way they move, above MIN_SPEED) from BRAKE m/s² starts the spray and
// FULL gives RATE grains a frame. Each grain leaves the skates (the body's vertices within SKATE of its lowest point) forward and to
// the side, at OUT of the skater's speed (spread SPREAD rad either side) and UP m/s upward, falls, slides to a stop on the ice (FRICTION
// m/s²) and fades over LIFE s. A grain is drawn as a soft puff SIZE m across, OPACITY at most, growing by GROW as it spreads: spray
// reads as a cloud, single ice crystals would be under a pixel. Grains are seeded by skater, frame and index, so the spray at any time is the same however it is reached
// (capture frames out of order, motion blur sub-frames).
import * as THREE from "three";
import type { Person } from "./people";
import type { SceneData } from "./types";

const SPRAY = {
	SIGMA: 3,
	MIN_SPEED: 2,
	BRAKE: 6,
	FULL: 12,
	RATE: 50,
	SKATE: 0.05,
	OUT: [0.4, 1.0],
	SPREAD: 0.7,
	UP: [0.2, 0.9],
	FRICTION: 6,
	LIFE: 0.7,
	SIZE: 0.2,
	GROW: 2.0,
	OPACITY: 0.5,
	G: 9.81,
} as const;

type XY = [number, number];

function gauss(a: XY[], s: number): XY[] {
	// a: [n][2], smoothed along n (edges held)
	const r = Math.ceil(3 * s),
		w = Array.from({ length: 2 * r + 1 }, (_, k) => Math.exp(-((k - r) ** 2) / (2 * s * s)));
	return a.map((_, i) => {
		let x = 0,
			y = 0,
			t = 0;
		w.forEach((wk, k) => {
			const p = a[Math.min(Math.max(i + k - r, 0), a.length - 1)]!;
			x += wk * p[0];
			y += wk * p[1];
			t += wk;
		});
		return [x / t, y / t];
	});
}

function rng(seed: number) {
	// mulberry32
	return () => {
		seed |= 0;
		seed = (seed + 0x6d2b79f5) | 0;
		let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

type Grain = { t: number; x: number; y: number; vx: number; vy: number; vz: number };

export class Spray {
	grains: Grain[];
	fps: number;
	pos: THREE.BufferAttribute;
	alpha: THREE.BufferAttribute;
	grow: THREE.BufferAttribute;
	points: THREE.Points<THREE.BufferGeometry, THREE.ShaderMaterial>;

	constructor(S: SceneData, people: Person[]) {
		const fps = S.fps,
			grains: Grain[] = [];
		people.forEach((p) => {
			if (p.meta.goalie) return;
			// over exported frames, the body held at its ends outside its own (where no shot shows anyone)
			const V = p.V,
				n = p.frames,
				v0 = p.at(p.body.first);
			let low = Infinity;
			for (let i = 0; i < V; i++) low = Math.min(low, v0[3 * i + 2]!);
			const feet: number[] = [];
			for (let i = 0; i < V; i++) if (v0[3 * i + 2]! - low < SPRAY.SKATE * 1000) feet.push(i);
			const mean = (f: number, idx: number[]): XY => {
				const v = p.at(f);
				let x = 0,
					y = 0;
				for (const i of idx) {
					x += v[i * 3]!;
					y += v[i * 3 + 1]!;
				}
				return [x / idx.length / 1000, y / idx.length / 1000];
			};
			const all = Array.from({ length: V }, (_, i) => i);
			const r = gauss(
				Array.from({ length: n }, (_, f) => mean(f, all)),
				SPRAY.SIGMA,
			);
			const d = (a: XY[], f: number): XY => [
				((a[Math.min(f + 1, n - 1)]![0] - a[Math.max(f - 1, 0)]![0]) / 2) * fps,
				((a[Math.min(f + 1, n - 1)]![1] - a[Math.max(f - 1, 0)]![1]) / 2) * fps,
			];
			const vel = r.map((_, f) => d(r, f)),
				acc = vel.map((_, f) => d(vel, f));
			for (let f = 2; f < n - 2; f++) {
				const vf = vel[f]!,
					af = acc[f]!,
					sp = Math.hypot(...vf);
				if (sp < SPRAY.MIN_SPEED || !p.meta.seen[f]) continue;
				const u = [vf[0] / sp, vf[1] / sp] as const,
					brake = -(af[0] * u[0] + af[1] * u[1]);
				const k = Math.round(SPRAY.RATE * Math.min(Math.max((brake - SPRAY.BRAKE) / (SPRAY.FULL - SPRAY.BRAKE), 0), 1));
				if (!k) continue;
				const foot = mean(f, feet),
					rand = rng(((p.meta.key + 1) * 7919 + f) * 104729);
				for (let g = 0; g < k; g++) {
					const side = rand() < 0.5 ? -1 : 1,
						a = side * SPRAY.SPREAD * rand(),
						s = sp * (SPRAY.OUT[0] + (SPRAY.OUT[1] - SPRAY.OUT[0]) * rand());
					const c = Math.cos(a),
						sn = Math.sin(a);
					grains.push({
						t: f + rand(),
						x: foot[0] + 0.1 * (rand() - 0.5),
						y: foot[1] + 0.1 * (rand() - 0.5),
						vx: s * (u[0] * c - u[1] * sn),
						vy: s * (u[0] * sn + u[1] * c),
						vz: SPRAY.UP[0] + (SPRAY.UP[1] - SPRAY.UP[0]) * rand(),
					});
				}
			}
		});
		this.grains = grains;
		this.fps = fps;
		const g = new THREE.BufferGeometry(),
			m = Math.max(grains.length, 1);
		this.pos = new THREE.BufferAttribute(new Float32Array(m * 3), 3).setUsage(THREE.DynamicDrawUsage);
		this.alpha = new THREE.BufferAttribute(new Float32Array(m), 1).setUsage(THREE.DynamicDrawUsage);
		this.grow = new THREE.BufferAttribute(new Float32Array(m), 1).setUsage(THREE.DynamicDrawUsage);
		g.setAttribute("position", this.pos);
		g.setAttribute("alpha", this.alpha);
		g.setAttribute("grow", this.grow);
		const uniforms = { size: { value: SPRAY.SIZE as number }, height: { value: 768 } };
		this.points = new THREE.Points(
			g,
			new THREE.ShaderMaterial({
				uniforms,
				transparent: true,
				depthWrite: false,
				vertexShader: /* glsl */ `
uniform float size, height; attribute float alpha; attribute float grow; varying float vA;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0); vA = alpha;
  gl_PointSize = max(size * grow * projectionMatrix[1][1] * height / 2.0 / -mv.z, 1.0); gl_Position = projectionMatrix * mv;
}`,
				fragmentShader: /* glsl */ `
varying float vA;
void main() {
  float r = length(gl_PointCoord - 0.5) * 2.0; if (r > 1.0) discard;
  gl_FragColor = vec4(vec3(0.93, 0.95, 0.98), vA * exp(-4.0 * r * r));
  #include <colorspace_fragment>
}`,
			}),
		);
		this.points.frustumCulled = false;
		this.points.onBeforeRender = (renderer) => {
			uniforms.height.value = renderer.getRenderTarget()?.height ?? renderer.getDrawingBufferSize(new THREE.Vector2()).y;
		};
	}

	set(f: number) {
		// f: fractional exported frame
		const P = this.pos.array,
			A = this.alpha.array,
			W = this.grow.array,
			{ G, FRICTION, LIFE, GROW, OPACITY } = SPRAY;
		let m = 0;
		for (const g of this.grains) {
			const age = (f - g.t) / this.fps;
			if (age < 0 || age > LIFE) continue;
			const up = g.vz / G,
				fly = Math.min(age, 2 * up),
				z = g.vz * fly - 0.5 * G * fly * fly;
			const h = Math.hypot(g.vx, g.vy),
				slide = Math.max(age - fly, 0),
				stop = h / FRICTION;
			const along = h * fly + h * Math.min(slide, stop) - 0.5 * FRICTION * Math.min(slide, stop) ** 2,
				k = h > 0 ? along / h : 0;
			P[3 * m] = g.x + g.vx * k;
			P[3 * m + 1] = g.y + g.vy * k;
			P[3 * m + 2] = Math.max(z, 0.004);
			W[m] = 1 + ((GROW - 1) * age) / LIFE;
			A[m++] = OPACITY * (1 - age / LIFE);
		}
		this.points.geometry.setDrawRange(0, m);
		this.pos.needsUpdate = this.alpha.needsUpdate = this.grow.needsUpdate = true;
	}
}
