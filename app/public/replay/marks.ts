// Skate marks: the lines everyone's blades cut in the ice as they skate, from their own motion. Each foot's track is its ankle
// ring's centre (skates.ts FEET) over the exported frames, smoothed over SIGMA frames; between two frames that both have that foot
// down (its ankle within LIFT of the lowest it gets) and that it moved between (MOVE m at least, under JUMP m/s: faster is a
// track's jump, not a stride), the blade cut a curve through them (Catmull-Rom, in SUB pieces). The lines build up as the moment
// plays (each drawn from its frame on, so the ice at any time is the same however it is reached) and stay: frost scratched up
// pale, WIDTH across, OPACITY at most. Far off a line is drawn a pixel wide and fainter by as much (no shimmer from lines under a
// pixel).
import * as THREE from "three";
import type { Person } from "./people";
import { FEET } from "./skates";

const MARKS = { SIGMA: 1, SUB: 4, LIFT: 0.045, MOVE: 0.01, JUMP: 12, WIDTH: 0.01, OPACITY: 0.32, Z: 0.0015 } as const;

/** Catmull-Rom between b and c (a before, d after) at u in [0, 1]. */
const spline = (a: number, b: number, c: number, d: number, u: number) =>
	b + 0.5 * u * (c - a + u * (2 * a - 5 * b + 4 * c - d + u * (3 * (b - c) + d - a)));

/** xy of each point, smoothed along the track over sigma points (ends held). */
function smoothed(at: (readonly [number, number, number])[], sigma: number) {
	const r = Math.ceil(3 * sigma),
		n = at.length;
	return at.map((_, i) => {
		let x = 0,
			y = 0,
			t = 0;
		for (let k = -r; k <= r; k++) {
			const w = Math.exp(-(k * k) / (2 * sigma * sigma)),
				p = at[Math.min(Math.max(i + k, 0), n - 1)]!;
			x += w * p[0];
			y += w * p[1];
			t += w;
		}
		return [x / t, y / t] as const;
	});
}

export class Marks {
	mesh: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>;

	constructor(people: Person[], fps: number) {
		const pos: number[] = [],
			side: number[] = [],
			edge: number[] = [],
			frame: number[] = [],
			idx: number[] = [];
		for (const p of people) {
			const n = p.frames;
			for (const foot of FEET) {
				const at = Array.from({ length: n }, (_, f) => {
					const v = p.at(f);
					let x = 0,
						y = 0,
						z = 0;
					for (const i of foot.ankle) {
						x += v[3 * i]!;
						y += v[3 * i + 1]!;
						z += v[3 * i + 2]!;
					}
					const k = foot.ankle.length * 1000;
					return [x / k, y / k, z / k] as const;
				});
				const low = at.map((q) => q[2]).sort((a, b) => a - b)[Math.floor(n * 0.05)] ?? 0,
					down = (f: number) => p.meta.seen[f] && at[f]![2] - low < MARKS.LIFT,
					track = smoothed(at, MARKS.SIGMA),
					q = (f: number) => track[Math.min(Math.max(f, 0), n - 1)]!;
				for (let f = 0; f + 1 < n; f++) {
					const [x0, y0] = q(f),
						[x1, y1] = q(f + 1);
					const d = Math.hypot(x1 - x0, y1 - y0);
					if (!down(f) || !down(f + 1) || d < MARKS.MOVE || d * fps > MARKS.JUMP) continue;
					const a = q(f - 1),
						e = q(f + 2),
						pts = Array.from({ length: MARKS.SUB + 1 }, (_, k) => {
							const u = k / MARKS.SUB;
							return [spline(a[0], x0, x1, e[0], u), spline(a[1], y0, y1, e[1], u)] as const;
						});
					for (let k = 0; k < MARKS.SUB; k++) {
						const [xa, ya] = pts[k]!,
							[xb, yb] = pts[k + 1]!,
							l = Math.hypot(xb - xa, yb - ya) || 1,
							nx = -(yb - ya) / l,
							ny = (xb - xa) / l,
							b = pos.length / 3;
						for (const [x, y, t] of [
							[xa, ya, f + k / MARKS.SUB],
							[xb, yb, f + (k + 1) / MARKS.SUB],
						] as const)
							for (const s of [-1, 1]) {
								pos.push(x, y, MARKS.Z);
								side.push(s * nx, s * ny);
								edge.push(s);
								frame.push(t);
							}
						idx.push(b, b + 2, b + 1, b + 1, b + 2, b + 3);
					}
				}
			}
		}
		const geo = new THREE.BufferGeometry();
		geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
		geo.setAttribute("side", new THREE.Float32BufferAttribute(side, 2));
		geo.setAttribute("edge", new THREE.Float32BufferAttribute(edge, 1));
		geo.setAttribute("frame", new THREE.Float32BufferAttribute(frame, 1));
		geo.setIndex(idx);
		const uniforms = { now: { value: 0 }, height: { value: 768 } };
		this.mesh = new THREE.Mesh(
			geo,
			new THREE.ShaderMaterial({
				uniforms,
				transparent: true,
				depthWrite: false,
				polygonOffset: true,
				polygonOffsetFactor: -2,
				vertexShader: /* glsl */ `
uniform float now, height; attribute vec2 side; attribute float edge, frame; varying float vA, vE;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float px = 2.0 * -mv.z / (projectionMatrix[1][1] * height), w = max(${MARKS.WIDTH.toFixed(4)}, px);   // m a pixel spans there
  vA = frame > now ? 0.0 : ${MARKS.OPACITY.toFixed(3)} * ${MARKS.WIDTH.toFixed(4)} / w; vE = edge;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position + vec3(side * w * 0.5, 0.0), 1.0);
}`,
				fragmentShader: /* glsl */ `
varying float vA, vE;
void main() {
  if (vA <= 0.0) discard;
  gl_FragColor = vec4(vec3(0.97, 0.98, 1.0), vA * smoothstep(1.0, 0.3, abs(vE)));   // soft along its edges
  #include <colorspace_fragment>
}`,
			}),
		);
		this.mesh.frustumCulled = false;
		this.mesh.userData.ao = false; // no normals or motion of its own: the ice under it speaks for it
		this.mesh.onBeforeRender = (renderer) => {
			uniforms.height.value = renderer.getRenderTarget()?.height ?? renderer.getDrawingBufferSize(new THREE.Vector2()).y;
		};
	}

	/** The marks cut by fractional exported frame f. */
	set(f: number) {
		this.mesh.material.uniforms.now!.value = f;
	}
}
