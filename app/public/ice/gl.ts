// The shapes (shapes.ts) as three.js meshes: laid out on the CPU through the camera each update, in its normalised device
// coordinates, so they draw with whatever camera renders them as they were laid out (its view offset included), over whatever is
// there: transparent, no depth test or write. Ribbons smooth their own edges (their half width carried to the fragment, half a
// device px of ramp each side); areas have none. Occlusion is the shader's shown(wp), wp the rink point under the fragment: by
// default hidden where the scene's depth (tDepth, perspective, near and far) is nearer than wp by more than BIAS m, when depthOn;
// a caller's own (buildIce's shown) can use its uniforms as it likes. Two meshes, two materials: on the ice (onIce 1) and in the
// air (onIce 0), sharing every other uniform.
import * as THREE from "three";
import type { Element, Project } from "./model";
import { layout, type Rink, type Shape } from "./shapes";

const BIAS = 0.1;

/** rink metres to CSS px over a `size` (CSS px) view through camera (its projection matrix, view offset and all) */
export function projectThrough(camera: THREE.Camera, size: THREE.Vector2): Project {
	const m = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse),
		v = new THREE.Vector4();
	return (x, y, z = 0) => {
		v.set(x, y, z, 1).applyMatrix4(m);
		if (v.w < 1e-3) return null;
		return [((v.x / v.w + 1) / 2) * size.x, ((1 - v.y / v.w) / 2) * size.y];
	};
}

const DEFAULT_SHOWN = /* glsl */ `
float shown(vec3 wp) {
  if (depthOn < 0.5) return 1.0;
  float d = texture2D(tDepth, gl_FragCoord.xy / viewport).x, scene = near * far / (far - d * (far - near));
  return step(-(viewMatrix * vec4(wp, 1.0)).z, scene + ${BIAS.toFixed(2)});
}`;

export type IceOptions = { uniforms?: Record<string, THREE.IUniform>; shown?: string };

/**
 * The language's meshes, in `group`: `update(elements, camera, size, dpr, rink)` lays a frame out (size in CSS px of the view the
 * camera renders, dpr its device pixel ratio); `uniforms` holds opacity, viewport (device px), tDepth, depthOn, near and far,
 * and any of the caller's.
 */
export function buildIce({ uniforms: own = {}, shown = DEFAULT_SHOWN }: IceOptions = {}) {
	const uniforms: Record<string, THREE.IUniform> = {
		opacity: { value: 1 },
		viewport: { value: new THREE.Vector2(1, 1) },
		tDepth: { value: null },
		depthOn: { value: 0 },
		near: { value: 0.1 },
		far: { value: 600 },
		...own,
	};
	const material = (onIce: number) =>
		new THREE.ShaderMaterial({
			uniforms: { ...uniforms, onIce: { value: onIce } },
			transparent: true,
			side: THREE.DoubleSide, // the shapes wind either way
			depthTest: false,
			depthWrite: false,
			vertexShader: /* glsl */ `
attribute vec3 wp; attribute vec4 rgba; attribute vec2 edge;
varying vec3 vWp; varying vec4 vRgba; varying vec2 vEdge;
void main() { vWp = wp; vRgba = rgba; vEdge = edge; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
			fragmentShader: /* glsl */ `
uniform float opacity, onIce, depthOn, near, far; uniform vec2 viewport; uniform sampler2D tDepth;
varying vec3 vWp; varying vec4 vRgba; varying vec2 vEdge;
${shown}
void main() {
  float cover = vEdge.y > 0.0 ? clamp(vEdge.y + 0.5 - abs(vEdge.x), 0.0, 1.0) : 1.0;   // a ribbon's edges, smoothed over a px
  gl_FragColor = vec4(vRgba.rgb, vRgba.a * cover * opacity * shown(vWp));
}`,
		});
	const meshes = [1, 0].map((onIce) => {
		const m = new THREE.Mesh(new THREE.BufferGeometry(), material(onIce));
		m.frustumCulled = false;
		m.renderOrder = onIce ? 0 : 1;
		return m;
	});
	const group = new THREE.Group();
	group.add(...meshes);
	const size = new THREE.Vector2();
	return {
		group,
		uniforms,
		/** The shapes of `elements` laid out through camera, for a view `css` px at device pixel ratio `dpr`. */
		update(elements: Element[], camera: THREE.Camera, css: THREE.Vector2, dpr: number, rink: Rink) {
			camera.updateMatrixWorld();
			uniforms.viewport!.value.set(css.x * dpr, css.y * dpr);
			const shapes = layout(elements, projectThrough(camera, size.copy(css)), rink);
			for (const [k, m] of meshes.entries())
				fill(
					m.geometry,
					shapes.filter((s) => s.ice === (k === 0)),
					css,
					dpr,
				);
			return shapes;
		},
		dispose() {
			for (const m of meshes) {
				m.geometry.dispose();
				(m.material as THREE.Material).dispose();
			}
			group.removeFromParent();
		},
	};
}

/** The attributes a geometry keeps from frame to frame, with room for `cap` vertices and `icap` indices. */
type Buffers = {
	cap: number;
	icap: number;
	pos: THREE.BufferAttribute;
	wp: THREE.BufferAttribute;
	col: THREE.BufferAttribute;
	edge: THREE.BufferAttribute;
	index: THREE.BufferAttribute;
};

/**
 * The shapes' triangles into geometry g: positions in NDC, each ribbon half a device px wider for its smoothing. Its attributes
 * are kept and only what is drawn is sent each frame; they are made anew, half as large again, only when the shapes outgrow them.
 */
function fill(g: THREE.BufferGeometry, shapes: Shape[], css: THREE.Vector2, dpr: number) {
	let n = 0,
		m = 0;
	for (const s of shapes) {
		n += s.v.length;
		m += s.tris.length;
	}
	let b = g.userData.buffers as Buffers | undefined;
	if (!b || b.cap < n || b.icap < m) {
		const cap = Math.max(Math.ceil(n * 1.5), 256),
			icap = Math.max(Math.ceil(m * 1.5), 512),
			made = (k: number) => new THREE.BufferAttribute(new Float32Array(cap * k), k).setUsage(THREE.DynamicDrawUsage);
		g.dispose(); // the old buffers freed on the GPU; the geometry is set up again when next drawn
		b = {
			cap,
			icap,
			pos: made(3),
			wp: made(3),
			col: made(4),
			edge: made(2),
			index: new THREE.BufferAttribute(new Uint32Array(icap), 1).setUsage(THREE.DynamicDrawUsage),
		};
		g.setAttribute("position", b.pos);
		g.setAttribute("wp", b.wp);
		g.setAttribute("rgba", b.col);
		g.setAttribute("edge", b.edge);
		g.setIndex(b.index);
		g.userData.buffers = b;
	}
	const pos = b.pos.array as Float32Array,
		wp = b.wp.array as Float32Array,
		col = b.col.array as Float32Array,
		edge = b.edge.array as Float32Array,
		index = b.index.array as Uint32Array;
	let at = 0,
		i = 0;
	for (const s of shapes) {
		for (const t of s.tris) index[i++] = at + t;
		const [r, gr, bl, a] = s.rgba;
		for (const v of s.v) {
			const h = v.h > 0 ? v.h + 0.5 / dpr : 0;
			pos[3 * at] = ((v.x + v.nx * v.s * h) / css.x) * 2 - 1;
			pos[3 * at + 1] = 1 - ((v.y + v.ny * v.s * h) / css.y) * 2;
			pos[3 * at + 2] = 0;
			wp[3 * at] = v.w[0];
			wp[3 * at + 1] = v.w[1];
			wp[3 * at + 2] = v.w[2];
			col[4 * at] = r;
			col[4 * at + 1] = gr;
			col[4 * at + 2] = bl;
			col[4 * at + 3] = a;
			edge[2 * at] = v.s * (v.h * dpr + 0.5);
			edge[2 * at + 1] = v.h * dpr;
			at++;
		}
	}
	for (const [attr, k] of [
		[b.pos, 3],
		[b.wp, 3],
		[b.col, 4],
		[b.edge, 2],
	] as const) {
		attr.clearUpdateRanges();
		attr.addUpdateRange(0, n * k);
		attr.needsUpdate = true;
	}
	b.index.clearUpdateRanges();
	b.index.addUpdateRange(0, m);
	b.index.needsUpdate = true;
	g.setDrawRange(0, m);
}
