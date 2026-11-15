// The outline round one person (their body and gear), drawn over the finished frame: the person alone drawn into a mask through
// the frame's camera (their own materials, nothing else in the scene, so the line shows through whoever stands in front of them),
// then a line WIDTH CSS px wide in their team's colour along the mask's edge and a faint wash (FILL) inside it.
import * as THREE from "three";
import { FullScreenQuad } from "three/addons/postprocessing/Pass.js";

const LAYER = 31, // the camera layer the person is put on while the mask is drawn
	WIDTH = 2.5, // CSS px
	FILL = 0.1,
	TAPS = 16; // round the edge

export class Outline {
	private mask = new THREE.WebGLRenderTarget(1, 1);
	private quad: FullScreenQuad;
	private material: THREE.ShaderMaterial;
	private size = new THREE.Vector2();
	private clear = new THREE.Color();

	constructor() {
		this.material = new THREE.ShaderMaterial({
			uniforms: {
				tMask: { value: this.mask.texture },
				texel: { value: new THREE.Vector2() },
				width: { value: WIDTH },
				color: { value: new THREE.Vector3(1, 1, 1) },
				strength: { value: 1 },
			},
			vertexShader: /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
			fragmentShader: /* glsl */ `
uniform sampler2D tMask; uniform vec2 texel; uniform float width, strength; uniform vec3 color;
varying vec2 vUv;
float on(vec2 uv) { return smoothstep(0.0, 0.05, texture2D(tMask, uv).a); }
void main() {
  float inside = on(vUv), near = 0.0;
  for (int i = 0; i < ${TAPS}; i++) {
    vec2 d = vec2(cos(6.2831853 * float(i) / ${TAPS}.0), sin(6.2831853 * float(i) / ${TAPS}.0)) * texel;
    near = max(near, max(on(vUv + d * width), on(vUv + d * width * 0.5)));
  }
  gl_FragColor = vec4(color, (near * (1.0 - inside) + ${FILL.toFixed(2)} * inside) * strength);
}`,
			transparent: true,
			depthTest: false,
			depthWrite: false,
		});
		this.quad = new FullScreenQuad(this.material);
	}

	/**
	 * Draw the outline of `parts` (a person's body and gear) in `rgb` (the display's colour space, 0 to 1) at `strength` (0 to 1),
	 * over what is on the screen, through `camera`.
	 */
	draw(
		renderer: THREE.WebGLRenderer,
		scene: THREE.Scene,
		camera: THREE.Camera,
		parts: THREE.Object3D[],
		rgb: number[],
		strength: number,
	) {
		renderer.getDrawingBufferSize(this.size);
		if (this.mask.width !== this.size.x || this.mask.height !== this.size.y) this.mask.setSize(this.size.x, this.size.y);
		const marked: THREE.Object3D[] = [];
		for (const p of parts)
			p.traverse((o) => {
				if (!o.layers.isEnabled(LAYER)) {
					o.layers.enable(LAYER);
					marked.push(o);
				}
			});
		const layers = camera.layers.mask,
			background = scene.background,
			alpha = renderer.getClearAlpha(),
			autoClear = renderer.autoClear;
		renderer.getClearColor(this.clear);
		camera.layers.set(LAYER);
		scene.background = null;
		renderer.setRenderTarget(this.mask);
		renderer.setClearColor(0x000000, 0);
		renderer.clear();
		renderer.render(scene, camera);
		camera.layers.mask = layers;
		scene.background = background;
		for (const o of marked) o.layers.disable(LAYER);
		renderer.setClearColor(this.clear, alpha);
		const u = this.material.uniforms;
		u.texel!.value.set(1 / this.size.x, 1 / this.size.y);
		u.width!.value = WIDTH * renderer.getPixelRatio();
		u.color!.value.set(rgb[0], rgb[1], rgb[2]);
		u.strength!.value = strength;
		renderer.setRenderTarget(null);
		renderer.autoClear = false;
		this.quad.render(renderer);
		renderer.autoClear = autoClear;
	}

	dispose() {
		this.mask.dispose();
		this.material.dispose();
		this.quad.dispose();
	}
}
