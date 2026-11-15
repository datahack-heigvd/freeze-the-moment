// Broadcast look: render through an EffectComposer instead of straight to the screen.
//   bloom    a light glow around the brightest parts (helmet and glass highlights), as a broadcast lens gives; from THRESHOLD up,
//            above the ice (the brightest broad surface), so the ice keeps its markings and scratches
//   neutral  Khronos PBR Neutral tone mapping: colours as they are up to the highlights, which roll off softly (the ice stays ice
//            white rather than going grey, as it does under AgX)
//   agx      AgX tone mapping at EXPOSURE, for comparison
//   ao       ground-truth ambient occlusion: contact darkening where players, boards and goals meet the ice
//   grain    faint per-frame sensor noise
//   blur     motion blur from a velocity buffer, as games do it: each pixel smeared along how far it moves on screen over the shutter
//            (the viewer's Motion), the scene rendered once
// The frame takes three full-size passes, whatever the look: the scene (multisampled), Combine (the scene made sane, darkened by
// the AO and smeared by the blur, in one pass) and Final (the bloom added, grain, tone mapping and sRGB, onto the screen). AO and
// blur share one geometry pass at AO_SCALE of the frame (GeometryPass): view normals and depth for AO, screen motion for blur, with
// people blended between their frames as they are drawn (people.ts). What has userData.ao false (and what is under it) is left out
// of it, but for a style: the nets, which are see-through, and the stands, whose rows would read as crevices and darken the crowd;
// there, as where nothing is, motion is that of the far distance (the camera turning). The bloom is built at half size from
// Combine's output and added by Final.
// Pixels that come out of the scene not finite (a NaN from a degenerate normal, say) are set to black, and the rest kept within
// [0, BRIGHTEST], before bloom, which would otherwise spread one such pixel (a few reach -50000) over a square of the frame.
// Depth of field (render's Focus): a thin lens focused at d m, whose circle of confusion at depth D is `blur` of the frame's
// width times |1 - d/D| (far behind the subject: blur; in front of it, more), at most `most` of it. Bokeh, at AO_SCALE on the
// geometry pass's depth, gathers each pixel's disk (TAPS_DOF golden-angle taps): a tap counts where its own disk reaches the pixel,
// a tap behind the pixel no wider than the pixel's own (a sharp subject's edge takes no blur from behind it); Combine mixes it in
// by how blurred the pixel is.
// Quality (QUALITY, setQuality) sets the scene's multisampling, whether there is AO, the blur's most taps and whether there is
// bloom, all without a shader of the look's changing (one compiling stalls the frame); the scale (setScale) renders everything but
// Final at a share of the canvas's size, which Final scales up, sharpened.
// Styles (setStyle; styles.ts): Final draws the frame realistic, as above, or as pixel art, an inked vector illustration or the
// minimal view of the data, from the scene, the geometry pass's normals and depth, and who is who (its normal alpha, from
// userData.styleId), a uniform's change. The styles go without the AO, take the geometry pass at the frame's full size (their
// edges come from it), and minimal and vector are smoothed by FXAA after Final.
// Overlays: the scene's children with userData.overlay (the ice graphics, ice/gl.ts) are left out of the frame and drawn over it
// once it is finished, with the geometry pass's depth (their userData.uniforms: tDepth, depthOn, near, far, viewport) to hide
// them behind what is nearer.
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { GTAOPass } from "three/addons/postprocessing/GTAOPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { FullScreenQuad, Pass } from "three/addons/postprocessing/Pass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { FXAAShader } from "three/addons/shaders/FXAAShader.js";
import type { Perf } from "./perf";
import type { GoalScene } from "./scene";
import { STYLE_GLSL, STYLES, type Style, type StyleUniforms, Styling, styleUniforms } from "./styles";

export type Look = "bloom" | "neutral" | "agx" | "ao" | "grain" | "blur";
export const LOOKS: readonly Look[] = ["bloom", "neutral", "ao", "grain"];

/** The spotlight (post.spot): the arena dimmed to `ambient` of its light (the stands to half that), a cone from `height` m over
 * the subject lighting `radius` m round it on the ice to `key` of it, as each surface faces the lamp (wrapped by `wrap`: the
 * helmet and shoulders bright, the sides falling off), the subject's shadow pooled on the ice under them (`shadow`: how dark,
 * how wide in m), its beam seen through `haze` of its light per m of air. It is
 * worked out per pixel from the geometry pass's depth, drawn at the frame's full size while it is on (no blocky edge round the
 * subject). */
const SPOT = { height: 14, radius: 3.2, ambient: 0.12, key: 1.15, haze: 0.009, wrap: 0.5, shadow: [0.6, 0.4] } as const;

export type Quality = "high" | "medium" | "low";
/** Per quality: the scene's MSAA samples, AO, the blur's most taps (at most TAPS), bloom. */
export const QUALITY: Record<Quality, { samples: number; ao: boolean; taps: number; bloom: boolean }> = {
	high: { samples: 4, ao: true, taps: 12, bloom: true },
	medium: { samples: 2, ao: true, taps: 8, bloom: true },
	low: { samples: 0, ao: false, taps: 6, bloom: false },
};

const BLOOM = { strength: 0.25, radius: 0.5, threshold: 0.97 },
	EXPOSURE = 1.15,
	GRAIN = 0.025,
	AO_SCALE = 0.5,
	AO_SAMPLES = 12,
	AO_THICKNESS = 0.1, // m assumed behind what the depth shows: thicker, ice seen low (a head camera) darkens in blocks around the near players
	TAPS = 12,
	AO_INTENSITY = 0.8,
	BRIGHTEST = 64, // linear; well over any highlight bloom picks out
	SMEAR = { most: 40, step: 2 }, // px a pixel is smeared at most, at full size; px between the blur's taps at most
	SHARPEN = 0.6, // of AMD's contrast-adaptive sharpening, at the smallest scale; none at full size
	TAPS_DOF = 20;

/** Looks named in a ?look= list. */
export const parseLook = (s: string) =>
	s.split(",").filter((x): x is Look => (["bloom", "neutral", "agx", "ao", "grain", "blur"] as string[]).includes(x));

/** How the frame moves over the shutter: the camera's view-projection as it opens and closes, and the frames of motion it spans. */
export type Motion = { open: THREE.Matrix4; close: THREE.Matrix4; span: number };

/** A lens's focus: distance d (m) along the view, the blur far behind it and the most blur, both as shares of the frame's width. */
export type Focus = { d: number; blur: number; most: number };

const ZERO = new THREE.Matrix4().set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
const QUAD =
	"varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }";

/**
 * Normals (view space, packed as MeshNormalMaterial packs them; alpha (styleId + 1) / 16, styleId the userData.styleId of the
 * object or its nearest ancestor with one, else 0), depth and screen motion (uv over the shutter) of the scene, in one render to
 * two targets. People are blended between their frames as their own material does (userData.blend; next and nextNormal),
 * and move by the difference; anything else moves by userData.motion (its change of world matrix in a frame) and the camera.
 */
class GeometryPass extends Pass {
	target = new THREE.WebGLRenderTarget(1, 1, {
		count: 2,
		type: THREE.HalfFloatType,
		minFilter: THREE.NearestFilter,
		magFilter: THREE.NearestFilter,
		depthTexture: new THREE.DepthTexture(1, 1),
	});
	motion: Motion | null = null;
	/** with what has userData.ao false (the styles' view of the arena) */
	all = false;
	/** of the frame's size it is drawn at: AO_SCALE, or 1 for the styles' edges and the spotlight */
	scale = AO_SCALE;
	private full = new THREE.Vector2(1, 1);
	private material: THREE.ShaderMaterial;
	private left: THREE.Object3D[] = [];
	private kept: boolean[] = [];
	private was: boolean[] = [];
	private vp = new THREE.Matrix4();
	private saved = new THREE.Color();

	constructor(
		private scene: THREE.Scene,
		private camera: THREE.PerspectiveCamera,
	) {
		super();
		this.needsSwap = false;
		scene.traverse((o) => {
			const points = o instanceof THREE.Points || o instanceof THREE.Line;
			if (o.userData.ao === false || points) {
				this.left.push(o);
				this.kept.push(!points);
			}
		});
		const u = {
			blend: { value: 0 },
			moving: { value: 0 },
			head: { value: new THREE.Vector3() },
			styleId: { value: 0 },
			span: { value: 0 },
			objectMotion: { value: new THREE.Matrix4() },
			openVP: { value: new THREE.Matrix4() },
			closeVP: { value: new THREE.Matrix4() },
		};
		this.material = new THREE.ShaderMaterial({
			glslVersion: THREE.GLSL3,
			uniforms: u,
			vertexShader: /* glsl */ `
attribute vec3 next, nextNormal; attribute float tuck; uniform float blend, moving, span; uniform vec3 head; uniform mat4 objectMotion, openVP, closeVP;
varying vec3 vNormal; varying vec4 vOpen, vClose;
void main() {
  float b = blend * moving;
  vec3 n = mix(normal, nextNormal, b), p = mix(position, next, b);
  if (tuck > 0.0) p -= tuck * normalize(p - head);
  vec4 w = modelMatrix * vec4(p, 1.0);
  vec3 d = (mat3(modelMatrix) * (next - position) * moving + (objectMotion * vec4(p, 1.0)).xyz) * span;
  vOpen = openVP * vec4(w.xyz - 0.5 * d, 1.0);
  vClose = closeVP * vec4(w.xyz + 0.5 * d, 1.0);
  vNormal = dot(n, n) > 0.0 ? normalMatrix * n : vec3(0.0, 0.0, 1.0);
  gl_Position = projectionMatrix * viewMatrix * w;
}`,
			fragmentShader: /* glsl */ `
layout(location = 0) out vec4 gNormal; layout(location = 1) out vec4 gMotion;
uniform float styleId; varying vec3 vNormal; varying vec4 vOpen, vClose;
void main() {
  gNormal = vec4(normalize(vNormal) * 0.5 + 0.5, (styleId + 1.0) / 16.0);
  gMotion = vec4(0.5 * (vClose.xy / vClose.w - vOpen.xy / vOpen.w), 0.0, 1.0);
}`,
		});
		this.material.onBeforeRender = (_r, _s, _c, _g, object) => {
			// per object, though one material draws them all
			const blend = object.userData.blend as number | undefined;
			u.blend.value = blend ?? 0;
			u.moving.value = blend === undefined ? 0 : 1;
			u.objectMotion.value.copy((object.userData.motion as THREE.Matrix4 | undefined) ?? ZERO);
			let o: THREE.Object3D | null = object;
			while (o && o.userData.styleId === undefined) o = o.parent;
			u.styleId.value = (o?.userData.styleId as number | undefined) ?? 0;
			this.material.uniformsNeedUpdate = true;
		};
	}

	override setSize(w: number, h: number) {
		this.full.set(w, h);
		this.fit();
	}

	/** The target at scale of the frame (the AO's setSize sizes it too, at AO_SCALE). */
	private fit() {
		const w = Math.max(1, Math.round(this.full.x * this.scale)),
			h = Math.max(1, Math.round(this.full.y * this.scale));
		if (this.target.width !== w || this.target.height !== h) this.target.setSize(w, h);
	}

	override render(renderer: THREE.WebGLRenderer) {
		this.fit();
		const { scene, camera, left, kept, was, all } = this,
			u = this.material.uniforms,
			vp = this.vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
		u.span!.value = this.motion?.span ?? 0;
		u.openVP!.value.copy(this.motion?.open ?? vp);
		u.closeVP!.value.copy(this.motion?.close ?? vp);
		const background = scene.background,
			alpha = renderer.getClearAlpha();
		renderer.getClearColor(this.saved);
		was.length = left.length;
		for (let i = 0; i < left.length; i++) {
			was[i] = left[i]!.visible;
			left[i]!.visible &&= all && kept[i]!;
		}
		scene.background = null;
		scene.overrideMaterial = this.material;
		renderer.setRenderTarget(this.target);
		renderer.setClearColor(0x7777ff, 0); // a normal facing the camera; no motion recorded (alpha 0)
		renderer.clear();
		renderer.render(scene, camera);
		renderer.setClearColor(this.saved, alpha);
		scene.overrideMaterial = null;
		scene.background = background;
		for (let i = 0; i < left.length; i++) left[i]!.visible = was[i]!;
	}

	override dispose() {
		this.target.dispose();
		this.material.dispose();
	}
}

/** The scene into its own multisampled target (samples), the one target of the frame that needs them. */
class ScenePass extends Pass {
	target: THREE.WebGLRenderTarget;

	constructor(
		private scene: THREE.Scene,
		private camera: THREE.PerspectiveCamera,
	) {
		super();
		this.needsSwap = false;
		this.target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType });
	}

	set samples(n: number) {
		if (n === this.target.samples) return;
		this.target.samples = n;
		this.target.dispose(); // made again, so multisampled, at its next use
	}

	override setSize(w: number, h: number) {
		this.target.setSize(w, h);
	}

	override render(renderer: THREE.WebGLRenderer) {
		renderer.setRenderTarget(this.target);
		renderer.clear();
		renderer.render(this.scene, this.camera);
	}

	override dispose() {
		this.target.dispose();
	}
}

/**
 * The scene's depth of field at AO_SCALE (see the top of this file): rgb the blurred scene (darkened by the AO), a how blurred.
 * Gathered, then settled: each pixel's colour a 3x3 tent of its blurred neighbours' (by how blurred), which smooths away the grain
 * of the taps' per-pixel spin while how blurred stays each pixel's own, so the edges of what is in focus stay where they are.
 */
class Bokeh extends Pass {
	target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType });
	private gathered = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType });
	uniforms = {
		tScene: { value: null as THREE.Texture | null },
		tDepth: { value: null as THREE.Texture | null },
		tAO: { value: null as THREE.Texture | null },
		darken: { value: 0 },
		focus: { value: 1 },
		blur: { value: 0 }, // px of this target, far behind the focus
		most: { value: 0 }, // px
		near: { value: 0.1 },
		far: { value: 600 },
	};
	private quad: FullScreenQuad;
	private settle = new FullScreenQuad(
		new THREE.ShaderMaterial({
			uniforms: { tGathered: { value: this.gathered.texture } },
			vertexShader: QUAD,
			fragmentShader: /* glsl */ `
uniform sampler2D tGathered;
void main() {
  ivec2 at = ivec2(gl_FragCoord.xy), last = textureSize(tGathered, 0) - 1;
  vec4 own = texelFetch(tGathered, at, 0);
  vec3 sum = vec3(0.0); float weight = 0.0;
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    vec4 b = texelFetch(tGathered, clamp(at + ivec2(x, y), ivec2(0), last), 0);
    float w = float((2 - abs(x)) * (2 - abs(y))) * b.a;
    sum += b.rgb * w; weight += w;
  }
  gl_FragColor = vec4(weight > 0.0 ? sum / weight : own.rgb, own.a);
}`,
		}),
	);

	constructor() {
		super();
		this.needsSwap = false;
		this.quad = new FullScreenQuad(
			new THREE.ShaderMaterial({
				uniforms: this.uniforms,
				vertexShader: QUAD,
				fragmentShader: /* glsl */ `
#include <packing>
uniform sampler2D tScene, tDepth, tAO; uniform float darken, focus, blur, most, near, far;
varying vec2 vUv;
${SANE}
float depthAt(vec2 uv) { return -perspectiveDepthToViewZ(texture2D(tDepth, uv).x, near, far); }
float coc(float D) { return min(0.5 * blur * abs(1.0 - focus / D), 0.5 * most); } // radius, px
vec3 shaded(vec2 uv) {
  vec3 c = sane(texture2D(tScene, uv)).rgb;
  return darken > 0.0 ? c * mix(vec3(1.0), texture2D(tAO, uv).rgb, darken) : c;
}
void main() {
  vec2 px = 1.0 / vec2(textureSize(tDepth, 0));
  float D = depthAt(vUv), own = coc(D), reach = 0.5 * most, spin = 6.2831853 * fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  vec3 sum = shaded(vUv); float weight = 1.0, blurred = own;
  for (int i = 0; i < ${TAPS_DOF}; i++) {
    float r = reach * sqrt((float(i) + 0.5) / ${TAPS_DOF}.0), a = float(i) * 2.3999632 + spin;
    vec2 uv = vUv + r * vec2(cos(a), sin(a)) * px;
    float d = depthAt(uv), c = coc(d);
    if (d > D) c = min(c, own); // behind: no wider than this pixel's own disk
    float w = clamp(c - r + 1.0, 0.0, 1.0);
    sum += shaded(uv) * w; weight += w;
    if (w > 0.0) blurred = max(blurred, c);
  }
  gl_FragColor = vec4(sum / weight, smoothstep(0.5, 1.5, blurred));
}`,
			}),
		);
	}

	override setSize(w: number, h: number) {
		for (const t of [this.target, this.gathered])
			t.setSize(Math.max(1, Math.round(w * AO_SCALE)), Math.max(1, Math.round(h * AO_SCALE)));
	}

	override render(renderer: THREE.WebGLRenderer) {
		renderer.setRenderTarget(this.gathered);
		this.quad.render(renderer);
		renderer.setRenderTarget(this.target);
		this.settle.render(renderer);
	}

	override dispose() {
		this.target.dispose();
		this.gathered.dispose();
		this.quad.dispose();
		this.settle.dispose();
	}
}

/** GLSL: a scene pixel made sane (see the top of this file). */
const SANE = `vec4 sane(vec4 c) { return any(isnan(c)) || any(isinf(c)) ? vec4(0.0, 0.0, 0.0, 1.0) : clamp(c, 0.0, ${BRIGHTEST.toFixed(1)}); }`;

/**
 * The scene made sane, darkened by the AO (AO, by darken) and, while there is motion (BLUR, moving), each pixel smeared along its
 * screen motion (GeometryPass), or that of a faster one near it whose smear reaches it: taps every SMEAR.step px at most, up to most
 * taps.
 */
function combine(ao: boolean, blur: boolean) {
	return {
		defines: { ...(ao ? { AO: 1 } : {}), ...(blur ? { BLUR: 1, TAPS } : {}) },
		uniforms: {
			darken: { value: AO_INTENSITY },
			taps: { value: TAPS },
			tBokeh: { value: null as THREE.Texture | null },
			dof: { value: 0 },
			tDiffuse: { value: null },
			tMotion: { value: null as THREE.Texture | null },
			tAO: { value: null as THREE.Texture | null },
			moving: { value: 0 },
			most: { value: SMEAR.most },
			openVP: { value: new THREE.Matrix4() },
			closeVP: { value: new THREE.Matrix4() },
			unproject: { value: new THREE.Matrix4() },
			eye: { value: new THREE.Vector3() },
			resolution: { value: new THREE.Vector2(1, 1) },
		},
		vertexShader: QUAD,
		fragmentShader: /* glsl */ `
uniform sampler2D tDiffuse, tMotion, tAO, tBokeh; uniform float moving, most, darken, dof; uniform int taps; uniform mat4 openVP, closeVP, unproject; uniform vec3 eye; uniform vec2 resolution;
varying vec2 vUv;
${SANE}
vec4 shaded(vec2 uv) {
  vec4 c = sane(texture2D(tDiffuse, uv));
#ifdef AO
  if (darken > 0.0) c.rgb *= mix(vec3(1.0), texture2D(tAO, uv).rgb, darken);
#endif
  if (dof > 0.0) { vec4 b = texture2D(tBokeh, uv); c.rgb = mix(c.rgb, b.rgb, b.a); }
  return c;
}
#ifdef BLUR
// px moved over the shutter; where the geometry pass recorded nothing, that of the far distance in this direction
vec2 motionAt(vec2 uv) {
  vec4 m = texture2D(tMotion, uv);
  if (m.a > 0.5) return m.xy * resolution;
  vec4 w = unproject * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
  vec3 d = w.xyz / w.w - eye;
  vec4 o = openVP * vec4(d, 0.0), c = closeVP * vec4(d, 0.0);
  return 0.5 * (c.xy / c.w - o.xy / o.w) * resolution;
}
#endif
void main() {
  vec4 here = shaded(vUv);
#ifdef BLUR
  if (moving < 0.5) { gl_FragColor = here; return; }
  vec2 own = motionAt(vUv), m = own;
  for (int k = 0; k < 4; k++) {
    vec2 o = (k < 2 ? vec2(k == 0 ? 1.0 : -1.0, 0.0) : vec2(0.0, k == 2 ? 1.0 : -1.0)) * most * 0.5;
    vec2 n = motionAt(vUv + o / resolution);
    if (dot(n, n) > dot(m, m)) m = n;
  }
  float len = length(m);
  if (len < 0.5) { gl_FragColor = here; return; }
  if (len > most) { m *= most / len; len = most; }
  int n = min(taps, max(2, int(ceil(len / ${SMEAR.step.toFixed(1)}))));
  float reach = 0.5 * length(own), j = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) - 0.5;
  vec4 sum = here; float weight = 1.0;
  for (int i = 0; i < TAPS; i++) {
    if (i >= n) break;
    float t = (float(i) + 0.5 + j) / float(n) - 0.5, d = abs(t) * len;
    vec2 uv = vUv + m * t / resolution;
    float w = reach >= d || 0.5 * length(motionAt(uv)) >= d ? 1.0 : 0.0;
    sum += shaded(uv) * w; weight += w;
  }
  gl_FragColor = sum / weight;
#else
  gl_FragColor = here;
#endif
}`,
	};
}

const ACROSS = new THREE.Vector2(1, 0),
	DOWN = new THREE.Vector2(0, 1);

/** UnrealBloomPass's glow, left in its first mip's target for Final to add instead of blended over the frame. */
class Bloom extends UnrealBloomPass {
	constructor(size: THREE.Vector2) {
		super(size, BLOOM.strength, BLOOM.radius, BLOOM.threshold);
		this.needsSwap = false;
	}

	get texture() {
		return this.renderTargetsHorizontal[0]!.texture;
	}

	override render(renderer: THREE.WebGLRenderer, _write: THREE.WebGLRenderTarget, read: THREE.WebGLRenderTarget) {
		// UnrealBloomPass.render up to its composite: the bright parts, blurred down the mips, the mips put together
		const own = this as unknown as {
			highPassUniforms: Record<string, THREE.IUniform>;
			materialHighPassFilter: THREE.Material;
			separableBlurMaterials: THREE.ShaderMaterial[];
			compositeMaterial: THREE.ShaderMaterial;
			bloomTintColors: THREE.Vector3[];
			_fsQuad: { material: THREE.Material; render(r: THREE.WebGLRenderer): void };
		};
		const quad = own._fsQuad,
			autoClear = renderer.autoClear,
			clear = renderer.getClearColor(new THREE.Color()),
			alpha = renderer.getClearAlpha();
		renderer.autoClear = false;
		renderer.setClearColor(this.clearColor, 0);
		own.highPassUniforms.tDiffuse!.value = read.texture;
		own.highPassUniforms.luminosityThreshold!.value = this.threshold;
		quad.material = own.materialHighPassFilter;
		renderer.setRenderTarget(this.renderTargetBright);
		renderer.clear();
		quad.render(renderer);
		let input = this.renderTargetBright;
		for (let i = 0; i < this.nMips; i++) {
			const m = own.separableBlurMaterials[i]!;
			quad.material = m;
			m.uniforms.colorTexture!.value = input.texture;
			m.uniforms.direction!.value = ACROSS;
			renderer.setRenderTarget(this.renderTargetsHorizontal[i]!);
			renderer.clear();
			quad.render(renderer);
			m.uniforms.colorTexture!.value = this.renderTargetsHorizontal[i]!.texture;
			m.uniforms.direction!.value = DOWN;
			renderer.setRenderTarget(this.renderTargetsVertical[i]!);
			renderer.clear();
			quad.render(renderer);
			input = this.renderTargetsVertical[i]!;
		}
		const c = own.compositeMaterial;
		quad.material = c;
		c.uniforms.bloomStrength!.value = this.strength;
		c.uniforms.bloomRadius!.value = this.radius;
		c.uniforms.bloomTintColors!.value = own.bloomTintColors;
		renderer.setRenderTarget(this.renderTargetsHorizontal[0]!);
		renderer.clear();
		quad.render(renderer);
		renderer.setClearColor(clear, alpha);
		renderer.autoClear = autoClear;
	}
}

/**
 * OutputPass (tone mapping and the output colour space, onto the screen) with the bloom added (bloom 1, its texture tBloom),
 * grain and, below full size (sharpen above 0), a contrast-adaptive sharpening of the frame it scales up, and the spotlight
 * (spot above 0: SPOT, from the geometry pass's depth); or, with a style (style above 0), the frame in that style.
 */
class Final extends OutputPass {
	constructor(styles: StyleUniforms) {
		super();
		Object.assign(this.uniforms, styles, {
			tBloom: { value: null },
			bloom: { value: 0 },
			grain: { value: 0 },
			seed: { value: 0 },
			sharpen: { value: 0 },
			texelSize: { value: new THREE.Vector2(1, 1) },
			spot: { value: 0 },
			spotAt: { value: new THREE.Vector4(0, 0, 0, SPOT.radius) },
			spotView: { value: new THREE.Matrix3() },
		});
		// a raw shader's samplers default to lowp, which some GPUs read at half precision: too coarse for the depth the styles use
		this.material.fragmentShader = /* glsl */ `
precision highp float; precision highp sampler2D;
uniform sampler2D tDiffuse, tBloom; uniform float bloom, grain, seed, sharpen, spot; uniform vec2 texelSize; uniform vec4 spotAt; uniform mat3 spotView;
#include <tonemapping_pars_fragment>
#include <colorspace_pars_fragment>
varying vec2 vUv;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233)) + seed) * 43758.5453); }
vec4 display(vec4 c) {                                   // tone mapped, in the output colour space
#ifdef LINEAR_TONE_MAPPING
  c.rgb = LinearToneMapping(c.rgb);
#elif defined(REINHARD_TONE_MAPPING)
  c.rgb = ReinhardToneMapping(c.rgb);
#elif defined(CINEON_TONE_MAPPING)
  c.rgb = CineonToneMapping(c.rgb);
#elif defined(ACES_FILMIC_TONE_MAPPING)
  c.rgb = ACESFilmicToneMapping(c.rgb);
#elif defined(AGX_TONE_MAPPING)
  c.rgb = AgXToneMapping(c.rgb);
#elif defined(NEUTRAL_TONE_MAPPING)
  c.rgb = NeutralToneMapping(c.rgb);
#elif defined(CUSTOM_TONE_MAPPING)
  c.rgb = CustomToneMapping(c.rgb);
#endif
#ifdef SRGB_TRANSFER
  c = sRGBTransferOETF(c);
#endif
  return c;
}
${STYLE_GLSL}
// the spotlight: a soft-edged cone down from SPOT.height over spotAt (xyz on the ice, w its radius there)
float spotCone(vec3 p) {
  float r = spotAt.w * (${SPOT.height.toFixed(1)} - p.z) / ${SPOT.height.toFixed(1)};
  return p.z < ${SPOT.height.toFixed(1)} ? 1.0 - smoothstep(0.6, 1.15, length(p.xy - spotAt.xy) / max(r, 1e-3)) : 0.0;
}
// at uv: how lit the point seen is (the arena dim, the cone's pool bright; nothing drawn by the geometry pass, the stands, darker
// still) and how many m of the view ray run through the cone's haze before it
vec2 spotlight(vec2 uv) {
  vec4 a = A(uv); vec3 eye = at(uv, -1.0), dir = normalize(at(uv, 1.0) - eye);
  bool none = a.w < -0.5; vec3 p = none ? eye + dir * 80.0 : world(uv);
  // the key: as the surface (its normal, view to world) faces the lamp, wrapped; on the ice, the subject's shadow under them
  vec3 n = normalize(spotView * vec3(a.xy, sqrt(max(1.0 - dot(a.xy, a.xy), 0.0)))), toLamp = normalize(vec3(spotAt.xy, ${SPOT.height.toFixed(1)}) - p);
  float face = clamp((dot(n, toLamp) + ${SPOT.wrap.toFixed(2)}) / ${(1 + SPOT.wrap).toFixed(2)}, 0.0, 1.0);
  vec2 off = p.xy - spotAt.xy; float shade = p.z < 0.05 ? 1.0 - ${SPOT.shadow[0].toFixed(2)} * exp(-dot(off, off) / ${(SPOT.shadow[1] ** 2).toFixed(3)}) : 1.0;
  float lit = none ? ${(SPOT.ambient * 0.5).toFixed(3)} : mix(${SPOT.ambient.toFixed(3)}, ${SPOT.key.toFixed(3)} * face * shade, spotCone(p));
  // the beam's air: thicker up toward the lamp, none within a few m of the eye (inside the beam the view stays clear)
  float len = min(length(p - eye), 60.0), j = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))), h = 0.0;
  for (int i = 0; i < 16; i++) {
    float t = len * (float(i) + j) / 16.0; vec3 q = eye + dir * t;
    h += spotCone(q) * (0.3 + 0.7 * smoothstep(0.0, ${SPOT.height.toFixed(1)}, q.z)) * smoothstep(1.0, 5.0, t);
  }
  return vec2(lit, h * len / 16.0);
}
void main() {
  if (style > 0) { gl_FragColor = vec4(styled(), 1.0); return; }
  vec4 c = texture2D(tDiffuse, vUv);
  if (sharpen > 0.0) {
    vec3 n = texture2D(tDiffuse, vUv + vec2(0.0, texelSize.y)).rgb, s = texture2D(tDiffuse, vUv - vec2(0.0, texelSize.y)).rgb,
         e = texture2D(tDiffuse, vUv + vec2(texelSize.x, 0.0)).rgb, w = texture2D(tDiffuse, vUv - vec2(texelSize.x, 0.0)).rgb;
    vec3 lo = clamp(min(c.rgb, min(min(n, s), min(e, w))), 0.0, 1.0), hi = clamp(max(c.rgb, max(max(n, s), max(e, w))), 0.0, 1.0);
    vec3 k = -sqrt(clamp(min(lo, 1.0 - hi) / max(hi, 1e-4), 0.0, 1.0)) * sharpen * 0.2;
    c.rgb = max((c.rgb + (n + s + e + w) * k) / (1.0 + 4.0 * k), 0.0);
  }
  c.rgb += bloom * texture2D(tBloom, vUv).rgb;
  if (spot > 0.0) { vec2 l = spotlight(vUv); c.rgb = c.rgb * mix(1.0, l.x, spot) + spot * l.y * ${SPOT.haze.toFixed(3)} * vec3(1.0, 0.95, 0.86); }
  c.rgb *= 1.0 + grain * (hash(gl_FragCoord.xy) - 0.5) * 2.0 * (1.0 - dot(c.rgb, vec3(0.2126, 0.7152, 0.0722)));
  gl_FragColor = display(c);
}`;
		this.material.needsUpdate = true;
	}
}

/** A render of scene through camera with the chosen look and quality, and its resize and scale. Each pass is a GPU span. */
export function broadcastLook(
	renderer: THREE.WebGLRenderer,
	scene: THREE.Scene,
	camera: THREE.PerspectiveCamera,
	look: readonly Look[],
	W: number,
	H: number,
	perf: Perf,
	quality: Quality = "high",
) {
	renderer.toneMapping = look.includes("neutral")
		? THREE.NeutralToneMapping
		: look.includes("agx")
			? THREE.AgXToneMapping
			: THREE.NoToneMapping;
	renderer.toneMappingExposure = look.includes("agx") ? EXPOSURE : 1;
	const target = new THREE.WebGLRenderTarget(W, H, { type: THREE.HalfFloatType });
	const composer = new EffectComposer(renderer, target);
	const add = (name: string, pass: Pass) => {
		const render = pass.render.bind(pass);
		pass.render = (...args) => {
			perf.gpuBegin(name);
			render(...args);
			perf.gpuEnd();
		};
		composer.addPass(pass);
	};
	let ratio = renderer.getPixelRatio(),
		scale = 1;
	composer.setPixelRatio(ratio);
	composer.setSize(W, H);
	const scenePass = new ScenePass(scene, camera);
	add("scene", scenePass);
	const wantsAO = look.includes("ao"),
		wantsBlur = look.includes("blur"),
		geometry = new GeometryPass(scene, camera); // for the AO, the blur, the depth of field and the styles: on while one of them is
	add("geometry", geometry);
	let ao: GTAOPass | null = null;
	if (wantsAO) {
		// on the geometry pass's normals and depth, not a render of its own; its result is darkened in by Combine
		ao = new GTAOPass(scene, camera, W, H);
		const own = ao as unknown as { normalRenderTarget: THREE.WebGLRenderTarget }; // its setSize sizes this
		own.normalRenderTarget.dispose();
		own.normalRenderTarget = geometry.target;
		ao.setGBuffer(geometry.target.depthTexture!, geometry.target.textures[0]!);
		ao.updateGtaoMaterial({ samples: AO_SAMPLES, thickness: AO_THICKNESS });
		ao.output = GTAOPass.OUTPUT.Off;
		ao.needsSwap = false;
		const pass = ao;
		ao.setSize = (w, h) =>
			GTAOPass.prototype.setSize.call(pass, Math.max(1, Math.round(w * AO_SCALE)), Math.max(1, Math.round(h * AO_SCALE)));
		add("ao", ao);
	}
	const bokeh = new Bokeh(),
		b = bokeh.uniforms;
	b.tScene.value = scenePass.target.texture;
	b.tDepth.value = geometry.target.depthTexture;
	b.tAO.value = ao?.pdRenderTarget.texture ?? null;
	add("dof", bokeh);
	const mix = new ShaderPass(combine(!!ao, wantsBlur), "scene"); // reads the scene's target, not the read buffer
	// set after ShaderPass copies the uniforms, which it cannot for a target's texture
	mix.uniforms.tDiffuse!.value = scenePass.target.texture;
	mix.uniforms.tAO!.value = ao?.pdRenderTarget.texture ?? null;
	mix.uniforms.tMotion!.value = geometry.target.textures[1]!;
	mix.uniforms.tBokeh!.value = bokeh.target.texture;
	add("combine", mix);
	const bloom = look.includes("bloom") ? new Bloom(new THREE.Vector2(W, H)) : null;
	if (bloom) add("bloom", bloom);
	const styles = styleUniforms(),
		final = new Final(styles);
	final.uniforms.tBloom!.value = bloom?.texture ?? null;
	final.uniforms.grain!.value = look.includes("grain") ? GRAIN : 0;
	styles.tNormal.value = geometry.target.textures[0]!;
	styles.tDepth.value = geometry.target.depthTexture;
	let styling: Styling | null = null;
	add("final", final);
	const fxaa = new ShaderPass(FXAAShader); // after Final for the minimal and vector styles, whose edges are hard
	fxaa.enabled = false;
	add("fxaa", fxaa);
	const size = new THREE.Vector2();
	/** What is drawn over the finished frame (userData.overlay), its uniforms (userData.uniforms) given the scene's depth. */
	const drawOverlays = (overlays: THREE.Object3D[]) => {
		renderer.getDrawingBufferSize(size);
		for (const o of overlays) {
			o.visible = true;
			const u = o.userData.uniforms as Record<string, THREE.IUniform> | undefined;
			if (!u) continue;
			u.tDepth!.value = geometry.target.depthTexture;
			u.depthOn!.value = 1;
			u.near!.value = camera.near;
			u.far!.value = camera.far;
			u.viewport!.value.copy(size);
		}
		const clear = renderer.autoClear;
		renderer.autoClear = false;
		renderer.setRenderTarget(null);
		for (const o of overlays) renderer.render(o, camera);
		renderer.autoClear = clear;
	};
	let frame = 0,
		shading = false; // the quality has AO, which a style goes without
	const post = {
		/** whether the look has motion blur, and so takes the frame's Motion */
		blur: wantsBlur,
		/** whether the scene's depth is drawn every frame, for what is drawn over the look (depthTexture) */
		depth: false,
		/** the spotlight: how strong (0 off to 1), on whom (a point on the ice) */
		spot: { strength: 0, at: new THREE.Vector3() },
		/** the scene's depth as last drawn (at the geometry pass's scale), while `depth` is on */
		get depthTexture() {
			return geometry.target.depthTexture!;
		},
		quality,
		setQuality(to: Quality) {
			const q = QUALITY[to];
			post.quality = to;
			scenePass.samples = q.samples;
			shading = !!ao && q.ao;
			mix.uniforms.taps!.value = q.taps;
			if (bloom) bloom.enabled = q.bloom;
			final.uniforms.bloom!.value = bloom && q.bloom ? 1 : 0;
		},
		/** Draw the frames of `scene` in `style`. */
		setStyle(style: Style, of: GoalScene) {
			styles.style.value = STYLES.indexOf(style);
			fxaa.enabled = style === "minimal" || style === "vector";
			if (style !== "realistic") styling ??= new Styling(of, styles);
		},
		/** The frame at source time `time`, blurred by `motion` (with the blur look) and by the lens at `focus`. */
		render(motion: Motion | null = null, focus: Focus | null = null, time = 0) {
			const u = mix.uniforms,
				read = composer.readBuffer,
				lens = focus,
				styled = styles.style.value > 0,
				overlays = scene.children.filter((o) => o.userData.overlay && o.visible);
			const spot = post.spot.strength > 0 && !styled;
			geometry.enabled = shading || wantsBlur || !!lens || styled || spot || post.depth || overlays.length > 0;
			// the styles' edges and the spotlight's edge round whoever stands in it are drawn per pixel: the depth at the frame's size
			geometry.scale = styled || spot ? 1 : AO_SCALE;
			final.uniforms.spot!.value = spot ? post.spot.strength : 0;
			if (spot) {
				final.uniforms.spotAt!.value.set(post.spot.at.x, post.spot.at.y, 0, SPOT.radius);
				final.uniforms.spotView!.value.setFromMatrix4(camera.matrixWorld);
				styles.unproject.value.multiplyMatrices(camera.matrixWorld, camera.projectionMatrixInverse);
				styles.clip.value.set(camera.near, camera.far);
			}
			geometry.all = styled;
			if (ao) ao.enabled = shading && !styled;
			u.darken!.value = b.darken.value = shading && !styled ? AO_INTENSITY : 0;
			bokeh.enabled = !!lens;
			u.dof!.value = lens ? 1 : 0;
			if (lens) {
				const w = bokeh.target.width;
				b.focus.value = lens.d;
				b.blur.value = lens.blur * w;
				b.most.value = lens.most * w;
				b.near.value = camera.near;
				b.far.value = camera.far;
			}
			final.uniforms.seed!.value = (frame++ % 997) * 0.37;
			final.uniforms.sharpen!.value = scale < 1 ? (SHARPEN * (1 - scale)) / 0.5 : 0;
			final.uniforms.texelSize!.value.set(1 / read.width, 1 / read.height);
			geometry.motion = motion;
			// Final draws onto the screen, or, before the FXAA, into a target of the frame's (scaled) size
			if (fxaa.enabled) fxaa.uniforms.resolution!.value.set(1 / read.width, 1 / read.height);
			if (styled)
				styling?.update(
					time,
					camera,
					fxaa.enabled ? size.set(read.width, read.height) : renderer.getDrawingBufferSize(size),
					geometry.target,
				);
			u.moving!.value = wantsBlur && motion ? 1 : 0;
			if (wantsBlur && motion) {
				u.openVP!.value.copy(motion.open);
				u.closeVP!.value.copy(motion.close);
				u.unproject!.value.multiplyMatrices(camera.matrixWorld, camera.projectionMatrixInverse);
				u.eye!.value.setFromMatrixPosition(camera.matrixWorld);
				u.resolution!.value.set(read.width, read.height);
				u.most!.value = SMEAR.most * (read.width / Math.max(renderer.getDrawingBufferSize(size).x, 1));
			}
			for (const o of overlays) o.visible = false;
			composer.render();
			if (overlays.length) drawOverlays(overlays);
		},
		setSize(w: number, h: number) {
			W = w;
			H = h;
			ratio = renderer.getPixelRatio();
			composer.setPixelRatio(ratio * scale);
			composer.setSize(w, h);
		},
		/** Render at s (at most 1) of the canvas's size, scaled up by Final. */
		setScale(s: number) {
			scale = Math.min(Math.max(s, 0.25), 1);
			composer.setPixelRatio(ratio * scale);
			composer.setSize(W, H);
		},
		dispose() {
			composer.dispose();
			target.dispose();
		},
	};
	post.setQuality(quality);
	return post;
}
