// Styles of the broadcast look (post.ts), drawn by Final after its tone mapping, on display colours; switching is a uniform (style),
// never a recompile:
//   realistic  the frame as it is
//   pixel      320 px sprite art in the PICO-8 palette, ordered dithering, outlined
//   vector     an inked illustration: four muted tones a colour (Kuwahara-flattened), bold outlines round the people, thin inside
//   minimal    the system's view underneath the rendering: each player a solid figure in their team's kit colour, shaded and
//              outlined, officials grey; light cool ice with a 1 m and 5 m grid from the centre and the rink's lines drawn crisply
//              from its geometry (blue lines 18.26 m inside the goal lines, as measured on the scenes' ice atlas); boards, sponsors,
//              glass and goals subdued in cool grey, the stands darker; the puck dark, ringed in orange. With TRACKS: jersey-number
//              tags, velocity arrows on the ice, people hollow in frames the broadcast does not see them, the puck's trail while no
//              one carries it
// They read who is where from the geometry pass: its normal target's alpha holds (styleId + 1) / 16 (0 where nothing is), styleId
// the userData.styleId of each object or its nearest ancestor with one (Styling: the teams, the officials, the puck; 5 more while
// unseen, with TRACKS; OVERLAY_ID drawings), and its depth.
import * as THREE from "three";
import type { GoalScene } from "./scene";
import { param } from "./util";

export type Style = "realistic" | "pixel" | "vector" | "minimal";
export const STYLES: readonly Style[] = ["realistic", "pixel", "vector", "minimal"];

/** minimal with its tracks (number tags, velocity arrows, the unseen hollow, the puck's trail) or without */
const TRACKS = false;

/** the styleId of drawings on the ice (tactical.ts), which every style shows as they are rendered */
export const OVERLAY_ID = 10;

const ID = { team0: 1, team1: 2, official: 3, puck: 4 },
	LAG = 0.3, // s the velocity is taken over
	CARRY = 1.5, // m from the puck within which a skater carries it
	HEAD = 1.7, // m up where a tag hangs
	SRGB = THREE.SRGBColorSpace;

const idOf = (team: number) => [ID.official, ID.team0, ID.team1][team + 1]!;

/** Final's uniforms for the styles. */
export const styleUniforms = () => ({
	style: { value: 0 }, // index in STYLES
	debug: { value: 0 }, // minimal's terms per pixel instead of the frame (?styledebug=1)
	tNormal: { value: null as THREE.Texture | null },
	tDepth: { value: null as THREE.Texture | null },
	unproject: { value: new THREE.Matrix4() }, // camera.matrixWorld * projectionMatrixInverse
	clip: { value: new THREE.Vector2(0.05, 600) }, // camera near, far
	res: { value: new THREE.Vector2(1, 1) }, // drawing buffer px
	gres: { value: new THREE.Vector2(1, 1) }, // the geometry pass's px
	kit1: { value: new THREE.Vector3(1, 1, 1) },
	kit2: { value: new THREE.Vector3(1, 1, 1) },
	rink: { value: new THREE.Vector3(30, 15, 8.5) }, // half length, half width, corner radius (m)
	goalX: { value: 26 },
	puck: { value: new THREE.Vector3() }, // screen px, 1 when on screen
	feet: { value: Array.from({ length: 24 }, () => new THREE.Vector3()) }, // each contact shadow's centre (m), 1 while shown
	skater: { value: Array.from({ length: 24 }, () => new THREE.Vector4()) },
	kind: { value: new Array<number>(24).fill(0) },
	label: { value: Array.from({ length: 24 }, () => new THREE.Vector4()) },
	trail: { value: Array.from({ length: 16 }, () => new THREE.Vector2()) },
	trailOn: { value: 0 },
});
export type StyleUniforms = ReturnType<typeof styleUniforms>;

/**
 * Each team's kit colour (sRGB): the commonest saturated hue over its skaters' textures (or vertex colours, linear), lifted to mid
 * lightness so dark kits still read; white where a team has none.
 */
export function kitColours(scene: GoalScene) {
	const bins = [0, 1].map(() => Array.from({ length: 12 }, () => [0, 0, 0, 0])),
		cv = document.createElement("canvas"),
		cx = cv.getContext("2d", { willReadFrequently: true })!,
		hsl = { h: 0, s: 0, l: 0 },
		c = new THREE.Color(),
		k = { r: 0, g: 0, b: 0 };
	cv.width = cv.height = 32;
	for (const p of scene.people) {
		if (p.meta.team < 0 || p.meta.goalie) continue;
		const img = (p.mesh.material.uniforms.map?.value as THREE.Texture | null)?.image as CanvasImageSource | undefined,
			col = p.mesh.geometry.getAttribute("color"),
			px: [number, number, number][] = [];
		if (img) {
			cx.drawImage(img, 0, 0, 32, 32);
			const d = cx.getImageData(0, 0, 32, 32).data;
			for (let i = 0; i < d.length; i += 4) px.push([d[i]! / 255, d[i + 1]! / 255, d[i + 2]! / 255]);
		} else if (col)
			for (let i = 0; i < col.count; i += 37) {
				c.setRGB(col.getX(i), col.getY(i), col.getZ(i)).getRGB(k, SRGB);
				px.push([k.r, k.g, k.b]);
			}
		for (const [r, g, b] of px) {
			c.setRGB(r, g, b, SRGB).getHSL(hsl, SRGB);
			if (hsl.s < 0.3 || hsl.l < 0.08 || hsl.l > 0.85) continue;
			const bin = bins[p.meta.team]![Math.floor(hsl.h * 12) % 12]!;
			bin[0]! += r;
			bin[1]! += g;
			bin[2]! += b;
			bin[3]!++;
		}
	}
	return bins.map((b) => {
		const [r, g, bl, n] = b.reduce((m, x) => (x[3]! > m[3]! ? x : m)) as [number, number, number, number];
		if (n <= 20) return new THREE.Vector3(0.95, 0.95, 0.97);
		const { h, s } = c.setRGB(r / n, g / n, bl / n, SRGB).getHSL(hsl, SRGB);
		c.setHSL(h, Math.max(s, 0.6), 0.55, SRGB).getRGB(k, SRGB);
		return new THREE.Vector3(k.r, k.g, k.b);
	});
}

/** The styles' hold on a scene: who is who (userData.styleId), the kit colours, and each frame's camera, puck and tracks. */
export class Styling {
	private parts: THREE.Object3D[][];
	private v = new THREE.Vector3();

	constructor(
		private scene: GoalScene,
		private u: StyleUniforms,
	) {
		this.parts = scene.people.map((p, j) => [p.mesh, scene.stickMeshes[j]!.mesh, ...p.gear.map((g) => g.group)]);
		for (const [j, os] of this.parts.entries()) for (const o of os) o.userData.styleId = idOf(scene.people[j]!.meta.team);
		scene.puck.userData.styleId = ID.puck;
		u.debug.value = Number(param("styledebug") ?? 0);
		const [k1, k2] = kitColours(scene),
			r = scene.S.rink;
		u.kit1.value.copy(k1!);
		u.kit2.value.copy(k2!);
		u.rink.value.set(r.hx, r.hy, r.corner);
		u.goalX.value = r.goal_x;
	}

	/** The frame at source time t through camera, onto a drawing buffer of size, from a geometry pass of `geometry` px. */
	update(t: number, camera: THREE.PerspectiveCamera, size: THREE.Vector2, geometry: { width: number; height: number }) {
		const { scene, u, v } = this,
			S = scene.S,
			f = Math.min(Math.max(t - S.frame0, 0), S.frames - 1);
		u.unproject.value.multiplyMatrices(camera.matrixWorld, camera.projectionMatrixInverse);
		u.clip.value.set(camera.near, camera.far);
		u.res.value.copy(size);
		u.gres.value.set(geometry.width, geometry.height);
		const shown = scene.shadows.filter((m) => m.visible);
		for (const [k, p] of u.feet.value.entries()) {
			const s = shown[k];
			if (s) p.set(s.position.x, s.position.y, 1);
			else p.setZ(0);
		}
		const on = scene.puck.visible && v.copy(scene.puck.position).project(camera).z < 1;
		u.puck.value.set(((v.x + 1) * size.x) / 2, ((v.y + 1) * size.y) / 2, on ? 1 : 0);
		if (TRACKS) this.tracks(f, camera, size);
	}

	/**
	 * minimal's tracks at fractional exported frame f: who is unseen (styleId 5 more), skaters (xy m, velocity over LAG s) and their
	 * kind (team id, negative for a goalie), tags (screen px over each head, jersey number, team id, negative while unseen) and,
	 * while no one is within CARRY m of the puck or the shot flies, its last 16 frames.
	 */
	private tracks(f: number, camera: THREE.PerspectiveCamera, size: THREE.Vector2) {
		const { scene, u, v } = this,
			S = scene.S,
			i = Math.round(f),
			f0 = Math.max(f - LAG * S.fps, 0),
			dt = (f - f0) / S.fps,
			at = (xy: [number, number][], g: number, mm = 1): [number, number] => {
				const a = Math.min(Math.floor(g), xy.length - 1),
					b = Math.min(a + 1, xy.length - 1),
					w = g - a;
				return [(xy[a]![0] + (xy[b]![0] - xy[a]![0]) * w) / mm, (xy[a]![1] + (xy[b]![1] - xy[a]![1]) * w) / mm];
			},
			puck = at(S.puck.xy, f);
		scene.people.forEach((p, j) => {
			const id = idOf(p.meta.team) + (p.meta.seen[i] ? 0 : 5);
			for (const o of this.parts[j]!) o.userData.styleId = id;
		});
		u.kind.value.fill(0);
		for (const l of u.label.value) l.w = 0;
		let near = Infinity;
		scene.people
			.filter((p) => p.mesh.visible && p.meta.team >= 0)
			.slice(0, 24)
			.forEach((p, k) => {
				const [x, y] = at(p.meta.root, f, 1000),
					[x0, y0] = at(p.meta.root, f0, 1000),
					id = p.meta.team + 1,
					n = p.meta.jersey ? parseInt(p.meta.jersey, 10) : -1;
				u.skater.value[k]!.set(x, y, dt ? (x - x0) / dt : 0, dt ? (y - y0) / dt : 0);
				u.kind.value[k] = p.meta.goalie ? -id : id;
				if (!p.meta.goalie) near = Math.min(near, Math.hypot(x - puck[0], y - puck[1]));
				v.set(x, y, HEAD).project(camera);
				if (v.z < 1 && n >= 0)
					u.label.value[k]!.set(((v.x + 1) * size.x) / 2, ((v.y + 1) * size.y) / 2, n % 100, p.meta.seen[i] ? id : -id);
			});
		const s = S.puck.shot,
			flying = !!s && f >= s.release - S.frame0 && f <= s.in_net - S.frame0;
		u.trailOn.value = flying || near >= CARRY ? 1 : 0;
		for (const [k, p] of u.trail.value.entries()) p.set(...at(S.puck.xy, Math.max(f - k, 0)));
	}
}

/**
 * GLSL ES 1.0 for Final, after its display() (tone mapping and the output colour space) and its tDiffuse, tBloom and bloom:
 * styled(), the frame in the chosen style.
 */
export const STYLE_GLSL = /* glsl */ `
${TRACKS ? "#define TRACKS" : ""}
uniform int style, debug; uniform sampler2D tNormal, tDepth; uniform mat4 unproject; uniform vec2 clip, res, gres, trail[16]; uniform vec3 rink, kit1, kit2, puck;
uniform vec3 feet[24]; uniform vec4 skater[24], label[24]; uniform float goalX, trailOn, kind[24];
vec3 C(vec2 uv) { return display(vec4(texture2D(tDiffuse, uv).rgb + bloom * texture2D(tBloom, uv).rgb, 1.0)).rgb; }
float L(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
vec3 saturate3(vec3 c, float s) { return clamp(mix(vec3(L(c)), c, s), 0.0, 1.0); }
float solid(float id) { return id > 9.5 ? 0.0 : id; }    // drawings on the ice (OVERLAY_ID) are the ice to ink and outlines
vec4 A(vec2 uv) {                                        // view normal xy, view depth (m; far where nothing is), style id (-1 nothing)
  vec4 n = texture2D(tNormal, uv); float d = texture2D(tDepth, uv).x;
  return vec4(n.xy * 2.0 - 1.0, n.a > 0.0 ? clip.x * clip.y / (clip.y - d * (clip.y - clip.x)) : 1e4, floor(n.a * 16.0 + 0.5) - 1.0); }
vec3 at(vec2 uv, float z) { vec4 w = unproject * vec4(uv * 2.0 - 1.0, z, 1.0); return w.xyz / w.w; }   // z: NDC depth
vec3 world(vec2 uv) { return at(uv, texture2D(tDepth, uv).x * 2.0 - 1.0); }
vec3 gpoint(vec2 uv) { return world((floor(uv * gres) + 0.5) / gres); }   // the geometry pass's own point under uv
vec2 ground(vec2 uv) { vec3 a = at(uv, -1.0), b = at(uv, 1.0); return mix(a, b, a.z / (a.z - b.z)).xy; }   // where the view ray meets z = 0
float ink(vec2 uv, float w) {                            // 1 on silhouettes (inverse depth bends), creases (normal turns) and id borders, w geometry px
  vec2 d = w / gres; vec4 c = A(uv), l = A(uv - vec2(d.x, 0.0)), r = A(uv + vec2(d.x, 0.0)), u = A(uv + vec2(0.0, d.y)), b = A(uv - vec2(0.0, d.y));
  float curv = (abs(1.0 / l.z + 1.0 / r.z - 2.0 / c.z) + abs(1.0 / u.z + 1.0 / b.z - 2.0 / c.z)) * c.z;
  float turn = length(l.xy - r.xy) + length(u.xy - b.xy), id = float(solid(l.w) != solid(r.w) || solid(u.w) != solid(b.w) || solid(l.w) != solid(c.w));
  return max(max(smoothstep(0.03, 0.1, curv), smoothstep(0.5, 0.9, turn)), id); }
float outline(vec2 uv, float w) {                        // 1 within w geometry px of a border between someone and something else
  vec4 c = A(uv); float e = 0.0;
  for (int i = 0; i < 8; i++) { vec4 n = A(uv + vec2(cos(float(i) * 0.785), sin(float(i) * 0.785)) * w / gres);
    if (max(solid(n.w), solid(c.w)) > 0.5 && (solid(n.w) != solid(c.w) || abs(1.0 / n.z - 1.0 / c.z) * c.z > 0.15)) e = 1.0; }
  return e; }

vec3 pal(int i) {                                        // PICO-8
  if (i == 0) return vec3(0.0); if (i == 1) return vec3(0.11, 0.17, 0.33); if (i == 2) return vec3(0.49, 0.15, 0.33); if (i == 3) return vec3(0.0, 0.53, 0.32);
  if (i == 4) return vec3(0.67, 0.32, 0.21); if (i == 5) return vec3(0.37, 0.34, 0.31); if (i == 6) return vec3(0.76, 0.76, 0.78); if (i == 7) return vec3(1.0, 0.95, 0.91);
  if (i == 8) return vec3(1.0, 0.0, 0.3); if (i == 9) return vec3(1.0, 0.64, 0.0); if (i == 10) return vec3(1.0, 0.93, 0.15); if (i == 11) return vec3(0.0, 0.89, 0.21);
  if (i == 12) return vec3(0.16, 0.68, 1.0); if (i == 13) return vec3(0.51, 0.46, 0.61); if (i == 14) return vec3(1.0, 0.47, 0.66); return vec3(1.0, 0.8, 0.67); }
float b2(vec2 a) { a = floor(a); return fract(dot(a, vec2(0.5, a.y * 0.75))); }
vec3 pixel() {
  vec2 cell = floor(gl_FragCoord.xy / 4.0), uv = (cell * 4.0 + 2.0) / res; vec3 c = saturate3(C(uv), 1.3) + (b2(0.5 * cell) * 0.25 + b2(cell) - 0.5) * 0.18;
  vec3 best = pal(0); float bd = 9.0; for (int i = 0; i < 16; i++) { vec3 e = c - pal(i); float d = dot(e, e); if (d < bd) { bd = d; best = pal(i); } }
  return ink(uv, 3.0) > 0.5 ? pal(1) : best; }

vec3 hsv(vec3 c) { vec4 k = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0), p = mix(vec4(c.bg, k.wz), vec4(c.gb, k.xy), step(c.b, c.g)), q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y); return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + 1e-10)), d / (q.x + 1e-10), q.x); }
vec3 rgb(vec3 c) { vec3 p = abs(fract(c.xxx + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0); return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y); }
vec4 quad(vec2 dir) { vec3 m = vec3(0.0), s = vec3(0.0);
  for (int i = 0; i <= 3; i++) for (int j = 0; j <= 3; j++) { vec3 c = C(vUv + dir * vec2(float(i), float(j)) / res); m += c; s += c * c; }
  m /= 16.0; s = abs(s / 16.0 - m * m); return vec4(m, s.r + s.g + s.b); }
vec3 vector() {                                          // flattened (Kuwahara), then four muted tones a colour, inked
  vec4 a = quad(vec2(-1.0, -1.0)), b = quad(vec2(1.0, -1.0)), c = quad(vec2(-1.0, 1.0)), d = quad(vec2(1.0, 1.0));
  vec4 m = a; if (b.a < m.a) m = b; if (c.a < m.a) m = c; if (d.a < m.a) m = d;
  vec3 h = hsv(m.rgb); float v = h.z > 0.72 ? 0.9 : h.z > 0.45 ? 0.68 : h.z > 0.22 ? 0.45 : 0.22;
  vec3 col = rgb(vec3(h.x, h.y < 0.15 ? 0.0 : floor(h.y * 0.8 * 4.0 + 0.5) / 4.0, v)) * (v < 0.5 ? vec3(0.9, 0.93, 1.05) : v > 0.8 ? vec3(1.02, 1.0, 0.97) : vec3(1.0));
  return col * (1.0 - max(outline(vUv, 2.2), 0.6 * ink(vUv, 1.0))); }

float sobel(vec2 uv) {                                   // luminance edge strength
  vec2 d = 1.0 / res; float tl = L(C(uv + d * vec2(-1.0, 1.0))), t = L(C(uv + d * vec2(0.0, 1.0))), tr = L(C(uv + d)), l = L(C(uv - d * vec2(1.0, 0.0)));
  float r = L(C(uv + d * vec2(1.0, 0.0))), bl = L(C(uv - d)), b = L(C(uv - d * vec2(0.0, 1.0))), br = L(C(uv + d * vec2(1.0, -1.0)));
  return length(vec2(tr + 2.0 * r + br - tl - 2.0 * l - bl, tl + 2.0 * t + tr - bl - 2.0 * b - br)); }
float sdRink(vec2 p) { vec2 q = abs(p) - rink.xy + rink.z; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - rink.z; }   // < 0 on the ice
float band(float d, float m, float w) { w = max(w, 0.6 * m); return smoothstep(w + m, w, abs(d)); }   // half-width w m, at least ~1 px
vec3 markings(vec2 p, float m, vec3 col, vec3 grid) {    // a 1 m and 5 m grid from the centre, fading at the boards and far off; the lines
  vec2 g1 = abs(fract(p + 0.5) - 0.5) / m, g5 = abs(fract(p / 5.0 + 0.5) - 0.5) * 5.0 / m, q = abs(p);
  float edge = smoothstep(0.0, 3.0, -sdRink(p)), bx = goalX - 18.26, ex = goalX - 6.0;
  col += grid * edge * (0.05 * (1.0 - clamp(min(g1.x, g1.y), 0.0, 1.0)) * smoothstep(0.2, 0.08, m) + 0.13 * (1.0 - clamp(min(g5.x, g5.y) - 0.3, 0.0, 1.0)));
  vec3 red = vec3(0.8, 0.28, 0.28), blue = vec3(0.3, 0.5, 0.9); float on = step(sdRink(p), 0.0);
  col = mix(col, vec3(0.32, 0.45, 0.7), 0.35 * on * step(length(vec2(q.x - goalX, p.y)), 1.83) * step(q.x, goalX));   // the creases
  col = mix(col, blue, 0.8 * on * max(band(q.x - bx, m, 0.15), band(length(p) - 4.5, m, 0.025)));   // blue lines, centre circle
  col = mix(col, red, 0.75 * on * max(max(band(p.x, m, 0.15), band(q.x - goalX, m, 0.025)), band(length(q - vec2(ex, 7.0)) - 4.5, m, 0.025)));   // red, goal lines, circles
  float spot = min(min(length(q - vec2(ex, 7.0)), length(q - vec2(bx - 1.5, 7.0))), length(p));
  return mix(col, red, 0.85 * on * smoothstep(0.3 + m, 0.3, spot)); }
vec3 team(float id) { return id < 1.5 ? kit1 : id < 2.5 ? kit2 : vec3(0.6); }
#ifdef TRACKS
float seg(vec2 p, vec2 a, vec2 b) { vec2 pa = p - a, ba = b - a; return length(pa - ba * clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0)); }
float stroke(float d, float m, float px) { return smoothstep(px * m, 0.4 * px * m, d); }
vec2 arrows(vec2 p, float m) {                           // each skater's velocity arrow (where they are in 0.5 s, at most 4.5 m): strength, team
  vec2 best = vec2(0.0);
  for (int i = 0; i < 24; i++) { float k = kind[i]; if (k <= 0.0) continue; vec4 s = skater[i]; vec2 v = s.zw * 0.5, u = normalize(v + 1e-6); v = u * min(length(v), 4.5);
    vec2 n = vec2(-u.y, u.x), tip = s.xy + v;
    float a = length(v) > 0.4 ? stroke(min(seg(p, s.xy + 0.3 * u, tip), min(seg(p, tip, tip - 0.5 * u + 0.3 * n), seg(p, tip, tip - 0.5 * u - 0.3 * n))), m, 2.0) : 0.0;
    if (a > best.x) best = vec2(a, k); }
  return best; }
float font(int d) {                                      // 3 x 5 digits, bit row * 3 + column from the top left
  if (d == 0) return 31599.0; if (d == 1) return 29850.0; if (d == 2) return 29671.0; if (d == 3) return 31207.0; if (d == 4) return 18925.0;
  if (d == 5) return 31183.0; if (d == 6) return 31695.0; if (d == 7) return 9383.0; if (d == 8) return 31727.0; return 31215.0; }
float digit(int d, vec2 q) { return q.x < 0.0 || q.y < 0.0 || q.x > 2.0 || q.y > 4.0 ? 0.0 : mod(floor(font(d) / exp2(q.y * 3.0 + q.x)), 2.0); }
vec3 labels(vec2 px, vec3 col) {                         // jersey numbers in a team-coloured tag over each head; hollow while unseen
  for (int i = 0; i < 24; i++) { vec4 t = label[i]; if (t.w == 0.0) continue; int n = int(t.z + 0.5), nd = n > 9 ? 2 : 1;
    float tw = float(nd * 12 - 3); vec2 c = t.xy + vec2(0.0, 18.0), h = vec2(tw * 0.5 + 5.0, 11.5), d = abs(px - c) - h + 3.0; if (d.x > 4.0 || d.y > 4.0) continue;
    float sd = length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - 3.0, box = smoothstep(0.8, -0.8, sd), rim = box * smoothstep(-2.2, -1.2, sd);
    vec3 k = team(abs(t.w)), fg = t.w < 0.0 ? k * 0.8 : L(k) > 0.55 ? vec3(0.1) : vec3(1.0);
    vec2 q = floor(vec2(px.x - c.x + tw * 0.5, c.y + 7.5 - px.y) / 3.0); int j = int(floor(q.x / 4.0)); q.x -= float(j) * 4.0;
    float g = j < 0 || j >= nd ? 0.0 : digit(nd == 2 && j == 0 ? n / 10 : n - n / 10 * 10, q);
    col = mix(col, t.w < 0.0 ? mix(vec3(0.95), k, rim) : k, box); col = mix(col, fg, g * box); }
  return col; }
#endif
float shade(vec2 p) {                                    // the contact shadows round everyone's feet (scene.ts's falloff)
  float s = 1.0; for (int i = 0; i < 24; i++) { vec2 d = p - feet[i].xy; s *= 1.0 - 0.35 * feet[i].z * exp(-dot(d, d) / 0.18); } return s; }
vec3 minimal() {                                         // the system's view: light cool ice, grid and lines as landmarks; the players in kit colour
  vec4 a = A(vUv); vec3 w = gpoint(vUv), c = C(vUv), slate = vec3(0.62, 0.68, 0.78), orange = vec3(1.0, 0.45, 0.0);
  float l = L(c), near = step(sdRink(w.xy), 2.0) * step(w.z, 2.5) * float(a.z < 1e3);   // boards, glass and goals; beyond them the stands
  bool ice = a.w < 0.5 && a.z < 1e3 && abs(w.z) < 0.05;
  vec2 g = ground(vUv);                                  // the ice under the pixel itself, m a px there
  float m = 0.7 * max(length(ground(vUv + vec2(1.0 / res.x, 0.0)) - g), length(ground(vUv + vec2(0.0, 1.0 / res.y)) - g));
  vec3 col = slate * (0.12 + mix(0.12, 0.38, near) * l) + slate * near * (0.25 * ink(vUv, 1.0) * float(a.w < 0.5) + 0.12 * smoothstep(0.15, 0.45, sobel(vUv)));
  if (near > 0.5 && a.w < 0.5 && abs(abs(w.x) - goalX) < 1.5 && abs(w.y) < 1.5 && w.z < 1.4) col = mix(col, saturate3(c, 0.6) * 0.8, 0.6);   // the goals
  if (ice) {                                             // the ice, its one colour, darkened only by the contact shadows
    col = markings(g, m, vec3(0.86, 0.9, 0.95) * mix(1.0, shade(g), 0.7), -vec3(0.55, 0.5, 0.4));
#ifdef TRACKS
    vec2 ar = arrows(g, m); col = mix(col, team(ar.y) * 0.7, ar.x);
    if (trailOn > 0.5) for (int i = 0; i < 15; i++) col = mix(col, orange, (1.0 - float(i) / 15.0) * stroke(seg(g, trail[i], trail[i + 1]), m, 3.5));
#endif
  }
  if (debug == 1) return vec3(float(ice), shade(g), near);   // red the ice, green its contact shading, blue the boards and glass
  float id = a.w > 4.5 ? a.w - 5.0 : a.w;               // 5 more: someone the broadcast does not see in this frame
  if (id > 0.5) col = team(id) * (0.55 + 0.45 * sqrt(max(1.0 - dot(a.xy, a.xy), 0.0))) * (1.0 - 0.6 * ink(vUv, 1.0));   // brightest facing the camera
#ifdef TRACKS
  if (a.w > 5.5 && a.w < 8.5) col = mix(mix(vec3(0.9, 0.92, 0.95), team(id), 0.2), team(id) * 0.8, ink(vUv, 2.0));   // guessed: hollow
#endif
  if (id > 3.5) col = vec3(0.05);
  if (a.w > 9.5) col = C(vUv);                           // a drawing on the ice (OVERLAY_ID), as it is
  col = mix(col, orange, 0.9 * puck.z * smoothstep(1.5, 0.0, abs(length(gl_FragCoord.xy - puck.xy) - 9.0)));
#ifdef TRACKS
  col = labels(gl_FragCoord.xy, col);
#endif
  return col; }

vec3 styled() { return style == 1 ? pixel() : style == 2 ? vector() : minimal(); }
`;
