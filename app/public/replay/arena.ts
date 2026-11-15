// The arena: ice, boards, ledge, tiered stands, the glass above the boards and the roof.
//
// Projective lookup uses the calibrated main-camera centre (S.source.camera) and the reference atlas surfaces (render.py: ice,
// vertical boards, a flat ledge, a uniform slope). Procedural crowd rows add inferred relief, not measured stand depth.
// Atlas fallback content and approximate geometry can still stretch or repeat in novel views; projection does not establish
// source coverage for surfaces the camera never saw. The roof above the stands' back wall is inferred too (the footage never shows
// it): a dark ceiling with dim light rigs, so that low cameras looking up see an arena at night, not a void.
import * as THREE from "three";
import { type Atlas, loadAtlas, type Scoreboard } from "./atlas";
import { arenaLight, ENV_BRDF_GLSL, LAMP, TRUSS } from "./light";
import type { Perf } from "./perf";
import type { Rink, SceneData, Vec3 } from "./types";
import { param } from "./util";

const TREAD = 0.8,
	PEOPLE = 1.0,
	HEADS = 0.35; // stand row depth, height of a seated row's front, depth of their heads (m)
const f = (x: number) => Number(x).toFixed(6);

type Edge = [number, number, number, number];

/** Rink outline (rounded rectangle) by arc length s, counter-clockwise from the +x end centre: point and outward normal. */
export function outline(rink: Rink): (s: number) => Edge {
	const { hx, hy, corner: c } = rink,
		a = hx - c,
		b = hy - c,
		pi = Math.PI;
	const ends = [b, (c * pi) / 2, 2 * a, (c * pi) / 2, 2 * b, (c * pi) / 2, 2 * a, (c * pi) / 2, b].reduce<number[]>((acc, x) => {
		acc.push((acc.at(-1) ?? 0) + x);
		return acc;
	}, []);
	const arcs: Partial<Record<number, Vec3>> = { 1: [a, b, 0], 3: [-a, b, pi / 2], 5: [-a, -b, pi], 7: [a, -b, (3 * pi) / 2] };
	return (s) => {
		s = ((s % rink.perim) + rink.perim) % rink.perim;
		let seg = ends.findIndex((e) => s < e);
		if (seg < 0) seg = 8;
		const u = s - (seg ? ends[seg - 1]! : 0),
			arc = arcs[seg];
		if (arc) {
			const [cx, cy, a0] = arc,
				t = a0 + u / c;
			return [cx + c * Math.cos(t), cy + c * Math.sin(t), Math.cos(t), Math.sin(t)];
		}
		return (
			[
				[hx, u, 1, 0],
				null,
				[a - u, hy, 0, 1],
				null,
				[-hx, b - u, -1, 0],
				null,
				[-a + u, -hy, 0, -1],
				null,
				[hx, -b + u, 1, 0],
			] as (Edge | null)[]
		)[seg]!;
	};
}

// The rink's outline, the reference surface's profile and its (s, h) coordinates, in GLSL.
function rinkGlsl(rink: Rink) {
	const { hx, hy, corner: c, wall, ledge, slope, perim } = rink,
		a = hx - c,
		b = hy - c;
	return /* glsl */ `
const float RA = ${f(a)}, RB = ${f(b)}, RC = ${f(c)}, WALL = ${f(wall)}, LEDGE = ${f(ledge)}, SLOPE = ${f(slope)}, PERIM = ${f(perim)};
const float K = ${f(Math.hypot(1, slope))}, PI = 3.14159265;
float sdRink(vec2 p) { vec2 q = abs(p) - vec2(RA, RB); return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - RC; }
float outlineS(vec2 p) {
  float x = p.x, y = p.y, th = atan(y - sign(y) * RB, x - sign(x) * RA);
  if (th < 0.0) th += 2.0 * PI;
  if (x > RA && abs(y) <= RB) return mod(y, PERIM);
  if (x > RA && y > RB) return RB + RC * th;
  if (abs(x) <= RA && y > RB) return RB + RC * PI / 2.0 + (RA - x);
  if (x < -RA && y > RB) return RB + RC * PI / 2.0 + 2.0 * RA + RC * (th - PI / 2.0);
  if (x < -RA && abs(y) <= RB) return RB + RC * PI + 2.0 * RA + (RB - y);
  if (x < -RA && y < -RB) return 3.0 * RB + RC * PI + 2.0 * RA + RC * (th - PI);
  if (abs(x) <= RA && y < -RB) return 3.0 * RB + 3.0 * RC * PI / 2.0 + 2.0 * RA + (x + RA);
  return 3.0 * RB + 3.0 * RC * PI / 2.0 + 4.0 * RA + RC * (th - 3.0 * PI / 2.0);
}
float outH(float sd, float z) { return sd < 0.02 ? z : sd < LEDGE ? WALL + sd : WALL + LEDGE + (sd - LEDGE) * K; }`;
}

// The crowd's motion: cells (m along the stands, m up them), sway and jump heights (m), the share of cells jumping in a cheer; the
// cheer after the puck is in: from REACT s, up over RISE, held to HOLD, gone by END.
const CROWD = { CELL: [0.6, 0.9], SWAY: 0.012, JUMP: 0.11, JUMPERS: 0.65, REACT: 0.3, RISE: 0.6, HOLD: 4, END: 8 } as const;

// The projective lookup (see the top of this file), in GLSL, after rinkGlsl and the atlas.
function projectionGlsl(camera: Vec3) {
	return /* glsl */ `
const vec3 C0 = vec3(${camera.map(f).join(", ")});
// The atlas's lowest board rows mix in the ice wherever the footage's ice edge and the boards don't line up: the kick plate is read
// from KICK up, and the boards darken into a contact line at the ice.
const float KICK = 0.08;
// The stands' reference: the slope, continued over the rink as a plateau at board height (stand points never resolve onto the ice)
float above(vec3 q) { float sd = sdRink(q.xy); return q.z - (sd < LEDGE ? WALL : WALL + SLOPE * (sd - LEDGE)); }
// Where the broadcast ray through a stand or ledge point P meets their reference surface (P itself when not within 6 m of P).
vec3 broadcastHit(vec3 P) {
  vec3 d = P - C0; float L = length(d); d /= L;
  float t0 = max(L - 6.0, 0.5), t1 = L + 6.0;
  if (above(C0 + d * t0) <= 0.0 || above(C0 + d * t1) > 0.0) return P;
  for (int i = 0; i < 26; i++) { float tm = 0.5 * (t0 + t1); if (above(C0 + d * tm) > 0.0) t0 = tm; else t1 = tm; }
  return C0 + d * t1;
}
// The crowd in the stands, a still photo in the atlas, cut into seat-sized CROWD cells that move it: swaying a little, then after a
// goal (cheer up to 1) most bouncing as people jumping, each cell at its own pace (source time crowdT, s). Height to look up lower.
uniform float crowdT, cheer;
float cellHash(vec2 c) { return fract(sin(dot(c, vec2(127.1, 311.7))) * 43758.5453); }
float crowdLift(vec2 sh, float sd) {
  vec2 c = floor(sh / vec2(${f(CROWD.CELL[0])}, ${f(CROWD.CELL[1])})); float a = cellHash(c), b = cellHash(c + 17.0);
  float sway = ${f(CROWD.SWAY)} * sin(2.0 * PI * ((0.2 + 0.2 * b) * crowdT + a));
  float jump = (a < ${f(CROWD.JUMPERS)} ? ${f(CROWD.JUMP)} : ${f(CROWD.JUMP / 4)}) * abs(sin(PI * ((1.6 + 0.6 * b) * crowdT + a)));
  return smoothstep(LEDGE, LEDGE + 0.5, sd) * mix(sway, jump, cheer);
}
// Ice and the faces of the boards are the reference surface itself: looked up directly. Ledge and stands facing the broadcast
// camera through its ray; faces it saw edge-on or never looked at, and the stands between the rink and the camera (it looked down
// past them, so its rays there hit what it never saw), increasingly horizontally onto the slope at the same height, where the
// atlas holds the stands filled in. So do points whose ray meets no reference within 6 m (the back wall above the top row: above
// the atlas, the dark roof).
vec3 arenaColor(vec3 P, vec3 n) {
  float sd = sdRink(P.xy);
  vec2 outward = normalize(vec2(sdRink(P.xy + vec2(0.05, 0.0)) - sdRink(P.xy - vec2(0.05, 0.0)), sdRink(P.xy + vec2(0.0, 0.05)) - sdRink(P.xy - vec2(0.0, 0.05))));
  float facing = dot(n, normalize(C0 - P)) * (1.0 - smoothstep(-2.0, 2.0, dot(outward, C0.xy - P.xy)));
  // read by height up the profile: on an upright face (a row's front) from its height, on a face turned up (the flat a row sits on,
  // the slope of the crowd's relief) from its distance out (its height barely changes across it, which would stretch one row of the
  // atlas over it, streaks seen from above)
  float tread = smoothstep(0.15, 0.5, abs(n.z)) * step(LEDGE, sd);
  vec2 plain = vec2(outlineS(P.xy), mix(outH(max(LEDGE + (P.z - WALL) / SLOPE, LEDGE), P.z), outH(sd, P.z), tread)), uv = plain;
  if (sd < 0.02 && P.z < WALL) uv = plain = vec2(outlineS(P.xy), max(P.z, KICK));
  else if (facing > 0.0) { vec3 q = broadcastHit(P); if (q != P) uv = vec2(outlineS(q.xy), outH(max(sdRink(q.xy), 0.0), q.z)); }
  float lift = crowdLift(plain, sd); plain.y -= lift; uv.y -= lift;
  // Implicit texture gradients require uniform control flow across the fragment quad.
  // Stands the atlas holds laid out straight on (atlasPlain) are read by height too.
  float seen = smoothstep(0.15, 0.35, facing) * (1.0 - atlasPlain(plain.x, plain.y));
  vec3 outer = mix(atlasOutColor(plain.x, plain.y), atlasOutColor(uv.x, uv.y), seen), inner = atlasIceColor(P.xy);
  float contact = mix(0.72, 1.0, smoothstep(0.0, 0.06, P.z));
  return sd < -0.02 && P.z < 0.05 ? inner : outer * (sd < 0.02 && P.z < WALL ? contact : 1.0);
}`;
}

const DEBUG: Record<string, Record<string, number>> = { hits: { DEBUG_HITS: 1 }, stretch: { DEBUG_STRETCH: 1 } };

function projected(atlas: Atlas, rink: Rink, camera: Vec3) {
	return new THREE.ShaderMaterial({
		defines: DEBUG[param("debug") ?? ""] ?? {},
		uniforms: {
			...atlas.uniforms,
			crowdT: { value: 0 },
			cheer: { value: 0 },
		},
		side: THREE.DoubleSide,
		vertexShader: /* glsl */ `varying vec3 wp; void main() { vec4 w = modelMatrix * vec4(position, 1.0); wp = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
		fragmentShader:
			rinkGlsl(rink) +
			atlas.glsl +
			projectionGlsl(camera) +
			/* glsl */ `
varying vec3 wp;
// Faces turned away from the rink (behind the heads; the broadcast never saw them) are shaded darker, tops a little.
float shade(vec3 P, vec3 n) {
  vec2 e = vec2(0.05, 0.0); vec2 o = normalize(vec2(sdRink(P.xy + e.xy) - sdRink(P.xy - e.xy), sdRink(P.xy + e.yx) - sdRink(P.xy - e.yx)));
  float away = dot(n.xy, o);
  return mix(1.0, 0.86, smoothstep(0.6, 0.95, n.z)) * mix(1.0, 0.5, smoothstep(-0.1, 0.6, away));
}
void main() {
  vec3 n = normalize(cross(dFdx(wp), dFdy(wp))); if (dot(n, cameraPosition - wp) < 0.0) n = -n;   // the visible side's normal
  vec3 col = arenaColor(wp, n) * (sdRink(wp.xy) > 0.5 ? shade(wp, n) : 1.0);
  gl_FragColor = vec4(col, 1.0);
#ifdef DEBUG_HITS
  { vec3 q = broadcastHit(wp); float sd = sdRink(q.xy);
    gl_FragColor = vec4(q == wp ? vec3(1, 0, 1) : sd < 0.0 ? vec3(0.9) : sd < 0.02 ? vec3(0, 0, 1) : sd < LEDGE ? vec3(0, 1, 0) : vec3(1, 0.5, 0), 1.0); }
#endif
#ifdef DEBUG_STRETCH
  { vec3 q = broadcastHit(wp); vec2 uv = vec2(outlineS(q.xy), outH(max(sdRink(q.xy), 0.0), q.z));   // atlas metres per surface metre:
    float k = length(fwidth(uv)) / length(fwidth(wp));                                                   // red magnified (streaks), green ~1
    gl_FragColor = vec4(1.0 - smoothstep(0.33, 0.8, k), smoothstep(0.33, 0.8, k) * (1.0 - smoothstep(1.5, 3.0, k)), smoothstep(1.5, 3.0, k), 1.0); }
#endif
#include <colorspace_fragment>
}`,
	});
}

// Profile of the boards, ledge and stands as (sd, z) points outward from the rink: up the boards, across the ledge, then one block
// per row of seated people (a 1 m front centred on the reference slope, where the atlas has the crowd, the tops of the heads,
// then down behind them to the next row) up to the crowd's top (crowdTop, m up the profile: above it the atlas holds beams and
// walls, which rows would cut into streaks), the plain slope on to the top of the atlas, then a back wall.
function profile(rink: Rink, crowdTop = Infinity): [number, number][] {
	const { wall, ledge, slope, out_res: res } = rink,
		k = Math.hypot(1, slope),
		hMax = rink.out_size[0] * res;
	const sdAt = (h: number) => ledge + (h - wall - ledge) / k,
		sdTop = sdAt(hMax),
		sdRows = Math.min(sdAt(crowdTop), sdTop),
		ref = (sd: number) => wall + slope * (sd - ledge),
		pts: [number, number][] = [
			[0, 0],
			[0, wall],
			[ledge, wall],
		];
	let sd = ledge + 0.15;
	for (; sd + TREAD <= sdRows; sd += TREAD) {
		pts.push([sd, Math.max(ref(sd) - PEOPLE / 2, wall)], [sd, ref(sd) + PEOPLE / 2], [sd + HEADS, ref(sd) + PEOPLE / 2]);
	}
	if (sd < sdTop) pts.push([sd, ref(sd)], [sdTop, ref(sdTop)]);
	const [sdEnd, zEnd] = pts.at(-1)!;
	pts.push([sdEnd, zEnd + 8]);
	return pts;
}

// A surface swept along the rink outline: for every outline sample, the profile points (sd outward, z up). Corners are duplicated
// so faces stay flat-shaded steps rather than a smoothed slope.
function sweep(rink: Rink, prof: [number, number][], step = 0.25) {
	const at = outline(rink),
		ns = Math.ceil(rink.perim / step),
		pos: number[] = [],
		idx: number[] = [],
		m = prof.length;
	for (let i = 0; i <= ns; i++) {
		const [bx, by, nx, ny] = at((rink.perim * i) / ns);
		for (const [sd, z] of prof) pos.push(bx + nx * sd, by + ny * sd, z);
	}
	for (let i = 0; i < ns; i++)
		for (let j = 0; j < m - 1; j++) {
			const a = i * m + j,
				b = a + m;
			idx.push(a, b, a + 1, a + 1, b, b + 1);
		}
	const g = new THREE.BufferGeometry();
	g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
	g.setIndex(idx);
	return g;
}

const MIRROR = 2,
	LEAST = 360,
	BLUR = 2.5; // the mirror is rendered at 1/2^MIRROR of the frame (at least LEAST px tall), and read as soft as mip level BLUR of the frame would be

// The ice: its atlas, the roof's lamps mirrored in it (the atlas's atlasGlare, read along each view's reflected ray, so the streaks
// move with the view as they would; from the broadcast camera they are the footage's), and a soft mirror of everything standing on
// it (players, sticks, goals, puck; not the arena), both by ice's reflectance (Fresnel, f0 0.018, stronger toward grazing). Close
// up the surface is not flat (ICE): a slow unevenness in two sizes bends what it mirrors, and skates have cut it. A cut is a
// groove a few mm wide along an arc, as a skate's edge leaves it: in each of a few layers of cells, arcs about one far centre,
// here and there, in stretches, more where the ice is worn. Its sides tilt what the ice mirrors and catch the light on one side,
// and the frost in it is white and matte, so it breaks the mirror rather than shining. Each cut is drawn as wide as a pixel sees
// it across (its own width and the pixel's footprint across it, its strength less by as much), so from low over the ice the cuts
// that run away from the camera stay fine lines and those across the view melt into the surface, with no shimmer; from the
// broadcast's height there are none, and the ice is the footage's. The mirror is rendered from the camera reflected in the ice
// before the ice is drawn, at a quarter of the frame's resolution (MIRROR; more for a small frame, which would leave it blocky),
// and read back blurred; once per frame (the scene's userData.frame, set by its viewer), however often the frame draws the scene
// (the broadcast look's geometry pass draws it again). Timed as a GPU span when the scene's userData.perf is a Perf (perf.ts).
const ICE = {
	ROLL: 0.004, // slope of the unevenness
	CUT: 0.045, // slope of a cut's sides
	WIDTH: 0.0035, // m: a cut's width, about
	SPACING: 0.09, // m between the places a cell's cuts may be
	FROST: [0.92, 0.95, 0.98], // the frost in a cut
} as const;
/** Whether the ice shows the skates' cuts: 1 but at low quality (viewer.ts sets it with the quality), where their cost close up is saved. */
export const skateCuts = { value: 1 };
function ice(rink: Rink, atlas: Atlas, arena: THREE.Object3D) {
	const { hx, hy, corner: c } = rink,
		sh = new THREE.Shape();
	sh.moveTo(hx, -hy + c);
	sh.lineTo(hx, hy - c);
	sh.absarc(hx - c, hy - c, c, 0, Math.PI / 2, false);
	sh.lineTo(-hx + c, hy);
	sh.absarc(-hx + c, hy - c, c, Math.PI / 2, Math.PI, false);
	sh.lineTo(-hx, -hy + c);
	sh.absarc(-hx + c, -hy + c, c, Math.PI, 1.5 * Math.PI, false);
	sh.lineTo(hx - c, -hy);
	sh.absarc(hx - c, -hy + c, c, 1.5 * Math.PI, 2 * Math.PI, false);
	const target = new THREE.WebGLRenderTarget(2, 2, { generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter });
	const toTexture = new THREE.Matrix4(),
		mirror = new THREE.PerspectiveCamera(),
		size = new THREE.Vector2(),
		pos = new THREE.Vector3(),
		fwd = new THREE.Vector3(),
		up = new THREE.Vector3();
	const mat = new THREE.ShaderMaterial({
		uniforms: {
			...atlas.uniforms,
			mirror: { value: target.texture },
			toMirror: { value: toTexture },
			level: { value: BLUR - MIRROR },
			cutsOn: skateCuts,
		},
		vertexShader: /* glsl */ `varying vec3 wp; void main() { vec4 w = modelMatrix * vec4(position, 1.0); wp = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
		fragmentShader:
			rinkGlsl(rink) +
			atlas.glsl +
			ENV_BRDF_GLSL +
			/* glsl */ `
uniform sampler2D mirror; uniform mat4 toMirror; uniform float level, cutsOn;
varying vec3 wp;
float iceHash(vec2 c) { return fract(sin(dot(c, vec2(127.1, 311.7))) * 43758.5453); }
float iceNoise(vec2 p) {
  vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(iceHash(i), iceHash(i + vec2(1, 0)), f.x), mix(iceHash(i + vec2(0, 1)), iceHash(i + 1.0), f.x), f.y);
}
vec2 iceHash2(vec2 c) { return fract(sin(vec2(dot(c, vec2(127.1, 311.7)), dot(c, vec2(269.5, 183.3)))) * 43758.5453); }
// the slow unevenness: the surface's tilt (its slope along x and y) at p
vec2 iceRoll(vec2 p) {
  vec2 e = vec2(0.37, 0.0), f = vec2(0.08, 0.0);
  return ${f(ICE.ROLL)} * vec2(iceNoise(p / 1.7 + e) - iceNoise(p / 1.7 - e), iceNoise(p / 1.7 + e.yx) - iceNoise(p / 1.7 - e.yx))
    + ${f(ICE.ROLL * 0.6)} * vec2(iceNoise(p / 0.31 + f) - iceNoise(p / 0.31 - f), iceNoise(p / 0.31 + f.yx) - iceNoise(p / 0.31 - f.yx));
}
// the skates' cuts at p, a pixel there spanning px and py of the ice: the tilt of their sides (xy) and the frost in them (z)
vec3 iceCuts(vec2 p, vec2 px, vec2 py) {
  vec3 cuts = vec3(0.0);
  if (cutsOn < 0.5 || min(length(px), length(py)) > ${f(ICE.SPACING * 0.5)}) return cuts;
  for (int i = 0; i < 4; i++) {
    float fi = float(i), G = 2.6 + 1.1 * fi;
    vec2 g = p / G + 17.3 * fi, id = floor(g), at = fract(g), h = iceHash2(id + 31.0 * fi);
    float turn = 6.2832 * h.x, R = G * (0.8 + 1.8 * h.y);
    vec2 q = p - (id + 0.5 - 17.3 * fi) * G - R * vec2(cos(turn), sin(turn));  // from the centre the cell's arcs are about
    float r = length(q), u = r / ${f(ICE.SPACING)}, line = floor(u), x = (fract(u) - 0.5) * ${f(ICE.SPACING)};
    vec2 across = q / r, k = iceHash2(vec2(line, fi) + 7.0 * id);
    float w = ${f(ICE.WIDTH)} * (0.6 + 0.9 * k.y),
      seen = abs(dot(px, across)) + abs(dot(py, across)),
      wide = sqrt(w * w + seen * seen),
      on = step(0.72, k.x)
        * smoothstep(0.5, 0.62, iceNoise(vec2(atan(q.y, q.x) * r / 0.9, line + 13.0 * fi)))   // in stretches along the arc
        * smoothstep(0.0, 0.18, min(min(at.x, 1.0 - at.x), min(at.y, 1.0 - at.y)))            // ending inside its cell
        * smoothstep(0.3, 0.7, iceNoise(p / 4.0 + 9.0 * fi))                                  // where the ice is worn
        * (w / wide) * (1.0 - smoothstep(${f(ICE.SPACING * 0.25)}, ${f(ICE.SPACING * 0.5)}, wide)),
      d = x / wide;
    cuts.xy += across * (${f(ICE.CUT * 1.65)} * on * d * exp(-d * d));
    cuts.z += on * exp(-d * d);
  }
  return vec3(cuts.xy, min(cuts.z, 1.0));
}
void main() {
  vec3 cuts = iceCuts(wp.xy, dFdx(wp.xy), dFdy(wp.xy));
  vec3 v = normalize(cameraPosition - wp), n = normalize(vec3(-iceRoll(wp.xy) - cuts.xy, 1.0));
  vec4 m = toMirror * vec4(wp, 1.0);
  vec4 r = textureLod(mirror, m.xy / m.w + n.xy * 0.6, level);
  float nv = clamp(dot(n, v), 1e-3, 1.0);
  vec2 ab = envBrdf(0.1, nv); float refl = (0.018 * ab.x + ab.y) * (1.0 - 0.8 * cuts.z);   // ice: f0 0.018 (n = 1.31), its glare's lobe (ice_glare.py); frost mirrors little
  float k = min(0.03 + 0.8 * refl, 0.35) * r.a;
  float edge = mix(1.0, 0.82, smoothstep(-0.1, 0.0, sdRink(wp.xy)));          // scuffed, shadowed ice along the boards
  float lit = 1.0 - 2.0 * dot(cuts.xy, vec2(0.6, 0.8));                        // a cut's side towards the lamps lighter, the other darker
  vec3 ice = mix(atlasIceColor(wp.xy) * edge, vec3(${ICE.FROST.join(", ")}), 0.3 * cuts.z) * lit + vec3(atlasGlare(wp, reflect(-v, n)) * refl);
  gl_FragColor = vec4(mix(ice, r.rgb / max(r.a, 1e-3), k), 1.0);
  #include <colorspace_fragment>
}`,
	});
	const mesh = new THREE.Mesh(new THREE.ShapeGeometry(sh, 24), mat);
	let busy = false,
		drawn: unknown;
	mesh.onBeforeRender = (renderer, scene, camera) => {
		const frame = scene.userData.frame;
		if (busy || (frame !== undefined && frame === drawn)) return;
		busy = true;
		drawn = frame;
		// a fraction of what the frame is drawn into: the target the scene is rendered at (scaled by the frame budget), else the canvas
		const into = renderer.getRenderTarget();
		if (into) size.set(into.width, into.height);
		else renderer.getDrawingBufferSize(size);
		const shift = Math.min(Math.max(Math.log2(size.y / LEAST), 0), MIRROR);
		target.setSize(Math.max(Math.round(size.x / 2 ** shift), 1), Math.max(Math.round(size.y / 2 ** shift), 1));
		mat.uniforms.level!.value = BLUR - shift;
		const e = camera.matrixWorld.elements;
		pos.set(e[12]!, e[13]!, -e[14]!);
		fwd.set(-e[8]!, -e[9]!, e[10]!);
		up.set(e[4]!, e[5]!, -e[6]!);
		mirror.position.copy(pos);
		mirror.up.copy(up);
		mirror.lookAt(fwd.add(pos));
		mirror.projectionMatrix.copy(camera.projectionMatrix);
		mirror.projectionMatrixInverse.copy(camera.projectionMatrixInverse);
		mirror.updateMatrixWorld();
		toTexture
			.set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 1)
			.multiply(mirror.projectionMatrix)
			.multiply(mirror.matrixWorldInverse);
		const bg = scene.background,
			rt = renderer.getRenderTarget(),
			clear = renderer.getClearAlpha();
		arena.visible = mesh.visible = false;
		scene.background = null;
		renderer.setClearAlpha(0);
		const perf = scene.userData.perf as Perf | undefined;
		perf?.gpuBegin("ice mirror");
		renderer.setRenderTarget(target);
		renderer.clear();
		renderer.render(scene, mirror);
		perf?.gpuEnd();
		renderer.setRenderTarget(rt);
		renderer.setClearAlpha(clear);
		scene.background = bg;
		arena.visible = mesh.visible = true;
		busy = false;
	};
	return mesh;
}

// Glass above the boards: 1.5 cm thick panels GLASS high all round (as this arena's is), joints every PANEL m, standing on
// the boards' top rail. The faces are clear and reflect the arena (the roof's rows of lights above the horizon, the dim stands below)
// more toward grazing angles; the top edges and joints show the faint green of acrylic edges.
const PANEL = 1.5,
	PANE = 0.015,
	GLASS = 1.8;

function glass(rink: Rink) {
	const at = outline(rink),
		ns = Math.ceil(rink.perim / 0.1),
		pos: number[] = [],
		idx: number[] = [];
	const ring = [
		[0.02, 0],
		[0.02, 1],
		[0.02 + PANE, 1],
		[0.02 + PANE, 0],
	] as const; // inner face up, across the top, outer face down
	for (let i = 0; i <= ns; i++) {
		const [bx, by, nx, ny] = at((rink.perim * i) / ns);
		for (const [d, top] of ring) pos.push(bx + nx * d, by + ny * d, rink.wall + top * GLASS);
	}
	for (let i = 0; i < ns; i++)
		for (let j = 0; j < 3; j++) {
			const a = i * 4 + j,
				b = a + 4;
			idx.push(a, b, a + 1, a + 1, b, b + 1);
		}
	const g = new THREE.BufferGeometry();
	g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
	g.setIndex(idx);
	const mat = new THREE.ShaderMaterial({
		transparent: true,
		depthWrite: false,
		side: THREE.DoubleSide,
		vertexShader: /* glsl */ `varying vec3 wp; void main() { vec4 w = modelMatrix * vec4(position, 1.0); wp = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
		fragmentShader:
			rinkGlsl(rink) +
			/* glsl */ `
varying vec3 wp;
const float PANEL = ${f(PANEL)};
vec3 environment(vec3 r) {                               // the arena around the glass, as reflected and soft (a pane is never a
  float up = smoothstep(0.15, 0.6, r.z), down = 1.0 - smoothstep(-0.08, 0.0, r.z);    // clean mirror): bright ice below the horizon,
  return mix(mix(vec3(0.05, 0.05, 0.06), vec3(0.62, 0.61, 0.58), up), vec3(0.72, 0.76, 0.8), down);   // dim stands, the lit roof above
}
void main() {
  vec3 n = normalize(cross(dFdx(wp), dFdy(wp))), v = normalize(cameraPosition - wp); if (dot(n, v) < 0.0) n = -n;
  float c = clamp(dot(n, v), 0.0, 1.0), fres = 0.04 + 0.96 * pow(1.0 - c, 5.0);
  vec3 env = environment(reflect(-v, n));
  float s = outlineS(wp.xy), joint = 1.0 - smoothstep(0.004, 0.012, abs(fract(s / PANEL + 0.5) - 0.5) * PANEL);
  float top = smoothstep(0.5, 0.9, n.z);
  vec3 col = mix(mix(env, vec3(0.45, 0.55, 0.52), joint), vec3(0.75, 0.88, 0.82), top);
  float a = clamp(fres * (0.25 + 0.75 * env.g) * 0.7 + 0.025 + 0.22 * joint + 0.7 * top, 0.0, 0.9);
  gl_FragColor = vec4(col, a);
  #include <colorspace_fragment>
}`,
	});
	const m = new THREE.Mesh(g, mat);
	m.renderOrder = 2;
	return m;
}

// The roof over it all, which the footage never shows: a flat ceiling at the top of the stands' back wall (top, m), out to the wall
// (sd, m outward from the rink), seen only from below (its faces turned down, so a camera above it, as the tactical one is, looks
// through). Night dark navy, a little lighter over the ice where its lights hang, near black at the walls; over the ice, dim trusses every
// TRUSS m across the rink's length, a lamp every LAMP m along each (light.ts, whose light they give), softly haloed.
function roof(rink: Rink, sd: number, top: number) {
	const at = outline(rink),
		ns = Math.ceil(rink.perim / 0.5),
		pos: number[] = [],
		idx: number[] = [];
	for (let i = 0; i < ns; i++) {
		const [bx, by, nx, ny] = at((rink.perim * i) / ns);
		pos.push(bx + nx * sd, by + ny * sd, top);
	}
	pos.push(0, 0, top);
	// a fan from the centre, wound clockwise seen from above: facing down
	for (let i = 0; i < ns; i++) idx.push(ns, (i + 1) % ns, i);
	const g = new THREE.BufferGeometry();
	g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
	g.setIndex(idx);
	const mat = new THREE.ShaderMaterial({
		vertexShader: /* glsl */ `varying vec3 wp; void main() { vec4 w = modelMatrix * vec4(position, 1.0); wp = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
		fragmentShader:
			rinkGlsl(rink) +
			/* glsl */ `
varying vec3 wp;
const float WALL_SD = ${f(sd)}, TRUSS = ${f(TRUSS)}, LAMP = ${f(LAMP)};
float line(float x, float w) { float d = abs(fract(x + 0.5) - 0.5), a = fwidth(x); return 1.0 - smoothstep(w - a, w + a, d); }
void main() {
  float sd = sdRink(wp.xy), over = 1.0 - smoothstep(0.0, 6.0, sd);                     // over the ice, fading out past the boards
  vec3 col = mix(vec3(0.022, 0.026, 0.045), vec3(0.004, 0.005, 0.012), smoothstep(-4.0, WALL_SD, sd));  // lit by the rigs, dark at the walls
  vec2 g = wp.xy / vec2(TRUSS * 2.0, TRUSS);                                            // trusses along x, cross members along y
  float truss = max(line(g.y, 0.25 / TRUSS), 0.6 * line(g.x, 0.15 / TRUSS));
  col = mix(col, vec3(0.045, 0.05, 0.065), truss * over);
  vec2 l = vec2(wp.x / LAMP, wp.y / TRUSS), c = (fract(l + 0.5) - 0.5) * vec2(LAMP, TRUSS);
  float r = length(c), lamp = (1.0 - smoothstep(0.12, 0.2, r)) + 0.12 * exp(-r * 1.6);
  col += vec3(0.4, 0.38, 0.33) * lamp * over;
  gl_FragColor = vec4(col, 1.0);
  #include <colorspace_fragment>
}`,
	});
	return new THREE.Mesh(g, mat);
}

// The centre-hung scoreboard (scoreboard.py): a cube over centre ice with the face the broadcast camera saw on all four sides, over a
// ring tapering below it; underside and top dark.
function scoreboard({ x, width: w, ring, z: [z0, z1, z2], map }: Scoreboard) {
	const pos: number[] = [],
		uv: number[] = [],
		idx: number[] = [],
		rows = [
			[z0, ring / 2],
			[z1, w / 2],
			[z2, w / 2],
		] as const;
	for (const [nx, ny] of [
		[0, -1],
		[1, 0],
		[0, 1],
		[-1, 0],
	] as const) {
		const rx = -ny,
			ry = nx,
			first = pos.length / 3; // right, seen from in front of the side
		for (const [z, hw] of rows)
			for (const side of [-1, 1]) {
				pos.push(x + nx * hw + rx * hw * side, ny * hw + ry * hw * side, z);
				uv.push(0.5 + (side * hw) / w, (z2 - z) / (z2 - z0));
			}
		for (const r of [0, 2]) idx.push(first + r, first + r + 1, first + r + 2, first + r + 1, first + r + 3, first + r + 2);
	}
	const g = new THREE.BufferGeometry();
	g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
	g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
	g.setIndex(idx);
	const sides = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ map, side: THREE.DoubleSide })),
		dark = new THREE.MeshBasicMaterial({ color: 0x101216, side: THREE.DoubleSide });
	const cap = (hw: number, z: number) => {
		const m = new THREE.Mesh(new THREE.PlaneGeometry(2 * hw, 2 * hw), dark);
		m.position.set(x, 0, z);
		return m;
	};
	return new THREE.Group().add(sides, cap(ring / 2, z0), cap(w / 2, z2));
}

/** setTime(f): the crowd at source frame f (from the scene's first). */
export type Arena = THREE.Group & {
	keepOut?: THREE.Box3;
	/** the scoreboard over centre ice, when the atlas has one */
	scoreboard?: THREE.Object3D;
	setTime: (f: number) => void;
};

export async function buildArena(S: SceneData, base: string): Promise<Arena> {
	const atlas = await loadAtlas(base, S.rink),
		stands = new THREE.Group(),
		crowd = projected(atlas, S.rink, S.source.camera),
		step = (e0: number, e1: number, x: number) => {
			const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
			return t * t * (3 - 2 * t);
		},
		sheet = ice(S.rink, atlas, stands),
		panes = glass(S.rink),
		g: Arena = Object.assign(new THREE.Group(), {
			setTime(f: number) {
				const after = S.puck.shot ? (f - S.puck.shot.in_net) / S.fps - CROWD.REACT : -Infinity;
				crowd.uniforms.crowdT!.value = f / S.fps;
				crowd.uniforms.cheer!.value = step(0, CROWD.RISE, after) * (1 - step(CROWD.HOLD, CROWD.END, after));
			},
		});
	const prof = profile(S.rink, atlas.crowdTop),
		[sdWall, top] = prof.at(-1)!;
	arenaLight.roofZ.value = top;
	arenaLight.rinkHalf.value.set(S.rink.hx, S.rink.hy);
	stands.add(new THREE.Mesh(sweep(S.rink, prof), crowd), panes, roof(S.rink, sdWall, top));
	if (atlas.scoreboard) {
		// the space a free camera is kept out of
		const board = scoreboard(atlas.scoreboard);
		stands.add(board);
		g.scoreboard = board;
		g.keepOut = new THREE.Box3().setFromObject(board).expandByScalar(0.5);
	}
	stands.userData.ao = false; // contact shading for things on the ice (post.ts): the stands' stepped rows are no contact
	g.add(sheet, stands);
	return g;
}
