// The arena's light, one model for everything on the ice (people, their gear, sticks, goals). An ice rink is lit from straight
// above by a grid of lamps under the roof (a lamp every LAMP m along each truss, trusses every TRUSS m), and the white ice throws
// much of it back up, so a player is lit from above and from below and barely from the dark stands at the sides.
//   Diffuse: the roof's light from above, the ice's from below (cooler), scaled so a surface facing sideways keeps its colour as
// the broadcast saw it (the textures are the broadcast's colours, already lit).
//   The lamps' radiance follows from the ice: lit to about 1 (it shows near its albedo), the roof over it must average about 1/pi,
// so a lamp (a 0.3 m disc in each LAMP x TRUSS cell) is near 95: a sharp reflection of one is far over white, and blooms.
//   Specular: GGX's split sum. What a surface reflects along r is traced: up, to the roof plane, where the lamp grid is read blurred
// by the reflection's spread (roughness times the distance), so a helmet shows each lamp it faces; down, the ice; round the
// horizon, the stands. Weighted by an analytic fit of the GGX lobe's energy (Karis) and Schlick's Fresnel.
//   Cloth has a sheen (a soft rim of the light behind and round it, as woven fibres scatter), a helmet or stick shaft a clear coat (COAT_ROUGH).
// Linear colours in, linear colours out.
import * as THREE from "three";

/** The roof's lamps: one every LAMP m along each truss, the trusses every TRUSS m across the rink's length (m). */
export const LAMP = 3.5,
	TRUSS = 7;
/** How much more light a surface facing the roof takes than one facing sideways, and how much less one facing the ice: the
 * lamps overhead outshine the ice's bounce, so a player is bright over the shoulders and darker under the arms (as broadcast). */
const TOP = 0.42;

/** The arena's uniforms every lit material shares (buildArena sets them): the roof's height and the rink's half size (m). */
export const arenaLight = {
	roofZ: { value: 18 },
	rinkHalf: { value: new THREE.Vector2(30, 15) },
};

/** GLSL: `envBrdf(rough, nv)`, GGX's split-sum energy (scale and bias of f0) for roughness and n.v (Karis' analytic fit). */
export const ENV_BRDF_GLSL = /* glsl */ `
vec2 envBrdf(float rough, float nv) {
  vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022), c1 = vec4(1.0, 0.0425, 1.04, -0.04), r = rough * c0 + c1;
  float a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
  return vec2(-1.04, 1.04) * a004 + r.zw;
}`;

/**
 * GLSL: `arenaLit(Surface s, vec3 p, vec3 v)`, the light a surface at p seen from direction v sends back. Surface: albedo (linear),
 * n (unit, world), rough (0 mirror to 1 matte), f0 (reflectance head on; 0.04 for plastics and cloth), sheen (0 to 1), coat (a clear
 * coat's strength, 0 to 1), ao (the occlusion of light, 0 to 1).
 */
export const LIGHT_GLSL = /* glsl */ `
uniform float roofZ; uniform vec2 rinkHalf;
struct Surface { vec3 albedo; vec3 n; float rough; float f0; float sheen; float coat; float ao; };
const vec3 ROOF_LAMP = vec3(95.0, 91.0, 83.0), ROOF_DARK = vec3(0.03, 0.03, 0.045), ICE_BOUNCE = vec3(0.78, 0.82, 0.88),
  STANDS = vec3(0.06, 0.055, 0.06);
const float LAMP = ${LAMP.toFixed(2)}, TRUSS = ${TRUSS.toFixed(2)}, COAT_ROUGH = 0.34;
float overIce(vec2 q) { vec2 d = abs(q) - rinkHalf; return 1.0 - smoothstep(0.0, 8.0, max(d.x, d.y)); }
// the roof seen at q, blurred over spread m (at least a lamp's glare, 0.7 m: a reflection never shows the lamps as points) where they
// hang over the ice
vec3 roofAt(vec2 q, float spread) {
  vec2 c = (fract(vec2(q.x / LAMP, q.y / TRUSS) + 0.5) - 0.5) * vec2(LAMP, TRUSS);
  float w = max(0.7, spread), lamp = 0.025 / (w * w) * exp(-dot(c, c) / (w * w));   // a lamp's light spread over its blur, its energy kept
  float mean = 3.14159 * 0.025 / (LAMP * TRUSS);                                     // the grid's mean, which a wide blur tends to
  float grid = mix(lamp, mean, smoothstep(0.35 * LAMP, LAMP, w));
  return ROOF_DARK + ROOF_LAMP * grid * overIce(q);
}
// what is seen from p along r, blurred by roughness
vec3 arenaSeen(vec3 p, vec3 r, float rough) {
  float a = rough * rough, horizon = 0.06 + 0.5 * a;
  vec3 stands = STANDS, ice = ICE_BOUNCE * (0.55 + 0.45 * smoothstep(0.0, 0.5, -r.z));
  vec3 up = stands;
  if (r.z > 0.02) {
    float d = (roofZ - p.z) / r.z;                                                    // along r to the roof
    up = roofAt(p.xy + r.xy * d, a * d * 0.8 + 0.02 * d);
  }
  return r.z >= 0.0 ? mix(stands, up, smoothstep(0.0, horizon, r.z)) : mix(stands, ice, smoothstep(0.0, horizon, -r.z));
}
// light reaching a surface facing n, 1 facing sideways: TOP more facing the roof, TOP less and cooler facing the ice
vec3 arenaIrradiance(vec3 n) {
  return mix(vec3(0.93, 0.96, 1.02), vec3(1.02, 1.0, 0.97), 0.5 + 0.5 * n.z) * (1.0 + ${TOP.toFixed(2)} * n.z);
}
${ENV_BRDF_GLSL}
vec3 arenaLit(Surface s, vec3 p, vec3 v) {
  float nv = clamp(dot(s.n, v), 1e-3, 1.0);
  vec2 ab = envBrdf(s.rough, nv);
  vec3 spec = arenaSeen(p, reflect(-v, s.n), s.rough) * (s.f0 * ab.x + ab.y);
  float kd = 1.0 - (s.f0 * ab.x + ab.y);
  vec3 col = s.albedo * arenaIrradiance(s.n) * kd * s.ao + spec * mix(1.0, s.ao, 0.6);
  // sheen: fibres catch the light round the silhouette, most from the light behind and above (the roof, the ice)
  float rim = pow(1.0 - nv, 3.0);
  col += s.sheen * rim * mix(s.albedo, vec3(1.0), 0.35) * arenaIrradiance(-v) * 0.55 * s.ao;
  // a clear coat: a glossy reflection over the base (the lamps as soft streaks, not points: no coat is a mirror), its Fresnel
  // taking from what is under it
  if (s.coat > 0.0) {
    vec2 cb = envBrdf(COAT_ROUGH, nv); float fc = s.coat * (0.04 * cb.x + cb.y);
    col = col * (1.0 - fc) + arenaSeen(p, reflect(-v, s.n), COAT_ROUGH) * fc;
  }
  return col;
}`;

/** How a surface takes the light: roughness (0 mirror to 1 matte), reflectance head on (f0: 0.04 for plastics, paint and cloth,
 * far more for bare metal) and a clear coat's strength. */
export type Finish = { rough: number; f0: number; coat: number };

/** The gear's finishes. */
export const FINISH = {
	/** a helmet: glossy painted plastic under a clear coat */
	shell: { rough: 0.3, f0: 0.04, coat: 1 },
	/** a goalie's mask: painted fibreglass, satin */
	mask: { rough: 0.38, f0: 0.04, coat: 0.5 },
	/** a cage's coated wire */
	wire: { rough: 0.35, f0: 0.04, coat: 0.4 },
	/** a goalie's pads and gloves: synthetic leather */
	pad: { rough: 0.5, f0: 0.04, coat: 0.15 },
	/** a skate's blade holder: moulded plastic */
	holder: { rough: 0.3, f0: 0.04, coat: 0.3 },
	/** a skate's runner: polished steel */
	steel: { rough: 0.18, f0: 0.6, coat: 0 },
	/** a stick's shaft: carbon fibre under a clear coat */
	carbon: { rough: 0.45, f0: 0.04, coat: 0.7 },
	/** cloth tape on a blade or a knob */
	tape: { rough: 0.85, f0: 0.04, coat: 0 },
	/** a goal's painted steel frame */
	paint: { rough: 0.3, f0: 0.04, coat: 0.8 },
	/** a goal's padding and fabric */
	fabric: { rough: 0.75, f0: 0.04, coat: 0 },
} satisfies Record<string, Finish>;

const VERTEX = /* glsl */ `
varying vec3 vN, vP, vC;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0); vP = w.xyz; vN = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * viewMatrix * w;
#ifdef PAINTED
  vC = color;
#endif
}`;

/**
 * A material under the arena's light in `finish`, both sides lit: its colour the one texel of `map` at `uv` (a person's texture, where
 * the footage shows the gear), else `colour` (sRGB); painted: the geometry's own per-vertex colours (linear) instead.
 */
export function litMaterial(
	finish: Finish,
	{
		map = null,
		uv = [0, 0],
		colour = [1, 1, 1],
		painted = false,
	}: { map?: THREE.Texture | null; uv?: readonly number[]; colour?: readonly number[]; painted?: boolean } = {},
) {
	return new THREE.ShaderMaterial({
		uniforms: {
			map: { value: map },
			crown: { value: new THREE.Vector2(uv[0], uv[1]) },
			colour: { value: new THREE.Color().setRGB(colour[0]!, colour[1]!, colour[2]!, THREE.SRGBColorSpace) },
			finish: { value: new THREE.Vector3(finish.rough, finish.f0, finish.coat) },
			...arenaLight,
		},
		defines: painted ? { PAINTED: 1 } : map ? { MAP: 1 } : {},
		vertexColors: painted,
		side: THREE.DoubleSide,
		vertexShader: VERTEX,
		fragmentShader:
			LIGHT_GLSL +
			/* glsl */ `
uniform sampler2D map; uniform vec2 crown; uniform vec3 colour, finish; varying vec3 vN, vP, vC;
void main() {
#ifdef PAINTED
  vec3 albedo = vC;
#elif defined(MAP)
  vec3 albedo = texture2D(map, crown).rgb;
#else
  vec3 albedo = colour;
#endif
  vec3 n = normalize(vN) * (gl_FrontFacing ? 1.0 : -1.0), v = normalize(cameraPosition - vP);
  gl_FragColor = vec4(arenaLit(Surface(albedo, n, finish.x, finish.y, 0.0, finish.z, 1.0), vP, v), 1.0);
  #include <colorspace_fragment>
}`,
	});
}
