// The arena atlases as GLSL: atlasIceColor(xy) for a rink point on the ice, atlasOutColor(s, h) for the boards, ledge and stands by
// arc length s along the rink outline and height h along the profile. Colours come back linear; beyond the atlas (the roof) dark.
//
// With <base>/atlas.json (arena_atlas.py: the shared high-resolution atlas) the ice is 1 cm per texel, the boards band (h below
// band.top) 1 cm folded into strips stacked vertically, and the stands 2.5 cm; the band hands over to the stands just below its top.
// Without it, the goal's own atlas_ice.jpg and atlas_out.jpg from render.py. With atlas.json's "scoreboard" (scoreboard.py) also the
// centre-hung scoreboard's size and face texture. With stands.plain (stands_quilt.py) atlasPlain(s, h): 1 where the stands hold crowd
// laid out straight on rather than what the broadcast camera saw, to be read by height.
import * as THREE from "three";
import type { Rink } from "./types";
import { request, texture } from "./util";

const f = (x: number) => Number(x).toFixed(6);
const ROOF = "vec3(0.0067, 0.0067, 0.0080)";

export type Scoreboard = {
	x: number;
	width: number;
	ring: number;
	z: [number, number, number];
	file: string;
	map: THREE.Texture;
};
/**
 * The arena atlases' uniforms and GLSL. Besides the colours (top of this file): atlasGlare(p, r), the roof's radiance along r from
 * a point p on the ice (ice_glare.py: the lamps the broadcast saw mirrored in the ice, put back on the roof; 0 without it), linear.
 */
/** crowdTop: the height up the stands (m) the seated tiers end at, when the atlas knows it (stands_quilt.py). */
export type Atlas = { uniforms: Record<string, THREE.IUniform>; glsl: string; scoreboard?: Scoreboard; crowdTop?: number };

type Grid = { file: string; res: number; cols: number; rows: number };
type AtlasJson = {
	ice: Grid & { x0: number; y0: number };
	band: Grid & { strip_cols: number; strip_rows: number; strips: number; pad: number; top: number };
	stands: Grid & { plain?: string; crowd_top?: number };
	glare?: Grid & { matte: string; z: number; x0: number; y0: number; max: number };
	scoreboard?: Omit<Scoreboard, "map">;
};

async function legacy(base: string, rink: Rink): Promise<Atlas> {
	const [ice, out] = await Promise.all([texture(`${base}/atlas_ice.jpg`), texture(`${base}/atlas_out.jpg`)]);
	const [ir, ic] = rink.ice_size,
		[or, oc] = rink.out_size;
	return {
		uniforms: { atlasIce: { value: ice }, atlasOut: { value: out } },
		glsl: /* glsl */ `
uniform sampler2D atlasIce, atlasOut;
float atlasPlain(float s, float h) { return 0.0; }
float atlasGlare(vec3 p, vec3 r) { return 0.0; }
vec3 atlasIceColor(vec2 xy) {
  return texture2D(atlasIce, vec2(((xy.x + ${f(rink.hx)}) / ${f(rink.ice_res)} + 0.5) / ${f(ic)}, ((xy.y + ${f(rink.hy)}) / ${f(rink.ice_res)} + 0.5) / ${f(ir)})).rgb;
}
vec3 atlasOutColor(float s, float h) {
  vec3 colour = texture2D(atlasOut, vec2((s / ${f(rink.out_res)} + 0.5) / ${f(oc)}, (h / ${f(rink.out_res)} + 0.5) / ${f(or)})).rgb;
  return h > ${f(or * rink.out_res)} ? ${ROOF} : colour;
}`,
	};
}

export async function loadAtlas(base: string, rink: Rink): Promise<Atlas> {
	const r = await request(`${base}/atlas.json`).catch(() => null);
	if (!r?.ok) return legacy(base, rink);
	const m = (await r.json()) as AtlasJson,
		{ ice: I, band: B, stands: T, glare: G } = m;
	// with the glare taken out (ice_glare.py) the ice is its matte colour, the glare drawn from the roof for each view
	const [ice, band, stands] = await Promise.all([G ? G.matte : I.file, B.file, T.file].map((file) => texture(`${base}/${file}`)));
	const roof = G ? await texture(`${base}/${G.file}`) : null;
	if (roof) roof.colorSpace = THREE.NoColorSpace;
	const scoreboard = m.scoreboard && { ...m.scoreboard, map: await texture(`${base}/${m.scoreboard.file}`) };
	const plain = T.plain ? await texture(`${base}/${T.plain}`) : null;
	if (plain) plain.colorSpace = THREE.NoColorSpace;
	return {
		scoreboard,
		crowdTop: T.crowd_top,
		uniforms: {
			atlasIce: { value: ice },
			atlasBand: { value: band },
			atlasStands: { value: stands },
			atlasPlainMap: { value: plain },
			atlasRoof: { value: roof },
		},
		glsl: /* glsl */ `
uniform sampler2D atlasIce, atlasBand, atlasStands, atlasPlainMap, atlasRoof;
float atlasGlare(vec3 p, vec3 r) {
  ${
		G
			? `if (r.z <= 0.0) return 0.0;
  vec2 q = p.xy + r.xy * (${f(G.z)} - p.z) / r.z, uv = (q - vec2(${f(G.x0)}, ${f(G.y0)})) / vec2(${f(G.cols * G.res)}, ${f(G.rows * G.res)});
  float e = texture2D(atlasRoof, uv).r;
  return uv == clamp(uv, 0.0, 1.0) ? e * e * ${f(G.max)} : 0.0;`
			: "return 0.0;"
	}
}
float atlasPlain(float s, float h) {
  ${plain ? `return h > ${f(T.rows * T.res)} ? 0.0 : texture2D(atlasPlainMap, vec2(s / ${f(T.cols * T.res)}, h / ${f(T.rows * T.res)})).r;` : "return 0.0;"}
}
vec3 atlasIceColor(vec2 xy) {
  return texture2D(atlasIce, vec2(((xy.x - ${f(I.x0)}) / ${f(I.res)} + 0.5) / ${f(I.cols)}, ((xy.y - ${f(I.y0)}) / ${f(I.res)} + 0.5) / ${f(I.rows)})).rgb;
}
vec3 atlasBandColor(float s, float h) {
  float len = ${f(B.strip_cols * B.res)}, k = clamp(floor(s / len), 0.0, ${f(B.strips - 1)});
  float col = (s - k * len) / ${f(B.res)} + ${f(B.pad)}, row = k * ${f(B.strip_rows)} + ${f(B.pad)} + h / ${f(B.res)};
  return texture2D(atlasBand, vec2((col + 0.5) / ${f(B.cols)}, (row + 0.5) / ${f(B.rows)})).rgb;
}
vec3 atlasStandsColor(float s, float h) {
  vec3 colour = texture2D(atlasStands, vec2((s / ${f(T.res)} + 0.5) / ${f(T.cols)}, (h / ${f(T.res)} + 0.5) / ${f(T.rows)})).rgb;
  return h > ${f(T.rows * T.res)} ? ${ROOF} : colour;
}
vec3 atlasOutColor(float s, float h) {
  float t = smoothstep(${f(B.top - 0.25)}, ${f(B.top - 0.05)}, h);
  vec3 band = atlasBandColor(s, h), stands = atlasStandsColor(s, h);
  return mix(band, stands, t);
}`,
	};
}
