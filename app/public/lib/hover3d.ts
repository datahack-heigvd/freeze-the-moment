// A moment card's frame as a window on a small baked 3D world: the card stays put and the pointer moves the point of view inside it,
// x round the play and y up and down (up, higher), eased on a spring that overshoots a little.
// - A goal rebuilt in 3D has an orbit strip (scripts/build-orbits.ts): a grid of rendered views, each with its depth, and its still
//   (the middle view) at rest. Between views, the four nearest are each reprojected to the exact camera the pointer asks for, by
//   their depth, and blended by nearness and by how well each lands, so what one view could not see another fills in.
// - Any other card has a depth map (scripts/build-depth.ts) and parallax round the middle of the frame.
// Two WebGL canvases serve every card: one is lent to the card hovered and taken back once it is at rest again, so a card the
// pointer leaves for the next eases back on its own; it joins a card only once it has drawn that card's own pictures, so a card
// never shows another's. The pictures load on the first hover, at low priority and
// after the open replay's scene, and are dropped if the card is scrolled away before they are in. On touch screens the card crossing the middle of the screen drifts slowly, or follows the phone's tilt where the browser
// reports it.
// Nothing moves for prefers-reduced-motion.
import { afterScene } from "./priority";
import { orbitStrips } from "./site";

/** An orbit strip: cols × rows views swinging `swing`° round and `rise`° up and down from the middle one at `lift`° up, `distance` m
 * from the pivot through a `fov`° lens, `tiles` a row in the images; depth as nearness from `nearest` (1) to `farthest` m (0). */
export type OrbitStrip = {
	cols: number;
	rows: number;
	swing: number;
	rise: number;
	tiles: number;
	lift: number;
	distance: number;
	fov: number;
	nearest: number;
	farthest: number;
};

const STIFF = 110,
	DAMP = 0.62,
	DRIFT = 0.35,
	ROLL = 30,
	PARALLAX = 0.05, // 1/s² spring; of critical; turns a second, halved; degrees of phone tilt for the whole swing; of the width
	INTENT = 150; // ms the pointer rests on a card before onIntent

export const stripOf = (id: number) => orbitStrips[id] ?? null;
export const stillUrl = (id: number) => `/site/moments/${id}/still.webp`;
const stripUrls = (id: number) => [`/site/moments/${id}/orbit.webp`, `/site/moments/${id}/orbit-depth.webp`];
const depthUrl = (id: number) => `/site/moments/${id}/depth.png`;

type Card = { frame: HTMLElement; layer: HTMLElement; thumb: string; onIntent?: () => void };

/** Starts the effect on a card's frame, the canvas lent into `layer`; `onIntent` once the pointer rests on it. Returns its stop. */
export function hover3d(id: number, { frame, layer, thumb, onIntent }: Card): () => void {
	let intent: ReturnType<typeof setTimeout> | undefined;
	const rest = () => {
		if (onIntent && intent === undefined) intent = setTimeout(onIntent, INTENT);
	};
	const restless = () => {
		clearTimeout(intent);
		intent = undefined;
	};
	frame.addEventListener("pointerenter", rest);
	frame.addEventListener("pointerleave", restless);
	if (matchMedia("(prefers-reduced-motion: reduce)").matches)
		return () => {
			restless();
			frame.removeEventListener("pointerenter", rest);
			frame.removeEventListener("pointerleave", restless);
		};
	const urls = stripOf(id) ? stripUrls(id) : [thumb, depthUrl(id)],
		// -1..1 across and down
		goal = { x: 0, y: 0 },
		now = { x: 0, y: 0 },
		speed = { x: 0, y: 0 };
	let active = false,
		mine: Stage | null = null,
		raf = 0,
		last = 0,
		drifting = false,
		rolled = false;

	const step = (t: number) => {
		raf = 0;
		const dt = Math.min((t - (last || t)) / 1000, 1 / 20);
		last = t;
		if (drifting && !rolled) {
			goal.x = Math.sin((t / 1000) * DRIFT * Math.PI) * 0.85;
			goal.y = Math.sin((t / 1000) * DRIFT * 0.62 * Math.PI) * 0.6;
		}
		let moving = drifting;
		for (const k of ["x", "y"] as const) {
			speed[k] += ((goal[k] - now[k]) * STIFF - speed[k] * 2 * DAMP * Math.sqrt(STIFF)) * dt;
			now[k] += speed[k] * dt;
			if (Math.abs(goal[k] - now[k]) > 1e-3 || Math.abs(speed[k]) > 1e-3) moving = true;
		}
		if (moving || active) mine?.draw(layer, id, now.x, now.y);
		else {
			mine?.release(layer); // at rest the still underneath is the same picture
			mine = null;
		}
		if (moving) wake();
		else last = 0;
	};
	const wake = () => {
		if (!raf) raf = requestAnimationFrame(step);
	};

	const enter = () => {
		active = true;
		mine = stageFor(layer);
		mine.lend(layer, id, urls).then(
			(ok) => ok && wake(),
			() => {},
		);
	};
	const leave = () => {
		active = false;
		goal.x = goal.y = 0;
		wake();
	};
	const move = (e: PointerEvent) => {
		if (e.pointerType === "touch") return;
		const r = frame.getBoundingClientRect();
		if (!active) enter();
		goal.x = Math.min(Math.max(((e.clientX - r.left) / r.width) * 2 - 1, -1), 1);
		goal.y = Math.min(Math.max(((e.clientY - r.top) / r.height) * 2 - 1, -1), 1);
		wake();
	};
	const out = (e: PointerEvent) => e.pointerType !== "touch" && leave();
	frame.addEventListener("pointermove", move);
	frame.addEventListener("pointerleave", out);

	// touch: drift while crossing the middle of the screen (one card at a time has the canvas), or follow the phone's tilt
	const roll = (e: DeviceOrientationEvent) => {
		if (e.gamma === null || !drifting) return;
		rolled = true;
		goal.x = Math.min(Math.max(e.gamma / ROLL, -1), 1);
		goal.y = Math.min(Math.max(((e.beta ?? 45) - 45) / ROLL, -1), 1);
		wake();
	};
	const seen = new IntersectionObserver(
		([entry]) => {
			if (entry!.isIntersecting) {
				drifting = true;
				enter();
			} else if (drifting) {
				drifting = false;
				leave();
			}
		},
		{ rootMargin: "-45% 0px -45% 0px" },
	);
	if (matchMedia("(hover: none)").matches) {
		seen.observe(frame);
		addEventListener("deviceorientation", roll);
	}
	// scrolled well away: let go of the canvas, and of pictures still on their way
	const near = new IntersectionObserver(
		([entry]) => {
			if (entry!.isIntersecting) return;
			for (const s of stages) s.release(layer);
			cancel(id);
		},
		{ rootMargin: "50%" },
	);
	near.observe(frame);

	return () => {
		restless();
		frame.removeEventListener("pointerenter", rest);
		frame.removeEventListener("pointerleave", restless);
		near.disconnect();
		frame.removeEventListener("pointermove", move);
		frame.removeEventListener("pointerleave", out);
		seen.disconnect();
		removeEventListener("deviceorientation", roll);
		cancelAnimationFrame(raf);
		for (const s of stages) s.release(layer);
	};
}

const VERTEX =
	"#version 300 es\nin vec2 p; out vec2 uv; void main() { uv = vec2(p.x, -p.y) * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }";

// The four nearest views, each reprojected to the wanted camera. A camera is placed as build-orbits.ts placed it: on a sphere round
// the pivot (the origin here), looking at it, up +z. For each pixel p of the wanted view, Newton steps find the pixel q of a view
// whose surface (unprojected by its depth) projects onto p; how far it still misses tells how much to trust it there.
const ORBIT = `#version 300 es
precision highp float;
uniform sampler2D color, depth;
uniform vec2 grid, at, steps, inset; uniform float tiles, tileRows, lift, dist, tanHalf, aspect, nearest, farthest;
in vec2 uv; out vec4 outColor;
struct Cam { vec3 c, r, u, f; };
Cam camAt(vec2 g) {
  float a = (g.x - (grid.x - 1.0) * 0.5) * steps.x, e = lift - (g.y - (grid.y - 1.0) * 0.5) * steps.y;
  vec3 d = vec3(cos(e) * cos(a), cos(e) * sin(a), sin(e));
  Cam k; k.c = d * dist; k.f = -d; k.r = normalize(cross(k.f, vec3(0.0, 0.0, 1.0))); k.u = cross(k.r, k.f); return k;
}
vec2 tileUv(vec2 g, vec2 q) {
  float v = g.y * grid.x + g.x;
  return (vec2(mod(v, tiles), floor(v / tiles)) + clamp(q, inset, 1.0 - inset)) / vec2(tiles, tileRows);
}
float zOf(vec2 g, vec2 q) { return 1.0 / (texture(depth, tileUv(g, q)).r * (1.0 / nearest - 1.0 / farthest) + 1.0 / farthest); }
vec3 unproject(Cam k, vec2 q, float z) { vec2 n = vec2(q.x * 2.0 - 1.0, 1.0 - q.y * 2.0) * tanHalf * z; return k.c + k.f * z + k.r * n.x * aspect + k.u * n.y; }
vec2 project(Cam k, vec3 P) { vec3 d = P - k.c; float z = dot(d, k.f); vec2 n = vec2(dot(d, k.r) / aspect, dot(d, k.u)) / (z * tanHalf); return vec2(n.x * 0.5 + 0.5, 0.5 - n.y * 0.5); }
void main() {
  Cam want = camAt(at);
  vec2 g0 = min(floor(at), grid - 2.0), f = at - g0;
  vec4 sum = vec4(0.0);
  for (int n = 0; n < 4; n++) {
    vec2 g = g0 + vec2(n % 2, n / 2), w2 = mix(1.0 - f, f, vec2(n % 2, n / 2));
    float w = w2.x * w2.y;
    if (w < 1e-4) continue;
    Cam k = camAt(g);
    vec2 q = uv;
    for (int i = 0; i < 6; i++) q += uv - project(want, unproject(k, q, zOf(g, q)));
    vec2 miss = (uv - project(want, unproject(k, q, zOf(g, q)))) * vec2(aspect, 1.0) * 400.0;
    float inside = step(0.0, q.x) * step(q.x, 1.0) * step(0.0, q.y) * step(q.y, 1.0);
    float trust = w * inside * exp(-dot(miss, miss)) + 1e-6 * w;
    sum += vec4(texture(color, tileUv(g, q)).rgb, 1.0) * trust;
  }
  outColor = vec4(sum.rgb / sum.a, 1.0);
}`;

// One picture moved round the middle of the frame: each pixel looks up the frame shifted by how much nearer it is (white near) than
// the middle, times the point of view's offset, refined over a few steps so near edges cover what is behind them.
const PARALLAX_SHADER = `#version 300 es
precision highp float;
uniform sampler2D color, depth; uniform vec2 tilt; in vec2 uv; out vec4 outColor;
void main() {
  float focus = texture(depth, vec2(0.5)).r;
  vec2 at = uv;
  for (int i = 0; i < 6; i++) at = uv + tilt * (texture(depth, at).r - focus);
  outColor = vec4(texture(color, at).rgb, 1.0);
}`;

// A card's two pictures (colour and depth), downloaded once for both canvases.
const files = new Map<number, { blobs: Promise<Blob[]>; abort: AbortController; done: boolean }>();
const fetched = (id: number, urls: string[]) => {
	let f = files.get(id);
	if (!f) {
		const abort = new AbortController(),
			entry: { blobs: Promise<Blob[]>; abort: AbortController; done: boolean } = {
				abort,
				done: false,
				blobs: afterScene().then(() =>
					Promise.all(
						urls.map((u) =>
							fetch(u, { signal: abort.signal, priority: "low" }).then((r) =>
								r.ok ? r.blob() : Promise.reject(new Error(`${u}: ${r.status}`)),
							),
						),
					),
				),
			};
		entry.blobs.then(
			() => (entry.done = true),
			() => files.get(id) === entry && files.delete(id),
		);
		files.set(id, (f = entry));
	}
	return f.blobs;
};

/** Drop card `id`'s pictures if they are still on their way. */
function cancel(id: number) {
	const f = files.get(id);
	if (!f || f.done) return;
	f.abort.abort();
	files.delete(id);
}

type Stage = ReturnType<typeof makeStage>;
const STAGES = 2,
	stages: Stage[] = [];
/** The canvas for `into`: the one it has, else a free one, else the one lent longest ago. */
const stageFor = (into: HTMLElement) =>
	stages.find((s) => s.owns(into)) ??
	stages.find((s) => s.free()) ??
	(stages.length < STAGES ? stages[stages.push(makeStage()) - 1]! : stages.reduce((a, b) => (a.lentAt() < b.lentAt() ? a : b)));

// A canvas: owned by one card at a time, with that card's two pictures on its textures. It is put into the owner's frame only by
// draw, once those pictures are up and drawn, so it never shows a frame of the card before.
function makeStage() {
	let canvas: HTMLCanvasElement | null = null,
		gl: WebGL2RenderingContext | null = null,
		owner: HTMLElement | null = null,
		lent = 0,
		shown = -1,
		uploading: { id: number; done: Promise<void> } | null = null;
	const programs: { orbit: WebGLProgram; parallax: WebGLProgram } = {} as never,
		textures: WebGLTexture[] = [];

	const setup = () => {
		canvas = document.createElement("canvas");
		canvas.className = "absolute inset-0 block size-full";
		gl = canvas.getContext("webgl2");
		if (!gl) return;
		const compile = (fragment: string) => {
			const program = gl!.createProgram()!;
			for (const [type, src] of [
				[gl!.VERTEX_SHADER, VERTEX],
				[gl!.FRAGMENT_SHADER, fragment],
			] as const) {
				const s = gl!.createShader(type)!;
				gl!.shaderSource(s, src);
				gl!.compileShader(s);
				gl!.attachShader(program, s);
			}
			gl!.bindAttribLocation(program, 0, "p");
			gl!.linkProgram(program);
			gl!.useProgram(program);
			gl!.uniform1i(gl!.getUniformLocation(program, "color"), 0);
			gl!.uniform1i(gl!.getUniformLocation(program, "depth"), 1);
			return program;
		};
		programs.orbit = compile(ORBIT);
		programs.parallax = compile(PARALLAX_SHADER);
		gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
		gl.enableVertexAttribArray(0);
		gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
		for (let i = 0; i < 2; i++) {
			const t = gl.createTexture()!;
			gl.activeTexture(gl.TEXTURE0 + i);
			gl.bindTexture(gl.TEXTURE_2D, t);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
			textures.push(t);
		}
	};

	// decoded only while shown: a strip's two pictures are some 40 MB once decoded
	const upload = (id: number) => {
		if (shown === id) return Promise.resolve();
		if (uploading?.id === id) return uploading.done;
		const done = (async () => {
			const pictures = await Promise.all(
				(await files.get(id)!.blobs).map((b) => createImageBitmap(b, { premultiplyAlpha: "none", colorSpaceConversion: "none" })),
			);
			if (uploading?.id === id && gl) {
				for (const [i, p] of pictures.entries()) {
					gl.activeTexture(gl.TEXTURE0 + i);
					gl.bindTexture(gl.TEXTURE_2D, textures[i]!);
					gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, p);
				}
				shown = id;
				uploading = null;
			}
			for (const p of pictures) p.close();
		})();
		uploading = { id, done };
		return done;
	};

	const render = (into: HTMLElement, id: number, x: number, y: number) => {
		const scale = Math.min(devicePixelRatio, 2),
			w = Math.round(into.clientWidth * scale),
			h = Math.round(into.clientHeight * scale);
		if (canvas!.width !== w || canvas!.height !== h) [canvas!.width, canvas!.height] = [w, h];
		gl!.viewport(0, 0, w, h);
		const strip = stripOf(id);
		if (strip) {
			const p = programs.orbit,
				u = (name: string) => gl!.getUniformLocation(p, name),
				rad = Math.PI / 180;
			gl!.useProgram(p);
			gl!.uniform2f(u("grid"), strip.cols, strip.rows);
			gl!.uniform2f(u("at"), ((x + 1) / 2) * (strip.cols - 1), ((y + 1) / 2) * (strip.rows - 1));
			gl!.uniform2f(
				u("steps"),
				(2 * strip.swing * rad) / Math.max(strip.cols - 1, 1),
				(2 * strip.rise * rad) / Math.max(strip.rows - 1, 1),
			);
			gl!.uniform2f(u("inset"), 0.5 / 640, 0.5 / 360);
			gl!.uniform1f(u("tiles"), strip.tiles);
			gl!.uniform1f(u("tileRows"), Math.ceil((strip.cols * strip.rows) / strip.tiles));
			gl!.uniform1f(u("lift"), strip.lift * rad);
			gl!.uniform1f(u("dist"), strip.distance);
			gl!.uniform1f(u("tanHalf"), Math.tan((strip.fov * rad) / 2));
			gl!.uniform1f(u("aspect"), 16 / 9);
			gl!.uniform1f(u("nearest"), strip.nearest);
			gl!.uniform1f(u("farthest"), strip.farthest);
		} else {
			gl!.useProgram(programs.parallax);
			gl!.uniform2f(gl!.getUniformLocation(programs.parallax, "tilt"), -x * PARALLAX * 2, -y * PARALLAX * 2 * (w / h));
		}
		gl!.drawArrays(gl!.TRIANGLE_STRIP, 0, 4);
	};

	return {
		owns: (into: HTMLElement) => owner === into,
		free: () => owner === null,
		lentAt: () => lent,
		/** Make `into` the owner, for card `id`; true once its pictures are up and it still owns the canvas. */
		async lend(into: HTMLElement, id: number, urls: string[]) {
			if (owner !== into) canvas?.remove(); // the card before shows its own still again
			owner = into;
			lent = performance.now();
			await fetched(id, urls);
			if (owner !== into) return false;
			if (!canvas) setup();
			if (!gl) return false;
			await upload(id);
			return owner === into && shown === id;
		},
		release(from: HTMLElement) {
			if (owner !== from) return;
			owner = null;
			canvas?.remove();
		},
		draw(into: HTMLElement, id: number, x: number, y: number) {
			if (!gl || !canvas || owner !== into || shown !== id) return;
			render(into, id, x, y);
			if (canvas.parentElement !== into) into.append(canvas); // with this card's frame already drawn
		},
	};
}
