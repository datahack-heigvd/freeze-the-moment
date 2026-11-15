// The 3D goals' hover strips: each goal frozen at the shot's release and seen from a grid of cameras round the scorer and the puck,
// cols across swinging `swing`° either way, rows down rising `rise`° either way (orbit/capture.ts, rendered headless). Into
// store/site/moments/<moment>/: the views TILES a row in orbit.webp, their nearness the same way in orbit-depth.webp (lossless, as
// the card reprojects by it), and the middle view on its own (still.webp), the card's frame at rest. store/site/orbits.json lists
// the moments that have one, with the grid and the cameras.
// `bun scripts/build-orbits.ts [--grid 7x3] [scene ...]` builds those scenes' only, with that grid.
import { mkdtemp, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { $ } from "bun";
import { chromium } from "playwright";
import type { OrbitStrip } from "../public/lib/hover3d";
import type { SceneData } from "../public/replay/types";
import type { Orbit, OrbitCamera } from "./orbit/capture";
import capture from "./orbit/capture.html";
import { ASSETS, serveFrom } from "./serve";

const SCENES = process.env.SCENES ?? join(import.meta.dir, "../../store/scenes");
const GAME = join(import.meta.dir, "../../store/site/game.json");
const MOMENTS = join(import.meta.dir, "../../store/site/moments.json");
const THUMBS = join(import.meta.dir, "../../store/site/moments");
const OUT = join(import.meta.dir, "../../store/site/orbits.json");

const ORBIT: Orbit = { cols: 7, rows: 3, swing: 20, rise: 10, width: 640, height: 360 };
const TILES = 3; // views a row in the image: 1920 px wide, inside the 4096 px textures every phone takes
const QUALITY = 78;

const args = process.argv.slice(2),
	grid = args.indexOf("--grid");
if (grid >= 0) {
	const [cols, rows] = args.splice(grid, 2)[1]!.split("x").map(Number);
	Object.assign(ORBIT, { cols, rows });
}

type Event = { t: number; shotT?: number; kind: string };
const { events }: { events: Event[] } = await Bun.file(GAME).json();
const moments: number[] = await Bun.file(MOMENTS).json();

// every scene, with the goal moment whose shot falls in its frames
const names = (await readdir(SCENES)).sort(),
	built = (
		await Promise.all(names.map(async (n) => ((await Bun.file(join(SCENES, n, "scene.json")).exists()) ? n : null)))
	).filter((n) => n !== null);
const jobs: { scene: string; moment: number }[] = [];
for (const scene of built.filter((n) => !args.length || args.includes(n))) {
	const S: SceneData = await Bun.file(join(SCENES, scene, "scene.json")).json(),
		from = S.source.start + S.frame0 / S.fps,
		to = S.source.start + (S.frame0 + S.frames - 1) / S.fps,
		moment = moments.find((i) => {
			const e = events[i]!,
				at = e.shotT ?? e.t;
			return e.kind === "goal" && at >= from - 1 && at <= to + 1;
		});
	if (moment === undefined || !S.puck.shot)
		console.log(`${scene}: ${moment === undefined ? "no goal moment" : "no shot"}, skipped`);
	else jobs.push({ scene, moment });
}

const server = Bun.serve({
	port: 0,
	routes: {
		"/": capture,
		"/scenes/*": serveFrom(SCENES, "/scenes/"),
		"/assets/*": serveFrom(ASSETS, "/assets/"),
	},
});
const browser = await chromium.launch({
	args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
});
const strips: Record<string, OrbitStrip> = await Bun.file(OUT)
	.json()
	.catch(() => ({}));
const n = ORBIT.cols * ORBIT.rows,
	tile = `tile=${TILES}x${Math.ceil(n / TILES)}`,
	name = (i: number) => String(i).padStart(3, "0");
try {
	for (const { scene, moment } of jobs) {
		const started = performance.now(),
			page = await browser.newPage({ viewport: { width: ORBIT.width, height: ORBIT.height } });
		page.on("pageerror", (e) => console.error(`${scene}: ${e.message}`));
		await page.goto(server.url.href);
		await page.waitForFunction(() => "orbit" in window);
		type Shot = { views: { color: string; depth: string }[]; camera: OrbitCamera };
		const { views, camera }: Shot = await page.evaluate(
			([id, o]) => (window as unknown as { orbit: (id: string, o: Orbit) => Promise<Shot> }).orbit(id, o),
			[scene, ORBIT] as const,
		);
		await page.close();
		const dir = await mkdtemp(join(tmpdir(), "orbit-")),
			png = (url: string) => Buffer.from(url.slice(url.indexOf(",") + 1), "base64");
		await Promise.all(
			views.flatMap((v, i) => [
				Bun.write(join(dir, `c${name(i)}.png`), png(v.color)),
				Bun.write(join(dir, `d${name(i)}.png`), png(v.depth)),
			]),
		);
		const at = join(THUMBS, String(moment)),
			color = join(at, "orbit.webp"),
			depth = join(at, "orbit-depth.webp");
		await $`mkdir -p ${at}`;
		await $`ffmpeg -loglevel error -y -i ${join(dir, "c%03d.png")} -vf ${tile} -frames:v 1 -c:v libwebp -quality ${QUALITY} ${color}`;
		await $`ffmpeg -loglevel error -y -i ${join(dir, "d%03d.png")} -vf ${tile},format=gray -frames:v 1 -c:v libwebp -lossless 1 -compression_level 6 ${depth}`;
		await $`ffmpeg -loglevel error -y -i ${join(dir, `c${name((n - 1) / 2)}.png`)} -c:v libwebp -quality ${QUALITY} ${join(at, "still.webp")}`;
		await rm(dir, { recursive: true });
		strips[moment] = { cols: ORBIT.cols, rows: ORBIT.rows, swing: ORBIT.swing, rise: ORBIT.rise, tiles: TILES, ...camera };
		const kb = (f: string) => (Bun.file(f).size / 1e3).toFixed(0);
		console.log(
			`${scene} → moment ${moment}: ${kb(color)} + ${kb(depth)} KB depth, ${((performance.now() - started) / 1000).toFixed(0)} s`,
		);
	}
} finally {
	await browser.close();
	server.stop();
}
await Bun.write(OUT, `${JSON.stringify(strips)}\n`);
