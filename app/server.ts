import { mkdir, readdir, stat } from "node:fs/promises";
import { join } from "node:path";
import { brotliCompressSync, constants } from "node:zlib";
import tailwind from "bun-plugin-tailwind";
import index from "./public/index.html";
import type { SceneData, SceneInfo } from "./public/replay/types";

const VIDEO = Bun.file(
	process.env.VIDEO ?? join(import.meta.dir, "../store/data/20251231_115035_SpenglerCup_Kam1 31.12.25 Game 11.mp4"),
);
// The goals' reconstructed 3D scenes (pipeline/goal/scene_build.py, into store/scenes), served as they are: nothing is copied.
const SCENES = process.env.SCENES ?? join(import.meta.dir, "../store/scenes");
// The AR overlay's per-segment camera and tracks (pipeline/calib/scripts/export_ar.py).
const AR = process.env.AR ?? join(import.meta.dir, "../store/ar");
// The site's data from the match: the feed, the players' stats, the moments and their frames (dvc.yaml's game and moments stages).
const SITE = process.env.SITE ?? join(import.meta.dir, "../store/site");
// Inter's font files, each fetched only when the page shows its characters (the @font-face rules in index.html).
const FONTS = join(import.meta.dir, "node_modules/@fontsource-variable/inter/files");
const startedAt = Date.now();
const arIndex = await Bun.file(join(AR, "index.json"))
	.json()
	.catch(() => null);
// The live edge starts a few seconds before the first AR window that contains a goal, so the demo plays straight into it.
const startOffset = Number(process.env.START ?? (arIndex ? arIndex.defaultStart - 6 : 5280));

function serveVideo(req: Request): Response {
	const size = VIDEO.size;
	const headers = {
		"Content-Type": "video/mp4",
		"Accept-Ranges": "bytes",
		"Cache-Control": "no-store",
	};
	const match = /^bytes=(\d*)-(\d*)$/.exec(req.headers.get("range") ?? "");
	if (!match) {
		return new Response(VIDEO, { headers: { ...headers, "Content-Length": String(size) } });
	}
	let start = match[1] ? Number(match[1]) : size - Number(match[2]);
	let end = match[1] && match[2] ? Number(match[2]) : size - 1;
	end = Math.min(end, size - 1);
	start = Math.max(start, 0);
	if (start > end) {
		return new Response(null, { status: 416, headers: { "Content-Range": `bytes */${size}` } });
	}
	return new Response(VIDEO.slice(start, end + 1), {
		status: 206,
		headers: {
			...headers,
			"Content-Range": `bytes ${start}-${end}/${size}`,
			"Content-Length": String(end - start + 1),
		},
	});
}

// JSON and binaries go gzipped (once, kept in GZIPPED); the body files are gzip streams already and go as they are.
const GZIPPED = join(import.meta.dir, ".cache/gz"),
	SQUEEZE = /\.(json|bin)$/,
	squeezing = new Map<string, Promise<string>>();

/**
 * Files under `dir`, addressed by the part of the URL after `prefix`; checked again on each visit (no-cache), a 304 while they are
 * as they were (ETag from their size and time).
 */
function serveFrom(dir: string, prefix: string) {
	return async (req: Request): Promise<Response> => {
		const path = join(dir, decodeURIComponent(new URL(req.url).pathname.slice(prefix.length))),
			file = Bun.file(path);
		if (!path.startsWith(`${dir}/`) || !(await file.exists())) return new Response("Not found", { status: 404 });
		const { size, mtimeMs } = await stat(path),
			version = `${size.toString(36)}-${Math.round(mtimeMs).toString(36)}`,
			gzip = SQUEEZE.test(path) && /\bgzip\b/.test(req.headers.get("Accept-Encoding") ?? ""),
			etag = `"${version}${gzip ? "-gz" : ""}"`,
			headers: Record<string, string> = {
				ETag: etag,
				"Last-Modified": new Date(mtimeMs).toUTCString(),
				"Cache-Control": "no-cache",
				Vary: "Accept-Encoding",
			};
		if (req.headers.get("If-None-Match") === etag) return new Response(null, { status: 304, headers });
		if (!gzip) return new Response(file, { headers });
		const out = join(GZIPPED, `${Bun.hash(path).toString(36)}-${version}.gz`);
		let job = squeezing.get(out);
		if (!job) {
			job = (async () => {
				if (!(await Bun.file(out).exists())) {
					await mkdir(GZIPPED, { recursive: true });
					await Bun.write(out, Bun.gzipSync(await file.bytes(), { level: 9 }));
				}
				return out;
			})().finally(() => squeezing.delete(out));
			squeezing.set(out, job);
		}
		return new Response(Bun.file(await job), { headers: { ...headers, "Content-Encoding": "gzip", "Content-Type": file.type } });
	};
}

type Built = { type: string; bytes: Uint8Array<ArrayBuffer>; br?: Uint8Array<ArrayBuffer>; gzip?: Uint8Array<ArrayBuffer> };

/**
 * The page and its bundle for production: built once at start (minified) and kept in memory, the text files with a Brotli and a
 * gzip copy. The bundle's files have a hash in their name and are cached for good; every other path is the page, checked on each
 * visit.
 */
async function buildSite(): Promise<(req: Request) => Response> {
	const built = await Bun.build({
		entrypoints: [join(import.meta.dir, "public/index.html")],
		minify: true,
		publicPath: "/",
		plugins: [tailwind],
		define: { "process.env.NODE_ENV": JSON.stringify("production") },
	});
	if (!built.success) throw new AggregateError(built.logs, "the site did not build");
	const files = new Map<string, Built>();
	for (const out of built.outputs) {
		const bytes = new Uint8Array(await out.arrayBuffer()),
			text = /^(text\/|application\/(javascript|json)|image\/svg)/.test(out.type);
		files.set(out.path.replace(/^\./, ""), {
			type: out.type,
			bytes,
			...(text && {
				br: new Uint8Array(brotliCompressSync(bytes, { params: { [constants.BROTLI_PARAM_QUALITY]: 11 } })),
				gzip: Bun.gzipSync(bytes, { level: 9 }),
			}),
		});
	}
	const page = files.get("/index.html");
	if (!page) throw new Error("the site's build has no index.html");
	return (req) => {
		const path = new URL(req.url).pathname,
			file = path === "/index.html" ? undefined : files.get(path),
			{ type, bytes, br, gzip } = file ?? page,
			accept = req.headers.get("Accept-Encoding") ?? "",
			encoding = br && /\bbr\b/.test(accept) ? "br" : gzip && /\bgzip\b/.test(accept) ? "gzip" : null;
		return new Response((encoding === "br" ? br : encoding === "gzip" ? gzip : null) ?? bytes, {
			headers: {
				"Content-Type": type,
				"Cache-Control": file ? "public, max-age=31536000, immutable" : "no-cache",
				Vary: "Accept-Encoding",
				...(encoding && { "Content-Encoding": encoding }),
			},
		});
	};
}

/** Every scene with a scene.json. */
async function listScenes(): Promise<SceneInfo[]> {
	const names = (await readdir(SCENES)).sort();
	const built = (
		await Promise.all(names.map(async (n) => ((await Bun.file(join(SCENES, n, "scene.json")).exists()) ? n : null)))
	).filter((n) => n !== null);
	return Promise.all(
		built.map(async (id) => {
			const S: SceneData = await Bun.file(join(SCENES, id, "scene.json")).json();
			const scorer = S.people.find((p) => p.key === S.shots.pov?.own);
			return {
				id,
				from: S.source.start + S.frame0 / S.fps,
				to: S.source.start + (S.frame0 + S.frames - 1) / S.fps,
				scorer: scorer ? { name: scorer.name, jersey: scorer.jersey, team: scorer.team } : null,
				shots: Object.keys(S.shots),
			};
		}),
	);
}

Bun.serve({
	port: Number(process.env.PORT ?? 3000),
	routes: {
		// every other path is a page of the client-side router: Bun's live bundle while developing, a built one in production
		"/*": process.env.NODE_ENV === "production" ? await buildSite() : index,
		"/video": serveVideo,
		"/site/*": serveFrom(SITE, "/site/"),
		"/ar/*": serveFrom(AR, "/ar/"),
		"/fonts/*": serveFrom(FONTS, "/fonts/"),
		"/assets/*": serveFrom(join(import.meta.dir, "public/assets"), "/assets/"),
		"/scenes/*": serveFrom(SCENES, "/scenes/"),
		"/api/scenes": async () => Response.json(await listScenes()),
		"/api/clock": () => Response.json({ startedAt, startOffset, now: Date.now() }),
	},
	fetch: () => new Response("Not found", { status: 404 }),
});

console.log(`live on http://localhost:${process.env.PORT ?? 3000}`);
