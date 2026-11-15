import * as THREE from "three";
import type { BodyLayout, Rink } from "./types";

type BinType<T> = new (buffer: ArrayBuffer) => T;

/** Prefetches (prefetchScene) still landing, by URL and range; each settles without failing. */
const warming = new Map<string, Promise<unknown>>();
/** Ranges prefetched, which the HTTP cache does not keep for script requests: held until a load takes them, the oldest let go past HELD bytes. */
const held = new Map<string, { bytes: ArrayBuffer; status: number; headers: Headers }>(),
	HELD = 64 << 20;
const requestKey = (url: string, init?: RequestInit) => `${url} ${new Headers(init?.headers).get("Range") ?? ""}`;

/** `fetch`, once a prefetch of the same request has landed: then answered from the bytes it holds, or else the HTTP cache. */
export async function request(url: string, init?: RequestInit) {
	const key = requestKey(url, init);
	await warming.get(key);
	const kept = held.get(key);
	if (!kept) return fetch(url, init);
	held.delete(key);
	return new Response(kept.bytes, kept);
}

/** A request fetched whole at low priority ahead of the load that asks for it with `request`; its bytes. */
export function warm(url: string, init: RequestInit = {}): Promise<ArrayBuffer> {
	const key = requestKey(url, init),
		job = fetch(url, { ...init, priority: "low" }).then(async (r) => {
			if (!r.ok) throw new Error(`${url}: ${r.status}`);
			const bytes = await r.arrayBuffer();
			if (r.status === 206) hold(key, { bytes, status: r.status, headers: r.headers });
			return bytes;
		}),
		landed = job.then(
			() => {},
			() => {},
		);
	warming.set(key, landed);
	landed.then(() => warming.get(key) === landed && warming.delete(key));
	return job;
}

function hold(key: string, kept: { bytes: ArrayBuffer; status: number; headers: Headers }) {
	held.delete(key);
	held.set(key, kept);
	let total = 0;
	for (const k of [...held.keys()].reverse()) if ((total += held.get(k)!.bytes.byteLength) > HELD) held.delete(k);
}

export async function bin<T>(url: string, Type: BinType<T>): Promise<T> {
	const r = await request(url);
	if (!r.ok) throw new Error(`${url}: ${r.status}`);
	return new Type(await r.arrayBuffer());
}

/** A body file's first chunk ends by this share of its raw bytes, compressed (at most 0.32 in every scene so far). */
const HEAD = 0.35;
/** before a body file that came in broken is fetched again whole (it may have been caught while being written) */
const AGAIN_MS = 1000;

/** The bytes a body file's head range asks for: to past its first chunk. */
export const bodyHead = (V: number, B: BodyLayout) => Math.ceil(HEAD * 2 * Math.min(B.chunk, B.frames) * V * 3);

/**
 * A body file (scene_export.py), streamed: one gzip stream, flushed after each chunk of B.chunk frames, each chunk int16 [n, V, 3]
 * in B.step mm as low bytes then high bytes, row 0 as it is, row 1 its change from row 0, each later row its change in change (the
 * second difference). Fetched in two ranges, the head (to past the first chunk) at `priority`, the rest at low priority once `later`
 * is had, so everyone's first chunk (and all else starting takes) comes before anyone's rest, as few connections as the browser has. Each chunk is decoded into `v` (int16
 * [B.frames, V, 3] mm) as soon as its bytes are in; then `rows(n)` is told the n rows decoded so far. `bytes(n, of)`: the
 * compressed bytes in so far, of the file's. When the two ranges are not of one file (it was written again between them) or it
 * does not decode, it is logged and fetched again whole, once; `rows` only ever grows, the rows had held until the new ones pass them.
 */
export async function body(
	url: string,
	V: number,
	B: BodyLayout,
	v: Int16Array,
	rows: (n: number) => void,
	bytes: (n: number, of: number) => void,
	priority: RequestPriority,
	later: Promise<unknown>,
) {
	let had = 0;
	const grow = (n: number) => n > had && rows((had = n));
	try {
		await decode(await ranged(url, bodyHead(V, B), bytes, priority, later), V, B, v, grow);
	} catch (e) {
		console.error(`${url}: ${(e as Error).message} at frame ${had} of ${B.frames}; fetching it again whole`);
		await new Promise((r) => setTimeout(r, AGAIN_MS));
		try {
			const r = await fetch(url, { priority: had ? "low" : priority });
			if (!r.ok) throw new Error(`${r.status}`);
			await decode(joined([r], bytes, Number(r.headers.get("Content-Length"))), V, B, v, grow);
		} catch (e) {
			console.error(`${url}: ${(e as Error).message} again; held at frame ${had} of ${B.frames}`);
			throw e;
		}
	}
}

/** A body file as two ranges, its head at `priority` and its rest at low priority once `later` is had, checked to be of one file. */
async function ranged(
	url: string,
	head: number,
	bytes: (n: number, of: number) => void,
	priority: RequestPriority,
	later: Promise<unknown>,
) {
	const part = async (range: string, priority: RequestPriority) => {
			const r = await request(url, { headers: { Range: `bytes=${range}` }, priority });
			if (!r.ok) throw new Error(`${r.status}`);
			return r;
		},
		size = (r: Response) =>
			Number(r.status === 206 ? r.headers.get("Content-Range")?.split("/")[1] : r.headers.get("Content-Length")),
		first = await part(`0-${head - 1}`, priority),
		of = size(first);
	if (first.status !== 206 || of <= head) return joined([first], bytes, of);
	const rest = later
		.then(() => part(`${head}-`, "low"))
		.then((r) => {
			if (r.status !== 206 || size(r) !== of || r.headers.get("ETag") !== first.headers.get("ETag"))
				throw new Error("written again between its two ranges");
			return r;
		});
	rest.catch(() => {}); // when the head fails to decode first, its rest goes unread
	return joined([first, rest], bytes, of);
}

/** The bodies of `parts` one after the other, each asked for its body once the one before has ended, counted out to `bytes`. */
function joined(parts: (Response | Promise<Response>)[], bytes: (n: number, of: number) => void, of: number) {
	let got = 0,
		k = 0,
		from: ReadableStreamDefaultReader<Uint8Array> | null = null;
	return new ReadableStream<Uint8Array<ArrayBuffer>>({
		async pull(out) {
			for (;;) {
				from ??= (await parts[k]!).body!.getReader();
				const read = await from.read();
				if (!read.done) {
					bytes((got += read.value.byteLength), of);
					return out.enqueue(read.value as Uint8Array<ArrayBuffer>);
				}
				if (++k === parts.length) return out.close();
				from = null;
			}
		},
	});
}

/**
 * A body file's gzip stream decoded into `v` from row 0, `rows` told the rows decoded after each chunk. The inflating and undoing
 * of the differences happen in a worker (unpacker), so the page keeps drawing while scenes load: the compressed bytes go over as
 * they come in, each chunk comes back decoded and is copied into `v`.
 */
async function decode(
	stream: ReadableStream<Uint8Array<ArrayBuffer>>,
	V: number,
	B: BodyLayout,
	v: Int16Array,
	rows: (n: number) => void,
) {
	const job = ++unpacker.jobs,
		reader = stream.getReader();
	let failed: Error | null = null;
	const unpacked = new Promise<void>((ok, fail) =>
		unpacker.waiting.set(job, (d) => {
			if (d.error) fail((failed = new Error(d.error)));
			else if (d.chunk) {
				v.set(d.chunk, d.at * V * 3);
				rows(d.at + d.chunk.length / (V * 3));
			} else ok();
		}),
	);
	unpacked.catch(() => {});
	unpacker.worker.postMessage({ job, V, B: { frames: B.frames, chunk: B.chunk, step: B.step } });
	try {
		for (;;) {
			const read = await reader.read();
			if (failed) throw failed;
			if (read.done) break;
			// a view of part of a larger buffer goes over as its own copy
			const bytes =
				read.value.byteOffset || read.value.byteLength !== read.value.buffer.byteLength ? read.value.slice() : read.value;
			unpacker.worker.postMessage({ job, bytes }, [bytes.buffer]);
		}
		unpacker.worker.postMessage({ job });
		await unpacked;
	} catch (e) {
		unpacker.worker.postMessage({ job, abort: true });
		void reader.cancel().catch(() => {});
		throw e;
	} finally {
		unpacker.waiting.delete(job);
	}
}

/** The body files' gzip streams in, by job: rows of B.chunk frames out, each int16 [n, V, 3] mm, as each is had. */
function unpack() {
	const jobs = new Map<number, WritableStreamDefaultWriter<BufferSource>>();
	const run = async (
		job: number,
		reader: ReadableStreamDefaultReader<Uint8Array>,
		V: number,
		B: Pick<BodyLayout, "frames" | "chunk" | "step">,
	) => {
		const row = V * 3;
		let done = 0,
			buf = new Uint8Array(0),
			over: Uint8Array | null = null; // read past the chunk before, for this one
		while (done < B.frames) {
			const n = Math.min(B.chunk, B.frames - done) * row;
			if (buf.length !== 2 * n) buf = new Uint8Array(2 * n);
			for (let filled = 0; filled < 2 * n; ) {
				let value: Uint8Array | null = over;
				over = null;
				if (!value) {
					const read = await reader.read();
					if (read.done) throw new Error("ended early");
					value = read.value;
				}
				const take = Math.min(value.length, 2 * n - filled);
				buf.set(value.subarray(0, take), filled);
				filled += take;
				if (take < value.length) over = value.subarray(take);
			}
			const v = new Int16Array(n);
			for (let i = 0; i < n; i++) v[i] = buf[i]! | (buf[n + i]! << 8);
			// Int16Array wraps, as the differences were taken
			for (let i = row; i < n; i++) v[i]! += i < 2 * row ? v[i - row]! : 2 * v[i - row]! - v[i - 2 * row]!;
			for (let i = 0; i < n; i++) v[i]! *= B.step;
			postMessage({ job, at: done, chunk: v }, [v.buffer]);
			done += n / row;
		}
		postMessage({ job });
	};
	self.onmessage = ({ data: d }) => {
		if (d.V) {
			const inflate = new DecompressionStream("gzip");
			jobs.set(d.job, inflate.writable.getWriter());
			run(d.job, inflate.readable.getReader(), d.V, d.B)
				.catch((e: Error) => postMessage({ job: d.job, error: e.message }))
				.finally(() => jobs.delete(d.job));
			return;
		}
		const writer = jobs.get(d.job);
		if (!writer) return;
		if (d.bytes) writer.write(d.bytes).catch(() => {});
		else if (d.abort) writer.abort().catch(() => {});
		else writer.close().catch(() => {});
	};
}

const unpacker = {
	worker: new Worker(URL.createObjectURL(new Blob([`(${unpack})()`]))),
	jobs: 0,
	waiting: new Map<number, (d: { error?: string; at: number; chunk?: Int16Array }) => void>(),
};
unpacker.worker.onmessage = ({ data }) => unpacker.waiting.get(data.job)?.(data);

/** A colour texture, decoded off the page's thread (an ImageBitmap): only its upload is left to the page, when it is first drawn. */
export async function texture(url: string): Promise<THREE.Texture> {
	await warming.get(requestKey(url));
	const bitmap = await new THREE.ImageBitmapLoader().setOptions({ premultiplyAlpha: "none" }).loadAsync(url),
		t = new THREE.Texture(bitmap);
	Object.assign(t, {
		colorSpace: THREE.SRGBColorSpace,
		flipY: false,
		anisotropy: 8,
		minFilter: THREE.LinearMipmapLinearFilter,
		needsUpdate: true,
	});
	return t;
}

export const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

/** The page's query parameters, read by the parts of the scene that offer review switches (?guard=cage, ?debug=hits). */
export const param = (name: string) => new URLSearchParams(location.search).get(name);

/** Signed distance (m) from the rink's outline in plan: negative on the ice. */
export function sdRink({ hx, hy, corner }: Rink, x: number, y: number) {
	const qx = Math.abs(x) - (hx - corner),
		qy = Math.abs(y) - (hy - corner);
	return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - corner;
}
