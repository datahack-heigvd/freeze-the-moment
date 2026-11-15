// What a render was made from, as one hash: for the film's takes (build-film.ts, build-ui-film.ts) to tell a take whose inputs are
// unchanged from one to render again. Code is hashed by its content; data (a scene's or the site's, large and written once) by
// each file's path, size and time.
import { createHash } from "node:crypto";
import { stat } from "node:fs/promises";
import { join } from "node:path";

/** A string as is; a file, or every file under a directory, by its content (`code`) or by its path, size and time (`data`). */
export type Input = string | { code: string } | { data: string };

const ALL = new Bun.Glob("**/*");

/** The files under `path` (relative to it, in order), or the file itself (""); none if it is missing. */
async function files(path: string) {
	const s = await stat(path).catch(() => null);
	if (!s) return [];
	if (s.isFile()) return [""];
	return (await Array.fromAsync(ALL.scan({ cwd: path, onlyFiles: true }))).sort();
}

export async function fingerprint(inputs: Input[]): Promise<string> {
	const hash = createHash("sha1");
	for (const input of inputs) {
		if (typeof input === "string") hash.update(`s:${input}\n`);
		else if ("code" in input)
			for (const f of await files(input.code)) hash.update(`c:${f}\n`).update(await Bun.file(join(input.code, f)).bytes());
		else
			for (const f of await files(input.data)) {
				const s = await stat(join(input.data, f));
				hash.update(`d:${f}:${s.size}:${s.mtimeMs}\n`);
			}
	}
	return hash.digest("hex");
}

/** Whether `out` is there, made from `key` (its `<out>.key` beside it). */
export async function made(out: string, key: string) {
	const k = Bun.file(`${out}.key`);
	return (await Bun.file(out).exists()) && (await k.exists()) && (await k.text()) === key;
}

/** Keep that `out` was made from `key`. */
export const keep = (out: string, key: string) => Bun.write(`${out}.key`, key);

/** fn over items, at most n at a time, in order of start. */
export async function pool<T>(items: T[], n: number, fn: (item: T) => Promise<void>) {
	let next = 0;
	await Promise.all(
		Array.from({ length: Math.min(n, items.length) }, async () => {
			while (next < items.length) await fn(items[next++]!);
		}),
	);
}
