// The local server the headless build scripts' pages read from.
import { join } from "node:path";

/** The files under `dir`, at their paths after `prefix`. */
export const serveFrom = (dir: string, prefix: string) => async (req: Request) => {
	const file = Bun.file(join(dir, decodeURIComponent(new URL(req.url).pathname.slice(prefix.length))));
	return file.name!.startsWith(`${dir}/`) && (await file.exists())
		? new Response(file)
		: new Response("Not found", { status: 404 });
};

/** The viewer's own assets (the arena's, the players' cloth). */
export const ASSETS = join(import.meta.dir, "../public/assets");
