import { join } from "node:path";
import { $ } from "bun";

const VIDEO =
	process.env.VIDEO ?? join(import.meta.dir, "../../store/data/20251231_115035_SpenglerCup_Kam1 31.12.25 Game 11.mp4");
const GAME = join(import.meta.dir, "../../store/site/game.json");
const OUT = join(import.meta.dir, "../../store/site/moments.json");
const THUMBS = join(import.meta.dir, "../../store/site/moments");

type Event = { t: number; shotT?: number; kind: "goal" | "penalty" | "shot"; period: number; clock: number; xg: number | null };
const { events }: { events: Event[] } = await Bun.file(GAME).json();

const CHANCES = 6;
const picked = new Set<number>();
for (const [i, e] of events.entries()) if (e.kind === "goal") picked.add(i);
events
	.map((e, i) => [e, i] as const)
	.filter(([e]) => e.kind === "shot" && e.xg != null)
	.sort((a, b) => b[0].xg! - a[0].xg!)
	.slice(0, CHANCES)
	.forEach(([, i]) => {
		picked.add(i);
	});
// a double minor is one moment, not two
const seen = new Set<string>();
events.forEach((e, i) => {
	const key = `${e.period}:${e.clock}`;
	if (e.kind !== "penalty" || seen.has(key)) return;
	seen.add(key);
	picked.add(i);
});

await $`mkdir -p ${THUMBS}`;
const moments = [...picked].sort((a, b) => events[a]!.t - events[b]!.t);
for (const i of moments) {
	const e = events[i]!;
	// the video second of the shot itself; for a goal the event is stamped a little later
	const at = e.kind === "penalty" ? e.t : (e.shotT ?? e.t);
	await $`ffmpeg -loglevel error -y -ss ${at} -i ${VIDEO} -frames:v 1 -vf scale=640:-2 -q:v 4 ${join(THUMBS, `${i}.jpg`)}`;
}

await Bun.write(OUT, `${JSON.stringify(moments)}\n`);
console.log(`${moments.length} moments`);
