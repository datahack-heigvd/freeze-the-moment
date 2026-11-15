// Each goal scene's stretch of the broadcast's sound, for the 3D replay (replay/sound.ts), into store/site/audio/: <scene>.m4a, cut from
// the match video as it is (AAC, copied), and <scene>@<tempo>.m4a for each slower speed of the replay, slowed with its pitch kept
// (rubberband: smoother than a browser stretches it live). Needs ffmpeg with librubberband.
// `bun scripts/build-audio.ts [scene ...]` builds those scenes' only.
import { mkdir, readdir } from "node:fs/promises";
import { join } from "node:path";
import { $ } from "bun";
import type { SceneData } from "../public/replay/types";

const STORE = join(import.meta.dir, "../../store");
const VIDEO = process.env.VIDEO ?? join(STORE, "data/20251231_115035_SpenglerCup_Kam1 31.12.25 Game 11.mp4");
const SCENES = process.env.SCENES ?? join(STORE, "scenes");
const OUT = join(STORE, "site/audio");
const TEMPOS = ["0.5", "0.25"]; // the replay's slower speeds (replay/viewer.ts TEMPOS)

const GAME = join(STORE, "site/game.json");
const { events }: { events: { t: number; shotT?: number }[] } = await Bun.file(GAME).json();
const length = Number((await $`ffprobe -v error -show_entries format=duration -of csv=p=0 ${VIDEO}`.text()).trim());
const only = process.argv.slice(2);
await mkdir(OUT, { recursive: true });

for (const id of (await readdir(SCENES)).sort()) {
	const file = Bun.file(join(SCENES, id, "scene.json"));
	if ((only.length && !only.includes(id)) || !(await file.exists())) continue;
	const S: SceneData = await file.json(),
		start = S.source.start + S.frame0 / S.fps,
		dur = S.frames / S.fps,
		cut = join(OUT, `${id}.m4a`);
	// a scene of another match has no sound in this video: its moment holds one of the feed's events
	if (start + dur > length || !events.some((e) => (e.shotT ?? e.t) >= start && (e.shotT ?? e.t) <= start + dur)) {
		console.log(`${id}: not in this video, skipped`);
		continue;
	}
	await $`ffmpeg -v error -y -ss ${start} -i ${VIDEO} -t ${dur} -vn -c:a copy -movflags +faststart ${cut}`;
	for (const tempo of TEMPOS) {
		const filter = `rubberband=tempo=${tempo}:transients=smooth:detector=soft:window=long:pitchq=quality:channels=together`;
		await $`ffmpeg -v error -y -i ${cut} -af ${filter} -c:a aac -b:a 160k -movflags +faststart ${join(OUT, `${id}@${tempo}.m4a`)}`;
	}
	console.log(id);
}
