import { use } from "react";
import { useMoments } from "../components/Moments";
import type { SceneInfo } from "../replay/types";
import type { Moment } from "./media";

let scenes: Promise<SceneInfo[]> | undefined;

/** The goals rebuilt in 3D (store/scenes, listed by server.ts). Fetched once. */
const loadScenes = () => (scenes ??= fetch("/api/scenes").then((r) => (r.ok ? (r.json() as Promise<SceneInfo[]>) : [])));

/** A 3D replay and the final's goal it shows. */
export type Replay = { scene: SceneInfo; moment: Moment };

/** The moment whose shot falls inside the scene's frames (the scenes are cut from the final's broadcast around each goal). */
const momentOf = (scene: SceneInfo, moments: Moment[]) =>
	moments.find(
		(m) => m.kind === "Goal" && (m.event.shotT ?? m.event.t) >= scene.from - 1 && (m.event.shotT ?? m.event.t) <= scene.to + 1,
	);

/** The final's goals rebuilt in 3D, in game order, up to the live edge: a goal still to come is never given away. */
export function useReplays(): Replay[] {
	const scenes = use(loadScenes());
	const moments = useMoments();
	return scenes
		.flatMap((scene) => {
			const moment = momentOf(scene, moments);
			return moment ? [{ scene, moment }] : [];
		})
		.sort((a, b) => a.moment.event.t - b.moment.event.t);
}

/** Each goal moment's 3D replay, by moment id. */
export function useReplayIds(): Map<number, string> {
	return new Map(useReplays().map((r) => [r.moment.id, r.scene.id] as const));
}
