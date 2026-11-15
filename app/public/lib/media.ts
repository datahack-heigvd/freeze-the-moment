import { use } from "react";
import { type Index, loadIndex } from "../player/ar";
import type { Game, GameEvent } from "../player/events";
import { momentIds } from "./site";

let media: Promise<{ game: Game; index: Index }> | undefined;

/** The featured game's events and tracking, and the index of its AR segments. Fetched once. */
const loadMedia = () =>
	(media ??= Promise.all([fetch("/site/game.json").then((r) => r.json() as Promise<Game>), loadIndex()]).then(
		([game, index]) => ({
			game,
			index,
		}),
	));

/** Suspends until the game data has arrived. */
export const useMedia = () => use(loadMedia());

type MomentKind = "Goal" | "Chance" | "Penalty";
export type Moment = { id: number; kind: MomentKind; event: GameEvent; thumb: string };

/** Goals, the biggest chances and the penalties, each with a frame grabbed from the broadcast (see scripts/build-moments.ts). */
export const momentsOf = (game: Game): Moment[] =>
	momentIds.map((id) => {
		const event = game.events[id]!;
		return {
			id,
			kind: event.kind === "goal" ? "Goal" : event.kind === "penalty" ? "Penalty" : "Chance",
			event,
			thumb: `/site/moments/${id}.jpg`,
		};
	});

/** Where in the video a viewer should land to see a moment play out. */
export const lead = (event: GameEvent) => event.t - 8;
