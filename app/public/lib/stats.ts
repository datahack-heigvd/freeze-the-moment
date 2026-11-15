import { useSearchParams } from "react-router";
import type { Game, GameEvent } from "../player/events";
import { useLiveEdge } from "./clock";

type Side = GameEvent["team"];
export const SIDES: readonly Side[] = ["Home", "Away"];

/**
 * How much of the final a viewer is allowed to know about, as a video second: all of it, or, played as live (?live), what has
 * happened up to the live edge.
 */
export function useVisibleUntil(): number {
	const live = useSearchParams()[0].has("live");
	const edge = useLiveEdge(live);
	return live ? edge : Number.POSITIVE_INFINITY;
}

export const eventsUntil = (game: Game, t: number) => game.events.filter((e) => e.t <= t);

/** Seconds into regulation time. */
export const gameSecond = (e: GameEvent) => (e.period - 1) * 1200 + e.clock;

export type SideStats = { goals: number; shotsOnGoal: number; attempts: number; xg: number; penalties: number };

export function sideStats(events: GameEvent[], side: Side): SideStats {
	const mine = events.filter((e) => e.team === side);
	const goals = mine.filter((e) => e.kind === "goal");
	const shots = mine.filter((e) => e.kind === "shot");
	return {
		goals: goals.length,
		shotsOnGoal: goals.length + shots.filter((e) => e.result === "Saved").length,
		attempts: goals.length + shots.length,
		xg: [...goals, ...shots].reduce((sum, e) => sum + (e.xg ?? 0), 0),
		penalties: mine.filter((e) => e.kind === "penalty").length,
	};
}

export const scoreOf = (events: GameEvent[]): [number, number] => [
	sideStats(events, "Home").goals,
	sideStats(events, "Away").goals,
];

/** Goals per period, for the scoreboard table. */
export function periodScores(events: GameEvent[], periods: number): [number, number][] {
	return Array.from({ length: periods }, (_, p) => scoreOf(events.filter((e) => e.period === p + 1)));
}
