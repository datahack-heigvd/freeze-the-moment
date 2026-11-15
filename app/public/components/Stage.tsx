import { memo, useEffect, useLayoutEffect, useRef, useState } from "react";
import { SIDE, team } from "../lib/league";
import { useMedia } from "../lib/media";
import { COLOR, TEAM_COLOR } from "../player/events";
import { mountPlayer, type Scoreboard } from "../player/mount";
import { TeamLogo } from "./TeamLogo";

type Props = {
	live: boolean;
	/** Video second to open at. Replays default to puck drop. */
	startAt?: number;
	className?: string;
};

/** The broadcast with its overlays, timeline and score. Live follows the edge; otherwise the whole game can be replayed. */
export function Stage({ live, startAt, className = "" }: Props) {
	const { game, index } = useMedia();
	const root = useRef<HTMLDivElement>(null);
	const [score, setScore] = useState<Scoreboard>({ home: 0, away: 0, period: 0 });
	const opening = startAt ?? (live ? undefined : game.periods[0]!.t - 5);

	// a layout effect, so the player is torn down before React detaches its elements
	useLayoutEffect(
		() => mountPlayer(root.current!, { game, index, live, startAt: opening, onScore: setScore }),
		[game, index, live, opening],
	);

	return (
		<div data-stage-frame className={`flex min-h-0 flex-col ${className}`}>
			<ScoreHeader score={score} />
			<div ref={root} className="flex min-h-0 flex-1 flex-col">
				<StageScreen />
				<footer className="flex shrink-0 flex-wrap items-center justify-between gap-x-6 gap-y-1.5 px-4 py-2.5 text-xs text-dim sm:px-8">
					<span>Camera 1 &middot; Eisstadion Davos</span>
					<ul className="flex flex-wrap items-center gap-x-4 gap-y-1" id="legend" />
				</footer>
			</div>
		</div>
	);
}

function ScoreHeader({ score }: { score: Scoreboard }) {
	const sides = [
		{ team: team(SIDE.Home), n: score.home, color: TEAM_COLOR.Home },
		{ team: team(SIDE.Away), n: score.away, color: TEAM_COLOR.Away },
	] as const;
	return (
		<header className="grid shrink-0 grid-cols-[1fr_auto_1fr] items-center gap-3 px-4 py-3 sm:gap-6 sm:px-8">
			<div className="flex min-w-0 items-center gap-3">
				<TeamLogo id={sides[0].team.id} className="size-9 sm:size-10" />
				<p className="min-w-0 truncate text-base font-semibold sm:text-lg">
					<span className="sm:hidden">{sides[0].team.short}</span>
					<span className="hidden sm:inline">{sides[0].team.name}</span>
				</p>
			</div>
			<div className="text-center">
				<p className="flex items-center justify-center text-3xl font-bold tabular-nums sm:text-4xl">
					<Digit value={sides[0].n} color={sides[0].color} />
					<span className="px-2.5 text-dim sm:px-3">:</span>
					<Digit value={sides[1].n} color={sides[1].color} />
				</p>
				<p className="mt-0.5 text-xs text-dim">{score.period ? `Period ${score.period}` : "Pre-game"}</p>
			</div>
			<div className="flex min-w-0 items-center justify-end gap-3 text-right">
				<p className="min-w-0 truncate text-base font-semibold sm:text-lg">
					<span className="sm:hidden">{sides[1].team.short}</span>
					<span className="hidden sm:inline">{sides[1].team.name}</span>
				</p>
				<TeamLogo id={sides[1].team.id} className="size-9 sm:size-10" />
			</div>
		</header>
	);
}

/** A score digit that pops in its team colour when the team scores. */
function Digit({ value, color }: { value: number; color: string }) {
	const el = useRef<HTMLSpanElement>(null);
	const previous = useRef(value);
	useEffect(() => {
		if (value === previous.current + 1 && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
			el.current?.animate(
				[
					{ transform: "scale(1.4)", color },
					{ transform: "scale(1)", color: COLOR.paper },
				],
				{ duration: 900, easing: "cubic-bezier(.2,.8,.2,1)" },
			);
		}
		previous.current = value;
	}, [value, color]);
	return (
		<span ref={el} className="inline-block">
			{value}
		</span>
	);
}

/** Static markup that mountPlayer fills in; React never re-renders it. */
const StageScreen = memo(function StageScreen() {
	return (
		<div id="stage" className="relative min-h-0 w-full flex-1 overflow-hidden bg-black">
			<video id="v" autoPlay muted playsInline preload="auto" className="block h-full w-full cursor-pointer object-cover" />
			<canvas id="ar" className="pointer-events-none absolute inset-0 h-full w-full" />
			<div id="ar-labels" className="pointer-events-none absolute inset-0 overflow-hidden" />

			<div data-ui className="pointer-events-none absolute inset-x-0 top-0 h-28 bg-gradient-to-b from-black/40 to-transparent" />
			<div
				data-ui
				className="pointer-events-none absolute inset-x-0 bottom-0 h-52 bg-gradient-to-t from-black/70 via-black/30 to-transparent"
			/>

			<div className="pointer-events-none absolute inset-x-4 top-3 flex flex-wrap items-start justify-between gap-2 sm:inset-x-8 sm:top-5">
				<button type="button" id="status" className="pill pointer-events-auto" />
				<div data-ui className="pointer-events-auto flex flex-wrap items-center justify-end gap-2">
					<button type="button" id="ar-jump" className="pill" hidden>
						Watch with AR
						<svg
							aria-hidden="true"
							viewBox="0 0 16 16"
							fill="none"
							stroke="currentColor"
							strokeWidth="1.6"
							strokeLinecap="round"
							strokeLinejoin="round"
						>
							<path d="M3 8h10M9 4l4 4-4 4" />
						</svg>
					</button>
					<div id="presets" className="seg" role="group" aria-label="Overlay" />
				</div>
			</div>

			<div id="banner" aria-live="polite" />

			<div data-ui className="absolute inset-x-4 bottom-3 flex items-center gap-3 sm:inset-x-8 sm:bottom-5">
				<button type="button" id="play" className="pill icon" aria-label="Pause" />
				<div className="relative flex-1">
					<svg id="momentum" className="pointer-events-none absolute inset-x-0 bottom-9 block h-11 w-full overflow-visible" />
					<div id="markers" className="absolute inset-x-0 bottom-2.5 z-10 h-5" />
					<div className="pointer-events-none absolute inset-x-0 top-[calc(50%-1.5px)] z-20 h-[3px] rounded-full bg-white/25">
						<div id="fill" className="h-full rounded-full bg-paper" />
					</div>
					<i
						id="dot"
						className="pointer-events-none absolute bottom-[4px] z-20 size-3 -translate-x-1/2 rounded-full bg-paper shadow-[0_0_0_3px_rgb(0_0_0/0.35)]"
					/>
					<input
						id="scrub"
						type="range"
						min="0"
						max="1"
						step="0.1"
						defaultValue="0"
						aria-label="Scrub"
						className="block h-5 w-full cursor-pointer"
					/>
				</div>
				<button type="button" id="mute" className="pill icon" aria-label="Unmute" />
				<button type="button" id="fullscreen" className="pill icon" aria-label="Full screen" />
			</div>
			<div id="tip" className="pointer-events-none absolute z-30 hidden" />
		</div>
	);
});
