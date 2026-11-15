import type { ReactNode } from "react";
import { Link } from "react-router";
import { SIDE } from "../lib/league";
import { lead, type Moment, momentsOf, useMedia } from "../lib/media";
import { useReplayIds } from "../lib/replays";
import { useVisibleUntil } from "../lib/stats";
import { eventDetails, periodClock, player } from "../player/events";
import { watchFrom } from "../player/mount";
import { MomentFrame } from "./MomentFrame";
import { TeamLogo } from "./TeamLogo";

/** Moments a viewer may already know about: played as live, nothing past the live edge. */
export function useMoments(): Moment[] {
	const { game } = useMedia();
	const until = useVisibleUntil();
	return momentsOf(game).filter((m) => m.event.t <= until);
}

const KIND_LABEL = { Goal: "Goal", Chance: "Big chance", Penalty: "Penalty" } as const;

const title = (m: Moment) => [KIND_LABEL[m.kind], player(m.event.player).last].filter(Boolean).join(" · ");
const subtitle = (m: Moment) => [periodClock(m.event), ...eventDetails(m.event).slice(0, 2)].join(" · ");

/** A moment with its broadcast frame. Opens the game at that point, or rewinds the stage already on the page when `inPlace`. */
export function MomentCard({
	moment,
	className = "",
	inPlace = false,
}: {
	moment: Moment;
	className?: string;
	inPlace?: boolean;
}) {
	const { event } = moment;
	const replay = useReplayIds().get(moment.id);
	const body = (
		<>
			<MomentFrame moment={moment} scene={replay}>
				<div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-black/20" />
				<span className="absolute left-2.5 top-2.5">
					<TeamLogo id={SIDE[event.team]} className="size-7" />
				</span>
				<span className={`badge absolute right-2.5 top-2.5 ${moment.kind === "Goal" ? "bg-paper text-ink" : "bg-black/60"}`}>
					{KIND_LABEL[moment.kind]}
				</span>
				<span className="absolute bottom-2.5 right-2.5 grid size-8 place-items-center rounded-full bg-black/60 opacity-0 transition-opacity group-hover:opacity-100">
					<svg viewBox="0 0 16 16" className="size-3.5" fill="currentColor" aria-hidden="true">
						<path d="M5 3.5v9l7.5-4.5z" />
					</svg>
				</span>
			</MomentFrame>
			<p className="mt-2.5 truncate text-sm font-semibold">{title(moment)}</p>
			<p className="mt-0.5 truncate text-xs text-dim">{subtitle(moment)}</p>
		</>
	);
	const card = (
		<Card
			moment={moment}
			body={body}
			className={replay ? "group block w-full text-left" : `group block text-left ${className}`}
			inPlace={inPlace}
		/>
	);
	if (!replay) return card;
	// the goal's 3D replay, a link of its own over the card's frame
	return (
		<div className={`relative ${className}`}>
			{card}
			<div className="pointer-events-none absolute inset-x-0 top-0 aspect-video">
				<Link
					to={`/replay/${replay}`}
					className="badge pointer-events-auto absolute bottom-2.5 left-2.5 bg-black/70 transition-colors hover:bg-paper hover:text-ink"
				>
					3D replay
				</Link>
			</div>
		</div>
	);
}

function Card({ moment, body, className, inPlace }: { moment: Moment; body: ReactNode; className: string; inPlace: boolean }) {
	const { event } = moment;
	if (!inPlace) {
		return (
			<Link to={`/final?t=${lead(event)}`} className={className}>
				{body}
			</Link>
		);
	}
	return (
		<button
			type="button"
			className={className}
			onClick={() => {
				watchFrom(lead(event));
				window.scrollTo({ top: 0, behavior: "smooth" });
			}}
		>
			{body}
		</button>
	);
}
