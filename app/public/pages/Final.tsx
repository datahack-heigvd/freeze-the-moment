import { NavLink, Outlet, useSearchParams } from "react-router";
import { MomentCard, useMoments } from "../components/Moments";
import { Container, Section } from "../components/Section";
import { Stage } from "../components/Stage";

const TABS = [
	["/final", "Watch"],
	["/final/recap", "Recap"],
	["/final/stats", "Stats"],
] as const;

/** The final: its broadcast with the AR overlay, its recap and its stats. */
export function FinalPage() {
	return (
		<>
			<div className="border-b border-hairline">
				<Container>
					<nav className="tabs" aria-label="The final">
						{TABS.map(([to, label]) => (
							<NavLink key={to} to={to} end>
								{label}
							</NavLink>
						))}
					</nav>
				</Container>
			</div>
			<Outlet />
		</>
	);
}

/** Header, tabs and the stage's own bars take what they need; the broadcast gets the rest of the screen. */
const STAGE = "h-[calc(100svh-var(--header-h)-3rem)] min-h-[34rem]";

/**
 * The broadcast from puck drop, or from ?t= (a video second). ?live plays it as if it were on now: from the server's clock, the
 * goals and moments appearing only once they have happened.
 */
export function Watch() {
	const [params] = useSearchParams();
	const t = Number(params.get("t"));
	return (
		<>
			<Stage live={params.has("live")} startAt={params.has("t") && Number.isFinite(t) ? t : undefined} className={STAGE} />
			<Container>
				<JumpTo />
			</Container>
		</>
	);
}

/** Skip straight to a goal or a big chance. */
function JumpTo() {
	const moments = useMoments().filter((m) => m.kind !== "Penalty");
	return (
		<Section title="Jump to a moment" hint="Goals and the biggest chances. Pick one to rewind the broadcast to it.">
			<div className="rail -mx-4 px-4 sm:-mx-8 sm:px-8">
				{moments.map((m) => (
					<MomentCard key={m.id} moment={m} className="w-64" inPlace />
				))}
			</div>
		</Section>
	);
}
