import { Link } from "react-router";
import { ShotMap, StatBar, XgRace } from "../components/Charts";
import { GameHeader, PeriodTable } from "../components/GameHeader";
import { PlayerCard, topPlayers } from "../components/Players";
import { Container, Empty, Section } from "../components/Section";
import { TeamLogo } from "../components/TeamLogo";
import { featuredGame, minutes, type PlayerStats, SIDE, team } from "../lib/league";
import { lead, useMedia } from "../lib/media";
import { playerStats } from "../lib/site";
import { eventsUntil, periodScores, SIDES, scoreOf, sideStats, useVisibleUntil } from "../lib/stats";
import { eventDetails, periodClock, player, TEAM_COLOR } from "../player/events";

/** The final's events, trimmed to what the viewer is allowed to know yet. */
function useKnown() {
	const { game } = useMedia();
	const until = useVisibleUntil();
	return { game, until, events: eventsUntil(game, until), done: until === Number.POSITIVE_INFINITY };
}

export function Recap() {
	const { events, done } = useKnown();
	const goals = events.filter((e) => e.kind === "goal");
	const periods = Math.max(1, ...events.map((e) => e.period));
	return (
		<Container>
			<GameHeader game={featuredGame} score={scoreOf(events)} />
			<PeriodTable game={featuredGame} periods={periodScores(events, periods)} total={scoreOf(events)} />

			<Section title="Goals">
				{goals.length === 0 ? (
					<Empty>No goals yet.</Empty>
				) : (
					<ul className="panel divide-y divide-hairline">
						{goals.map((g) => (
							<li key={g.t}>
								<Link
									to={`/final?t=${lead(g)}`}
									className="flex items-center gap-4 px-4 py-3.5 transition-colors hover:bg-panel-hover sm:px-5"
								>
									<span className="w-16 shrink-0 text-xs tabular-nums text-dim">{periodClock(g)}</span>
									<TeamLogo id={SIDE[g.team]} className="size-7" />
									<span className="min-w-0 flex-1">
										<span className="block truncate font-semibold">{player(g.player).name}</span>
										<span className="block truncate text-xs text-dim">{eventDetails(g).join(" · ")}</span>
									</span>
									<span className="badge hidden sm:inline-flex">Watch</span>
								</Link>
							</li>
						))}
					</ul>
				)}
			</Section>

			<Section title="Expected goals" hint="How much danger each team created, shot by shot. Dots are goals.">
				<div className="panel p-4 sm:p-6">
					<XgRace events={events} />
					<div className="mt-3 flex items-center justify-center gap-6 text-xs text-dim">
						{SIDES.map((side) => (
							<span key={side} className="flex items-center gap-2">
								<i className="h-0.5 w-5 rounded-full" style={{ background: TEAM_COLOR[side] }} />
								{team(SIDE[side]).name} {sideStats(events, side).xg.toFixed(2)}
							</span>
						))}
					</div>
				</div>
			</Section>

			<Section
				title="Shot maps"
				hint="Every attempt, turned to face the goal the team was attacking. Bigger means better chance."
			>
				<div className="grid gap-4 sm:grid-cols-2">
					{SIDES.map((side) => (
						<ShotMap key={side} events={events} side={side} />
					))}
				</div>
			</Section>

			{done && (
				<Section title="Three stars">
					<div className="grid gap-3 sm:grid-cols-3">
						{topPlayers(3).map((p, i) => (
							<div key={p.id} className="relative">
								<span className="badge absolute -top-2 left-4 z-10 bg-paper text-ink">{["1st", "2nd", "3rd"][i]} star</span>
								<PlayerCard player={p} className="pt-6" />
							</div>
						))}
					</div>
				</Section>
			)}
		</Container>
	);
}

const sum = (list: PlayerStats[], pick: (p: PlayerStats) => number) => list.reduce((total, p) => total + pick(p), 0);

export function Stats() {
	const { events, done } = useKnown();
	const [home, away] = SIDES.map((side) => sideStats(events, side)) as [
		ReturnType<typeof sideStats>,
		ReturnType<typeof sideStats>,
	];
	const skaters = SIDES.map((side) => playerStats.filter((p) => p.team === SIDE[side]));
	const [homeSkaters, awaySkaters] = skaters as [PlayerStats[], PlayerStats[]];
	return (
		<Container>
			<GameHeader game={featuredGame} score={scoreOf(events)} />
			<div className="flex items-center justify-between pb-1 pt-2 text-sm font-semibold">
				<span className="flex items-center gap-2">
					<TeamLogo id={SIDE.Home} className="size-6" />
					{team(SIDE.Home).short}
				</span>
				<span className="flex items-center gap-2">
					{team(SIDE.Away).short}
					<TeamLogo id={SIDE.Away} className="size-6" />
				</span>
			</div>
			<div className="panel divide-y divide-hairline px-4 py-1 sm:px-6">
				<StatBar label="Goals" home={home.goals} away={away.goals} />
				<StatBar label="Shots on goal" home={home.shotsOnGoal} away={away.shotsOnGoal} />
				<StatBar label="Shot attempts" home={home.attempts} away={away.attempts} />
				<StatBar label="Expected goals" home={home.xg} away={away.xg} digits={2} />
				<StatBar label="Penalties" home={home.penalties} away={away.penalties} />
				{done && (
					<>
						<StatBar
							label="Distance skated (km)"
							home={sum(homeSkaters, (p) => p.distance) / 1000}
							away={sum(awaySkaters, (p) => p.distance) / 1000}
							digits={1}
						/>
						<StatBar
							label="Faceoffs won"
							home={sum(homeSkaters, (p) => p.faceoffsWon)}
							away={sum(awaySkaters, (p) => p.faceoffsWon)}
						/>
						<StatBar label="Passes completed" home={sum(homeSkaters, (p) => p.passes)} away={sum(awaySkaters, (p) => p.passes)} />
					</>
				)}
			</div>

			{done ? (
				SIDES.map((side, i) => <SkaterTable key={side} side={side} skaters={skaters[i]!} />)
			) : (
				<p className="mt-6 text-sm text-dim">Skater tracking stats arrive after the final whistle.</p>
			)}
		</Container>
	);
}

function SkaterTable({ side, skaters }: { side: keyof typeof SIDE; skaters: PlayerStats[] }) {
	const id = SIDE[side];
	const rows = [...skaters].sort((a, b) => b.toi - a.toi);
	return (
		<Section title={team(id).name}>
			<div className="panel overflow-x-auto">
				<table className="w-full min-w-[34rem] text-sm">
					<thead className="text-xs text-dim">
						<tr>
							<th className="w-10 py-2.5 pl-4 text-left font-medium">#</th>
							<th className="text-left font-medium">Player</th>
							{["G", "A", "S", "xG", "TOI", "km", "Top km/h"].map((h) => (
								<th key={h} className="px-2 text-right font-medium last:pr-4">
									{h}
								</th>
							))}
						</tr>
					</thead>
					<tbody>
						{rows.map((p) => (
							<tr key={p.id} className="border-t border-hairline tabular-nums">
								<td className="py-2 pl-4 text-dim">{p.jersey}</td>
								<td>
									<span className="font-medium">{p.name}</span>
								</td>
								<td className="px-2 text-right">{p.goals}</td>
								<td className="px-2 text-right">{p.assists}</td>
								<td className="px-2 text-right">{p.shots}</td>
								<td className="px-2 text-right">{p.xg.toFixed(2)}</td>
								<td className="px-2 text-right">{minutes(p.toi)}</td>
								<td className="px-2 text-right">{(p.distance / 1000).toFixed(1)}</td>
								<td className="px-2 pr-4 text-right">{p.topSpeed.toFixed(1)}</td>
							</tr>
						))}
					</tbody>
				</table>
			</div>
		</Section>
	);
}
