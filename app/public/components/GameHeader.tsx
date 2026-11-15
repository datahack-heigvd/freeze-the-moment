import { formatDate, type LeagueGame, league, team } from "../lib/league";
import { TeamLogo } from "./TeamLogo";

function Side({ id }: { id: LeagueGame["home"] }) {
	return (
		<div className="flex min-w-0 flex-col items-center gap-3 text-center">
			<TeamLogo id={id} className="size-16 sm:size-24" />
			<span className="text-sm font-semibold sm:text-lg">{team(id).name}</span>
		</div>
	);
}

/** Who played, the score so far and when and where it happened. */
export function GameHeader({ game, score }: { game: LeagueGame; score: [number, number] }) {
	return (
		<header className="pb-6 pt-8 text-center sm:pt-12">
			<p className="text-sm text-dim">
				{game.stage} &middot; {formatDate(game.date)} &middot; {league.edition.venue}
			</p>
			<div className="mt-6 grid grid-cols-[1fr_auto_1fr] items-center gap-4 sm:gap-10">
				<Side id={game.home} />
				<p className="text-4xl font-bold tabular-nums sm:text-6xl">
					{score[0]} : {score[1]}
				</p>
				<Side id={game.away} />
			</div>
		</header>
	);
}

/** Goals per period for both teams, with the total. */
export function PeriodTable({
	game,
	periods,
	total,
}: {
	game: LeagueGame;
	periods: [number, number][];
	total: [number, number];
}) {
	const row = (id: LeagueGame["home"], i: 0 | 1) => (
		<tr key={id} className="border-t border-hairline">
			<td className="py-2.5 pl-4">
				<span className="flex items-center gap-2.5">
					<TeamLogo id={id} className="size-6" />
					<span className="font-medium">{team(id).name}</span>
				</span>
			</td>
			{periods.map(([a, b], p) => (
				// biome-ignore lint/suspicious/noArrayIndexKey: a period is its place in the game
				<td key={p} className="text-center tabular-nums">
					{i === 0 ? a : b}
				</td>
			))}
			<td className="pr-4 text-right font-bold tabular-nums">{total[i]}</td>
		</tr>
	);
	return (
		<div className="panel overflow-hidden">
			<table className="w-full text-sm">
				<thead className="text-xs text-dim">
					<tr>
						<th className="py-2.5 pl-4 text-left font-medium">Goals</th>
						{periods.map((_, p) => (
							// biome-ignore lint/suspicious/noArrayIndexKey: a period is its place in the game
							<th key={p} className="w-12 text-center font-medium">
								P{p + 1}
							</th>
						))}
						<th className="w-14 pr-4 text-right font-medium">Total</th>
					</tr>
				</thead>
				<tbody>
					{row(game.home, 0)}
					{row(game.away, 1)}
				</tbody>
			</table>
		</div>
	);
}
