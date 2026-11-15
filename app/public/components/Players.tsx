import { type PlayerStats, team } from "../lib/league";
import { playerStats } from "../lib/site";
import { TeamLogo } from "./TeamLogo";

/** Scorers first, then assisters, then whoever created the most danger. */
const impact = (p: PlayerStats) => p.goals * 3 + p.assists * 2 + p.xg;

export const topPlayers = (n: number): PlayerStats[] => [...playerStats].sort((a, b) => impact(b) - impact(a)).slice(0, n);

const statLine = (p: PlayerStats) => `${p.goals} G · ${p.assists} A · ${p.shots} shots`;

export function PlayerCard({ player, className = "" }: { player: PlayerStats; className?: string }) {
	return (
		<div className={`panel flex items-center gap-4 p-4 ${className}`}>
			<span className="relative shrink-0">
				<span className="grid size-14 place-items-center rounded-full bg-white/8 text-xl font-bold tabular-nums">
					{player.jersey}
				</span>
				<span className="absolute -bottom-1 -right-1 rounded-full ring-2 ring-panel">
					<TeamLogo id={player.team} className="size-6" />
				</span>
			</span>
			<span className="min-w-0">
				<span className="block truncate font-semibold">{player.name}</span>
				<span className="block truncate text-xs text-dim">{team(player.team).name}</span>
				<span className="mt-1 block text-xs tabular-nums">{statLine(player)}</span>
			</span>
		</div>
	);
}
