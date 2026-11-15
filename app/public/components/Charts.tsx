import { SIDE, team } from "../lib/league";
import { gameSecond, SIDES } from "../lib/stats";
import { type GameEvent, TEAM_COLOR } from "../player/events";

/** One row of a head-to-head table: values at the edges, bars growing from the middle. */
export function StatBar({ label, home, away, digits = 0 }: { label: string; home: number; away: number; digits?: number }) {
	const total = home + away || 1;
	return (
		<div className="py-2.5">
			<div className="flex items-baseline justify-between text-sm">
				<span className="font-semibold tabular-nums">{home.toFixed(digits)}</span>
				<span className="text-xs text-dim">{label}</span>
				<span className="font-semibold tabular-nums">{away.toFixed(digits)}</span>
			</div>
			<div className="mt-1.5 flex gap-1" aria-hidden>
				<div className="flex h-1.5 flex-1 justify-end overflow-hidden rounded-full bg-white/6">
					<i className="h-full rounded-full" style={{ width: `${(home / total) * 100}%`, background: TEAM_COLOR.Home }} />
				</div>
				<div className="flex h-1.5 flex-1 overflow-hidden rounded-full bg-white/6">
					<i className="h-full rounded-full" style={{ width: `${(away / total) * 100}%`, background: TEAM_COLOR.Away }} />
				</div>
			</div>
		</div>
	);
}

const W = 600;
const H = 150;
const PAD = { l: 8, r: 8, t: 10, b: 22 };

/** Cumulative expected goals over regulation time, with a dot where each goal went in. */
export function XgRace({ events }: { events: GameEvent[] }) {
	const end = Math.max(3600, ...events.map(gameSecond));
	const series = SIDES.map((side) => {
		let total = 0;
		const points = events
			.filter((e) => e.team === side && e.kind !== "penalty" && e.xg != null)
			.sort((a, b) => gameSecond(a) - gameSecond(b))
			.map((e) => ({ e, x: gameSecond(e), y: (total += e.xg!) }));
		return { side, points, total };
	});
	const top = Math.max(0.5, ...series.map((s) => s.total));
	const x = (s: number) => PAD.l + (s / end) * (W - PAD.l - PAD.r);
	const y = (v: number) => H - PAD.b - (v / top) * (H - PAD.b - PAD.t);
	const step = (points: { x: number; y: number }[]) => {
		let d = `M${x(0)},${y(0)}`;
		for (const p of points) d += `H${x(p.x)}V${y(p.y)}`;
		return `${d}H${x(end)}`;
	};
	return (
		<svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Cumulative expected goals for both teams">
			{[1200, 2400].map((s) => (
				<line key={s} x1={x(s)} x2={x(s)} y1={PAD.t} y2={H - PAD.b} stroke="var(--hairline)" strokeDasharray="3 4" />
			))}
			{["P1", "P2", "P3"].map((label, i) => (
				<text key={label} x={x(i * 1200 + 600)} y={H - 6} textAnchor="middle" fontSize="11" fill="var(--dim)">
					{label}
				</text>
			))}
			<line x1={PAD.l} x2={W - PAD.r} y1={y(0)} y2={y(0)} stroke="var(--hairline)" />
			{series.map(({ side, points }) => (
				<g key={side}>
					<path d={step(points)} fill="none" stroke={TEAM_COLOR[side]} strokeWidth="2" strokeLinejoin="round" />
					{points
						.filter((p) => p.e.kind === "goal")
						.map((p) => (
							<circle key={p.e.t} cx={x(p.x)} cy={y(p.y)} r="4.5" fill={TEAM_COLOR[side]} stroke="#fff" strokeWidth="1.5" />
						))}
				</g>
			))}
		</svg>
	);
}

const RINK_R = 8.5;
const rinkPath = `M0,-15H${30 - RINK_R}A${RINK_R},${RINK_R} 0 0 1 30,${-15 + RINK_R}V${15 - RINK_R}A${RINK_R},${RINK_R} 0 0 1 ${30 - RINK_R},15H0Z`;

/** Where one team shot from, drawn on the half rink it attacks. Size is expected goals. */
export function ShotMap({ events, side }: { events: GameEvent[]; side: GameEvent["team"] }) {
	const shots = events.filter((e) => e.team === side && e.xy && e.kind !== "penalty");
	const color = TEAM_COLOR[side];
	const name = team(SIDE[side]).name;
	return (
		<figure>
			<svg viewBox="-1 -16 32 32" className="w-full rounded-xl bg-[#14181e]" role="img" aria-label={`${name} shots`}>
				<path d={rinkPath} fill="#1a2028" />
				<g fill="none" strokeWidth="0.3">
					<line x1="7.14" x2="7.14" y1="-15" y2="15" stroke="#6f8299" strokeWidth="0.6" />
					<line x1="26" x2="26" y1="-12" y2="12" stroke="#c8524d" />
					<line x1="0" x2="0" y1="-15" y2="15" stroke="#c8524d" strokeWidth="0.6" />
					{[-7, 7].map((fy) => (
						<circle key={fy} cx="20" cy={fy} r="4.5" stroke="#c8524d" />
					))}
					<path d="M26,-1.8A1.8,1.8 0 0 0 26,1.8Z" fill="#6f8299" fillOpacity="0.3" stroke="#c8524d" strokeWidth="0.2" />
					<rect x="26" y="-0.9" width="1.2" height="1.8" stroke="#9fb3c8" strokeWidth="0.2" />
				</g>
				{shots.map((e) => {
					// each team attacks one end per period: turn every shot to the right-hand goal
					const [rx, ry] = e.xy![0] < 0 ? [-e.xy![0], -e.xy![1]] : e.xy!;
					const r = 0.5 + Math.sqrt(e.xg ?? 0.02) * 2.6;
					return (
						<circle
							key={e.t}
							cx={rx}
							cy={-ry}
							r={r}
							fill={e.result === "Missed" ? "none" : color}
							fillOpacity={e.result === "Blocked" ? 0.35 : 0.85}
							stroke={e.kind === "goal" ? "#fff" : color}
							strokeWidth={e.kind === "goal" ? 0.45 : 0.25}
						/>
					);
				})}
			</svg>
			<figcaption className="mt-2 flex items-center gap-2 text-xs text-dim">
				<i className="size-2 rounded-full" style={{ background: color }} />
				{name}, {shots.length} attempts
			</figcaption>
		</figure>
	);
}
