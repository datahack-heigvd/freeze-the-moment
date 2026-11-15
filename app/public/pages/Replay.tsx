import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router";
import { MomentFrame } from "../components/MomentFrame";
import { Container, Empty, Section } from "../components/Section";
import { TeamLogo } from "../components/TeamLogo";
import { minutes, SIDE } from "../lib/league";
import { lead, useMedia } from "../lib/media";
import { sceneLoading } from "../lib/priority";
import { type Replay, useReplays } from "../lib/replays";
import { playerStats } from "../lib/site";
import { eventsUntil, scoreOf } from "../lib/stats";
import { periodClock, player, TEAM_COLOR } from "../player/events";
import type { PerfReport } from "../replay/perf";
import { LOOKS, type Look, parseLook } from "../replay/post";
import type { PersonMeta, Role } from "../replay/types";
import { MODES, type Mode, ReplayViewer } from "../replay/viewer";

const RATES = [
	[0.25, "¼×"],
	[0.5, "½×"],
	[1, "1×"],
] as const;

const MODE_LABEL: Record<Mode, string> = { replay: "Replay", player: "Player", tactical: "Tactical", explore: "Explore" };
const ROLE_LABEL: Record<Role, string> = { scorer: "Scorer", passer: "Passer", goalie: "Goalie" };
const ANGLE_LABEL: Record<string, string> = {
	drone: "Drone",
	bullettime: "Bullet time",
	puckcam: "Puck cam",
	goalcam: "Net cam",
	celebrate: "Celebration",
	goal: "Goal",
	replay: "Broadcast replay",
	goal_slow: "Slow motion",
};
const CLICK = { px: 6, ms: 400 }; // moved less, released sooner: a click on the stage, not a drag

const replayTitle = ({ moment }: Replay) => `Goal · ${player(moment.event.player).last}`;
const replaySubtitle = ({ moment }: Replay) => periodClock(moment.event);

/** A person's numbers over the final (scene team 0 the home side, 1 the away), for the card shown over them on hover. */
const finalStats = (m: PersonMeta): [string, string][] => {
	const s = playerStats.find((p) => p.team === SIDE[m.team === 1 ? "Away" : "Home"] && p.jersey === m.jersey);
	return s ? [[minutes(s.toi), "ice time"]] : [];
};

/** The goals strip above the stage takes this; the replay gets the rest of the screen. */
const STAGE = "h-[calc(100svh-var(--header-h)-3.25rem)] min-h-[30rem]";

/** The final's goals rebuilt in 3D from the broadcast: replayed, through a player's eyes, from above, or explored freely. */
export function ReplayPage() {
	const replays = useReplays();
	const { id } = useParams();
	const current = replays.find((r) => r.scene.id === id) ?? replays[0];
	const goals = useMedia().game.events.filter((e) => e.kind === "goal").length;

	return (
		<>
			{replays.length > 0 && <GoalStrip replays={replays} current={current} />}
			{current ? (
				<Stage key={current.scene.id} replay={current} />
			) : (
				<Container className="mt-6">
					<Empty>No 3D replays yet.</Empty>
				</Container>
			)}
			<Container>
				<div className="pb-2 pt-8 sm:pt-10">
					<h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Freeze the moment</h1>
					<p className="mt-2 max-w-2xl text-pretty text-dim">
						The goals of the Spengler Cup 2025 final, rebuilt in 3D from the TV broadcast alone: every skater, stick and the puck.
						Watch the replay, see it through a player's eyes, read the play from above, or stop time and walk around it.
					</p>
				</div>
				{replays.length > 1 && (
					<Section
						title="The goals of the final"
						hint={`${replays.length} of the ${goals} goals, each rebuilt from the broadcast around the shot.`}
					>
						<div className="grid gap-x-4 gap-y-7 sm:grid-cols-2 lg:grid-cols-4">
							{replays.map((r) => (
								<ReplayCard key={r.scene.id} replay={r} current={r === current} />
							))}
						</div>
					</Section>
				)}
			</Container>
		</>
	);
}

/** The goals rebuilt in 3D in game order, each with the final's score once it went in, the one shown highlighted; a pick opens
 * its replay. */
function GoalStrip({ replays, current }: { replays: Replay[]; current: Replay | undefined }) {
	const { game } = useMedia();
	const items = replays.map((r) => ({ replay: r, score: scoreOf(eventsUntil(game, r.moment.event.t)).join("–") }));
	return (
		<nav aria-label="Goals" className="border-b border-hairline">
			<div className="flex h-13 items-center gap-1.5 overflow-x-auto px-4 [scrollbar-width:none] sm:px-8">
				{items.map(({ replay: r, score }) => (
					<Link
						key={r.scene.id}
						to={`/replay/${r.scene.id}`}
						aria-current={r === current ? "page" : undefined}
						className="flex shrink-0 items-center gap-2 rounded-full px-3 py-1.5 text-sm text-dim transition-colors hover:bg-white/8 hover:text-paper aria-[current=page]:bg-paper aria-[current=page]:text-ink"
					>
						<i className="size-2 rounded-full" style={{ background: TEAM_COLOR[r.moment.event.team] }} />
						<span className="font-medium">{player(r.moment.event.player).last}</span>
						<span className="tabular-nums opacity-70">{score}</span>
					</Link>
				))}
			</div>
		</nav>
	);
}

function ReplayCard({ replay, current }: { replay: Replay; current: boolean }) {
	const { scene, moment } = replay;
	return (
		<Link to={`/replay/${scene.id}`} className="group block text-left" aria-current={current ? "page" : undefined}>
			<MomentFrame moment={moment} scene={scene.id} className={current ? "outline-2 -outline-offset-2 outline-away" : ""}>
				<div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-black/20" />
				<span className="absolute left-2.5 top-2.5">
					<TeamLogo id={SIDE[moment.event.team]} className="size-7" />
				</span>
				<span className="badge absolute right-2.5 top-2.5 bg-black/60">3D</span>
			</MomentFrame>
			<p className="mt-2.5 truncate text-sm font-semibold">{replayTitle(replay)}</p>
			<p className="mt-0.5 truncate text-xs text-dim">{replaySubtitle(replay)}</p>
		</Link>
	);
}

/** The broadcast look with motion blur, or the look named by ?look= (bloom,neutral,agx,ao,grain,blur; empty for none). */
const defaultLook = (): readonly Look[] => {
	const q = new URLSearchParams(location.search).get("look");
	return q === null ? [...LOOKS, "blur"] : parseLook(q);
};

/** The viewer for one replay and its controls. The scene loads when the stage mounts and is released when it unmounts. */
function Stage({ replay }: { replay: Replay }) {
	const { scene } = replay;
	const box = useRef<HTMLDivElement>(null);
	const scrub = useRef<HTMLInputElement>(null);
	const [viewer, setViewer] = useState<ReplayViewer | null>(null);
	const [progress, setProgress] = useState(0);
	const [error, setError] = useState<string | null>(null);
	const [mode, setMode] = useState<Mode>("replay");
	const [role, setRole] = useState<Role>("scorer");
	const [playing, setPlaying] = useState(true);
	const playingNow = useRef(playing); // read when the viewer opens, so a pause while loading holds
	playingNow.current = playing;
	const press = useRef<{ x: number; y: number; t: number } | null>(null);
	const [rate, setRate] = useState(1);
	const [follow, setFollow] = useState(true);
	const [angles, setAngles] = useState(false); // the free camera's planned angles, listed
	const [sound, setSound] = useState(true);
	const [perf, setPerf] = useState<PerfReport | null>(null);

	useEffect(() => {
		let live = true,
			opened: ReplayViewer | null = null;
		const opening = ReplayViewer.open(box.current!, scene.id, {
			look: defaultLook(),
			progress: (done, total) => live && setProgress(done / total),
		});
		sceneLoading(opening);
		opening
			.then((v) => {
				if (!live) return v.dispose();
				opened = v;
				v.playing = playingNow.current;
				v.onTick = (at) => {
					if (scrub.current) scrub.current.value = String(at);
				};
				v.onFollow = setFollow;
				v.onMode = (m, r) => {
					setMode(m);
					setRole(r);
				};
				v.statsOf = finalStats;
				v.onPerf = setPerf;
				setViewer(v);
			})
			.catch((e: unknown) => live && setError(String(e)));
		return () => {
			live = false;
			opened?.dispose();
		};
	}, [scene.id]);

	useEffect(() => {
		if (viewer) viewer.playing = playing;
	}, [viewer, playing]);
	useEffect(() => {
		if (viewer) viewer.rate = rate;
	}, [viewer, rate]);
	useEffect(() => {
		viewer?.setSound(sound);
	}, [viewer, sound]);
	useEffect(() => {
		// ?perf shows where the frame time goes
		if (viewer && new URLSearchParams(location.search).has("perf")) viewer.setPerf(true);
	}, [viewer]);

	const pickMode = (to: Mode, as = role) => {
		setMode(to);
		setRole(as);
		setAngles(false);
		viewer?.setMode(to, as);
	};
	const heads = viewer?.heads ?? [];
	const panel = !!viewer && (mode === "player" || mode === "explore"); // the mode has controls of its own
	const nameOf = (key: number) =>
		viewer?.scene.people
			.find((p) => p.meta.key === key)
			?.meta.name?.split(" ")
			.at(-1);

	const frame = useRef<HTMLDivElement>(null);
	const [full, setFull] = useState(false);
	useEffect(() => {
		const on = () => setFull(document.fullscreenElement === frame.current);
		document.addEventListener("fullscreenchange", on);
		return () => document.removeEventListener("fullscreenchange", on);
	}, []);
	const toggleFull = () =>
		(document.fullscreenElement ? document.exitFullscreen() : frame.current?.requestFullscreen())?.catch(() => {});

	return (
		<section ref={frame} className={`relative w-full overflow-hidden bg-black ${full ? "h-full" : STAGE}`}>
			{/* biome-ignore lint/a11y/noStaticElementInteractions: a click on the scene; the play button does the same */}
			<div
				ref={box}
				className="absolute inset-0"
				onContextMenu={(e) => e.preventDefault()}
				onPointerDown={(e) => {
					if (e.button === 0) press.current = { x: e.clientX, y: e.clientY, t: e.timeStamp };
				}}
				onPointerUp={(e) => {
					// a click on a player goes to them (their eyes, or the free camera round them; a first tap on a touch screen only
					// shows their card); elsewhere it pauses or plays, as on the game's video; a drag (the free camera's orbit) does neither
					const p = press.current;
					press.current = null;
					if (!p || e.button !== 0 || Math.hypot(e.clientX - p.x, e.clientY - p.y) >= CLICK.px || e.timeStamp - p.t >= CLICK.ms)
						return;
					if (!viewer?.choose(e.clientX, e.clientY, e.pointerType === "touch")) setPlaying((on) => !on);
				}}
			/>
			<div className="pointer-events-none absolute inset-x-0 top-0 h-28 bg-gradient-to-b from-black/50 to-transparent" />
			<div className="pointer-events-none absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-black/70 via-black/30 to-transparent" />
			{!viewer && (
				<div className="absolute inset-0 grid place-items-center text-sm text-dim">
					{error ? (
						<p className="px-6 text-center">The replay could not be loaded. {error}</p>
					) : (
						<div className="w-48 text-center">
							<p>{progress < 1 ? "Loading the scene" : "Building the scene"}</p>
							<div className="mt-3 h-1 overflow-hidden rounded-full bg-white/10">
								<div
									className="h-full rounded-full bg-paper transition-[width]"
									style={{ width: `${Math.round(progress * 100)}%` }}
								/>
							</div>
						</div>
					)}
				</div>
			)}

			<div className="pointer-events-none absolute inset-x-4 top-3 flex flex-wrap items-start justify-between gap-2 sm:inset-x-8 sm:top-5">
				<div className="pointer-events-auto flex flex-wrap items-center gap-2 max-sm:hidden">
					<span className="pill cursor-default bg-black/60">
						<TeamLogo id={SIDE[replay.moment.event.team]} className="size-5" />
						{replayTitle(replay)} &middot; {replaySubtitle(replay)}
					</span>
					<Link to={`/final?t=${lead(replay.moment.event)}`} className="pill bg-black/60">
						Watch it on the broadcast
					</Link>
				</div>
				<div className="seg pointer-events-auto max-w-full bg-black/60" role="group" aria-label="Mode">
					{MODES.map((m) => (
						<button
							type="button"
							key={m}
							className="inline-flex items-center gap-1.5 disabled:opacity-40 max-sm:!px-2.5 [&>svg]:size-3.5"
							aria-pressed={mode === m}
							disabled={!viewer || (m === "player" && !heads.length)}
							onClick={() => pickMode(m)}
						>
							{ICON[m]}
							{MODE_LABEL[m]}
						</button>
					))}
				</div>
			</div>

			{perf && <PerfPanel report={perf} />}

			<div className="absolute inset-x-4 bottom-3 sm:inset-x-8 sm:bottom-5">
				{/* the mode's own controls, standing on the bar */}
				<div className={`mb-3 flex items-end gap-3 ${panel ? "" : "hidden"}`}>
					<div className="min-w-0">
						{viewer && mode === "player" && (
							<div
								className="seg max-w-full overflow-x-auto bg-black/60 [scrollbar-width:none]"
								role="group"
								aria-label="Whose eyes"
							>
								{heads.map((h) => (
									<button type="button" key={h.role} aria-pressed={role === h.role} onClick={() => pickMode("player", h.role)}>
										{ROLE_LABEL[h.role]} · {nameOf(h.key)}
									</button>
								))}
							</div>
						)}
						{viewer && mode === "explore" && (
							<div className="flex flex-col items-start gap-2 text-xs">
								{angles && (
									<div className="seg flex max-w-full flex-wrap bg-black/60" role="group" aria-label="Angles">
										{viewer.angles.map((name) => (
											<button
												type="button"
												key={name}
												onClick={() => {
													viewer.lookFrom(name);
													setAngles(false);
												}}
											>
												{ANGLE_LABEL[name] ?? name}
											</button>
										))}
										<button type="button" aria-pressed={follow} onClick={() => viewer.setFollow(!follow)}>
											Follow {player(replay.moment.event.player).last}
										</button>
									</div>
								)}
								<div className="flex items-center gap-2">
									<button
										type="button"
										className={`badge ${angles ? "bg-paper text-ink" : "bg-black/60"}`}
										aria-expanded={angles}
										onClick={() => setAngles(!angles)}
									>
										Angles
									</button>
									{/* how to move, said once as the mode opens, then gone */}
									<span className="hint rounded-full bg-black/60 px-3 py-1 text-dim">
										<span className="pointer-coarse:hidden">Drag to orbit · right-drag to pan · scroll to zoom</span>
										<span className="hidden pointer-coarse:inline">Drag to orbit · pinch to zoom</span>
									</span>
								</div>
							</div>
						)}
					</div>
				</div>
				<div className="flex flex-wrap items-center gap-2 sm:flex-nowrap sm:gap-3">
					<input
						ref={scrub}
						type="range"
						min={0}
						max={1}
						step={0.001}
						defaultValue={0}
						aria-label="Position in the moment"
						className="order-first h-5 min-w-0 basis-full cursor-pointer accent-[var(--paper)] sm:order-none sm:basis-auto sm:flex-1"
						onInput={(e) => viewer?.seek(Number(e.currentTarget.value))}
					/>
					<button
						type="button"
						className="pill icon bg-black/60 sm:order-first"
						aria-label={playing ? "Pause" : "Play"}
						onClick={() => setPlaying(!playing)}
					>
						{playing ? ICON.pause : ICON.play}
					</button>
					<div className="seg bg-black/60" role="group" aria-label="Speed">
						{RATES.map(([value, label]) => (
							<button type="button" key={value} aria-pressed={rate === value} onClick={() => setRate(value)}>
								{label}
							</button>
						))}
					</div>
					<span className="flex-1 sm:hidden" />
					<button
						type="button"
						className="pill icon bg-black/60 aria-pressed:bg-paper"
						aria-pressed={!sound}
						aria-label={sound ? "Mute" : "Unmute"}
						onClick={() => setSound(!sound)}
					>
						{sound ? ICON.sound : ICON.muted}
					</button>
					<button
						type="button"
						className="pill icon bg-black/60"
						aria-label={full ? "Leave full screen" : "Full screen"}
						onClick={toggleFull}
					>
						{full ? ICON.exitFullscreen : ICON.enterFullscreen}
					</button>
				</div>
			</div>
		</section>
	);
}

const icon = (d: React.ReactNode, filled = false) => (
	<svg
		aria-hidden="true"
		viewBox="0 0 16 16"
		{...(filled
			? { fill: "currentColor" }
			: {
					fill: "none",
					stroke: "currentColor",
					strokeWidth: 1.5,
					strokeLinecap: "round" as const,
					strokeLinejoin: "round" as const,
				})}
	>
		{d}
	</svg>
);
/** The stage's icons (player/mount.ts draws the same). */
const ICON = {
	play: icon(<path d="M5 3.5v9l7.5-4.5z" />, true),
	pause: icon(
		<>
			<rect x="4" y="3.5" width="2.6" height="9" rx=".6" />
			<rect x="9.4" y="3.5" width="2.6" height="9" rx=".6" />
		</>,
		true,
	),
	muted: icon(<path d="M2.5 6h2.5l3.5-3v10L5 10H2.5zM11 6l3 4M14 6l-3 4" />),
	sound: icon(<path d="M2.5 6h2.5l3.5-3v10L5 10H2.5zM11 5.5a3.5 3.5 0 0 1 0 5M12.8 3.8a6 6 0 0 1 0 8.4" />),
	enterFullscreen: icon(<path d="M3 6V3h3M10 3h3v3M13 10v3h-3M6 13H3v-3" />),
	exitFullscreen: icon(<path d="M6 3v3H3M13 6h-3V3M10 13v-3h3M3 10h3v3" />),
	replay: icon(<path d="M2.5 6.5h11v6.5h-11zM2.5 6.5 3.5 3h10l-1 3.5M6 3 5 6.5M9.5 3l-1 3.5" />),
	player: icon(
		<>
			<path d="M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z" />
			<circle cx="8" cy="8" r="2" />
		</>,
	),
	tactical: icon(
		<>
			<rect x="2.5" y="2.5" width="11" height="11" rx="1.5" />
			<path d="M5 11l2.5-3 2 1.5L12 5.5M10 5.5h2v2" />
		</>,
	),
	explore: icon(
		<>
			<circle cx="8" cy="8" r="1.8" />
			<ellipse cx="8" cy="8" rx="6.2" ry="2.8" transform="rotate(-25 8 8)" />
		</>,
	),
};

const ms = (x: number) => x.toFixed(x < 10 ? 2 : 1);

/**
 * Where the frame time goes: frames a second, then CPU and GPU milliseconds a frame by part, the slowest first (GPU parts are the
 * scene, the ice's mirror and each pass of the broadcast look). The CPU parts are also in the browser's performance trace.
 */
function PerfPanel({ report: r }: { report: PerfReport }) {
	const rows = (title: string, spans: [string, number][]) => (
		<>
			<p className="mt-2 text-dim">{title}</p>
			{spans.map(([name, t]) => (
				<p key={name} className="flex justify-between gap-4">
					<span>{name}</span>
					<span className="tabular-nums">{ms(t)} ms</span>
				</p>
			))}
		</>
	);
	return (
		<div className="pointer-events-none absolute right-4 top-16 w-56 rounded-xl bg-black/75 sm:right-8 sm:top-20 px-3 py-2.5 font-mono text-[11px] leading-relaxed">
			<p className="flex justify-between gap-4 font-semibold">
				<span>{r.fps.toFixed(0)} fps</span>
				<span className="tabular-nums">{ms(r.frame)} ms</span>
			</p>
			{rows("CPU", r.cpu)}
			{!r.gpu ? (
				<p className="mt-2 text-dim">GPU timer not offered by this browser</p>
			) : r.gpu.length ? (
				rows("GPU", r.gpu)
			) : (
				<p className="mt-2 text-dim">GPU: timing the first frames</p>
			)}
			<p className="mt-2 text-dim">
				{r.calls} draws · {(r.triangles / 1e6).toFixed(2)} M triangles · {r.pixels[0]}×{r.pixels[1]} px
			</p>
		</div>
	);
}
