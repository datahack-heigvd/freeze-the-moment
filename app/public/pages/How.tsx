import { type PointerEvent, type ReactNode, type RefObject, use, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import * as THREE from "three";
import { Container, PageTitle } from "../components/Section";
import {
	at,
	BASE,
	blendPose,
	cameraAt,
	colours,
	footprint,
	frameUrl,
	GOAL,
	H,
	type How,
	type HowPerson,
	label,
	linesPath,
	loadHow,
	loadPose,
	loadSticks,
	outline,
	type PoseData,
	poseOf,
	project,
	type SticksData,
	viewUrl,
	W,
	type XY,
} from "../how/data";
import { type Layers, World } from "../how/world";
import { useReplays } from "../lib/replays";
import { TEAM_COLOR } from "../player/events";
import { prefetchScene } from "../replay/scene";

const STEPS = [
	{ name: "The broadcast", line: "One camera, one goal: all we start from." },
	{ name: "The camera", line: "Where the camera stands and how it looks, solved every frame." },
	{ name: "Who's who", line: "Every player found and followed, number and all." },
	{ name: "On the ice", line: "From the picture to where each one stands on the rink." },
	{ name: "Bodies", line: "Each player's pose, then a 3D body fitted to it." },
	{ name: "The arena", line: "Ice, boards and crowd, painted from frames where nobody is in the way." },
	{ name: "Stick and puck", line: "Each stick found in the picture, lifted into 3D from the hands; the puck followed in." },
	{ name: "Into 3D", line: "Now the camera can go anywhere." },
] as const;

/** The 3D each step draws, from the step that opens it (Bodies) on. */
const LAYERS: Partial<Record<number, Layers>> = {
	4: { whole: false, people: true, sticks: false, puck: false },
	5: { whole: true, people: false, sticks: false, puck: false },
	6: { whole: false, people: true, sticks: true, puck: true },
	7: { whole: true, people: true, sticks: true, puck: true },
};
/** The steps with a wipe, and what either side of it shows. */
const WIPES: Partial<Record<number, [string, string]>> = {
	2: ["Picture", "Tracked"],
	4: ["Pose", "3D bodies"],
	5: ["Picture", "3D arena"],
	6: ["Found", "3D sticks"],
	7: ["Picture", "3D"],
};
const SCENE_FROM = 2; // the step from which the 3D scene is fetched ahead, and opened one later
const WIPE = 0.45; // where the wipe rests on each step
const SOLVE_MS = 2200;
const GLIDE_MS = 1600;
const FILL_MS = 6000;
const MAP = { scale: 13.5, x: W / 2, y: H / 2 + 6 }; // the top-down rink across the stage: px per m, centre

/** An eased 0..1 over ms, from each change of `key` (and `again`). */
function useRun(key: unknown, ms: number, loop = false) {
	const [u, setU] = useState(0);
	const [again, setAgain] = useState(0);
	// biome-ignore lint/correctness/useExhaustiveDependencies: key and again only restart the run
	useEffect(() => {
		if (matchMedia("(prefers-reduced-motion: reduce)").matches) return setU(1);
		let raf = 0;
		const t0 = performance.now(),
			tick = (now: number) => {
				const x = (now - t0) / ms;
				setU(loop ? x % 1 : Math.min(x, 1));
				if (loop || x < 1) raf = requestAnimationFrame(tick);
			};
		setU(0);
		raf = requestAnimationFrame(tick);
		return () => cancelAnimationFrame(raf);
	}, [key, again, ms, loop]);
	return [u, () => setAgain((n) => n + 1)] as const;
}

const smooth = (x: number) => x * x * (3 - 2 * x);

/** What load gives, fetched once `when` first holds. */
function useLoaded<T>(load: () => Promise<T>, when: boolean) {
	const [got, setGot] = useState<T | null>(null);
	useEffect(() => {
		if (!when || got) return;
		let gone = false;
		load().then(
			(v) => !gone && setGot(v),
			() => {},
		);
		return () => {
			gone = true;
		};
	}, [load, when, got]);
	return got;
}

/** One goal, step by step from the broadcast to its 3D replay. */
export function HowPage() {
	const how = use(loadHow());
	return (
		<Container>
			<PageTitle title="How it works" subtitle="One goal, from the broadcast to the 3D replay." />
			<Steps how={how} />
		</Container>
	);
}

function Steps({ how }: { how: How }) {
	const [params, setParams] = useSearchParams();
	const [a, b] = how.frames;
	const step = Math.min(Math.max(Number(params.get("step") ?? 1) - 1, 0), STEPS.length - 1) || 0;
	const go = useCallback(
		(s: number) =>
			setParams(
				(p) => {
					p.set("step", String(Math.min(Math.max(s, 0), STEPS.length - 1) + 1));
					return p;
				},
				{ replace: true, preventScrollReset: true },
			),
		[setParams],
	);
	const [frame, setFrame] = useState(() => Math.min(Math.max(Number(params.get("t")) || how.puck.shot.release - 3, a), b));
	const [playing, setPlaying] = useState(false);
	const [hover, setHover] = useState<number | null>(null);
	const colour = useColours(how);
	const rail = useRef<HTMLOListElement>(null);

	// across a phone the rail scrolls sideways: the step shown kept in it
	useEffect(() => {
		const r = rail.current,
			li = r?.children[step] as HTMLElement | undefined;
		if (r && li && r.scrollWidth > r.clientWidth) r.scrollTo({ left: li.offsetLeft - 16, behavior: "smooth" });
	}, [step]);

	useEffect(() => {
		const key = (e: KeyboardEvent) => {
			const t = e.target as HTMLElement;
			if (e.altKey || e.ctrlKey || e.metaKey || t.closest("input, textarea, select, [contenteditable]")) return;
			if (e.key === "ArrowRight") go(step + 1);
			else if (e.key === "ArrowLeft") go(step - 1);
			else if (e.key === " " && !t.closest("button")) setPlaying((p) => !p);
			else return;
			e.preventDefault();
		};
		addEventListener("keydown", key);
		return () => removeEventListener("keydown", key);
	}, [go, step]);

	useEffect(() => {
		if (!playing) return;
		const id = setInterval(() => setFrame((f) => (f >= b ? a : f + 1)), 1000 / how.fps);
		return () => clearInterval(id);
	}, [playing, a, b, how.fps]);

	// the moment's frames, nearest the one shown first
	// biome-ignore lint/correctness/useExhaustiveDependencies: once, later frames come from the cache
	useEffect(() => {
		const order = Array.from({ length: b - a + 1 }, (_, k) => a + k).sort((x, y) => Math.abs(x - frame) - Math.abs(y - frame));
		const imgs = order.map((i) =>
			Object.assign(new Image(), { decoding: "async", fetchPriority: i === frame ? "high" : "low", src: frameUrl(i) }),
		);
		return () => {
			for (const im of imgs) im.src = "";
		};
	}, []);

	useEffect(() => {
		if (step >= SCENE_FROM) void prefetchScene(GOAL).catch(() => {});
	}, [step]);

	return (
		<div className="grid gap-4 pb-12 pt-4 lg:grid-cols-[13rem_minmax(0,1fr)] lg:gap-8">
			<ol ref={rail} className="relative -mx-4 flex gap-1 overflow-x-auto px-4 lg:mx-0 lg:flex-col lg:px-0" aria-label="Steps">
				{STEPS.map((s, k) => (
					<li key={s.name} className="shrink-0">
						<button
							type="button"
							onClick={() => go(k)}
							aria-current={k === step ? "step" : undefined}
							className={`flex w-full items-center gap-3 rounded-card px-3 py-2 text-left text-sm transition-colors ${k === step ? "bg-panel-hover text-paper" : "text-dim hover:bg-panel hover:text-paper"}`}
						>
							<span
								className={`grid size-6 shrink-0 place-items-center rounded-full text-xs font-semibold tabular-nums ${k === step ? "bg-paper text-ink" : k < step ? "bg-white/15 text-paper" : "border border-white/15"}`}
							>
								{k + 1}
							</span>
							<span className="whitespace-nowrap">{s.name}</span>
						</button>
					</li>
				))}
			</ol>
			<div className="min-w-0">
				<Stage
					how={how}
					step={step}
					frame={frame}
					setFrame={setFrame}
					playing={playing}
					hover={hover}
					setHover={setHover}
					colour={colour}
				/>
				<div className="mt-3 flex items-center gap-3">
					<button
						type="button"
						className="grid size-9 shrink-0 place-items-center rounded-full border border-white/15 text-paper disabled:opacity-30"
						onClick={() => go(step - 1)}
						disabled={step === 0}
						aria-label="Previous step"
					>
						<span aria-hidden>&larr;</span>
					</button>
					<button
						type="button"
						className="grid size-9 shrink-0 place-items-center rounded-full bg-paper text-ink"
						onClick={() => setPlaying((p) => !p)}
						aria-label={playing ? "Pause" : "Play"}
					>
						{playing ? <PauseIcon /> : <PlayIcon />}
					</button>
					<input
						type="range"
						min={a}
						max={b}
						value={frame}
						onChange={(e) => {
							setPlaying(false);
							setFrame(Number(e.target.value));
						}}
						className="min-w-0 flex-1 accent-paper"
						aria-label="Moment"
					/>
					<button
						type="button"
						className="grid size-9 shrink-0 place-items-center rounded-full border border-white/15 text-paper disabled:opacity-30"
						onClick={() => go(step + 1)}
						disabled={step === STEPS.length - 1}
						aria-label="Next step"
					>
						<span aria-hidden>&rarr;</span>
					</button>
				</div>
				<p className="mt-3 min-h-6 text-paper">
					<span className="mr-2 text-dim tabular-nums">
						{step + 1}/{STEPS.length}
					</span>
					{STEPS[step]!.line}
				</p>
				<Roster how={how} frame={frame} hover={hover} setHover={setHover} colour={colour} />
			</div>
		</div>
	);
}

/** Each person's colour, their team's as the site shows it (the scorer's team from the goal's replay). */
function useColours(how: How) {
	const replay = useReplays().find((r) => r.scene.id === GOAL);
	return useMemo(() => {
		const scorer = replay?.scene.scorer?.team ?? 0,
			side = replay?.moment.event.team ?? "Home",
			other = side === "Home" ? "Away" : "Home";
		return colours(how, (team) => TEAM_COLOR[team === scorer ? side : other]);
	}, [how, replay]);
}

type StageProps = {
	how: How;
	step: number;
	frame: number;
	setFrame: (f: number) => void;
	playing: boolean;
	hover: number | null;
	setHover: (k: number | null) => void;
	colour: Map<number, string>;
};

function Stage({ how, step, frame, setFrame, playing, hover, setHover, colour }: StageProps) {
	const box = useRef<HTMLDivElement>(null);
	const host = useRef<HTMLDivElement>(null);
	const [world, setWorld] = useState<World | null>(null);
	const [loading, setLoading] = useState<number | null>(null);
	const [wipe, setWipe] = useState(WIPE);
	const [flown, setFlown] = useState<"no" | "flying" | "yes">("no");
	const i = at(how, frame);
	const camera = useMemo(() => cameraAt(poseOf(how.cams[i]!)), [how, i]);
	const wants3d = step >= SCENE_FROM + 1;
	const layers = LAYERS[step];
	const free = step === 7 && flown !== "no";
	const sides = free ? undefined : WIPES[step],
		wiping = !!sides;

	const [solve, solveAgain] = useRun(step === 1 ? "solve" : "rest", SOLVE_MS);
	const rough = useMemo(() => roughCamera(how, frame, smooth(solve)), [how, frame, solve]);

	// the 3D scene, opened once it is near and kept while the page is
	const opening = useRef(false);
	useEffect(() => {
		const into = host.current;
		if (!wants3d || opening.current || !into) return;
		opening.current = true;
		setLoading(0);
		World.open(into, how, (d, t) => setLoading(t ? d / t : 0)).then(
			(w) => {
				if (!into.isConnected) return w.dispose();
				setWorld(w);
				setLoading(null);
				Object.assign(window, { howWorld: w }); // for headless checks
			},
			() => setLoading(null),
		);
	}, [wants3d, how]);
	useEffect(() => () => world?.dispose(), [world]);

	useEffect(() => {
		if (layers) world?.set(layers);
	}, [world, layers]);
	useEffect(() => {
		world?.setTime(frame, playing ? 1 : 0);
	}, [world, frame, playing]);
	useEffect(() => world?.setHover(hover), [world, hover]);
	useEffect(() => {
		if (world) world.onTime = (t) => setFrame(Math.round(Math.min(Math.max(t, how.frames[0]), how.frames[1])));
	}, [world, setFrame, how]);

	// a wipe opens from the right on each step that has one; leaving Into 3D lands the camera back
	// biome-ignore lint/correctness/useExhaustiveDependencies: on each step only, not when the flight or the world change
	useEffect(() => {
		setWipe(WIPE);
		if (step !== 7 && flown !== "no") {
			world?.land();
			setFlown("no");
		}
	}, [step]);

	const clip = wiping ? { clipPath: `inset(0 0 0 ${wipe * 100}%)` } : undefined;
	const people = how.people.filter((p) => p.outline[i]);
	const hit = !free && step !== 3;
	const shown = layers && world;
	const solidPicture = step === 2 || step === 4 || step === 6;
	const pose = useLoaded(loadPose, step >= 3);
	const sticks = useLoaded(loadSticks, step >= 5);

	return (
		<div
			ref={box}
			className="relative aspect-video w-full touch-pan-y select-none overflow-hidden rounded-card bg-black"
			onPointerLeave={() => setHover(null)}
		>
			{/* before: the broadcast as it is */}
			<img
				src={frameUrl(frame)}
				alt=""
				className={`absolute inset-0 size-full transition-opacity duration-500 ${step === 3 ? "opacity-25" : free ? "opacity-0" : ""}`}
				draggable={false}
			/>
			<svg viewBox={`0 0 ${W} ${H}`} className="pointer-events-none absolute inset-0 size-full" aria-hidden>
				{step === 4 && pose && <Skeletons pose={pose} i={i} hover={hover} />}
				{step === 6 && sticks && <StickOutlines sticks={sticks} i={i} />}
			</svg>
			{/* after: what the step adds, left of the wipe the broadcast (and what is found in it) still */}
			<div className="absolute inset-0" style={clip}>
				{solidPicture && (
					<img
						src={frameUrl(frame)}
						alt=""
						className="absolute inset-0 size-full"
						style={step === 2 ? undefined : { filter: "brightness(.42) saturate(.5)" }}
						draggable={false}
					/>
				)}
				<div ref={host} className={`absolute inset-0 transition-opacity duration-500 ${shown ? "opacity-100" : "opacity-0"}`} />
				<svg viewBox={`0 0 ${W} ${H}`} className="pointer-events-none absolute inset-0 size-full" aria-hidden>
					{step === 2 && people.map((p) => <Shape key={p.key} p={p} i={i} fill={colour.get(p.key)!} hover={hover} tag />)}
					{step === 6 && world && <Sticks how={how} world={world} camera={camera} frame={frame} hover={hover} colour={colour} />}
				</svg>
			</div>
			<svg viewBox={`0 0 ${W} ${H}`} className="pointer-events-none absolute inset-0 size-full" aria-hidden="true">
				{step === 1 && <Solve how={how} camera={rough} done={solve >= 1} again={solveAgain} />}
				{step === 6 && <PuckPath how={how} camera={camera} frame={frame} />}
				{step === 3 && <IceMap how={how} frame={frame} camera={camera} hover={hover} setHover={setHover} colour={colour} />}
				{hit &&
					people.map((p) => (
						// biome-ignore lint/a11y/noStaticElementInteractions: hover highlights only, for pointers
						<polygon
							key={p.key}
							points={p.outline[i]!.join(" ")}
							className="pointer-events-auto cursor-pointer"
							fill="transparent"
							stroke={hover === p.key && step !== 2 ? colour.get(p.key) : "none"}
							strokeWidth={2}
							onPointerEnter={() => setHover(p.key)}
							onPointerLeave={() => setHover(null)}
							onClick={() => setHover(hover === p.key ? null : p.key)}
						/>
					))}
				{hit && hover !== null && <Tag p={how.people.find((p) => p.key === hover)!} i={i} colour={colour.get(hover)!} />}
			</svg>
			{step === 1 && <CameraMap how={how} camera={rough} />}
			{step === 5 && <AtlasFill how={how} />}
			{sides && (
				<>
					<Side className="left-3">{sides[0]}</Side>
					<Side className="right-3">{sides[1]}</Side>
					<Wipe at={wipe} set={setWipe} box={box} />
				</>
			)}
			{step === 7 && (
				<button
					type="button"
					className="absolute bottom-3 right-3 rounded-full bg-paper px-4 py-2 text-sm font-semibold text-ink shadow-overlay disabled:opacity-60"
					disabled={!world || flown === "flying"}
					onClick={() => {
						if (!world) return;
						if (flown === "no") {
							setFlown("flying");
							world.fly(() => setFlown("yes"));
						} else {
							setFlown("flying");
							world.land(() => setFlown("no"));
						}
					}}
				>
					{flown === "no" ? "Fly into 3D" : "Back to the broadcast"}
				</button>
			)}
			{wants3d && layers && !world && (
				<div className="absolute inset-0 grid place-items-center bg-black/40" role="status" aria-label="Loading the 3D scene">
					<div className="h-1 w-40 overflow-hidden rounded-full bg-white/15">
						<div className="h-full bg-paper transition-[width]" style={{ width: `${(loading ?? 0) * 100}%` }} />
					</div>
				</div>
			)}
		</div>
	);
}

function Side({ className, children }: { className: string; children: ReactNode }) {
	return (
		<span className={`pointer-events-none absolute top-3 rounded-full bg-black/70 px-3 py-1 text-xs text-paper ${className}`}>
			{children}
		</span>
	);
}

/** The divider between before (left) and after (right), dragged anywhere along it. */
function Wipe({ at: x, set, box }: { at: number; set: (x: number) => void; box: RefObject<HTMLDivElement | null> }) {
	const move = (e: PointerEvent) => {
		const r = box.current!.getBoundingClientRect();
		set(Math.min(Math.max((e.clientX - r.left) / r.width, 0.02), 0.98));
	};
	return (
		<div
			className="absolute inset-y-0 z-10 -ml-4 w-8 cursor-ew-resize touch-none"
			style={{ left: `${x * 100}%` }}
			onPointerDown={(e) => {
				e.currentTarget.setPointerCapture(e.pointerId);
				move(e);
			}}
			onPointerMove={(e) => e.currentTarget.hasPointerCapture(e.pointerId) && move(e)}
			role="slider"
			aria-label="Before and after"
			aria-valuemin={0}
			aria-valuemax={100}
			aria-valuenow={Math.round(x * 100)}
			tabIndex={0}
			onKeyDown={(e) => {
				if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
					e.stopPropagation();
					e.preventDefault();
					set(Math.min(Math.max(x + (e.key === "ArrowLeft" ? -0.05 : 0.05), 0.02), 0.98));
				}
			}}
		>
			<div className="absolute inset-y-0 left-1/2 w-px bg-paper/80" />
			<div className="absolute left-1/2 top-1/2 grid size-8 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-paper text-xs text-ink shadow-overlay">
				<span aria-hidden>&harr;</span>
			</div>
		</div>
	);
}

/** A person's outline in the picture, faded while someone else is picked. */
function Shape({ p, i, fill, hover, tag }: { p: HowPerson; i: number; fill: string; hover: number | null; tag?: boolean }) {
	const pts = p.outline[i]!,
		dim = hover !== null && hover !== p.key;
	const top = topOf(pts);
	return (
		<g style={{ opacity: dim ? 0.2 : 1, transition: "opacity .2s" }}>
			<polygon
				points={pts.join(" ")}
				style={{ fill, fillOpacity: 0.42, stroke: fill, strokeOpacity: 0.95 }}
				strokeWidth={1.4}
				strokeLinejoin="round"
			/>
			{tag && p.team >= 0 && p.jersey && (
				<text
					x={top[0]}
					y={top[1] - 6}
					textAnchor="middle"
					className="fill-paper text-[15px] font-bold"
					style={{ paintOrder: "stroke", stroke: "#000a", strokeWidth: 3 }}
				>
					{p.jersey}
				</text>
			)}
		</g>
	);
}

function topOf(pts: number[]): XY {
	let k = 1;
	for (let j = 3; j < pts.length; j += 2) if (pts[j]! < pts[k]!) k = j;
	return [pts[k - 1]!, pts[k]!];
}

/** The hovered person's name over them. */
function Tag({ p, i, colour }: { p: HowPerson; i: number; colour: string }) {
	const pts = p.outline[i];
	if (!pts) return null;
	const [x, y] = topOf(pts);
	return (
		<text
			x={x}
			y={Math.max(y - 10, 18)}
			textAnchor="middle"
			className="text-[17px] font-semibold"
			style={{ fill: colour, paintOrder: "stroke", stroke: "#000c", strokeWidth: 4 }}
		>
			{label(p)}
		</text>
	);
}

/** Everyone's pose as found in the picture: the keypoints of the body fitted to them, limb by limb. */
function Skeletons({ pose, i, hover }: { pose: PoseData; i: number; hover: number | null }) {
	return (
		<g strokeLinecap="round">
			{Object.entries(pose.people).map(([key, rows]) => {
				const kp = rows[i];
				if (!kp) return null;
				return (
					<g key={key} style={{ opacity: hover !== null && hover !== Number(key) ? 0.2 : 1 }}>
						{pose.limbs.map(([a, b], k) => (
							<line
								key={`${a}-${b}`}
								x1={kp[2 * a]}
								y1={kp[2 * a + 1]}
								x2={kp[2 * b]}
								y2={kp[2 * b + 1]}
								stroke={`rgb(${pose.colours[k]!.join(" ")})`}
								strokeWidth={2.6}
							/>
						))}
						{Array.from({ length: kp.length / 2 }, (_, j) => (
							// biome-ignore lint/suspicious/noArrayIndexKey: a keypoint is its joint's index
							<circle key={j} cx={kp[2 * j]} cy={kp[2 * j + 1]} r={1.6} fill="#fff" />
						))}
					</g>
				);
			})}
		</g>
	);
}

/** The sticks as found in the picture: their outlines. */
function StickOutlines({ sticks, i }: { sticks: SticksData; i: number }) {
	return (
		<g>
			{(sticks[i] ?? []).map((poly, k) => (
				<polygon
					// biome-ignore lint/suspicious/noArrayIndexKey: a frame's stick outlines have no identity of their own
					key={k}
					points={poly.join(" ")}
					fill="var(--away)"
					fillOpacity={0.55}
					stroke="var(--away)"
					strokeWidth={2.5}
					strokeLinejoin="round"
					style={{ filter: "drop-shadow(0 0 2px #000)" }}
				/>
			))}
		</g>
	);
}

/** Everyone's stick in 3D through the frame's camera, from the top hand down to the blade, and where the blade has been. */
function Sticks({
	how,
	world,
	camera,
	frame,
	hover,
	colour,
}: {
	how: How;
	world: World;
	camera: THREE.PerspectiveCamera;
	frame: number;
	hover: number | null;
	colour: Map<number, string>;
}) {
	return (
		<g strokeLinecap="round">
			{how.people.map((p) => {
				const st = world.stick(p.key, frame);
				if (!st || !p.seen[at(how, frame)]) return null;
				const top = project(camera, st[0].x, st[0].y, st[0].z),
					heel = project(camera, st[1].x, st[1].y, st[1].z);
				if (!top || !heel) return null;
				let path = "";
				for (let f = Math.max(frame - 10, how.frames[0]); f <= frame; f++) {
					const b = world.stick(p.key, f),
						q = b && project(camera, b[1].x, b[1].y, b[1].z);
					if (q) path += `${path ? "L" : "M"}${q[0].toFixed(1)} ${q[1].toFixed(1)}`;
				}
				const c = colour.get(p.key)!;
				return (
					<g key={p.key} style={{ opacity: hover !== null && hover !== p.key ? 0.2 : 1 }}>
						<path d={path} fill="none" style={{ stroke: c }} strokeOpacity={0.6} strokeWidth={2} />
						<line x1={top[0]} y1={top[1]} x2={heel[0]} y2={heel[1]} stroke="#000" strokeOpacity={0.7} strokeWidth={6} />
						<line x1={top[0]} y1={top[1]} x2={heel[0]} y2={heel[1]} style={{ stroke: c }} strokeWidth={3.5} />
						<circle cx={heel[0]} cy={heel[1]} r={4} style={{ fill: c }} stroke="#000" strokeWidth={1.5} />
					</g>
				);
			})}
		</g>
	);
}

/**
 * The rink's lines through the frame's camera, starting from a rough guess at it and settling onto the solve: an illustration
 * of the fit, not what the solver went through.
 */
function Solve({ how, camera, done, again }: { how: How; camera: THREE.PerspectiveCamera; done: boolean; again: () => void }) {
	return (
		<g>
			<path
				d={linesPath(camera, how.rink.lines)}
				fill="none"
				stroke={!done ? "var(--away)" : "var(--paper)"}
				strokeWidth={2}
				strokeLinejoin="round"
				style={{ transition: "stroke .4s" }}
			/>
			<foreignObject x={12} y={12} width={440} height={40}>
				<div className="flex items-center gap-2 text-[15px]">
					<span className="rounded-full bg-black/70 px-3 py-1 text-paper">
						{done ? "Fitted" : "Illustration: a rough first guess"}
					</span>
					{done && (
						<button
							type="button"
							className="pointer-events-auto rounded-full bg-black/70 px-3 py-1 text-dim hover:text-paper"
							onClick={again}
						>
							Again
						</button>
					)}
				</div>
			</foreignObject>
		</g>
	);
}

const GUESS = { shift: new THREE.Vector3(6, -5, 4), yaw: 0.14, pitch: -0.07, zoom: 1.3 };

/** The camera a fraction u from a rough guess (u 0) to frame's solve (u 1). */
function roughCamera(how: How, frame: number, u: number) {
	const solve = poseOf(how.cams[at(how, frame)]!),
		turn = new THREE.Quaternion()
			.setFromAxisAngle(new THREE.Vector3(0, 0, 1), GUESS.yaw)
			.multiply(solve.quaternion)
			.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), GUESS.pitch)),
		guess = { position: solve.position.clone().add(GUESS.shift), quaternion: turn, fov: solve.fov * GUESS.zoom };
	return cameraAt(blendPose(guess, solve, u));
}

/** A top-down rink, in its own box: x m across, y up. */
function rinkBox(how: How, pad: number, below: number) {
	const { hx, hy } = how.rink;
	return { x0: -hx - pad, y0: -hy - below, w: 2 * (hx + pad), h: 2 * hy + pad + below };
}
const rinkPath = (pts: XY[], to: (p: XY) => XY) =>
	`${pts
		.map(
			(p, k) =>
				`${k ? "L" : "M"}${to(p)
					.map((v) => v.toFixed(2))
					.join(" ")}`,
		)
		.join("")}Z`;

/** Inset: the rink from above, the camera on it and what it sees of the ice. */
function CameraMap({ how, camera }: { how: How; camera: THREE.PerspectiveCamera }) {
	const c = camera.position,
		box = rinkBox(how, 3, 22),
		to = ([x, y]: XY): XY => [x - box.x0, box.y0 + box.h - y],
		seen = footprint(camera),
		ice = outline(how);
	return (
		<svg
			viewBox={`0 0 ${box.w} ${box.h}`}
			className="absolute bottom-3 right-3 w-[34%] rounded-card bg-black/75 p-1.5 shadow-overlay"
			aria-label="The camera over the rink"
		>
			<defs>
				<clipPath id="how-ice">
					<path d={rinkPath(ice, to)} />
				</clipPath>
			</defs>
			<path d={rinkPath(ice, to)} fill="rgb(236 232 223 / 0.08)" />
			<path d={rinkPath(seen, to)} fill="var(--away)" fillOpacity={0.35} clipPath="url(#how-ice)" />
			{how.rink.lines.slice(1).map((l, k) => (
				<path
					// biome-ignore lint/suspicious/noArrayIndexKey: the rink's lines are fixed
					key={k}
					d={l.map((p, j) => `${j ? "L" : "M"}${to(p).join(" ")}`).join("")}
					fill="none"
					stroke="rgb(236 232 223 / 0.35)"
					strokeWidth={0.3}
				/>
			))}
			<path d={rinkPath(ice, to)} fill="none" stroke="var(--paper)" strokeWidth={0.4} />
			<path
				d={`M${to([c.x, c.y]).join(" ")}L${to(seen[0]!).join(" ")}M${to([c.x, c.y]).join(" ")}L${to(seen[8]!).join(" ")}`}
				stroke="var(--away)"
				strokeWidth={0.35}
				strokeDasharray="1 1"
			/>
			<circle cx={to([c.x, c.y])[0]} cy={to([c.x, c.y])[1]} r={1.4} fill="var(--away)" />
		</svg>
	);
}

/** Everyone's dots gliding from their feet in the picture onto the rink seen from above, with where they came from. */
function IceMap({
	how,
	frame,
	camera,
	hover,
	setHover,
	colour,
}: {
	how: How;
	frame: number;
	camera: THREE.PerspectiveCamera;
	hover: number | null;
	setHover: (k: number | null) => void;
	colour: Map<number, string>;
}) {
	const [u, again] = useRun("glide", GLIDE_MS);
	const e = smooth(u),
		map = ([x, y]: XY): XY => [MAP.x + x * MAP.scale, MAP.y - y * MAP.scale],
		place = (q: XY): XY | null => {
			const top = map(q);
			if (e >= 1) return top;
			const foot = project(camera, q[0], q[1]);
			return foot ? [foot[0] + (top[0] - foot[0]) * e, foot[1] + (top[1] - foot[1]) * e] : top;
		},
		trail = (xy: (XY | null)[]) =>
			xy
				.slice(Math.max(frame - 40, 0), frame + 1)
				.filter((q): q is XY => !!q)
				.map(
					(q, k) =>
						`${k ? "L" : "M"}${map(q)
							.map((v) => v.toFixed(1))
							.join(" ")}`,
				)
				.join(""),
		puck = how.puck.xy[frame],
		puckAt = puck && how.puck.show[frame]! > 0.3 ? place(puck) : null;
	return (
		<g>
			<g style={{ opacity: e }}>
				<path d={rinkPath(outline(how), map)} fill="rgb(11 12 14 / 0.6)" stroke="var(--paper)" strokeWidth={1.5} />
				{how.rink.lines.slice(1).map((l, k) => (
					<path
						// biome-ignore lint/suspicious/noArrayIndexKey: the rink's lines are fixed
						key={k}
						d={l.map((p, j) => `${j ? "L" : "M"}${map(p).join(" ")}`).join("")}
						fill="none"
						stroke="rgb(236 232 223 / 0.3)"
						strokeWidth={1.2}
					/>
				))}
				{how.people.map((p) => (
					<path
						key={p.key}
						d={trail(p.xy)}
						fill="none"
						style={{ stroke: colour.get(p.key), opacity: hover === null || hover === p.key ? 0.55 : 0.12 }}
						strokeWidth={2}
						strokeLinecap="round"
					/>
				))}
				<path
					d={trail(how.puck.xy.map((q, k) => (how.puck.show[k]! > 0.3 ? q : null)))}
					fill="none"
					stroke="var(--paper)"
					strokeWidth={1.5}
					strokeDasharray="3 3"
				/>
			</g>
			{how.people.map((p) => {
				const q = p.xy[frame],
					d = q && place(q);
				if (!d) return null;
				const seen = p.seen[at(how, frame)],
					dim = hover !== null && hover !== p.key;
				return (
					// biome-ignore lint/a11y/noStaticElementInteractions: hover highlights only, for pointers
					<g
						key={p.key}
						transform={`translate(${d[0].toFixed(1)} ${d[1].toFixed(1)})`}
						className="pointer-events-auto cursor-pointer"
						style={{ opacity: dim ? 0.25 : 1 }}
						onPointerEnter={() => setHover(p.key)}
						onPointerLeave={() => setHover(null)}
						onClick={() => setHover(hover === p.key ? null : p.key)}
					>
						<circle
							r={hover === p.key ? 13 : 10}
							style={{
								fill: colour.get(p.key),
								stroke: "#000",
								strokeWidth: 1.5,
								strokeDasharray: seen ? undefined : "3 2",
								fillOpacity: seen ? 1 : 0.5,
							}}
						/>
						<text y={4.5} textAnchor="middle" className="fill-ink text-[11px] font-bold">
							{p.team >= 0 ? p.jersey : ""}
						</text>
						{hover === p.key && (
							<text
								y={-18}
								textAnchor="middle"
								className="text-[15px] font-semibold"
								style={{ fill: colour.get(p.key), paintOrder: "stroke", stroke: "#000c", strokeWidth: 4 }}
							>
								{label(p)}
							</text>
						)}
					</g>
				);
			})}
			{puckAt && <circle cx={puckAt[0]} cy={puckAt[1]} r={4} fill="var(--paper)" stroke="#000" strokeWidth={1.5} />}
			{u >= 1 && (
				<foreignObject x={12} y={12} width={120} height={40}>
					<button
						type="button"
						className="pointer-events-auto rounded-full bg-black/70 px-3 py-1 text-[15px] text-dim hover:text-paper"
						onClick={again}
					>
						Again
					</button>
				</foreignObject>
			)}
		</g>
	);
}

/** The puck's path so far through the moment, through the frame's camera, and a ring on it. */
function PuckPath({ how, camera, frame }: { how: How; camera: THREE.PerspectiveCamera; frame: number }) {
	const [a] = how.frames;
	let d = "",
		pen = false,
		last: XY | null = null;
	for (let f = a; f <= frame; f++) {
		const q = how.puck.xy[f],
			p = q && how.puck.show[f]! > 0.3 ? project(camera, q[0], q[1], 0.02) : null;
		if (!p) {
			pen = false;
			continue;
		}
		d += `${pen ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`;
		pen = true;
		last = p;
	}
	return (
		<g>
			<path
				d={d}
				fill="none"
				stroke="var(--live)"
				strokeWidth={2.5}
				strokeLinecap="round"
				strokeLinejoin="round"
				style={{ filter: "drop-shadow(0 0 3px #000)" }}
			/>
			{last && <circle cx={last[0]} cy={last[1]} r={9} fill="none" stroke="var(--live)" strokeWidth={2} />}
		</g>
	);
}

/**
 * Inset: the arena's atlas (stands above, ice below) filling in view by view, each texel from the first view that saw it clear of
 * players (first_*.png), and the view it is at.
 */
function AtlasFill({ how }: { how: How }) {
	const canvas = useRef<HTMLCanvasElement>(null);
	const [parts, setParts] = useState<{ img: HTMLImageElement; first: ImageData; mask: HTMLCanvasElement }[] | null>(null);
	const [u] = useRun("fill", FILL_MS, true);
	const n = Math.min(Math.floor(Math.min(u * 1.25, 1) ** 2.2 * how.atlas.views), how.atlas.views - 1); // slow at first, where the early views fill most

	useEffect(() => {
		let gone = false;
		const image = (src: string) =>
			new Promise<HTMLImageElement>((ok, fail) => {
				const im = new Image();
				im.onload = () => ok(im);
				im.onerror = fail;
				im.src = src;
			});
		const part = async (name: string) => {
			const [img, first] = await Promise.all([image(`${BASE}/atlas_${name}.webp`), image(`${BASE}/first_${name}.png`)]),
				c = document.createElement("canvas");
			c.width = first.width;
			c.height = first.height;
			const g = c.getContext("2d", { willReadFrequently: true })!;
			g.drawImage(first, 0, 0);
			return { img, first: g.getImageData(0, 0, c.width, c.height), mask: c };
		};
		Promise.all([part("out"), part("ice")]).then(
			(p) => !gone && setParts(p),
			() => {},
		);
		const views = Array.from({ length: how.atlas.views }, (_, n) => Object.assign(new Image(), { src: viewUrl(n) })); // a dozen kB each
		return () => {
			gone = true;
			for (const v of views) v.src = "";
		};
	}, [how]);

	useEffect(() => {
		const c = canvas.current;
		if (!c || !parts) return;
		const g = c.getContext("2d")!,
			w = c.width;
		g.clearRect(0, 0, w, c.height);
		let y = 0;
		for (const { img, first, mask } of parts) {
			const h = Math.round((w * first.height) / first.width),
				m = mask.getContext("2d")!,
				out = m.createImageData(first.width, first.height);
			for (let k = 0; k < first.data.length; k += 4) out.data[k + 3] = first.data[k]! <= n ? 255 : 0;
			m.putImageData(out, 0, 0);
			g.save();
			g.beginPath();
			g.rect(0, y, w, h);
			g.clip();
			g.drawImage(img, 0, y, w, h);
			g.globalCompositeOperation = "destination-in";
			g.imageSmoothingEnabled = true;
			g.drawImage(mask, 0, y, w, h);
			g.restore();
			y += h + 6;
		}
	}, [parts, n]);

	const height = Math.round((480 * 170) / 824) + Math.round((480 * 302) / 598) + 6;
	return (
		<div className="pointer-events-none absolute bottom-3 left-3 w-[36%] rounded-card bg-black/80 p-2 shadow-overlay">
			<canvas ref={canvas} width={480} height={height} className="block w-full" />
			<div className="mt-1.5 flex items-center gap-2">
				<img src={viewUrl(n)} alt="" className="aspect-video w-1/3 rounded-sm" />
				<span className="text-xs tabular-nums text-dim">
					view {n + 1}/{how.atlas.views}
				</span>
			</div>
		</div>
	);
}

/** Who is in the moment, to pick out in every layer. */
function Roster({
	how,
	frame,
	hover,
	setHover,
	colour,
}: {
	how: How;
	frame: number;
	hover: number | null;
	setHover: (k: number | null) => void;
	colour: Map<number, string>;
}) {
	const i = at(how, frame);
	return (
		<ul className="mt-4 flex flex-wrap gap-1.5" aria-label="Players">
			{how.people
				.filter((p) => p.seen.some(Boolean))
				.map((p) => (
					<li key={p.key}>
						<Chip
							active={hover === p.key}
							on={!!p.seen[i]}
							colour={colour.get(p.key)!}
							onPick={(on) => setHover(on ? p.key : null)}
						>
							{label(p)}
						</Chip>
					</li>
				))}
		</ul>
	);
}

function Chip({
	active,
	on,
	colour,
	onPick,
	children,
}: {
	active: boolean;
	on: boolean;
	colour: string;
	onPick: (on: boolean) => void;
	children: ReactNode;
}) {
	return (
		<button
			type="button"
			className={`flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors ${active ? "border-paper text-paper" : "border-white/10 text-dim hover:text-paper"} ${on ? "" : "opacity-50"}`}
			onPointerEnter={() => onPick(true)}
			onPointerLeave={() => onPick(false)}
			onFocus={() => onPick(true)}
			onBlur={() => onPick(false)}
		>
			<span className="size-2.5 rounded-full" style={{ background: colour }} />
			{children}
		</button>
	);
}

function PlayIcon() {
	return (
		<svg viewBox="0 0 16 16" className="size-3.5" aria-hidden>
			<path d="M4 2.5v11l9-5.5z" fill="currentColor" />
		</svg>
	);
}

function PauseIcon() {
	return (
		<svg viewBox="0 0 16 16" className="size-3.5" aria-hidden>
			<path d="M4 2.5h3v11H4zM9 2.5h3v11H9z" fill="currentColor" />
		</svg>
	);
}
