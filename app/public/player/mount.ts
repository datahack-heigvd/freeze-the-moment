import { PRESETS, type PresetName } from "../ice/model";
import { liveEdge, syncClock } from "../lib/clock";
import { createAR, type Index } from "./ar";
import { createBanner } from "./banner";
import {
	COLOR,
	eventDetails,
	eventTitle,
	type Game,
	type GameEvent,
	markerHTML,
	periodAt,
	periodClock,
	playerHTML,
	TEAM_COLOR,
	TEAMS,
} from "./events";
import { createMomentum } from "./momentum";

export type Scoreboard = { home: number; away: number; period: number };

export type PlayerOptions = {
	game: Game;
	index: Index;
	/** Live follows the broadcast edge and cannot seek past it; otherwise the whole recording is rewatchable. */
	live: boolean;
	/** Video second to open at; live opens at the edge when this is left out. */
	startAt?: number;
	onScore: (score: Scoreboard) => void;
};

let active: { replay(t: number): void } | undefined;
/** Jump the mounted player to video second t, leaving the live edge if it is following it. */
export const watchFrom = (t: number) => active?.replay(t);

const svgIcon = (d: string, filled = false) =>
	`<svg viewBox="0 0 16 16" ${filled ? 'fill="currentColor"' : 'fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"'}>${d}</svg>`;
const ICON = {
	play: svgIcon('<path d="M5 3.5v9l7.5-4.5z"/>', true),
	pause: svgIcon(
		'<rect x="4" y="3.5" width="2.6" height="9" rx=".6"/><rect x="9.4" y="3.5" width="2.6" height="9" rx=".6"/>',
		true,
	),
	muted: svgIcon('<path d="M2.5 6h2.5l3.5-3v10L5 10H2.5z"/><path d="M11 6l3 4M14 6l-3 4"/>'),
	sound: svgIcon('<path d="M2.5 6h2.5l3.5-3v10L5 10H2.5z"/><path d="M11 5.5a3.5 3.5 0 0 1 0 5M12.8 3.8a6 6 0 0 1 0 8.4"/>'),
	enterFullscreen: svgIcon('<path d="M3 6V3h3M10 3h3v3M13 10v3h-3M6 13H3v-3"/>'),
	exitFullscreen: svgIcon('<path d="M6 3v3H3M13 6h-3V3M10 13v-3h3M3 10h3v3"/>'),
};

const pad = (n: number) => String(Math.floor(n)).padStart(2, "0");
const hms = (s: number) => `${pad(s / 3600)}:${pad((s % 3600) / 60)}:${pad(s % 60)}`;

// Matches the native range thumb travel (12px wide) so dots line up with the markers.
const thumbLeft = (f: number) => `calc(6px + (100% - 12px) * ${f})`;

/** Wires the stage markup of components/Stage.tsx to the video, the AR overlay and the timeline; returns the teardown. */
export function mountPlayer(root: HTMLElement, { game, index, live, startAt, onScore }: PlayerOptions): () => void {
	const $ = <T extends HTMLElement>(id: string) => root.querySelector<T>(`#${id}`)!;
	const abort = new AbortController();
	const { signal } = abort;
	const v = $<HTMLVideoElement>("v");
	const stage = $("stage");
	const scrub = $<HTMLInputElement>("scrub");
	const status = $<HTMLButtonElement>("status");
	const playBtn = $("play");
	const muteBtn = $("mute");
	const fullscreenBtn = $("fullscreen");
	// the scoreboard goes fullscreen with the stage, so the score stays in view
	const stageFrame = root.closest<HTMLElement>("[data-stage-frame]")!;
	const markers = $("markers");
	const tip = $("tip");
	const dot = $("dot");
	const fill = $("fill");
	const presets = $("presets");
	const legend = $("legend");
	const arLabels = $("ar-labels");
	const momentum = $("momentum") as unknown as SVGSVGElement;

	let following = live;
	let scrubbing = false;

	const edge = () =>
		live ? Math.min(liveEdge(), v.duration || Number.POSITIVE_INFINITY) : v.duration || Number.POSITIVE_INFINITY;

	function seek(t: number) {
		v.currentTime = Math.max(0, Math.min(t, edge()));
	}

	function goLive() {
		following = true;
		seek(edge());
		v.play().catch(() => {});
	}

	function replay(t: number) {
		following = false;
		seek(t);
		v.play().catch(() => {});
	}
	active = { replay };

	/** The one Live control: red and inert at the live edge, otherwise how far behind you are and a way back. */
	let statusKey = "";
	function renderStatus(atLive: boolean, lag: number) {
		const key = atLive ? "live" : hms(lag).replace(/^00:/, "").replace(/^0/, "");
		if (key === statusKey) return;
		statusKey = key;
		status.classList.toggle("bg-live", atLive);
		status.classList.toggle("text-white", atLive);
		status.classList.toggle("cursor-default", atLive);
		status.setAttribute("aria-label", atLive ? "Watching live" : `${key} behind live, go live`);
		status.innerHTML = atLive
			? `<i class="size-1.5 animate-pulse rounded-full bg-white motion-reduce:animate-none"></i>Live`
			: `<span class="tabular-nums text-dim">-${key}</span>Go live`;
	}
	if (!live) {
		status.innerHTML = "Replay";
		status.disabled = true;
		status.classList.add("cursor-default");
	}

	function tick() {
		if (!v.duration) return;
		const e = edge();
		if (live) {
			if (v.currentTime > e) seek(e);
			if (following && Math.abs(e - v.currentTime) > 3) seek(e);
			renderStatus(following && e - v.currentTime < 3, e - v.currentTime);
		}
		scrub.max = String(e);
		if (!scrubbing) scrub.value = String(v.currentTime);
		placeMarkers(e);
		dot.style.left = thumbLeft(Number(scrub.value) / e);
		fill.style.width = thumbLeft(Number(scrub.value) / e);
		arJump.hidden = ar.segments.some((s) => v.currentTime >= s.t0 - 10 && v.currentTime < s.t1);
	}

	const togglePlay = () => (v.paused ? v.play() : v.pause());
	const syncPlay = () => {
		playBtn.innerHTML = v.paused ? ICON.play : ICON.pause;
		playBtn.setAttribute("aria-label", v.paused ? "Play" : "Pause");
	};
	const syncMute = () => {
		muteBtn.innerHTML = v.muted ? ICON.muted : ICON.sound;
		muteBtn.setAttribute("aria-label", v.muted ? "Unmute" : "Mute");
	};
	v.addEventListener("play", syncPlay, { signal });
	v.addEventListener("pause", syncPlay, { signal });
	v.addEventListener("volumechange", syncMute, { signal });
	syncPlay();
	syncMute();

	const toggleFullscreen = () =>
		(document.fullscreenElement ? document.exitFullscreen() : stageFrame.requestFullscreen()).catch(() => {});
	const syncFullscreen = () => {
		const on = document.fullscreenElement === stageFrame;
		fullscreenBtn.innerHTML = on ? ICON.exitFullscreen : ICON.enterFullscreen;
		fullscreenBtn.setAttribute("aria-label", on ? "Exit full screen" : "Full screen");
	};
	fullscreenBtn.hidden = !document.fullscreenEnabled;
	document.addEventListener("fullscreenchange", syncFullscreen, { signal });
	syncFullscreen();

	scrub.addEventListener(
		"pointerdown",
		() => {
			scrubbing = true;
		},
		{ signal },
	);
	scrub.addEventListener(
		"input",
		() => {
			following = false;
			seek(Number(scrub.value));
			tick();
		},
		{ signal },
	);
	scrub.addEventListener(
		"change",
		() => {
			scrubbing = false;
			following = live && edge() - v.currentTime < 3;
		},
		{ signal },
	);
	playBtn.onclick = togglePlay;
	// a click on a player or a shot of the overlay keeps its card up; elsewhere it pauses or plays
	v.onclick = (e) => {
		if (!ar.press(e.clientX, e.clientY)) void togglePlay();
	};
	status.onclick = () => live && goLive();
	// muted only while the browser asks for it: with sound once the visitor has used the page (a browser lets it play then), from
	// their first press if they came straight here, unless they chose with the button
	let chosen = false;
	const unmute = (e?: Event) => {
		if (chosen || !v.muted || (e?.target instanceof Node && muteBtn.contains(e.target))) return;
		v.muted = false;
		syncMute();
		if (!v.paused) return;
		v.play().catch((e: unknown) => {
			// refused with sound (not merely cut short by a seek): muted, as autoplay allows
			if (!(e instanceof DOMException && e.name === "NotAllowedError")) return;
			v.muted = true;
			syncMute();
			v.play().catch(() => {});
		});
	};
	if (navigator.userActivation?.hasBeenActive) unmute();
	for (const type of ["pointerdown", "keydown"]) document.addEventListener(type, unmute, { capture: true, signal });
	muteBtn.onclick = () => {
		chosen = true;
		v.muted = !v.muted;
	};
	fullscreenBtn.onclick = toggleFullscreen;
	v.ondblclick = toggleFullscreen;
	document.addEventListener(
		"keydown",
		(e) => {
			const target = e.target;
			if (e.code === "KeyF" && !(target instanceof Element && target.closest("select, textarea, input:not(#scrub)")))
				toggleFullscreen();
			if (target instanceof Element && target.closest("a, button, select, textarea, input:not(#scrub)")) return;
			if (e.code === "Space") {
				e.preventDefault();
				togglePlay();
			}
			if (target === scrub) return;
			if (e.code === "ArrowLeft") {
				following = false;
				seek(v.currentTime - 10);
			}
			if (e.code === "ArrowRight") seek(v.currentTime + 10);
		},
		{ signal },
	);
	v.addEventListener(
		"pause",
		() => {
			following = false;
		},
		{ signal },
	);

	const drawMomentum = createMomentum(momentum, game);
	const updateBanner = createBanner($("banner"), game);

	const ar = createAR($<HTMLCanvasElement>("ar"), arLabels, v, index, game);
	const arJump = $("ar-jump");
	arJump.onclick = () => replay(ar.start - 6);

	const PRESET_KEY = "overlay-preset";
	function choosePreset(name: PresetName) {
		ar.preset(name);
		for (const b of presets.children) b.setAttribute("aria-pressed", String((b as HTMLElement).dataset.preset === name));
		try {
			localStorage.setItem(PRESET_KEY, name);
		} catch {}
	}
	for (const [name, label] of PRESETS) {
		const btn = document.createElement("button");
		btn.dataset.preset = name;
		btn.textContent = label;
		btn.onclick = () => choosePreset(name);
		presets.append(btn);
	}
	let saved: string | null = null;
	try {
		saved = localStorage.getItem(PRESET_KEY);
	} catch {}
	choosePreset(PRESETS.find(([n]) => n === saved)?.[0] ?? "players");

	function showTip(ev: GameEvent, el: HTMLElement) {
		tip.innerHTML = `<i class="absolute -bottom-1 size-2.5 -translate-x-1/2 rotate-45 bg-[#101114]"></i>
			<div class="card relative whitespace-nowrap px-3 py-2 text-xs">
				<p class="flex items-center gap-1.5 font-semibold"><i class="size-2 rounded-full" style="background:${TEAM_COLOR[ev.team]}"></i>${eventTitle(ev)}<span class="font-normal text-dim">${TEAMS[ev.team]}</span></p>
				<p class="mt-1 text-sm font-semibold">${playerHTML(ev.player)}</p>
				<p class="mt-0.5 text-dim">${[periodClock(ev), ...eventDetails(ev)].join(" · ")}</p>
			</div>`;
		tip.classList.remove("hidden");
		const s = stage.getBoundingClientRect();
		const m = el.getBoundingClientRect();
		const cx = m.left + m.width / 2 - s.left;
		const half = tip.offsetWidth / 2;
		const left = Math.min(Math.max(cx, half + 8), s.width - half - 8);
		tip.style.left = `${left}px`;
		tip.style.top = `${m.top - s.top}px`;
		tip.style.setProperty("--dx", `${cx - left}px`);
	}

	const markerEls = game.events.map((ev) => {
		const el = document.createElement("button");
		el.className =
			"absolute bottom-0 flex h-full w-2.5 -translate-x-1/2 cursor-pointer items-end justify-center transition-transform hover:scale-125";
		el.setAttribute("aria-label", `${eventTitle(ev)}, ${TEAMS[ev.team]}, ${periodClock(ev)}`);
		el.innerHTML = markerHTML(ev, TEAM_COLOR[ev.team]);
		el.onmouseenter = el.onfocus = () => showTip(ev, el);
		el.onmouseleave = el.onblur = () => tip.classList.add("hidden");
		el.onclick = () => replay(ev.t - 8);
		markers.append(el);
		return el;
	});

	const goals = game.events.filter((ev) => ev.kind === "goal");
	let lastScore = "";

	function updateScore(t: number) {
		const done = goals.filter((g) => g.t <= t);
		const home = done.filter((g) => g.team === "Home").length;
		const period = periodAt(game, t);
		const key = `${home}:${done.length - home}:${period}`;
		if (key === lastScore) return;
		lastScore = key;
		onScore({ home, away: done.length - home, period });
	}

	function placeMarkers(e: number) {
		drawMomentum(e);
		game.events.forEach((ev, i) => {
			const el = markerEls[i]!;
			el.hidden = ev.t > e;
			el.style.left = thumbLeft(ev.t / e);
		});
	}

	legend.innerHTML = [
		...(["goal", "penalty", "shot"] as const).map(
			(kind) =>
				`<li class="flex items-center gap-1.5"><span class="flex h-4 w-2.5 items-end justify-center">${markerHTML({ kind, result: "Saved" }, COLOR.paper)}</span>${kind[0]!.toUpperCase() + kind.slice(1)}</li>`,
		),
		`<li class="flex items-center gap-1.5"><i class="size-2 rounded-full bg-home"></i><i class="-ml-1 size-2 rounded-full bg-away"></i>Colour is the team</li>`,
	].join("");

	let raf = 0;
	(function frame() {
		if (v.duration) {
			updateScore(v.currentTime);
			updateBanner(v.currentTime);
		}
		raf = requestAnimationFrame(frame);
	})();

	let idleTimer: ReturnType<typeof setTimeout>;
	function wake() {
		stage.classList.remove("idle");
		clearTimeout(idleTimer);
		idleTimer = setTimeout(() => {
			if (!v.paused && !scrubbing) stage.classList.add("idle");
		}, 2500);
	}
	for (const type of ["pointermove", "pointerdown", "keydown"]) stage.addEventListener(type, wake, { signal });
	document.addEventListener("keydown", wake, { signal });
	v.addEventListener("pause", wake, { signal });
	wake();

	v.addEventListener(
		"loadedmetadata",
		async () => {
			if (live) await syncClock();
			if (startAt === undefined) goLive();
			else replay(startAt);
		},
		{ signal },
	);
	v.src = "/video";

	const tickTimer = setInterval(tick, 200);
	const clockTimer = live ? setInterval(syncClock, 60000) : undefined;

	return () => {
		abort.abort();
		cancelAnimationFrame(raf);
		clearTimeout(idleTimer);
		clearInterval(tickTimer);
		clearInterval(clockTimer);
		ar.destroy();
		if (document.fullscreenElement === stageFrame) document.exitFullscreen().catch(() => {});
		v.pause();
		v.removeAttribute("src");
		v.load();
		for (const el of [markers, presets, legend, arLabels, momentum, $("banner")]) el.replaceChildren();
		if (active?.replay === replay) active = undefined;
	};
}
