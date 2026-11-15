// A goal's 3D replay in a page element, in one of four modes (Mode):
//   replay    the cut sequence (S.cuts: parts of the planned shots back to back, consecutive in time; CUTS), cutting hard from one
//             part to the next, each part played as its own shot plays it; where the scene has none, the broadcast sequence
//             (BROADCAST): the director's approach shot, then the goal shot, gliding from one to the other over the end of their
//             overlap (at most GLIDE s of the moment), or cutting where they do not overlap
//   player    a player's head camera (S.heads, by role) over the whole moment, their body hidden and their stick drawn
//   tactical  the play from above (tactical.ts: its camera, trails, the pass and the shot, the card), always in its style
//   explore   the free camera (below), which can be put where any planned shot's camera is at that time (lookFrom)
// Whatever plays, plays on loop at `rate` (real time to begin with) over the whole
// reconstructed moment: before a shot the camera keeps its first framing and after it its last, carried along with the scorer, so
// the play leading up to the goal and what follows it are seen too. It spins up from frozen (SPIN s) and winds down to frozen at the
// end (over the last WIND s), as slow motion does, its sound faded out; it holds the moment's last frame HOLD s, then the camera glides
// (GLIDE s, eased) back to its start while the moment winds back. Changing mode or shot glides the same way; paused, it glides
// to where the new shot is at the moment that stands still, which stays still. Between the source's frames
// (25 a second) everyone's pose and the camera are interpolated, so playback is as smooth as the display.
//
// Motion blur (the "blur" look, post.ts): each frame is exposed over SHUTTER of a source frame (about 1/80 s, as a broadcast
// camera's shutter): the scene and the planned camera as the shutter opens and closes (Motion) give every pixel its smear, so what
// the camera follows stays sharp. While time stands still (paused, a glide, bullet time's frozen moment) a frame is sharp.
//
// The free camera plays the whole reconstructed moment (every source frame of the scene, not one shot's). It starts in an orbit
// around the scorer, ORBIT m off at the angle the camera had (between LIFT degrees above the ice), glides there, and follows him as
// the moment plays (until panned). It glides (damped) and stays where the scene holds up: above
// the ice (no lower than LOW, looking down at its target from at least FLAT degrees), inside the glass (WALL in from the boards) and
// under the roof (CEILING); dragged past a limit it is drawn in toward its target along its line of sight, so the view keeps its aim.
// With sound on, the broadcast's own sound of the moment (/site/audio/<id>.m4a, scripts/build-audio.ts) plays along at the moment's speed, on the
// audio thread (sound.ts): at a slower speed picked, a copy slowed to it offline with its pitch kept (<id>@<tempo>.m4a), as stretching it
// live loses more. That speed comes from the shot's pace, not from frame times, and the sound's own clock is kept in step by
// nudging its speed NUDGE either way when it is more than STEADY s off; it is only restarted where the moment is when more than
// JUMP s off (a scrub, a loop). A shot that eases the moment a little (a drone or POV shot at 0.8 or so) is heard at its speed
// with its pitch kept; one that slows it right down (slow motion, bullet time) goes the way films do, by FILM (from none at the
// first share of slowing to full at the second): muffled, lower (its pitch going down with the speed, to SLOWEST of it) and in a
// reverb; frozen, it keeps rumbling on at SLOWEST and comes back in step when time runs again. Paused, holding or gliding, it
// stops as a tape does (TAPE s, pitched down and fading) and comes back up with the moment's spin-up. A browser that holds sound
// until the page is used gets it from the first press on the page. It is heard from where the camera is (spatial.ts): the
// broadcast fuller near the stands, the skates and the puck placed where they are, moving round the listener as the camera does.
// The scene comes from /scenes/<id> (store/scenes, served by server.ts). Every frame is timed (perf.ts) when perf is on.
// Depth of field (post.ts): on in the drone and the POV (`dof` "auto"), every shot ("all") or none ("off"; ?dof=), never in the
// minimal and vector styles (LENS_STYLES): far behind the subject blurs DOF of the frame's width, at most MOST
// of it (POV_MOST in the POV, where the ice at the skater's feet should stay readable). Autofocus picks the subject: the drone
// and the net camera whoever has the puck (the nearest visible skater within REACH m of it, at chest height), else the puck; the puck
// camera the puck; the POV, of the puck, the net and the others, what is nearest the centre of view and over 1 m off; any other shot
// the scorer, else the puck; a head camera as the POV. In the cut sequence each part focuses as its shot does, afresh at each cut. The focus distance follows it critically damped in diopters
// (as a focus ring turns), over about AF s of playback (of the page's time while paused, so a still frame comes into focus).
// The style (post.ts, styles.ts; ?style=) draws the frames realistic, as pixel art, as a vector illustration or as the minimal view
// of the data; the tactical mode in its own (TACTICAL_STYLE) whatever is picked.
// How much is rendered is kept within a frame budget (quality.ts: "auto", or a fixed quality); nothing is posed or drawn while the
// replay is off screen or the page hidden. ?bench=1 measures a run of the moment (bench.ts); ?quality= picks the setting.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { projectThrough } from "../ice/gl";
import { createLabels } from "../ice/labels";
import { type Card, color, rgba } from "../ice/model";
import { skateCuts } from "./arena";
import { bench } from "./bench";
import { Outline } from "./outline";
import { clothWeave, Person } from "./people";
import { FRAME, Perf, type PerfReport } from "./perf";
import { motionOf, pickPerson } from "./pick";
import { broadcastLook, type Focus, type Look, type Motion, type Quality } from "./post";
import { Budget, SETTINGS, type Setting } from "./quality";
import { blendPose, cameraPose, GoalScene, type Pose } from "./scene";
import { Sound } from "./sound";
import { Space } from "./spatial";
import { STYLES, type Style } from "./styles";
import { buildTactical, TACTICAL_STYLE, tacticalPose } from "./tactical";

import type { Cam, Head, PersonMeta, Role, Shot } from "./types";
import { param, sdRink } from "./util";
import { Wire } from "./wire";

const GLIDE = 1.2,
	HOLD = 1,
	GRAB = 6, // px a press moves before it is a drag that takes the camera
	SPOT_EASE = 0.3, // s the spotlight eases on and off over
	ORBIT = 9,
	LIFT = [12, 40] as const,
	LOW = 0.4,
	FLAT = 6,
	WALL = 0.6,
	CEILING = 25,
	NEAR = 1.5,
	FAR = 45, // s; s; m; degrees; m; degrees above the horizontal; m in from the boards; m; m from the target, closest and farthest
	FOLLOW = 4, // 1/s: how quickly the orbit's centre catches up with the scorer
	AIM = 1, // m above the ice: the point of the scorer a camera following him before or after its shot keeps framed (his chest)
	COAST = 1, // s over which such a camera's own speed hands over to the scorer's
	INSET = 0.05, // m further in than a camera may go that a point it looks at is kept, so it is never on the limit itself
	PIXEL_RATIO = 1.5, // at most: a high-density display renders 1.5 px per CSS px, not 2 or 3
	REPORT = 500, // ms between perf reports
	LOOK = 0.25, // s ahead the people's frames are built
	REWIND = 16, // the people's frames a glide goes through
	DOF = 0.007,
	MOST = 0.018,
	POV_MOST = 0.012,
	REACH = 1.5,
	AF = 0.15,
	SPIN = 1.2, // s the moment takes to spin up from frozen to its speed
	WIND = 0.6, // s of the timeline over which it winds down to frozen at its end (evenly, as a skater stops: speed ∝ √distance left)
	STOP = 0.06, // of its speed it keeps at least while winding down, so it reaches the end
	SHUTTER = 0.3, // of a source frame
	STEADY = 0.06,
	NUDGE = 0.05, // the pitch-kept copy hides it
	JUMP = 0.25,
	FILM = [0.35, 0.7],
	SLOWEST = 0.3,
	EASE = 0.15, // share of the speed; s for the effect to settle
	TAPE = 0.35, // s a tape stop takes: the sound slowing to a quarter, pitched down, muffled and faded out
	OUTLINE_IN = 0.12,
	CARD_REVEAL = 0.4, // s the hovered person's outline and card take to come in
	CARD_UP = 2.1, // m up where the hovered person's card is pinned, over their helmet
	KEEP = { top: 64, bottom: 104 }; // px along the view's top and bottom under the page's controls, where no card goes

/** The broadcast sequence's name among the shots: APPROACH, then GOAL. */
export const BROADCAST = "broadcast";
/** The cut sequence's name among the shots: S.cuts's parts. */
export const CUTS = "cuts";

export type DofSetting = "off" | "auto" | "all";
export const DOF_SETTINGS: readonly DofSetting[] = ["off", "auto", "all"];
/** the styles with depth of field: minimal (the analysis view) and vector (flat graphics) stay sharp throughout */
const LENS_STYLES: readonly Style[] = ["realistic", "pixel"];
const AUTOFOCUS = ["drone", "pov", "puckcam", "goalcam"];
const APPROACH = "approach",
	GOAL = "goal";
/** The tactical mode's shot: tactical.ts's camera over the whole moment. */
const TACTICAL = "tactical";
/** A head camera's shot: HEAD and its player's role. */
const HEAD = "head:";
/** The lens a shot focuses as: a head camera's the POV's. */
const lensOf = (name: string) => (name.startsWith(HEAD) ? "pov" : name);

export type Mode = "replay" | "player" | "tactical" | "explore";
export const MODES: readonly Mode[] = ["replay", "player", "tactical", "explore"];
/** The planned shots the free camera can be put at (lookFrom), in this order. */
const ANGLES = ["drone", "bullettime", "puckcam", "goalcam", "celebrate", "goal", "replay", "goal_slow"];

/** A part of a shot's timeline: shot `shot`'s frames a to b (not b), from timeline frame `at`. */
type Part = { shot: string; a: number; b: number; at: number };

const UP = new THREE.Vector3(0, 0, 1),
	RIGHT = new THREE.Vector3(1, 0, 0);

/** Pose p, held in place, turned as an operator pans and tilts from point a to point b: about the vertical, then its own right axis. */
function pan(p: Pose, a: THREE.Vector3, b: THREE.Vector3) {
	const da = a.clone().sub(p.position),
		db = b.clone().sub(p.position),
		turn = Math.atan2(db.y, db.x) - Math.atan2(da.y, da.x),
		yaw = Math.atan2(Math.sin(turn), Math.cos(turn)),
		tilt = Math.atan2(db.z, Math.hypot(db.x, db.y)) - Math.atan2(da.z, Math.hypot(da.x, da.y));
	p.quaternion
		.premultiply(new THREE.Quaternion().setFromAxisAngle(UP, yaw))
		.multiply(new THREE.Quaternion().setFromAxisAngle(RIGHT, tilt));
}

const ease = (x: number) => {
	const t = Math.min(Math.max(x, 0), 1);
	return t * t * (3 - 2 * t);
};

/** How far the sound goes the way films do at a shot's pace: 0 eased a little (pace 0.65 and up), 1 slowed right down (0.3 and below). */
export const filmStrength = (pace: number) => ease((1 - Math.min(pace, 1) - FILM[0]!) / (FILM[1]! - FILM[0]!));

/** The speeds with a copy of the sound slowed offline (scripts/build-audio.ts's TEMPOS): those of the speed switch (Replay.tsx) below 1. */
const TEMPOS: readonly number[] = [0.5, 0.25];

export type ViewerOptions = { spray?: boolean; look?: readonly Look[]; progress?: (done: number, total: number) => void };
/**
 * A frame set from outside: the moment at source time `time`, moving `pace` source frames a frame there (0 frozen), seen from
 * `pose` as shot `seen` is seen (whose player it hides, whose lens it has: "free" for none); the camera at the shutter's opening and
 * closing for its motion blur.
 */
/** A frame set from outside (see `directed`): the moment's time and its pace, the camera, whose view it is, the camera as the
 * shutter opens and closes; asFilmed (0 to 1), how much the players are blurred as the broadcast's shutter caught them however fast time
 * runs (a frozen moment a photograph of the play, as the broadcast's own frame is), and at least by how far they move while the
 * shutter is open; lens,
 * focused `d` m ahead with `strength` of the depth of field's blur (whatever `dof` is), else the depth of field as `dof` has it. */
export type Directed = {
	time: number;
	pace: number;
	pose: Pose;
	seen: string;
	shutter?: [Pose, Pose];
	asFilmed?: number;
	lens?: { d: number; strength: number };
};

export class ReplayViewer {
	scene: GoalScene;
	renderer: THREE.WebGLRenderer;
	camera = new THREE.PerspectiveCamera(40, 1, 0.05, 600);
	controls: OrbitControls;
	perf: Perf;
	shot: string;
	mode: Mode = "replay";
	/** whose head camera the player mode has */
	role: Role = "scorer";
	/** the players with a head camera (S.heads) */
	heads: Head[];
	/** the shots made here for the modes, by name: the head cameras */
	private made = new Map<string, Shot>();
	/** the tactical mode's drawings, while it is on */
	private board: ReturnType<typeof buildTactical> | null = null;
	/** the tactical camera, for the aspect it was framed at */
	private above: { aspect: number; pose: Pose } | null = null;
	private running = true;
	/** whether the moment plays: set going again, it spins up from frozen over SPIN */
	get playing() {
		return this.running;
	}
	set playing(on: boolean) {
		if (on && !this.running) this.resumed = this.clock();
		this.running = on;
	}
	free = false;
	/** the free camera's centre keeps to the scorer */
	follow = true;
	/** the arena dimmed, a spotlight on the scorer (post.ts SPOT): 0 off to 1, eased to over SPOT_EASE */
	spotlight = 0;
	/** the scene's meshes as a wireframe over the band of the picture between these fractions of its width (wire.ts) */
	wire: readonly [number, number] = [0, 0];
	private wires: Wire | null = null;
	private lit = 0;
	/** playback speed, 1 real time */
	rate = 1;
	/** s a glide from one camera to another takes */
	glideFor = GLIDE;
	/** place in the shot's timeline (fractional frames): the source frames before the shot, its output frames, those after it */
	k = 0;
	/** the free camera's place in the moment: fractional source frame from the scene's first */
	f = 0;
	/** Each frame set from outside while not null (a film's take, scripts/film): drawn at that time from that camera, whatever the
	 * mode, the shot, the free camera and the playback (all left as they are) */
	directed: Directed | null = null;
	/** the time (ms) the moment plays and the camera glides by: the display's, or capture's own */
	private clock = () => performance.now();
	/** the shaders compiled, before the first frame is drawn */
	private compiled: Promise<void>;
	/** called every frame with the progress through the shot, or the moment in free mode (0 to 1) */
	onTick: (at: number) => void = () => {};
	/** called when panning lets the free camera's centre go */
	onFollow: (follow: boolean) => void = () => {};
	/** the mode changed from within (a grab, a person picked) */
	onMode: (mode: Mode, role: Role) => void = () => {};
	/** a person's own numbers over the whole game (value, label), for the card shown when they are under the pointer */
	statsOf: (meta: PersonMeta) => [string, string][] = () => [];
	/** called every REPORT ms while perf is on */
	onPerf: (report: PerfReport) => void = () => {};
	/** what is rendered at what quality and size */
	budget = new Budget();
	/** where the depth of field is on */
	dof: DofSetting = "auto";
	/** how the frame is drawn (styles.ts; ?style=) */
	style: Style = "realistic";
	/** the focus distance as it follows the subject: two stages, in diopters (1/m); null to start again at the subject */
	private pull: [number, number] | null = null;
	/** the shot seen this frame (a part's, in the cut sequence) and the one the lens last focused for */
	private seen = "";
	private focused = "";
	/** the last frame: the CPU's ms on it (all, posing, rendering) and what else it did (bench.ts reads it) */
	stats = { cpu: 0, pose: 0, render: 0, notes: [] as string[] };
	private post!: ReturnType<typeof broadcastLook>;
	/** the replay is in view (some of it on screen) */
	private shown = true;
	private visibility: IntersectionObserver;
	private resize: ResizeObserver;
	private last: number | null = null;
	private glide: { from: Pose; time: number; start: number } | null = null;
	/** when the moment last set going (ms: first played, a glide ended, play pressed), spun up from it over SPIN; NaN till first played */
	private resumed = NaN;
	/** the loop's own share of real time: up from frozen after resumed, down to frozen over the timeline's last WIND s */
	private spin = 1;
	/** s left to hold the last frame */
	private hold = HOLD;
	/** a press on the stage while the camera is not free, until it is a drag (grab) or ends */
	private press: PointerEvent | null = null;
	/** scene time of the last frame drawn */
	private time = 0;
	private frames = 0;
	private reported = 0;
	private wantPerf = false;
	private scorer: number;
	/** whom the free camera's orbit keeps to (index in scene.people): the scorer, or a person picked */
	private target: number;
	/** the pointer over the view (-1 to 1 each way) while no button is down, the person under it (index, -1 none) and since when */
	private pointer: THREE.Vector2 | null = null;
	private hovered = -1;
	private hoverSince = 0;
	/** the person a tap on a touch screen chose (meta.key), marked out as if under a pointer until the next tap */
	private chosen: number | null = null;
	private outline = new Outline();
	/** the hovered person's card, as DOM over the view */
	private cardLayer: HTMLDivElement;
	private cards: ReturnType<typeof createLabels>;
	private sound = false;
	private audio: Sound | null = null;
	private space: Space | null = null;
	/** the sound's speed nudged -1 (it is ahead), 0 or 1 (behind) */
	private nudge = 0;

	static async open(container: HTMLElement, id: string, { look = [], ...options }: ViewerOptions = {}) {
		const scene = await GoalScene.load(`/scenes/${encodeURIComponent(id)}`, options);
		return new ReplayViewer(container, id, scene, look);
	}

	private constructor(
		private container: HTMLElement,
		private id: string,
		scene: GoalScene,
		look: readonly Look[],
	) {
		this.scene = scene;
		const S = scene.S;
		this.heads = S.heads ?? [];
		for (const h of this.heads)
			this.made.set(HEAD + h.role, { frames: h.cams.map((_, i) => S.frame0 + i), cams: h.cams, own: h.key });
		this.shot = this.opening;
		const own = S.shots.pov?.own;
		this.scorer = Math.max(
			scene.people.findIndex((p) => p.meta.key === own),
			0,
		);
		this.target = this.scorer;
		// no multisampling here: the broadcast look draws the scene into its own multisampled target
		this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
		this.renderer.setPixelRatio(Math.min(devicePixelRatio, PIXEL_RATIO));
		this.renderer.domElement.className = "block size-full touch-none";
		container.append(this.renderer.domElement);
		this.renderer.info.autoReset = false; // a frame's draws add up over every pass, reset once a frame
		this.perf = new Perf(this.renderer.getContext() as WebGL2RenderingContext);
		scene.root.userData.perf = this.perf;
		this.camera.up.set(0, 0, 1);
		this.controls = new OrbitControls(this.camera, this.renderer.domElement);
		Object.assign(this.controls, {
			enabled: false,
			enableDamping: true,
			dampingFactor: 0.08,
			screenSpacePanning: true,
			minDistance: NEAR,
			maxDistance: FAR,
			maxPolarAngle: THREE.MathUtils.degToRad(90 - FLAT),
		});
		const dom = this.renderer.domElement;
		dom.addEventListener("pointerdown", (e) => {
			// OrbitControls pans with the right button, or the left with a modifier
			if (this.free && this.follow && (e.button === 2 || e.shiftKey || e.ctrlKey || e.metaKey)) this.setFollow(false);
			if (!this.free) this.press = e;
		});
		// a press on the replay or a player's eyes that becomes a drag takes the camera there (grab), the drag going on as the orbit's
		dom.addEventListener("pointermove", (e) => {
			const p = this.press;
			if (!p || e.pointerId !== p.pointerId || Math.hypot(e.clientX - p.clientX, e.clientY - p.clientY) < GRAB) return;
			this.press = null;
			this.grab();
			if (this.free) dom.dispatchEvent(new PointerEvent("pointerdown", p));
		});
		for (const end of ["pointerup", "pointercancel"]) dom.addEventListener(end, () => (this.press = null));
		// a mouse or pen over the view, no button down: who is under it is marked out (each frame, as the people move)
		dom.addEventListener("pointermove", (e) => {
			if (e.pointerType === "touch" || e.buttons) return void (this.pointer = null);
			const r = dom.getBoundingClientRect();
			(this.pointer ??= new THREE.Vector2()).set(
				((e.clientX - r.left) / r.width) * 2 - 1,
				-((e.clientY - r.top) / r.height) * 2 + 1,
			);
		});
		dom.addEventListener("pointerleave", () => (this.pointer = null));
		this.cardLayer = document.createElement("div");
		this.cardLayer.style.cssText = "position:absolute;inset:0;pointer-events:none;overflow:hidden;";
		container.append(this.cardLayer);
		this.cards = createLabels(this.cardLayer, KEEP);
		const { width: W, height: H } = this.size();
		this.renderer.setSize(W, H, false);
		const dof = param("dof") as DofSetting | null;
		if (dof && DOF_SETTINGS.includes(dof)) this.dof = dof;
		const style = param("style") as Style | null;
		if (style && STYLES.includes(style)) this.style = style;
		const asked = param("quality") as Setting | null;
		this.budget.set(asked && SETTINGS.includes(asked) ? asked : "auto");
		this.perf.measure = this.budget.setting === "auto";
		this.setLook(look);
		this.resize = new ResizeObserver(() => {
			const { width, height } = this.size();
			this.renderer.setSize(width, height, false);
			this.post.setSize(width, height);
		});
		this.resize.observe(container);
		this.visibility = new IntersectionObserver(([e]) => (this.shown = !!e?.isIntersecting));
		this.visibility.observe(container);
		// shaders compiled ahead, in parallel where the browser can, so the first frames do not stall on them
		this.compiled = this.renderer
			.compileAsync(this.scene.root, this.camera)
			.catch(() => {})
			// each frame timed by the page's clock (as everything else here is), not the display's frame timestamp: the same in a
			// browser, and in step with a page whose clock is driven frame by frame (a recording)
			.then(() => this.renderer.setAnimationLoop(() => this.frame(this.clock())));
		if (param("bench")) void bench(this.benchTarget());
		if (param("debug")) Object.assign(window, { replayViewer: this }); // for the console and headless checks
	}

	private size() {
		return { width: Math.max(this.container.clientWidth, 1), height: Math.max(this.container.clientHeight, 1) };
	}

	/** The replay mode's shot: the cut sequence, else the broadcast sequence, else the first planned shot. */
	private get opening() {
		const S = this.scene.S;
		return S.cuts?.length ? CUTS : S.shots[APPROACH] && S.shots[GOAL] ? BROADCAST : Object.keys(S.shots)[0]!;
	}

	/** The planned shots of ANGLES the scene has. */
	get angles() {
		return ANGLES.filter((name) => name in this.scene.S.shots);
	}

	/** The scene's planned shot `name`, or one made here. */
	private shotOf(name: string): Shot | undefined {
		return this.scene.S.shots[name] ?? this.made.get(name);
	}

	/** Player `key`'s head camera at every source frame, if the scene has it. */
	private headOf(key: number | null | undefined): Cam[] | undefined {
		return this.heads.find((h) => h.key === key)?.cams;
	}

	/** The style drawn: the one picked, the tactical mode's own while it is on. */
	private get drawnStyle() {
		return this.board ? TACTICAL_STYLE : this.style;
	}

	setLook(look: readonly Look[]) {
		this.post?.dispose();
		this.wires?.dispose();
		this.wires = null;
		const { width, height } = this.size();
		this.post = broadcastLook(this.renderer, this.scene.root, this.camera, look, width, height, this.perf, this.budget.quality);
		this.post.setScale(this.budget.scale);
		this.post.setStyle(this.drawnStyle, this.scene);
		this.budget.reset();
	}

	/** The close-up detail a quality draws: the cloth's weave at high only, the skates' cuts in the ice but at low. */
	private detail(quality: Quality) {
		clothWeave.value = quality === "high" ? 1 : 0;
		skateCuts.value = quality === "low" ? 0 : 1;
	}

	/** Render at a fixed quality, or keep within the frame budget ("auto"). */
	setQuality(setting: Setting) {
		this.budget.set(setting);
		this.perf.measure = setting === "auto";
		this.post.setQuality(this.budget.quality);
		this.detail(this.budget.quality);
		if (setting === "high") this.scene.sharpen();
		this.post.setScale(this.budget.scale);
	}

	private benchTarget() {
		return {
			id: this.id,
			renderer: this.renderer,
			stats: this.stats,
			setting: () => this.budget.setting,
			setQuality: (s: Setting) => this.setQuality(s),
			dof: () => this.dof,
			setDof: (d: DofSetting) => this.setDof(d),
			restart: () => {
				if (this.free) this.setFree(false);
				this.setBoard(false);
				this.mode = "replay";
				this.shot = this.opening;
				this.glide = null;
				this.k = 0;
				this.hold = HOLD;
				this.rate = 1;
				this.playing = true;
				this.resumed = this.clock();
			},
			describe: () => `${this.budget.quality} at ${this.budget.scale.toFixed(2)}`,
		};
	}

	setPerf(on: boolean) {
		this.wantPerf = on; // switched between frames, never inside one
	}

	/** Into mode `mode` (the player mode with `role`'s head camera), glided to as a change of shot. */
	setMode(mode: Mode, role = this.role) {
		if (mode === this.mode && role === this.role) return;
		this.mode = mode;
		this.role = role;
		this.target = this.scorer;
		this.chosen = null;
		this.setBoard(mode === "tactical");
		if (mode === "explore") this.setFree(true);
		else this.setShot(mode === "player" ? HEAD + role : mode === "tactical" ? TACTICAL : this.opening);
	}

	/** The tactical drawings on or off, and the style drawn with them. */
	private setBoard(on: boolean) {
		if (on === !!this.board) return;
		if (on) {
			this.board = buildTactical(this.scene, KEEP);
			this.scene.root.add(this.board.root);
			// under the cards: what the pointer is on shows over the board's own marks
			if (this.board.overlay) this.container.insertBefore(this.board.overlay, this.cardLayer);
		} else {
			this.board!.dispose();
			this.board = null;
		}
		this.post.setStyle(this.drawnStyle, this.scene);
	}

	/**
	 * The free camera glided to where planned shot `name`'s camera is at this time, looking as it looks, its orbit's centre on its
	 * line of sight as far off as the scorer (within the orbit's limits); its centre no longer following him.
	 */
	lookFrom(name: string) {
		const S = this.scene.S,
			x = THREE.MathUtils.clamp(this.along(name, S.frame0 + this.f), 0, this.span(name).length),
			{ position, quaternion, fov } = this.planned(name, x).pose;
		this.startGlide();
		this.hold = HOLD;
		this.orbitFrom(position, quaternion, fov);
	}

	/**
	 * Into the explore mode from where the camera is, as it is (a drag on the replay or a player's eyes takes the camera): the free
	 * camera there, its orbit's centre on its line of sight as far off as the scorer; its centre no longer following him.
	 */
	grab() {
		if (this.free || this.mode === "tactical") return;
		this.mode = "explore";
		this.free = true;
		this.controls.enabled = true;
		this.f = Math.min(Math.max(this.time - this.scene.S.frame0, 0), this.scene.S.frames - 1);
		this.orbitFrom(this.camera.position.clone(), this.camera.quaternion.clone(), this.camera.fov);
		this.onMode(this.mode, this.role);
	}

	/** The free camera's orbit from `position`, looking along `quaternion`: its centre on that line as far off as its target. */
	private orbitFrom(position: THREE.Vector3, quaternion: THREE.Quaternion, fov: number) {
		const ahead = new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion),
			reach = THREE.MathUtils.clamp(this.targetAt().setZ(AIM).sub(position).dot(ahead), NEAR, FAR);
		this.setFollow(false);
		this.camera.position.copy(position);
		this.camera.fov = fov;
		this.controls.target.copy(position).addScaledVector(ahead, reach);
		this.bound();
	}

	/** Shot `name` made outside the scene (as from another camera's track), to be set or cut to as a planned one. */
	addShot(name: string, shot: Shot) {
		this.made.set(name, shot);
		this.partsOf.delete(name);
	}

	/** Cut straight to point `at` (0 to 1) of shot `name`, out of the free camera if it is on, the moment already at its speed. */
	cut(name: string, at = 0) {
		this.setShot(name);
		this.glide = null;
		this.seek(at);
		this.resumed = this.clock() - SPIN * 1000;
	}

	/**
	 * Off the display, for offline capture: at fixed high quality, on a clock of its own that only `step(ms)` moves, which draws the
	 * frame `ms` on from the last; as smooth at any frame rate as rendering takes.
	 */
	async capture() {
		await this.compiled;
		this.renderer.setAnimationLoop(null);
		this.setQuality("high");
		let now = this.last ?? 0;
		this.clock = () => now;
		return (ms: number) => this.frame((now += ms));
	}

	/** Glide into the start of shot `name` (paused, into where it is at the moment), out of the free camera if it is on. */
	setShot(name: string) {
		this.startGlide();
		this.shot = name;
		this.k = this.playing ? 0 : this.reach(name, this.time);
		this.hold = HOLD;
		if (this.free) this.setFree(false);
		// the glide's first frames sent to be built now, before its first display frame
		this.scene.prepare([this.time, ...this.rewind(this.time, this.startTime()).slice(0, 3)]);
	}

	/** Where on shot `name`'s timeline it is at source time `time` (its start or end if outside it). */
	private reach(name: string, time: number) {
		let lo = 0,
			hi = this.span(name).length;
		for (let i = 0; i < 30; i++) {
			const m = (lo + hi) / 2;
			if (this.planned(name, m).time < time) lo = m;
			else hi = m;
		}
		return hi;
	}

	/** Jump to a point of the shot (of the moment, in free mode), 0 to 1. */
	seek(at: number) {
		if (this.free) this.f = at * (this.scene.S.frames - 1);
		else this.k = at * this.span().length;
		this.hold = HOLD;
	}

	setFollow(follow: boolean) {
		this.follow = follow;
		this.onFollow(follow);
	}

	/** Into the free camera: an orbit round its target (the scorer, or a person picked), glided to; or back to the planned shot. */
	setFree(free: boolean) {
		if (free === this.free) return;
		this.startGlide();
		this.free = free;
		this.controls.enabled = free;
		this.hold = HOLD;
		if (!free) return;
		this.f = Math.min(Math.max(this.time - this.scene.S.frame0, 0), this.scene.S.frames - 1);
		this.setFollow(true);
		const target = this.targetAt().setZ(1),
			from = this.camera.position.clone().sub(target),
			flat = Math.hypot(from.x, from.y) || 1;
		const lift = THREE.MathUtils.degToRad(
			THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(Math.atan2(from.z, flat)), LIFT[0], LIFT[1]),
		);
		const p = target
			.clone()
			.add(
				new THREE.Vector3((from.x / flat) * Math.cos(lift), (from.y / flat) * Math.cos(lift), Math.sin(lift)).multiplyScalar(
					ORBIT,
				),
			);
		this.controls.target.copy(target);
		this.camera.position.copy(p);
		this.camera.lookAt(target);
		this.bound();
	}

	/** The lens's focus this frame, the camera posed; `dt` s of playback since the last. Null where the depth of field is off. */
	private focus(dt: number): Focus | null {
		const name = this.directed?.seen ?? (this.free ? "free" : this.seen),
			lens = lensOf(name);
		if (name !== this.focused) this.pull = null; // a cut: the new camera's lens is already on its subject
		this.focused = name;
		const told = this.directed?.lens;
		if (told && LENS_STYLES.includes(this.drawnStyle))
			return told.strength > 0 ? { d: told.d, blur: DOF * told.strength, most: MOST * told.strength } : null;
		if (this.dof === "off" || !LENS_STYLES.includes(this.drawnStyle) || (this.dof === "auto" && !AUTOFOCUS.includes(lens)))
			return (this.pull = null);
		const eye = new THREE.Vector3().setFromMatrixPosition(this.camera.matrixWorld),
			ahead = this.camera.getWorldDirection(new THREE.Vector3()),
			r = 1 / Math.max(this.subject(name, eye, ahead).sub(eye).dot(ahead), 0.5),
			a = 1 - Math.exp(-dt / AF),
			p = (this.pull ??= [r, r]);
		p[0] += a * (r - p[0]);
		p[1] += a * (p[0] - p[1]);
		return { d: 1 / p[1], blur: DOF, most: lens === "pov" ? POV_MOST : MOST };
	}

	/** What the lens of shot `name` focuses on (see the top of this file), from eye looking ahead. */
	private subject(name: string, eye: THREE.Vector3, ahead: THREE.Vector3) {
		const { S, people, puck: disc } = this.scene,
			puck = new THREE.Vector3(disc.position.x, disc.position.y, 0.05),
			own = this.shotOf(name)?.own,
			others = people.filter((p) => p.mesh.visible && p.meta.key !== own).map((p) => p.root()),
			lens = lensOf(name);
		if (lens === "puckcam") return puck;
		if (lens === "drone" || lens === "goalcam") {
			const near = others.reduce<[number, number] | null>(
				(m, o) => (!m || Math.hypot(o[0] - puck.x, o[1] - puck.y) < Math.hypot(m[0] - puck.x, m[1] - puck.y) ? o : m),
				null,
			);
			return near && Math.hypot(near[0] - puck.x, near[1] - puck.y) < REACH ? new THREE.Vector3(near[0], near[1], 1.0) : puck;
		}
		if (lens === "pov") {
			const shot = S.puck.shot,
				[nx] = S.puck.xy[shot ? Math.min(Math.max(shot.in_net, 0), S.frames - 1) : 0]!,
				net = new THREE.Vector3(Math.sign(nx) * (S.rink.hx - 4), 0, 0.6),
				off = (v: THREE.Vector3) => {
					const r = v.clone().sub(eye),
						d = r.length();
					return d < 1 ? Infinity : Math.acos(Math.min(r.dot(ahead) / d, 1));
				};
			return [puck, net, ...others.map(([x, y]) => new THREE.Vector3(x, y, 1.2))].reduce((m, v) => (off(v) < off(m) ? v : m));
		}
		const scorer = S.shots.pov?.own !== undefined && people[this.scorer];
		return scorer ? new THREE.Vector3(...scorer.root(), 1.0) : puck;
	}

	/** Draw the frames realistic, as pixel art, as a vector illustration or as the minimal view of the data. */
	setStyle(style: Style) {
		this.style = style;
		this.post.setStyle(this.drawnStyle, this.scene);
	}

	/** Depth of field in the drone and the POV, every shot or none. */
	setDof(dof: DofSetting) {
		this.dof = dof;
	}

	/** The scorer's mean ground position as last posed. */
	private scorerAt() {
		const [x, y] = this.scene.people[this.scorer]?.root() ?? [0, 0];
		return new THREE.Vector3(x, y, 0);
	}

	/** The free camera's target's mean ground position as last posed. */
	private targetAt() {
		const [x, y] = this.scene.people[this.target]?.root() ?? [0, 0];
		return new THREE.Vector3(x, y, 0);
	}

	/** Whether person p can be picked: a player or goalie on the ice, shown, not the one whose eyes the view is. */
	private pickable = (p: Person) => p.meta.team >= 0 && !this.scene.hidden.has(p.meta.key);

	/** The meta.key of whoever is at client point (x, y) in the view, or null. */
	private pickAt(x: number, y: number): number | null {
		const r = this.renderer.domElement.getBoundingClientRect(),
			ndc = new THREE.Vector2(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1),
			j = pickPerson(this.scene.people, this.camera, ndc, this.pickable);
		return j < 0 ? null : this.scene.people[j]!.meta.key;
	}

	/**
	 * A click or a tap at client point (x, y): onto the person there (focusOn), but a first tap on a touch screen (`touch`) only
	 * marks them out, as a pointer over them would. False if it did nothing: no one there, and no one marked out to let go of.
	 */
	choose(x: number, y: number, touch: boolean): boolean {
		const key = this.pickAt(x, y),
			had = this.chosen;
		this.chosen = touch && key !== had ? key : null;
		if (key !== null && this.chosen === null) this.focusOn(key);
		return key !== null || had !== null;
	}

	/**
	 * Onto person `key`: through their own eyes if they have a head camera (but in the free camera); else, or in the free camera, the
	 * free camera's orbit round them, following them. The owner is told the mode (onMode).
	 */
	focusOn(key: number) {
		const j = this.scene.people.findIndex((p) => p.meta.key === key),
			head = this.heads.find((h) => h.key === key);
		if (j < 0) return;
		if (head && this.mode !== "explore") this.setMode("player", head.role);
		else {
			if (this.mode !== "explore") this.setMode("explore");
			this.target = j;
			this.setFollow(true);
		}
		this.onMode(this.mode, this.role);
	}

	/** The hovered person's card: who they are, their speed in this moment, their numbers over the game. */
	private hoverCard(p: Person, f: number, now: number): Card {
		const m = p.meta,
			{ now: speed, top } = motionOf(this.scene.S, p, f),
			[x, y] = p.root();
		return {
			kind: "card",
			key: `hover${m.key}`,
			x,
			y,
			z: CARD_UP,
			anchor: "above",
			team: m.team as 0 | 1,
			kicker: `${m.goalie ? "Goalie" : "Skater"}${m.jersey ? ` · #${m.jersey}` : ""}`,
			title: m.name?.split(" ").at(-1) ?? "Unidentified",
			stats: [
				...(m.goalie
					? []
					: ([
							[speed === null ? "-" : `${Math.round(speed)}`, "km/h"],
							[`${Math.round(top)}`, "top km/h"],
						] as [string, string][])),
				...this.statsOf(m),
			].slice(0, 3),
			p: (now - this.hoverSince) / 1000 / CARD_REVEAL,
		};
	}

	private pose(): Pose {
		return { position: this.camera.position.clone(), quaternion: this.camera.quaternion.clone(), fov: this.camera.fov };
	}

	/** The source frames a glide from source time `from` to `to` blends the people between, rounded: REWIND + 1 of them. */
	private rewind(from: number, to: number) {
		return Array.from({ length: REWIND + 1 }, (_, k) => Math.round(from + (k * (to - from)) / REWIND));
	}

	/** The source time the shot (or the free camera) starts at. */
	private startTime() {
		return this.free ? this.scene.S.frame0 : this.planned(this.shot, 0).time;
	}

	private startGlide() {
		this.glide = { from: this.pose(), time: this.time, start: this.last ?? this.clock() };
	}

	/**
	 * Shot `name`'s timeline: `lead` source frames before it, its `n` output frames (its parts'), then the rest of the moment;
	 * `length` frames. The broadcast sequence's and the tactical camera's are the moment's source frames.
	 */
	private span(name = this.shot) {
		const S = this.scene.S;
		if (name === BROADCAST || name === TACTICAL) return { lead: 0, n: S.frames, length: S.frames - 1 };
		const parts = this.parts(name),
			last = parts.at(-1)!,
			lead = parts[0]!.at,
			n = last.at + last.b - last.a - lead,
			tail = Math.max(S.frame0 + S.frames - 1 - this.shotOf(last.shot)!.frames[last.b - 1]!, 0);
		return { lead, n, length: lead + n - 1 + tail };
	}

	private partsOf = new Map<string, Part[]>();

	/** Shot `name`'s parts: the cut sequence's, or a shot's frames whole. */
	private parts(name: string) {
		let parts = this.partsOf.get(name);
		if (!parts) {
			const S = this.scene.S,
				cuts = name === CUTS ? S.cuts! : [{ shot: name, frames: [0, this.shotOf(name)!.frames.length] as const }];
			let at = Math.max(this.shotOf(cuts[0]!.shot)!.frames[cuts[0]!.frames[0]]! - S.frame0, 0);
			parts = cuts.map(({ shot, frames: [a, b] }) => {
				const part = { shot, a, b, at };
				at += b - a;
				return part;
			});
			this.partsOf.set(name, parts);
		}
		return parts;
	}

	/** Where shot `name`'s timeline is at source frame t. */
	private along(name: string, t: number) {
		const fr = this.shotOf(name)!.frames,
			{ lead, n } = this.span(name);
		if (t <= fr[0]!) return lead + t - fr[0]!;
		if (t >= fr[n - 1]!) return lead + n - 1 + t - fr[n - 1]!;
		let i = 0;
		while (fr[i + 1]! <= t) i++;
		return lead + i + (t - fr[i]!) / (fr[i + 1]! - fr[i]! || 1);
	}

	// Shot `name` at x on its timeline: scene time, camera (both interpolated between the frames around x) and the shot seen (whose
	// people it hides, whose lens it has); before and after the shot, its first or last camera flying on (or, before, back) at its
	// own speed (its climb or descent easing off), handed over to the scorer's over COAST, held at the scene's limits where it meets
	// them (keepIn, frame by frame) and turned to frame him as it did; a POV's camera its player's head camera (headOf).
	// A part runs on into its shot's next frame until the next part cuts in. The broadcast
	// sequence: the approach and the goal shot at that time, blended from one to the other over the end of their overlap. The tactical
	// camera: tactical.ts's, still.
	private planned(name: string, x: number): { time: number; pose: Pose; hides: string } {
		const S = this.scene.S;
		if (name === TACTICAL) return { time: S.frame0 + x, pose: this.overhead(), hides: TACTICAL };
		if (name === BROADCAST) {
			const time = S.frame0 + x,
				a = this.planned(APPROACH, this.along(APPROACH, time)),
				g = this.planned(GOAL, this.along(GOAL, time)),
				from = S.shots[GOAL]!.frames[0]!,
				to = S.shots[APPROACH]!.frames.at(-1)!,
				glide = Math.min(GLIDE * S.fps, Math.max(to - from, 0)),
				w = glide > 0 ? ease((time - (to - glide)) / glide) : time >= from ? 1 : 0;
			return { time, pose: blendPose(a.pose, g.pose, w), hides: w < 0.5 ? APPROACH : GOAL };
		}
		const parts = this.parts(name),
			first = parts[0]!,
			last = parts.at(-1)!,
			end = last.at + last.b - last.a - 1;
		if (x < first.at || x > end) {
			const [part, edge, over] = x < first.at ? [first, first.a, x - first.at] : [last, last.b - 1, x - end],
				sh = this.shotOf(part.shot)!,
				from = sh.frames[edge]!,
				time = from + over,
				head = this.headOf(sh.own);
			if (head) {
				const k = Math.min(Math.max(time - S.frame0, 0), head.length - 1),
					i = Math.floor(k);
				return {
					time,
					pose: blendPose(cameraPose(head[i]!, S.height), cameraPose(head[Math.min(i + 1, head.length - 1)]!, S.height), k - i),
					hides: part.shot,
				};
			}
			const pose = cameraPose(sh.cams[edge]!, S.height),
				scorer = this.scene.people[this.scorer];
			if (scorer) {
				// its own velocity (m per source frame) at that end, handed over to the scorer's over COAST
				const near = edge === 0 ? 1 : edge - 1,
					c0 = sh.cams[edge]!.C,
					c1 = sh.cams[near]!.C,
					dt = Math.abs(sh.frames[edge]! - sh.frames[near]!),
					v = new THREE.Vector3(c0[0]! - c1[0]!, c0[1]! - c1[1]!, c0[2]! - c1[2]!).multiplyScalar(
						dt > 0 ? (edge === 0 ? -1 : 1) / dt : 0,
					),
					// where time stands still at that end (bullet time's frozen ends) its orbit step is no speed through the moment: it
					// follows the scorer from the start
					coast = dt > 0 ? COAST * S.fps : 1e-6,
					root = (f: number) => scorer.rootAt(f - S.frame0),
					chest = (f: number) => new THREE.Vector3(...root(f), AIM),
					// its limits as the shot left it (no nearer the glass than WALL, no lower than LOW, no nearer the scorer's chest than
					// NEAR, or than it already was), each easing out to its own over COAST
					edgeIn = Math.min(WALL, -sdRink(S.rink, c0[0]!, c0[1]!)),
					edgeLow = Math.min(LOW, c0[2]!),
					edgeNear = Math.min(NEAR, chest(from).distanceTo(pose.position)),
					step = Math.sign(over),
					place = pose.position.clone(),
					carried = pose.position.clone();
				for (let u = 0; u < Math.abs(over); u++) {
					const d = Math.min(Math.abs(over) - u, 1),
						[xa, ya] = root(from + step * u),
						[xb, yb] = root(from + step * (u + d)),
						w = Math.exp(-(u + d / 2) / coast),
						dx = xb - xa,
						dy = yb - ya;
					place.add(new THREE.Vector3(v.x * d * step * w + dx * (1 - w), v.y * d * step * w + dy * (1 - w), v.z * d * step * w));
					carried.add(new THREE.Vector3(dx, dy, 0));
					const eased = Math.exp(-(u + d) / (COAST * S.fps)),
						margin = WALL + (edgeIn - WALL) * eased,
						low = LOW + (edgeLow - LOW) * eased,
						near = NEAR + (edgeNear - NEAR) * eased,
						him = chest(from + step * (u + d));
					this.keepIn(place, margin, low);
					const off = place.clone().sub(him);
					if (off.length() < near) this.keepIn(place.copy(him).add(off.setLength(near)), margin, low);
				}
				const [x1, y1] = root(time),
					target = this.inward(new THREE.Vector3(x1, y1, AIM));
				pose.position.copy(place);
				pan(pose, target.clone().add(place).sub(carried), target);
			}
			return { time, pose, hides: part.shot };
		}
		const part = parts.findLast((p) => p.at <= x)!,
			sh = this.shotOf(part.shot)!,
			k = part.a + x - part.at,
			i = Math.floor(k),
			j = Math.min(i + 1, sh.frames.length - 1),
			a = k - i;
		return {
			time: sh.frames[i]! + (sh.frames[j]! - sh.frames[i]!) * a,
			pose: blendPose(cameraPose(sh.cams[i]!, S.height), cameraPose(sh.cams[j]!, S.height), a),
			hides: part.shot,
		};
	}

	/** The tactical camera, framed for the stage's aspect. */
	private overhead() {
		const { width, height } = this.size(),
			aspect = width / height;
		if (this.above?.aspect !== aspect) this.above = { aspect, pose: tacticalPose(this.scene.S, aspect) };
		return this.above.pose;
	}

	private setPose(p: Pose) {
		const { width, height } = this.size();
		this.camera.position.copy(p.position);
		this.camera.quaternion.copy(p.quaternion);
		this.camera.fov = p.fov;
		this.camera.aspect = width / height;
		this.camera.updateProjectionMatrix();
	}

	private frame(now: number): void {
		const start = performance.now(),
			built = Person.built,
			programs = this.renderer.info.programs?.length ?? 0,
			stats = this.stats;
		stats.notes.length = 0;
		this.perf.enabled = this.wantPerf;
		if (!this.wantPerf && this.reported) {
			this.perf.reset();
			this.reported = 0;
		}
		const S = this.scene.S;
		const elapsed = this.last === null ? 0 : Math.max(0, now - this.last) / 1000,
			was = this.free ? this.f : this.k;
		this.last = now;
		if (!this.shown || document.hidden) {
			// out of view: nothing posed, drawn or heard, and the budget measures afresh when it is back
			if (this.audio?.playing) void this.audio.stop();
			this.space?.quiet();
			this.budget.reset();
			return;
		}
		if (this.glide && now - this.glide.start >= this.glideFor * 1000) {
			this.resumed = this.glide.start + this.glideFor * 1000;
			this.glide = null;
		}
		const end = this.free ? S.frames - 1 : this.span().length;
		if (this.playing && !this.glide && !this.directed) {
			// from frozen (the start, a glide back, a pause) it spins up, and it winds down to frozen at the end, the people and the
			// sound (hear) with it
			if (Number.isNaN(this.resumed)) this.resumed = now;
			const left = (end - (this.free ? this.f : this.k)) / (WIND * S.fps);
			this.spin = ease((now - this.resumed) / 1000 / SPIN) * (left >= 1 ? 1 : Math.max(Math.sqrt(Math.max(left, 0)), STOP));
			const at = Math.min((this.free ? this.f : this.k) + elapsed * S.fps * this.rate * this.spin, end);
			if (this.free) this.f = at;
			else this.k = at;
			if (at >= end && (this.hold -= elapsed) <= 0) {
				// held long enough: wind back to the start
				this.startGlide();
				this.hold = HOLD;
				if (this.free) this.f = 0;
				else this.k = 0;
			}
		}
		const a = this.glide ? ease((now - this.glide.start) / 1000 / this.glideFor) : 1;
		this.scene.root.userData.frame = ++this.frames;
		this.renderer.info.reset();
		let drawn: Pose;
		this.perf.span("pose", () => {
			const d = this.directed;
			if (d) {
				this.time = d.time;
				this.seen = d.seen;
				this.scene.useShot(this.shotOf(d.seen));
				this.scene.setTime(d.time, d.pace, [d.time + LOOK * S.fps * d.pace]);
				drawn = d.pose;
				this.setPose(drawn);
				return;
			}
			const plan = this.planned(this.shot, this.k),
				to = this.free ? S.frame0 + this.f : plan.time;
			this.scene.useShot(this.shotOf(plan.hides));
			this.seen = plan.hides;
			if (this.free) this.scene.useShot(undefined);
			// gliding, time moves on the camera's eased curve, the people blended between REWIND frames on the way (each built
			// ahead), not built at every frame between (too many); holding at the end, the first of the loop's glide built ahead
			const steps = this.glide
					? this.rewind(this.glide.time, to)
					: (this.free ? this.f : this.k) >= end
						? this.rewind(to, this.startTime()).slice(0, 3)
						: [],
				s = Math.min(Math.floor(a * REWIND), REWIND - 1),
				keys: [number, number] | undefined = this.glide ? [steps[s]!, steps[s + 1]!] : undefined;
			this.time = keys ? keys[0] + (keys[1] - keys[0]) * (a * REWIND - s) : to;
			// where time is LOOK s on (round to the start past the end), for the people's frames to be built by then
			const look = (this.free ? this.f : this.k) + LOOK * S.fps * this.rate,
				next = this.free ? S.frame0 + (look > end ? 0 : look) : this.planned(this.shot, look > end ? 0 : look).time;
			this.scene.setTime(this.time, this.glide ? 1 : this.pace() * this.spin, [next, ...steps], keys);
			if (!this.free) {
				drawn = this.glide ? blendPose(this.glide.from, plan.pose, a) : plan.pose;
				this.setPose(drawn);
				return;
			}
			if (this.follow) {
				// the orbit's centre eases after the scorer and the camera moves with it
				const target = this.controls.target,
					step = this.targetAt()
						.setZ(1)
						.sub(target)
						.multiplyScalar(1 - Math.exp(-FOLLOW * elapsed));
				target.add(step);
				this.camera.position.add(step);
			}
			this.bound();
			// gliding in, the camera already sits on the orbit (where the controls work); the frame is drawn from the way there
			drawn = this.glide ? blendPose(this.glide.from, this.pose(), a) : this.pose();
		});
		const posed = performance.now();
		this.onTick((this.free ? this.f : this.k) / Math.max(end, 1));
		this.hear((this.free ? this.f : this.k) !== was);
		this.perf.span("render", () => {
			const orbit = this.free && this.glide ? this.pose() : null;
			this.setPose(drawn);
			const hovered = this.scene.people[this.hovered];
			this.board?.update(this.time, this.camera, hovered?.meta.key);
			if (this.sound)
				this.space?.update(
					this.camera,
					this.time - this.scene.S.frame0,
					this.rate * (this.glide ? 1 : this.pace() * this.spin),
					this.playing && !this.glide,
				);
			this.perf.gpuBegin(FRAME);
			this.lit += (this.spotlight - this.lit) * (1 - Math.exp(-elapsed / SPOT_EASE));
			if (Math.abs(this.lit - this.spotlight) < 1e-3) this.lit = this.spotlight;
			this.post.spot.strength = this.lit;
			if (this.lit > 0) this.post.spot.at.copy(this.scorerAt());
			this.post.render(
				this.post.blur ? (this.directed ? this.directedMotion(this.directed) : this.motion(end)) : null,
				this.focus(this.playing ? elapsed * this.rate : elapsed),
				this.time,
			); // paused, the lens still settles on what is seen
			if (this.wire[1] > this.wire[0]) (this.wires ??= new Wire(this.renderer)).draw(this.scene.root, this.camera, this.wire);
			this.markHovered(now);
			this.perf.gpuEnd();
			if (orbit) this.setPose(orbit);
		});
		const done = performance.now();
		stats.cpu = done - start;
		stats.pose = posed - start;
		stats.render = done - posed;
		if (Person.built > built) stats.notes.push(`built ${Person.built - built}`);
		if ((this.renderer.info.programs?.length ?? 0) > programs) stats.notes.push("compiled");
		this.perf.frame(elapsed * 1000);
		const change = this.budget.frame(elapsed * 1000, stats.cpu, this.perf.gpuFrame);
		this.perf.gpuFrame = null;
		if (change) {
			stats.notes.push(`to ${change.quality} at ${change.scale.toFixed(2)}`);
			this.post.setQuality(change.quality);
			this.detail(change.quality);
			this.post.setScale(change.scale);
		}
		if (this.perf.enabled && now - this.reported > REPORT) {
			this.reported = now;
			const { calls, triangles } = this.renderer.info.render,
				size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
			this.onPerf(this.perf.report({ calls, triangles }, [size.x, size.y]));
		}
	}

	// How the frame moves over the shutter (see the top of this file); null while time stands still.
	private motion(end: number): Motion | null {
		if (!this.playing || this.glide) return null;
		if (this.free) {
			const vp = this.viewProjection(this.pose());
			return { open: vp, close: vp, span: SHUTTER };
		}
		const at = (x: number) => this.planned(this.shot, Math.min(Math.max(x, 0), end)),
			open = at(this.k - SHUTTER / 2),
			close = at(this.k + SHUTTER / 2),
			span = close.time - open.time;
		if (span <= 0) return null;
		// across a cut the camera does not move: the frame's own, still
		const [o, c] = open.hides === close.hides ? [open.pose, close.pose] : [at(this.k).pose, at(this.k).pose];
		return { open: this.viewProjection(o), close: this.viewProjection(c), span };
	}

	// A directed frame's: the camera from its pose at the shutter's opening to its closing, the moment over SHUTTER at its pace (or,
	// as filmed, over SHUTTER of a source frame).
	private directedMotion(d: Directed): Motion | null {
		const span = Math.max(d.pace, d.asFilmed ?? 0) * SHUTTER;
		if (!d.shutter && span <= 0) return null;
		const [o, c] = d.shutter ?? [d.pose, d.pose];
		return { open: this.viewProjection(o), close: this.viewProjection(c), span };
	}

	private view = new THREE.PerspectiveCamera();

	/** Projection times view of a camera at pose p, framed as the camera is. */
	private viewProjection(p: Pose) {
		const c = this.view;
		c.position.copy(p.position);
		c.quaternion.copy(p.quaternion);
		c.fov = p.fov;
		c.aspect = this.camera.aspect;
		c.near = this.camera.near;
		c.far = this.camera.far;
		c.updateProjectionMatrix();
		c.updateMatrixWorld();
		return c.projectionMatrix.clone().multiply(c.matrixWorldInverse);
	}

	/** Point t, where it is nearer the boards than `margin`, moved straight in to `margin` inside them (the nearest such point). */
	private inset(t: THREE.Vector3, margin: number) {
		const { hx, hy, corner } = this.scene.S.rink,
			cx = THREE.MathUtils.clamp(t.x, -(hx - corner), hx - corner),
			cy = THREE.MathUtils.clamp(t.y, -(hy - corner), hy - corner),
			r = Math.hypot(t.x - cx, t.y - cy),
			k = r > corner - margin ? (corner - margin) / r : 1;
		return t.set(cx + (t.x - cx) * k, cy + (t.y - cy) * k, t.z);
	}

	/** A point the camera looks at: inside the camera's limits by INSET more than WALL, its height between the ice and 3 m. */
	private inward(t: THREE.Vector3) {
		this.inset(t, WALL + INSET);
		return t.setZ(THREE.MathUtils.clamp(t.z, 0, 3));
	}

	/** Camera place p kept within its limits by the shortest move: `margin` in from the glass, out of the scoreboard by its nearest side,
	 * between `low` and CEILING. A step into a limit keeps only its part along it, so a camera meeting the glass stops there. */
	private keepIn(p: THREE.Vector3, margin = WALL, low = LOW) {
		this.inset(p, margin);
		p.z = THREE.MathUtils.clamp(p.z, low, CEILING);
		const box = this.scene.keepOut;
		if (box?.containsPoint(p)) {
			const out = [p.x - box.min.x, box.max.x - p.x, p.y - box.min.y, box.max.y - p.y, p.z - box.min.z, box.max.z - p.z],
				i = out.indexOf(Math.min(...out));
			if (i === 0) p.x = box.min.x;
			else if (i === 1) p.x = box.max.x;
			else if (i === 2) p.y = box.min.y;
			else if (i === 3) p.y = box.max.y;
			else if (i === 4) p.z = box.min.z;
			else p.z = box.max.z;
		}
		return p;
	}

	/** Camera place p, looking at t, drawn in toward t along its line of sight as far as it must to keep inside the glass (WALL in
	 * from the boards), out of the scoreboard and under CEILING; no lower than LOW. */
	private within(t: THREE.Vector3, p: THREE.Vector3) {
		const rink = this.scene.S.rink,
			keepOut = this.scene.keepOut,
			out = (q: THREE.Vector3) => q.z > CEILING || sdRink(rink, q.x, q.y) > -WALL || !!keepOut?.containsPoint(q);
		p.z = Math.max(p.z, LOW);
		if (out(p)) {
			let lo = 0,
				hi = 1; // the share of the way from the target to the camera that is still allowed
			const q = new THREE.Vector3();
			for (let i = 0; i < 20; i++) {
				const m = (lo + hi) / 2;
				if (out(q.lerpVectors(t, p, m))) hi = m;
				else lo = m;
			}
			p.lerpVectors(t, p, lo);
			p.z = Math.max(p.z, LOW);
		}
		return p;
	}

	// The free camera's limits (see the top of this file).
	private bound() {
		const { controls, camera } = this;
		this.inward(controls.target);
		controls.update();
		this.within(controls.target, camera.position);
		camera.lookAt(controls.target);
		camera.updateMatrixWorld();
	}

	setSound(on: boolean) {
		this.sound = on;
		if (on && !this.audio) {
			this.audio = new Sound(`/site/audio/${encodeURIComponent(this.id)}`, TEMPOS);
			this.space = new Space(this.audio, this.scene);
		}
		if (!on) {
			void this.audio?.stop();
			this.space?.quiet();
		}
	}

	// The broadcast's sound kept to the moment; `moved`: the timeline playing moved on this frame (see the top of this file). Only
	// what changes reaches the sound (sound.ts), scheduled on its own clock.
	private hear(moved: boolean) {
		const sound = this.audio;
		if (!sound || !this.sound) return;
		if (!this.playing || this.glide || !moved) {
			// not cut but stopped as a tape does (paused, a shot changed, the loop's hold and glide back)
			void sound.tapeStop(TAPE);
			sound.note(null, "stopped");
			return;
		}
		const S = this.scene.S,
			end = this.free ? S.frames - 1 : this.span().length;
		if (this.spin < 1 && end - (this.free ? this.f : this.k) < WIND * S.fps) {
			// the picture winding down to frozen at the end: the sound is not slowed with it (stretched that far it breaks up), it
			// plays on at its speed and fades out as the picture stops (speed ∝ √distance left: WIND s of timeline take twice that)
			void sound.fadeOut(2 * WIND);
			sound.note(null, "ending");
			return;
		}
		const tempo = TEMPOS.includes(this.rate) ? this.rate : 1,
			m = (this.time - S.frame0) / S.fps, // s of the moment shown
			pace = this.pace() * this.spin,
			slow = 1 - Math.min(pace, 1), // how much the shot (and the loop's spin) slows the moment: 0 none, 1 frozen
			speed = this.rate * Math.max(pace, SLOWEST), // s of the moment heard a second
			heard = sound.heard(),
			off = heard === null ? null : heard - m;
		// ahead: a little slower until back in step; behind: a little faster; frozen, it rumbles on and is put back as time runs
		if (slow < 0.9 && off !== null)
			this.nudge = Math.abs(off) < STEADY / 3 ? 0 : off > STEADY ? -1 : off < -STEADY ? 1 : this.nudge;
		else this.nudge = 0;
		const rate = (speed / tempo) * (1 + NUDGE * this.nudge);
		if (!sound.playing || sound.tempo !== tempo || (slow < 0.9 && off !== null && Math.abs(off) > JUMP)) {
			void sound.start(m, speed, rate, tempo);
			this.stats.notes.push("sound");
		} else void sound.setRate(rate);
		sound.effect(filmStrength(pace), EASE);
		sound.note(off, "playing");
	}

	/** Source frames the moment moves for each frame of the timeline playing, where it is: 1 but in a slowed or frozen shot. */
	private pace() {
		if (this.free) return 1;
		const end = this.span().length,
			x = Math.min(Math.max(this.k, 0.25), end - 0.25);
		return Math.max(this.planned(this.shot, x + 0.25).time - this.planned(this.shot, x - 0.25).time, 0) / 0.5;
	}

	/**
	 * Who is under the pointer (or was chosen with a tap), marked out over the frame just drawn: their outline in their team's colour and their card over
	 * their head (none while a film's take is drawn or the camera is being dragged).
	 */
	private markHovered(now: number) {
		const j = this.directed
			? -1
			: this.pointer
				? pickPerson(this.scene.people, this.camera, this.pointer, this.pickable)
				: this.scene.people.findIndex((p) => p.meta.key === this.chosen && this.pickable(p));
		if (j !== this.hovered) {
			this.hovered = j;
			this.hoverSince = now;
			this.renderer.domElement.style.cursor = j >= 0 ? "pointer" : "";
		}
		const p = this.scene.people[j],
			size = { x: Math.max(this.cardLayer.clientWidth, 1), y: Math.max(this.cardLayer.clientHeight, 1) };
		if (!p) return this.cards.place([], projectThrough(this.camera, new THREE.Vector2(size.x, size.y)));
		const strength = Math.min((now - this.hoverSince) / 1000 / OUTLINE_IN, 1);
		this.outline.draw(
			this.renderer,
			this.scene.root,
			this.camera,
			[p.mesh, ...p.gear.map((g) => g.group)],
			rgba(color(p.meta.team), 1),
			strength,
		);
		const f = Math.min(Math.max(this.time - this.scene.S.frame0, 0), this.scene.S.frames - 1);
		this.cards.place([this.hoverCard(p, f, now)], projectThrough(this.camera, new THREE.Vector2(size.x, size.y)));
	}

	dispose() {
		this.outline.dispose();
		this.cards.dispose();
		this.cardLayer.remove();
		this.space?.dispose();
		this.audio?.dispose();
		this.renderer.setAnimationLoop(null);
		this.resize.disconnect();
		this.visibility.disconnect();
		this.controls.dispose();
		this.board?.dispose();
		this.post.dispose();
		this.scene.root.traverse((o) => {
			if (o instanceof THREE.Mesh || o instanceof THREE.Points) {
				o.geometry.dispose();
				for (const m of [o.material].flat()) m.dispose();
			}
		});
		this.renderer.dispose();
		this.renderer.forceContextLoss();
		this.renderer.domElement.remove();
	}
}
