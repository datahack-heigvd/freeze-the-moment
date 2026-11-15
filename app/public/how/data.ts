// The "How it works" page's data for one goal (pipeline/how_export.py): the moment's frames undistorted into one pinhole camera
// each, so that the rink, the outlines, the puck and the 3D scene line up with the picture through a plain three.js camera.
import * as THREE from "three";
import { blendPose, cameraPose, type Pose } from "../replay/scene";
import type { Vec3 } from "../replay/types";

export const GOAL = "musa";
export const BASE = `/site/how/${GOAL}`;
export const W = 960,
	H = 540;

/** A frame's camera: focal length in page pixels, rows of R (rink to OpenCV camera axes), centre (m). */
export type HowCam = { f: number; R: [Vec3, Vec3, Vec3]; C: Vec3 };
export type XY = [number, number];

/** xy: rink position (m) at every scene frame; seen, outline (page px, flat) and iou over the moment's frames only. */
export type HowPerson = {
	key: number;
	team: number;
	jersey: string | null;
	name: string | null;
	goalie: boolean;
	xy: (XY | null)[];
	seen: boolean[];
	outline: (number[] | null)[];
	iou: (number | null)[];
};

export type How = {
	fps: number;
	/** the moment's first and last scene frame */
	frames: [number, number];
	cams: HowCam[];
	rink: { hx: number; hy: number; corner: number; goal_x: number; lines: XY[][] };
	people: HowPerson[];
	puck: { xy: (XY | null)[]; show: number[]; shot: { release: number; in_net: number; rest: number; scorer: number } };
	/** views: atlas views in all, the first moment_from of them before the moment; texel sizes (m) of first_*.png */
	atlas: { views: number; moment_from: number; ice_res: number; out_res: number; perim: number };
};

// the end zones' face-off circles (IIHF: 6 m out from the goal line, 7 m either side, 4.5 m across) and goal creases (1.83 m), which
// rink_lines leaves out; the circles checked against those painted on the ice in the moment's frames
const FACEOFF = { out: 6, side: 7, r: 4.5 },
	CREASE = 1.83;

const arc = (cx: number, cy: number, r: number, from: number, to: number, n = 64): XY[] =>
	Array.from({ length: n + 1 }, (_, k) => {
		const t = from + ((to - from) * k) / n;
		return [cx + r * Math.cos(t), cy + r * Math.sin(t)];
	});

function markings({ goal_x }: How["rink"]): XY[][] {
	const out: XY[][] = [];
	for (const end of [-1, 1]) {
		const x = end * (goal_x - FACEOFF.out);
		for (const side of [-1, 1]) out.push(arc(x, side * FACEOFF.side, FACEOFF.r, 0, 2 * Math.PI));
		const toward = end < 0 ? 0 : Math.PI; // the crease bulges toward centre ice
		out.push(arc(end * goal_x, 0, CREASE, toward - Math.PI / 2, toward + Math.PI / 2, 24));
	}
	return out;
}

let how: Promise<How> | null = null;
export const loadHow = () =>
	(how ??= fetch(`${BASE}/how.json`)
		.then((r) => r.json() as Promise<How>)
		.then((h) => {
			h.rink.lines.push(...markings(h.rink));
			return h;
		}));

/** Each fitted body's 2D keypoints (OpenPose 18, page px, flat) per frame of the moment by person key, and the limbs between them. */
export type PoseData = {
	limbs: [number, number][];
	colours: [number, number, number][];
	people: Record<string, (number[] | null)[]>;
};
/** The stick outlines found in each frame of the moment (page px, flat). */
export type SticksData = number[][][];

const once = <T>(file: string) => {
	let got: Promise<T> | null = null;
	return () => (got ??= fetch(`${BASE}/${file}`).then((r) => r.json() as Promise<T>));
};
export const loadPose = once<PoseData>("pose.json");
export const loadSticks = once<SticksData>("sticks.json");

export const frameUrl = (i: number) => `${BASE}/frames/${i}.webp`;
export const viewUrl = (n: number) => `${BASE}/views/${n}.webp`;

export const poseOf = (c: HowCam): Pose => cameraPose({ fx: c.f, R: c.R, C: c.C }, H);
export { blendPose };

/** A camera at `pose` for the page's frame. */
export function cameraAt(pose: Pose, camera = new THREE.PerspectiveCamera(pose.fov, W / H, 0.1, 600)) {
	camera.fov = pose.fov;
	camera.aspect = W / H;
	camera.position.copy(pose.position);
	camera.quaternion.copy(pose.quaternion);
	camera.updateProjectionMatrix();
	camera.updateMatrixWorld();
	return camera;
}

const v = new THREE.Vector3();
/** Rink point (m) to page pixels through camera, or null behind it. */
export function project(camera: THREE.Camera, x: number, y: number, z = 0): XY | null {
	v.set(x, y, z).applyMatrix4(camera.matrixWorldInverse);
	if (v.z > -0.1) return null; // behind the camera
	v.applyMatrix4(camera.projectionMatrix);
	return [((v.x + 1) / 2) * W, ((1 - v.y) / 2) * H];
}

/** Polylines on the rink as SVG path data through camera, broken where they go behind it. */
export function linesPath(camera: THREE.Camera, lines: XY[][]) {
	let d = "";
	for (const line of lines) {
		let pen = false;
		for (const [x, y] of line) {
			const p = project(camera, x, y);
			if (!p) {
				pen = false;
				continue;
			}
			d += `${pen ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`;
			pen = true;
		}
	}
	return d;
}

/** Where the frame's edge meets the ice through camera (rink m), as far as `far` m out: the camera's footprint. */
export function footprint(camera: THREE.PerspectiveCamera, far = 70): XY[] {
	const o = camera.position,
		out: XY[] = [],
		edge: XY[] = [];
	for (let k = 0; k <= 8; k++) edge.push([-1 + k / 4, -1]);
	for (let k = 0; k <= 8; k++) edge.push([1, -1 + k / 4]);
	for (let k = 0; k <= 8; k++) edge.push([1 - k / 4, 1]);
	for (let k = 0; k <= 8; k++) edge.push([-1, 1 - k / 4]);
	for (const [x, y] of edge) {
		const d = new THREE.Vector3(x, y, 0.5).unproject(camera).sub(o).normalize(),
			t = d.z < -1e-3 ? Math.min(-o.z / d.z, far) : far;
		out.push([o.x + d.x * t, o.y + d.y * t]);
	}
	return out;
}

/** The rink's outline (m), as rink_lines draws it: its first line. */
export const outline = (h: How) => h.rink.lines[0]!;

/** Index into the moment's per-frame lists for scene frame i. */
export const at = (h: How, i: number) => i - h.frames[0];

/** Each person's colour: their team's hue, a shade of it each; the officials grey. */
export function colours(h: How, teamColour: (team: number) => string) {
	const out = new Map<number, string>();
	for (const team of [0, 1, -1]) {
		const members = h.people.filter((p) => p.team === team);
		members.forEach((p, k) => {
			const shade = members.length > 1 ? k / (members.length - 1) : 0.5;
			out.set(
				p.key,
				team < 0
					? `hsl(0 0% ${62 + 18 * shade}%)`
					: `color-mix(in oklab, ${teamColour(team)} ${100 - 45 * Math.abs(shade - 0.35)}%, ${shade > 0.35 ? "white" : "black"})`,
			);
		});
	}
	return out;
}

/** A person's short label: number and last name, or what they are. */
export const label = (p: HowPerson) =>
	p.team < 0 ? "Official" : `${p.jersey ?? ""} ${p.name?.split(" ").slice(-1)[0] ?? ""}`.trim();
