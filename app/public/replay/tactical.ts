// The Tactical view: a coach's board of the goal from above, on the ice of the minimal style (styles.ts; TACTICAL_STYLE), in the
// language of the ice graphics (ice/): rings and tags, trails, the pass, the carry and the shot, the zones and the shot's card, as
// the preset (ice/model.ts PRESETS; "tactics" to begin with) has them. Its camera frames the play from the pass to the net.
// The drawings are an overlay (userData.overlay, post.ts): drawn over the finished frame, hidden where the scene is nearer;
// the tags and the card are DOM in `overlay`, over the canvas. What hangs over the ice (the scoreboard) is hidden while it is built.
import * as THREE from "three";
import { buildIce, projectThrough } from "../ice/gl";
import { createLabels, type Keep } from "../ice/labels";
import { layersOf, only, type PresetName } from "../ice/model";
import { sceneElements } from "../ice/scene";
import type { GoalScene, Pose } from "./scene";
import type { Style } from "./styles";
import type { Anatomy, SceneData } from "./types";

export const TACTICAL_STYLE: Style = "minimal";

const LEAD = 1, // s of the play framed before the pass
	MARGIN = 2.5, // m round the play in the frame
	OVERHEAD = 5, // m up from which what is over the ice is hidden
	FOV = 30,
	TILT = 10, // degrees off straight down, from the near side
	ROOM = { x: 6, y: 3 }; // m kept in frame beside the release for the card, away from the net

const clampFrame = (S: SceneData, f: number) => Math.min(Math.max(Math.round(f), 0), S.frames - 1);
/** Index in S.people of whoever has meta.key `key`. */
const index = (S: SceneData, key: number) => S.people.findIndex((p) => p.key === key);
/** Where person j (an index) stood at exported frame f. */
const rootAt = (S: SceneData, j: number, f: number): [number, number] => {
	const r = S.people[j]!.root[Math.min(Math.max(f, 0), S.people[j]!.root.length - 1)]!;
	return [r[0] / 1000, r[1] / 1000];
};
/** The goal line's x the shot goes to: the end it was taken in. */
const goalX = (S: SceneData, a: Anatomy) => (Math.sign(a.from[0]) || 1) * S.rink.goal_x;

/** The first frame of the play: LEAD s before the pass, or before the scorer has the puck; the release without an anatomy. */
const playFrom = (S: SceneData, a: Anatomy | null) =>
	clampFrame(S, a ? (a.pass_frame ?? a.received ?? a.release) - LEAD * S.fps : S.frames / 2);
/** Whether the puck is seen between exported frames f and f + 1. */
const shown = (S: SceneData, f: number) => (S.puck.show[f] ?? 0) > 0.5 && (S.puck.show[f + 1] ?? 0) > 0.5;

/** The play's extent on the ice: the puck, the scorer and the passer from playFrom to the net, the net and room for the card. */
function region(S: SceneData, a: Anatomy | null) {
	const box = new THREE.Box2(),
		f0 = playFrom(S, a),
		f1 = clampFrame(S, a?.in_net ?? f0 + S.fps);
	for (let f = f0; f <= f1; f++) {
		if (shown(S, f)) box.expandByPoint(new THREE.Vector2(...S.puck.xy[f]!));
		for (const k of [a?.scorer, a?.passer])
			if (k != null && index(S, k) >= 0) box.expandByPoint(new THREE.Vector2(...rootAt(S, index(S, k), f)));
	}
	if (a) {
		const gx = goalX(S, a),
			[x, y] = a.from;
		box.expandByPoint(new THREE.Vector2(gx + Math.sign(gx) * 1.5, -1.5)).expandByPoint(new THREE.Vector2(gx, 1.5));
		box.expandByPoint(new THREE.Vector2(x - Math.sign(gx) * ROOM.x, y + (y >= 0 ? 1 : -1) * ROOM.y));
	}
	return box.expandByScalar(MARGIN);
}

/** A camera from above (TILT off straight down, from the near side) framing the play at `aspect` (width over height). */
export function tacticalPose(S: SceneData, aspect: number): Pose {
	const box = region(S, S.anatomy),
		c = box.getCenter(new THREE.Vector2()),
		e = box.getSize(new THREE.Vector2()).multiplyScalar(0.5),
		t = Math.tan(THREE.MathUtils.degToRad(FOV / 2)),
		d = Math.max(e.y / t, e.x / (aspect * t)) * 1.08, // the tilt's foreshortening
		tilt = THREE.MathUtils.degToRad(TILT),
		target = new THREE.Vector3(c.x, c.y, 0),
		position = new THREE.Vector3(c.x, c.y - d * Math.sin(tilt), d * Math.cos(tilt));
	const quaternion = new THREE.Quaternion().setFromRotationMatrix(
		new THREE.Matrix4().lookAt(position, target, new THREE.Vector3(0, 1, 0)),
	);
	return { position, quaternion, fov: FOV };
}

/**
 * The Tactical view's drawings for `scene` (its cards kept clear of `keep`): add `root` to it and `overlay` over the canvas (in a positioned container the canvas
 * fills), and `update(time, camera)` each frame (time in source frames, as setTime's; the camera the frame is drawn through).
 */
export function buildTactical(scene: GoalScene, keep: Keep) {
	const S = scene.S,
		ice = buildIce(),
		overlay = document.createElement("div"),
		labels = createLabels(overlay, keep),
		size = new THREE.Vector2();
	overlay.style.cssText = "position:absolute;inset:0;pointer-events:none;overflow:hidden;";
	let layers = layersOf("tactics");
	ice.group.userData.overlay = true;
	ice.group.userData.uniforms = ice.uniforms;
	// what hangs over the ice (the scoreboard) would hide the play from above
	const over: THREE.Object3D[] = [];
	scene.arena.traverse((o) => {
		if (o instanceof THREE.Mesh && o.visible && new THREE.Box3().setFromObject(o).min.z > OVERHEAD) over.push(o);
	});
	for (const o of over) o.visible = false;
	return {
		root: ice.group,
		overlay,
		anatomy: S.anatomy,
		/** Show what preset `name` shows. */
		setPreset(name: PresetName) {
			layers = layersOf(name);
		},
		/** Draw the moment as it is at source time `time`, through `camera`; person `hover` (meta.key) marked out. */
		update(time: number, camera: THREE.Camera, hover?: number) {
			const f = Math.min(Math.max(time - S.frame0, 0), S.frames - 1),
				elements = only(sceneElements(S, f, scene.hidden, hover), layers);
			size.set(Math.max(overlay.clientWidth, 1), Math.max(overlay.clientHeight, 1));
			ice.update(elements, camera, size, Math.min(devicePixelRatio, 2), S.rink);
			labels.place(elements, projectThrough(camera, size));
		},
		dispose() {
			ice.dispose();
			labels.dispose();
			overlay.remove();
			for (const o of over) o.visible = true;
		},
	};
}
