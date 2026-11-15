// A goal frozen at the shot's release and seen from a grid of cameras round the scorer and the puck, for build-orbits.ts to pack
// into a card's hover strip: across, swinging round the play; down, from higher to lower. They sit DISTANCE m out (pulled in for all
// of them alike where one would be past the boards), centred on the goal shot's side of the play and LIFT° up. Full broadcast look;
// no motion blur, since time stands still. Each view comes with its depth, as the card reprojects the views between them: the scene
// drawn again into a depth texture, then as nearness (1/z from NEAREST m, 1, to FARTHEST, 0) in grey.
import * as THREE from "three";
import { FullScreenQuad } from "three/addons/postprocessing/Pass.js";
import { Perf } from "../../public/replay/perf";
import { broadcastLook, LOOKS } from "../../public/replay/post";
import { cameraPose, GoalScene } from "../../public/replay/scene";
import { sdRink } from "../../public/replay/util";

const FOV = 30,
	DISTANCE = 8.5,
	LIFT = [16, 26] as const,
	CENTRE = 1,
	WALL = 0.6,
	CLOSEST = 4, // degrees; m; degrees above the horizontal, least and most; m up the pivot sits; m in from the boards; m
	NEAREST = 1.5,
	FARTHEST = 120; // m

/** cols × rows views, swinging `swing`° either way round the play and `rise`° up and down from the middle one */
export type Orbit = { cols: number; rows: number; swing: number; rise: number; width: number; height: number };

/** Where the views were taken from, round the pivot: the middle view's lift (°), the distance (m), the field of view (°), and the
 * depth's range (m). */
export type OrbitCamera = { lift: number; distance: number; fov: number; nearest: number; farthest: number };

/** Each view of scene `id`'s orbit, its colour and its depth as PNG data URLs, row by row from the highest, each row from the left. */
async function orbit(
	id: string,
	{ cols, rows, swing, rise, width, height }: Orbit,
): Promise<{ views: { color: string; depth: string }[]; camera: OrbitCamera }> {
	const scene = await GoalScene.load(`/scenes/${encodeURIComponent(id)}`);
	await scene.streamed; // every body in, not held at its first frames
	const S = scene.S,
		shot = S.puck.shot;
	if (!shot) throw new Error(`${id}: no shot`);
	const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
	renderer.setPixelRatio(1);
	renderer.setSize(width, height, false);
	document.body.append(renderer.domElement);
	const camera = new THREE.PerspectiveCamera(FOV, width / height, 0.05, 600);
	camera.up.set(0, 0, 1);
	const post = broadcastLook(
		renderer,
		scene.root,
		camera,
		LOOKS,
		width,
		height,
		new Perf(renderer.getContext() as WebGL2RenderingContext),
	)!;
	const f = shot.release,
		time = S.frame0 + f;
	scene.setTime(time, 0);
	const scorer = scene.people.find((p) => p.meta.key === S.shots.pov?.own),
		[px, py] = S.puck.xy[f]!,
		[sx, sy] = scorer ? scorer.rootAt(f) : [px, py],
		pivot = new THREE.Vector3((px + sx) / 2, (py + sy) / 2, CENTRE);
	const goal = S.shots.goal ?? Object.values(S.shots)[0]!,
		near = goal.frames.reduce((b, fr, i) => (Math.abs(fr - time) < Math.abs(goal.frames[b]! - time) ? i : b), 0),
		from = cameraPose(goal.cams[near]!, S.height).position.sub(pivot),
		azimuth = Math.atan2(from.y, from.x),
		lift = THREE.MathUtils.clamp(
			Math.atan2(from.z, Math.hypot(from.x, from.y)),
			THREE.MathUtils.degToRad(LIFT[0]),
			THREE.MathUtils.degToRad(LIFT[1]),
		);
	const across = (i: number, n: number) => (2 * i) / Math.max(n - 1, 1) - 1,
		dirs = Array.from({ length: cols * rows }, (_, v) => {
			const a = azimuth + THREE.MathUtils.degToRad(swing) * across(v % cols, cols),
				e = lift - THREE.MathUtils.degToRad(rise) * across(Math.floor(v / cols), rows);
			return new THREE.Vector3(Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e));
		});
	let d = DISTANCE;
	while (d > CLOSEST && dirs.some((u) => sdRink(S.rink, pivot.x + u.x * d, pivot.y + u.y * d) > -WALL)) d -= 0.25;
	const target = new THREE.WebGLRenderTarget(width, height, { depthTexture: new THREE.DepthTexture(width, height) }),
		nearness = new FullScreenQuad(
			new THREE.ShaderMaterial({
				uniforms: {
					tDepth: { value: target.depthTexture },
					cameraNear: { value: camera.near },
					cameraFar: { value: camera.far },
				},
				vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
				fragmentShader: `#include <packing>
uniform sampler2D tDepth; uniform float cameraNear, cameraFar; varying vec2 vUv;
void main() {
  float z = -perspectiveDepthToViewZ(texture2D(tDepth, vUv).x, cameraNear, cameraFar);
  gl_FragColor = vec4(vec3(clamp((1.0 / z - ${(1 / FARTHEST).toFixed(6)}) / ${(1 / NEAREST - 1 / FARTHEST).toFixed(6)}, 0.0, 1.0)), 1.0);
}`,
			}),
		);
	const views = dirs.map((u) => {
		camera.position.copy(pivot).addScaledVector(u, d);
		camera.lookAt(pivot);
		camera.updateMatrixWorld();
		post.render();
		const color = renderer.domElement.toDataURL("image/png");
		renderer.setRenderTarget(target);
		renderer.clear();
		renderer.render(scene.root, camera);
		renderer.setRenderTarget(null);
		nearness.render(renderer);
		return { color, depth: renderer.domElement.toDataURL("image/png") };
	});
	return { views, camera: { lift: THREE.MathUtils.radToDeg(lift), distance: d, fov: FOV, nearest: NEAREST, farthest: FARTHEST } };
}

Object.assign(window, { orbit });
