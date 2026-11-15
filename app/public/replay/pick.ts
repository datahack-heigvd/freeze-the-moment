// Who is under a point of the view, and their speed in the play shown. Each person is picked as an upright column where they
// stand (COLUMN m round its axis, TALL m high), not by their body's triangles: nearest the line of sight wins, so a skater is
// found from any camera, from above as well as from the ice, and a hand or a stick out to the side does not take the pointer.
import * as THREE from "three";
import type { Person } from "./people";
import type { SceneData } from "./types";

const COLUMN = 0.5, // m
	TALL = 1.9, // m
	SPAN = 0.4; // s over which a speed is measured: shorter shakes with the tracking

const ray = new THREE.Raycaster(),
	base = new THREE.Vector3(),
	top = new THREE.Vector3();

/** Index in `people` of the one at `ndc` (the view's -1 to 1) through `camera`, or -1; those `can` turns down are left out. */
export function pickPerson(people: Person[], camera: THREE.Camera, ndc: THREE.Vector2, can: (p: Person) => boolean): number {
	ray.setFromCamera(ndc, camera);
	let best = -1,
		near = COLUMN * COLUMN;
	people.forEach((p, j) => {
		if (!p.mesh.visible || !can(p)) return;
		const [x, y] = p.root();
		const d = ray.ray.distanceSqToSegment(base.set(x, y, 0), top.set(x, y, TALL));
		if (d < near) [best, near] = [j, d];
	});
	return best;
}

/** Person `p`'s speed at exported frame f and their top speed over the moment (km/h; now null where they are not seen). */
export function motionOf(S: SceneData, p: Person, f: number) {
	const r = p.meta.root,
		seen = p.meta.seen,
		w = Math.max(1, Math.round((SPAN * S.fps) / 2)),
		at = (i: number) => r[Math.min(Math.max(i, 0), r.length - 1)]!,
		speed = (i: number) => {
			const a = at(i - w),
				b = at(i + w);
			return (Math.hypot(b[0] - a[0], b[1] - a[1]) / 1000 / ((2 * w) / S.fps)) * 3.6;
		},
		known = (i: number) => !!seen[Math.max(i - w, 0)] && !!seen[Math.min(i + w, r.length - 1)];
	let fastest = 0;
	for (let i = w; i < r.length - w; i++) if (known(i)) fastest = Math.max(fastest, speed(i));
	const i = Math.round(f);
	return { now: known(i) ? speed(i) : null, top: fastest };
}
