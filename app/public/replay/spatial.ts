// The moment's sound placed in the scene, heard from where the camera is (the Web Audio listener carried with it):
//   the bed     the broadcast's own sound (sound.ts), mostly the crowd, as heard from the camera: full and bright near the stands (as the
//               broadcast's microphones are), softer and duller further from them (BED: the camera's distance to the nearest stands,
//               the boards' top within the rink and the slope outside it, over FAR m)
//   skates      each skater's blades on the ice from where they are: a hiss rising with their speed, a scrape on a hard stop (braking
//               as spray.ts measures it), each a band of noise (SKATE) through its own HRTF panner, so it moves round the listener
//   the puck    a crack where it changes speed or way sharply (a stick, a pass taken, the boards: HIT m/s of change, GAP s apart at
//               least), a short sound synthesised once, at the puck's place; the goal itself is the broadcast's own (the bed)
// Each sound is set by the moment's frame, so it is the same however that frame is reached: skates follow the frame shown, the
// puck's sounds fire as playing time passes their frame (not when scrubbed or wound back) at the moment's speed (deeper in slow
// motion). All of it goes through the sound's film effect and mute.
import type * as THREE from "three";
import type { GoalScene } from "./scene";
import type { Sound } from "./sound";

const BED = { FAR: 25, LOW: 3500, QUIET: 0.7 } as const; // m; Hz and gain heard FAR off the stands
const SKATE = {
	HISS: 4200,
	SCRAPE: 1600, // Hz, the bands' centres
	FAST: 8, // m/s: full hiss
	BRAKE: [4, 12], // m/s²: braking from no scrape to full
	LEVEL: 0.22,
	SIGMA: 3, // frames the track is smoothed over
	REF: 3, // m: heard at full level within this
} as const;
const PUCK = { HIT: 9, GAP: 0.25, LEVEL: 0.9, REF: 4 } as const;

type Voice = { panner: PannerNode; gain: GainNode; band: BiquadFilterNode };
type Hit = { frame: number; x: number; y: number };

/** A panner with HRTF, its level falling off past `ref` m. */
function panner(c: AudioContext, ref: number) {
	return new PannerNode(c, { panningModel: "HRTF", distanceModel: "inverse", refDistance: ref, rolloffFactor: 1.3 });
}

/** Mono sound `seconds` long from f(t) per sample. */
function synth(c: AudioContext, seconds: number, f: (t: number) => number) {
	const n = Math.round(seconds * c.sampleRate),
		b = c.createBuffer(1, n, c.sampleRate),
		d = b.getChannelData(0);
	for (let i = 0; i < n; i++) d[i] = f(i / c.sampleRate);
	return b;
}

const noise = () => Math.random() * 2 - 1;

/** The sounds synthesised once: a loop of noise, a stick's crack. */
function sounds(c: AudioContext) {
	const band = (lo: number, hi: number) => {
		// noise through a band: one pole below `hi`, less one below `lo`
		const a = 1 - Math.exp((-2 * Math.PI * hi) / c.sampleRate),
			b = 1 - Math.exp((-2 * Math.PI * lo) / c.sampleRate);
		let p = 0,
			q = 0;
		return () => {
			const n = noise();
			p += a * (n - p);
			q += b * (n - q);
			return p - q;
		};
	};
	const click = band(1500, 9000),
		knock = band(80, 500);
	return {
		noise: synth(c, 2, noise),
		crack: synth(c, 0.12, (t) => Math.exp(-t / 0.006) * click() * 2.2 + Math.exp(-t / 0.02) * knock() * 0.8),
	};
}

/** Signed distance (m) to the rink's outline from its half sizes and corner radius; negative inside. */
function sdRink(x: number, y: number, hx: number, hy: number, r: number) {
	const qx = Math.abs(x) - (hx - r),
		qy = Math.abs(y) - (hy - r);
	return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - r;
}

export class Space {
	private skaters: { voice: Voice; speed: Float32Array; brake: Float32Array; at: (f: number) => [number, number] }[];
	private hits: Hit[];
	private buffers: ReturnType<typeof sounds>;
	private last: number | null = null;

	constructor(
		private sound: Sound,
		private scene: GoalScene,
	) {
		const c = sound.context,
			S = scene.S;
		this.buffers = sounds(c);
		this.skaters = scene.people
			.filter((p) => !p.meta.goalie)
			.map((p) => {
				const n = p.meta.root.length,
					smooth = (a: number[]) => {
						const r = Math.ceil(3 * SKATE.SIGMA);
						return a.map((_, i) => {
							let s = 0,
								w = 0;
							for (let k = -r; k <= r; k++) {
								const e = Math.exp(-(k * k) / (2 * SKATE.SIGMA ** 2));
								s += e * a[Math.min(Math.max(i + k, 0), n - 1)]!;
								w += e;
							}
							return s / w;
						});
					},
					x = smooth(p.meta.root.map((q) => q[0] / 1000)),
					y = smooth(p.meta.root.map((q) => q[1] / 1000)),
					d = (a: number[], i: number) => ((a[Math.min(i + 1, n - 1)]! - a[Math.max(i - 1, 0)]!) / 2) * S.fps,
					vx = x.map((_, i) => d(x, i)),
					vy = y.map((_, i) => d(y, i)),
					speed = Float32Array.from(vx, (v, i) => (p.meta.seen[i] ? Math.hypot(v, vy[i]!) : 0)),
					brake = Float32Array.from(vx, (v, i) => {
						const s = speed[i]!;
						return s < 2 ? 0 : -(d(vx, i) * v + d(vy, i) * vy[i]!) / s;
					});
				const source = new AudioBufferSourceNode(c, { buffer: this.buffers.noise, loop: true }),
					band = new BiquadFilterNode(c, { type: "bandpass", frequency: SKATE.HISS, Q: 0.9 }),
					gain = new GainNode(c, { gain: 0 }),
					pan = panner(c, SKATE.REF);
				source.loopStart = Math.random();
				source.connect(band).connect(gain).connect(pan).connect(sound.events);
				source.start(0, Math.random() * 2);
				return { voice: { panner: pan, gain, band }, speed, brake, at: (f: number) => p.rootAt(f) };
			});
		this.hits = puckHits(scene);
	}

	/**
	 * Heard at fractional source frame `frame` (from the scene's first), the moment moving at `speed` moment s a second while
	 * `playing` (0 frozen), from `camera`.
	 */
	update(camera: THREE.Camera, frame: number, speed: number, playing: boolean) {
		const c = this.sound.context,
			t = c.currentTime,
			e = camera.matrixWorld.elements,
			L = c.listener,
			[px, py, pz] = [e[12]!, e[13]!, e[14]!],
			ease = 0.04;
		// the listener: at the camera, facing as it does (-z), its up its y
		if (L.positionX) {
			L.positionX.setTargetAtTime(px, t, ease);
			L.positionY.setTargetAtTime(py, t, ease);
			L.positionZ.setTargetAtTime(pz, t, ease);
			L.forwardX.setTargetAtTime(-e[8]!, t, ease);
			L.forwardY.setTargetAtTime(-e[9]!, t, ease);
			L.forwardZ.setTargetAtTime(-e[10]!, t, ease);
			L.upX.setTargetAtTime(e[4]!, t, ease);
			L.upY.setTargetAtTime(e[5]!, t, ease);
			L.upZ.setTargetAtTime(e[6]!, t, ease);
		} else {
			L.setPosition(px, py, pz);
			L.setOrientation(-e[8]!, -e[9]!, -e[10]!, e[4]!, e[5]!, e[6]!);
		}
		// the bed: from the nearest stands
		const R = this.scene.S.rink,
			sd = sdRink(px, py, R.hx, R.hy, R.corner),
			off = sd < 0 ? Math.hypot(-sd, Math.max(pz - R.wall, 0)) : Math.max(pz - (R.wall + R.slope * Math.max(sd - R.ledge, 0)), 0),
			near = Math.exp(-off / BED.FAR);
		this.sound.bedTone.frequency.setTargetAtTime(Math.exp(Math.log(BED.LOW) * (1 - near) + Math.log(20000) * near), t, 0.2);
		this.sound.bed.gain.setTargetAtTime(BED.QUIET + (1 - BED.QUIET) * near, t, 0.2);
		// the skates, at the frame shown, their level with the moment's speed (silent frozen)
		const moving = playing ? Math.min(Math.sqrt(Math.max(speed, 0)), 1) : 0;
		for (const s of this.skaters) {
			const i = Math.min(Math.max(Math.round(frame), 0), s.speed.length - 1),
				[x, y] = s.at(frame),
				hiss = Math.min(s.speed[i]! / SKATE.FAST, 1),
				scrape = Math.min(Math.max((s.brake[i]! - SKATE.BRAKE[0]) / (SKATE.BRAKE[1] - SKATE.BRAKE[0]), 0), 1),
				v = s.voice;
			v.panner.positionX.setTargetAtTime(x, t, ease);
			v.panner.positionY.setTargetAtTime(y, t, ease);
			v.panner.positionZ.setTargetAtTime(0.05, t, ease);
			v.gain.gain.setTargetAtTime(SKATE.LEVEL * moving * (0.35 * hiss * hiss + 1.6 * scrape), t, 0.05);
			v.band.frequency.setTargetAtTime(SKATE.HISS + (SKATE.SCRAPE - SKATE.HISS) * scrape, t, 0.05);
		}
		// the puck's hits passed since the last frame heard, while playing on (not a jump)
		const was = this.last;
		this.last = frame;
		if (!playing || was === null || frame <= was || frame - was > 4) return;
		for (const h of this.hits) if (h.frame > was && h.frame <= frame) this.play(h, speed);
	}

	private play(h: Hit, speed: number) {
		const c = this.sound.context,
			src = new AudioBufferSourceNode(c, { buffer: this.buffers.crack }),
			p = panner(c, PUCK.REF),
			g = new GainNode(c, { gain: PUCK.LEVEL });
		src.playbackRate.value = Math.min(Math.max(speed, 0.35), 1);
		p.positionX.value = h.x;
		p.positionY.value = h.y;
		p.positionZ.value = 0.02;
		src.connect(g).connect(p).connect(this.sound.events);
		src.onended = () => p.disconnect();
		src.start();
	}

	/** Silent till updated again (the sound turned off). */
	quiet() {
		const t = this.sound.context.currentTime;
		for (const s of this.skaters) s.voice.gain.gain.setTargetAtTime(0, t, 0.02);
		this.last = null;
	}

	dispose() {
		for (const s of this.skaters) s.voice.panner.disconnect();
	}
}

/** The puck's hits: sharp changes of its velocity while seen (at least GAP s apart). */
function puckHits(scene: GoalScene): Hit[] {
	const P = scene.S.puck,
		fps = scene.S.fps,
		n = P.xy.length,
		hits: Hit[] = [],
		v = (i: number) => [(P.xy[i + 1]![0] - P.xy[i]![0]) * fps, (P.xy[i + 1]![1] - P.xy[i]![1]) * fps] as const;
	let lastAt = -Infinity;
	for (let i = 1; i + 1 < n; i++) {
		if (P.show[i - 1]! < 0.5 || P.show[i]! < 0.5 || P.show[i + 1]! < 0.5) continue;
		const a = v(i - 1),
			b = v(i);
		if (Math.hypot(b[0] - a[0], b[1] - a[1]) < PUCK.HIT || (i - lastAt) / fps < PUCK.GAP) continue;
		hits.push({ frame: i, x: P.xy[i]![0], y: P.xy[i]![1] });
		lastAt = i;
	}
	return hits;
}
