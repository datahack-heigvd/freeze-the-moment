// A moment's broadcast sound (/site/audio/<id>.m4a, scripts/build-audio.ts), played on the audio thread: decoded once into buffers (the sound as it
// is, and the copies slowed offline with their pitch kept, scripts/build-audio.ts) and read by an AudioWorklet (`stretch`), so a busy
// page (WebGL, post-processing) never starves it, and nothing is written to a media element as frames are drawn. The worklet reads
// from one position two ways at once: its pitch kept (WSOLA: overlapping windowed grains, each placed where it best continues the
// one before; GRAIN samples, half overlapping, SEARCH either way) and resampled (as tape or a film's projector slows); a speed (the
// `rate` AudioParam: buffer seconds a second) moves it. Their gains mix the two through a low-pass, then dry and through a reverb
// (a decaying noise ROOM s long, SMALL_ROOM on a phone-sized screen, connected only while it is heard): the slow-motion effect
// (effect). Every change is scheduled on the audio clock: a speed or the effect eased in, a tape stop (tapeStop: over TAPE s
// slowing to a quarter, the resampled copy taking over, muffled, faded, then stopped), a restart elsewhere (start: out and back in
// over FADE s, never a seek of a playing sound). Where it is (heard) is the worklet's own reading head, reported to the page every
// REPORT s and run on from there by the audio clock, less the output's latency: what is heard now. A page not served securely (plain
// http to a LAN address) has no AudioWorklet: there buffer sources play both copies resampled, sharing one speed (a constant source
// into their playbackRate), and where they are is run on from the speeds set. The owner keeps it in step with
// its picture, restarting it only when it is far off (a scrub, a loop, a shot changed). Sound on iOS: the session is "playback"
// (navigator.audioSession, so the ringer switch does not mute it), and the context is resumed by the first press on the page.
// With ?debug=audio a readout (window.audioDebug and a corner of the page): speed changes and restarts a second, the offset from
// the picture (ms), the output's latency and which reader plays.
import { param } from "./util";

// the worklet's scope (stretch runs there, as text)
declare const GRAIN_: number, SEARCH_: number, FADE_: number, REPORT_: number, sampleRate: number, currentTime: number;
declare class AudioWorkletProcessor {
	readonly port: MessagePort;
}
declare function registerProcessor(name: string, processor: unknown): void;

const GRAIN = 1024,
	SEARCH = 256, // samples
	FADE = 0.01, // s, each way
	REPORT = 0.05, // s
	ROOM = 2.4,
	SMALL_ROOM = 1.2, // s
	MUFFLE = 600,
	SLOWEST = 0.0625; // Hz; of the speed

/** The worklet: one sound, read from one position, its pitch kept (output 0) and resampled (output 1). */
function stretch() {
	const N = GRAIN_,
		H = N / 2,
		D = SEARCH_,
		window = Float32Array.from({ length: N }, (_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N));
	class Stretch extends AudioWorkletProcessor {
		static get parameterDescriptors() {
			return [{ name: "rate", defaultValue: 1, minValue: 0, maxValue: 4, automationRate: "k-rate" }];
		}
		data: Float32Array[] = [];
		/** the resampled head and the grains' nominal start (buffer samples) */
		pos = 0;
		grain = 0;
		last = 0;
		acc = [new Float32Array(N), new Float32Array(N)];
		/** samples of the grains' sum finished, at the front of acc */
		ready = 0;
		at = 0;
		playing = false;
		env = 0;
		want: number | null = null;
		stopAt = Infinity;
		told = 0;
		constructor() {
			super();
			this.port.onmessage = ({ data: d }) => {
				if (d.data) this.data = d.data;
				if (d.start !== undefined) {
					this.want = d.start;
					this.stopAt = Infinity;
					if (!this.playing) this.jump();
				}
				if (d.stop !== undefined) this.stopAt = d.stop;
			};
		}
		jump() {
			this.pos = this.grain = this.want!;
			this.last = this.grain - H;
			this.want = null;
			for (const a of this.acc) a.fill(0);
			this.ready = 0;
			this.at = 0;
			this.playing = true;
		}
		x(c: number, i: number) {
			const d = this.data[Math.min(c, this.data.length - 1)];
			return d && i >= 0 && i < d.length ? d[i]! : 0;
		}
		mono(i: number) {
			return this.x(0, i) + this.x(1, i);
		}
		/** the next grain, where it best continues the last, added in; H more samples finished */
		next(rate: number) {
			const nominal = Math.round(this.grain),
				natural = this.last + H;
			let best = nominal,
				top = -Infinity;
			for (let k = -D; k <= D; k += 4) {
				const s = nominal + k;
				let xy = 0,
					yy = 0;
				for (let n = 0; n < H; n += 4) {
					const a = this.mono(s + n);
					xy += a * this.mono(natural + n);
					yy += a * a;
				}
				const c = xy / Math.sqrt(yy + 1e-9);
				if (c > top) {
					top = c;
					best = s;
				}
			}
			for (let c = 0; c < 2; c++) {
				const a = this.acc[c]!;
				a.copyWithin(0, H);
				a.fill(0, N - H);
				for (let n = 0; n < N; n++) a[n]! += window[n]! * this.x(c, best + n);
			}
			this.last = best;
			this.grain += H * rate;
			this.ready = H;
			this.at = 0;
		}
		process(_: Float32Array[][], outputs: Float32Array[][], params: Record<string, Float32Array>) {
			const [kept, film] = outputs as [Float32Array[], Float32Array[]],
				rate = params.rate![0]!,
				n = kept[0]!.length,
				step = 1 / (FADE_ * sampleRate);
			if (currentTime >= this.stopAt) {
				this.playing = false;
				this.stopAt = Infinity;
				this.env = 0;
			}
			if (!this.playing && this.want === null) return true;
			for (let i = 0; i < n; i++) {
				// out to nothing for a restart, then back in where it was asked for
				const target = this.want === null && this.playing ? 1 : 0;
				this.env += Math.sign(target - this.env) * Math.min(step, Math.abs(target - this.env));
				if (this.want !== null && this.env === 0) this.jump();
				if (this.at >= this.ready) this.next(rate);
				const p = Math.floor(this.pos),
					f = this.pos - p;
				for (let c = 0; c < 2; c++) {
					kept[c]![i] = this.acc[c]![this.at]! * this.env;
					film[c]![i] = (this.x(c, p) * (1 - f) + this.x(c, p + 1) * f) * this.env;
				}
				this.at++;
				this.pos += rate;
			}
			if (currentTime - this.told >= REPORT_) {
				this.told = currentTime;
				this.port.postMessage({ pos: this.pos, time: currentTime + n / sampleRate });
			}
			return true;
		}
	}
	registerProcessor("stretch", Stretch);
}

const WORKLET = URL.createObjectURL(
	new Blob([`const GRAIN_ = ${GRAIN}, SEARCH_ = ${SEARCH}, FADE_ = ${FADE}, REPORT_ = ${REPORT};\n(${stretch})();`], {
		type: "text/javascript",
	}),
);

/**
 * What reads the sound into the two copies' gains (`kept`, `film`): `rate` its speed (buffer s a second); `play` from buffer
 * position pos (samples), after fading out what plays (FADE s) when `playing`; `stop` at audio clock time `at`. The worklet tells
 * where its head is (`tells`); without one (a page not served securely has no AudioWorklet) buffer sources play both copies
 * resampled, their speed one shared param, and where they are is worked out from it.
 */
type Reader = {
	rate: AudioParam;
	tells: boolean;
	play(buffer: AudioBuffer, pos: number, playing: boolean): void;
	stop(at: number): void;
};

function worklet(c: AudioContext, kept: GainNode, film: GainNode, told: (d: { pos: number; time: number }) => void): Reader {
	const node = new AudioWorkletNode(c, "stretch", { numberOfInputs: 0, numberOfOutputs: 2, outputChannelCount: [2, 2] });
	node.connect(kept, 0);
	node.connect(film, 1);
	node.port.onmessage = ({ data }) => told(data);
	let data: AudioBuffer | null = null;
	return {
		rate: node.parameters.get("rate")!,
		tells: true,
		play(buffer, pos) {
			if (data !== buffer) {
				data = buffer;
				const channels = Array.from({ length: buffer.numberOfChannels }, (_, ch) => buffer.getChannelData(ch).slice());
				node.port.postMessage(
					{ data: channels },
					channels.map((d) => d.buffer),
				);
			}
			node.port.postMessage({ start: pos });
		},
		stop: (at) => node.port.postMessage({ stop: at }),
	};
}

function sources(c: AudioContext, kept: GainNode, film: GainNode): Reader {
	const rate = new ConstantSourceNode(c);
	rate.start();
	let live: { nodes: AudioBufferSourceNode[]; envs: GainNode[] } | null = null;
	const stop = (at: number) => {
		for (const n of live?.nodes ?? []) n.stop(at);
		live = null;
	};
	return {
		rate: rate.offset,
		tells: false,
		play(buffer, pos, playing) {
			const t = c.currentTime,
				at = t + (playing ? FADE : 0);
			for (const e of live?.envs ?? []) {
				e.gain.setValueAtTime(1, t);
				e.gain.linearRampToValueAtTime(0, at);
			}
			stop(at);
			const envs = [kept, film].map((out) => new GainNode(c, { gain: 0 }).connect(out) as GainNode),
				nodes = envs.map((env) => {
					const n = new AudioBufferSourceNode(c, { buffer, playbackRate: 0 });
					rate.connect(n.playbackRate);
					n.connect(env);
					n.start(at, Math.max(pos / c.sampleRate, 0));
					return n;
				});
			for (const e of envs) {
				e.gain.setValueAtTime(0, at);
				e.gain.linearRampToValueAtTime(1, at + FADE);
			}
			live = { nodes, envs };
		},
		stop,
	};
}

type Debug = { rateWrites: number; starts: number; offset: number | null; latency: number; state: string; reader: string };
const DEBUG = param("debug") === "audio";

/** The file of the sound `base` (a path without its extension) at `tempo`: its own, or its copy slowed to it. */
const sourceOf = (base: string, tempo: number) => (tempo === 1 ? `${base}.m4a` : `${base}@${tempo}.m4a`);

/** The sound of `url`, with its copies slowed offline at `tempos` (fetched ahead, decoded once wanted). */
export class Sound {
	readonly context: AudioContext;
	/** what is heard of the moment while its sound is muted, 0, or not, 1 */
	readonly master: GainNode;
	/** the broadcast as heard from where the camera is (spatial.ts): its tone (a low-pass) and level, before the film effect */
	readonly bedTone: BiquadFilterNode;
	readonly bed: GainNode;
	/** where sounds placed in the scene (spatial.ts) come in: through the film effect, muted with the rest */
	readonly events: GainNode;
	/** the whole mix (the broadcast and the sounds placed in the scene), faded out together as it ends or stops */
	private mix: GainNode;
	/** the copy playing: 1, or one of the tempos */
	tempo = 1;
	private reader: Promise<Reader>;
	private kept: GainNode;
	private film: GainNode;
	private low: BiquadFilterNode;
	private dry: GainNode;
	private wet: GainNode;
	private room: ConvolverNode;
	private roomOn = false;
	private roomOff = 0;
	/** the files fetched and decoded, by their URL */
	private bytes = new Map<string, Promise<ArrayBuffer>>();
	private decoded = new Map<string, Promise<AudioBuffer>>();
	private tempos: readonly number[];
	/** the worklet's head as last told (buffer s, audio clock s), and the speed since */
	private told = { pos: 0, time: 0 };
	private rate = 1;
	private strength = -1;
	private wanted: [number, number, boolean] = [0, 0.15, true];
	playing = false;
	/** counts for the readout: speed changes and restarts, and the owner's last offset from its picture (ms) */
	private debug: Debug = { rateWrites: 0, starts: 0, offset: null, latency: 0, state: "", reader: "" };

	constructor(
		private url: string,
		tempos: readonly number[] = [],
	) {
		this.tempos = tempos;
		const audioSession = (navigator as Navigator & { audioSession?: { type: string } }).audioSession;
		if (audioSession) audioSession.type = "playback";
		const c = (this.context = new AudioContext({ latencyHint: "playback" }));
		this.master = new GainNode(c);
		this.kept = new GainNode(c);
		this.film = new GainNode(c, { gain: 0 });
		this.low = new BiquadFilterNode(c, { type: "lowpass", frequency: 20000, Q: 0.7 });
		this.dry = new GainNode(c);
		this.wet = new GainNode(c, { gain: 0 });
		this.room = new ConvolverNode(c);
		const seconds = matchMedia("(max-width: 700px)").matches ? SMALL_ROOM : ROOM,
			n = Math.round(seconds * c.sampleRate),
			tail = c.createBuffer(2, n, c.sampleRate);
		for (let ch = 0; ch < 2; ch++) {
			const d = tail.getChannelData(ch);
			for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / n) ** 3;
		}
		this.room.buffer = tail;
		this.bedTone = new BiquadFilterNode(c, { type: "lowpass", frequency: 20000, Q: 0.5 });
		this.bed = new GainNode(c);
		this.events = new GainNode(c);
		this.mix = new GainNode(c);
		this.kept.connect(this.bedTone);
		this.film.connect(this.bedTone);
		this.bedTone.connect(this.bed).connect(this.mix);
		this.events.connect(this.mix);
		this.mix.connect(this.low);
		this.low.connect(this.dry).connect(this.master).connect(c.destination);
		this.wet.connect(this.master);
		this.reader = (
			isSecureContext && c.audioWorklet ? c.audioWorklet.addModule(WORKLET) : Promise.reject(new Error("no AudioWorklet"))
		)
			.then(() => worklet(c, this.kept, this.film, (d) => (this.told = d)))
			.catch(() => sources(c, this.kept, this.film));
		void this.reader.then((r) => (this.debug.reader = r.tells ? "worklet" : "buffer sources"));
		if (url) this.prefetch(url);
		// iOS lets sound start from the end of a touch, others from its start or a key
		const resume = () => this.resume();
		for (const e of ["pointerdown", "pointerup", "touchend", "click", "keydown"])
			addEventListener(e, resume, { capture: true, signal: this.presses.signal });
		if (DEBUG) readout(this.debug, this);
	}

	/** Another moment's sound (its URL) from now on, silent till started; what was fetched for others let go. */
	use(url: string) {
		if (url === this.url) return;
		void this.stop();
		this.url = url;
		for (const m of [this.bytes, this.decoded]) for (const k of m.keys()) if (!k.startsWith(url)) m.delete(k);
		this.prefetch(url);
	}

	/** The sound of `url` and its slowed copies fetched ahead. */
	prefetch(url: string) {
		for (const t of [1, ...this.tempos]) this.fetch(sourceOf(url, t));
	}

	private fetch(src: string) {
		let b = this.bytes.get(src);
		if (!b) {
			b = fetch(src).then((r) => {
				if (!r.ok) throw new Error(`${src}: ${r.status}`);
				return r.arrayBuffer();
			});
			b.catch(() => {});
			this.bytes.set(src, b);
		}
		return b;
	}

	private buffer(tempo: number) {
		const src = sourceOf(this.url, tempo);
		let d = this.decoded.get(src);
		if (!d) {
			d = this.fetch(src).then((bytes) => this.context.decodeAudioData(bytes.slice(0)));
			d.catch(() => {});
			this.decoded.set(src, d);
		}
		return d;
	}

	/** The output's latency (s): how long after the audio clock a sample is heard. */
	get latency() {
		return this.context.outputLatency || this.context.baseLatency || 0;
	}

	/** The moment's second heard now (by the copy playing; null while silent or not yet told). */
	heard(): number | null {
		if (!this.playing || !this.told.time) return null;
		const sr = this.context.sampleRate,
			now = this.context.currentTime;
		return ((this.told.pos + (now - this.told.time) * this.rate * sr) / sr - this.latency * this.rate) * this.tempo;
	}

	/**
	 * Play from the moment's second `m`, as heard `ahead` s from now (what it will be then, at `speed` moment s a second: the
	 * output's latency and the restart's fade made up), from the copy at `tempo`, at `rate` buffer s a second.
	 */
	async start(m: number, speed: number, rate: number, tempo = this.tempo) {
		const c = this.context,
			delay = this.playing ? FADE : 0, // till the restart: the sound playing faded out first
			ahead = delay + this.latency,
			pos = ((m + ahead * speed) / tempo) * c.sampleRate,
			run = ++this.run;
		this.playing = true;
		this.tempo = tempo;
		this.told = { pos, time: c.currentTime + delay };
		this.debug.starts++;
		this.resume();
		const asked = c.currentTime,
			[reader, buffer] = await Promise.all([this.reader, this.buffer(tempo)]);
		if (run !== this.run) return;
		// the moment went on while the sound was being got ready (decoded, the worklet loaded)
		const late = c.currentTime - asked,
			at = pos + late * (speed / tempo) * c.sampleRate;
		this.told = { pos: at, time: c.currentTime + delay };
		reader.play(buffer, at, delay > 0);
		this.cancel(reader);
		this.mix.gain.setTargetAtTime(1, c.currentTime + delay, FADE);
		this.strength = -1;
		this.effect(...this.wanted);
		void this.setRate(rate, true);
	}
	private run = 0;

	/** The speed (buffer s a second), eased in (or set at once), when it changes by more than a little. */
	async setRate(rate: number, now = false) {
		rate = Math.min(Math.max(rate, SLOWEST), 4);
		if (!now && Math.abs(rate - this.rate) <= 0.005 * this.rate) return;
		const reader = await this.reader,
			p = reader.rate,
			t = this.context.currentTime,
			sr = this.context.sampleRate;
		// where it is, from the speed it had, when nothing tells it
		if (!reader.tells && this.told.time)
			this.told = { pos: this.told.pos + Math.max(t - this.told.time, 0) * this.rate * sr, time: Math.max(t, this.told.time) };
		this.rate = rate;
		if (now) p.setValueAtTime(rate, t);
		else p.setTargetAtTime(rate, t, 0.03);
		this.debug.rateWrites++;
	}

	/**
	 * How far the sound goes the way films do (0 not, 1 slowed right down), eased in over `ease` s when it changes: the resampled
	 * copy taking over from the pitch-kept one, and, `room`, muffled and in the reverb.
	 */
	effect(strength: number, ease: number, room = true) {
		this.wanted = [strength, ease, room];
		if (Math.abs(strength - this.strength) < 0.01) return;
		this.strength = strength;
		const t = this.context.currentTime,
			r = room ? strength : 0,
			wet = 0.7 * r;
		this.kept.gain.setTargetAtTime(Math.cos((strength * Math.PI) / 2), t, ease);
		this.film.gain.setTargetAtTime(Math.sin((strength * Math.PI) / 2), t, ease);
		this.low.frequency.setTargetAtTime(Math.exp(Math.log(20000) * (1 - r) + Math.log(MUFFLE) * r), t, ease);
		this.dry.gain.setTargetAtTime(1 - 0.45 * r, t, ease);
		this.wet.gain.setTargetAtTime(wet, t, ease);
		// the reverb only while it is heard (it costs the audio thread), let go once its level and tail have died away
		if (wet > 0.001 && !this.roomOn) {
			this.low.connect(this.room).connect(this.wet);
			this.roomOn = true;
		}
		clearTimeout(this.roomOff);
		if (wet <= 0.001 && this.roomOn)
			this.roomOff = window.setTimeout(
				() => {
					this.low.disconnect(this.room);
					this.room.disconnect();
					this.roomOn = false;
				},
				(5 * ease + ROOM) * 1000,
			);
	}

	/** Stopped as a tape does: over TAPE s slowing to a quarter, the resampled copy taking over, muffled and faded (the whole mix
	 * with it), then silent. */
	async tapeStop(tape: number) {
		if (!this.playing) return;
		this.playing = false;
		this.run++;
		const reader = await this.reader,
			t = this.context.currentTime,
			p = reader.rate;
		this.cancel(reader);
		p.setValueAtTime(this.rate, t);
		p.linearRampToValueAtTime(Math.max(this.rate / 4, SLOWEST), t + tape);
		this.kept.gain.setTargetAtTime(0, t, 0.03);
		this.film.gain.setTargetAtTime(1, t, 0.03);
		this.mix.gain.setTargetAtTime(0, t + tape / 3, tape / 4);
		this.low.frequency.setTargetAtTime(MUFFLE, t, tape / 3);
		this.strength = -1;
		reader.stop(t + tape);
		this.debug.rateWrites++;
	}

	/** Faded out over `fade` s as it plays (its speed and pitch kept), the whole mix together, then silent: an ending heard, not
	 * slowed down. */
	async fadeOut(fade: number) {
		if (!this.playing) return;
		this.playing = false;
		this.run++;
		const reader = await this.reader,
			t = this.context.currentTime;
		this.cancel(reader);
		this.mix.gain.setTargetAtTime(0, t, fade / 4);
		this.strength = -1;
		reader.stop(t + fade);
	}

	/** Heard or not (eased over a few ms); called from a press, it also lets a browser that holds sound start it. */
	setMuted(muted: boolean) {
		this.resume();
		this.master.gain.setTargetAtTime(muted ? 0 : 1, this.context.currentTime, 0.02);
	}

	/** Silent at once. */
	async stop() {
		this.playing = false;
		this.run++;
		(await this.reader).stop(0);
	}

	/** The owner's offset from its picture (s, the sound ahead positive), for the readout. */
	note(offset: number | null, state: string) {
		this.debug.offset = offset === null ? null : offset * 1000;
		this.debug.state = state;
	}

	/** What was scheduled from now on let go, each value held where it got to. */
	private cancel(reader: Reader) {
		const t = this.context.currentTime;
		for (const g of [this.kept.gain, this.film.gain, this.mix.gain, this.low.frequency, reader.rate])
			g.cancelAndHoldAtTime ? g.cancelAndHoldAtTime(t) : g.cancelScheduledValues(t);
	}

	/** Lets a browser that holds sound start it; a closed context stays closed. */
	private resume() {
		if (this.context.state !== "closed") void this.context.resume().catch(() => {});
	}
	private presses = new AbortController();

	dispose() {
		this.presses.abort();
		void this.context.close();
	}
}

/** ?debug=audio: the sound's counts a second, its offset from the picture and its latency, in a corner and as window.audioDebug. */
function readout(d: Debug, sound: Sound) {
	const box = document.createElement("pre");
	box.style.cssText =
		"position:fixed;left:8px;bottom:8px;z-index:99;margin:0;padding:6px 8px;border-radius:6px;background:rgba(0,0,0,.7);color:#9f9;font:11px/1.35 ui-monospace,monospace;pointer-events:none;";
	document.body.append(box);
	let last = { rateWrites: 0, starts: 0, t: performance.now() };
	const out = { rateWrites: 0, starts: 0, offset: null as number | null, latency: 0, state: "" };
	(window as unknown as { audioDebug: typeof out }).audioDebug = out;
	setInterval(() => {
		const now = performance.now(),
			dt = (now - last.t) / 1000;
		out.rateWrites = +((d.rateWrites - last.rateWrites) / dt).toFixed(1);
		out.starts = +((d.starts - last.starts) / dt).toFixed(1);
		out.offset = d.offset === null ? null : Math.round(d.offset);
		out.latency = Math.round(sound.latency * 1000);
		out.state = `${d.state} ${sound.context.state} (${d.reader})`;
		last = { rateWrites: d.rateWrites, starts: d.starts, t: now };
		box.textContent = `sound  ${out.state}\nspeed changes/s  ${out.rateWrites}\nrestarts/s  ${out.starts}\noffset  ${out.offset ?? "-"} ms\nlatency  ${out.latency} ms`;
	}, 500);
}
