// Where a replay's frame time goes. CPU spans (performance.now) and GPU spans (EXT_disjoint_timer_query_webgl2, when the browser
// offers it) by name, averaged over the last WINDOW frames; every CPU span is also a performance.measure ("replay:<name>"), so it
// shows in the browser's performance trace next to everything else. GPU spans nest: opening one inside another ends the outer one's
// query and resumes it after, so each pass's time is its own (the ice's mirror render inside the scene render, say). With `frames`
// on and the spans off, only the outermost span (the whole frame, FRAME) is timed: gpuFrame is then what the frame took on the
// GPU, as it is (all its spans together) with the spans on.
const WINDOW = 60;
/** the span around a whole frame: its own time is what no other span takes */
export const FRAME = "other";

type Pending = { name: string; query: WebGLQuery; frame: number };

export type PerfReport = {
	fps: number;
	/** ms per frame, wall clock */
	frame: number;
	cpu: [string, number][];
	/** null when the browser has no GPU timer */
	gpu: [string, number][] | null;
	calls: number;
	triangles: number;
	pixels: [number, number];
};

export class Perf {
	enabled = false;
	/** time each frame on the GPU as a whole (gpuFrame), even with the spans off */
	measure = false;
	/** GPU ms of the latest frame whose timing is in, or null (no timer, or none yet) */
	gpuFrame: number | null = null;
	private skipped = 0;
	private gl: WebGL2RenderingContext;
	private timer: { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number } | null;
	private cpu = new Map<string, number[]>();
	private gpu = new Map<string, number[]>();
	private frames: number[] = [];
	private stack: string[] = [];
	private open: Pending | null = null;
	private pending: Pending[] = [];
	private count = 0;
	/** GPU ms by name of the frames whose queries are still coming in */
	private collected = new Map<number, Map<string, number>>();

	constructor(gl: WebGL2RenderingContext) {
		this.gl = gl;
		this.timer = gl.getExtension("EXT_disjoint_timer_query_webgl2");
	}

	/** Time fn on the CPU as span `name`. */
	span<T>(name: string, fn: () => T): T {
		if (!this.enabled) return fn();
		const t0 = performance.now();
		try {
			return fn();
		} finally {
			const t1 = performance.now();
			push(this.cpu, name, t1 - t0);
			performance.measure(`replay:${name}`, { start: t0, end: t1 });
		}
	}

	/** Open GPU span `name` (closed by gpuEnd); nested spans pause the one around them. */
	gpuBegin(name: string) {
		if (!this.timer || !(this.enabled || (this.measure && name === FRAME && !this.stack.length))) {
			this.skipped++;
			return;
		}
		this.stop();
		this.stack.push(name);
		this.start(name);
	}

	gpuEnd() {
		if (this.skipped) {
			this.skipped--;
			return;
		}
		this.stop();
		this.stack.pop();
		const outer = this.stack.at(-1);
		if (outer) this.start(outer);
	}

	private start(name: string) {
		const query = this.gl.createQuery()!;
		this.gl.beginQuery(this.timer!.TIME_ELAPSED_EXT, query);
		this.open = { name, query, frame: this.count };
	}

	private stop() {
		if (!this.open) return;
		this.gl.endQuery(this.timer!.TIME_ELAPSED_EXT);
		this.pending.push(this.open);
		this.open = null;
	}

	/** One frame done, `ms` after the last: collect the GPU spans that have finished. */
	frame(ms: number) {
		if (!this.enabled && !this.measure) return;
		this.count++;
		if (this.enabled) {
			this.frames.push(ms);
			if (this.frames.length > WINDOW) this.frames.shift();
		}
		if (!this.timer) return;
		const gl = this.gl,
			disjoint = gl.getParameter(this.timer.GPU_DISJOINT_EXT) as boolean;
		while (this.pending.length && gl.getQueryParameter(this.pending[0]!.query, gl.QUERY_RESULT_AVAILABLE)) {
			const { name, query, frame } = this.pending.shift()!,
				spans = this.collected.get(frame) ?? new Map<string, number>();
			if (!disjoint) spans.set(name, (spans.get(name) ?? 0) + (gl.getQueryParameter(query, gl.QUERY_RESULT) as number) / 1e6);
			this.collected.set(frame, spans);
			gl.deleteQuery(query);
		}
		const waiting = this.pending[0]?.frame ?? this.count; // frames before this one have all their spans
		for (const [frame, spans] of this.collected) {
			if (frame >= waiting) continue;
			let total = 0;
			for (const [name, t] of spans) {
				total += t;
				if (this.enabled) push(this.gpu, name, t);
			}
			this.gpuFrame = total;
			this.collected.delete(frame);
		}
	}

	report(info: { calls: number; triangles: number }, pixels: [number, number]): PerfReport {
		const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / Math.max(a.length, 1);
		const table = (m: Map<string, number[]>) =>
			[...m].map(([name, t]) => [name, mean(t)] as [string, number]).sort((a, b) => b[1] - a[1]);
		const frame = mean(this.frames);
		return {
			fps: frame ? 1000 / frame : 0,
			frame,
			cpu: table(this.cpu),
			gpu: this.timer ? table(this.gpu) : null,
			...info,
			pixels,
		};
	}

	reset() {
		this.cpu.clear();
		this.gpu.clear();
		this.collected.clear();
		this.frames = [];
	}
}

function push(m: Map<string, number[]>, name: string, t: number) {
	const a = m.get(name) ?? [];
	a.push(t);
	if (a.length > WINDOW) a.shift();
	m.set(name, a);
}
