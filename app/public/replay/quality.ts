// How much the replay renders, kept within a frame budget. "auto" starts at high quality, full size, and every WINDOW frames
// looks at what frames took: on the GPU (Perf.gpuFrame) where the browser times it, else on the wall clock, which can only show
// being late (the display's refresh caps it). Late, it first renders smaller (a STEP smaller, down to SMALLEST of the canvas's
// size), and only at SMALLEST drops a quality (high, medium, low); but not when the frame's own CPU time is most of the budget, as
// neither would help then. With the GPU's time under ROOM of the budget for CALM windows in a row it renders larger again; on the
// wall clock it can only try, after WAIT windows without a late one, and waits twice as long each time a try comes back late.
// Quality never comes back up by itself: it was dropped because the smallest size did not fit. A fixed setting (high, medium,
// low) is that, at full size.
import type { Quality } from "./post";

export type Setting = "auto" | Quality;
export const SETTINGS: readonly Setting[] = ["auto", "high", "medium", "low"];

const BUDGET = 1000 / 60, // ms
	LATE = 1.15, // a wall-clock frame this many budgets long was late (one refresh missed)
	CPU = 0.7, // of the budget: a frame whose CPU time is more is held up by the CPU, not the GPU
	ROOM = 0.6,
	CALM = 4,
	WINDOW = 30,
	STEP = 0.85,
	SMALLEST = 0.6,
	WAIT = 8;
const LOWER: Record<Quality, Quality | null> = { high: "medium", medium: "low", low: null };

export class Budget {
	setting: Setting = "auto";
	quality: Quality = "high";
	scale = 1;
	private gpu: number[] = [];
	private wall: number[] = [];
	private cpu: number[] = [];
	private calm = 0;
	private wait = WAIT;
	private since = 0;
	/** the last change was a try at a larger size, on the wall clock */
	private trying = false;

	set(setting: Setting) {
		this.setting = setting;
		this.quality = setting === "auto" ? "high" : setting;
		this.scale = 1;
		this.reset();
	}

	/** Start measuring afresh: after a change of size or quality, or a pause (its frames say nothing of the load). */
	reset() {
		this.gpu = [];
		this.wall = [];
		this.cpu = [];
		this.calm = 0;
		this.since = 0;
	}

	/**
	 * One frame drawn: `ms` since the last on the wall clock, `cpu` the ms the CPU spent on it, `gpu` its GPU time if timed. Returns
	 * what to change, if anything: the scale and the quality to render at from now on.
	 */
	frame(ms: number, cpu: number, gpu: number | null): { scale: number; quality: Quality } | null {
		if (this.setting !== "auto") return null;
		this.wall.push(ms);
		this.cpu.push(cpu);
		if (gpu !== null) this.gpu.push(gpu);
		if (this.wall.length < WINDOW) return null;
		const timed = this.gpu.length >= WINDOW / 2,
			load = percentile(timed ? this.gpu : this.wall, 0.75) / BUDGET,
			late = timed ? load > 0.9 : load > LATE,
			held = percentile(this.cpu, 0.75) > CPU * BUDGET;
		this.gpu = [];
		this.wall = [];
		this.cpu = [];
		this.since++;
		if (late) {
			this.calm = 0;
			if (this.trying) this.wait *= 2;
			this.trying = false;
			this.since = 0;
			if (held && !timed) return null;
			if (this.scale > SMALLEST) return this.change(Math.max(this.scale * STEP, SMALLEST), this.quality);
			const lower = LOWER[this.quality];
			return lower ? this.change(this.scale, lower) : null;
		}
		this.trying = false;
		if (this.scale >= 1) return null;
		if (timed ? load < ROOM && ++this.calm >= CALM : this.since >= this.wait) {
			this.calm = 0;
			this.since = 0;
			this.trying = !timed;
			return this.change(Math.min(this.scale / STEP, 1), this.quality);
		}
		return null;
	}

	private change(scale: number, quality: Quality) {
		this.scale = scale;
		this.quality = quality;
		return { scale, quality };
	}
}

function percentile(a: number[], p: number) {
	const s = [...a].sort((x, y) => x - y);
	return s[Math.min(s.length - 1, Math.floor(p * s.length))]!;
}
