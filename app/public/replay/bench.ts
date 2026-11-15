// ?bench=1: the moment played from its start at each quality in turn (high, high with depth of field on, medium, low, then auto:
// WARM s to settle, then SECONDS measured; the first shot has no depth of field but where it is on in all shots) and summed up one line each, shown over the page to copy and logged ("BENCH ..."): frames per second (mean), frame time
// (p50, p95, p99), the CPU's time on a frame (performance.now() round the viewer's frame: all, posing, rendering) and the stalls
// (frames over STALL ms), each with what its frame did (built people's frames, compiled shaders, changed quality or size, swapped the
// sound) or, with little CPU time of its own, "outside JS" (the GPU, garbage collection or the browser). A first line says what
// it ran on: the GPU the browser reports, the canvas, the pixel ratio, draw calls and triangles.
import type * as THREE from "three";
import type { Setting } from "./quality";
import type { DofSetting } from "./viewer";

const WARM = 1.5,
	SECONDS = 8,
	STALL = 50, // ms
	RUNS: readonly (readonly [Setting, DofSetting])[] = [
		["high", "auto"],
		["high", "all"],
		["medium", "auto"],
		["low", "auto"],
		["auto", "auto"],
	];

export type BenchTarget = {
	id: string;
	renderer: THREE.WebGLRenderer;
	stats: { cpu: number; pose: number; render: number; notes: string[] };
	setting(): Setting;
	setQuality(s: Setting): void;
	dof(): DofSetting;
	setDof(d: DofSetting): void;
	/** from the start of the default shot, playing at real time */
	restart(): void;
	/** the quality rendered at and the scale */
	describe(): string;
};

const f1 = (x: number) => x.toFixed(1),
	at = (a: number[], p: number) => {
		const s = [...a].sort((x, y) => x - y);
		return s[Math.min(s.length - 1, Math.floor(p * s.length))] ?? 0;
	},
	wait = (s: number) => new Promise((r) => setTimeout(r, s * 1000));

/** One run at setting, with depth of field as `dof`: its line. */
async function run(v: BenchTarget, setting: Setting, dof: DofSetting) {
	v.setQuality(setting);
	v.setDof(dof);
	v.restart();
	await wait(WARM);
	const ms: number[] = [],
		cpu: number[] = [],
		pose: number[] = [],
		render: number[] = [],
		stalls: string[] = [];
	let last: number | null = null;
	await new Promise<void>((done) => {
		const start = performance.now();
		const tick = (now: number) => {
			// after the viewer's frame (its callback was asked for first): its stats are this frame's
			const s = v.stats;
			if (last !== null) {
				const d = now - last;
				ms.push(d);
				cpu.push(s.cpu);
				pose.push(s.pose);
				render.push(s.render);
				if (d > STALL)
					stalls.push(
						`${Math.round(d)} ms: cpu ${Math.round(s.cpu)}${s.notes.length ? `, ${s.notes.join(", ")}` : s.cpu < STALL / 2 ? ", outside JS" : ""}`,
					);
			}
			last = now;
			if (now - start < SECONDS * 1000) requestAnimationFrame(tick);
			else done();
		};
		requestAnimationFrame(tick);
	});
	const mean = ms.reduce((a, x) => a + x, 0) / Math.max(ms.length, 1);
	return [
		`${setting}${dof === "all" ? " + focus" : ""}${setting === "auto" ? ` (ended ${v.describe()})` : ""}`,
		`${f1(1000 / mean)} fps mean`,
		`frame ms p50 ${f1(at(ms, 0.5))}, p95 ${f1(at(ms, 0.95))}, p99 ${f1(at(ms, 0.99))}`,
		`cpu ms p50 ${f1(at(cpu, 0.5))}, p95 ${f1(at(cpu, 0.95))} (pose p50 ${f1(at(pose, 0.5))}, render p50 ${f1(at(render, 0.5))})`,
		`stalls >${STALL} ms: ${stalls.length}${stalls.length ? ` [${stalls.slice(0, 6).join("; ")}${stalls.length > 6 ? "; ..." : ""}]` : ""}`,
	].join(" | ");
}

export async function bench(v: BenchTarget) {
	const was = v.setting(),
		focus = v.dof(),
		lines: string[] = [];
	for (const [s, d] of RUNS) lines.push(await run(v, s, d));
	const gl = v.renderer.getContext(),
		info = gl.getExtension("WEBGL_debug_renderer_info"),
		canvas = v.renderer.domElement,
		{ calls, triangles } = v.renderer.info.render;
	lines.unshift(
		[
			`replay bench ${v.id}`,
			String(info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)),
			`canvas ${canvas.width}x${canvas.height}, dpr ${devicePixelRatio}`,
			`${calls} calls, ${(triangles / 1e6).toFixed(2)}M tris`,
		].join(" | "),
	);
	v.setQuality(was);
	v.setDof(focus);
	const text = lines.join("\n");
	console.log(`BENCH ${text}`);
	Object.assign(window, { replayBench: text });
	show(text);
	return text;
}

function show(text: string) {
	const box = document.createElement("div"),
		area = document.createElement("textarea"),
		copy = document.createElement("button");
	box.style.cssText =
		"position:fixed;left:12px;right:12px;bottom:12px;z-index:9999;display:flex;gap:8px;padding:8px;background:#000c;border-radius:8px";
	area.readOnly = true;
	area.value = text;
	area.rows = 6;
	area.style.cssText =
		"flex:1;font:12px ui-monospace,monospace;color:#fff;background:transparent;border:0;resize:none;white-space:pre";
	copy.textContent = "Copy";
	copy.style.cssText = "padding:4px 12px;border-radius:6px;background:#fff;color:#000;font:600 13px system-ui";
	copy.onclick = () => void navigator.clipboard.writeText(text).then(() => (copy.textContent = "Copied"));
	box.append(area, copy);
	document.body.append(box);
}
