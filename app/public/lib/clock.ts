import { useEffect, useState } from "react";

type Clock = { startedAt: number; startOffset: number; skew: number };
let clock: Clock | undefined;

/** Reads the server's broadcast clock and the offset between its time and ours. */
export async function syncClock() {
	const t0 = Date.now();
	const r: { startedAt: number; startOffset: number; now: number } = await (
		await fetch("/api/clock", { cache: "no-store" })
	).json();
	clock = { startedAt: r.startedAt, startOffset: r.startOffset, skew: r.now + (Date.now() - t0) / 2 - Date.now() };
}

/** Video second the live broadcast is at right now; 0 until the clock has synced. */
export const liveEdge = () => (clock ? clock.startOffset + (Date.now() + clock.skew - clock.startedAt) / 1000 : 0);

/** The live edge as React state, refreshed every second while `active`. */
export function useLiveEdge(active: boolean): number {
	const [edge, setEdge] = useState(liveEdge);
	useEffect(() => {
		if (!active) return;
		let stale = false;
		syncClock().then(() => stale || setEdge(liveEdge()));
		const timer = setInterval(() => setEdge(liveEdge()), 1000);
		return () => {
			stale = true;
			clearInterval(timer);
		};
	}, [active]);
	return edge;
}
