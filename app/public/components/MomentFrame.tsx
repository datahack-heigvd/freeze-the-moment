import { type ReactNode, useEffect, useRef } from "react";
import { hover3d, stillUrl, stripOf } from "../lib/hover3d";
import type { Moment } from "../lib/media";
import { afterScene, thrifty } from "../lib/priority";
import { prefetchScene } from "../replay/scene";

/**
 * A moment's frame on a card, a window on its baked 3D world under the pointer (lib/hover3d.ts): a goal rebuilt in 3D shows its
 * render, any other moment its broadcast frame. `children` are laid over it. With `scene`, the goal's 3D replay, resting the pointer
 * on it fetches that scene ahead (after any replay already loading, and not for a viewer saving data or on a slow connection),
 * so opening it is quick.
 */
export function MomentFrame({
	moment,
	scene,
	className = "",
	children,
}: {
	moment: Moment;
	scene?: string;
	className?: string;
	children?: ReactNode;
}) {
	const frame = useRef<HTMLDivElement>(null),
		layer = useRef<HTMLDivElement>(null);
	useEffect(() => {
		const f = frame.current!;
		// one prefetch a hover, dropped if the pointer leaves before it is in
		let ahead: { abort: AbortController; done: boolean } | null = null;
		const fetchAhead = (id: string) => {
				if (ahead && !ahead.abort.signal.aborted) return;
				const job = { abort: new AbortController(), done: false };
				ahead = job;
				void afterScene()
					.then(() => (job.abort.signal.aborted ? Promise.reject(job.abort.signal.reason) : prefetchScene(id, job.abort.signal)))
					.then(
						() => {
							job.done = true;
						},
						() => {},
					);
			},
			leave = () => {
				if (ahead && !ahead.done) ahead.abort.abort();
			},
			stop = hover3d(moment.id, {
				frame: f,
				layer: layer.current!,
				thumb: moment.thumb,
				onIntent: scene && !thrifty() ? () => fetchAhead(scene) : undefined,
			});
		f.addEventListener("pointerleave", leave);
		return () => {
			f.removeEventListener("pointerleave", leave);
			leave();
			stop();
		};
	}, [moment.id, moment.thumb, scene]);
	return (
		<div
			ref={frame}
			className={`relative aspect-video overflow-hidden rounded-xl bg-panel shadow-[inset_0_0_0_1px_var(--hairline)] ${className}`}
		>
			<img
				src={stripOf(moment.id) ? stillUrl(moment.id) : moment.thumb}
				alt=""
				loading="lazy"
				decoding="async"
				fetchPriority="low"
				className="size-full object-cover"
			/>
			<div ref={layer} className="absolute inset-0" />
			{children}
		</div>
	);
}
