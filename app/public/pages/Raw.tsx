import { useEffect, useRef } from "react";
import { liveEdge, syncClock } from "../lib/clock";

/**
 * The broadcast alone: the match video filling the screen with the browser's own controls, and no overlay or header. It opens at
 * the live edge, muted (browsers only autoplay silent video), and is then the viewer's to pause, seek and unmute.
 */
export function RawPage() {
	const video = useRef<HTMLVideoElement>(null);

	useEffect(() => {
		const v = video.current!;
		syncClock().then(() => {
			v.currentTime = liveEdge();
			v.play().catch(() => {});
		});
	}, []);

	return (
		<video
			ref={video}
			src="/video"
			controls
			muted
			playsInline
			preload="auto"
			className="fixed inset-0 size-full bg-black object-contain"
		/>
	);
}
