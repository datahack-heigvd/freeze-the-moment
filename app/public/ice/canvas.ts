// The shapes (shapes.ts) filled on a 2D canvas, whose own smoothing does their edges: each shape one path of its loops, filled.
import type { Shape } from "./shapes";

const css = ([r, g, b, a]: Shape["rgba"]) => `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},${a})`;

/** Fill `shapes` on ctx (in CSS px under its transform): those on the ice, the ones in the air, or all. */
export function drawShapes(ctx: CanvasRenderingContext2D, shapes: Shape[], which: "ice" | "air" | "all" = "all") {
	for (const s of shapes) {
		if (which !== "all" && s.ice !== (which === "ice")) continue;
		ctx.beginPath();
		for (const loop of s.loops)
			loop.forEach((i, k) => {
				const v = s.v[i]!,
					x = v.x + v.nx * v.s * v.h,
					y = v.y + v.ny * v.s * v.h;
				if (k) ctx.lineTo(x, y);
				else ctx.moveTo(x, y);
			});
		ctx.closePath();
		ctx.fillStyle = css(s.rgba);
		ctx.fill();
	}
}
