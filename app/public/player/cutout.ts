/** Outlines of everybody on the ice, per frame (see pipeline/calib/scripts/export_masks.py): ice-painted graphics are erased where a person stands in front of them. */
export type Cutout = { offsets: Uint32Array; data: Int16Array };

export function parseCutout(buf: ArrayBuffer): Cutout {
	const n = new Uint32Array(buf, 0, 1)[0]!;
	return { offsets: new Uint32Array(buf, 4, n + 1), data: new Int16Array(buf, 4 * (n + 2)) };
}

/** Erase the outlines of frame i from what is already drawn on ctx (video pixel coordinates). */
export function eraseOutlines(ctx: CanvasRenderingContext2D, c: Cutout, i: number) {
	const a = c.offsets[i];
	const b = c.offsets[i + 1];
	if (a === undefined || b === undefined || a === b) return;
	ctx.beginPath();
	for (let p = a; p < b; ) {
		const n = c.data[p++]!;
		ctx.moveTo(c.data[p]! / 8, c.data[p + 1]! / 8);
		for (let j = 1; j < n; j++) ctx.lineTo(c.data[p + 2 * j]! / 8, c.data[p + 2 * j + 1]! / 8);
		ctx.closePath();
		p += 2 * n;
	}
	ctx.globalCompositeOperation = "destination-out";
	ctx.fillStyle = "#000";
	ctx.fill();
	ctx.strokeStyle = "#000";
	ctx.lineWidth = 1.5;
	ctx.stroke();
	ctx.globalCompositeOperation = "source-over";
}
