// Each moment's depth map, for the card's parallax where there is no orbit strip (lib/hover3d.ts): Depth Anything V2 (small, ONNX,
// on the CPU) over the broadcast frame, near white, at half the frame's size, into store/site/moments/<moment>/depth.png.
import { join } from "node:path";
import { pipeline, RawImage } from "@huggingface/transformers";
import { $ } from "bun";

const MOMENTS = join(import.meta.dir, "../../store/site/moments.json");
const THUMBS = join(import.meta.dir, "../../store/site/moments");
const MODEL = "onnx-community/depth-anything-v2-small";

const moments: number[] = await Bun.file(MOMENTS).json();
const estimate = await pipeline("depth-estimation", MODEL, { dtype: "fp32" });
for (const id of moments) {
	const frame = await RawImage.read(join(THUMBS, `${id}.jpg`)),
		out = await estimate(frame),
		{ depth } = Array.isArray(out) ? out[0]! : out;
	await $`mkdir -p ${join(THUMBS, String(id))}`;
	const half = await depth.resize(frame.width >> 1, frame.height >> 1);
	await half.save(join(THUMBS, String(id), "depth.png"));
	console.log(`${id}: ${half.width}×${half.height}`);
}
