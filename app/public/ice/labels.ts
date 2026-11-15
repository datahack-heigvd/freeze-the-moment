// The language's text, as DOM over the view (the page's type, crisp at any pixel ratio, untouched by the frame's look): each
// player's jersey tag over their head, pushed up out of the way of the tags nearer the camera (the puck carrier's with a white
// triangle; .ice-tag in styles.css), and the glass cards (Card in model.ts), the one style for numbers: the team's emblem, then a
// name and a kicker on one line and the numbers on the next (the big one first), the net's mouth to the right. A card is pinned
// over or under its point, or off to the side of it with a leader, towards `toward`; it comes in as its p goes from 0 to 1 (the
// leader drawn, the card wiped in, its numbers counting up). Elements are keyed, so only their place changes from frame to frame.
import { type Card, color, type Element, logo, type Project, PUCK } from "./model";

const HEAD = 2.05, // m up where a tag hangs
	GAP = 36, // px from a leader's point to its card
	NET = 2.6, // em high the net's mouth on a card
	TAG = { min: 11, max: 28 }, // px, the tags' type
	NARROW = 560; // px wide a view under which the cards are set smaller

/** px along a view's top and bottom that the page covers with its controls, where no card goes */
export type Keep = { top: number; bottom: number };

const ease = (p: number) => 1 - (1 - Math.min(Math.max(p, 0), 1)) ** 3;
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);
const el = <K extends keyof HTMLElementTagNameMap>(tag: K, css: string, parent?: HTMLElement, text = "") => {
	const e = document.createElement(tag);
	e.style.cssText = css;
	e.textContent = text;
	parent?.append(e);
	return e;
};
const PLACED = "position:absolute;left:0;top:0;will-change:transform;";
/** the dim type on a card */
export const DIM = "rgba(226,232,244,.62)";
/** A card's glass: dark and see-through, what is behind it blurred, a hairline round it. */
export const GLASS =
	"border-radius:12px;color:#fff;background:rgba(13,18,29,.72);-webkit-backdrop-filter:blur(14px) saturate(1.3);" +
	"backdrop-filter:blur(14px) saturate(1.3);box-shadow:0 8px 24px rgba(8,14,30,.3),inset 0 0 0 1px rgba(255,255,255,.1);";

/** A card's DOM, and how to fill and place it. Sized in em off one type size, smaller on a narrow view. */
function cardOf(root: HTMLElement, svg: SVGSVGElement, c: Card) {
	const box = el(
			"div",
			PLACED +
				GLASS +
				"z-index:1;display:flex;align-items:center;gap:.65em;padding:.55em .85em .6em .6em;white-space:nowrap;line-height:1.15;" +
				"font-variant-numeric:tabular-nums;",
			root,
		),
		mark = logo(c.team),
		badge = mark
			? el("span", "position:relative;flex:none;width:2.3em;height:2.3em;border-radius:50%;background:#fff;", box)
			: null,
		text = el("div", "", box),
		head = el("div", "display:flex;align-items:baseline;gap:.45em;", text),
		kicker = el(
			"span",
			`font-size:.72em;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:${c.team < 0 ? PUCK : color(c.team)};`,
			head,
			c.kicker,
		),
		line =
			c.value !== undefined || c.stats?.length
				? el("div", "display:flex;align-items:baseline;gap:.8em;margin-top:.15em;", text)
				: null,
		item = (size: number, unit: string) => {
			const s = el("span", "", line!),
				v = el("span", `font-size:${size}em;font-weight:700;letter-spacing:-.01em;`, s);
			el("span", `font-size:.8em;font-weight:500;margin-left:.25em;color:${DIM};`, s, unit);
			return v;
		},
		value = c.value !== undefined ? item(1.6, c.unit ?? "") : null,
		stats = c.stats?.map(([, name]) => item(1.15, name)),
		leader = document.createElementNS("http://www.w3.org/2000/svg", "path"),
		dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
	if (badge && mark) el("img", "position:absolute;inset:15%;width:70%;height:70%;object-fit:contain;", badge).src = mark;
	if (c.title) head.prepend(el("span", "font-size:1.15em;font-weight:700;letter-spacing:-.01em;", undefined, c.title));
	if (c.net) {
		// the net's mouth, NET em high beside the numbers: open in green, covered in red, under the posts and the bar
		const grid = c.net,
			cols = grid[0]!.length,
			rows = grid.length,
			cv = el("canvas", `height:${NET}em;aspect-ratio:${cols + 2}/${rows + 1};display:block;flex:none;`, box),
			x = cv.getContext("2d")!,
			k = 4; // canvas px a cell, drawn down to the CSS size
		cv.width = (cols + 2) * k;
		cv.height = (rows + 1) * k;
		for (const [r, cells] of grid.entries())
			for (const [j, o] of cells.entries()) {
				x.fillStyle = o ? "#2fbf68" : "#ef4b4b";
				x.fillRect((1 + j) * k, (1 + r) * k, k, k);
			}
		x.strokeStyle = "#fff";
		x.lineWidth = k * 0.9;
		x.beginPath();
		x.moveTo(k / 2, cv.height);
		x.lineTo(k / 2, k / 2);
		x.lineTo(cv.width - k / 2, k / 2);
		x.lineTo(cv.width - k / 2, cv.height);
		x.stroke();
	}
	for (const [a, v] of [
		["fill", "none"],
		["stroke", "rgba(13,18,29,.85)"],
		["stroke-width", "1.5"],
	])
		leader.setAttribute(a!, v!);
	for (const [a, v] of [
		["r", "6"],
		["fill", "none"],
		["stroke", PUCK],
		["stroke-width", "2.5"],
	])
		dot.setAttribute(a!, v!);
	if (c.anchor === "leader") svg.append(leader, dot);
	const count = (s: string, n: number) => s.replace(/\d+(\.\d+)?/, (m) => (Number(m) * n).toFixed(m.split(".")[1]?.length ?? 0));
	return {
		/**
		 * Fill it as c, `p` of the way in, pinned through P in a view `W` by `H` CSS px, at opacity `alpha`, kept `keep` px clear of the
		 * view's top and bottom.
		 */
		show(c: Card, P: Project, W: number, H: number, alpha: number, keep: Keep) {
			const at = P(c.x, c.y, c.z ?? 0),
				p = c.p;
			box.style.visibility = at && p > 0 ? "visible" : "hidden";
			leader.style.visibility = dot.style.visibility = box.style.visibility;
			if (!at || p <= 0) return;
			const size = `${W < NARROW ? 12 : 13.5}px`;
			if (box.style.fontSize !== size) box.style.fontSize = size;
			const n = ease(p),
				e = ease(p * 1.6),
				bw = box.offsetWidth,
				bh = box.offsetHeight,
				top = keep.top + 8,
				bottom = H - keep.bottom - 8;
			if (value) value.textContent = count(c.value!, n);
			for (const [k, [v]] of (c.stats ?? []).entries()) stats![k]!.textContent = count(v, n);
			kicker.style.opacity = `${ease(p * 2)}`;
			box.style.opacity = `${alpha * Math.min(Math.max((p - 0.15) * 3, 0), 1)}`;
			leader.style.opacity = dot.style.opacity = `${alpha}`;
			let x: number, y: number, clip: string;
			if (c.anchor === "leader") {
				const to = c.toward ? P(c.toward[0], c.toward[1], 0) : null,
					left = !!to && to[0] < at[0],
					up = !!to && at[1] < to[1],
					line = ease(p * 3);
				x = Math.min(Math.max(left ? at[0] - GAP - bw : at[0] + GAP, 8), W - bw - 8);
				y = Math.min(Math.max(up ? at[1] - GAP - bh : at[1] + GAP, top), bottom - bh);
				const ex = left ? x + bw : x,
					ey = up ? y + bh : y;
				dot.setAttribute("cx", `${at[0]}`);
				dot.setAttribute("cy", `${at[1]}`);
				leader.setAttribute("d", `M${at[0]},${at[1]}L${at[0] + (ex - at[0]) * line},${at[1] + (ey - at[1]) * line}`);
				y += (1 - e) * (up ? 10 : -10);
				clip = `inset(0 ${left ? 0 : (1 - e) * 100}% 0 ${left ? (1 - e) * 100 : 0}% round 12px)`;
			} else {
				x = Math.min(Math.max(at[0] - bw / 2, 8), W - bw - 8);
				y = Math.min(Math.max(c.anchor === "above" ? at[1] - bh - 10 : at[1] + 14, top), bottom - bh);
				y += (1 - e) * (c.anchor === "above" ? 8 : -8);
				clip = "inset(0 round 12px)";
			}
			box.style.transform = `translate(${x}px,${y}px)`;
			box.style.clipPath = clip;
		},
		remove() {
			box.remove();
			leader.remove();
			dot.remove();
		},
	};
}

/** Labels over a view in `root` (positioned, the view filling it), cards kept clear of `keep`: `place(elements, P)` each frame. */
export function createLabels(root: HTMLElement, keep: Keep = { top: 0, bottom: 0 }) {
	const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"),
		tags = new Map<string, HTMLElement>(),
		cards = new Map<string, ReturnType<typeof cardOf>>();
	svg.setAttribute("style", "position:absolute;inset:0;z-index:1;width:100%;height:100%;overflow:visible;pointer-events:none;");
	root.append(svg);
	return {
		/** Tags for the players, cards for the cards, through P (rink metres to CSS px over root), at opacity `alpha`. */
		place(elements: Element[], P: Project, alpha = 1) {
			const seen = new Set<string>(),
				W = root.clientWidth,
				H = root.clientHeight,
				placed: { l: number; r: number; t: number; b: number }[] = [],
				wanted: { key: string; x: number; y: number; size: number; text: string; team: number; carrier: boolean }[] = [];
			for (const e of elements) {
				if (e.kind === "player") {
					const head = P(e.x, e.y, HEAD),
						foot = P(e.x, e.y, 0);
					if (head && foot)
						wanted.push({
							key: e.key,
							x: head[0],
							y: head[1],
							size: Math.min(TAG.max, Math.max(TAG.min, Math.abs(foot[1] - head[1]) * 0.16)),
							text: e.tag,
							team: e.team,
							carrier: e.carrier,
						});
				} else if (e.kind === "card") {
					seen.add(`card ${e.key}`);
					let c = cards.get(e.key);
					if (!c) cards.set(e.key, (c = cardOf(root, svg, e)));
					c.show(e, P, W, H, alpha, keep);
				}
			}
			// nearest the camera keeps its place; anything it would cover moves up
			wanted.sort((a, b) => b.y - a.y);
			for (const t of wanted) {
				const w = t.size * (0.62 * t.text.length + 0.9),
					h = t.size * (t.carrier ? 2.4 : 1.75),
					box = { l: t.x - w / 2, r: t.x + w / 2, b: t.y, t: t.y - h };
				for (let n = 0; n < 6; n++) {
					const hit = placed.find((o) => box.l < o.r && box.r > o.l && box.t < o.b && box.b > o.t);
					if (!hit) break;
					const dy = box.b - hit.t + 2;
					box.b -= dy;
					box.t -= dy;
				}
				placed.push(box);
				seen.add(`tag ${t.key}`);
				let tag = tags.get(t.key);
				if (!tag) {
					tag = el("div", PLACED, root);
					tags.set(t.key, tag);
				}
				const html = `<div>${t.carrier ? "<u></u>" : ""}<b>${esc(t.text)}</b><i></i><s></s></div>`;
				if (tag.dataset.html !== html) {
					tag.innerHTML = html;
					tag.dataset.html = html;
				}
				tag.className = `ice-tag${t.team === 1 ? " away" : ""}`;
				tag.style.setProperty("--team", color(t.team));
				tag.style.setProperty("--lift", `${t.y - box.b}px`);
				tag.style.fontSize = `${t.size}px`;
				tag.style.opacity = `${alpha}`;
				tag.style.transform = `translate(${t.x}px,${t.y}px)`;
			}
			for (const [k, tag] of tags)
				if (!seen.has(`tag ${k}`)) {
					tag.remove();
					tags.delete(k);
				}
			for (const [k, c] of cards)
				if (!seen.has(`card ${k}`)) {
					c.remove();
					cards.delete(k);
				}
		},
		clear() {
			for (const t of tags.values()) t.remove();
			for (const c of cards.values()) c.remove();
			tags.clear();
			cards.clear();
		},
		dispose() {
			this.clear();
			svg.remove();
		},
	};
}
