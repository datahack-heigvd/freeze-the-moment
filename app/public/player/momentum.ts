import { curveStepAfter, line, scaleLinear, select } from "d3";
import { COLOR, type Game, TEAM_COLOR } from "./events";

const H = 44;
const PAD = 8;
const CHIP_H = 16;

type Point = { t: number; v: number };

function cumulative(game: Game, team: "Home" | "Away"): Point[] {
	let total = 0;
	const points: Point[] = [{ t: 0, v: 0 }];
	for (const e of game.events) {
		if (e.team !== team || e.kind === "penalty" || e.xg == null) continue;
		total += e.xg;
		points.push({ t: e.t, v: total });
	}
	return points;
}

/** Cumulative expected goals per team along the same time axis as the scrub bar. */
export function createMomentum(svgEl: SVGSVGElement, game: Game) {
	const svg = select(svgEl);
	const series = (["Home", "Away"] as const).map((team) => ({ team, points: cumulative(game, team) }));
	svg
		.append("text")
		.attr("x", 0)
		.attr("y", 8)
		.attr("font-size", 10)
		.attr("font-weight", 600)
		.attr("fill", COLOR.paper)
		.attr("fill-opacity", 0.75)
		.attr("paint-order", "stroke")
		.attr("stroke", "rgb(0 0 0 / 0.55)")
		.attr("stroke-width", 3)
		.text("Expected goals");

	return function draw(edge: number) {
		const w = svgEl.clientWidth;
		if (!w) return;
		svg.attr("viewBox", `0 0 ${w} ${H}`);
		const visible = series.map((s) => {
			const points = s.points.filter((p) => p.t <= edge);
			const last = points.at(-1)!;
			return { ...s, points: [...points, { t: edge, v: last.v }] };
		});
		const max = Math.max(0.5, ...visible.map((s) => s.points.at(-1)!.v));
		const x = scaleLinear([0, edge], [0, w]);
		const y = scaleLinear([0, max], [H - 2, PAD]);

		const g = svg
			.selectAll<SVGGElement, (typeof visible)[number]>("g.s")
			.data(visible, (d) => d.team)
			.join((enter) => {
				const e = enter.append("g").attr("class", "s");
				e.append("path").attr("class", "halo");
				e.append("path").attr("class", "line");
				const chip = e.append("g").attr("class", "chip");
				chip.append("rect").attr("rx", 4).attr("height", CHIP_H).attr("fill", "rgb(16 17 20 / 0.92)");
				chip.append("text").attr("text-anchor", "middle").attr("font-size", 10.5).attr("font-weight", 600).attr("dy", 11.5);
				return e;
			});
		const color = (d: (typeof visible)[number]) => TEAM_COLOR[d.team];
		const stepLine = (d: (typeof visible)[number]) =>
			line<Point>()
				.curve(curveStepAfter)
				.x((p) => x(p.t))
				.y((p) => y(p.v))(d.points);
		g.select("path.halo")
			.attr("fill", "none")
			.attr("stroke", "rgb(0 0 0 / 0.45)")
			.attr("stroke-width", 4.5)
			.attr("stroke-linejoin", "round")
			.attr("d", stepLine);
		g.select("path.line")
			.attr("fill", "none")
			.attr("stroke", color)
			.attr("stroke-width", 1.75)
			.attr("stroke-linejoin", "round")
			.attr("d", stepLine);

		// End-of-line chips, centred on their line and pushed apart so they never overlap.
		const ends = visible.map((d) => y(d.points.at(-1)!.v));
		const hi = ends[0]! <= ends[1]! ? 0 : 1;
		const lo = 1 - hi;
		const tops = [0, 0];
		tops[hi] = ends[hi]! - CHIP_H / 2;
		tops[lo] = Math.max(ends[lo]! - CHIP_H / 2, tops[hi]! + CHIP_H + 2);
		const chip = g.select<SVGGElement>("g.chip").attr("transform", (_, i) => `translate(${w + 6},${tops[i]})`);
		chip
			.select("text")
			.attr("fill", color)
			.text((d) => d.points.at(-1)!.v.toFixed(2));
		chip.select("rect").attr("width", 34);
		chip.select("text").attr("x", 17);
	};
}
