import { DIM, GLASS } from "../ice/labels";
import { logo } from "../ice/model";
import { eventDetails, eventTitle, type Game, type GameEvent, player, TEAM_COLOR } from "./events";

const DURATION = { goal: 8, penalty: 5 } as const;

/** An event as a glass card, the overlay's one style (ice/labels.ts): the team's emblem, the player's name with what happened as
 * a kicker in the team's colour, the details under it; for a goal, the score it made to the right, the scoring side's number in
 * its colour. */
function html(e: GameEvent, game: Game): string {
	const color = TEAM_COLOR[e.team],
		mark = logo(e.team === "Away" ? 1 : 0),
		p = player(e.player);
	let score = "";
	if (e.kind === "goal") {
		const done = game.events.filter((g) => g.kind === "goal" && g.t <= e.t);
		const home = done.filter((g) => g.team === "Home").length;
		const digit = (n: number, team: "Home" | "Away") => `<span style="${team === e.team ? `color:${color}` : ""}">${n}</span>`;
		score = `<div style="flex:none;margin-left:.6em;font-size:1.9em;font-weight:700;letter-spacing:-.01em">${digit(home, "Home")}<span style="margin:0 .2em;font-weight:400;color:${DIM}">–</span>${digit(done.length - home, "Away")}</div>`;
	}
	return `<div style="${GLASS}display:flex;align-items:center;gap:.75em;padding:.65em 1.1em .7em .7em;font-size:15px;line-height:1.2;font-variant-numeric:tabular-nums">
		${mark ? `<span style="position:relative;flex:none;width:2.6em;height:2.6em;border-radius:50%;background:#fff"><img src="${mark}" alt="" style="position:absolute;inset:15%;width:70%;height:70%;object-fit:contain"></span>` : ""}
		<div style="min-width:0;flex:1">
			<p style="display:flex;align-items:baseline;gap:.5em"><span class="truncate" style="font-size:1.25em;font-weight:700;letter-spacing:-.01em">${p.name}</span><span style="font-size:.72em;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:${color}">${eventTitle(e)}</span></p>
			<p class="sm:truncate" style="margin-top:.2em;font-size:.86em;font-weight:500;color:${DIM}">${[p.num && `#${p.num}`, ...eventDetails(e)].filter(Boolean).join(" · ")}</p>
		</div>
		${score}
	</div>`;
}

/** Lower-third announcement for goals and penalties, shown for a few seconds after each one. */
export function createBanner(root: HTMLElement, game: Game) {
	const events = game.events.filter((e): e is GameEvent & { kind: "goal" | "penalty" } => e.kind !== "shot");
	let shown: GameEvent | undefined;
	return function update(t: number) {
		const e = events.findLast((ev) => t >= ev.t && t - ev.t < DURATION[ev.kind]);
		if (e === shown) return;
		shown = e;
		if (e) root.innerHTML = html(e, game);
		root.classList.toggle("show", !!e);
	};
}
