import { join } from "node:path";
import { csvParse } from "d3";

const SRC = join(
	import.meta.dir,
	"../../store/data/U.S. Collegiate Selects_HC Davos_Wisehockey_Match_Statistics_31_12_2025 All.csv",
);
const OUT = join(import.meta.dir, "../../store/site/players.json");

const TEAM: Record<string, string> = { "HC Davos": "davos", "U.S. Collegiate Selects": "uscs" };

const rows = csvParse((await Bun.file(SRC).text()).replace(/^﻿/, ""));
const num = (v: string | undefined) => Number(v || 0);

const players = rows.map((r) => ({
	id: `${TEAM[r.Team!]}-${r.Jersey}`,
	team: TEAM[r.Team!]!,
	jersey: r.Jersey!,
	name: `${r.FirstName} ${r.LastName}`,
	goals: num(r.G),
	assists: num(r.A),
	pim: num(r.PIM),
	plus: num(r["+"]),
	minus: num(r["-"]),
	shots: num(r.S),
	xg: num(r.xGsum),
	toi: num(r["TOI (s)"]),
	shifts: num(r.Shifts),
	distance: num(r["Dist (m)"]),
	topSpeed: num(r["SPD TOP (km/h)"]),
	faceoffsWon: num(r.FOW),
	faceoffs: num(r["No. FO"]),
	passes: num(r.Passes),
	passAttempts: num(r["All passes"]),
}));

await Bun.write(OUT, `${JSON.stringify(players)}\n`);
console.log(`${players.length} players`);
