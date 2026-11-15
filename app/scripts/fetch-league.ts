import { join } from "node:path";
import { $ } from "bun";

/**
 * Snapshot of the real tournament data the site shows around the one game we have video for:
 * the 2025 Spengler Cup (results and standings, from Wikipedia) and the 2026 schedule (spenglercup.ch).
 * It runs once and the output is committed, so the site never depends on the network at runtime.
 */
const PUBLIC = join(import.meta.dir, "../public");
const OUT = join(PUBLIC, "data/league.json");
const UA = "Mozilla/5.0 (spengler-fan-site prototype)";

const WIKI = "https://en.wikipedia.org/w/index.php?title=2025_Spengler_Cup&action=raw";
const SCHEDULE = "https://www.spenglercup.ch/en/tournament/games-standing";
const HOME = "https://www.spenglercup.ch/en";
const CDN = "https://cdn.spenglercup.ch/sites/default/files/team";
const COMMONS = "https://upload.wikimedia.org/wikipedia";

type Team = { id: string; name: string; short: string; country: string; logo: string; source: string };

const TEAMS: Team[] = [
	{
		id: "davos",
		name: "HC Davos",
		short: "DAV",
		country: "SUI",
		logo: "davos.png",
		source: `${CDN}/2026-01/HCDavos_darkBackground_0.png`,
	},
	{
		id: "uscs",
		name: "U.S. Collegiate Selects",
		short: "USA",
		country: "USA",
		logo: "uscs.png",
		source: `${CDN}/2026-03/SC-NCAA-Wappen-Logo-rot.png`,
	},
	{
		id: "canada",
		name: "Team Canada",
		short: "CAN",
		country: "CAN",
		logo: "canada.png",
		source: `${COMMONS}/en/thumb/5/5f/Hockey_Canada.svg/500px-Hockey_Canada.svg.png`,
	},
	{
		id: "fribourg",
		name: "HC Fribourg-Gottéron",
		short: "FRI",
		country: "SUI",
		logo: "fribourg.png",
		source: `${COMMONS}/en/thumb/a/a4/HC_Fribourg-Gott%C3%A9ron_logo.svg/500px-HC_Fribourg-Gott%C3%A9ron_logo.svg.png`,
	},
	{
		id: "sparta",
		name: "HC Sparta Praha",
		short: "SPA",
		country: "CZE",
		logo: "sparta.png",
		source: `${COMMONS}/commons/thumb/2/2b/HC_Sparta_Praha_logo.svg/500px-HC_Sparta_Praha_logo.svg.png`,
	},
	{
		id: "helsinki",
		name: "IFK Helsinki",
		short: "HEL",
		country: "FIN",
		logo: "helsinki.png",
		source: `${COMMONS}/commons/thumb/2/21/HIFK-J%C3%A4%C3%A4kiekko.svg/500px-HIFK-J%C3%A4%C3%A4kiekko.svg.png`,
	},
	{
		id: "scl-tigers",
		name: "SCL Tigers",
		short: "SCL",
		country: "SUI",
		logo: "scl-tigers.png",
		source: `${CDN}/2026-01/SCLTigers.png`,
	},
	{
		id: "mannheim",
		name: "Adler Mannheim",
		short: "MAN",
		country: "GER",
		logo: "mannheim.png",
		source: `${CDN}/2026-05/ADLER_Logo_4C_2015_out.png`,
	},
	{
		id: "ilves",
		name: "Ilves Tampere",
		short: "ILV",
		country: "FIN",
		logo: "ilves.png",
		source: `${CDN}/2026-05/ilves-official-logo.png`,
	},
	{
		id: "frolunda",
		name: "Frölunda HC",
		short: "FRO",
		country: "SWE",
		logo: "frolunda.png",
		source: `${CDN}/2026-02/Frolunda-HC-RGB.png`,
	},
];

const fetchText = async (url: string) => {
	const r = await fetch(url, { headers: { "User-Agent": UA } });
	if (!r.ok) throw new Error(`${url}: ${r.status}`);
	return r.text();
};

const teamByName = (name: string) => {
	const n = name.toLowerCase();
	return TEAMS.find(
		(t) =>
			n.includes(t.name.toLowerCase().replace(/^hc /, "")) ||
			(t.id === "uscs" && n.includes("collegiate")) ||
			(t.id === "helsinki" && n.includes("helsinki")),
	);
};

const MONTHS = [
	"january",
	"february",
	"march",
	"april",
	"may",
	"june",
	"july",
	"august",
	"september",
	"october",
	"november",
	"december",
];
const isoDate = (s: string) => {
	const [d, m, y] = s.trim().split(/\s+/);
	return `${y}-${String(MONTHS.indexOf(m!.toLowerCase()) + 1).padStart(2, "0")}-${d!.padStart(2, "0")}`;
};
const wikiText = (s: string) =>
	s
		.replace(/\{\{[^}]*\}\}/g, "")
		.replace(/\[\[(?:[^\]|]*\|)?([^\]]*)\]\]/g, "$1")
		.replace(/'''?/g, "")
		.trim();

const pair = (s: string): [number, number] => {
	const m = /(\d+)\D+(\d+)/.exec(s);
	return [Number(m![1]), Number(m![2])];
};

async function parseWikipedia() {
	const wiki = await fetchText(WIKI);

	const standings = [
		...wiki.matchAll(/===Group \[\[[^\]|]*\|(\w+)\]\]===\s*\{\{#invoke:Sports table\|main([\s\S]*?)\n\}\}/g),
	].map(([, group, body]) => {
		const field = (key: string, code: string) => Number(new RegExp(`\\b${key}_${code}\\s*=\\s*(\\d+)`).exec(body!)?.[1] ?? 0);
		const codes = /team_order\s*=\s*([\w, ]+)/
			.exec(body!)![1]!
			.split(",")
			.map((c) => c.trim());
		const rows = codes.map((code, i) => {
			const name = wikiText(new RegExp(`name_${code}\\s*=\\s*(.*)`).exec(body!)![1]!);
			const [w, otw, otl, l, gf, ga] = [
				field("win", code),
				field("OTwin", code),
				field("OTloss", code),
				field("loss", code),
				field("gf", code),
				field("ga", code),
			];
			const result = new RegExp(`result${i + 1}\\s*=\\s*(\\w+)`).exec(body!)?.[1];
			return {
				team: teamByName(name)!.id,
				pld: w + otw + otl + l,
				w,
				otw,
				otl,
				l,
				gf,
				ga,
				pts: 3 * w + 2 * otw + otl,
				advance: result === "SF" ? "Semifinals" : "Quarterfinals",
			};
		});
		return { group: group!, rows };
	});

	const stageAt = (index: number) => {
		const heading = [...wiki.slice(0, index).matchAll(/\n===([^=\n]+)===/g)].at(-1)?.[1] ?? "";
		return /Group/.test(heading)
			? "Group stage"
			: /Quarter/.test(heading)
				? "Quarterfinal"
				: /Semi/.test(heading)
					? "Semifinal"
					: "Final";
	};

	const games = [...wiki.matchAll(/\{\{Hockeybox2([\s\S]*?)\n\}\}/g)].map((m) => {
		const field = (k: string) => new RegExp(`\\|\\s*${k}\\s*=([^\\n]*)`).exec(m[1]!)?.[1]?.trim() ?? "";
		const [home, away] = [teamByName(wikiText(field("team1")))!, teamByName(wikiText(field("team2")))!];
		const optional = (a: string, b: string): [number, number] | undefined =>
			field(a) && field(b) ? [Number(field(a)), Number(field(b))] : undefined;
		return {
			stage: stageAt(m.index!),
			date: isoDate(field("date")),
			time: field("time").replace(/\D+/, ":"),
			home: home.id,
			away: away.id,
			score: pair(field("score")),
			periods: [...field("periods").matchAll(/(\d+)–(\d+)/g)].map((p) => [Number(p[1]), Number(p[2])] as [number, number]),
			shots: optional("shots1", "shots2"),
			pim: optional("penalties1", "penalties2"),
		};
	});
	games.sort((a, b) => `${a.date} ${a.time}`.localeCompare(`${b.date} ${b.time}`));
	return { standings, games: games.map((g, i) => ({ id: `g${i + 1}`, no: i + 1, ...g })) };
}

async function parseSchedule() {
	const html = await fetchText(SCHEDULE);
	const rows = [...html.matchAll(/<tr class="transition-colors rounded-xl[^>]*cursor-pointer">([\s\S]*?)<\/tr>/g)].map(
		(m) => m[1]!,
	);
	const strip = (s: string) => s.replace(/<[^>]+>/g, "").trim();
	const slot = (name: string) => {
		const team = TEAMS.find((t) => t.name === name);
		return team ? { team: team.id } : { label: name };
	};
	return rows.map((row, i) => {
		const cells = [...row.matchAll(/<td[^>]*>([\s\S]*?)<\/td>/g)].map((c) => strip(c[1]!));
		const [dd, mm, yyyy] = cells[0]!.split(".");
		const names = [...row.matchAll(/alt="([^"]*) Logo"/g)].map((a) => a[1]!.replace(/&amp;/g, "&"));
		return {
			id: `f${i + 1}`,
			no: i + 1,
			date: `${yyyy}-${mm}-${dd}`,
			time: cells[1]!,
			home: slot(names[0]!),
			away: slot(names[1]!),
		};
	});
}

async function downloadLogos() {
	const dir = join(PUBLIC, "assets/teams");
	await $`mkdir -p ${dir}`;
	for (const t of TEAMS) {
		const file = join(dir, t.logo);
		if (await Bun.file(file).exists()) continue;
		const res = await fetch(t.source, { headers: { "User-Agent": UA } });
		if (!res.ok) throw new Error(`${t.source}: ${res.status}`);
		await Bun.write(file, res);
		// emblems are shown at most ~300px wide
		await $`ffmpeg -loglevel error -y -i ${file} -vf scale='min(320,iw)':'min(320,ih)':force_original_aspect_ratio=decrease,format=rgba ${file}.tmp.png`.quiet();
		await $`mv ${file}.tmp.png ${file}`;
	}
	// the tournament wordmark is inline SVG in the site header
	const header = /<svg[^>]*><title>SpenglerCup-big<\/title>[\s\S]*?<\/svg>/.exec(await fetchText(HOME))![0];
	await Bun.write(
		join(PUBLIC, "assets/spengler-cup.svg"),
		header
			.replace(/ width="\d+" height="\d+"/, "")
			.replace("<svg ", '<svg xmlns:xlink="http://www.w3.org/1999/xlink" ')
			.replace("SpenglerCup-big", "Spengler Cup"),
	);
}

const [{ standings, games }, fixtures] = await Promise.all([parseWikipedia(), parseSchedule(), downloadLogos()]);

const league = {
	fetchedAt: new Date().toISOString().slice(0, 10),
	sources: [WIKI.replace("/w/index.php?title=", "/wiki/").replace("&action=raw", ""), SCHEDULE],
	teams: Object.fromEntries(TEAMS.map(({ source, logo, ...t }) => [t.id, { ...t, logo: `/assets/teams/${logo}` }])),
	edition: { name: "Spengler Cup 2025", venue: "Eisstadion Davos", from: "2025-12-26", to: "2025-12-31", champion: "davos" },
	standings,
	games,
	next: { name: "Spengler Cup 2026", fixtures },
};
await Bun.write(OUT, `${JSON.stringify(league, null, "\t")}\n`);
console.log(`${games.length} games, ${standings.length} groups, ${fixtures.length} fixtures`);
