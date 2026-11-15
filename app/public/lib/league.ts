import leagueJson from "../data/league.json";

export type TeamId =
	| "davos"
	| "uscs"
	| "canada"
	| "fribourg"
	| "sparta"
	| "helsinki"
	| "scl-tigers"
	| "mannheim"
	| "ilves"
	| "frolunda";
export type Team = { id: TeamId; name: string; short: string; country: string; logo: string };

export type LeagueGame = {
	id: string;
	no: number;
	stage: "Group stage" | "Quarterfinal" | "Semifinal" | "Final";
	date: string;
	time: string;
	home: TeamId;
	away: TeamId;
	score: [number, number];
	periods: [number, number][];
	shots?: [number, number];
	pim?: [number, number];
};
type League = {
	fetchedAt: string;
	sources: string[];
	teams: Record<TeamId, Team>;
	edition: { name: string; venue: string; from: string; to: string; champion: TeamId };
	games: LeagueGame[];
};

export type PlayerStats = {
	id: string;
	team: TeamId;
	jersey: string;
	name: string;
	goals: number;
	assists: number;
	pim: number;
	plus: number;
	minus: number;
	shots: number;
	xg: number;
	toi: number;
	shifts: number;
	distance: number;
	topSpeed: number;
	faceoffsWon: number;
	faceoffs: number;
	passes: number;
	passAttempts: number;
};

export const league = leagueJson as unknown as League;

export const team = (id: TeamId): Team => league.teams[id];
/** The one game with video and tracking behind it: the 2025 final. */
export const featuredGame = league.games.find((g) => g.id === "g11")!;

/** game.json calls the first-listed side "Home" and the second "Away". */
export const SIDE = { Home: featuredGame.home, Away: featuredGame.away } as const;

const fmt = (opts: Intl.DateTimeFormatOptions) => new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", ...opts });
const at = (iso: string) => new Date(`${iso}T12:00:00Z`);
export const formatDate = (iso: string) => fmt({ day: "numeric", month: "short", year: "numeric" }).format(at(iso));

const pad = (n: number) => String(Math.floor(n)).padStart(2, "0");
export const minutes = (s: number) => `${Math.floor(s / 60)}:${pad(s % 60)}`;
