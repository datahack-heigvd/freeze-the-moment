// The site's data from the match (store/site, served under /site/): built by the pipeline from the organisers' feed and video, kept
// in the bucket and never in git. The small files are read once, before the first page draws; the feed itself (game.json) is
// fetched when a page needs it (media.ts).
import type { OrbitStrip } from "./hover3d";
import type { PlayerStats } from "./league";

const json = async <T>(name: string): Promise<T> => {
	const r = await fetch(`/site/${name}`);
	if (!r.ok) throw new Error(`/site/${name}: ${r.status}`);
	return r.json() as Promise<T>;
};

const [players, moments, orbits] = await Promise.all([
	json<PlayerStats[]>("players.json"),
	json<number[]>("moments.json"),
	json<Record<string, OrbitStrip>>("orbits.json"),
]);

/** Every skater's and goalie's numbers over the final (scripts/build-players.ts). */
export const playerStats = players;
/** The feed's events picked as moments: every goal, the biggest chances and the penalties (scripts/build-moments.ts). */
export const momentIds = moments;
/** The goals' orbit strips, by moment id (scripts/build-orbits.ts). */
export const orbitStrips = orbits;
