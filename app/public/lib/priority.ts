// The open replay's scene comes first on the network: card pictures and prefetches wait until its first part is in.
let loading: Promise<void> = Promise.resolve();

/** Mark `opening` as the open replay's load: card fetches wait for it. */
export function sceneLoading(opening: Promise<unknown>) {
	const done = opening.then(
		() => {},
		() => {},
	);
	loading = done;
}

/** Settles once the open replay's scene is in (at once if none is loading). */
export const afterScene = () => loading;

/** The viewer asked to save data, or the connection is slow (2g or 3g): nothing is fetched ahead for it. */
export function thrifty() {
	const c = (navigator as Navigator & { connection?: { saveData?: boolean; effectiveType?: string } }).connection;
	return !!c && (!!c.saveData || /2g|3g/.test(c.effectiveType ?? ""));
}
