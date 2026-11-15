import { type TeamId, team } from "../lib/league";

type Props = { id: TeamId; className?: string; plain?: boolean };

/** A team's emblem. Most are drawn for white backgrounds, so by default they sit on a white disc. */
export function TeamLogo({ id, className = "size-9", plain = false }: Props) {
	const t = team(id);
	return (
		<span className={`relative inline-block shrink-0 ${plain ? "" : "rounded-full bg-white"} ${className}`}>
			<img
				src={t.logo}
				alt={`${t.name} emblem`}
				className={`absolute object-contain ${plain ? "inset-0 size-full" : "inset-[14%] size-[72%]"}`}
			/>
		</span>
	);
}
