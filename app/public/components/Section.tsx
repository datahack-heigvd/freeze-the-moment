import type { ReactNode } from "react";
import { Link } from "react-router";

/** Centered content column shared by every page. */
export function Container({ children, className = "" }: { children: ReactNode; className?: string }) {
	return <div className={`mx-auto w-full max-w-6xl px-4 sm:px-8 ${className}`}>{children}</div>;
}

type SectionProps = {
	title: string;
	hint?: string;
	more?: { to: string; label: string };
	children: ReactNode;
	className?: string;
};

export function Section({ title, hint, more, children, className = "" }: SectionProps) {
	return (
		<section className={`py-8 ${className}`}>
			<div className="mb-4 flex items-end justify-between gap-4">
				<div>
					<h2 className="text-lg font-semibold sm:text-xl">{title}</h2>
					{hint && <p className="mt-0.5 text-sm text-dim">{hint}</p>}
				</div>
				{more && (
					<Link to={more.to} className="shrink-0 text-sm font-medium text-dim transition-colors hover:text-paper">
						{more.label} <span aria-hidden>&rarr;</span>
					</Link>
				)}
			</div>
			{children}
		</section>
	);
}

export function PageTitle({ title, subtitle }: { title: string; subtitle?: string }) {
	return (
		<div className="pb-2 pt-8 sm:pt-12">
			<h1 className="text-3xl font-bold tracking-tight sm:text-4xl">{title}</h1>
			{subtitle && <p className="mt-2 max-w-xl text-dim">{subtitle}</p>}
		</div>
	);
}

export function Empty({ children }: { children: ReactNode }) {
	return <p className="panel px-5 py-8 text-center text-sm text-dim">{children}</p>;
}
