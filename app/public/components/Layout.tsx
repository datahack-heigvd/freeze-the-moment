import { Suspense } from "react";
import { Link, NavLink, Outlet, ScrollRestoration } from "react-router";
import { Container } from "./Section";

const LINKS = [
	["/", "3D Replays", "Replays"],
	["/final", "The final", "Final"],
	["/how", "How it works", "How"],
] as const;

export function Layout() {
	return (
		<>
			<Header />
			<main className="min-h-[60svh]">
				<Suspense
					fallback={
						<div
							className="mx-auto mt-24 size-6 animate-spin rounded-full border-2 border-white/20 border-t-paper"
							role="status"
							aria-label="Loading"
						/>
					}
				>
					<Outlet />
				</Suspense>
			</main>
			<Footer />
			<ScrollRestoration />
		</>
	);
}

function Header() {
	return (
		<header className="sticky top-0 z-40 border-b border-hairline bg-ink/90 backdrop-blur">
			<Container className="flex h-(--header-h) items-center gap-3 sm:gap-8">
				<Link to="/" className="flex shrink-0 items-center gap-3" aria-label="Freeze the moment, home">
					<img src="/assets/spengler-cup.svg" alt="" className="h-7 w-auto sm:h-8" />
					<span className="hidden text-sm font-semibold tracking-tight lg:inline">Freeze the moment</span>
				</Link>
				<nav className="tabs -mr-4 min-w-0 self-stretch overflow-x-auto [scrollbar-width:none]" aria-label="Main">
					{LINKS.map(([to, label, short]) => (
						<NavLink key={to} to={to} end={to === "/"} className="whitespace-nowrap max-sm:!px-2.5">
							<span className="max-sm:hidden">{label}</span>
							<span className="sm:hidden">{short}</span>
						</NavLink>
					))}
				</nav>
			</Container>
		</header>
	);
}

function Footer() {
	return (
		<footer className="mt-16 border-t border-hairline py-10 text-sm text-dim">
			<Container className="flex flex-col gap-2">
				<p className="text-pretty">
					Freeze the moment: the Spengler Cup 2025 final's goals rebuilt in 3D from the TV broadcast alone. A prototype for the
					Data-hack 2026 challenge.
				</p>
				<p className="text-pretty">
					Broadcast and match data (Wisehockey) provided by the organisers for the challenge. Tournament data: Wikipedia and
					spenglercup.ch. Logos belong to their owners.
				</p>
			</Container>
		</footer>
	);
}
