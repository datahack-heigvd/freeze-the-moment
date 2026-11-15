import { Link } from "react-router";
import { Container } from "../components/Section";

export function NotFound() {
	return (
		<Container className="py-24 text-center">
			<p className="text-6xl font-bold text-dim">404</p>
			<h1 className="mt-4 text-xl font-semibold">We could not find that page</h1>
			<Link to="/" className="btn btn-primary mt-8">
				Back to the 3D replays
			</Link>
		</Container>
	);
}
