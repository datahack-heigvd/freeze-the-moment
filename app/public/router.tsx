import { createBrowserRouter } from "react-router";
import { Layout } from "./components/Layout";
import { FinalPage, Watch } from "./pages/Final";
import { Recap, Stats } from "./pages/FinalTabs";
import { HowPage } from "./pages/How";
import { NotFound } from "./pages/NotFound";
import { RawPage } from "./pages/Raw";
import { ReplayPage } from "./pages/Replay";

export const router = createBrowserRouter([
	// the broadcast alone, live, full screen
	{ path: "raw", Component: RawPage },
	{
		Component: Layout,
		children: [
			{ index: true, Component: ReplayPage },
			{ path: "replay/:id", Component: ReplayPage },
			{
				path: "final",
				Component: FinalPage,
				children: [
					{ index: true, Component: Watch },
					{ path: "recap", Component: Recap },
					{ path: "stats", Component: Stats },
				],
			},
			{ path: "how", Component: HowPage },
			{ path: "*", Component: NotFound },
		],
	},
]);
