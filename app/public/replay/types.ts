// A goal's reconstructed scene as pipeline/scene_export.py writes it (scene.json), and the 3D replays the server lists (/api/scenes).

export type Vec3 = [number, number, number];

export type Rink = {
	hx: number;
	hy: number;
	corner: number;
	wall: number;
	ledge: number;
	slope: number;
	stand_sd: number;
	goal_x: number;
	ice_res: number;
	out_res: number;
	perim: number;
	ice_size: [number, number];
	out_size: [number, number];
};

/** A planned camera: fx at the scene's width, R rows (rink to OpenCV camera axes), C the centre. */
export type Cam = { fx: number; R: [Vec3, Vec3, Vec3]; C: Vec3 };

/** A planned shot; one filmed from a player's eyes (own) has their head camera in the scene's heads. */
export type Shot = { frames: number[]; cams: Cam[]; hide?: number[]; own?: number | null };

export type Role = "scorer" | "passer" | "goalie";

/** Player `key`'s head camera at every source frame of the scene, by their part in the goal. */
export type Head = { key: number; role: Role; cams: Cam[] };

/** root: the mean ground position (mm) at each exported frame */
export type PersonMeta = {
	key: number;
	team: number;
	/** null for a player the tracking never identified */
	jersey: string | null;
	name: string | null;
	goalie: boolean;
	seen: boolean[];
	stick: boolean;
	root: [number, number][];
};

/** The body files' frames: exported frames first .. first + frames - 1, in `step` mm, encoded in chunks of `chunk` frames. */
export type BodyLayout = { first: number; frames: number; step: number; chunk: number };

export type SceneData = {
	fps: number;
	frame0: number;
	frames: number;
	vertices: number;
	faces: number;
	width: number;
	height: number;
	rink: Rink;
	people: PersonMeta[];
	body: BodyLayout;
	/** shot: the goal's frames (release, in the net, at rest), when the scene has one */
	puck: { xy: [number, number][]; show: number[]; shot?: { release: number; in_net: number; rest: number } };
	shots: Record<string, Shot>;
	heads?: Head[];
	anatomy: Anatomy | null;
	/** the cut sequence: parts of the shots, consecutive in time, each shots[shot]'s frames a to b (not b) */
	cuts?: { shot: string; frames: [number, number] }[];
	source: { start: number; camera: Vec3 };
};

/** One replay as /api/scenes lists it. */
export type SceneInfo = {
	id: string;
	/** video seconds of the scene's first and last source frame (the 2025 final's broadcast; other matches have their own) */
	from: number;
	to: number;
	scorer: Pick<PersonMeta, "name" | "jersey" | "team"> | null;
	shots: string[];
};

/** The goal's make-up (scene.json's anatomy): frames are exported frames, people their meta.key, lengths m. */
export type Anatomy = {
	release: number;
	in_net: number;
	pass_frame: number | null;
	passer: number | null;
	scorer: number;
	goalie: number | null;
	distance: number;
	/** degrees off the centre line */
	angle: number;
	from: [number, number];
	/** the frame the scorer takes the pass */
	received: number | null;
	/** the net's mouth, w by h cells of `cell` m, rows from the crossbar down, columns from the shooter's left: 1 open, 0 covered;
	 * share open. Null for a scramble at the crease. */
	open: { w: number; h: number; cell: number; grid: number[][]; share: number } | null;
};
