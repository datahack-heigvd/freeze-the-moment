// The scene's meshes as they are built, drawn over the finished frame: the picture dimmed under a wireframe of every triangle the
// viewer draws, the hidden lines hidden (the surfaces' depth drawn first, pushed back a little so their own edges pass). The arena
// in thin cool lines (LINE), the people, their sticks and gear (the scene's children with userData.wire "figure") warm and fainter
// (FIGURE: their meshes are dense, so at a distance their lines shade them as a volume); what has userData.wire false is left out.
// It covers a band of the frame from one side to the other (`span`, fractions of its width), a line of light along each edge of it
// inside the frame (EDGE), as the film's scans: swept across it comes in, swept on it goes.
import * as THREE from "three";

const DIM = 0.88,
	LINE = { colour: new THREE.Color(0.62, 0.84, 1), opacity: 0.5 },
	FIGURE = { colour: new THREE.Color(1, 0.8, 0.45), opacity: 0.32 },
	EDGE = { width: 28, core: 1.5, colour: new THREE.Color(0.87, 0.95, 1) }; // px: the glow's half width, the line's

export class Wire {
	private depth = new THREE.MeshBasicMaterial({
		colorWrite: false,
		side: THREE.DoubleSide,
		polygonOffset: true,
		polygonOffsetFactor: 1,
		polygonOffsetUnits: 1,
	});
	private lines = new THREE.MeshBasicMaterial({
		wireframe: true,
		transparent: true,
		depthWrite: false,
		blending: THREE.AdditiveBlending,
	});
	private shade = new THREE.Mesh(
		new THREE.PlaneGeometry(2, 2),
		new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, depthTest: false, depthWrite: false }),
	);
	private flat = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
	/** a line of light down the frame at x (NDC), `px` NDC units a pixel across */
	private edge = new THREE.Mesh(
		new THREE.PlaneGeometry(2, 2),
		new THREE.ShaderMaterial({
			uniforms: { x: { value: 0 }, px: { value: 0 }, colour: { value: EDGE.colour } },
			vertexShader: /* glsl */ `varying vec2 vP; void main() { vP = position.xy; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
			fragmentShader: /* glsl */ `
uniform float x, px; uniform vec3 colour; varying vec2 vP;
void main() {
  float d = abs(vP.x - x) / px;
  float a = 0.5 * exp(-d * d / ${(EDGE.width * EDGE.width) / 4}.0) + 0.8 * clamp(${EDGE.core / 2} + 0.5 - d, 0.0, 1.0);
  gl_FragColor = vec4(colour * a, 1.0);
}`,
			transparent: true,
			depthTest: false,
			depthWrite: false,
			blending: THREE.AdditiveBlending,
		}),
	);

	constructor(private renderer: THREE.WebGLRenderer) {}

	/** Over what the renderer last drew to the screen: `scene` seen from `camera` as a wireframe, over the band `span` of the frame. */
	draw(scene: THREE.Scene, camera: THREE.Camera, [from, to]: readonly [number, number]) {
		const a = Math.max(from, 0),
			b = Math.min(to, 1);
		if (b <= a) return;
		const r = this.renderer,
			{ x: w, y: h } = r.getSize(new THREE.Vector2()),
			auto = r.autoClear,
			override = scene.overrideMaterial,
			background = scene.background;
		r.setRenderTarget(null);
		r.autoClear = false;
		r.setScissor(a * w, 0, (b - a) * w, h);
		r.setScissorTest(true);
		this.shade.material.opacity = DIM;
		r.render(this.shade, this.flat);
		r.clearDepth();
		const hidden: THREE.Object3D[] = [];
		scene.traverseVisible((o) => {
			if (o.userData.wire === false) hidden.push(o);
		});
		for (const o of hidden) o.visible = false;
		scene.background = null;
		scene.overrideMaterial = this.depth;
		r.render(scene, camera);
		scene.overrideMaterial = this.lines;
		const shown = scene.children.filter((o) => o.visible);
		for (const [part, look] of [
			[false, LINE],
			[true, FIGURE],
		] as const) {
			for (const o of shown) o.visible = (o.userData.wire === "figure") === part;
			this.lines.color.copy(look.colour);
			this.lines.opacity = look.opacity;
			r.render(scene, camera);
		}
		for (const o of shown) o.visible = true;
		scene.overrideMaterial = override;
		scene.background = background;
		for (const o of hidden) o.visible = true;
		r.setScissorTest(false);
		const u = this.edge.material.uniforms;
		u.px!.value = 2 / w;
		for (const e of [from, to]) {
			if (e <= 0 || e >= 1) continue;
			u.x!.value = 2 * e - 1;
			r.render(this.edge, this.flat);
		}
		r.autoClear = auto;
	}

	dispose() {
		this.depth.dispose();
		this.lines.dispose();
		this.shade.geometry.dispose();
		this.shade.material.dispose();
		this.edge.geometry.dispose();
		this.edge.material.dispose();
	}
}
