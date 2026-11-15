import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const VIDEO = join(import.meta.dir, "../../../store/data/20251231_115035_SpenglerCup_Kam1 31.12.25 Game 11.mp4");
const KF = new URL("../labels/keyframes.json", import.meta.url).pathname;
const FRAMES = new URL("./frames/", import.meta.url).pathname;
const load = (): any[] => (existsSync(KF) ? JSON.parse(readFileSync(KF, "utf8")) : []);

async function frame(t: number) {
  const path = `${FRAMES}${t.toFixed(2)}.jpg`;
  if (!existsSync(path)) {
    const p = Bun.spawn(["ffmpeg", "-v", "error", "-ss", String(t), "-i", VIDEO, "-frames:v", "1", "-vf", "scale=1920:1080", "-q:v", "3", "-y", path]);
    await p.exited;
  }
  return new Response(Bun.file(path), { headers: { "cache-control": "max-age=3600" } });
}

Bun.serve({
  port: 3100,
  async fetch(req) {
    const u = new URL(req.url);
    if (u.pathname === "/") return new Response(Bun.file(new URL("./index.html", import.meta.url).pathname));
    if (u.pathname === "/rink.json") return new Response(Bun.file(new URL("./rink.json", import.meta.url).pathname));
    if (u.pathname === "/frame") return frame(Math.round(Number(u.searchParams.get("t")) * 100) / 100);
    if (u.pathname === "/api/keyframes" && req.method === "GET") return Response.json(load());
    if (u.pathname === "/api/keyframes" && req.method === "POST") {
      const k = await req.json();
      const all = load().filter((x) => Math.abs(x.t - k.t) > 0.005);
      all.push(k); all.sort((a, b) => a.t - b.t);
      writeFileSync(KF, JSON.stringify(all, null, 1));
      return Response.json(all);
    }
    if (u.pathname === "/api/keyframes" && req.method === "DELETE") {
      const t = Number(u.searchParams.get("t"));
      const all = load().filter((x) => Math.abs(x.t - t) > 0.005);
      writeFileSync(KF, JSON.stringify(all, null, 1));
      return Response.json(all);
    }
    return new Response("not found", { status: 404 });
  },
});
console.log("labeling on http://localhost:3100");
