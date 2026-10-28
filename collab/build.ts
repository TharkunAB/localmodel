import { mkdir, rename } from "node:fs/promises";
import { join } from "node:path";

const directory = import.meta.dir;
const result = await Bun.build({
  entrypoints: [join(directory, "upstream/packages/collab-web/index.html")],
  outdir: join(directory, "dist"),
  target: "browser",
  minify: true,
  sourcemap: "none",
  define: { "process.env.NODE_ENV": JSON.stringify("production") },
  naming: { entry: "[hash].[ext]", chunk: "[hash].[ext]", asset: "[hash].[ext]" },
  tsconfig: join(directory, "tsconfig.json"),
});
if (!result.success) throw new AggregateError(result.logs, "Collaboration browser build failed");
const html = result.outputs.find((output) => output.path.endsWith(".html"));
if (!html) throw new Error("Collaboration browser build did not emit HTML");
await mkdir(join(directory, "dist"), { recursive: true });
await rename(html.path, join(directory, "dist/index.html"));
console.log(`Built ${result.outputs.length} local browser assets`);
