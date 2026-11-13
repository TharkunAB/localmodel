import { appendFileSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const root = process.env.OMP_BENCH_ROOT;
if (!root) throw new Error("OMP_BENCH_ROOT is required");
const original = globalThis.fetch.bind(globalThis);
let count = 0;

function phase(): string {
  try { return String(JSON.parse(readFileSync(join(root!, "phase.json"), "utf8")).name); }
  catch { return "startup"; }
}

function record(event: Record<string, unknown>): void {
  appendFileSync(join(root!, "network.jsonl"), JSON.stringify({ at: Date.now(), ...event }) + "\n");
}

async function observe(response: Response, id: number, name: string, start: number): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let pending = "";
  let first = false;
  try {
    for (;;) {
      const next = await reader.read();
      if (next.done) break;
      pending += decoder.decode(next.value, { stream: true });
      let end: number;
      while ((end = pending.indexOf("\n")) >= 0) {
        const line = pending.slice(0, end).replace(/\r$/, "");
        pending = pending.slice(end + 1);
        if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
        let chunk: any;
        try { chunk = JSON.parse(line.slice(6)); } catch { continue; }
        if (!first && chunk.choices?.some((choice: any) => {
          const delta = choice.delta;
          return delta && (delta.content || delta.reasoning_content || delta.tool_calls);
        })) {
          first = true;
          record({ kind: "first_token", id, phase: name, latency_ms: performance.now() - start });
        }
        if (chunk.timings || chunk.usage) {
          record({ kind: "timings", id, phase: name, timings: chunk.timings, usage: chunk.usage });
        }
      }
    }
    record({ kind: "complete", id, phase: name, elapsed_ms: performance.now() - start });
  } catch (error: unknown) {
    record({ kind: "stream_error", id, phase: name, error: String(error) });
  } finally {
    reader.releaseLock();
  }
}

globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
  const url = String(input instanceof Request ? input.url : input);
  const parsed = new URL(url);
  const model = parsed.hostname === "127.0.0.1" && parsed.port === "8899"
    && parsed.pathname === "/v1/chat/completions";
  if (!model) return original(input, init);
  const name = phase();
  const id = ++count;
  let body: unknown = null;
  if (typeof init?.body === "string") body = JSON.parse(init.body);
  record({ kind: "request", id, phase: name, url, body });
  if (process.env.OMP_BENCH_CAPTURE_ONLY) {
    writeFileSync(process.env.OMP_BENCH_CAPTURE_ONLY, JSON.stringify(body) + "\n");
    process.exit(0);
  }
  if (count > 64) throw new Error("The benchmark exceeded 64 model requests in one OMP process");
  const start = performance.now();
  const response = await original(input, init);
  record({ kind: "headers", id, phase: name, status: response.status, latency_ms: performance.now() - start });
  void observe(response.clone(), id, name, start);
  return response;
}) as typeof fetch;
