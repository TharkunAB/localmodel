import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const savedFetch = globalThis.fetch.bind(globalThis);
const root = process.env.OMP_BENCH_ROOT;
if (!root) throw new Error("OMP_BENCH_ROOT is required");
const log = process.env.OMP_BENCH_LOG ?? join(root, "network.jsonl");
mkdirSync(dirname(log), { recursive: true });
let requests = 0;
let modelRequests = 0;
const maxRequests = Number(process.env.OMP_BENCH_MAX_REQUESTS ?? "12");
if (!Number.isSafeInteger(maxRequests) || maxRequests < 1) {
  throw new Error("OMP_BENCH_MAX_REQUESTS must be a positive integer");
}

function record(event: Record<string, unknown>): void {
  appendFileSync(log, JSON.stringify({ at: new Date().toISOString(), ...event }) + "\n");
}

function object(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown> : undefined;
}

async function observe(response: Response, id: number, started: number): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buffer = "";
  let raw = "";
  let first = false;
  try {
    for (;;) {
      const next = await reader.read();
      if (next.done) break;
      const text = decoder.decode(next.value, { stream: true });
      raw += text;
      buffer += text;
      let end: number;
      while ((end = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, end).replace(/\r$/, "");
        buffer = buffer.slice(end + 1);
        if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
        let parsed: unknown;
        try { parsed = JSON.parse(line.slice(6)); } catch { continue; }
        const chunk = object(parsed);
        if (!chunk) continue;
        if (!first && Array.isArray(chunk.choices)) {
          const meaningful = chunk.choices.some((choice: unknown): boolean => {
            const delta = object(object(choice)?.delta);
            return Boolean(delta && (delta.content || delta.reasoning_content || delta.tool_calls));
          });
          if (meaningful) {
            first = true;
            record({ kind: "first_token", id, latency_ms: performance.now() - started });
          }
        }
        if (chunk.timings || chunk.usage) {
          record({ kind: "timings", id, timings: chunk.timings, usage: chunk.usage });
        }
      }
    }
    raw += decoder.decode();
    record({ kind: "response", id, status: response.status, elapsed_ms: performance.now() - started, text: raw });
  } catch (error: unknown) {
    record({ kind: "response_error", id, error: String(error), text: raw });
  } finally {
    reader.releaseLock();
  }
}

globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
  const url = String(input instanceof Request ? input.url : input);
  const parsed = new URL(url);
  const id = ++requests;
  let body: unknown = null;
  if (init?.body && typeof init.body === "string") {
    try { body = JSON.parse(init.body); } catch { body = init.body; }
  }
  const modelRequest = parsed.hostname === "127.0.0.1" && parsed.port === "8899"
    && parsed.pathname === "/v1/chat/completions";
  const started = performance.now();
  record({ kind: "request", id, url, body });
  const capture = process.env.OMP_BENCH_CAPTURE_ONLY;
  if (capture && modelRequest) {
    mkdirSync(dirname(capture), { recursive: true });
    writeFileSync(capture, JSON.stringify(body) + "\n");
    record({ kind: "capture_only", id, path: capture });
    // Exit before the provider can retry. The driver uses a disposable session copy.
    process.exit(0);
  }
  if (modelRequest) {
    modelRequests += 1;
    if (modelRequests > maxRequests) {
      record({ kind: "budget_exceeded", id, model_requests: modelRequests, limit: maxRequests });
      process.exit(76);
    }
    record({ kind: "generation_started", id });
  }
  const response = await savedFetch(input, init);
  if (modelRequest) {
    record({ kind: "response_headers", id, status: response.status, latency_ms: performance.now() - started });
    void observe(response.clone(), id, started);
  }
  return response;
}) as typeof fetch;
