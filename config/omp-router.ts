/** Route simple OMP requests to an idle local Qwen server. */
import { execFile } from "node:child_process";
import { get } from "node:http";
import { availableParallelism, loadavg } from "node:os";
import {
	createAssistantMessageEventStream,
	streamSimple,
	type AssistantMessageEvent,
	type Context,
	type Model,
	type ModelSpec,
	type SimpleStreamOptions,
} from "@oh-my-pi/pi-ai";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { Effort, THINKING_EFFORTS } from "@oh-my-pi/pi-catalog/effort";
import { kNoAuth, type ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

const OUTPUT_LIMIT = 16_000;
const CONTEXT_RESERVE = 4_096;
const METADATA_BYTES = 65_536;
const zeroCost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
type Metadata = Record<string, unknown>;

function record(value: unknown): Metadata {
	return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Metadata : {};
}

function positiveInteger(value: unknown): number | undefined {
	return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : undefined;
}

/** Accept the managed alias or an explicit Flash-Next name, never dense 27B. */
export function isFlashNext(id: string): boolean {
	const name = id.split("/").at(-1) ?? "";
	return name.toLowerCase() === "qwen3.8" || /^qwen3\.8[-_]flash[-_]next(?:[-_.]|$)/i.test(name);
}

/** This whitelist is a heuristic. Unclear requests and coding work stay remote. */
export function isSimpleRequest(prompt: string): boolean {
	const text = prompt.trim().replace(/^(?:please\s+)?(?:(?:can|could|would) you\s+)?(?:please\s+)?/i, "");
	if (!text || Buffer.byteLength(text) > 800 || text.split("\n").length > 8 || text.includes("```")) return false;
	if (/\b(?:implement|debug|refactor|architect\w*|design|prove|proof|theorem|security|audit|review|benchmark|optim\w*|migrat\w*|concurr\w*|deadlock|segfault|performance|vulnerab\w*|investigate|analy[sz]e|bug|error|fail\w*|broken|thorough\w*|complex)\b/i.test(text)) return false;
	return /^(?:(?:what (?:is|are|does)|what['’]s|define|explain|summari[sz]e|translate|rewrite|rephrase|calculate|how many|fix (?:the )?(?:typos|grammar))\b|(?:hi|hello|hey|thanks|thank you)[!. ]*$)/i.test(text);
}

/** Count serialized bytes as an upper bound, with room for template framing. */
export function fitsLocalContext(context: Context, window: number, output: number): boolean {
	return Buffer.byteLength(JSON.stringify(context), "utf8") + output + CONTEXT_RESERVE <= window;
}

function latestTurn(context: Context): { prompt: string; previous?: { provider: string; model: string } } {
	let user = -1;
	for (let index = context.messages.length - 1; index >= 0; index--) {
		if (context.messages[index].role === "user") { user = index; break; }
	}
	const message = context.messages[user];
	const prompt = message?.role === "user"
		? typeof message.content === "string" ? message.content
			: message.content.filter(part => part.type === "text")
				// Ignore only OMP's exact date/cwd reminder when classifying the prompt.
				.filter(part => !/^<system-reminder>\nToday: \d{4}-\d{2}-\d{2}; current working directory: '[^\n]*'\. Do not repeat this information in your reply\.\n<\/system-reminder>$/.test(part.text))
				.map(part => part.text).join("\n")
		: "";
	for (let index = context.messages.length - 1; index > user; index--) {
		const previous = context.messages[index];
		if (previous.role === "assistant") return { prompt, previous };
	}
	return { prompt };
}

function hasImages(context: Context): boolean {
	return context.messages.some(message => Array.isArray(message.content)
		&& message.content.some(part => part.type === "image"));
}

function loopbackBase(value: string): URL {
	const url = new URL(value);
	if (url.protocol !== "http:" || !["127.0.0.1", "[::1]"].includes(url.hostname) || url.username || url.password
		|| url.search || url.hash || url.pathname.replace(/\/$/, "") !== "/v1") {
		throw new Error("OMP router endpoints must use HTTP on an IPv4 or IPv6 loopback address at /v1, without credentials or queries.");
	}
	return url;
}

/** Node HTTP does not use proxy variables or follow redirects. */
async function metadata(base: URL, path: string, signal?: AbortSignal): Promise<unknown> {
	const deadline = AbortSignal.timeout(1_200);
	return new Promise((resolve, reject) => {
		const request = get(new URL(path, base), {
			signal: signal ? AbortSignal.any([signal, deadline]) : deadline,
		}, response => {
			if (response.statusCode !== 200) {
				response.resume();
				reject(new Error(`Local metadata returned HTTP ${response.statusCode}.`));
				return;
			}
			const chunks: Buffer[] = [];
			let bytes = 0;
			response.on("data", (chunk: Buffer) => {
				bytes += chunk.length;
				if (bytes > METADATA_BYTES) request.destroy(new Error("Local metadata is too large."));
				else chunks.push(chunk);
			});
			response.on("error", reject);
			response.on("end", () => {
				try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8"))); }
				catch (error) { reject(error); }
			});
		});
		request.on("error", reject);
	});
}

async function gpuBusy(signal?: AbortSignal): Promise<boolean> {
	return new Promise(resolve => {
		execFile("nvidia-smi", ["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
			{ encoding: "utf8", timeout: 1_000, maxBuffer: 4_096, signal }, (error, stdout) => {
				const values = stdout.trim().split(/\s+/).map(Number);
				resolve(Boolean(error) || !stdout.trim() || values.some(value => !Number.isFinite(value) || value < 0 || value >= 50));
			});
	});
}

async function localModel(base: URL, signal?: AbortSignal): Promise<{ model: Model; busy: boolean } | undefined> {
	try {
		const [healthValue, catalogValue, slotsValue] = await Promise.all([
			metadata(base, "/health", signal), metadata(base, "/v1/models", signal), metadata(base, "/slots", signal),
		]);
		const health = record(healthValue);
		const catalog = record(catalogValue).data;
		if (health.status !== "ok" || health.loaded === false || !Array.isArray(catalog) || catalog.length !== 1
			|| !Array.isArray(slotsValue) || slotsValue.length === 0) return undefined;
		const info = record(catalog[0]);
		if (typeof info.id !== "string" || !isFlashNext(info.id) || record(info.status).value === "unloaded") return undefined;
		const slots = slotsValue.map(record);
		const capacities = [health.max_context, info.max_model_len, info.context_length, info.max_context_length,
			record(info.meta).n_ctx, ...slots.map(slot => slot.n_ctx)].map(positiveInteger)
			.filter((value): value is number => value !== undefined);
		if (capacities.length === 0) return undefined;
		const contextWindow = Math.min(...capacities);
		const output = positiveInteger(info.max_output_tokens) ?? positiveInteger(info.max_tokens) ?? OUTPUT_LIMIT;
		return {
			busy: slots.some(slot => slot.is_processing !== false),
			model: buildModel({
				provider: "bonsai", id: info.id, name: info.id, api: "openai-completions", baseUrl: base.href.replace(/\/$/, ""),
				reasoning: true, thinking: { mode: "effort", efforts: [...THINKING_EFFORTS], defaultLevel: Effort.XHigh },
				input: ["text"], supportsTools: true, cost: zeroCost,
				contextWindow, maxTokens: Math.min(OUTPUT_LIMIT, output, Math.max(1, Math.floor(contextWindow / 2))),
				compat: {
					maxTokensField: "max_tokens", supportsDeveloperRole: false, supportsStore: false,
					supportsReasoningEffort: true, reasoningContentField: "reasoning_content",
					streamFirstEventTimeoutMs: 3_600_000, streamIdleTimeoutMs: 3_600_000,
					extraBody: { temperature: 1.0, top_p: 0.95 },
				},
			}),
		};
	} catch (error) {
		if (signal?.aborted) throw error;
		return undefined;
	}
}

async function selectModel(remote: Model, base: URL, context: Context, options?: SimpleStreamOptions): Promise<{ model: Model; reason: string; followup: boolean }> {
	const turn = latestTurn(context);
	const followup = turn.previous !== undefined;
	const chooseRemote = (reason: string) => ({ model: remote, reason, followup });
	if (hasImages(context)) return chooseRemote("images stay remote");
	if (turn.previous && (turn.previous.provider !== "bonsai" || !isFlashNext(turn.previous.model))) {
		return chooseRemote("continue the remote turn");
	}
	if (!followup && !isSimpleRequest(turn.prompt)) return chooseRemote("complex or unclassified request");
	if (!followup && loadavg()[0] / availableParallelism() >= 0.75) return chooseRemote("local CPU load is high");
	const local = await localModel(base, options?.signal);
	if (!local) return chooseRemote("local Flash-Next is unavailable");
	if (turn.previous && turn.previous.model !== local.model.id) return chooseRemote("local model changed during the turn");
	const outputLimit = positiveInteger(local.model.maxTokens) ?? OUTPUT_LIMIT;
	const output = Math.min(positiveInteger(options?.maxTokens) ?? outputLimit, outputLimit);
	const contextWindow = positiveInteger(local.model.contextWindow);
	if (!contextWindow || !fitsLocalContext(context, contextWindow, output)) return chooseRemote("conversation exceeds the local context budget");
	if (!followup && (local.busy || await gpuBusy(options?.signal))) return chooseRemote("local server or GPU is busy");
	options?.signal?.throwIfAborted();
	return { model: local.model, reason: followup ? "continue the local turn" : "simple request; local Flash-Next is idle", followup };
}

/** Forward physical identities and hooks. Retry only before any response content. */
export function routedStream(remote: Model, base: URL, context: Context, options?: SimpleStreamOptions) {
	const outer = createAssistantMessageEventStream();
	void (async () => {
		try {
			const selected = await selectModel(remote, base, context, options);
			if (!selected.followup) process.stderr.write(`[omp-router] ${JSON.stringify(`${selected.model.provider}/${selected.model.id}`)}: ${selected.reason}\n`);
			let target = selected.model;
			for (let attempt = 0; attempt < 2; attempt++) {
				options?.signal?.throwIfAborted();
				const pending: AssistantMessageEvent[] = [];
				let emitted = false;
				let retry = false;
				try {
					const inner = streamSimple(target, context, {
						...options, apiKey: kNoAuth,
						...(target.provider === "bonsai" ? {
							maxTokens: Math.min(positiveInteger(options?.maxTokens) ?? OUTPUT_LIMIT, positiveInteger(target.maxTokens) ?? OUTPUT_LIMIT),
						} : {}),
					});
					outer.forwardLocalWorkFrom(inner);
					for await (const event of inner) {
						if (!emitted && event.type === "start") { pending.push(event); continue; }
						if (!emitted && event.type === "error" && event.reason !== "aborted" && !options?.signal?.aborted
							&& target.provider === "bonsai" && attempt === 0) { retry = true; break; }
						for (const buffered of pending) outer.push(buffered);
						pending.length = 0;
						emitted = true;
						outer.push(event);
					}
					if (!retry) {
						for (const buffered of pending) outer.push(buffered);
						outer.end(await inner.result());
						return;
					}
				} catch (error) {
					if (emitted || target.provider !== "bonsai" || attempt !== 0 || options?.signal?.aborted) throw error;
					retry = true;
				} finally { outer.forwardLocalWorkFrom(undefined); }
				if (retry) {
					process.stderr.write("[omp-router] Local request failed before output; retrying once on the remote model.\n");
					target = remote;
				}
			}
		} catch (error) { outer.fail(error); }
	})();
	return outer;
}

export default function (pi: ExtensionAPI): void {
	const raw = process.env.OMP_REMOTE_ROUTER_CONFIG;
	if (!raw) return;
	const config = JSON.parse(raw) as { remote: ModelSpec; localBaseUrl: string };
	const remote = buildModel(config.remote);
	loopbackBase(remote.baseUrl);
	const contextWindow = positiveInteger(remote.contextWindow);
	const maxTokens = positiveInteger(remote.maxTokens);
	if (remote.provider !== "model" || remote.api !== "openai-completions"
		|| !contextWindow || !maxTokens) {
		throw new Error("OMP router requires the discovered remote OpenAI-compatible model.");
	}
	const local = loopbackBase(config.localBaseUrl);
	const model = {
		id: "auto", name: "Auto: remote / local Flash-Next", reasoning: true, thinking: remote.thinking,
		input: ["text"] as ("text" | "image")[], supportsTools: true, cost: zeroCost,
		contextWindow, maxTokens,
		compactionModel: `${remote.provider}/${remote.id}`,
	};
	pi.registerProvider("router", {
		baseUrl: remote.baseUrl, apiKey: kNoAuth, authHeader: false, api: "localmodel-router",
		models: [model], streamSimple: (_model, context, options) => routedStream(remote, local, context, options),
	});
}
