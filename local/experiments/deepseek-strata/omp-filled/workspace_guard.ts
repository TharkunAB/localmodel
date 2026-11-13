/** Restrict this benchmark to workspace reads, clamp.py edits, and its validator. */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { lstatSync, readFileSync, realpathSync } from "node:fs";
import { join, resolve, sep } from "node:path";
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

const validatorHash = "55ddef940b70b621a70e337cbd6e8325c2132d689715d873e92254eee12b7fb5";
const maxSourceBytes = 65536;
// Use the benchmark's function boundary before each edit and validator run.
// Reject builtin aliases and annotations that could execute during import.
const checkSource = `import ast, builtins, sys
t = ast.parse(sys.stdin.read())
functions = {'clamp', 'clamp_index', 'clamp_many', 'clamp_span'}
calls = functions | {'min', 'max', 'len', 'ValueError'}
assert all(isinstance(n, ast.FunctionDef) or isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str) for n in t.body)
for n in ast.walk(t):
 assert not isinstance(n, (ast.Attribute, ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef))
 if isinstance(n, ast.FunctionDef):
  assert n.name in functions and not n.decorator_list and n.returns is None
  assert all(isinstance(v, ast.Constant) for v in n.args.defaults)
  assert all(v is None or isinstance(v, ast.Constant) for v in n.args.kw_defaults)
  assert not getattr(n, 'type_params', [])
 if isinstance(n, ast.arg):
  assert n.annotation is None and not n.arg.startswith('__')
 if isinstance(n, ast.Name):
  assert not n.id.startswith('__') and (n.id not in vars(builtins) or n.id in calls)
 if isinstance(n, ast.Call):
  assert isinstance(n.func, ast.Name) and n.func.id in calls
`;

function validateSource(source: string): void {
	if (Buffer.byteLength(source) > maxSourceBytes) throw new Error("clamp.py exceeds 64 KiB");
	execFileSync("/usr/bin/python3", ["-I", "-c", checkSource], {
		input: source, timeout: 2000, maxBuffer: 8192, stdio: ["pipe", "pipe", "pipe"],
	});
}

export default function (pi: ExtensionAPI): void {
	const root = process.env.OMP_BENCH_ROOT;
	if (!root) throw new Error("OMP_BENCH_ROOT is required by the benchmark guard");
	const workspace = realpathSync(join(root, "workspace"));
	const subject = join(workspace, "clamp.py");
	const validator = join(workspace, "verify.py");
	const regular = (path: string): void => {
		const stat = lstatSync(path);
		if (!stat.isFile() || stat.nlink !== 1) throw new Error("Only ordinary benchmark files are allowed");
	};
	const checkValidator = (): void => {
		regular(subject);
		regular(validator);
		if (createHash("sha256").update(readFileSync(validator)).digest("hex") !== validatorHash) {
			throw new Error("The benchmark validator differs from its pin");
		}
	};
	checkValidator();
	validateSource(readFileSync(subject, "utf8"));
	pi.on("tool_call", event => {
		try {
			checkValidator();
			const input = event.input as Record<string, unknown>;
			if (event.toolName === "read" || event.toolName === "edit") {
				if (typeof input.path !== "string" || /[\0\r\n]/.test(input.path)) throw new Error("Invalid path");
				const path = event.toolName === "read" ? input.path.replace(/:\d+(?:-\d+)?$/, "") : input.path;
				const target = realpathSync(resolve(workspace, path));
				if (!target.startsWith(workspace + sep)) throw new Error("Read and edit must stay inside the benchmark workspace");
				regular(target);
				if (event.toolName === "read") return;
				if (target !== subject) throw new Error("Only clamp.py may be edited; verify.py and configuration are protected");
				const keys = new Set(["i", "path", "old_string", "new_string", "replace_all"]);
				if (Object.keys(input).some(key => !keys.has(key)) || typeof input.old_string !== "string" ||
					typeof input.new_string !== "string" || input.old_string.length === 0 ||
					(input.replace_all !== undefined && typeof input.replace_all !== "boolean")) {
					throw new Error("Use the ordinary replace edit on clamp.py");
				}
				const text = readFileSync(subject, "utf8");
				const parts = text.split(input.old_string);
				if (parts.length < 2 || (parts.length > 2 && input.replace_all !== true)) throw new Error("Use an exact, unambiguous replacement");
				validateSource(parts.join(input.new_string));
				return;
			}
			if (event.toolName === "bash") {
				const keys = new Set(["i", "command", "cwd", "timeout", "pty", "async"]);
				if (Object.keys(input).some(key => !keys.has(key)) || typeof input.command !== "string" ||
					!/^python3? verify\.py clamp\.py [0-3]$/.test(input.command) ||
					(input.cwd !== undefined && (typeof input.cwd !== "string" || realpathSync(resolve(workspace, input.cwd)) !== workspace)) ||
					(input.pty !== undefined && input.pty !== false) || (input.async !== undefined && input.async !== false)) {
					throw new Error("Run only python3 verify.py clamp.py STAGE (0 through 3), in the benchmark workspace");
				}
				validateSource(readFileSync(subject, "utf8"));
				return;
			}
			throw new Error("Only read, edit, and the pinned validator bash command are allowed");
		} catch {
			return { block: true, reason: "Benchmark guard: read workspace files, edit only clamp.py with standalone pure functions, and run only python3 verify.py clamp.py STAGE (0 through 3). Do not change configuration or the validator." };
		}
	});
}
