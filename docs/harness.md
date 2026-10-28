# Local coding harness

The checkout contains the launchers, profile defaults, runtime patch manifest,
dependency locks, local search plugin, and optional collaboration source.
Model weights, runtime installations, credentials, sessions, and private
certificates stay outside version control.

The packaged setup supports Ubuntu on Linux x86_64, including WSL2. A fresh
installation does not require the original user's home directory or a global
OMP installation.

## Install from a fresh checkout

Install Git, Bash, Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and [Bun](https://bun.com/docs/installation/). OMP requires Bun 1.3.14 or later.
The supplied OMP dependency lock was checked with Bun 1.4.2. Keep `uv` and
`bun` on `PATH`. Model and search launchers also require a working systemd
user manager and cgroup v2. On WSL2, enable systemd before starting services.

From the checkout, preview and install the base environment, OMP, and search:

```bash
bin/setup-localmodel --harness --search --dry-run
bin/setup-localmodel --harness --search --run
```

The root installer creates the Python environment and launcher links. It
refuses to replace existing files or links that point elsewhere. Use `--bin-dir DIRECTORY` to
select a different launcher directory. Add that directory to `PATH`.

The OMP step runs this command. It can also run separately:

```bash
uv run --no-project --python .venv/bin/python python bin/setup-harness --dry-run
uv run --no-project --python .venv/bin/python python bin/setup-harness --run
```

`setup-harness` installs `@oh-my-pi/pi-coding-agent@18.8.6` and the Linux x86_64
native package from `config/omp-package/bun.lock`. It uses an isolated temporary
directory, disables package scripts, and checks the reviewed OMP bundle before
it activates the installation. It omits optional speech and transformer
packages. The native package is an explicit dependency, so it remains installed.
It does not install Bun, update a global package, start a service, or fetch a
model. A repeated install checks and keeps the existing managed package.

SearXNG has its own Python environment and pinned source checkout under
`search/`. Its installer records the installed Python versions there. Model
runtime installation is separate. Use the root installer's `--bonsai` or
`--strata` option when the selected model requires that backend. See the
root README for model-specific dependencies and downloads.

## Start the components

Run search in one terminal:

```bash
searxng-local
```

Run the selected local model in another terminal. For example:

```bash
localmodel list
localmodel fetch qwen3.8
localmodel start qwen3.8
```

The fetch command downloads the selected model and its required runtime. Check
the model's documented disk, RAM, and GPU requirements first. The setup
installer does not perform this step.

Start the coding harness from the project you want to work on:

```bash
cd /path/to/project
omp-local --mem 32
```

`omp-local` discovers the active `localmodel` endpoint. All conversation and
task roles use that model. Search uses SearXNG at port 8888. URL reads fetch
the URL directly.

For a remote main model, expose its OpenAI-compatible endpoint on local port
8898. An SSH tunnel is sufficient. Keep a local task model ready on port 8899:

```bash
omp-remote --mem 32
```

`omp-remote` discovers the main model on port 8898 through IPv4 or IPv6. Its
context limit defaults to 1048576 tokens and decreases when the server reports
a smaller limit. The output limit is at most 262144 tokens. The main, plan,
small, slow, and title roles use this endpoint.

The task, scout, reviewer, security-reviewer, and sonic agents use the live
model on port 8899. This includes Qwen3.8. The provider name `bonsai` remains
for compatibility; it does not require a Bonsai model. Discovery has no cached
fallback. If either required endpoint is unavailable, `omp-remote` exits.
Relaunch after changing a model or its context limit.

Remote sessions saved with `deepseek-v4.1` or a `deepseek-v4.1-*` profile can
resume with another served profile in that family. The current server must
report one matching model. Other model families retain OMP's normal checks.

## Dependencies and state

| Component | Runtime dependencies | Local installation or state |
| --- | --- | --- |
| Python launchers | Python 3.12; root `.venv` | Root scripts and `.venv/` |
| OMP | Bun >=1.3.14; locked OMP and native packages | `state/omp-package/18.8.6/` |
| Reviewed OMP patches | Input and output SHA256 in `config/omp-runtime.json` | `state/omp-runtime/18.8.6/managed/` |
| Local OMP profile | Ready model; healthy SearXNG | `state/omp-local/` |
| Remote OMP profile | Ports 8898 and 8899; healthy SearXNG | `state/omp-remote/` |
| SearXNG | Pinned source; its Python requirements; Granian | `search/`, `state/searxng/` |
| Memory limits | Linux cgroup v2; systemd user manager; Python 3 | `libexec/capped*`, profile peak logs |
| OpenCode, optional | External OpenCode v2.0.26 binary; reviewed goal plugin | `~/.opencode/bin/opencode`, `state/opencode*/` |
| OpenCode goal plugin | Plugin 0.11.1; Zod 4.6.5 | `state/plugins/opencode-goal-plugin-0.11.1/` |
| OpenCode search plugin | OpenCode plugin API; no npm dependencies | `config/searxng-plugin/` |
| Collaboration, optional | Bun; checked-in browser source and dependency lock; private TLS certificate | `collab/`, `state/collab/` |

OMP first selects the managed package. If that directory is absent, it can
use `omp` from `PATH`. Either source must match the exact reviewed version and
input checksum. A partial or changed managed installation causes an error.
It does not cause a fallback to the global package. The launcher checks the
patched output checksum before execution and sets `PI_PACKAGE_DIR` to the
selected package. It leaves the global package unchanged.

The local and remote profiles have separate sessions, authentication files,
caches, and preferences. The launcher seeds a missing `agent/config.yml`
from `config/omp.json`. It preserves an existing `config.yml` or `config.yaml`.
Settings changed through OMP therefore persist. The generated `launcher.yml`
overlay controls model routing, context thresholds, search, and network
settings. Edit persistent defaults or profile settings, rather than this
generated overlay. The launchers reserve `--profile` for their isolated state.

OMP defaults to `xhigh` reasoning. Compaction uses the conversation model,
runs synchronously, and keeps 20000 recent tokens. It starts at the lower of
80% of context or context minus the output limit and 16384 tokens.

OMP supports its native `/goal` and `/skill:NAME` commands. Skills load from
the user and project `.agents/skills` directories. A profile links the user's
`~/AGENTS.md` when that file exists. Project instructions continue to apply.

## Search and memory checks

Every interactive OMP launch checks the configured SearXNG `/healthz` endpoint
without a proxy or redirect. It requires HTTP 200 and `OK`. This check sends
no search query. Help, version, config, agent-listing, and completion commands
work without search or a model. The required OMP package and Bun must still
be installed.

`--mem` must be the first launcher option. Bare sizes use GiB. For example,
`--mem 32` sets a shared 32 GiB limit with no swap for the foreground harness
and its ordinary child processes. The model and search servers have separate
limits. The bundled memory recorder removes the dependency on a separate
`heavy-management` checkout. Peak records remain under the profile's state
directory. An external OpenCode `--server` cannot receive this limit.

OMP's shared LSP broker and background-service launcher are disabled by the
managed overlay. Optional upstream daemons can use separate scopes. In
particular, the text-prediction daemon has previously been observed outside
the foreground scope. Do not treat `--mem` as a limit for every process in the
user account.

The runtime disables automatic xAI search, hosted URL-reader fallbacks,
session uploads, and collaboration destinations outside the configured list.
SearXNG itself contacts the search engines in `config/searxng.yml`; direct URL
reads contact the requested website. These settings do not provide a network
firewall for project commands or explicit user configuration.

## Optional OpenCode

The supplied OpenCode profiles target v2.0.26 at `~/.opencode/bin/opencode`.
This repository does not contain a pinned OpenCode binary installer. Install
that compatible version separately before using these launchers. A current
upstream release can have a different plugin API or profile schema.

Install the separately reviewed goal plugin and Zod dependency:

```bash
uv run --no-project --python .venv/bin/python python bin/setup-harness --dry-run --with-opencode-goal
uv run --no-project --python .venv/bin/python python bin/setup-harness --run --with-opencode-goal
```

Link the optional OpenCode launchers with `bin/setup-localmodel --opencode --run`.
If an existing launcher is a wrapper file instead of a matching symbolic link,
setup preserves it and stops. You can keep using that wrapper or select a
separate `--bin-dir`. The base setup does not replace OpenCode launchers.

The installer checks both archive SHA256 values and every extracted file
against `config/opencode-goal-plugin-manifest.json`. It executes no npm
scripts. Existing installations must match that manifest. The local SearXNG
plugin ships in the checkout and needs no package installation.

```bash
opencode-local --mem 32
opencode-remote --mem 32
opencode-local service status
opencode-remote service status
```

OpenCode uses separate profile state. Service commands work without a ready
model. They keep a saved model configuration when discovery fails. A fresh
service configuration starts with no allowed model providers. The OpenCode
main routes follow the local endpoint or port 8898. Its optional general and
explore agents retain their existing Bonsai route and cached-metadata behavior.
This differs from OMP remote task discovery, which requires a live port 8899.

To start an OpenCode service with a memory limit, put the limit before
`service start`. Restarting a service interrupts its work. A service already
running outside the requested scope needs an explicit `service restart` to
receive the limit. Setup does not start or restart these services.

## Optional private collaboration

Collaboration is disabled when `config/omp-collab.json` is absent. Each machine
needs its own configuration and TLS certificate. The example files use a
reserved hostname and loopback addresses. They are templates, not a working
service configuration.

To configure a new private relay, copy the examples only when the destination
files do not already exist:

```bash
cp -n config/omp-collab.example.json config/omp-collab.json
cp -n config/omp-collab-service.example.json config/omp-collab-service.json
```

Set the real private hostname, listen address, and permitted client CIDRs.
Keep `relayUrl`, `webUrl`, `origin`, and `allowedOrigins` consistent. Client
origins must use HTTPS and WSS. Generate a certificate with your private CA
for that hostname. Store the server certificate, private key, and CA file at
the configured paths under `state/collab/tls/`. Restrict the key to its owner.
Trust that CA in the client browser. OMP reads the configured CA file only for
its collaboration connection; setup does not alter the system trust store.

Build the supplied browser client:

```bash
bun install --cwd collab --frozen-lockfile --ignore-scripts
bun run --cwd collab build
```

The upstream source revision and local changes are recorded in
`collab/upstream-revision.json` and `collab/AUDIT.md`. The source and dependency
locks ship in this checkout. The generated `collab/dist/` directory does not.

For a user service, write a machine-specific `config/omp-collab.service`.
Replace the example paths with the paths on this machine:

```ini
[Unit]
Description=Private OMP collaboration
After=network.target

[Service]
Type=simple
WorkingDirectory=/absolute/path/to/local-models
ExecStart=/absolute/path/to/bun /absolute/path/to/local-models/collab/server.ts --config /absolute/path/to/local-models/config/omp-collab-service.json
Restart=on-failure
RestartSec=3
UMask=0077
MemoryMax=512M
MemoryHigh=384M
MemorySwapMax=0
TasksMax=64
NoNewPrivileges=yes
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX

[Install]
WantedBy=default.target
```

Review the addresses, certificate, and unit before enabling the service:

```bash
omp-collab install
omp-collab start
omp-collab info
```

The install command refuses to replace another unit. It enables the unit for
future user-manager starts. Core setup leaves collaboration files and services
unchanged. Keep machine configuration and certificates out of version control.

## Updates and recovery

OMP updates require a new reviewed input bundle, patches, output checksum,
package manifest, and dependency lock. Updating only a global `omp` executable
does not update the managed runtime. Never bypass a checksum failure to make
an unreviewed package launch.

If setup reports an existing directory with different files, inspect that
directory before moving or replacing it. The installer does not delete an
existing installation or profile. Keep profile state when changing runtime
packages. Model, search, and harness processes must be stopped separately
when they are no longer needed.
