# Local collaboration source review

The source is pinned to OMP `v18.8.6`, commit
`f068751e2f1dbdbc195977776d47a26db8697495`.
The source repository is <https://github.com/can1357/oh-my-pi>.
`upstream-revision.json` records the source files, local changes, and built asset hashes.

## Browser

The browser uses the upstream React guest client and AES-256-GCM codec.
The browser keeps the room key and control token in its URL fragment and memory.
The upstream client stores the display name and theme in local storage.
It does not store transcripts in local storage.
The browser history can retain a collaboration link.

The local `index.html` replaces the upstream entry point.
It removes the Umami script at `um.can.ac` and all hosted page metadata.
The local link parser uses the browser origin as its relay origin.
It rejects links for other relay origins.
The local TypeScript config resolves only the required source packages.
All other browser source files are unchanged.

The browser has no HTTP client for transcripts.
It sends encrypted frames through its WebSocket connection.
The response policy restricts scripts, fonts, and WebSocket connections to the local service.
The policy blocks external images and frames.
Markdown escapes raw HTML and rejects script URL schemes.
KaTeX uses MathML with `trust: false`.
Tool output uses React text rendering unless a local syntax highlighter is present.
The local build does not provide a syntax highlighter.

The six browser runtime packages match the upstream lock entries, including their integrity hashes.
They are React, React DOM, scheduler, Lucide React, KaTeX, and Commander.
Dependencies were installed with `--ignore-scripts`.
`bun.lock` pins the complete private install.
The global OMP install is unchanged.

## Relay

`server.ts` adapts upstream `packages/collab-web/scripts/local-relay.ts`.
It retains the room creation, sender identifier, broadcast, direct delivery, and room teardown protocol.
It does not decrypt frames or write session data.
It logs only its listening origin at startup.
There is no `/share` endpoint or transcript archive.

The service checks the remote peer address, HTTP Host, and browser WebSocket Origin.
Clients without an Origin header are accepted for the OMP terminal client.
TLS certificates and allowed addresses come from the service configuration.
Only generated browser assets, `/healthz`, and the public `/ca.pem` are served.
The service does not expose private TLS keys.

The relay limits rooms to 32, guests per room to 16, and total sockets to 128.
It limits each frame to 2 MiB and each socket buffer to 8 MiB.
It limits each sender to 64 MiB in a 10-second window.
WebSocket heartbeats remove inactive connections.
Rooms expire after 24 hours.
The clients can reconnect after room closure.

The service trusts the LAN and the bearer links.
The relay does not authenticate users or inspect their encrypted requests.
The OMP host validates the control token and enforces view access.
The server binds only to the node's Tailscale IPv4 address on port 7467.
The peer filter accepts Tailnet IPv4 addresses.
The relay checks browser Origin and Host against the configured Tailscale name and IP address.
TLS uses a private CA that the OMP wrappers trust explicitly.
The CA file path is passed to the private runtime. Its contents are read at socket creation.
Matti accepted a browser certificate warning for this deployment.
There is no Tailscale Serve, Funnel, Windows forwarding rule, or custom Windows CA trust.

## Validation

The static browser build ran under a 2 GiB cgroup limit.
The measured build peak was 52 MiB.
The generated browser assets total about 652 KiB.
The new TypeScript service and build script pass strict type checking.

`state/collab-checks/relay-check.ts` uses a synthetic host and the upstream browser `GuestClient`.
It checks trusted TLS, static assets, private relay parsing, encrypted snapshots,
view and control token handshakes, sender identifiers, wrong keys, duplicate hosts,
missing rooms, and room teardown.
It also checks Host, Origin, and CIDR rejection, path traversal, missing files,
blocked sharing paths, and private key paths.
`state/collab-checks/summary.json` records the result.
No real OMP session content or model request was used.

This review does not establish that the full upstream browser or its dependencies have no defects.
The relay caps and browser response policy limit specific failure paths.
