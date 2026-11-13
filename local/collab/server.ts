// Relay routing follows upstream v18.8.6 packages/collab-web/scripts/local-relay.ts.
import { BlockList, isIP } from "node:net";
import { resolve, sep } from "node:path";

export interface ServiceConfig {
  hostname: string;
  port: number;
  origin: string;
  allowedOrigins?: string[];
  allowCidrs: string[];
  tls?: { cert: string; key: string };
  caFile?: string;
  assetsDir?: string;
}

interface SocketData {
  roomId: string;
  role: "host" | "guest";
  peerId: number;
  openedAt: number;
  windowStart: number;
  windowBytes: number;
}
type RelaySocket = Bun.ServerWebSocket<SocketData>;
interface Room {
  host: RelaySocket;
  guests: Map<number, RelaySocket>;
  nextPeerId: number;
}

const MAX_ROOMS = 32;
const MAX_GUESTS = 16;
const MAX_SOCKETS = 128;
const MAX_FRAME_BYTES = 2 * 1024 * 1024;
const MAX_BUFFER_BYTES = 8 * 1024 * 1024;
const MAX_WINDOW_BYTES = 64 * 1024 * 1024;
const WINDOW_MS = 10_000;
const MAX_ROOM_MS = 24 * 60 * 60 * 1000;
const ROOM_PATH_RE = /^\/r\/([A-Za-z0-9_-]{10,64})$/;

/** Start the local relay and static browser service. */
export function startServer(config: ServiceConfig, root = resolve(import.meta.dir, "..")) {
  const origin = new URL(config.origin).origin;
  const origins = new Set([origin, ...(config.allowedOrigins ?? []).map((value) => new URL(value).origin)]);
  const hosts = new Set([...origins].map((value) => new URL(value).host));
  const peers = new BlockList();
  for (const cidr of config.allowCidrs) {
    const [address, prefix] = cidr.split("/");
    const family = isIP(address!);
    if (!family || prefix === undefined) throw new Error(`Invalid allowed peer CIDR: ${cidr}`);
    peers.addSubnet(address!, Number(prefix), family === 6 ? "ipv6" : "ipv4");
  }
  const assets = resolve(root, config.assetsDir ?? "collab/dist");
  const rooms = new Map<string, Room>();
  const sockets = new Set<RelaySocket>();
  let pending = 0;
  const websocketOrigins = [...origins].map((value) => value.replace(/^https:/, "wss:").replace(/^http:/, "ws:"));
  const headers = {
    "Content-Security-Policy": [
      "default-src 'none'", "script-src 'self'", "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob:", "font-src 'self'", `connect-src 'self' ${websocketOrigins.join(" ")}`,
      "base-uri 'none'", "form-action 'none'", "frame-ancestors 'none'", "object-src 'none'",
    ].join("; "),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cache-Control": "no-store",
  };
  const response = (body: BodyInit | null, status = 200, extra: Record<string, string> = {}) =>
    new Response(body, { status, headers: { ...headers, ...extra } });
  const send = (socket: RelaySocket, message: string | Buffer): void => {
    if (socket.getBufferedAmount() > MAX_BUFFER_BYTES) {
      socket.close(4008, "receiver is too slow");
      return;
    }
    socket.send(message);
  };
  const closeRoom = (roomId: string, room: Room): void => {
    rooms.delete(roomId);
    for (const guest of room.guests.values()) {
      send(guest, JSON.stringify({ t: "room-closed" }));
      guest.close(4001, "room closed");
    }
    room.guests.clear();
  };
  const server = Bun.serve<SocketData>({
    hostname: config.hostname,
    port: config.port,
    maxRequestBodySize: 1024,
    tls: config.tls ? { cert: Bun.file(resolve(root, config.tls.cert)), key: Bun.file(resolve(root, config.tls.key)) } : undefined,
    async fetch(req, srv): Promise<Response | undefined> {
      const address = (srv.requestIP(req)?.address ?? "").replace(/^::ffff:/, "");
      const family = isIP(address);
      if (!family || !peers.check(address, family === 6 ? "ipv6" : "ipv4")) return response(null, 403);
      if (!hosts.has(req.headers.get("host") ?? "")) return response(null, 403);
      if (req.method !== "GET" && req.method !== "HEAD") return response(null, 405);
      const url = new URL(req.url);
      const match = ROOM_PATH_RE.exec(url.pathname);
      if (match) {
        const requestOrigin = req.headers.get("origin");
        if (requestOrigin !== null && !origins.has(requestOrigin)) return response(null, 403);
        const role = url.searchParams.get("role");
        if (role !== "host" && role !== "guest") return response(null, 404);
        if (sockets.size + pending >= MAX_SOCKETS) return response(null, 503);
        const now = Date.now();
        const data: SocketData = { roomId: match[1]!, role, peerId: 0, openedAt: now, windowStart: now, windowBytes: 0 };
        pending++;
        if (srv.upgrade(req, { data })) return undefined;
        pending--;
        return response(null, 426);
      }
      if (url.pathname === "/healthz") return response(req.method === "HEAD" ? null : "ok\n", 200, { "Content-Type": "text/plain" });
      if (url.pathname === "/ca.pem" && config.caFile) {
        return response(req.method === "HEAD" ? null : Bun.file(resolve(root, config.caFile)), 200, {
          "Content-Type": "application/x-pem-file", "Content-Disposition": "attachment; filename=omp-local-ca.pem",
        });
      }
      // Only generated static files are reachable. Session paths have no HTTP route.
      let pathname: string;
      try { pathname = decodeURIComponent(url.pathname); } catch { return response(null, 404); }
      if (pathname.includes("\\") || pathname.includes("\0") || pathname.split("/").some((part) => part === "." || part === ".." || part.startsWith("."))) return response(null, 404);
      const filePath = resolve(assets, pathname === "/" ? "index.html" : pathname.slice(1));
      if (!filePath.startsWith(`${assets}${sep}`)) return response(null, 404);
      const file = Bun.file(filePath);
      if (!await file.exists()) return response(null, 404);
      return response(req.method === "HEAD" ? null : file, 200, { "Content-Type": file.type });
    },
    websocket: {
      maxPayloadLength: MAX_FRAME_BYTES,
      backpressureLimit: MAX_BUFFER_BYTES,
      closeOnBackpressureLimit: true,
      idleTimeout: 60,
      sendPings: true,
      open(ws) {
        pending = Math.max(0, pending - 1);
        sockets.add(ws);
        const { roomId, role } = ws.data;
        if (role === "host") {
          if (rooms.has(roomId)) { ws.close(4009, "a host is already connected for this room"); return; }
          if (rooms.size >= MAX_ROOMS) { ws.close(4008, "relay room limit"); return; }
          rooms.set(roomId, { host: ws, guests: new Map(), nextPeerId: 1 });
          return;
        }
        const room = rooms.get(roomId);
        if (!room) { ws.close(4004, "no such room"); return; }
        if (room.guests.size >= MAX_GUESTS || room.nextPeerId >= 0xffffffff) { ws.close(4029, "room guest limit"); return; }
        const peerId = room.nextPeerId++;
        ws.data.peerId = peerId;
        room.guests.set(peerId, ws);
        send(room.host, JSON.stringify({ t: "peer-joined", peer: peerId }));
      },
      message(ws, message) {
        if (typeof message === "string" || message.byteLength < 4) return;
        const room = rooms.get(ws.data.roomId);
        if (!room) return;
        const now = Date.now();
        if (now - ws.data.windowStart >= WINDOW_MS) { ws.data.windowStart = now; ws.data.windowBytes = 0; }
        ws.data.windowBytes += message.byteLength;
        if (ws.data.windowBytes > MAX_WINDOW_BYTES) { ws.close(4008, "relay byte rate limit"); return; }
        if (ws.data.role === "host") {
          if (room.host !== ws) return;
          const peerId = new DataView(message.buffer, message.byteOffset, 4).getUint32(0, false);
          if (peerId === 0) { for (const guest of room.guests.values()) send(guest, message); }
          else { const guest = room.guests.get(peerId); if (guest) send(guest, message); }
          return;
        }
        if (room.guests.get(ws.data.peerId) !== ws) return;
        new DataView(message.buffer, message.byteOffset, 4).setUint32(0, ws.data.peerId, false);
        send(room.host, message);
      },
      close(ws) {
        sockets.delete(ws);
        const { roomId, role, peerId } = ws.data;
        const room = rooms.get(roomId);
        if (!room) return;
        if (role === "host") { if (room.host === ws) closeRoom(roomId, room); return; }
        if (room.guests.get(peerId) === ws && room.guests.delete(peerId)) send(room.host, JSON.stringify({ t: "peer-left", peer: peerId }));
      },
    },
    error() { return response(null, 500); },
  });
  const cleanup = setInterval(() => {
    const now = Date.now();
    for (const [roomId, room] of rooms) {
      if (now - room.host.data.openedAt >= MAX_ROOM_MS) {
        closeRoom(roomId, room);
        room.host.close(4001, "room lifetime limit; share the session again");
      }
    }
  }, 30_000);
  cleanup.unref();
  return {
    port: server.port!,
    stop() { clearInterval(cleanup); for (const [roomId, room] of rooms) closeRoom(roomId, room); server.stop(true); },
  };
}

if (import.meta.main) {
  const flag = process.argv.indexOf("--config");
  const configPath = flag >= 0 ? process.argv[flag + 1] : resolve(import.meta.dir, "../config/omp-collab-service.json");
  if (!configPath) throw new Error("--config requires a file path");
  const config = await Bun.file(configPath).json() as ServiceConfig;
  const server = startServer(config);
  console.log(`OMP private collaboration listening on ${config.origin}`);
  const stop = () => { server.stop(); process.exit(0); };
  process.on("SIGTERM", stop);
  process.on("SIGINT", stop);
}
