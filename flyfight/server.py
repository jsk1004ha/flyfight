"""HTTP/WebSocket view of a sampled environment from the training process.

Training never waits for the browser. A latest-only bounded queue carries real
rollout frames; it is disabled entirely while no viewer is connected. Remote
control is restricted to an explicit allowlist, same-origin + per-run token.
"""
from __future__ import annotations
import asyncio
from dataclasses import asdict
import hmac
import json
import multiprocessing as mp
from pathlib import Path
import queue
import secrets
import webbrowser
from urllib.parse import urlsplit
from aiohttp import web,WSMsgType
from .live_view import LiveTrainingView
from .training import Config,train_worker

ROOT = Path(__file__).resolve().parents[1]


def training_ready(runtime: dict, view_ready: bool, error: str | None, worker_alive: bool) -> bool:
    return worker_alive and view_ready and error is None and runtime.get("state") == "training"


def create_app(cfg: Config, *, open_browser: bool = False, port: int = 8765,
               public_origin: str | None = None) -> web.Application:
    if public_origin is not None:
        parsed = urlsplit(public_origin)
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
                or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
            raise ValueError('public_origin must be an HTTPS origin without credentials or path')
        public_origin = public_origin.rstrip('/')
    allowed_hosts = {f"127.0.0.1:{port}",f"localhost:{port}",f"[::1]:{port}"}
    if public_origin:
        allowed_hosts.add(urlsplit(public_origin).netloc)
    ctx = mp.get_context("spawn")
    status_q, live_q = ctx.Queue(maxsize=8),ctx.Queue(maxsize=1)
    stop,pause,save,live_subscribers = ctx.Event(),ctx.Event(),ctx.Event(),ctx.Event()
    worker = ctx.Process(target=train_worker,args=(asdict(cfg),status_q,live_q,stop,pause,save,live_subscribers),daemon=False)
    token = secrets.token_urlsafe(32)
    runtime = {"state":"starting","envs":cfg.envs,"update":0,"env_steps":0}
    sockets: dict = {}
    live_view = LiveTrainingView(cfg)
    watch_paused = False
    latest = None
    last_error = None

    @web.middleware
    async def local_only(request,handler):
        # Kubernetes probes address the Pod IP, not the public viewer host. These
        # two read-only routes expose only a boolean and never a token or metrics.
        probe = request.method in ("GET", "HEAD") and request.path in ("/health/live", "/health/ready")
        if request.host not in allowed_hosts and not probe:
            raise web.HTTPForbidden(text="Unexpected Host")
        return await handler(request)

    app = web.Application(middlewares=[local_only],client_max_size=16384)

    async def home(request):
        html = (ROOT/"viewer"/"FlyFight_Viewer.html").read_text(encoding="utf-8")
        settings = json.dumps({"token":token,"wsPath":"/ws","readOnly":bool(public_origin)})
        html = html.replace("</head>",f"<script>window.FLYFIGHT_NATIVE={settings};</script></head>")
        return web.Response(text=html,content_type="text/html",headers={"Cache-Control":"no-store",
            "X-Content-Type-Options":"nosniff","Referrer-Policy":"no-referrer","X-Frame-Options":"DENY"})

    async def health(request):
        view_ready = runtime.get("state") in ("training","stopped")
        return web.json_response({**runtime,"view_ready":view_ready,
                                  "train_paused":pause.is_set(),"viewer_paused":watch_paused,
                                  "viewer_speed":1.,"view_source":"training_live","error":last_error})

    async def live(request):
        healthy = last_error is None and runtime.get("state") not in ("error", "stopped")
        return web.json_response({"alive":healthy},status=200 if healthy else 503,
                                 headers={"Cache-Control":"no-store"})

    async def ready(request):
        view_ready = runtime.get("state") == "training"
        healthy = training_ready(runtime,view_ready,last_error,worker.is_alive())
        return web.json_response({"ready":healthy},status=200 if healthy else 503,
                                 headers={"Cache-Control":"no-store"})

    async def send(ws,message):
        # Slow clients lose the connection instead of accumulating unbounded snapshots.
        try:
            await asyncio.wait_for(ws.send_str(json.dumps(message,ensure_ascii=False,allow_nan=False)),.3)
            return True
        except (TimeoutError,ConnectionError,RuntimeError):
            await ws.close()
            return False

    async def websocket(request):
        nonlocal watch_paused
        expected_origin = public_origin or f"http://{request.host}"
        if request.headers.get("Origin") != expected_origin or not hmac.compare_digest(request.query.get("token",""),token):
            raise web.HTTPForbidden(text="Same-origin session token required")
        ws = web.WebSocketResponse(heartbeat=20,max_msg_size=4096)
        await ws.prepare(request)
        sockets[ws] = False
        live_subscribers.set()
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                    if not isinstance(data,dict) or data.get("type") != "command":
                        raise ValueError("Expected command object")
                    command = data.get("command")
                    if public_origin:
                        raise ValueError("Public spectators cannot control training or shared playback")
                    if command == "train_pause":
                        pause.set()
                    elif command == "train_resume":
                        pause.clear()
                    elif command == "checkpoint":
                        save.set()
                    elif command == "view_pause":
                        watch_paused = True
                    elif command == "view_resume":
                        watch_paused = False
                    elif command == "view_speed":
                        raise ValueError("view_speed is unavailable for live training")
                    else:
                        raise ValueError("Unsupported command")
                    await send(ws,{"type":"ack","command":command,"train_paused":pause.is_set(),
                                   "viewer_paused":watch_paused,"viewer_speed":1.})
                except (ValueError,TypeError) as exc:
                    await send(ws,{"type":"error","message":str(exc)})
        finally:
            sockets.pop(ws,None)
            if not sockets:
                live_subscribers.clear()
        return ws

    async def bridge():
        nonlocal latest,last_error
        while True:
            for _ in range(8):
                try:
                    status = status_q.get_nowait()
                except queue.Empty:
                    break
                if status.get("state") == "error":
                    last_error = status.get("error")
                runtime.update(status)
            if worker.exitcode not in (None,0) and last_error is None:
                last_error = f"Training process exited ({worker.exitcode}); inspect run_dir/error.log"
                runtime["state"] = "error"
            new_frame = False
            while True:
                try:
                    latest = live_q.get_nowait()
                    new_frame = True
                except queue.Empty:
                    break
            if new_frame:
                latest["training"] = dict(runtime,train_paused=pause.is_set(),error=last_error)
            # All clients get the same actual rollout sample. Slow clients are
            # disconnected by send(); training never waits or replays frames.
            for ws in list(sockets):
                if ws.closed:
                    sockets.pop(ws,None); continue
                if not sockets[ws]:
                    if await send(ws,live_view.hello()):
                        sockets[ws] = True
                        if latest is not None:
                            await send(ws,latest)
                elif new_frame and not watch_paused:
                    await send(ws,latest)
            await asyncio.sleep(.015)

    async def startup(app):
        worker.start()
        app["bridge_task"] = asyncio.create_task(bridge())
        print(f"Viewer: http://127.0.0.1:{port}  |  Ctrl+C saves and stops.",flush=True)
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{port}")

    async def cleanup(app):
        app["bridge_task"].cancel()
        try:
            await app["bridge_task"]
        except asyncio.CancelledError:
            pass
        for ws in list(sockets):
            await ws.close()
        stop.set(); pause.clear()
        # Wait for the current PPO update and final atomic checkpoint; don't kill mid-write.
        await asyncio.to_thread(worker.join)
        for q in (status_q,live_q):
            q.close()

    app.router.add_get("/",home)
    app.router.add_get("/health",health)
    app.router.add_get("/health/live",live)
    app.router.add_get("/health/ready",ready)
    app.router.add_get("/ws",websocket)
    app.on_startup.append(startup); app.on_cleanup.append(cleanup)
    return app
