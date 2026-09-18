"""Loopback-only HTTP/WebSocket viewer + isolated training process.

Training never waits for the browser. A bounded queue moves model snapshots
at update intervals; the spectator runs inference on CPU in its own thread.
Remote control is restricted to an explicit allowlist, same-origin + per-run
token. There is no arbitrary command execution, checkpoint upload or file API.
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
import torch
from aiohttp import web,WSMsgType
from .training import Config,train_worker
from .spectator import Spectator

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
    status_q, weights_q = ctx.Queue(maxsize=8),ctx.Queue(maxsize=1)
    stop,pause,save = ctx.Event(),ctx.Event(),ctx.Event()
    worker = ctx.Process(target=train_worker,args=(asdict(cfg),status_q,weights_q,stop,pause,save),daemon=False)
    token = secrets.token_urlsafe(32)
    runtime = {"state":"starting","envs":cfg.envs,"update":0,"env_steps":0}
    sockets: dict = {}
    watch = Spectator(cfg.map_path)
    watch_paused = False
    watch_speed = 1.
    latest = None
    last_error = None
    torch.set_num_threads(2)

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
        return web.json_response({**runtime,"view_ready":watch.models is not None,
                                  "train_paused":pause.is_set(),"viewer_paused":watch_paused,
                                  "viewer_speed":watch_speed,"error":last_error})

    async def live(request):
        healthy = last_error is None and runtime.get("state") not in ("error", "stopped")
        return web.json_response({"alive":healthy},status=200 if healthy else 503,
                                 headers={"Cache-Control":"no-store"})

    async def ready(request):
        healthy = training_ready(runtime,watch.models is not None,last_error,worker.is_alive())
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
        nonlocal watch_paused,watch_speed
        expected_origin = public_origin or f"http://{request.host}"
        if request.headers.get("Origin") != expected_origin or not hmac.compare_digest(request.query.get("token",""),token):
            raise web.HTTPForbidden(text="Same-origin session token required")
        ws = web.WebSocketResponse(heartbeat=20,max_msg_size=4096)
        await ws.prepare(request)
        sockets[ws] = False
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
                    elif command == "view_speed" and type(data.get("value")) in (int,float) and data["value"] in (.5,1,2,4,8):
                        watch_speed = float(data["value"])
                    else:
                        raise ValueError("Unsupported command")
                    await send(ws,{"type":"ack","command":command,"train_paused":pause.is_set(),
                                   "viewer_paused":watch_paused,"viewer_speed":watch_speed})
                except (ValueError,TypeError) as exc:
                    await send(ws,{"type":"error","message":str(exc)})
        finally:
            sockets.pop(ws,None)
        return ws

    async def bridge():
        nonlocal latest,last_error
        loop = asyncio.get_running_loop()
        last_watch_time = loop.time()
        next_frame = last_watch_time
        watch_accumulator = 0.
        while True:
            for _ in range(8):
                try:
                    status = status_q.get_nowait()
                except queue.Empty:
                    break
                if status.get("state") == "error":
                    last_error = status.get("error")
                runtime.update(status)
            try:
                blob = weights_q.get_nowait()
            except queue.Empty:
                blob = None
            if blob is not None:
                await asyncio.to_thread(watch.offer,blob)
            if worker.exitcode not in (None,0) and last_error is None:
                last_error = f"Training process exited ({worker.exitcode}); inspect run_dir/error.log"
                runtime["state"] = "error"
            now = loop.time()
            new_frame = False
            if sockets and watch.models is not None and not watch_paused:
                elapsed = min(now-last_watch_time,.25)
                last_watch_time = now
                watch_accumulator = min(watch_accumulator+elapsed*watch_speed,cfg.dt*16)
                # Simulation speed follows the selected multiplier, while browser traffic stays <=30 FPS.
                if now >= next_frame and watch_accumulator >= cfg.dt:
                    steps = min(16,int(watch_accumulator/cfg.dt))
                    latest = await asyncio.to_thread(watch.advance,steps)
                    watch_accumulator -= steps*cfg.dt
                    latest["training"] = dict(runtime,train_paused=pause.is_set(),error=last_error)
                    new_frame = True
                    next_frame = now+1/30
            else:
                last_watch_time = now
                watch_accumulator = 0.
                next_frame = now
            # All clients get latest-only state. No browser -> no spectator CPU ray rendering.
            for ws in list(sockets):
                if ws.closed:
                    sockets.pop(ws,None); continue
                if watch.models is not None and not sockets[ws]:
                    if await send(ws,watch.hello()):
                        sockets[ws] = True
                        if latest is not None:
                            await send(ws,latest)
                elif new_frame:
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
        for q in (status_q,weights_q):
            q.close()

    app.router.add_get("/",home)
    app.router.add_get("/health",health)
    app.router.add_get("/health/live",live)
    app.router.add_get("/health/ready",ready)
    app.router.add_get("/ws",websocket)
    app.on_startup.append(startup); app.on_cleanup.append(cleanup)
    return app
