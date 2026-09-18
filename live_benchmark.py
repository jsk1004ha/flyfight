"""Benchmark the real trainer, loopback spectator WebSocket, and system Chrome.

Example:
    python live_benchmark.py --duration 15 --output-json evidence/live_benchmark.json
    python live_benchmark.py --normal-config --device auto --duration 60
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Iterable
from urllib.request import urlopen

import torch
from playwright.sync_api import sync_playwright

from flyfight.training import Config


ROOT = Path(__file__).resolve().parent
SMALL_DEFAULTS = {
    "envs": 8,
    "hidden": 96,
    "width": 12,
    "height": 8,
    "horizon": 8,
    "epochs": 1,
    "minibatch_envs": 4,
    "threads": 2,
    "publish_every": 1,
}


def find_free_port() -> int:
    """Return a currently free loopback TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure FlyFight training and its real browser WebSocket viewer together."
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Measurement seconds")
    parser.add_argument("--warmup", type=float, default=3.0, help="Warmup seconds after native frames arrive")
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--ready-timeout", type=float, default=90.0)
    parser.add_argument("--stutter-ms", type=float, default=33.34, help="Frame delta counted as a stutter")
    parser.add_argument("--output-json", "--output", dest="output_json", default="evidence/live_benchmark.json")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--chrome-path", default=os.environ.get("CHROMIUM_PATH", ""))
    parser.add_argument("--headed", action="store_true", help="Show the system Chrome window during measurement")
    parser.add_argument("--normal-config", action="store_true", help="Use Config defaults for unspecified model/training sizes")
    parser.add_argument("--device", default="auto")
    for name in ("envs", "hidden", "width", "height", "horizon", "epochs", "minibatch_envs", "threads", "publish_every"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    args = parser.parse_args(argv)
    if args.duration <= 0 or args.warmup < 0 or args.sample_interval <= 0 or args.ready_timeout <= 0:
        parser.error("duration/interval/timeouts must be positive and warmup must be non-negative")
    defaults = Config()
    for name, small_value in SMALL_DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, getattr(defaults, name) if args.normal_config else small_value)
    if args.run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = str(ROOT / "runs" / f"live_benchmark_{stamp}")
    return args


def build_trainer_command(args: argparse.Namespace, port: int) -> list[str]:
    command = [sys.executable, "run.py", "--device", args.device]
    for name in ("envs", "hidden", "width", "height", "horizon", "epochs", "minibatch_envs", "threads", "publish_every"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    command += [
        "--save-every", "1000000",
        "--episode-seconds", "3",
        "--run-dir", str(args.run_dir),
        "--port", str(port),
    ]
    return command


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return round(float(ordered[index]), 3)


def _stats(values: Iterable[float]) -> dict:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"samples": 0, "mean": None, "min": None, "max": None, "p95": None}
    return {
        "samples": len(clean),
        "mean": round(sum(clean) / len(clean), 3),
        "min": round(min(clean), 3),
        "max": round(max(clean), 3),
        "p95": _percentile(clean, 0.95),
    }


def summarize_render(
    fps_samples: list[float],
    scale_samples: list[float],
    frame_ms: list[float],
    stutter_ms: float,
) -> dict:
    stutters = sum(delta > stutter_ms for delta in frame_ms)
    return {
        "fps": _stats(fps_samples),
        "render_scale": {
            "min": round(min(scale_samples), 3) if scale_samples else None,
            "max": round(max(scale_samples), 3) if scale_samples else None,
            "final": round(scale_samples[-1], 3) if scale_samples else None,
        },
        "frame_timing_ms": {
            **_stats(frame_ms),
            "stutter_threshold": stutter_ms,
            "stutter_count": stutters,
            "stutter_ratio": round(stutters / len(frame_ms), 6) if frame_ms else None,
        },
    }


def validate_native_samples(states: list[dict]) -> None:
    if not any(state.get("native") and state.get("external") for state in states):
        raise RuntimeError("Browser did not receive native frames through the real WebSocket spectator")


def describe_gpu_metrics(samples: list[float]) -> dict:
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        return {
            "cuda_available": False,
            "collected": False,
            "reason": "CUDA is unavailable in this Python/PyTorch runtime; no GPU metrics were collected.",
        }
    if not samples:
        return {
            "cuda_available": True,
            "collected": False,
            "reason": "CUDA is available, but reliable per-process nvidia-smi memory telemetry was unavailable.",
        }
    return {"cuda_available": True, "collected": True, "process_memory_mib": _stats(samples)}


def _chrome_path(requested: str) -> str:
    candidates = [requested]
    if sys.platform == "win32":
        candidates += [
            os.path.join(os.environ.get("PROGRAMFILES", ""), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
        ]
    else:
        candidates += [shutil.which(name) or "" for name in ("google-chrome", "chromium", "chromium-browser")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise FileNotFoundError("System Chrome/Chromium was not found; pass --chrome-path")


def _health(base_url: str) -> dict:
    with urlopen(base_url + "/health", timeout=2) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_health(base_url: str, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    latest_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Trainer/server exited before readiness (code {process.returncode})")
        try:
            health = _health(base_url)
            if health.get("error"):
                raise RuntimeError(str(health["error"]))
            if health.get("view_ready"):
                return health
        except Exception as exc:
            latest_error = exc
        time.sleep(0.1)
    raise TimeoutError(f"Server did not become viewer-ready: {latest_error}")


def _process_sampler(root_pid: int):
    try:
        import psutil
    except ImportError:
        return None, {"collected": False, "reason": "psutil is not installed."}
    root = psutil.Process(root_pid)
    primed: set[int] = set()

    def sample() -> tuple[dict, set[int]]:
        processes = [root] + root.children(recursive=True)
        live = []
        for process in processes:
            try:
                if process.pid not in primed:
                    process.cpu_percent(None)
                    primed.add(process.pid)
                live.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        cpu = rss = 0.0
        pids = set()
        for process in live:
            try:
                cpu += process.cpu_percent(None)
                rss += process.memory_info().rss / (1024 * 1024)
                pids.add(process.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return {"cpu_percent": cpu, "rss_mib": rss, "processes": len(pids)}, pids

    return sample, None


def _gpu_memory_for_pids(pids: set[int]) -> float | None:
    if not torch.cuda.is_available() or not pids:
        return None
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        output = subprocess.check_output(
            [executable, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    total = 0.0
    matched = False
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) in pids:
            try:
                total += float(fields[1])
                matched = True
            except ValueError:
                pass
    return total if matched else None


def _stop_process(process: subprocess.Popen, timeout: float = 90.0) -> dict:
    forced = False
    if process.poll() is None:
        process.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            forced = True
            process.kill()
            process.wait(timeout=10)
    return {"returncode": process.returncode, "forced": forced}


def _wait_for_checkpoint(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            return
        time.sleep(0.1)
    raise TimeoutError(f"Checkpoint command did not create {path}")


def run_benchmark(args: argparse.Namespace) -> dict:
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir).resolve()
    if any((run_dir / name).exists() for name in ("initial.pt", "latest.pt", "metrics.jsonl")):
        raise FileExistsError(f"Benchmark run directory already contains an experiment: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = output.with_suffix(".server.log")
    command = build_trainer_command(args, port)
    report = {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "base_url": base_url,
        "run_dir": str(run_dir),
        "server_log": str(log_path),
        "configuration": {name: getattr(args, name) for name in SMALL_DEFAULTS} | {"device": args.device},
    }
    shutdown = None
    process = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            sampler, process_unavailable = _process_sampler(process.pid)
            _wait_for_health(base_url, process, args.ready_timeout)
            chrome = _chrome_path(args.chrome_path)
            fps_samples: list[float] = []
            scale_samples: list[float] = []
            states: list[dict] = []
            health_samples: list[dict] = []
            process_samples: list[dict] = []
            gpu_samples: list[float] = []
            browser_errors: list[str] = []
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    executable_path=chrome,
                    headless=not args.headed,
                    args=["--no-sandbox", "--disable-dev-shm-usage"],
                )
                page = browser.new_page(viewport={"width": 1600, "height": 1050})
                page.on("pageerror", lambda error: browser_errors.append(str(error)))
                page.goto(base_url, wait_until="load", timeout=int(args.ready_timeout * 1000))
                page.wait_for_function(
                    "window.FlyFight && window.FlyFight.getState().native === true",
                    timeout=int(args.ready_timeout * 1000),
                )
                page.evaluate("""() => {
                    window.__flyfightFrameTimes = [];
                    let previous = performance.now();
                    function sample(now) {
                        const values = window.__flyfightFrameTimes;
                        values.push(now - previous);
                        if (values.length > 30000) values.splice(0, values.length - 30000);
                        previous = now;
                        requestAnimationFrame(sample);
                    }
                    requestAnimationFrame(sample);
                }""")
                page.wait_for_timeout(int(args.warmup * 1000))
                deadline = time.monotonic() + args.duration
                while time.monotonic() < deadline:
                    state = page.evaluate("window.FlyFight.getState()")
                    states.append(state)
                    if isinstance(state.get("fps"), (int, float)):
                        fps_samples.append(float(state["fps"]))
                    if isinstance(state.get("renderScale"), (int, float)):
                        scale_samples.append(float(state["renderScale"]))
                    health_samples.append(_health(base_url))
                    pids: set[int] = set()
                    if sampler is not None:
                        process_sample, pids = sampler()
                        process_samples.append(process_sample)
                    gpu_memory = _gpu_memory_for_pids(pids)
                    if gpu_memory is not None:
                        gpu_samples.append(gpu_memory)
                    page.wait_for_timeout(int(args.sample_interval * 1000))
                frame_ms = page.evaluate("window.__flyfightFrameTimes.slice()")
                # Ask the live server to save through its public browser command and
                # confirm the atomic file before Windows sends CTRL_BREAK to the tree.
                page.locator("#saveModelBtn").click()
                _wait_for_checkpoint(run_dir / "latest.pt", args.ready_timeout)
                browser.close()
            validate_native_samples(states)
            trainer_sps = [sample["env_steps_s"] for sample in health_samples if isinstance(sample.get("env_steps_s"), (int, float))]
            report.update({
                "status": "complete",
                "browser": {"executable": chrome, "headed": args.headed, "page_errors": browser_errors, "native_websocket_frames": True},
                "render": summarize_render(fps_samples, scale_samples, frame_ms, args.stutter_ms),
                "trainer": {
                    "env_steps_s": _stats(trainer_sps),
                    "final_health": health_samples[-1] if health_samples else None,
                },
                "process_tree": process_unavailable or {
                    "collected": True,
                    "cpu_percent": _stats(sample["cpu_percent"] for sample in process_samples),
                    "rss_mib": _stats(sample["rss_mib"] for sample in process_samples),
                    "max_processes": max((sample["processes"] for sample in process_samples), default=0),
                },
                "gpu": describe_gpu_metrics(gpu_samples),
                "measurement_seconds": args.duration,
            })
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if process is not None:
            shutdown = _stop_process(process)
        report["shutdown"] = shutdown
        report["checkpoint"] = {
            "path": str(run_dir / "latest.pt"),
            "preserved": (run_dir / "latest.pt").is_file(),
        }
        report["finished_at"] = datetime.now().astimezone().isoformat()
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["checkpoint"]["preserved"]:
        raise RuntimeError("Graceful shutdown did not preserve latest.pt")
    return report


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_benchmark(args)
    print(json.dumps({
        "status": report["status"],
        "output_json": str(Path(args.output_json).resolve()),
        "checkpoint": report["checkpoint"],
        "fps": report["render"]["fps"],
        "env_steps_s": report["trainer"]["env_steps_s"],
        "gpu": report["gpu"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
