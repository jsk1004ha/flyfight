from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

import live_benchmark


def test_find_free_port_returns_bindable_loopback_port():
    port = live_benchmark.find_free_port()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def test_build_trainer_command_uses_requested_small_configuration(tmp_path: Path):
    args = live_benchmark.parse_args([
        "--run-dir", str(tmp_path / "run"),
        "--device", "cpu",
        "--envs", "6",
        "--hidden", "64",
        "--width", "12",
        "--height", "8",
        "--horizon", "8",
        "--epochs", "1",
        "--minibatch-envs", "3",
        "--threads", "2",
    ])
    command = live_benchmark.build_trainer_command(args, 23456)
    assert command[:2] == [sys.executable, "run.py"]
    assert command[command.index("--port") + 1] == "23456"
    assert command[command.index("--envs") + 1] == "6"
    assert command[command.index("--run-dir") + 1] == str(tmp_path / "run")


def test_normal_config_uses_project_training_defaults_when_sizes_are_unspecified():
    args = live_benchmark.parse_args(["--normal-config"])
    defaults = live_benchmark.Config()
    assert args.envs == defaults.envs
    assert args.hidden == defaults.hidden
    assert args.width == defaults.width
    assert args.height == defaults.height


def test_summarize_render_reports_stutter_and_scale_range():
    summary = live_benchmark.summarize_render(
        fps_samples=[60.0, 54.0, 58.0],
        scale_samples=[1.0, 0.85, 0.85],
        frame_ms=[16.0, 17.0, 40.0, 15.0],
        stutter_ms=33.34,
    )
    assert summary["fps"]["mean"] == pytest.approx(57.333, abs=0.001)
    assert summary["render_scale"] == {"min": 0.85, "max": 1.0, "final": 0.85}
    assert summary["frame_timing_ms"]["stutter_count"] == 1
    assert summary["frame_timing_ms"]["stutter_ratio"] == pytest.approx(0.25)


def test_cuda_unavailable_is_reported_as_uncollected(monkeypatch):
    monkeypatch.setattr(live_benchmark.torch.cuda, "is_available", lambda: False)
    result = live_benchmark.describe_gpu_metrics([])
    assert result["collected"] is False
    assert result["cuda_available"] is False
    assert "CUDA" in result["reason"]


def test_validate_native_samples_rejects_page_without_websocket_frames():
    with pytest.raises(RuntimeError, match="WebSocket"):
        live_benchmark.validate_native_samples([{"native": False}, {"native": False}])
