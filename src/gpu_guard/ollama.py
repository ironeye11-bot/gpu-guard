"""Talk to the Ollama CLI. No model weights are read or uploaded."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence


class GpuGuardError(RuntimeError):
    """Base error for GPU Guard."""


class OllamaDown(GpuGuardError):
    """ollama binary missing, timed out, or not responding."""


class StopFailed(GpuGuardError):
    """ollama stop returned a non-zero exit code."""


Runner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]

# Longest marks first so 18b is not mistaken for 8b, 1.5b before 5b, etc.
_SIZE_RE = re.compile(
    r"(?<![0-9.])(72b|70b|32b|27b|18b|14b|8b|7b|3b|1\.5b)(?![0-9a-z])",
    re.IGNORECASE,
)


def _binary() -> str:
    return os.environ.get("OLLAMA_BINARY", "ollama")


def _default_runner(args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise OllamaDown(f"{args[0]} is not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise OllamaDown(f"{args[0]} timed out after {timeout:.0f}s") from exc


def classify_size(name: str) -> str:
    """Cheap size hint from a model tag. Used only for display."""
    match = _SIZE_RE.search(name or "")
    return match.group(1).lower() if match else "unknown"


def parse_ps(stdout: str) -> list[str]:
    """Parse `ollama ps` table. First column is the model name."""
    models: list[str] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or line.upper().startswith("NAME"):
            continue
        name = line.split()[0]
        if name and name not in models:
            models.append(name)
    return models


def loaded_models(runner: Runner | None = None) -> list[str]:
    run = runner or _default_runner
    proc = run([_binary(), "ps"], 10)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise OllamaDown(err or "ollama ps failed")
    return parse_ps(proc.stdout or "")


def stop_model(name: str, runner: Runner | None = None) -> None:
    target = name.strip()
    if not target:
        raise ValueError("model name is empty")
    run = runner or _default_runner
    proc = run([_binary(), "stop", target], 20)
    if proc.returncode != 0:
        raise StopFailed((proc.stderr or proc.stdout or f"failed to stop {target}").strip())


def stop_all(runner: Runner | None = None) -> list[str]:
    loaded = loaded_models(runner)
    stopped: list[str] = []
    for name in loaded:
        stop_model(name, runner)
        stopped.append(name)
    return stopped


@dataclass(frozen=True)
class GuardResult:
    target: str
    loaded_before: tuple[str, ...]
    still_loaded: tuple[str, ...]
    stopped: tuple[str, ...]
    already_resident: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "loaded_before": list(self.loaded_before),
            "still_loaded": list(self.still_loaded),
            "stopped": list(self.stopped),
            "already_resident": self.already_resident,
        }


def ensure_only(model_name: str, runner: Runner | None = None) -> GuardResult:
    """Stop every loaded model except ``model_name``. Does not pull or load it."""
    target = model_name.strip()
    if not target:
        raise ValueError("model name is empty")
    loaded = loaded_models(runner)
    stopped: list[str] = []
    for name in loaded:
        if name == target:
            continue
        stop_model(name, runner)
        stopped.append(name)
    still = tuple(name for name in loaded if name == target)
    return GuardResult(
        target=target,
        loaded_before=tuple(loaded),
        still_loaded=still,
        stopped=tuple(stopped),
        already_resident=target in loaded,
    )
