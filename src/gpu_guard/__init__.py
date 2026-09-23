"""Keep a consumer GPU from loading more than one local LLM at a time."""

from gpu_guard.ollama import (
    GpuGuardError,
    OllamaDown,
    StopFailed,
    classify_size,
    ensure_only,
    loaded_models,
    stop_all,
    stop_model,
)

__version__ = "1.0.0"
__all__ = [
    "GpuGuardError",
    "OllamaDown",
    "StopFailed",
    "classify_size",
    "ensure_only",
    "loaded_models",
    "stop_all",
    "stop_model",
    "__version__",
]
