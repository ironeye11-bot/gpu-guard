"""Command line for GPU Guard."""

from __future__ import annotations

import argparse
import json
import sys

from gpu_guard import __version__
from gpu_guard.ollama import (
    GpuGuardError,
    classify_size,
    ensure_only,
    loaded_models,
    stop_all,
    stop_model,
)


def _print_models(models: list[str]) -> None:
    if not models:
        print("no models loaded")
        return
    width = max(len(name) for name in models)
    print(f"{'MODEL'.ljust(width)}  SIZE")
    for name in models:
        print(f"{name.ljust(width)}  {classify_size(name)}")


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    # SUPPRESS so a global --json is not wiped by the subparser default False.
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="machine-readable output",
    )


def _want_json(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "json", False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gpu-guard",
        description="Keep one local LLM in VRAM. Stop the rest.",
    )
    parser.add_argument("--version", action="version", version=f"gpu-guard {__version__}")
    # Global --json (before subcommand): gpu-guard --json only model
    _add_json_flag(parser)
    sub = parser.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("ps", help="list models currently in VRAM")
    _add_json_flag(ps)

    only = sub.add_parser("only", help="stop every model except this one")
    only.add_argument("model", help="model tag, e.g. qwen2.5-coder:14b")
    _add_json_flag(only)

    stop = sub.add_parser("stop", help="stop one loaded model")
    stop.add_argument("model")
    _add_json_flag(stop)

    stop_all_p = sub.add_parser("stop-all", help="unload every model")
    _add_json_flag(stop_all_p)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "ps":
            models = loaded_models()
            if _want_json(args):
                print(json.dumps({"loaded": models}, indent=2))
            else:
                _print_models(models)
            return 0
        if args.cmd == "only":
            result = ensure_only(args.model)
            if _want_json(args):
                print(json.dumps(result.as_dict(), indent=2))
            else:
                print(f"target          {result.target}")
                print(f"was loaded      {', '.join(result.loaded_before) or '—'}")
                print(f"stopped         {', '.join(result.stopped) or '—'}")
                print(f"still in VRAM   {', '.join(result.still_loaded) or 'none (not preloaded)'}")
            return 0
        if args.cmd == "stop":
            stop_model(args.model)
            payload = {"stopped": [args.model]}
            print(json.dumps(payload, indent=2) if _want_json(args) else f"stopped {args.model}")
            return 0
        if args.cmd == "stop-all":
            stopped = stop_all()
            if _want_json(args):
                print(json.dumps({"stopped": stopped}, indent=2))
            else:
                print("stopped " + ", ".join(stopped) if stopped else "nothing loaded")
            return 0
    except GpuGuardError as exc:
        print(f"gpu-guard: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"gpu-guard: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
