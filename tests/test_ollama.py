from __future__ import annotations

import subprocess
import unittest
from typing import Sequence

from gpu_guard.ollama import (
    OllamaDown,
    StopFailed,
    classify_size,
    ensure_only,
    loaded_models,
    parse_ps,
    stop_all,
)


PS_TABLE = """\
NAME                   ID              SIZE      PROCESSOR    CONTEXT    UNTIL
qwen2.5-coder:14b      abc             9.0 GB    100% GPU     8192       Forever
llama3.1:8b            def             4.9 GB    100% GPU     4096       Forever
"""


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.loaded = ["qwen2.5-coder:14b", "llama3.1:8b"]

    def __call__(self, args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(args))
        cmd = list(args)
        if cmd[1] == "ps":
            rows = "NAME ID SIZE\n" + "\n".join(f"{name} x 1GB" for name in self.loaded)
            return subprocess.CompletedProcess(cmd, 0, rows, "")
        if cmd[1] == "stop":
            name = cmd[2]
            if name in self.loaded:
                self.loaded.remove(name)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 1, "", "unknown")


class ParseTests(unittest.TestCase):
    def test_parse_skips_header(self) -> None:
        self.assertEqual(
            parse_ps(PS_TABLE),
            ["qwen2.5-coder:14b", "llama3.1:8b"],
        )

    def test_classify_size(self) -> None:
        self.assertEqual(classify_size("qwen2.5-coder:14b"), "14b")
        self.assertEqual(classify_size("llama3.1:8b"), "8b")
        self.assertEqual(classify_size("custom"), "unknown")


class GuardTests(unittest.TestCase):
    def test_ensure_only_stops_the_other_model(self) -> None:
        runner = FakeRunner()
        result = ensure_only("qwen2.5-coder:14b", runner)
        self.assertEqual(result.stopped, ("llama3.1:8b",))
        self.assertTrue(result.already_resident)
        self.assertEqual(runner.loaded, ["qwen2.5-coder:14b"])

    def test_ensure_only_does_not_preload_missing_target(self) -> None:
        runner = FakeRunner()
        result = ensure_only("mistral:7b", runner)
        self.assertEqual(set(result.stopped), {"qwen2.5-coder:14b", "llama3.1:8b"})
        self.assertFalse(result.already_resident)
        self.assertEqual(runner.loaded, [])

    def test_stop_all(self) -> None:
        runner = FakeRunner()
        stopped = stop_all(runner)
        self.assertEqual(stopped, ["qwen2.5-coder:14b", "llama3.1:8b"])
        self.assertEqual(runner.loaded, [])

    def test_empty_target(self) -> None:
        with self.assertRaises(ValueError):
            ensure_only("  ", FakeRunner())

    def test_ollama_down(self) -> None:
        def boom(args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(list(args), 1, "", "connection refused")

        with self.assertRaises(OllamaDown):
            loaded_models(boom)

    def test_stop_failed(self) -> None:
        def bad(args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
            if list(args)[1] == "ps":
                return subprocess.CompletedProcess(list(args), 0, "NAME\nfoo x\n", "")
            return subprocess.CompletedProcess(list(args), 1, "", "busy")

        with self.assertRaises(StopFailed):
            ensure_only("bar", bad)


if __name__ == "__main__":
    unittest.main()
