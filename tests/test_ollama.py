from __future__ import annotations

import subprocess
import unittest
from typing import Sequence

from gpu_guard.cli import main as cli_main
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

    def test_classify_size_no_false_positive_18b(self) -> None:
        """18b must not match the 8b substring."""
        self.assertEqual(classify_size("foo:18b"), "18b")
        self.assertEqual(classify_size("model-18b-instruct"), "18b")

    def test_classify_size_matrix(self) -> None:
        cases = {
            "qwen2.5-coder:14b": "14b",
            "llama3.1:8b": "8b",
            "foo:18b": "18b",
            "bar:72b": "72b",
            "x:1.5b": "1.5b",
            "custom": "unknown",
            "qwen:32b": "32b",
            "model:70b": "70b",
            "mistral:7b": "7b",
            "phi:3b": "3b",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(classify_size(name), expected)


class CliJsonTests(unittest.TestCase):
    def test_json_after_subcommand_is_accepted(self) -> None:
        # argparse only — must not SystemExit(2) for unrecognized --json
        with self.assertRaises(SystemExit) as ctx:
            # --help exits 0; use parse via main with --help on only
            cli_main(["only", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_json_flag_in_help_for_only(self) -> None:
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                cli_main(["only", "--help"])
            except SystemExit:
                pass
        self.assertIn("--json", buf.getvalue())


    def test_json_before_and_after_subcommand(self) -> None:
        """Both `gpu-guard --json only x` and `gpu-guard only x --json` must emit JSON."""
        import io
        from contextlib import redirect_stdout, redirect_stderr
        from gpu_guard import ollama

        class Fake:
            def __call__(self, args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
                cmd = list(args)
                if cmd[1] == "ps":
                    return subprocess.CompletedProcess(cmd, 0, "NAME\n", "")
                if cmd[1] == "stop":
                    return subprocess.CompletedProcess(cmd, 0, "", "")
                return subprocess.CompletedProcess(cmd, 1, "", "unknown")

        original = ollama._default_runner
        ollama._default_runner = Fake()
        try:
            for argv in (["only", "x", "--json"], ["--json", "only", "x"]):
                buf = io.StringIO()
                err = io.StringIO()
                with redirect_stdout(buf), redirect_stderr(err):
                    code = cli_main(argv)
                self.assertEqual(code, 0, argv)
                out = buf.getvalue()
                self.assertIn('"target"', out, argv)
                self.assertIn('"stopped"', out, argv)
                self.assertNotIn("was loaded", out, argv)
        finally:
            ollama._default_runner = original

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
