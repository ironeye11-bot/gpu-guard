# GPU Guard

Ein lokales LLM im VRAM. Der Rest fliegt runter.

<p align="center">
  <img src="docs/hero.jpg" alt="product photo" width="100%">
</p>
<p align="center">
  <img src="docs/hero.svg" alt="GPU Guard — one local model in VRAM" width="100%">
</p>

**12 GB cards choke when two models sit in VRAM.**  
GPU Guard talks to the Ollama CLI and unloads everything except the one model you want. It does not pull weights. It does not start a server. It does not send telemetry.

<p align="center">
  <img src="docs/terminal.svg" alt="gpu-guard only example" width="86%">
</p>

## Install

```bash
pip install git+https://github.com/ironeye11-bot/gpu-guard.git
```

Needs `ollama` on `PATH`. Override the binary with `OLLAMA_BINARY` if you have to.

## Use

```bash
gpu-guard ps
gpu-guard only qwen2.5-coder:14b
gpu-guard stop llama3.1:8b
gpu-guard stop-all
gpu-guard --json only qwen2.5-coder:14b
```

`only` **never loads** the target. If it is not already resident, the other models are still stopped and VRAM is free for *you* to load it.

## Library

```python
from gpu_guard import ensure_only, loaded_models

print(loaded_models())
ensure_only("qwen2.5-coder:14b")
```

## Why this exists

Local coding agents stack a 14B and an 8B “just in case”. On a 3060 / 4060 / 4070 laptop that is a freeze, not a feature. One command, one occupant.

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

MIT
