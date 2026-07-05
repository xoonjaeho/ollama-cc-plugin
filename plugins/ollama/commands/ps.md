---
description: List ollama models currently loaded in memory (running)
argument-hint: ''
allowed-tools: Bash(python:*), Bash(py:*)
---

Show which ollama models are currently loaded in memory. Read-only.

Run (use `py -3` if `python` is missing; if neither exists, tell the user to install Python 3.x on PATH and stop):

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_companion.py" ps
```

Present the output as-is (model name, memory size, VRAM, expiry). If it reports none loaded, say so. If the daemon is down (the output says it is not reachable), show that message verbatim and stop.
