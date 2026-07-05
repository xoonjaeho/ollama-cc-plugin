---
description: List installed / available ollama models
argument-hint: ''
allowed-tools: Bash(python:*), Bash(py:*)
---

List the ollama models available to use (installed locally + registered cloud models). Read-only.

Run (use `py -3` if `python` is missing; if neither exists, tell the user to install Python 3.x on PATH and stop):

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_companion.py" list
```

Present the output as-is (name, `[cloud]` marker, size). If none are installed, relay the pull hint. If the daemon is down, show that message verbatim and stop.
