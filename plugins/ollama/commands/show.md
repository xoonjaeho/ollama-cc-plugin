---
description: Show details of an ollama model (family, parameters, quantization, size)
argument-hint: '<model>'
allowed-tools: Bash(python:*), Bash(py:*)
---

Show details for one model. Read-only.

Raw arguments:
$ARGUMENTS

Run with the model name the user gave (use `py -3` if `python` is missing):

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_companion.py" show <model>
```

Present the output as-is. If the model is not found (exit code 4), relay the "pull it" hint and stop. If the daemon is down, show that message verbatim.
