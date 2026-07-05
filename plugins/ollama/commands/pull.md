---
description: Download/install an ollama model (confirms before the download)
argument-hint: '<model>'
allowed-tools: Bash(python:*), Bash(py:*), AskUserQuestion
---

Install (download) an ollama model. A model can be several GB, so **confirm before downloading** — never pull without an explicit yes from the user.

Raw arguments:
$ARGUMENTS

Steps:

1. Run the preview (no `--yes`) with the model the user named (use `py -3` if `python` is missing):

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_companion.py" pull <model>
```

   This does **not** download. Exit code 10 means "confirmation required"; the printed line says whether the model is new or already installed.

2. Show that line to the user and get an explicit choice via `AskUserQuestion` (options: `Download <model>` / `Cancel`). Proceed only on an explicit yes.

3. Only after the user confirms, run the real download:

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_companion.py" pull <model> --yes
```

   Relay progress and the final status. If it errors (daemon down, not found, sign-in needed), show the error verbatim and stop.
