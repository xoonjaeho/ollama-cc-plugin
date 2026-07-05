---
description: Delete an installed ollama model (confirms before deleting)
argument-hint: '<model>'
allowed-tools: Bash(python:*), Bash(py:*), AskUserQuestion
---

Delete one installed ollama model. This is destructive (reversible only by re-pulling), so **confirm before deleting** — never delete without an explicit yes.

Raw arguments:
$ARGUMENTS

Steps:

1. Run the preview (no `--yes`) with the model the user named (use `py -3` if `python` is missing):

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_companion.py" rm <model>
```

   Exit code 10 = confirmation required (the line shows the model and the disk it frees). Exit code 4 = not installed (tell the user; stop).

2. Show that line and get an explicit choice via `AskUserQuestion` (options: `Delete <model>` / `Cancel`). Delete only the ONE model the user named — never expand to several or to all models.

3. Only after the user confirms, run:

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_companion.py" rm <model> --yes
```

   Confirm the result. If it errors, show it verbatim.
