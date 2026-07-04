---
description: Delegate a task to a FULL Claude Code session powered by an ollama model (ollama launch claude). Full write+shell on your real tree — more dangerous than /ollama:rescue.
argument-hint: '[--model <name>] [--resume <session-id>] <task>'
allowed-tools: Bash(ollama:*), Bash(python:*), Bash(py:*), Bash(mktemp:*), Write, AskUserQuestion
---

Delegate a task to a **full, real Claude Code session** driven by an ollama model, via `ollama launch claude`. The launched session has Claude Code's entire toolset (Read/Write/Bash/Edit), skills, MCP servers, and hooks — but its "brain" is the ollama model. Unlike `/ollama:ask` (raw model, no tools) and `/ollama:rescue` (isolated worktree, diff reviewed before apply), **this runs with full access on your real working tree and host, with no worktree isolation and no diff-review gate.** Treat every run as arbitrary code execution you are authorizing.

Raw arguments:
$ARGUMENTS

1. **Recursion guard.** If `OLLAMA_AS_CLAUDE_ACTIVE` is already set in the environment (`python -c "import os,sys; sys.exit(0 if os.environ.get('OLLAMA_AS_CLAUDE_ACTIVE') else 1)"` → exit 0 means set), you are already running inside an as-claude session. Refuse and stop, so a launched session cannot spawn another and run away.

2. **Resolve model + cloud-ness authoritatively.** Run `python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_companion.py" setup --json`. The model is `--model <name>` if the user gave one, else the JSON's `default_model` (which honors `OLLAMA_CC_MODEL` — never hardcode `glm-5.2:cloud`). **Validate:** the resolved model name MUST appear in the JSON `models` list. If it does not, or `models_error` is set, stop and report — do not launch an unvalidated model name (this both prevents a typo'd model and blocks argument injection through `--model`). Read that model's `cloud` flag from the list; fail closed (treat as cloud) if the model is unlisted.

3. **Validate `--resume`.** If the user passed `--resume <id>`, it MUST match a UUID, case-insensitive: `^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`. If it does not match, stop. If absent, start a fresh session (no resume). This keeps the id from injecting extra flags.

4. **Launch gate — disclosure + consent.** Use `AskUserQuestion` exactly once. State plainly, before launching:
   - This launches a **full Claude Code agent** (all tools, skills, MCP, hooks) powered by the ollama model, run with `--dangerously-skip-permissions`: it can **read, write, delete, and run shell/network commands on your REAL working tree and host, with no per-action approval** — arbitrary code execution.
   - There is **no worktree isolation and no diff review**; changes land directly. For editing tasks, **`/ollama:rescue` is safer** (isolated worktree, you review the diff before it applies). Prefer rescue unless you specifically need the full Claude Code harness on the ollama brain.
   - If the model resolved in step 2 is **cloud**, everything the agent reads AND does is exposed to ollama.com — a third party driving an agent on your machine (egress + remote control), not just a read of one file.
   Options: `Proceed — full-access agent on real tree (RCE)` / `Cancel`. On Cancel, stop.

5. **Write the task to a temp file with the Write tool** (get a path with `TASKF=$(mktemp)`, then the Write tool puts the task text into it). Never pass the task through a shell argument or heredoc — it is fed to the session on **stdin**, so any shell metacharacters or flag-like text in it cannot break the command or inject flags.

6. **Launch and capture the JSON.** The model and resume-id are inserted only after the validation in steps 2–3, so they cannot inject flags. Include `--resume` only if step 3 validated one. **Pipe** the launch stdout into a UTF-8 parser — do **not** redirect it to a file (`> file`); through `ollama launch` the JSON is not reliably captured that way. The parser reconfigures its stdout to UTF-8 so a non-ASCII `result` (e.g. Korean) does not crash on cp949/Windows consoles:
```bash
OLLAMA_AS_CLAUDE_ACTIVE=1 ollama launch claude --model "<validated-model>" -- \
  -p --dangerously-skip-permissions --output-format json [--resume "<validated-id>"] < "$TASKF" 2>/dev/null \
  | python -c "import sys, json; sys.stdout.reconfigure(encoding='utf-8'); d=json.load(sys.stdin); print(json.dumps({k: d.get(k) for k in ('is_error','session_id','total_cost_usd','result')}, ensure_ascii=False))"
```
Remove the task file when done: `rm -f "$TASKF"` (single file — never `rm -rf`).

7. **Report.**
   - Relay the JSON's `result` verbatim — that is the session's final answer / account of what it did.
   - Print the `session_id` and tell the user they can continue it: `/ollama:as-claude --resume <session_id> <follow-up>`.
   - Note once: the JSON's `total_cost_usd` is **Claude Code's own estimate against a placeholder price, not ollama's actual billing** — real usage is governed by your ollama plan.
   - If `is_error` is true, or the output is not valid JSON, show what came back and stop — do not paper over it.
