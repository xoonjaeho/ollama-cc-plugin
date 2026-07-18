---
description: Delegate a coding task to an ollama-model agent that edits files in an isolated worktree; review its diff before it touches your tree
argument-hint: '[--model <name>] [--allow-shell] [--timeout <sec>] <what the ollama agent should do>'
allowed-tools: Bash(python:*), Bash(py:*), Bash(git:*), Bash(mktemp:*), Write, AskUserQuestion, Agent
---

Delegate a task to the ollama agent. It reads and writes files in an **isolated git worktree**; nothing touches your real working tree until you approve the diff.

Raw arguments:
$ARGUMENTS

1. Repo = current working directory. Recognize `--timeout <sec>` as a flag (default `1800`, a whole-run wall-clock cap since a rescue writes files), not part of the task text. Verify it is a git work tree (`git rev-parse --is-inside-work-tree`); if not, tell the user `/ollama:rescue` needs a git repo and stop. **Resolve the model and its cloud-ness authoritatively** — run `python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_companion.py" setup --json`: the model is `--model <name>` if given, else the JSON's `default_model` (which honors `OLLAMA_CC_MODEL` — do NOT hard-code `glm-5.2:cloud`, or a user who set a local default gets a false cloud warning). Read that model's `cloud` flag from the JSON `models` list; fail closed (treat as cloud) if the model is not listed or `models_error` is set.

2. **Launch gate — disclosure + consent.** Use `AskUserQuestion` exactly once. State plainly, before delegating:
   - the ollama agent will read and write files in an isolated worktree off HEAD, and you will review its diff before anything is applied;
   - if the model resolved in step 1 is cloud, **everything the agent reads is sent to ollama.com** (egress). If it is a local model, no egress — you can skip the egress line.
   - **If (and only if) the raw arguments contain `--allow-shell`**, add a second, stronger line: the agent will also get a `run_shell` tool = **arbitrary code execution that is NOT contained by the worktree** — it can read, modify, delete, or exfiltrate any file on this host (including secrets/keys) and reach the network. With a cloud model this is a third party running code on your machine. Only proceed if you truly intend that.
   Options: `Proceed — isolated worktree, review diff before apply (Recommended)` / `Cancel`. On Cancel, stop. (When `--allow-shell` is set, make the proceed option name it, e.g. `Proceed WITH shell/RCE`.)

3. On proceed:
   - **Mint a fresh single-use launch token** (a short-lived nonce the runtime consumes; it fails closed without it):
```bash
TOK=$(mktemp); python -c "import secrets,sys; open(sys.argv[1],'w').write(secrets.token_hex(16))" "$TOK"; echo "$TOK"
```
   - **Write the user's task to a temp file using the Write tool** (get a path with `TASKF=$(mktemp)`, then the Write tool puts the task text into it). Do this with the Write tool, never `echo`/shell — the task is untrusted and must not pass through a shell command.

4. **Delegate** to the `ollama:ollama-rescue` subagent via the `Agent` tool (`subagent_type: "ollama:ollama-rescue"`). Give it in the prompt ONLY these paths, on separate lines: `repo: <cwd>`, `token: <$TOK>`, `task_file: <$TASKF>`, `timeout: <sec>` (the value parsed in step 1, else `1800`), `model: <name>` only if the user specified one, and `allow_shell: true` only if the raw arguments contained `--allow-shell` and the user confirmed the shell/RCE disclosure. **Never put the task text itself into the prompt — only its file path.** The subagent runs the runtime once and returns a JSON report. Do not do the run yourself.

5. Present the report's `final` summary and its `diff`. If `stop_reason` is not `done`, or the `diff` is empty, say so and stop — there is nothing to apply. If `cleanup_error` is present, mention the leaked worktree path.

6. **Apply gate — never auto-apply.** Use `AskUserQuestion`: `Apply the diff to your working tree` / `Discard`. On Discard, stop.

7. On Apply, enforce the base before touching the tree. **Apply from the report's `diff_file` path — never feed patch text through stdin or a heredoc; the diff is untrusted agent output and reconstructing it through the shell is unsafe.**
   - If the report's `dirty_base` is true, warn the user their working tree had uncommitted changes and recommend committing or stashing first, so a failed apply can be cleanly undone.
   - Check `git rev-parse HEAD` still equals the report's `base_sha`. If it moved, warn that the diff was built against a different base and stop.
   - Dry-run first: `git apply --check --3way "<diff_file>"`. If it fails, show the error and stop — do not force.
   - Apply: `git apply --3way "<diff_file>"`.
   - If apply fails or leaves conflict markers: undo exactly this patch with `git apply --reverse "<diff_file>"`, then remove only the files the patch newly created. **Never run a blanket `git checkout -- .`** — it would destroy the user's own uncommitted work and still miss patch-created files.
   - Show `git status --short` so the user sees exactly what landed.

8. Clean up the temp files once done (applied or discarded): remove `$TASKF` and the report's `diff_file`.

Pass `--allow-shell` to the agent **only** when the raw arguments contain `--allow-shell` and the user confirmed the shell/RCE disclosure in step 2 — otherwise never enable it, and the agent gets read/write only.
