---
description: Delegate a task to a FULL Claude Code session powered by an ollama model. Default runs in an isolated git worktree with diff-review apply-gate; --no-worktree keeps full direct access.
argument-hint: '[--model <name>] [--resume <session-id>] [--no-worktree] [--timeout <sec>] [--permission-mode <mode>] <task>'
allowed-tools: Bash(ollama:*), Bash(python:*), Bash(py:*), Bash(git:*), Bash(mktemp:*), Bash(timeout:*), Write, AskUserQuestion
---

Delegate a task to a **full, real Claude Code session** driven by an ollama model, via `ollama launch claude`. The launched session has Claude Code's entire toolset (Read/Write/Bash/Edit), skills, MCP servers, and hooks — but its "brain" is the ollama model. **By default this runs in an isolated git worktree off HEAD; you review the diff before it applies to your real tree.** The worktree is built off HEAD, so **uncommitted work is absent** — use `--no-worktree` for an uncommitted diff (it runs directly on your real tree, no isolation, no diff gate).

**It launches without asking.** There is no consent prompt: `/ollama:as-claude` starts a full-access agent (RCE on this host, and cloud egress if the model is cloud). The step-5 apply gate protects the repo's *git state* — it is not a sandbox boundary: the session's shell can write anywhere on this host, including the original checkout by absolute path, before you are ever asked to apply.

Raw arguments:
$ARGUMENTS

1. **Recursion guard.** If `OLLAMA_AS_CLAUDE_ACTIVE` is already set in the environment (`python -c "import os,sys; sys.exit(0 if os.environ.get('OLLAMA_AS_CLAUDE_ACTIVE') else 1)"` → exit 0 means set), you are already running inside an as-claude session. Refuse and stop.

2. **Resolve model + cloud-ness authoritatively.** Recognize `--timeout <sec>` as a flag (default `1800`, a whole-run wall-clock cap), not part of the task text; it bounds the worktree launch (step 5). Recognize `--permission-mode <mode>` the same way — one of `acceptEdits` / `auto` / `bypassPermissions` / `default` / `dontAsk` / `plan`, default `bypassPermissions`; anything else, stop and report rather than passing it through. Note for the user which one is in effect: only `bypassPermissions` runs unattended end to end — the others route tool calls through Claude Code's classifier or prompts, and in headless `-p` a non-approved call is **denied and the session continues**, so the run can come back partially done. The same mode is also stricter in a worktree than on the real tree: a worktree checkout has only tracked files, so an untracked `.claude/settings.local.json` allowlist is absent there, and a gitignored `.claude/rules/` is invisible too — the launched session then follows no project conventions without saying so. Check what the worktree actually sees with `git -C <repo> ls-files .claude | head`. Run `python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_companion.py" setup --json`. The model is `--model <name>` if the user gave one, else the JSON's `default_model` (which honors `OLLAMA_CC_MODEL` — never hardcode `glm-5.2:cloud`). **Validate:** the resolved model name MUST appear in the JSON `models` list. If it does not, or `models_error` is set, stop and report — do not launch an unvalidated model name (this both prevents a typo'd model and blocks argument injection through `--model`). Read that model's `cloud` flag from the list; fail closed (treat as cloud) if the model is unlisted.

3. **Validate `--resume`.** If the user passed `--resume <id>`, it MUST match a UUID, case-insensitive: `^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`. If it does not match, stop. If absent, start a fresh session. `--resume` forces a **direct** run (not worktree), because `claude --resume` is cwd-scoped and the worktree path is deleted after the run.

4. **Choose mode.**
   - If `--resume` or `--no-worktree` is present, go to step 6 (direct launch).
   - Else check `git rev-parse --is-inside-work-tree`. If true, go to step 5 (worktree launch). If false, warn that worktree isolation requires a git repo and fall back to step 6 (direct launch).

5. **Worktree launch + apply gate (default).**
   - **Launch disclosure — state it, do not ask.** In one line before launching: full Claude Code agent in a worktree off HEAD on `<model>`, permission mode `<mode>`; the worktree protects repo *state* only — the session's shell reaches the whole host — and a cloud model means everything it reads and does goes to ollama.com.
   - **Mint task file.** `TASKF=$(python -c "import tempfile,os; print((os.path.join(tempfile.mkdtemp(),'task.md')).replace(os.sep,'/'))")` (use `py -3` if `python` is missing). Use the `Write` tool to put the task text into `$TASKF`.
   - **Delegate to the worktree runtime.** Capture the JSON report to a temp file; do not feed untrusted patch text through stdin or a heredoc later.
     Use the `--timeout <sec>` parsed in step 2 (else `1800`); it is a hard wall-clock cap — the launched session's whole process tree is killed if it runs past it.
     ```bash
     REPORTF=$(mktemp)
     python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_agent.py" --as-claude --repo "<cwd>" --task-file "$TASKF" --model "<validated-model>" --timeout <sec> --permission-mode <mode> > "$REPORTF"
     ```
     `--as-claude` requires `--repo`; `--root` is rejected — the worktree is always built from the repo, never a bare root.
   - **Inspect the report.** Use a surrogate-safe UTF-8 parser to surface key fields:
     ```bash
     python -c "import sys, json; sys.stdout.reconfigure(encoding='utf-8', errors='replace'); d=json.load(open(sys.argv[1], encoding='utf-8')); [print(k+':', d.get(k,'')) for k in ('is_error','stop_reason','session_id','base_sha','dirty_base','has_diff','permission_denials','cleanup_error','diff_error','diff_file')]; print('--- result ---'); print(d.get('result') or d.get('final',''))" "$REPORTF"
     ```
     The `--as-claude` path reports `result` (from the launched Claude session). The read-only `--root --readonly` consult path used by `/ollama:adversarial-review` reports `final` instead — and a `tool_call_cap` stop there still carried a complete report, so read `final` first. When parsing raw `ollama launch claude --output-format json` output by hand, warning banners may be prepended; slice from the first `{"is_error"` rather than parsing from byte 0 (the worktree path's `ollama_agent.py`/`_launch_claude` already strips them — this bites hand-rolled captures).
   - **Extract gate variables** (paths/IDs generated by the helper, not the untrusted patch text). Do this BEFORE deciding whether to stop, so a nonzero-but-productive session's diff is not thrown away and cleanup can reach `$DIFF_FILE`:
     ```bash
     DIFF_FILE=$(python -c "import json,sys; sys.stdin.reconfigure(encoding='utf-8'); print(json.load(sys.stdin).get('diff_file',''))" < "$REPORTF")
     BASE_SHA=$(python -c "import json,sys; sys.stdin.reconfigure(encoding='utf-8'); print(json.load(sys.stdin).get('base_sha',''))" < "$REPORTF")
     HAS_DIFF=$(python -c "import json,sys; sys.stdin.reconfigure(encoding='utf-8'); print('yes' if json.load(sys.stdin).get('has_diff') else 'no')" < "$REPORTF")
     DIRTY_BASE=$(python -c "import json,sys; sys.stdin.reconfigure(encoding='utf-8'); print('yes' if json.load(sys.stdin).get('dirty_base') else 'no')" < "$REPORTF")
     SOURCE_FILES=$(python -c "import json,sys; sys.stdin.reconfigure(encoding='utf-8'); print(json.load(sys.stdin).get('source_files',0))" < "$REPORTF")
     ```
     **Cleanup command** — run it before EVERY stop below (applied, discarded, or error) so no task/patch temp file lingers; it uses only the already-allowed Python tool:
     ```bash
     python -c "import os,sys,shutil; shutil.rmtree(os.path.dirname(sys.argv[1]), ignore_errors=True); [os.remove(p) for p in sys.argv[2:] if p and os.path.exists(p)]" "$TASKF" "$REPORTF" "$DIFF_FILE" "${DIFF_FILE:+$DIFF_FILE.lease}"
     ```
   - **Decide — never auto-apply.** Judge on the captured diff, not on `is_error` alone:
     - Report is not valid JSON → show the raw output, run cleanup, and stop.
     - `diff_error` present → be explicit: the session made changes but the diff could not be captured and the worktree is already removed, so its work is lost (do NOT report "no changes"). Run cleanup and stop.
     - `HAS_DIFF` is not `yes` → nothing to apply. If `is_error` is true, say the session reported an error and produced no reviewable changes; else say there were no changes. **If `permission_denials` is above 0, lead with that instead** — the permission mode blocked N tool calls, so "no changes" means "it was not allowed to make them"; name the mode and suggest re-running at `bypassPermissions`. Run cleanup and stop.
     - `HAS_DIFF` is `yes` and `SOURCE_FILES` is `0` → the captured patch holds only build artifacts; the agent's real work most likely landed in the **main tree**, not the patch. Tell the user the patch contains no source files, and to run `git status` in the real repo to find the work. Do **NOT** offer the artifact-only patch at the apply gate. Run cleanup and stop.
     - `HAS_DIFF` is `yes` → proceed to the apply gate. **If `is_error` is also true, surface it as a warning** — the session reported an error but still made edits you can review and apply; do not discard them.
     - If `cleanup_error` is present, mention the leaked worktree path.
   - **Apply gate.** Show the `result` summary and the diff (`python -c "import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace'); print(open(sys.argv[1],'rb').read().decode('utf-8','replace'))" "$DIFF_FILE"`). Use `AskUserQuestion`: `Apply the diff to your working tree` / `Discard`. On Discard, run cleanup and stop.
     The captured patch includes `.gitignore`d files by design (`git add -Af`, so agent work cannot vanish), so it can be large — budget for it: apply with an exclude for `__pycache__`, and put "delete `__pycache__/` before you finish" in the task packet. To review it, split on `diff --git ` and list paths first rather than dumping the raw patch.
   - On Apply, reuse `/ollama:rescue` steps 6–8 (run cleanup before each stop below, too):
     - If `DIRTY_BASE` is `yes`, warn the user their working tree had uncommitted changes and recommend committing or stashing first, so a failed apply can be cleanly undone.
     - Check `git rev-parse HEAD` still equals `$BASE_SHA`. If it moved, warn that the diff was built against a different base, run cleanup, and stop.
     - Dry-run first: `git apply --check --3way "$DIFF_FILE"`. If it fails, show the error, run cleanup, and stop — do not force.
     - Apply: `git apply --3way "$DIFF_FILE"`.
     - If apply fails or leaves conflict markers: undo exactly this patch with `git apply --reverse "$DIFF_FILE"`, then remove only the files the patch newly created. **Never run a blanket `git checkout -- .`** — it would destroy the user's own uncommitted work and still miss patch-created files.
     - Show `git status --short` so the user sees exactly what landed, then run cleanup.
   - **Do not advertise a resume ID.** A worktree run is throwaway and **non-resumable by design**: sessions are cwd-scoped, so one born in an ephemeral worktree dies with it. Do not print a `/ollama:as-claude --resume <id>` hint.

6. **Direct launch (`--no-worktree`, `--resume`, or non-git cwd fallback).** The parsed `--timeout <sec>` (default `1800`) bounds this path via coreutils `timeout`; the launch block below resolves the binary (`timeout`, else macOS/Homebrew's `gtimeout`) and, if neither is installed, runs uncapped and says so rather than failing. Note: the calling Bash tool's own maximum is `600s`, so a run longer than that must use `run_in_background` or a sub-600s `--timeout`.
   - **Launch disclosure — state it, do not ask.** In one line before launching: full Claude Code agent (all tools, skills, MCP, hooks) on `<model>`, permission mode `<mode>`, running on your **real** tree with no isolation and no diff review — changes land directly, and at `bypassPermissions` that is arbitrary code execution on this host. A cloud model means everything it reads and does is exposed to ollama.com.
   - **Mint task file.** `TASKF=$(python -c "import tempfile,os; print((os.path.join(tempfile.mkdtemp(),'task.md')).replace(os.sep,'/'))")` (use `py -3` if `python` is missing). Use the `Write` tool to put the task text into `$TASKF`.
   - **Launch and capture the JSON.** `<permission-flags>` is **exactly one of these six literals, copied verbatim** — never build it by interpolating the user's text, so nothing from `$ARGUMENTS` reaches the shell: `--dangerously-skip-permissions` (for `bypassPermissions`, the default) · `--permission-mode acceptEdits` · `--permission-mode auto` · `--permission-mode default` · `--permission-mode dontAsk` · `--permission-mode plan`. If step 2's value is not one of the six modes, stop — do not pass it through. Never emit two of these literals at once: `--dangerously-skip-permissions` wins and the mode the user asked for is silently ignored. Include `--resume` only if step 3 validated one. Pipe the launch stdout into a UTF-8 parser that is surrogate-safe so a non-ASCII `result` does not crash the report. The `$WRAP` prefix below applies the cap from step 6's header; when no timeout binary exists it is empty and word-splits to nothing, leaving the launch unbounded:
     ```bash
     if command -v timeout >/dev/null 2>&1; then WRAP="timeout <sec>"; elif command -v gtimeout >/dev/null 2>&1; then WRAP="gtimeout <sec>"; else WRAP=""; echo "warning: neither timeout nor gtimeout found; running uncapped" >&2; fi
     OLLAMA_AS_CLAUDE_ACTIVE=1 $WRAP ollama launch claude --model "<validated-model>" -- \
       -p <permission-flags> --output-format json [--resume "<validated-id>"] < "$TASKF" 2>/dev/null \
       | python -c "import sys, json; sys.stdout.reconfigure(encoding='utf-8', errors='replace'); sys.stdin.reconfigure(encoding='utf-8'); d=json.load(sys.stdin); print(json.dumps({k: d.get(k) for k in ('is_error','session_id','total_cost_usd','result')}, ensure_ascii=False))"
     ```
   - Remove the task directory before returning (including on an error stop), via the already-allowed Python tool: `python -c "import os,sys,shutil; shutil.rmtree(os.path.dirname(sys.argv[1]), ignore_errors=True)" "$TASKF"`.
   - **Report.**
     - Relay the JSON's `result` verbatim — that is the session's final answer / account of what it did.
     - Print the `session_id` and tell the user they can continue it: `/ollama:as-claude --resume <session_id> <follow-up>`.
     - Note once: the JSON's `total_cost_usd` is **Claude Code's own estimate against a placeholder price, not ollama's actual billing** — real usage is governed by your ollama plan.
     - If `is_error` is true, or the output is not valid JSON, show what came back and stop — do not paper over it.
     - If the launch did NOT exit cleanly (a `timeout`/`gtimeout` kill, a non-zero exit, or output that is not valid JSON), the direct path has no diff capture and edits the **real** tree — so partial edits may be sitting in your working tree with no JSON to describe them. Run `git status --short` in the real repo and surface the result, leading with the fact that the session may have left partial edits. (This path has no worktree — do NOT copy the leaked-worktree recovery from step 5.)
