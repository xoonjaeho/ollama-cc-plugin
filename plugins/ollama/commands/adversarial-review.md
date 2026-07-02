---
description: Adversarially review your repo with an ollama model that explores it read-only; optionally steer the focus
argument-hint: '[--model <name>] [--base <ref>] [focus: what to challenge]'
disable-model-invocation: true
allowed-tools: Bash(python:*), Bash(py:*), Bash(git:*), Bash(mktemp:*), Write, AskUserQuestion
---

Run a READ-ONLY adversarial review: an ollama model explores your repo (`read_file`/`list_dir`/`grep_search` — no writes, no shell) and challenges the design and implementation, steered by your focus text.

Raw arguments:
$ARGUMENTS

1. Repo = current working directory. The model is `--model <name>` if given, else the runtime default (resolved authoritatively as `default_model` by step 2's `setup --json` — do not hard-code `glm-5.2:cloud`). Parse any focus text (everything after the flags) — e.g. "challenge the retry logic", "look for race conditions". If `--base <ref>` is given, run `git diff --stat <ref>...HEAD` to scope the review to recent changes.

2. **Cloud egress gate.** Determine whether the model is a cloud model: run `python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_companion.py" setup --json` and read the `cloud` flag for that model (fail closed — treat as cloud if unknown). If it is cloud, the files the model chooses to read are sent to ollama.com. Use `AskUserQuestion` once: `Proceed — read-only, sends read files to <model>` / `Cancel`. On Cancel, stop.

3. **Write the review task to a temp file with the Write tool** (`TASKF=$(mktemp)`, then the Write tool — never `echo`/shell, the focus text is untrusted). The task: adversarially review this repo — find bugs, risks, design flaws, unhandled edge cases — using `list_dir`/`grep_search`/`read_file` to explore; be specific with `file:line`; order by severity; end with a one-line verdict. Append the user's focus text if any, and the `git diff --stat` output from step 1 if `--base` was given.

4. Run the read-only agent (use `py -3` if `python` is missing):
```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_agent.py" --root "<cwd>" --readonly --task-file "<TASKF>" [--model <model>]
```

5. Return the report's `final` (the review) verbatim. This is read-only and advisory — do NOT fix anything. After presenting the findings, STOP and ask which issues (if any) the user wants addressed. Then remove the temp task file (`$TASKF`).
