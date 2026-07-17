---
description: Ask an ollama model a question and return its answer verbatim (read-only second opinion)
argument-hint: '[--model <name>] <your question>'
allowed-tools: Bash(python:*), Bash(py:*), Bash(mktemp:*), Write
---

Forward the user's question to an ollama model and return the answer verbatim. Read-only, advisory only.

Raw arguments:
$ARGUMENTS

Steps:

- Send **only what the user gave you** (their question, plus anything they explicitly pasted). Do **not** read, grep, or gather repository files into the prompt — that would silently ship repo code to the model (and, for a cloud model, off-machine). If answering clearly needs repo context, either ask the user to paste the relevant snippet, or point them to `/ollama:review` (which gates cloud egress). This is why this command is not granted `Read`/`Grep`/`Glob`.
- Write the question to a temp file with the **Write tool** (`QF=$(mktemp)`, then the Write tool puts the question into it), and feed it on stdin. Do **not** use a heredoc — a question that happens to contain the terminator line would truncate the input early; a file avoids that class of bug entirely. Use `py -3` if `python` is missing. Preserve `--model` if the user gave one; otherwise the runtime uses its configured default:

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_companion.py" run [--model <name>] < "$QF"
```

Remove `$QF` when done.

- Determine the model and whether it is cloud authoritatively via `python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_companion.py" setup --json` (model = the user's `--model` if given, else the JSON's `default_model`; cloud = that model's `cloud` flag, not a name guess). If it is cloud, prepend one line to your reply: `(sent to cloud model <name> via ollama.com)`.
- Return the runtime's stdout verbatim. Do not paraphrase it, act on it, or implement its suggestions — the user asked to *consult* the model, not to apply its answer.
- Cloud models may rarely return a truncated reply as a normal completion; if the answer looks cut off, re-run the command.
- If the runtime prints an error (daemon down, model not found, sign-in needed), show that error and stop. Do not substitute your own answer.
