# ollama plugin for Claude Code

Use local or cloud [ollama](https://ollama.com) models from inside Claude Code as a **second engine** — a quick second opinion, a code review from a different model, or a delegated agentic task, without leaving your workflow.

Two tiers:

- **Read-only consult/review** — a single stateless call to the local ollama daemon (`ollama_companion.py`).
- **Agentic rescue** — an ollama model runs a tool loop (read/write files, optionally shell) inside a throwaway **git worktree** and hands you a diff to review (`ollama_agent.py`). Modeled on the Codex plugin's rescue.

## What you get

| Command | Tier | What it does |
|---|---|---|
| `/ollama:setup` | — | check the daemon, list models, flag cloud sign-in |
| `/ollama:ask` | read-only | ask a model a question, answer returned verbatim |
| `/ollama:review` | read-only | review your git changes with a model (cloud egress gated) |
| `/ollama:adversarial-review` | read-only | a model **explores** your repo (read/list/grep) and challenges it; steerable focus |
| `/ollama:rescue` | **agentic** | delegate a coding task; the model edits files in an isolated worktree → you review the diff before it applies |

## Requirements

- **ollama** running locally (`ollama serve` or the Ollama app). Verify with `ollama --version`.
- **Python 3.x** on `PATH` (stdlib-only, no packages). On Windows, `py -3` is a fallback if `python` is missing.
- **git** on `PATH` (the agentic rescue uses `git worktree`).
- For **cloud models** (any model whose name contains `cloud`, e.g. `glm-5.2:cloud`): sign in once with `ollama signin`.

## Install

Local marketplaces take a **relative** path (`./…`), not an absolute one:

```bash
/plugin marketplace add ./ollama-cc-plugin
/plugin install ollama@ollama-cc
/reload-plugins
/ollama:setup
```

## Agentic rescue — safety ⚠️

`/ollama:rescue` lets an ollama model **read and write files**. Read this before using it:

- **Isolation.** The agent works in a throwaway `git worktree` off HEAD. Your real working tree is never touched by the agent; you get a diff and apply it yourself (via `git apply --3way`) only after reviewing it. Nothing is auto-applied.
- **Cloud egress.** With a cloud model, every file the agent reads is sent to `ollama.com`. `/ollama:rescue` discloses this and asks for consent before delegating.
- **`--allow-shell` is opt-in and is full RCE.** It gives the agent a `run_shell` tool. A shell is **not** contained by the worktree — it can read, modify, delete, or exfiltrate any file on the host (including secrets) and reach the network. With a cloud model that is a third party running code on your machine. The command shows a second, stronger disclosure and only enables it when you pass `--allow-shell` and confirm. Default is read/write only.
- **Keep it on-machine.** Set `OLLAMA_CC_MODEL` to a local model (see below) so nothing egresses.

```powershell
ollama pull <a-local-model>
$env:OLLAMA_CC_MODEL = "<a-local-model>"   # PowerShell
set OLLAMA_CC_MODEL=<a-local-model>        # cmd.exe
export OLLAMA_CC_MODEL=<a-local-model>     # bash
```

`--model <name>` overrides the default per command.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `OLLAMA_CC_MODEL` | `glm-5.2:cloud` | Default model for the commands |
| `OLLAMA_CC_HOST` | `http://127.0.0.1:11434` | ollama daemon base URL. Falls back to ollama's own `OLLAMA_HOST` (bare `host:port` accepted) when unset. |

## Relationship to `/octo:debate`

`/octo:debate` orchestrates **multiple** providers for consensus. This plugin is the opposite niche: one model, low ceremony — a fast single second opinion, or a single delegated agent.

## Runtime & tests

- `plugins/ollama/scripts/ollama_companion.py` — read-only tier (`setup`, `run`).
- `plugins/ollama/scripts/ollama_agent.py` — agentic tier (tool loop, path jail, worktree, gate token, shell).

Self-checks stub the HTTP/daemon layer (no real daemon needed):

```bash
cd plugins/ollama/scripts
python -m unittest test_ollama_companion test_ollama_agent
```
