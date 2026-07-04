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
| `/ollama:as-claude` | **full session** | run a task in a *real* Claude Code session powered by an ollama model (`ollama launch claude`) — full write+shell on your real tree, **no worktree, no diff gate**. More dangerous than rescue |

## Requirements

- **ollama** running locally (`ollama serve` or the Ollama app). Verify with `ollama --version`.
- **Python 3.x** on `PATH` (stdlib-only, no packages). On Windows, `py -3` is a fallback if `python` is missing.
- **git** on `PATH` (the agentic rescue uses `git worktree`).
- For **cloud models** (any model whose name contains `cloud`, e.g. `glm-5.2:cloud`): sign in once with `ollama signin`.

## Install

```bash
/plugin marketplace add xoonjaeho/ollama-cc-plugin
/plugin install ollama@ollama-cc
/reload-plugins
/ollama:setup
```

`/ollama:setup` checks the daemon, lists your models, and flags cloud sign-in.

<details>
<summary>Install from a local clone (development)</summary>

Local marketplaces take a **relative** path (`./…`), not an absolute one:

```bash
/plugin marketplace add ./ollama-cc-plugin
/plugin install ollama@ollama-cc
/reload-plugins
```
</details>

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

## as-claude — safety ⚠️⚠️

`/ollama:as-claude` runs your task inside a **full, real Claude Code session** (`ollama launch claude`) whose brain is an ollama model. It is stronger, and **more dangerous**, than `/ollama:rescue`:

- **No isolation, no diff gate.** The session runs with `--dangerously-skip-permissions` on your **real working tree and host** — it can read, write, delete, and run shell/network commands with no per-action approval and no diff to review. Every run is arbitrary code execution you authorize up front. For edits you want to review before they land, use `/ollama:rescue` instead (worktree-isolated).
- **Cloud egress + remote control.** With a cloud model, everything the session reads *and does* is exposed to `ollama.com` — a third party driving a full agent on your machine. The command discloses this and asks for consent once before launching.
- **Continue a session** with `--resume <session-id>` (printed after each run); override the model with `--model <name>`.
- The `total_cost_usd` the session reports is Claude Code's own placeholder estimate, **not** ollama's billing.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `OLLAMA_CC_MODEL` | `glm-5.2:cloud` | Default model for the commands |
| `OLLAMA_CC_HOST` | `http://127.0.0.1:11434` | ollama daemon base URL. Falls back to ollama's own `OLLAMA_HOST` (bare `host:port` accepted) when unset. |
| `OLLAMA_CC_NUM_CTX` | `32768` | Context window sent to the agentic rescue (`options.num_ctx`); the client-side char budget is derived from it. Lower it for a small-context local model so the agent's pinned system+task aren't front-truncated. |

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
