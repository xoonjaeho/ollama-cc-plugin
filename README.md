# ollama plugin for Claude Code

Use local or cloud [ollama](https://ollama.com) models from inside Claude Code as a **second engine** — a quick second opinion, a code review from a different model, or a delegated agentic task, without leaving your workflow.

Two tiers:

- **Read-only consult/review** — a single stateless call to the local ollama daemon (`ollama_companion.py`).
- **Agentic rescue** — an ollama model runs a tool loop (read/write files, optionally shell) inside a throwaway **git worktree** and hands you a diff to review (`ollama_agent.py`). Modeled on the Codex plugin's rescue.

## What you get

| Command | Tier | Engine | Timeout | Cloud egress | What it does |
|---|---|---|---|---|---|
| `/ollama:setup` | — | companion | — | none | check the daemon, list models, flag cloud sign-in |
| `/ollama:ask` | read-only | companion | 5 min · idle-gap | question sent (disclosed) | ask a model a question, answer returned verbatim |
| `/ollama:review` | read-only | companion | 5 min · idle-gap | diff sent (gated) | review your git changes with a model (cloud egress gated) |
| `/ollama:adversarial-review` | read-only | agent | 5 min · idle-gap | reads sent (gated) | a model **explores** your repo (read/list/grep) and challenges it; steerable focus |
| `/ollama:rescue` | **agentic** | agent | 30 min · total | reads sent (gated) | delegate a coding task; the model edits files in an isolated worktree → you review the diff before it applies |
| `/ollama:as-claude` | **full session** | agent → claude | 30 min · total\* | reads + actions (gated) | run a task in a *real* Claude Code session powered by an ollama model (`ollama launch claude`) — full write+shell on your real tree, **no worktree, no diff gate**. More dangerous than rescue |
| `/ollama:list` | read-only | companion | — | none | list installed / available models |
| `/ollama:ps` | read-only | companion | — | none | list running (in-memory) models |
| `/ollama:show` | read-only | companion | — | none | show a model's details (family, parameters, quantization, size) |
| `/ollama:pull` | manage | companion | — | none (downloads) | download/install a model — **confirms before the download** |
| `/ollama:rm` | manage | companion | — | none | delete an installed model — **confirms before deleting** |
| `/ollama:usage` | read-only | usage | — | cookie → ollama.com | show your ollama.com **cloud usage** (session + weekly %) |
| `/ollama:usage-login` | setup | usage | — | login → ollama.com | capture your ollama.com session via a one-time browser login (optional; needs Playwright) |

**Engine** — `companion` = a single stateless daemon call (`ollama_companion.py`); `agent` = the tool loop in a jailed root / worktree (`ollama_agent.py`); `agent → claude` = the agent launches a real Claude Code session; `usage` = the cloud-usage reader.

**Timeout** — the five working commands accept `--timeout <sec>` to override the default. `idle-gap` = the longest quiet gap allowed (a steadily-progressing run keeps going; only a real stall is killed), so a long review completes as long as it makes progress. `total` = a hard wall-clock cap on the whole run (its process tree is killed past it), used where the command writes files. `*` as-claude's cap applies to the default worktree launch; a `--no-worktree` / `--resume` run is unbounded. The other commands are quick local daemon calls with no user-facing timeout.

**Cloud egress** applies **only when the model is a cloud model** (name contains `cloud`); with a local model nothing leaves the machine. `gated` = asks for consent before sending; `disclosed` = proceeds but tells you it went to the cloud. `usage`/`usage-login` talk to `ollama.com` regardless of model (that's where the numbers live).

## Requirements

- **ollama** running locally (`ollama serve` or the Ollama app). Verify with `ollama --version`.
- **Python 3.x** on `PATH` (stdlib-only, no packages). On Windows, `py -3` is a fallback if `python` is missing.
- **git** on `PATH` (the agentic rescue uses `git worktree`).
- For **cloud models** (any model whose name contains `cloud`, e.g. `glm-5.2:cloud`): sign in once with `ollama signin`. Cloud models can rarely return a truncated reply as a normal completion; if the answer looks cut off, re-run the command.
- **Optional**, only for `/ollama:usage-login` (browser-based session capture): the `playwright` package (`pip install playwright`) + Google Chrome. The manual `/ollama:usage` setup needs nothing extra.

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

## Model management

`/ollama:list`, `/ollama:ps`, and `/ollama:show` are read-only views of the local daemon. `/ollama:pull` and `/ollama:rm` change state, so they **confirm first**: the command shows what will be downloaded (or the model and disk it frees) and only proceeds after you say yes — the underlying script refuses to mutate without an explicit `--yes`, and `rm` only ever deletes the single model you named. All of these talk only to the local daemon; no extra dependencies.

## Cloud usage (`/ollama:usage`)

Shows how much of your ollama.com cloud allowance you've used (session + weekly %). There is **no usage API** — the numbers live only on the cookie-gated `ollama.com/settings` page — so this reads them by replaying your logged-in browser **session cookie**. The reader validates the response (a real percentage must parse; a login page returning HTTP 200 is never treated as success) and caches the last good result, so a transient network blip never erases your numbers.

### One-time setup

`/ollama:usage` will report "needs login" until you store a session once. Two ways:

- **Manual (no dependency).** `/ollama:usage` walks you through it: in your logged-in Chrome, open `ollama.com/settings`, DevTools → Network → reload → right-click the `settings` request → **Copy as cURL**, and paste it. Only the durable session cookie is extracted and stored.
- **Browser (optional, needs Playwright + Chrome).** `/ollama:usage-login` opens a real Chrome window; you log in once (solving any CAPTCHA yourself), and it captures the cookie automatically. `--headless` re-captures later without a new login.

### Security

- The stored value is a **live session cookie — treat it like a password.** It is written to `~/.ollama-usage/session` with owner-only permissions (`0600` on Unix; on Windows the file sits in your user profile, which is already user-private — `chmod` there is best-effort).
- It is **only ever sent to `ollama.com`**, and is never logged or printed back to you.
- Only the durable `__Secure-session` cookie is kept — not your whole cookie jar.

### When it expires

Sessions don't last forever. When yours ends (logout, or the cookie's own lifetime — on the order of weeks), the reader detects the rejected session and reports **"needs login"**; just re-run the setup above. Everything degrades gracefully — a network hiccup keeps showing your last-known numbers (marked stale) and does **not** nag you to re-login; only a genuinely rejected session does.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `OLLAMA_CC_MODEL` | `glm-5.2:cloud` | Default model for the commands |
| `OLLAMA_CC_HOST` | `http://127.0.0.1:11434` | ollama daemon base URL. Falls back to ollama's own `OLLAMA_HOST` (bare `host:port` accepted) when unset. |
| `OLLAMA_CC_NUM_CTX` | `32768` | Context window sent to the agentic rescue (`options.num_ctx`); the client-side char budget is derived from it. Lower it for a small-context local model so the agent's pinned system+task aren't front-truncated. |
| `OLLAMA_USAGE_DIR` | `~/.ollama-usage` | Where the cloud-usage session cookie and cache are stored. |

## Relationship to `/octo:debate`

`/octo:debate` orchestrates **multiple** providers for consensus. This plugin is the opposite niche: one model, low ceremony — a fast single second opinion, or a single delegated agent.

## Runtime & tests

- `plugins/ollama/scripts/ollama_companion.py` — read-only tier (`setup`, `run`) + model management (`list`, `ps`, `show`, `pull`, `rm`).
- `plugins/ollama/scripts/ollama_agent.py` — agentic tier (tool loop, path jail, worktree, gate token, shell).
- `plugins/ollama/scripts/ollama_usage.py` — cloud-usage reader (`set-cookie`, `read`); stdlib only.
- `plugins/ollama/scripts/ollama_usage_login.py` — optional Playwright session capture.

Self-checks stub the HTTP/daemon/network layer (no real daemon, no network needed):

```bash
cd plugins/ollama/scripts
python -m unittest discover -p "test_*.py"
```

## License

MIT — see [LICENSE](LICENSE).
