---
description: Check the ollama daemon, list installed models, and flag cloud sign-in
argument-hint: ''
allowed-tools: Bash(python:*), Bash(py:*), Bash(ollama:*), AskUserQuestion
---

Check whether ollama is ready for this plugin.

Run the setup check (if `python` is not found, retry the same command once with `py -3` instead of `python`; if neither exists, tell the user to install Python 3.x on PATH and stop):

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_companion.py" setup --json
```

Interpret the JSON:

- `daemon: false` → the daemon is down. Show the `message` field verbatim (it tells the user to run `ollama serve` or launch the Ollama app). Stop here.
- `models_error` is set → the daemon is up but `/api/tags` failed, so the model list is unknown. Report that and suggest retrying; do NOT offer a pull (an empty `models` here means "couldn't list", not "none installed").
- `daemon: true`, no `models_error`, and (`models` is empty **or** `default_model_installed: false`) → use `AskUserQuestion` exactly once to offer pulling the default model (`default_model` from the JSON). Put the pull option first, suffixed `(Recommended)`:
  - `Pull <default_model> (Recommended)`
  - `Skip for now`
  - If the user chooses pull, run `ollama pull <default_model>`, then re-run the setup check above.
- If `has_cloud_models: true`, keep the `note_cloud_auth` guidance in your summary (cloud calls route through ollama.com and need `ollama signin` if they return an auth error).

Present a short human-readable status (daemon version, installed models with a `[cloud]` marker, default model). Do not start any long-running work.
