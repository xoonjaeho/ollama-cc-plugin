---
description: Capture your ollama.com session for /ollama:usage via a one-time browser login (optional; needs Playwright)
argument-hint: '[--headless]'
allowed-tools: Bash(python:*), Bash(py:*)
---

Optional convenience setup for `/ollama:usage`: opens a real Chrome window so the user logs in ONCE, then captures the session cookie automatically. Requires the optional `playwright` package + Google Chrome. (The no-dependency alternative is the manual cookie paste that `/ollama:usage` guides.)

Run (use `py -3` if `python` is missing):

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_usage_login.py"
```

- A Chrome window opens on ollama.com. Tell the user to log in there. They solve any CAPTCHA themselves — you do not. The script waits, captures the durable session cookie, stores it `0600`, and verifies it.
- Exit code 9 = Playwright or Chrome unavailable → relay the printed install hint, or point the user to the manual path (`/ollama:usage`).
- On success it prints the verified usage. **Never print any cookie value.**
- `--headless` re-captures from the saved profile without a new login — use it to refresh an expired stored cookie while the browser session is still alive.
