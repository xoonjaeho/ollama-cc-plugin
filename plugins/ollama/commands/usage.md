---
description: Show your ollama.com cloud usage (session + weekly %)
argument-hint: ''
allowed-tools: Bash(python:*), Bash(py:*), Bash(mktemp:*), Bash(rm:*), Write
---

Show ollama cloud usage (session + weekly %). The numbers live only on the cookie-gated ollama.com/settings page, read via a stored session cookie.

1. Read usage (use `py -3` if `python` is missing):

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_usage.py" read --json
```

2. Interpret the JSON:
   - `ok: true` → report `session_used` / `weekly_used` (and `session_remaining` / `weekly_remaining`, plus `session_reset` if present). If `stale: true`, add "(cached; refresh failed)".
   - `need_login: true` (exit 5 = expired, exit 8 = not set up) → the session is missing or expired. Help the user store one (step 3).
   - otherwise (exit 6 = transient) → show the last-good numbers and note that the refresh failed.

3. Set/refresh the session (only when needed). Two ways — offer the manual one by default; mention the browser one if the user prefers it:

   **Browser (optional, needs Playwright):** `/ollama:usage-login` opens Chrome, the user logs in once, and it captures the cookie automatically.

   **Manual (no dependency):** ask the user to, in their **logged-in Chrome**, open `https://ollama.com/settings`, press F12 → Network → reload → right-click the `settings` document request → Copy → **Copy as cURL**, and paste it. Write that paste to a temp file with the **Write tool** (`CF=$(mktemp)`), then feed it on stdin:

```bash
python "${CLAUDE_PLUGIN_ROOT}/scripts/ollama_usage.py" set-cookie < "$CF"
```

   Remove `$CF` afterward. The command keeps only the durable session cookie (stored `0600`) and verifies it. **Never print the cookie value back.** The stored session lasts until it expires (weeks) or the user logs out; re-run this step then.
