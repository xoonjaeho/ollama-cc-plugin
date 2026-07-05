#!/usr/bin/env python3
"""ollama.com cloud-usage reader for the ollama-cc plugin. stdlib only.

The usage numbers (session + weekly %) exist ONLY on the cookie-gated
https://ollama.com/settings HTML -- there is no usage API. This replays a stored
browser session cookie, validates the response *positively* (a parsed % -- never a
bare HTTP 200, which a login/Cloudflare page also returns), and caches the last
good result so a transient failure never erases it.

Subcommands:
  set-cookie   store a session cookie read from stdin (a cURL copy, a Cookie
               header, raw `k=v; k=v`, or a bare __Secure-session value)
  read         fetch usage, update the cache, print it (--json / --cache-only)

State lives under ~/.ollama-usage/ (override with OLLAMA_USAGE_DIR):
  session     the cookie -- a live credential, written 0600
  cache.json  last-good usage + freshness flags

Exit codes: 0 ok · 2 bad input · 5 session rejected (needs re-login) ·
            6 transient failure (kept last-good) · 8 no session configured
"""
import argparse
import json
import os
import re
import socket
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.request

SETTINGS_URL = "https://ollama.com/settings"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36")
SESSION_RE = re.compile(r"Session usage.*?(\d+(?:\.\d+)?)%\s*used", re.S)
WEEKLY_RE = re.compile(r"Weekly usage.*?(\d+(?:\.\d+)?)%\s*used", re.S)
RESET_RE = re.compile(r"Resets in ([^<.]+?)\s*[.<]")
DATA_TIME_RE = re.compile(r'data-time="([^"]+)"')
_DUR_RE = re.compile(r"(?:(\d+)|an?)\s*(second|minute|hour|day|week)s?", re.I)
_UNIT_SEC = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800}
_TRANSPORT = (urllib.error.URLError, socket.timeout, ConnectionError, OSError)


def _iso_to_epoch(text):
    """Parse an ISO-8601 UTC timestamp ('2026-07-04T23:00:00Z') to epoch seconds.
    The `.replace('Z', ...)` keeps this working on Python < 3.11, whose fromisoformat
    rejects a bare 'Z'."""
    if not text:
        return None
    try:
        from datetime import datetime
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
    except (ValueError, TypeError, OSError):
        return None


def _reset_to_epoch(text, now):
    """Fallback: convert ollama's relative reset text ("29 minutes", "an hour") to an
    absolute epoch. Coarser than the page's data-time, used only when it is absent."""
    if not text:
        return None
    m = _DUR_RE.search(text)
    if not m:
        return None
    n = int(m.group(1)) if m.group(1) else 1
    return int(now + n * _UNIT_SEC[m.group(2).lower()])


def _reset_at(body, marker, rel_text, now, end_marker=None):
    """Prefer the exact absolute reset time the page ships as `data-time` on the
    local-time div right after `marker` ("Session usage" / "Weekly usage"); fall back
    to parsing the coarse relative text only when the attribute is missing. The search
    is bounded to before `end_marker` so a section without data-time can't inherit the
    next section's timestamp."""
    i = body.find(marker)
    if i >= 0:
        end = body.find(end_marker, i) if end_marker else -1
        m = DATA_TIME_RE.search(body, i, end if end >= 0 else len(body))
        if m:
            epoch = _iso_to_epoch(m.group(1))
            if epoch is not None:
                return epoch
    return _reset_to_epoch(rel_text, now)


def _dir():
    return os.environ.get("OLLAMA_USAGE_DIR") or os.path.join(os.path.expanduser("~"), ".ollama-usage")


def _cookie_path():
    return os.path.join(_dir(), "session")


def _cache_path():
    return os.path.join(_dir(), "cache.json")


def _atomic_write(path, data, mode=None):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    try:
        os.chmod(d, 0o700)  # keep the state dir owner-only (best-effort on Windows)
    except OSError:
        pass
    # mkstemp makes a UNIQUE file at 0600: closes the world-readable window of a
    # write-then-chmod, and the shared-".tmp"-name collision when a detached refresh
    # writes concurrently with a foreground read.
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        if mode is not None:
            try:
                os.chmod(tmp, mode)
            except OSError:
                pass
        try:
            os.replace(tmp, path)
        except PermissionError:
            time.sleep(0.1)  # Windows: a concurrent reader may briefly hold the target
            os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def extract_cookie(text):
    """Reduce a paste to ONLY the durable __Secure-session cookie -- never the whole
    jar. A real logged-in /settings cURL / Cookie header always contains it, so this
    keeps just that one credential. A bare pasted token is taken to be its value."""
    text = text.strip()
    if not text:
        return None
    m = re.search(r"__Secure-session=([^;\s'\"]+)", text)
    if m:
        return "__Secure-session=" + m.group(1)
    if "=" not in text and not any(c.isspace() for c in text):  # a bare token value
        return "__Secure-session=" + text
    return None


def _fetch(cookie, timeout):
    req = urllib.request.Request(SETTINGS_URL, headers={
        "Cookie": cookie, "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.getcode(), r.geturl(), r.read().decode("utf-8", "replace"), None
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        return e.code, getattr(e, "url", SETTINGS_URL) or SETTINGS_URL, body, None
    except _TRANSPORT as e:
        return None, SETTINGS_URL, "", e


def read_usage(cookie, timeout=20):
    """Fetch and classify. 'ok' REQUIRES a parsed % -- HTTP 200 alone is never trusted."""
    code, final_url, body, transport_err = _fetch(cookie, timeout)
    if transport_err is not None:
        return {"status": "transient", "reason": "network error: %s"
                % (getattr(transport_err, "reason", transport_err))}
    s, w = SESSION_RE.search(body), WEEKLY_RE.search(body)
    if s and w:
        resets = RESET_RE.findall(body)
        s_reset = resets[0].strip() if len(resets) > 0 else None
        w_reset = resets[1].strip() if len(resets) > 1 else None
        now = time.time()
        return {"status": "ok",
                "session_used": float(s.group(1)), "weekly_used": float(w.group(1)),
                "session_reset": s_reset, "weekly_reset": w_reset,
                "session_reset_at": _reset_at(body, "Session usage", s_reset, now, "Weekly usage"),
                "weekly_reset_at": _reset_at(body, "Weekly usage", w_reset, now)}
    if code in (401, 403):
        return {"status": "auth", "reason": "session rejected (HTTP %s)" % code}
    if code and code >= 500:
        return {"status": "transient", "reason": "server error (HTTP %s)" % code}
    low = body.lower()
    # A real expired/invalid session redirects /settings to signin.ollama.com
    # (confirmed live); the body markers are a secondary signal.
    if ("signin" in final_url or "/login" in final_url
            or "sign in to" in low or "log in to" in low):
        return {"status": "auth", "reason": "not logged in (session expired?)"}
    return {"status": "unknown", "reason": "usage not found in response (HTTP %s)" % code}


def _load_cookie():
    try:
        with open(_cookie_path(), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _load_cache():
    try:
        with open(_cache_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _with_remaining(d):
    out = dict(d)
    for k in ("session", "weekly"):
        if "%s_used" % k in out:
            out["%s_remaining" % k] = round(100.0 - out["%s_used" % k], 1)
    return out


def _emit(d, as_json):
    if as_json:
        print(json.dumps(_with_remaining(d), ensure_ascii=False))
        return
    if "session_used" in d:
        tag = " (stale)" if d.get("stale") else ""
        print("ollama cloud usage%s: session %.1f%% used, weekly %.1f%% used"
              % (tag, d["session_used"], d["weekly_used"]))
        if d.get("session_reset"):
            print("  session resets in %s" % d["session_reset"])
    if d.get("need_login"):
        print("  session missing/expired -- %s" % d.get("reason", ""))
    elif "session_used" not in d:
        print("ollama cloud usage unavailable -- %s" % d.get("reason", "unknown"))


def cmd_set_cookie(args):
    cookie = extract_cookie(sys.stdin.read())
    if not cookie:
        print("error: no session cookie found in the input. Paste the /settings request as "
              "cURL, or the Cookie header, or the __Secure-session value.", file=sys.stderr)
        return 2
    _atomic_write(_cookie_path(), cookie, mode=stat.S_IRUSR | stat.S_IWUSR)
    res = read_usage(cookie, timeout=args.timeout)
    if res["status"] == "ok":
        print("session stored and verified: session %.1f%% / weekly %.1f%% used."
              % (res["session_used"], res["weekly_used"]))
        return 0
    print("session stored, but a test read returned no usage (%s: %s) -- the cookie may be "
          "wrong or expired." % (res["status"], res.get("reason", "")), file=sys.stderr)
    return 1


def cmd_read(args):
    cache = _load_cache()
    if args.cache_only:
        if not cache:
            _emit({"reason": "no cached usage yet"}, args.json)
            return 8
        _emit(cache, args.json)
        return 0
    cookie = _load_cookie()
    if not cookie:
        out = dict(cache)
        out.update({"ok": False, "need_login": True, "checked_ts": int(time.time()),
                    "reason": "no session configured -- run set-cookie"})
        _atomic_write(_cache_path(), json.dumps(out))  # stamp so the statusline refreshes at TTL, not every cycle
        _emit(out, args.json)
        return 8
    res = read_usage(cookie, timeout=args.timeout)
    now = int(time.time())
    if res["status"] == "ok":
        fresh = {"ok": True, "session_used": res["session_used"], "weekly_used": res["weekly_used"],
                 "session_reset": res.get("session_reset"), "weekly_reset": res.get("weekly_reset"),
                 "session_reset_at": res.get("session_reset_at"),
                 "weekly_reset_at": res.get("weekly_reset_at"),
                 "ts": now, "stale": False, "need_login": False}
        _atomic_write(_cache_path(), json.dumps(fresh))
        _emit(fresh, args.json)
        return 0
    # Failure: keep last-good values, annotate freshness. Only an auth failure sets
    # need_login; a transient/unknown failure must NOT (or a hiccup nags a re-login).
    out = dict(cache)
    out.update({"ok": False, "stale": True, "reason": res["reason"], "checked_ts": now})
    out["need_login"] = (res["status"] == "auth")
    _atomic_write(_cache_path(), json.dumps(out))
    _emit(out, args.json)
    return 5 if res["status"] == "auth" else 6


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass
    p = argparse.ArgumentParser(prog="ollama_usage")
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("set-cookie", help="store a session cookie read from stdin")
    sc.add_argument("--timeout", type=int, default=20)
    sc.set_defaults(func=cmd_set_cookie)

    rd = sub.add_parser("read", help="fetch usage, update cache, print")
    rd.add_argument("--json", action="store_true")
    rd.add_argument("--cache-only", action="store_true", help="print the cache without fetching")
    rd.add_argument("--timeout", type=int, default=20)
    rd.set_defaults(func=cmd_read)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
