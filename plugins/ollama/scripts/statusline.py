#!/usr/bin/env python3
"""Claude Code statusLine fragment: a compact ollama cloud-usage indicator.

Reads the cache ONLY -- never the network, never a browser -- so it stays instant.
When the cache is stale it fires a *detached, throttled* background refresh, which in
the manual acquisition mode is a plain HTTP fetch (no browser is ever spawned here).

Wire it up in settings.json, e.g.:
  "statusLine": {"type": "command",
                 "command": "python /path/to/scripts/statusline.py"}
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ollama_usage as ou

TTL = 180        # cache older than this -> trigger a background refresh
THROTTLE = 120   # minimum seconds between spawned refreshes (anti spawn-storm)
GREEN, AMBER, RED, DIM, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"


def _stamp_path():
    return os.path.join(ou._dir(), ".refresh-stamp")


def _spawn_detached(cmd):
    kw = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
              stderr=subprocess.DEVNULL, close_fds=True)
    if os.name == "nt":
        kw["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    try:
        subprocess.Popen(cmd, **kw)
    except Exception:
        pass  # a failed refresh must never break the status line


def maybe_refresh(last_check, now=None, spawn=_spawn_detached):
    """Fire a detached refresh only if the cache is stale AND we haven't spawned one
    within THROTTLE seconds. Returns True if a refresh was spawned."""
    now = time.time() if now is None else now
    if now - last_check <= TTL:
        return False
    stamp = _stamp_path()
    try:
        if os.path.exists(stamp) and now - os.path.getmtime(stamp) < THROTTLE:
            return False
    except OSError:
        pass
    try:
        os.makedirs(ou._dir(), exist_ok=True)
        with open(stamp, "w"):
            pass
    except OSError:
        pass
    spawn([sys.executable, os.path.join(os.path.dirname(__file__), "ollama_usage.py"), "read"])
    return True


def render(cache):
    if not cache or "session_used" not in cache and not cache.get("need_login"):
        return DIM + "ollama —" + RESET
    if cache.get("need_login"):
        return AMBER + "ollama ⚠ login" + RESET
    s, w = cache["session_used"], cache["weekly_used"]
    worst = max(s, w)
    color = GREEN if worst < 50 else AMBER if worst < 85 else RED
    stale = "~" if cache.get("stale") else ""
    return "%sollama %s%.0f%%·%.0f%%%s" % (color, stale, s, w, RESET)


def main():
    # Windows consoles default to cp949; force utf-8 so the ⚠/·/— glyphs below
    # don't crash the status line (the cp949 failure this plugin already fixes).
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass
    try:
        sys.stdin.read()  # Claude Code pipes session JSON on stdin; we don't need it
    except Exception:
        pass
    cache = ou._load_cache()
    last_check = max(cache.get("ts", 0), cache.get("checked_ts", 0)) if cache else 0
    maybe_refresh(last_check)
    sys.stdout.write(render(cache))
    return 0


if __name__ == "__main__":
    sys.exit(main())
