#!/usr/bin/env python3
"""Optional Playwright-based session capture for /ollama:usage.

The DEFAULT acquisition needs no dependency -- see `ollama_usage.py set-cookie`.
This is the convenience path: it opens a real Chrome window, the human logs in ONCE
(this script never touches the password or a CAPTCHA), and it captures the durable
__Secure-session cookie automatically. `--headless` re-captures from the persisted
profile without a new login, to refresh an expired stored cookie while the browser
session is still alive.

Requires the optional `playwright` package + Google Chrome. If either is missing it
prints how to install it (or use the manual path) and exits 9.

The stealth flag --disable-blink-features=AutomationControlled hides
navigator.webdriver so Google/ollama do not block the human login (pattern proven in
tbd/ig_browser).
"""
import argparse
import glob
import os
import stat
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ollama_usage as ou

STEALTH = ["--no-first-run", "--no-default-browser-check",
           "--disable-blink-features=AutomationControlled"]


def _profile_dir():
    return os.path.join(ou._dir(), "profile")


def _is_ollama_domain(d):
    # Exact ollama.com or a real subdomain -- not a lookalike like evilollama.com.
    d = (d or "").lstrip(".")
    return d == "ollama.com" or d.endswith(".ollama.com")


def store_from_cookies(cookies):
    """Extract the durable __Secure-session cookie from a Playwright context.cookies()
    list and store it (0600). Returns the stored Cookie string, or None if absent."""
    val = next((c.get("value") for c in cookies
                if c.get("name") == "__Secure-session"
                and _is_ollama_domain(c.get("domain"))), None)
    if not val:
        return None
    cookie = "__Secure-session=" + val
    ou._atomic_write(ou._cookie_path(), cookie, mode=stat.S_IRUSR | stat.S_IWUSR)
    return cookie


def _clean_locks(profile):
    # A crashed/killed prior run leaves Chromium singleton locks that block relaunch.
    for pat in ("SingletonLock", "SingletonCookie", "SingletonSocket", "lockfile"):
        for p in glob.glob(os.path.join(profile, "**", pat), recursive=True):
            try:
                os.unlink(p)
            except OSError:
                pass


def _wait_logged_in(page, timeout_s):
    # Passive poll only -- reloading mid-login breaks OAuth redirects (measured).
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if "Session usage" in page.inner_text("body"):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def capture(headless, login_timeout):
    from playwright.sync_api import sync_playwright  # lazy: optional dependency
    profile = _profile_dir()
    os.makedirs(profile, exist_ok=True)
    _clean_locks(profile)
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=profile, channel="chrome", headless=headless, args=STEALTH)
        try:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(ou.SETTINGS_URL, wait_until="domcontentloaded")
            if headless:
                time.sleep(2)
                ok = "Session usage" in page.inner_text("body")
            else:
                print("A Chrome window opened -- log into ollama.com there "
                      "(waiting up to %ds)..." % login_timeout, flush=True)
                ok = _wait_logged_in(page, login_timeout)
            if not ok:
                return None, ("profile not logged in" if headless
                              else "login not completed in time")
            cookies = ctx.cookies()
        finally:
            ctx.close()
    cookie = store_from_cookies(cookies)
    return (cookie, None) if cookie else (None, "no __Secure-session cookie after login")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass
    ap = argparse.ArgumentParser(prog="ollama_usage_login")
    ap.add_argument("--headless", action="store_true",
                    help="re-capture from the saved profile without a new login")
    ap.add_argument("--timeout", type=int, default=240, help="headful login wait (seconds)")
    args = ap.parse_args(argv)

    try:
        import playwright  # noqa: F401
    except ImportError:
        print("optional dependency 'playwright' is not installed.\n"
              "  install:  pip install playwright   (reuses your installed Chrome)\n"
              "  or use the no-dependency manual path:  ollama_usage.py set-cookie",
              file=sys.stderr)
        return 9

    try:
        cookie, err = capture(args.headless, args.timeout)
    except Exception as e:
        low = str(e).lower()
        if any(k in low for k in ("channel", "chrome", "executable", "browsertype")):
            print("could not launch Chrome via Playwright (%s).\n"
                  "  ensure Google Chrome is installed, or use ollama_usage.py set-cookie."
                  % e, file=sys.stderr)
            return 9
        print("login capture failed: %s" % e, file=sys.stderr)
        return 1
    if not cookie:
        print("session capture failed: %s. Try again, or use ollama_usage.py set-cookie."
              % err, file=sys.stderr)
        return 1
    res = ou.read_usage(cookie)
    if res["status"] == "ok":
        print("session captured and verified: session %.1f%% / weekly %.1f%% used."
              % (res["session_used"], res["weekly_used"]))
        return 0
    print("session captured, but a verify read returned no usage (%s)."
          % res.get("reason", ""), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
