"""Self-check for ollama_usage. Stubs the HTTP fetch and redirects state to a temp
dir -- never hits ollama.com, never touches the real ~/.ollama-usage.

Each test catches a concrete regression:
- trusting HTTP 200 (a login/Cloudflare page parsing as "ok") -> poisoned usage
- a transient failure erasing last-good or falsely flagging need_login -> re-login nag on a hiccup
- an auth failure NOT flagging re-login -> user never told the session died
- extract_cookie missing a real paste format -> set-cookie cannot store the session
"""
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout, redirect_stderr

import ollama_usage as ou

# Mirrors the real /settings raw HTML: the reset time is shipped as an exact
# data-time attribute on the local-time div (the "Resets in X" text is coarse).
SETTINGS_HTML = (
    '<html>...<span>Session usage</span><span>77.1% used</span>'
    '<div class="local-time" data-time="2026-07-04T23:00:00Z">Resets in 29 minutes.</div>'
    '<span>Weekly usage</span><span>14.6% used</span>'
    '<div class="local-time" data-time="2026-07-06T00:00:00Z">Resets in 1 day.</div>...</html>')
LOGIN_HTML = "<html><h1>Sign in to Ollama</h1><form>...</form></html>"


class _A:
    def __init__(self, **kw):
        self.json = kw.get("json", False)
        self.cache_only = kw.get("cache_only", False)
        self.timeout = kw.get("timeout", 20)


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._prev = os.environ.get("OLLAMA_USAGE_DIR")
        os.environ["OLLAMA_USAGE_DIR"] = self._tmp
        self._fetch = ou._fetch

    def tearDown(self):
        ou._fetch = self._fetch
        if self._prev is None:
            os.environ.pop("OLLAMA_USAGE_DIR", None)
        else:
            os.environ["OLLAMA_USAGE_DIR"] = self._prev
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _set_cookie(self, val="__Secure-session=abc"):
        ou._atomic_write(ou._cookie_path(), val, mode=0o600)

    def _prime_good(self):
        self._set_cookie()
        ou._fetch = lambda *a, **k: (200, ou.SETTINGS_URL, SETTINGS_HTML, None)
        with redirect_stdout(io.StringIO()):
            ou.cmd_read(_A())


class TestValidation(_Base):
    def test_parsed_usage_is_ok_and_cached(self):
        self._set_cookie()
        ou._fetch = lambda *a, **k: (200, ou.SETTINGS_URL, SETTINGS_HTML, None)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = ou.cmd_read(_A(json=True))
        self.assertEqual(rc, 0)
        d = json.loads(out.getvalue())
        self.assertTrue(d["ok"])
        self.assertEqual(d["session_used"], 77.1)
        self.assertEqual(d["weekly_used"], 14.6)
        self.assertEqual(d["session_remaining"], 22.9)
        self.assertEqual(d["session_reset"], "29 minutes")
        # reset_at comes from the exact data-time attribute, not the coarse relative text
        self.assertEqual(d["session_reset_at"], ou._iso_to_epoch("2026-07-04T23:00:00Z"))
        self.assertEqual(d["weekly_reset_at"], ou._iso_to_epoch("2026-07-06T00:00:00Z"))
        self.assertEqual(ou._load_cache()["session_used"], 77.1)  # persisted

    def test_reset_falls_back_to_relative_without_data_time(self):
        self._set_cookie()
        html = ("<span>Session usage</span><span>5% used</span><div>Resets in 10 minutes.</div>"
                "<span>Weekly usage</span><span>5% used</span><div>Resets in 2 days.</div>")
        ou._fetch = lambda *a, **k: (200, ou.SETTINGS_URL, html, None)
        out = io.StringIO()
        with redirect_stdout(out):
            ou.cmd_read(_A(json=True))
        d = json.loads(out.getvalue())
        self.assertAlmostEqual(d["session_reset_at"], time.time() + 10 * 60, delta=5)  # relative fallback

    def test_login_page_200_is_not_ok(self):
        self._set_cookie()
        ou._fetch = lambda *a, **k: (200, ou.SETTINGS_URL, LOGIN_HTML, None)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = ou.cmd_read(_A(json=True))
        self.assertEqual(rc, 5)  # HTTP 200 with a login page must NOT read as success
        d = json.loads(out.getvalue())
        self.assertFalse(d["ok"])
        self.assertTrue(d["need_login"])

    def test_real_redirect_to_signin_is_auth(self):
        # A real expired session: /settings 200-redirects to signin.ollama.com with a
        # body that lacks the "sign in to" phrase -- the URL host is the only signal.
        self._set_cookie()
        ou._fetch = lambda *a, **k: (
            200, "https://signin.ollama.com/?client_id=x",
            "<html><title>Sign in</title></html>", None)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = ou.cmd_read(_A(json=True))
        self.assertEqual(rc, 5)
        self.assertTrue(json.loads(out.getvalue())["need_login"])


class TestLastGood(_Base):
    def test_transient_keeps_last_good_and_does_not_nag_relogin(self):
        self._prime_good()
        ou._fetch = lambda *a, **k: (None, ou.SETTINGS_URL, "", ConnectionError("refused"))
        out = io.StringIO()
        with redirect_stdout(out):
            rc = ou.cmd_read(_A(json=True))
        self.assertEqual(rc, 6)
        d = json.loads(out.getvalue())
        self.assertEqual(d["session_used"], 77.1)  # last-good preserved
        self.assertTrue(d["stale"])
        self.assertFalse(d["need_login"])          # a hiccup must not demand re-login

    def test_auth_failure_flags_relogin_but_keeps_values(self):
        self._prime_good()
        ou._fetch = lambda *a, **k: (401, ou.SETTINGS_URL, "", None)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = ou.cmd_read(_A(json=True))
        self.assertEqual(rc, 5)
        d = json.loads(out.getvalue())
        self.assertTrue(d["need_login"])
        self.assertEqual(d["session_used"], 77.1)


class TestNotConfigured(_Base):
    def test_no_cookie_rc8(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = ou.cmd_read(_A(json=True))
        self.assertEqual(rc, 8)
        self.assertTrue(json.loads(out.getvalue())["need_login"])

    def test_cache_only_does_not_fetch(self):
        self._prime_good()
        called = []
        ou._fetch = lambda *a, **k: called.append(1) or (500, "", "", None)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = ou.cmd_read(_A(json=True, cache_only=True))
        self.assertEqual(rc, 0)
        self.assertEqual(called, [])                # cache-only never hits the network
        self.assertEqual(json.loads(out.getvalue())["session_used"], 77.1)


class TestIsoEpoch(unittest.TestCase):
    def test_utc_epoch_and_day_delta(self):
        a = ou._iso_to_epoch("2026-07-04T23:00:00Z")
        b = ou._iso_to_epoch("2026-07-05T23:00:00Z")
        self.assertEqual(b - a, 86400)   # exactly one day apart, timezone-independent
        self.assertGreater(a, 1_500_000_000)

    def test_junk_returns_none(self):
        self.assertIsNone(ou._iso_to_epoch("not-a-date"))
        self.assertIsNone(ou._iso_to_epoch(None))


class TestResetParsing(unittest.TestCase):
    def test_minutes(self):
        self.assertEqual(ou._reset_to_epoch("29 minutes", 1000), 1000 + 29 * 60)

    def test_one_day(self):
        self.assertEqual(ou._reset_to_epoch("1 day", 1000), 1000 + 86400)

    def test_an_hour_no_number(self):
        self.assertEqual(ou._reset_to_epoch("an hour", 1000), 1000 + 3600)

    def test_unparseable_returns_none(self):
        self.assertIsNone(ou._reset_to_epoch("soon", 1000))
        self.assertIsNone(ou._reset_to_epoch(None, 1000))


class TestExtractCookie(unittest.TestCase):
    def test_curl_prefers_secure_session(self):
        curl = "curl 'https://ollama.com/settings' -H 'cookie: a=1; __Secure-session=XYZ; b=2'"
        self.assertEqual(ou.extract_cookie(curl), "__Secure-session=XYZ")

    def test_cookie_header_with_secure_session(self):
        self.assertEqual(ou.extract_cookie("Cookie: a=1; __Secure-session=ZZZ; b=2"),
                         "__Secure-session=ZZZ")

    def test_bare_token_wrapped(self):
        self.assertEqual(ou.extract_cookie("abc123token"), "__Secure-session=abc123token")

    def test_cookie_header_without_secure_session_rejected(self):
        # never store the whole jar -- a paste lacking __Secure-session is rejected
        self.assertIsNone(ou.extract_cookie("Cookie: foo=bar; baz=qux"))

    def test_raw_pairs_without_secure_session_rejected(self):
        self.assertIsNone(ou.extract_cookie("a=1; b=2"))

    def test_empty_returns_none(self):
        self.assertIsNone(ou.extract_cookie("   "))


class TestSetCookie(_Base):
    def test_stores_and_verifies(self):
        ou._fetch = lambda *a, **k: (200, ou.SETTINGS_URL, SETTINGS_HTML, None)
        prev = sys.stdin
        sys.stdin = io.StringIO("__Secure-session=XYZ")
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                rc = ou.cmd_set_cookie(_A())
        finally:
            sys.stdin = prev
        self.assertEqual(rc, 0)
        self.assertIn("verified", out.getvalue())
        self.assertEqual(ou._load_cookie(), "__Secure-session=XYZ")


if __name__ == "__main__":
    unittest.main()
