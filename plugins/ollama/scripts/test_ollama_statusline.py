"""Self-check for the statusline: render formatting + the anti-spawn-storm throttle.
The throttle is the load-bearing bit -- without it the status line spawns a refresh on
every render (it runs on nearly every keystroke).
"""
import io
import json
import os
import shutil
import tempfile
import time
import unittest

import statusline as sl


class _RecordingStream:
    def __init__(self):
        self.encodings = []

    def reconfigure(self, encoding=None, **kw):
        self.encodings.append(encoding)

    def write(self, s):
        return len(s)

    def flush(self):
        pass


class TestRender(unittest.TestCase):
    def test_low_usage_is_green_with_both_percents(self):
        r = sl.render({"session_used": 0.1, "weekly_used": 14.6})
        self.assertIn("0%", r)
        self.assertIn("15%", r)
        self.assertIn(sl.GREEN, r)

    def test_high_usage_is_red(self):
        self.assertIn(sl.RED, sl.render({"session_used": 92.0, "weekly_used": 30.0}))

    def test_need_login(self):
        self.assertIn("login", sl.render({"need_login": True}))

    def test_empty_cache(self):
        self.assertIn("—", sl.render({}))  # em dash placeholder

    def test_stale_marker(self):
        self.assertIn("~", sl.render({"session_used": 1.0, "weekly_used": 1.0, "stale": True}))


class TestThrottle(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._prev = os.environ.get("OLLAMA_USAGE_DIR")
        os.environ["OLLAMA_USAGE_DIR"] = self._tmp

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("OLLAMA_USAGE_DIR", None)
        else:
            os.environ["OLLAMA_USAGE_DIR"] = self._prev
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_fresh_cache_does_not_spawn(self):
        spawned = []
        fired = sl.maybe_refresh(time.time(), spawn=lambda cmd: spawned.append(cmd))
        self.assertFalse(fired)
        self.assertEqual(spawned, [])

    def test_stale_spawns_once_then_throttled(self):
        spawned = []
        old = time.time() - sl.TTL - 10
        self.assertTrue(sl.maybe_refresh(old, spawn=lambda cmd: spawned.append(cmd)))
        self.assertEqual(len(spawned), 1)
        # a second render immediately after must be throttled by the fresh stamp
        self.assertFalse(sl.maybe_refresh(old, spawn=lambda cmd: spawned.append(cmd)))
        self.assertEqual(len(spawned), 1)


class TestMainReconfigure(unittest.TestCase):
    """The cp949 fix lives in main(): rendering the ⚠ glyph without the utf-8
    reconfigure crashes on a Windows console. A fresh checked_ts keeps it from
    spawning a real refresh during the test."""
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._prev = os.environ.get("OLLAMA_USAGE_DIR")
        os.environ["OLLAMA_USAGE_DIR"] = self._tmp
        with open(os.path.join(self._tmp, "cache.json"), "w") as f:
            f.write(json.dumps({"need_login": True, "reason": "x", "checked_ts": time.time()}))

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("OLLAMA_USAGE_DIR", None)
        else:
            os.environ["OLLAMA_USAGE_DIR"] = self._prev
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_main_forces_utf8_stdout(self):
        orig = (sl.sys.stdin, sl.sys.stdout, sl.sys.stderr)
        fake_out, fake_err = _RecordingStream(), _RecordingStream()
        sl.sys.stdin, sl.sys.stdout, sl.sys.stderr = io.StringIO(""), fake_out, fake_err
        try:
            rc = sl.main()
        finally:
            sl.sys.stdin, sl.sys.stdout, sl.sys.stderr = orig
        self.assertEqual(rc, 0)
        self.assertIn("utf-8", fake_out.encodings)


if __name__ == "__main__":
    unittest.main()
