"""Self-check for ollama_companion. Stubs the HTTP layer -- never touches a real
daemon. Run: python -m unittest test_ollama_companion  (from this directory).

Each test catches a concrete regression:
- is_cloud misclassifying a model -> wrong egress gating in review.md
- run not extracting .message.content -> empty answers
- main() dropping the utf-8 reconfigure -> the cp949 crash this plugin fixed
- an {"error": ...} body / empty content passing as success -> silent blank answers
- HTTP 401 / timeout / daemon-down not mapped -> unhelpful errors on the plugin's
  reason-for-existing failure modes
"""
import email.message
import io
import socket
import unittest
import urllib.error
from contextlib import redirect_stdout, redirect_stderr

import ollama_companion as oc


class _Args:
    def __init__(self, **kw):
        self.prompt = kw.get("prompt", "hi")
        self.model = kw.get("model")
        self.think = kw.get("think", False)
        self.show_thinking = kw.get("show_thinking", False)
        self.timeout = kw.get("timeout", 120)


class _RecordingStream:
    """A stdout/stderr stand-in that records reconfigure() calls."""
    def __init__(self):
        self.encodings = []
        self.buf = io.StringIO()

    def reconfigure(self, encoding=None, **kw):
        self.encodings.append(encoding)

    def write(self, s):
        return self.buf.write(s)

    def flush(self):
        pass


def _http_error(status, body):
    return urllib.error.HTTPError(
        oc._url("/api/chat"), status, "err", email.message.Message(),
        io.BytesIO(body.encode("utf-8")))


class TestIsCloud(unittest.TestCase):
    def test_cloud_tag(self):
        self.assertTrue(oc.is_cloud("glm-5.2:cloud"))

    def test_cloud_in_tag_without_colon(self):
        self.assertTrue(oc.is_cloud("qwen3.5:397b-cloud"))

    def test_remote_host_fallback(self):
        self.assertTrue(oc.is_cloud("weird", {"remote_host": "https://ollama.com:443"}))

    def test_local_is_not_cloud(self):
        self.assertFalse(oc.is_cloud("llama3.2:latest", {"details": {}}))


class TestRun(unittest.TestCase):
    def setUp(self):
        self._orig = oc._post

    def tearDown(self):
        oc._post = self._orig

    def test_extracts_content(self):
        oc._post = lambda *a, **k: {"message": {"content": "pong"}}
        out = io.StringIO()
        with redirect_stdout(out):
            rc = oc.cmd_run(_Args(prompt="ping"))
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue().strip(), "pong")

    def test_oversized_prompt_warns_but_proceeds(self):
        oc._post = lambda *a, **k: {"message": {"content": "ok"}}
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = oc.cmd_run(_Args(prompt="x" * (oc.PROMPT_WARN_CHARS + 1)))
        self.assertEqual(rc, 0)                       # warning must not block the call
        self.assertIn("may exceed", err.getvalue())

    def test_error_in_200_body_is_not_success(self):
        oc._post = lambda *a, **k: {"error": "model failed to load"}
        err = io.StringIO()
        with redirect_stderr(err):
            rc = oc.cmd_run(_Args())
        self.assertEqual(rc, 1)
        self.assertIn("model failed to load", err.getvalue())

    def test_empty_content_is_not_success(self):
        oc._post = lambda *a, **k: {"message": {"content": "   "}}
        err = io.StringIO()
        with redirect_stderr(err):
            rc = oc.cmd_run(_Args())
        self.assertEqual(rc, 1)
        self.assertIn("empty response", err.getvalue())

    def test_model_not_found_maps_to_pull_hint(self):
        def boom(*a, **k):
            raise _http_error(404, '{"error":"model \'x\' not found"}')
        oc._post = boom
        err = io.StringIO()
        with redirect_stderr(err):
            rc = oc.cmd_run(_Args(model="x:cloud"))
        self.assertEqual(rc, 4)
        self.assertIn("ollama pull", err.getvalue())

    def test_auth_error_maps_to_signin(self):
        def boom(*a, **k):
            raise _http_error(401, '{"error":"unauthorized"}')
        oc._post = boom
        err = io.StringIO()
        with redirect_stderr(err):
            rc = oc.cmd_run(_Args(model="glm-5.2:cloud"))
        self.assertEqual(rc, 5)
        self.assertIn("ollama signin", err.getvalue())

    def test_timeout_maps_to_rc6(self):
        def boom(*a, **k):
            raise socket.timeout("timed out")
        oc._post = boom
        err = io.StringIO()
        with redirect_stderr(err):
            rc = oc.cmd_run(_Args())
        self.assertEqual(rc, 6)

    def test_daemon_down_maps_to_rc3(self):
        def boom(*a, **k):
            raise urllib.error.URLError(ConnectionRefusedError("refused"))
        oc._post = boom
        err = io.StringIO()
        with redirect_stderr(err):
            rc = oc.cmd_run(_Args())
        self.assertEqual(rc, 3)
        self.assertIn("not reachable", err.getvalue())


class TestMainReconfigure(unittest.TestCase):
    """The cp949 fix lives in main(): deleting the reconfigure loop must fail here."""
    def test_main_forces_utf8_stdout(self):
        orig_post, orig_out, orig_err = oc._post, oc.sys.stdout, oc.sys.stderr
        fake_out, fake_err = _RecordingStream(), _RecordingStream()
        oc._post = lambda *a, **k: {"message": {"content": "ok"}}
        oc.sys.stdout, oc.sys.stderr = fake_out, fake_err
        try:
            rc = oc.main(["run", "ping"])
        finally:
            oc._post, oc.sys.stdout, oc.sys.stderr = orig_post, orig_out, orig_err
        self.assertEqual(rc, 0)
        self.assertIn("utf-8", fake_out.encodings)
        self.assertIn("utf-8", fake_err.encodings)


if __name__ == "__main__":
    unittest.main()
