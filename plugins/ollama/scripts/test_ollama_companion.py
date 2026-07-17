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
        self.stream = kw.get("stream", False)


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

    def test_stream_concatenates_content_chunks(self):
        def fake_stream(path, payload, timeout=None):
            assert payload.get("stream") is True   # streaming must request stream:true
            yield {"message": {"content": "Hel"}}
            yield {"message": {"content": "lo"}, "done": True}
        orig = oc._post_stream
        oc._post_stream = fake_stream
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                rc = oc.cmd_run(_Args(prompt="hi", stream=True))
        finally:
            oc._post_stream = orig
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue().strip(), "Hello")

    def test_stream_idle_timeout_maps_to_exit_6(self):
        def fake_stream(path, payload, timeout=None):
            raise socket.timeout("timed out")
            yield  # makes this a generator; the raise fires on first iteration
        orig = oc._post_stream
        oc._post_stream = fake_stream
        err = io.StringIO()
        try:
            with redirect_stderr(err):
                rc = oc.cmd_run(_Args(prompt="hi", stream=True))
        finally:
            oc._post_stream = orig
        self.assertEqual(rc, 6)
        self.assertIn("idle", err.getvalue())

    def test_stream_without_done_marker_flags_truncation(self):
        # stream ends with content but no terminal done:true -> flag as possibly truncated, not success
        def fake_stream(path, payload, timeout=None):
            yield {"message": {"content": "partial"}}
        orig = oc._post_stream
        oc._post_stream = fake_stream
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                rc = oc.cmd_run(_Args(prompt="hi", stream=True))
        finally:
            oc._post_stream = orig
        self.assertEqual(rc, 1)
        self.assertIn("partial", out.getvalue())            # partial content is still shown
        self.assertIn("truncat", err.getvalue().lower())    # and flagged

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

    def test_empty_response_retries_once_then_succeeds(self):
        calls = []
        def seq(*a, **k):
            calls.append(1)
            if len(calls) == 1:
                return {"message": {"content": ""}}
            return {"message": {"content": "retry-answer"}}
        oc._post = seq
        out = io.StringIO()
        with redirect_stdout(out):
            # Use a local model so _resolve_num_ctx does not make an extra /api/show call.
            rc = oc.cmd_run(_Args(model="llama3.2:latest"))
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(out.getvalue().strip(), "retry-answer")

    def test_empty_response_retry_once_then_fails(self):
        calls = []
        def empty(*a, **k):
            calls.append(1)
            return {"message": {"content": ""}}
        oc._post = empty
        err = io.StringIO()
        with redirect_stderr(err):
            rc = oc.cmd_run(_Args(model="llama3.2:latest"))
        self.assertEqual(rc, 1)
        self.assertEqual(len(calls), 2)
        self.assertIn("empty response", err.getvalue())

    def test_show_thinking_stream_does_not_dup_to_stdout(self):
        def fake_stream(path, payload, timeout=None):
            yield {"message": {"content": "ok", "thinking": "reasoning"}}
            yield {"message": {"content": "!"}, "done": True}
        orig = oc._post_stream
        oc._post_stream = fake_stream
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                rc = oc.cmd_run(_Args(stream=True, show_thinking=True))
        finally:
            oc._post_stream = orig
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue().strip(), "ok!")
        self.assertIn("reasoning", err.getvalue())

    def test_show_thinking_fallback_does_not_dup_to_stderr(self):
        oc._post = lambda *a, **k: {"message": {"content": "", "thinking": "only-reasoning"}}
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = oc.cmd_run(_Args(show_thinking=True))
        self.assertEqual(rc, 0)
        self.assertIn("only-reasoning", out.getvalue())
        self.assertNotIn("only-reasoning", err.getvalue())

    def test_show_thinking_stream_tees_reasoning_before_content(self):
        # Reasoning models emit thinking BEFORE content; --show-thinking must still show it
        # (buffered pre-content reasoning is flushed to stderr once the answer begins).
        def fake_stream(path, payload, timeout=None):
            yield {"message": {"thinking": "step-by-step"}}
            yield {"message": {"content": "answer"}, "done": True}
        orig = oc._post_stream
        oc._post_stream = fake_stream
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                rc = oc.cmd_run(_Args(stream=True, show_thinking=True, model="llama3.2:latest"))
        finally:
            oc._post_stream = orig
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue().strip(), "answer")    # answer only on stdout
        self.assertIn("step-by-step", err.getvalue())          # pre-content reasoning teed

    def test_empty_content_with_thinking_still_retries(self):
        # Reasoning present must NOT short-circuit the A2 retry: content is what counts;
        # thinking is only a fallback after the retry also yields no content.
        calls = []
        def seq(*a, **k):
            calls.append(1)
            if len(calls) == 1:
                return {"message": {"content": "", "thinking": "draft"}}
            return {"message": {"content": "real answer"}}
        oc._post = seq
        out = io.StringIO()
        with redirect_stdout(out):
            rc = oc.cmd_run(_Args(model="llama3.2:latest"))
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)                        # retried despite thinking present
        self.assertEqual(out.getvalue().strip(), "real answer")

    def test_stream_reasoning_only_without_done_flags_truncation(self):
        # A reasoning-only stream that never sees done:true was truncated: show the reasoning
        # but exit non-zero -- do not report a silent success.
        def fake_stream(path, payload, timeout=None):
            yield {"message": {"thinking": "partial reasoning"}}
        orig = oc._post_stream
        oc._post_stream = fake_stream
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                rc = oc.cmd_run(_Args(stream=True, model="llama3.2:latest"))
        finally:
            oc._post_stream = orig
        self.assertEqual(rc, 1)                                # truncation -> failure, not rc0
        self.assertIn("partial reasoning", out.getvalue())     # reasoning still shown
        self.assertIn("truncat", err.getvalue().lower())       # and flagged


class TestFinalText(unittest.TestCase):
    def test_thinking_fills_empty_content(self):
        self.assertEqual(oc._final_text({"content": "", "thinking": "x"}), "x")

    def test_whitespace_content_does_not_mask_thinking(self):
        self.assertEqual(oc._final_text({"content": "   ", "thinking": "y"}), "y")


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


class TestSetupRc(unittest.TestCase):
    """cmd_setup returns 1 when the daemon is up but the model list can't be fetched (distinct
    from rc 3 = daemon down), so a caller can detect the degraded result."""
    def setUp(self):
        self._orig = oc._get

    def tearDown(self):
        oc._get = self._orig

    def _run(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return oc.cmd_setup(type("A", (), {"json": False})())

    def test_ok_returns_0(self):
        oc._get = lambda path, timeout=10: {"version": "1"} if "version" in path else {"models": []}
        self.assertEqual(self._run(), 0)

    def test_tags_failure_returns_1(self):
        def g(path, timeout=10):
            if "version" in path:
                return {"version": "1"}
            raise urllib.error.URLError("tags unreachable")
        oc._get = g
        self.assertEqual(self._run(), 1)


if __name__ == "__main__":
    unittest.main()
