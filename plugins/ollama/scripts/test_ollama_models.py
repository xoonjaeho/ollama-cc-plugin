"""Self-check for the model-management subcommands (ps/show/pull/rm). Stubs the
HTTP layer -- never touches a real daemon, never downloads or deletes.

Each test catches a concrete regression:
- pull/rm mutating WITHOUT --yes -> the confirmation gate is the whole safety point
  (dropping the `if not args.yes` guard makes the mutating stub run -> these fail)
- ps/show not parsing the daemon's response shape -> empty or wrong listings
- rm on a not-installed model still calling /api/delete -> masks the real error
- daemon-down not mapped to rc 3 -> raw traceback instead of a hint
"""
import io
import unittest
from contextlib import redirect_stdout, redirect_stderr

import ollama_companion as oc


class _A:
    def __init__(self, **kw):
        self.model = kw.get("model")
        self.yes = kw.get("yes", False)
        self.json = kw.get("json", False)
        self.timeout = kw.get("timeout", 120)


class _Boundary(unittest.TestCase):
    def setUp(self):
        self._orig = (oc._get, oc._post, oc._post_stream, oc._delete)

    def tearDown(self):
        oc._get, oc._post, oc._post_stream, oc._delete = self._orig


class TestPs(_Boundary):
    def test_lists_running_model_name_and_size(self):
        oc._get = lambda *a, **k: {"models": [
            {"name": "llama3.2:latest", "size": 4 * 1024**3, "size_vram": 4 * 1024**3,
             "expires_at": "2026-07-04T12:00:00Z"}]}
        out = io.StringIO()
        with redirect_stdout(out):
            rc = oc.cmd_ps(_A())
        self.assertEqual(rc, 0)
        self.assertIn("llama3.2:latest", out.getvalue())
        self.assertIn("4.0 GB", out.getvalue())

    def test_none_running(self):
        oc._get = lambda *a, **k: {"models": []}
        out = io.StringIO()
        with redirect_stdout(out):
            rc = oc.cmd_ps(_A())
        self.assertEqual(rc, 0)
        self.assertIn("no models", out.getvalue().lower())

    def test_daemon_down_rc3(self):
        def boom(*a, **k):
            raise ConnectionError("refused")
        oc._get = boom
        with redirect_stderr(io.StringIO()):
            rc = oc.cmd_ps(_A())
        self.assertEqual(rc, 3)


class TestShow(_Boundary):
    def test_shows_family_params_quant(self):
        oc._post = lambda *a, **k: {"details": {
            "family": "llama", "parameter_size": "3.2B", "quantization_level": "Q4_K_M"}}
        oc._get = lambda *a, **k: {"models": []}  # size unknown -> line skipped
        out = io.StringIO()
        with redirect_stdout(out):
            rc = oc.cmd_show(_A(model="llama3.2"))
        self.assertEqual(rc, 0)
        v = out.getvalue()
        self.assertIn("llama", v)
        self.assertIn("3.2B", v)
        self.assertIn("Q4_K_M", v)


class TestPullGate(_Boundary):
    def test_pull_without_yes_does_not_download(self):
        called = []

        def stream(*a, **k):
            called.append(a)
            yield {}
        oc._get = lambda *a, **k: {"models": []}     # not installed
        oc._post_stream = stream
        out = io.StringIO()
        with redirect_stdout(out):
            rc = oc.cmd_pull(_A(model="llama3.2"))
        self.assertEqual(rc, oc.EXIT_CONFIRM)
        self.assertEqual(called, [])                  # the safety gate: no download
        self.assertIn("--yes", out.getvalue())

    def test_pull_with_yes_streams_and_succeeds(self):
        seen = {}

        def stream(path, payload, timeout=120):
            seen["path"] = path
            seen["payload"] = payload
            yield {"status": "downloading", "total": 100, "completed": 50}
            yield {"status": "success"}
        oc._post_stream = stream
        with redirect_stdout(io.StringIO()):
            rc = oc.cmd_pull(_A(model="llama3.2", yes=True))
        self.assertEqual(rc, 0)
        self.assertEqual(seen["path"], "/api/pull")
        self.assertEqual(seen["payload"]["model"], "llama3.2")
        self.assertTrue(seen["payload"]["stream"])

    def test_pull_truncated_stream_is_not_success(self):
        def stream(path, payload, timeout=120):
            yield {"status": "downloading", "total": 100, "completed": 40}
            # stream drops here -- no terminal {"status": "success"}
        oc._post_stream = stream
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = oc.cmd_pull(_A(model="llama3.2", yes=True))
        self.assertEqual(rc, 1)                        # must NOT report success
        self.assertNotIn("pulled", out.getvalue())
        self.assertIn("truncated", err.getvalue())


class TestRmGate(_Boundary):
    def test_rm_without_yes_does_not_delete(self):
        deleted = []
        oc._get = lambda *a, **k: {"models": [{"name": "llama3.2", "size": 2 * 1024**3}]}
        oc._delete = lambda *a, **k: deleted.append(a)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = oc.cmd_rm(_A(model="llama3.2"))
        self.assertEqual(rc, oc.EXIT_CONFIRM)
        self.assertEqual(deleted, [])                 # the safety gate: no delete
        self.assertIn("2.0 GB", out.getvalue())

    def test_rm_with_yes_deletes_named_model_only(self):
        seen = {}

        def dele(path, payload, timeout=120):
            seen["path"] = path
            seen["payload"] = payload
        oc._get = lambda *a, **k: {"models": [{"name": "llama3.2", "size": 2 * 1024**3}]}
        oc._delete = dele
        with redirect_stdout(io.StringIO()):
            rc = oc.cmd_rm(_A(model="llama3.2", yes=True))
        self.assertEqual(rc, 0)
        self.assertEqual(seen["path"], "/api/delete")
        self.assertEqual(seen["payload"], {"model": "llama3.2"})

    def test_rm_short_name_matches_latest_tag(self):
        seen = {}

        def dele(path, payload, timeout=120):
            seen["payload"] = payload
        oc._get = lambda *a, **k: {"models": [{"name": "all-minilm:latest", "size": 45 * 1024**2}]}
        oc._delete = dele
        with redirect_stdout(io.StringIO()):
            rc = oc.cmd_rm(_A(model="all-minilm", yes=True))   # bare name, installed as :latest
        self.assertEqual(rc, 0)
        self.assertEqual(seen["payload"], {"model": "all-minilm:latest"})  # delete by full tag

    def test_rm_missing_model_rc4_no_delete(self):
        deleted = []
        oc._get = lambda *a, **k: {"models": [{"name": "other:latest", "size": 1}]}
        oc._delete = lambda *a, **k: deleted.append(a)
        err = io.StringIO()
        with redirect_stderr(err):
            rc = oc.cmd_rm(_A(model="llama3.2", yes=True))
        self.assertEqual(rc, 4)
        self.assertEqual(deleted, [])
        self.assertIn("not installed", err.getvalue())


if __name__ == "__main__":
    unittest.main()
