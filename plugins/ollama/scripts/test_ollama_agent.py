"""Self-check for ollama_agent (P1 tool-loop). Stubs the /api/chat POST -- never
touches a real daemon. Run: python -m unittest test_ollama_agent (from here).

Each test catches a concrete regression:
- jail lets a path escape / reads .git / follows a symlink -> containment breach
- loop doesn't execute a tool_call or feed the result back -> dead agent
- a failed tool_call isn't turned into an error result -> orphaned turn / desync
- a bound (max_iters / loop-detect / malformed) doesn't trip -> runaway
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import ollama_agent as oa
import ollama_companion as oc


def _asst(content="", tool_calls=None):
    m = {"role": "assistant", "content": content}
    if tool_calls is not None:
        m["tool_calls"] = tool_calls
    return {"message": m}


def _call(name, args, cid="c1"):
    return {"id": cid, "function": {"name": name, "arguments": args}}


def _no_orphan(messages):
    """True iff no assistant turn is followed by fewer role:tool results than it has
    tool_calls (before the next assistant / the synth instruction). Catches the
    loop_detected mid-turn orphan the fallback must strip."""
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            need = len(m["tool_calls"])
            have = 0
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                have += 1
                j += 1
            if have < need:
                return False
            i = j
        else:
            i += 1
    return True


class _SeqPost:
    """Returns queued responses in order; records the payloads it was sent."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def __call__(self, path, payload, timeout=None):
        self.payloads.append(payload)
        return self.responses.pop(0) if self.responses else _asst(content="(end)")


class _AlwaysTool:
    """Always asks for a fresh (non-repeating) read -> exercises max_iters."""
    def __init__(self):
        self.n = 0

    def __call__(self, path, payload, timeout=None):
        self.n += 1
        return _asst(tool_calls=[_call("read_file", {"path": "f%d.txt" % self.n})])


class JailTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        with open(os.path.join(self.d, "a.txt"), "w") as f:
            f.write("hi")
        os.makedirs(os.path.join(self.d, ".git"), exist_ok=True)

    def test_accepts_in_root(self):
        self.assertTrue(oa.resolve_in_jail(self.d, "a.txt").endswith("a.txt"))

    def test_rejects_dotdot(self):
        with self.assertRaises(oa.JailError):
            oa.resolve_in_jail(self.d, os.path.join("..", "secret"))

    def test_rejects_absolute_escape(self):
        with self.assertRaises(oa.JailError):
            oa.resolve_in_jail(self.d, os.path.abspath(os.sep + "Windows"))

    def test_rejects_git(self):
        with self.assertRaises(oa.JailError):
            oa.resolve_in_jail(self.d, os.path.join(".git", "config"))

    def test_junction_escape_rejected(self):
        # Proves realpath actually resolves a Windows junction so the jail can't
        # be escaped through one. Skips gracefully if junctions need elevation.
        if sys.platform != "win32":
            self.skipTest("windows junction test")
        outside = tempfile.mkdtemp()
        link = os.path.join(self.d, "j")
        r = subprocess.run(["cmd", "/c", "mklink", "/J", link, outside],
                           capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("cannot create junction: %s" % (r.stderr or r.stdout).strip())
        with self.assertRaises(oa.JailError):
            oa.resolve_in_jail(self.d, os.path.join("j", "secret.txt"))

    def test_rejects_colon_ads(self):
        with self.assertRaises(oa.JailError):
            oa.resolve_in_jail(self.d, "a.txt:stream")

    def test_rejects_git_trailing_space(self):
        # NTFS strips trailing space -> ".git " would become a real .git dir
        with self.assertRaises(oa.JailError):
            oa.resolve_in_jail(self.d, os.path.join(".git ", "config"))


class LoopTest(unittest.TestCase):
    def setUp(self):
        self._orig = oa._post
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        oa._post = self._orig

    def test_read_then_done_feeds_result_back(self):
        with open(os.path.join(self.d, "a.txt"), "w") as f:
            f.write("hello")
        seq = _SeqPost([_asst(tool_calls=[_call("read_file", {"path": "a.txt"})]),
                        _asst(content="summary: hello")])
        oa._post = seq
        r = oa.run_agent("summarize a.txt", self.d)
        self.assertEqual(r["stop_reason"], "done")
        self.assertIn("summary", r["final"])
        self.assertTrue(r["actions"][0]["ok"])
        # the model's 2nd turn must have received a role:tool result carrying the id
        second = seq.payloads[1]["messages"]
        tool_msgs = [m for m in second if m.get("role") == "tool"]
        self.assertEqual(tool_msgs[0]["tool_call_id"], "c1")
        self.assertIn("hello", tool_msgs[0]["content"])

    def test_write_file_lands_in_root(self):
        seq = _SeqPost([_asst(tool_calls=[_call("write_file", {"path": "out.txt", "content": "X"})]),
                        _asst(content="done")])
        oa._post = seq
        r = oa.run_agent("write out.txt", self.d)
        self.assertEqual(r["stop_reason"], "done")
        with open(os.path.join(self.d, "out.txt")) as f:
            self.assertEqual(f.read(), "X")

    def test_write_file_rejects_oversize(self):
        big = "A" * (oa.WRITE_CAP + 1)
        with self.assertRaises(oa.JailError):
            oa.tool_write_file(self.d, {"path": "big.txt", "content": big})
        self.assertFalse(os.path.exists(os.path.join(self.d, "big.txt")))  # nothing written

    def test_jail_failure_becomes_error_result_not_crash(self):
        seq = _SeqPost([_asst(tool_calls=[_call("read_file", {"path": "../secret"})]),
                        _asst(content="ok")])
        oa._post = seq
        r = oa.run_agent("try to escape", self.d)
        self.assertEqual(r["stop_reason"], "done")
        self.assertFalse(r["actions"][0]["ok"])
        tool_msgs = [m for m in seq.payloads[1]["messages"] if m.get("role") == "tool"]
        self.assertTrue(tool_msgs[0]["content"].startswith("error:"))

    def test_max_iters_trips(self):
        oa._post = _AlwaysTool()
        r = oa.run_agent("loop forever", self.d, max_iters=3)
        self.assertEqual(r["stop_reason"], "max_iters")

    def test_loop_detected_on_repeated_identical_call(self):
        oa._post = _SeqPost([_asst(tool_calls=[_call("read_file", {"path": "same.txt"})])] * 4)
        r = oa.run_agent("repeat", self.d, max_iters=10)
        self.assertEqual(r["stop_reason"], "loop_detected")

    def test_unknown_tool_hits_malformed_cap(self):
        oa._post = _SeqPost([_asst(tool_calls=[_call("run_shell", {"cmd": "x"})])] * 5)
        r = oa.run_agent("bad tool", self.d, max_iters=10)
        self.assertEqual(r["stop_reason"], "malformed")

    def test_task_file_delivered_literally_not_via_shell(self):
        # the injection fix: task text comes from --task-file, so shell metachars
        # reach the model as literal task text and never touch a shell.
        tf = os.path.join(self.d, "task")
        with open(tf, "w") as f:
            f.write('summarize "$(whoami)"; echo pwn')
        seen = {}

        def stub(path, payload, timeout=None):
            seen["task"] = payload["messages"][1]["content"]
            return {"message": {"content": "done"}}
        oa._post = stub
        rc = oa.main(["--root", self.d, "--task-file", tf])
        self.assertEqual(rc, 0)
        self.assertIn("whoami", seen["task"])

    def test_error_body_is_not_done(self):
        oa._post = _SeqPost([{"message": {}, "error": "model failed to load"}])
        r = oa.run_agent("x", self.d)
        self.assertTrue(r["stop_reason"].startswith("api_error"))

    def test_non_dict_response_does_not_crash(self):
        oa._post = _SeqPost([None])  # bare JSON scalar from a broken server
        r = oa.run_agent("x", self.d)
        self.assertTrue(r["stop_reason"].startswith("api_error"))

    def test_multiple_tool_calls_one_turn_each_get_a_result(self):
        for n in ("a.txt", "b.txt"):
            with open(os.path.join(self.d, n), "w") as f:
                f.write(n)
        seq = _SeqPost([_asst(tool_calls=[_call("read_file", {"path": "a.txt"}, "c1"),
                                          _call("read_file", {"path": "b.txt"}, "c2")]),
                        _asst(content="read both")])
        oa._post = seq
        r = oa.run_agent("read both", self.d)
        self.assertEqual(r["stop_reason"], "done")
        self.assertEqual(len(r["actions"]), 2)
        tool_msgs = [m for m in seq.payloads[1]["messages"] if m.get("role") == "tool"]
        self.assertEqual({m["tool_call_id"] for m in tool_msgs}, {"c1", "c2"})

    def test_empty_turn_retry_yielding_tool_calls_executes_it(self):
        # An empty assistant turn triggers the A2 same-iteration retry; if the retry returns
        # tool_calls they must be executed, not dropped with an empty final (regression).
        with open(os.path.join(self.d, "a.txt"), "w") as f:
            f.write("hello")
        seq = _SeqPost([
            _asst(content=""),                                          # empty turn -> retry
            _asst(tool_calls=[_call("read_file", {"path": "a.txt"})]),  # retry yields a tool call
            _asst(content="summary: hello"),                            # final answer
        ])
        oa._post = seq
        r = oa.run_agent("summarize a.txt", self.d)
        self.assertEqual(r["stop_reason"], "done")
        self.assertEqual(r["final"], "summary: hello")   # not "" -- the retry tool call ran
        self.assertEqual(len(r["actions"]), 1)
        self.assertTrue(r["actions"][0]["ok"])

    def test_reread_after_write_is_not_auto_advanced(self):
        # read -> write (grows the file) -> read again: the second read must honor offset 0
        # and see the freshly written start, not auto-advance past it on stale read_progress.
        p = os.path.join(self.d, "f.txt")
        with open(p, "w") as fh:
            fh.write("OLD")                 # 3 bytes -> first read sets read_progress to 3
        new = "NEW-" + "Z" * 40             # 44 bytes, larger than the old size
        seq = _SeqPost([
            _asst(tool_calls=[_call("read_file", {"path": "f.txt"}, "r1")]),
            _asst(tool_calls=[_call("write_file", {"path": "f.txt", "content": new}, "w1")]),
            _asst(tool_calls=[_call("read_file", {"path": "f.txt"}, "r2")]),
            _asst(content="done"),
        ])
        oa._post = seq
        r = oa.run_agent("edit then verify", self.d)
        self.assertEqual(r["stop_reason"], "done")
        msgs = seq.payloads[-1]["messages"]
        r2 = [m for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") == "r2"]
        self.assertTrue(r2, "second read produced no tool result")
        self.assertTrue(r2[-1]["content"].startswith("NEW-"),
                        "re-read after write was auto-advanced past the new content: %r"
                        % r2[-1]["content"][:20])

    def test_read_file_is_bounded_and_offset_paginates(self):
        with open(os.path.join(self.d, "big.txt"), "w") as f:
            f.write("A" * (oa.READ_CAP + 100))
        first = oa.tool_read_file(self.d, {"path": "big.txt"})
        self.assertIn("[truncated", first)
        self.assertLessEqual(len(first.encode("utf-8")), oa.READ_CAP + 200)
        rest = oa.tool_read_file(self.d, {"path": "big.txt", "offset": oa.READ_CAP})
        self.assertNotIn("[truncated", rest)
        self.assertTrue(rest.startswith("A"))

    def test_next_read_offset_advances_a_reread(self):
        # re-requesting an already-served range with more file left -> next unread chunk
        self.assertEqual(oa._next_read_offset(0, 65536, 200000), 65536)

    def test_next_read_offset_honors_forward_request(self):
        # an explicit offset past what's served is a real request, not a re-read
        self.assertEqual(oa._next_read_offset(131072, 65536, 200000), 131072)

    def test_next_read_offset_no_advance_when_fully_read(self):
        # whole file already served -> do NOT advance, so a true spin still trips the loop guard
        self.assertEqual(oa._next_read_offset(0, 200000, 200000), 0)

    def test_derive_read_cap_clamps_floor(self):
        self.assertEqual(oa._derive_read_cap(1000), 16 * 1024)        # tiny budget -> floor

    def test_derive_read_cap_clamps_ceiling(self):
        self.assertEqual(oa._derive_read_cap(10_000_000), 96 * 1024)  # huge budget -> cap

    def test_derive_read_cap_scales_between(self):
        self.assertEqual(oa._derive_read_cap(150000), 50000)         # 150000 // 3, within range

    def test_system_prompt_readonly_omits_write_mandate(self):
        # a --readonly run has no write tool; it must not be told it FAILED for only reading
        ro = oa._system_prompt("/root", allow_write=False)
        rw = oa._system_prompt("/root", allow_write=True)
        self.assertNotIn("FAILED", ro)
        self.assertNotIn("write_file", ro)
        self.assertIn("write_file", rw)                   # write runs still get the write nudge
        self.assertIn("do NOT read the same file", ro)    # anti-re-read nudge applies in both modes

    def test_detect_context_length_non_dict_response_is_none(self):
        # a valid but wrong-shaped JSON body must not crash the best-effort probe
        orig = oc._post
        oc._post = lambda *a, **k: ["not", "a", "dict"]
        try:
            self.assertIsNone(oc._detect_context_length("x:cloud"))
        finally:
            oc._post = orig

    def test_resolve_num_ctx_env_override_wins(self):
        old = os.environ.get("OLLAMA_CC_NUM_CTX")
        os.environ["OLLAMA_CC_NUM_CTX"] = "8192"
        try:
            self.assertEqual(oa._resolve_num_ctx("x:cloud"), 8192)
        finally:
            os.environ.pop("OLLAMA_CC_NUM_CTX", None)
            if old is not None:
                os.environ["OLLAMA_CC_NUM_CTX"] = old

    def test_resolve_num_ctx_cloud_detected_is_clamped(self):
        old = os.environ.get("OLLAMA_CC_NUM_CTX")
        os.environ.pop("OLLAMA_CC_NUM_CTX", None)
        orig = oc._detect_context_length
        oc._detect_context_length = lambda m: 1_000_000              # a cloud model advertising a 1M window
        try:
            self.assertEqual(oa._resolve_num_ctx("x:cloud"), oc.NUM_CTX_CEILING)
        finally:
            oc._detect_context_length = orig
            if old is not None:
                os.environ["OLLAMA_CC_NUM_CTX"] = old

    def test_resolve_num_ctx_garbage_env_falls_to_safe_default(self):
        # a typo'd env (e.g. "32k") must NOT silently fall through to a cloud auto-crank
        def _boom(m):
            raise AssertionError("garbage env must not reach context detection")
        old = os.environ.get("OLLAMA_CC_NUM_CTX")
        os.environ["OLLAMA_CC_NUM_CTX"] = "32k"
        orig = oc._detect_context_length
        oc._detect_context_length = _boom
        try:
            self.assertEqual(oa._resolve_num_ctx("x:cloud"), 32768)
        finally:
            oc._detect_context_length = orig
            if old is None:
                os.environ.pop("OLLAMA_CC_NUM_CTX", None)
            else:
                os.environ["OLLAMA_CC_NUM_CTX"] = old

    def test_resolve_num_ctx_local_stays_conservative(self):
        # a local model is NOT auto-cranked (would risk a local-GPU OOM) -> conservative default,
        # and detection is never even attempted for it.
        def _boom(m):
            raise AssertionError("local model must not trigger context detection")
        old = os.environ.get("OLLAMA_CC_NUM_CTX")
        os.environ.pop("OLLAMA_CC_NUM_CTX", None)
        orig_c, orig_d = oc.is_cloud, oc._detect_context_length
        oc.is_cloud, oc._detect_context_length = (lambda m: False), _boom
        try:
            self.assertEqual(oa._resolve_num_ctx("llama3.2:latest"), 32768)
        finally:
            oc.is_cloud, oc._detect_context_length = orig_c, orig_d
            if old is not None:
                os.environ["OLLAMA_CC_NUM_CTX"] = old


class IdleGapTimeoutTest(unittest.TestCase):
    """readonly (adversarial-review) uses idle-gap timeout: each call gets the full window,
    no whole-run cap -- so a steadily-progressing review is not killed mid-flight (max_iters
    stays the backstop). A write run keeps the hard total cap. Catches the regression in both
    directions: idle_gap leaking to the write path (rescue loses its cap) or missing on the
    readonly path (review dies while still making progress)."""

    def setUp(self):
        self._orig_post = oa._post
        self._orig_time = oa.time
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        oa._post, oa.time = self._orig_post, self._orig_time
        shutil.rmtree(self.d, ignore_errors=True)

    def _jump_clock(self):
        """A fake clock that only advances when _post is called (below), so wall time is
        driven deterministically by the loop. Unknown attrs forward to the real time module."""
        real = self._orig_time

        class _C:
            t = 1000.0

            def monotonic(self):
                return self.t

            def __getattr__(self, n):
                return getattr(real, n)
        c = _C()
        oa.time = c
        return c

    def _advancing(self, clock, step=100000.0):
        """_post stub that jumps the clock far past any total cap on every call and asks for a
        fresh (non-repeating) read, so only a timeout / max_iters bound can stop the loop."""
        n = [0]

        def post(path, payload, timeout=None):
            clock.t += step
            n[0] += 1
            return _asst(tool_calls=[_call("read_file", {"path": "f%d.txt" % n[0]})])
        return post

    def test_readonly_idle_gap_has_no_total_cap(self):
        clock = self._jump_clock()
        oa._post = self._advancing(clock)
        r = oa.run_agent("x", self.d, max_iters=3, timeout_total=300, idle_gap=True)
        self.assertEqual(r["stop_reason"], "max_iters")   # NOT "timeout"

    def test_readonly_idle_gap_uses_fixed_per_call_timeout(self):
        # Even with the clock jumped far past the window, each call is given the full window as
        # its (idle) timeout -- never a shrunk-to-1 remainder.
        clock = self._jump_clock()
        with open(os.path.join(self.d, "a.txt"), "w") as f:
            f.write("hi")
        seen = []
        seq = [_asst(tool_calls=[_call("read_file", {"path": "a.txt"})]), _asst(content="ok")]

        def post(path, payload, timeout=None):
            clock.t += 100000.0
            seen.append(timeout)
            return seq.pop(0)
        oa._post = post
        r = oa.run_agent("x", self.d, timeout_total=300, idle_gap=True)
        self.assertEqual(r["stop_reason"], "done")
        self.assertEqual(seen, [300, 300])

    def test_write_path_keeps_total_cap(self):
        clock = self._jump_clock()
        oa._post = self._advancing(clock)
        r = oa.run_agent("x", self.d, max_iters=99, timeout_total=300)  # idle_gap default False
        self.assertEqual(r["stop_reason"], "timeout")

    def test_readonly_idle_gap_floors_nonsensical_timeout(self):
        # A user-supplied --timeout of 0/negative must not reach the socket as-is (0 -> a
        # nonblocking socket). The idle-gap branch floors it at 1 like the write branch.
        seen = []
        seq = [_asst(content="ok")]

        def post(path, payload, timeout=None):
            seen.append(timeout)
            return seq.pop(0)
        oa._post = post
        oa.run_agent("x", self.d, timeout_total=0, idle_gap=True)
        self.assertTrue(seen and all(t >= 1 for t in seen))


class FallbackTest(unittest.TestCase):
    def setUp(self):
        self._orig = oa._post
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        oa._post = self._orig

    def test_fallback_salvages_loop_detected(self):
        # allow_write=True (default) so the loop guard still trips on the 3rd
        # identical read; the forced-synthesis call must then salvage a final.
        with open(os.path.join(self.d, "same.txt"), "w") as f:
            f.write("DATA")
        seq = _SeqPost([_asst(tool_calls=[_call("read_file", {"path": "same.txt"})])] * 3
                       + [_asst(content="synthesized from same.txt")])
        oa._post = seq
        r = oa.run_agent("repeat read", self.d, max_iters=10)
        self.assertEqual(r["stop_reason"], "loop_detected")
        self.assertIn("synthesized", r["final"])
        # the synthesis call must offer no tools
        self.assertEqual(seq.payloads[-1]["tools"], [])

    def test_fallback_salvages_tool_call_cap(self):
        # aborts mid-turn (assistant turn appended, no tool results) -> the orphan
        # turn is stripped before the no-tools synthesis call salvages an answer.
        for n in ("a.txt", "b.txt"):
            with open(os.path.join(self.d, n), "w") as f:
                f.write(n)
        old = oa.TOOL_CALL_CAP
        oa.TOOL_CALL_CAP = 1
        try:
            seq = _SeqPost([_asst(tool_calls=[_call("read_file", {"path": "a.txt"}),
                                               _call("read_file", {"path": "b.txt"})]),
                            _asst(content="synthesized from context")])
            oa._post = seq
            r = oa.run_agent("read both", self.d, max_iters=10)
            self.assertEqual(r["stop_reason"], "tool_call_cap")
            self.assertIn("synthesized", r["final"])
        finally:
            oa.TOOL_CALL_CAP = old

    def test_readonly_repeated_read_does_not_hard_abort(self):
        # readonly idempotent reads are exempt from the loop guard, so the run
        # reaches a bounded stop (max_iters) and the fallback salvages an answer
        # instead of discarding it as loop_detected with empty final.
        with open(os.path.join(self.d, "same.txt"), "w") as f:
            f.write("X")
        seq = _SeqPost([_asst(tool_calls=[_call("read_file", {"path": "same.txt"})])] * 4
                       + [_asst(content="readonly synthesized")])
        oa._post = seq
        r = oa.run_agent("repeat readonly", self.d, allow_write=False, max_iters=4)
        self.assertNotEqual(r["stop_reason"], "loop_detected")
        self.assertIn("synthesized", r["final"])

    def test_fallback_respects_egress_budget(self):
        # no egress left -> the call is never made (degrades to ("", 0) without
        # spending more). Asserts _post is NOT called, so removing the guard fails
        # the test deterministically rather than relying on a daemon being down.
        called = []
        orig = oa._post
        oa._post = lambda path, payload, timeout=None: called.append(payload) or {"message": {"content": "LEAK"}}
        try:
            content, spent = oa._force_synthesis(
                [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}],
                "m", False, 32768, remaining_timeout=60, remaining_egress=0)
        finally:
            oa._post = orig
        self.assertEqual((content, spent), ("", 0))
        self.assertEqual(called, [])  # guard prevented the call entirely

    def test_fallback_respects_timeout(self):
        called = []
        orig = oa._post
        oa._post = lambda path, payload, timeout=None: called.append(payload) or {"message": {"content": "LEAK"}}
        try:
            content, spent = oa._force_synthesis(
                [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}],
                "m", False, 32768, remaining_timeout=0, remaining_egress=oa.EGRESS_BUDGET)
        finally:
            oa._post = orig
        self.assertEqual((content, spent), ("", 0))
        self.assertEqual(called, [])

    def test_strip_drops_orphan_assistant_turn(self):
        # tool_call_cap shape: 2 tool_calls, 0 results -> assistant turn dropped
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "t"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}, {"id": "c2"}]}]
        out = oa._strip_incomplete_trailing_turn(msgs)
        self.assertEqual([m["role"] for m in out], ["system", "user"])

    def test_strip_keeps_complete_trailing_turn(self):
        asst = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]}
        tool = {"role": "tool", "content": "r"}
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "t"},
                asst, tool]
        out = oa._strip_incomplete_trailing_turn(msgs)
        self.assertEqual(len(out), 4)            # nothing dropped
        self.assertIs(out[2], asst)             # the assistant turn is actually kept
        self.assertIs(out[3], tool)

    def test_strip_drops_partial_midturn_orphan(self):
        # loop_detected tripping on tc_k of N leaves last==tool with k<N results:
        # the helper must walk back through the trailing tool to the orphan assistant.
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "t"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}, {"id": "c2"}]},
                {"role": "tool", "tool_call_id": "c1", "content": "error: loop guard"}]
        out = oa._strip_incomplete_trailing_turn(msgs)
        self.assertEqual([m["role"] for m in out], ["system", "user"])  # orphan turn + partial result gone

    def test_fallback_salvages_loop_detected_multi_call_turn(self):
        # the guard trips on the FIRST tool_call of a 2-call turn, leaving 1 result
        # for 2 tool_calls. _strip must drop that orphan turn so the no-tools salvage
        # call doesn't ship an invalid conversation ollama would reject.
        for n in ("same.txt", "other.txt"):
            with open(os.path.join(self.d, n), "w") as f:
                f.write(n)
        seq = _SeqPost([_asst(tool_calls=[_call("read_file", {"path": "same.txt"}, "a1")]),
                        _asst(tool_calls=[_call("read_file", {"path": "same.txt"}, "a2")]),
                        _asst(tool_calls=[_call("read_file", {"path": "same.txt"}, "a3"),
                                          _call("read_file", {"path": "other.txt"}, "a4")]),
                        _asst(content="salvaged multi-call")])
        oa._post = seq
        r = oa.run_agent("repeat multi", self.d, max_iters=10)
        self.assertEqual(r["stop_reason"], "loop_detected")
        self.assertIn("salvaged", r["final"])
        # the fallback payload must carry no orphan assistant turn (every assistant's
        # tool_calls matched by its trailing role:tool results). Pre-fix this failed:
        # 2 tool_calls / 1 result -> ollama reject -> final "".
        self.assertTrue(_no_orphan(seq.payloads[-1]["messages"]))


class TruncateTest(unittest.TestCase):
    def test_truncate_pins_system_and_task(self):
        old = oa.CTX_CHAR_BUDGET
        oa.CTX_CHAR_BUDGET = 500
        try:
            sys_m = {"role": "system", "content": "SYSTEM"}
            task_m = {"role": "user", "content": "TASK"}
            msgs: list = [sys_m, task_m]
            for _ in range(50):
                msgs.append({"role": "assistant", "content": "x" * 40, "tool_calls": []})
                msgs.append({"role": "tool", "tool_name": "read_file", "content": "y" * 40})
            oa._truncate_history(msgs, 2)
            self.assertIs(msgs[0], sys_m)   # system survives
            self.assertIs(msgs[1], task_m)  # original task survives
            self.assertLess(len(msgs), 102)  # history was actually evicted
            if len(msgs) > 2:               # no orphaned role:tool at the front after eviction
                self.assertEqual(msgs[2]["role"], "assistant")
        finally:
            oa.CTX_CHAR_BUDGET = old


class EnvKnobTest(unittest.TestCase):
    def test_int_env_parses_and_falls_back(self):
        name = "OLLAMA_CC_NUM_CTX_TESTONLY"
        os.environ.pop(name, None)
        self.assertEqual(oc._int_env(name, 32768), 32768)        # unset -> default
        os.environ[name] = "8192"
        try:
            self.assertEqual(oc._int_env(name, 32768), 8192)     # parsed
            os.environ[name] = "not-an-int"
            self.assertEqual(oc._int_env(name, 32768), 32768)    # unparseable -> default, no crash
        finally:
            os.environ.pop(name, None)


def _init_repo():
    d = tempfile.mkdtemp()
    subprocess.run(["git", "init", "-q", d], check=True, capture_output=True)
    subprocess.run(["git", "-C", d, "config", "user.email", "t@example.com"], check=True, capture_output=True)
    subprocess.run(["git", "-C", d, "config", "user.name", "t"], check=True, capture_output=True)
    with open(os.path.join(d, "base.txt"), "w") as f:
        f.write("base\n")
    subprocess.run(["git", "-C", d, "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", d, "commit", "-q", "-m", "init"], check=True, capture_output=True)
    return d


class WorktreeTest(unittest.TestCase):
    def setUp(self):
        if shutil.which("git") is None:
            self.skipTest("git not on PATH")
        self._orig = oa._post

    def tearDown(self):
        oa._post = getattr(self, "_orig", oa._post)

    def test_non_git_refused(self):
        r = oa.run_agent_in_worktree("x", tempfile.mkdtemp())
        self.assertEqual(r["stop_reason"], "precondition")

    def test_worktree_isolates_and_captures_diff(self):
        repo = _init_repo()
        oa._post = _SeqPost([_asst(tool_calls=[_call("write_file", {"path": "new.txt", "content": "AGENT"})]),
                             _asst(content="done")])
        r = oa.run_agent_in_worktree("write new.txt", repo)
        self.assertEqual(r["stop_reason"], "done")
        self.assertFalse(os.path.exists(os.path.join(repo, "new.txt")))  # real tree untouched
        self.assertIn("new.txt", r["diff"])                              # change captured in the diff
        self.assertIn("AGENT", r["diff"])
        self.assertTrue(r["base_sha"])
        # worktree was cleaned up: only the main worktree remains
        wtlist = subprocess.run(["git", "-C", repo, "worktree", "list"],
                                capture_output=True, text=True).stdout
        self.assertEqual(wtlist.strip().count("\n"), 0)

    def test_captured_diff_is_apply_able(self):
        # the P2->P3 seam: the diff must cleanly apply onto the base tree that P3
        # will target. repo HEAD == base_sha and its tree is clean, so --check passes.
        repo = _init_repo()
        oa._post = _SeqPost([_asst(tool_calls=[_call("write_file", {"path": "new.txt", "content": "AGENT\n"})]),
                             _asst(content="done")])
        r = oa.run_agent_in_worktree("write new.txt", repo)
        proc = subprocess.run(["git", "-C", repo, "apply", "--check", "--3way", "-"],
                              input=r["diff"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_gitignored_agent_write_is_captured(self):
        # -Af: an agent write under a .gitignore'd path must still appear in the
        # diff, else the work silently vanishes across the P2->P3 seam.
        repo = _init_repo()
        with open(os.path.join(repo, ".gitignore"), "w") as f:
            f.write("build/\n")
        subprocess.run(["git", "-C", repo, "add", "-A"], check=True, capture_output=True)
        subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "ignore"], check=True, capture_output=True)
        oa._post = _SeqPost([_asst(tool_calls=[_call("write_file", {"path": "build/out.txt", "content": "ART"})]),
                             _asst(content="done")])
        r = oa.run_agent_in_worktree("write build/out.txt", repo)
        self.assertEqual(r["stop_reason"], "done")
        self.assertIn("build/out.txt", r["diff"])

    def test_diff_file_written_and_applies_from_path(self):
        # The apply path must use a file (no stdin/heredoc reconstruction of the
        # untrusted patch). The runtime writes diff_file; it applies onto the base.
        repo = _init_repo()
        oa._post = _SeqPost([_asst(tool_calls=[_call("write_file", {"path": "n.txt", "content": "Z\n"})]),
                             _asst(content="done")])
        r = oa.run_agent_in_worktree("write n.txt", repo)
        self.assertTrue(os.path.isfile(r["diff_file"]))
        proc = subprocess.run(["git", "-C", repo, "apply", "--check", "--3way", r["diff_file"]],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_diff_file_is_byte_exact_for_non_utf8(self):
        # A non-UTF-8 file's bytes must survive verbatim into diff_file, else `git apply`
        # fails; _git()'s text decode with errors="replace" would corrupt them to U+FFFD.
        repo = _init_repo()
        with open(os.path.join(repo, "latin.txt"), "wb") as f:
            f.write(b"caf\xe9\n")                        # Latin-1 e-acute = 0xE9, invalid UTF-8
        subprocess.run(["git", "-C", repo, "add", "-A"], check=True, capture_output=True)
        subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "latin"], check=True, capture_output=True)
        # the agent overwrites it -> the diff's removed line carries the raw 0xE9 byte
        oa._post = _SeqPost([_asst(tool_calls=[_call("write_file", {"path": "latin.txt", "content": "cafe\n"})]),
                             _asst(content="done")])
        r = oa.run_agent_in_worktree("edit latin.txt", repo)
        self.assertEqual(r["stop_reason"], "done")
        self.assertIn("diff_file", r)
        with open(r["diff_file"], "rb") as f:
            patch = f.read()
        self.assertIn(b"\xe9", patch)                    # raw non-UTF-8 byte preserved
        self.assertNotIn(b"\xef\xbf\xbd", patch)         # NOT the U+FFFD replacement


class ShellTest(unittest.TestCase):
    def setUp(self):
        self._orig = oa._post
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        oa._post = self._orig

    def test_shell_absent_by_default(self):
        tools, dispatch = oa._toolset(allow_shell=False)
        self.assertNotIn("run_shell", [t["function"]["name"] for t in tools])
        self.assertNotIn("run_shell", dispatch)

    def test_shell_present_and_runs_in_root_when_allowed(self):
        _, dispatch = oa._toolset(allow_shell=True)
        self.assertIn("run_shell", dispatch)
        oa.tool_run_shell(self.d, {"cmd": "python -c \"open('shell_ran.txt','w').write('ok')\""})
        self.assertTrue(os.path.isfile(os.path.join(self.d, "shell_ran.txt")))

    def test_run_shell_unreachable_without_allow_shell(self):
        oa._post = _SeqPost([_asst(tool_calls=[_call("run_shell", {"cmd": "echo hi"})])] * 4)
        r = oa.run_agent("try shell", self.d, allow_shell=False, max_iters=6)
        self.assertEqual(r["stop_reason"], "malformed")  # run_shell is 'unknown' when not allowed

    def test_shell_env_scrubs_secrets_but_keeps_path(self):
        os.environ["OLLAMA_SECRET_TEST"] = "LEAKVALUE"
        try:
            out = oa.tool_run_shell(self.d, {"cmd": (
                "python -c \"import os;print('SEC='+os.environ.get('OLLAMA_SECRET_TEST','ABSENT'));"
                "print('HASPATH='+str('PATH' in os.environ))\"")})
        finally:
            del os.environ["OLLAMA_SECRET_TEST"]
        self.assertIn("SEC=ABSENT", out)     # non-allowlisted parent secret is scrubbed
        self.assertNotIn("LEAKVALUE", out)
        self.assertIn("HASPATH=True", out)   # PATH preserved so real build/test commands work


class ReadToolsTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_list_dir_marks_directories(self):
        os.makedirs(os.path.join(self.d, "sub"))
        open(os.path.join(self.d, "f.txt"), "w").close()
        out = oa.tool_list_dir(self.d, {"path": "."})
        self.assertIn("f.txt", out)
        self.assertIn("sub/", out)

    def test_grep_search_finds_match_with_location(self):
        with open(os.path.join(self.d, "code.py"), "w") as f:
            f.write("alpha\nTARGET_TOKEN here\nbeta\n")
        out = oa.tool_grep_search(self.d, {"pattern": "TARGET_TOKEN"})
        self.assertIn("code.py:2:", out)

    def test_grep_search_jailed(self):
        with self.assertRaises(oa.JailError):
            oa.tool_grep_search(self.d, {"pattern": "x", "path": os.path.join("..", "..")})

    def test_readonly_toolset_excludes_write_and_shell(self):
        _, dispatch = oa._toolset(allow_write=False, allow_shell=False)
        self.assertEqual(set(dispatch), {"read_file", "list_dir", "grep_search"})


class GateTokenTest(unittest.TestCase):
    def _tok(self, value="nonce"):
        p = os.path.join(tempfile.mkdtemp(), "tok")
        with open(p, "w") as f:
            f.write(value)
        return p

    def test_fresh_token_consumed_single_use(self):
        p = self._tok()
        self.assertTrue(oa._consume_gate_token(p))
        self.assertFalse(os.path.exists(p))          # consumed
        self.assertFalse(oa._consume_gate_token(p))  # cannot replay

    def test_missing_or_empty_refused(self):
        self.assertFalse(oa._consume_gate_token(None))
        self.assertFalse(oa._consume_gate_token(os.path.join(tempfile.mkdtemp(), "nope")))
        self.assertFalse(oa._consume_gate_token(self._tok("")))

    def test_stale_token_refused(self):
        p = self._tok()
        os.utime(p, (time.time() - 300, time.time() - 300))
        self.assertFalse(oa._consume_gate_token(p))


class AsClaudeWorktreeTest(unittest.TestCase):
    """B1: as-claude worktree isolation, capture, and failure-path safety (launch seam mocked)."""

    def setUp(self):
        if shutil.which("git") is None:
            self.skipTest("git not on PATH")
        self._orig_launch = oa._launch_claude
        self._orig_git = oa._git
        self._orig_post = oa._post

    def tearDown(self):
        oa._launch_claude = self._orig_launch
        oa._git = self._orig_git
        oa._post = self._orig_post

    def _task_file(self, text="do a thing"):
        tf = os.path.join(tempfile.mkdtemp(), "task.txt")
        with open(tf, "w", encoding="utf-8") as f:
            f.write(text)
        return tf

    def _no_extra_worktree(self, repo):
        wtlist = subprocess.run(["git", "-C", repo, "worktree", "list"],
                                capture_output=True, text=True).stdout
        return wtlist.strip().count("\n") == 0

    def test_as_claude_worktree_captures_committed_change(self):
        # A launched session that commits inside the worktree must still have its change
        # captured. A --cached diff would miss it (index empty vs the new HEAD); the base_sha
        # diff catches it. Also proves isolation (real tree untouched) + worktree cleanup.
        repo = _init_repo()
        base_sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()

        def fake_launch(argv, cwd, task_file, env, timeout=None):
            self.assertEqual(env.get("OLLAMA_AS_CLAUDE_ACTIVE"), "1")
            self.assertEqual(argv[6:10], ["-p", "--dangerously-skip-permissions", "--output-format", "json"])
            with open(os.path.join(cwd, "committed.txt"), "w", encoding="utf-8") as f:
                f.write("committed\n")
            subprocess.run(["git", "-C", cwd, "add", "-A"], check=True, capture_output=True)
            subprocess.run(["git", "-C", cwd, "commit", "-q", "-m", "agent commit"],
                           check=True, capture_output=True)
            return {"is_error": False, "session_id": "00000000-0000-0000-0000-000000000000", "result": "done"}

        oa._launch_claude = fake_launch
        r = oa.run_as_claude_in_worktree(repo, self._task_file(), model="kimi-k2.7-code:cloud")
        self.assertEqual(r["base_sha"], base_sha)
        self.assertEqual(r["stop_reason"], "done")
        self.assertFalse(r["is_error"])
        self.assertTrue(r["has_diff"])
        with open(r["diff_file"], "r", encoding="utf-8") as f:
            self.assertIn("committed.txt", f.read())
        self.assertFalse(os.path.exists(os.path.join(repo, "committed.txt")))  # real tree untouched
        self.assertTrue(self._no_extra_worktree(repo))                          # worktree removed

    def test_launch_error_with_no_edits_offers_no_diff(self):
        # A failed launch that made no edits is flagged is_error with NO diff_file, so the
        # apply gate has nothing to offer -- and an untrusted diff_file key in the launch
        # JSON must NOT survive into the report (forged-patch defense).
        repo = _init_repo()
        oa._launch_claude = lambda *a, **k: {"is_error": True, "session_id": None,
                                             "result": "boom", "diff_file": "C:/forged.patch"}
        r = oa.run_as_claude_in_worktree(repo, self._task_file())
        self.assertTrue(r["is_error"])
        self.assertEqual(r["stop_reason"], "launch_error")
        self.assertFalse(r["has_diff"])
        self.assertNotIn("diff_file", r)          # the untrusted launch key did not leak through
        self.assertTrue(self._no_extra_worktree(repo))

    def test_edits_captured_even_when_launch_reports_error(self):
        # A session can make real edits and still exit nonzero; the diff is still offered.
        repo = _init_repo()

        def fake_launch(argv, cwd, task_file, env, timeout=None):
            with open(os.path.join(cwd, "edited.txt"), "w", encoding="utf-8") as f:
                f.write("x\n")
            return {"is_error": True, "session_id": None, "result": "exited nonzero"}

        oa._launch_claude = fake_launch
        r = oa.run_as_claude_in_worktree(repo, self._task_file())
        self.assertTrue(r["is_error"])
        self.assertTrue(r["has_diff"])
        with open(r["diff_file"], encoding="utf-8") as f:
            self.assertIn("edited.txt", f.read())
        self.assertTrue(self._no_extra_worktree(repo))

    def test_capture_failure_is_fatal_and_worktree_removed(self):
        # If diff capture fails, the session's work vanishes with the worktree -> it must be
        # fatal and surfaced (never silently "no changes"), with no diff_file to apply.
        repo = _init_repo()
        oa._launch_claude = lambda *a, **k: {"is_error": False, "session_id": None, "result": "ok"}
        real_run = oa.subprocess.run

        def failing_run(cmd, *a, **k):
            # fail only the capture diff (unique --no-textconv); worktree add/remove run real
            if isinstance(cmd, (list, tuple)) and "--no-textconv" in cmd:
                return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"boom")
            return real_run(cmd, *a, **k)

        oa.subprocess.run = failing_run
        try:
            r = oa.run_as_claude_in_worktree(repo, self._task_file())
        finally:
            oa.subprocess.run = real_run
        self.assertTrue(r["is_error"])
        self.assertEqual(r["stop_reason"], "diff_error")
        self.assertIn("diff_error", r)
        self.assertNotIn("diff_file", r)
        self.assertFalse(r["has_diff"])
        self.assertTrue(self._no_extra_worktree(repo))

    def test_cli_rejects_unsafe_as_claude_combos(self):
        repo = _init_repo()
        tf = self._task_file()
        with self.assertRaises(SystemExit):                       # --as-claude needs --repo
            oa.main(["--as-claude", "--task-file", tf])
        with self.assertRaises(SystemExit):                       # --as-claude needs --task-file
            oa.main(["--as-claude", "--repo", repo])
        with self.assertRaises(SystemExit):                       # --gate-token replayable if left with --as-claude
            oa.main(["--as-claude", "--repo", repo, "--task-file", tf, "--gate-token", tf])
        with self.assertRaises(SystemExit):                       # --resume dropped (worktree non-resumable)
            oa.main(["--as-claude", "--repo", repo, "--task-file", tf,
                     "--resume", "11111111-1111-1111-1111-111111111111"])

    def test_timeout_kills_the_process_tree(self):
        # Blocker fix: a launch timeout must kill the WHOLE tree, not just `ollama`. The fake
        # launch spawns a grandchild that would create a marker after a delay; if the tree is
        # killed the grandchild dies first and the marker never appears.
        d = tempfile.mkdtemp()
        marker = os.path.join(d, "grandchild_ran.txt")
        grandchild = "import time, sys; time.sleep(2); open(sys.argv[1], 'w').close()"
        parent = ("import subprocess, sys, time\n"
                  "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])\n"
                  "time.sleep(30)\n")
        argv = [sys.executable, "-c", parent, grandchild, marker]
        r = oa._launch_claude(argv, d, self._task_file(), dict(os.environ), timeout=1)
        self.assertTrue(r["is_error"])
        self.assertIn("timed out", r["result"])
        time.sleep(3.5)   # past the grandchild's 2s delay
        self.assertFalse(os.path.exists(marker), "grandchild survived the process-tree kill")


if __name__ == "__main__":
    unittest.main()
