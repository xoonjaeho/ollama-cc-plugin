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


def _asst(content="", tool_calls=None):
    m = {"role": "assistant", "content": content}
    if tool_calls is not None:
        m["tool_calls"] = tool_calls
    return {"message": m}


def _call(name, args, cid="c1"):
    return {"id": cid, "function": {"name": name, "arguments": args}}


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

    def test_read_file_is_bounded_and_offset_paginates(self):
        with open(os.path.join(self.d, "big.txt"), "w") as f:
            f.write("A" * (oa.READ_CAP + 100))
        first = oa.tool_read_file(self.d, {"path": "big.txt"})
        self.assertIn("[truncated", first)
        self.assertLessEqual(len(first.encode("utf-8")), oa.READ_CAP + 200)
        rest = oa.tool_read_file(self.d, {"path": "big.txt", "offset": oa.READ_CAP})
        self.assertNotIn("[truncated", rest)
        self.assertTrue(rest.startswith("A"))


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
        self.assertEqual(oa._int_env(name, 32768), 32768)        # unset -> default
        os.environ[name] = "8192"
        try:
            self.assertEqual(oa._int_env(name, 32768), 8192)     # parsed
            os.environ[name] = "not-an-int"
            self.assertEqual(oa._int_env(name, 32768), 32768)    # unparseable -> default, no crash
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


if __name__ == "__main__":
    unittest.main()
