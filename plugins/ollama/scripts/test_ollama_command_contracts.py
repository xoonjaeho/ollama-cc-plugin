"""Regression checks for the rescue/as-claude launch contracts.

Run: python -m unittest test_ollama_command_contracts (from this directory).
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import ollama_agent as oa


SCRIPTS_DIR = Path(__file__).resolve().parent
PLUGIN_DIR = SCRIPTS_DIR.parent


def _read(relative):
    return (PLUGIN_DIR / relative).read_text(encoding="utf-8")


def _init_repo(test):
    repo = tempfile.mkdtemp()
    test.addCleanup(shutil.rmtree, repo, ignore_errors=True)
    subprocess.run(["git", "init", "-q", repo], check=True, capture_output=True)
    subprocess.run(["git", "-C", repo, "config", "user.email", "t@example.com"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", repo, "config", "user.name", "t"],
                   check=True, capture_output=True)
    with open(os.path.join(repo, "base.txt"), "w") as f:
        f.write("base\n")
    subprocess.run(["git", "-C", repo, "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "init"],
                   check=True, capture_output=True)
    return repo


class AsClaudePoolIsolationTest(unittest.TestCase):
    def test_default_launch_disallows_claude_subagents(self):
        if shutil.which("git") is None:
            self.skipTest("git not on PATH")
        task_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, task_dir, ignore_errors=True)
        task_file = os.path.join(task_dir, "task.txt")
        with open(task_file, "w", encoding="utf-8") as f:
            f.write("do a thing")
        seen = {}
        original = oa._launch_claude

        def fake_launch(argv, cwd, task_file, env, timeout=None):
            seen["argv"] = argv
            return {"is_error": False, "session_id": None, "result": "done"}

        oa._launch_claude = fake_launch
        try:
            oa.run_as_claude_in_worktree(_init_repo(self), task_file)
        finally:
            oa._launch_claude = original

        deny = seen["argv"].index("--disallowed-tools")
        self.assertEqual(seen["argv"][deny + 1:], ["Agent", "Task"])

    def test_no_worktree_and_resume_launch_disallow_claude_subagents(self):
        text = _read("commands/as-claude.md")
        self.assertRegex(text, r"If `--resume` or `--no-worktree` is present, go to step 6")
        direct = text.split("6. **Direct launch", 1)[1]
        launch = re.search(r"OLLAMA_AS_CLAUDE_ACTIVE=1.*?\| python -c", direct, re.DOTALL)
        self.assertIsNotNone(launch)
        command = " ".join(launch.group(0).replace("\\", " ").split())
        self.assertIn("--disallowed-tools Agent Task", command)


class MaxItersForwardingTest(unittest.TestCase):
    def test_wrapper_command_includes_max_iters_only_when_supplied(self):
        wrapper = _read("agents/ollama-rescue.md")
        match = re.search(r"```bash\n([^\n]*ollama_agent\.py[^\n]*)\n```", wrapper)
        self.assertIsNotNone(match)
        template = match.group(1)
        marker = "[--max-iters <N>]"
        self.assertIn(marker, template)

        supplied = template.replace(marker, "--max-iters 37")
        omitted = template.replace(marker, "")
        self.assertIn("--max-iters 37", supplied)
        self.assertNotIn("--max-iters", omitted)
        self.assertRegex(wrapper, r"Add `--max-iters <N>` \*\*only if\*\*.*"
                                  r"If it is absent.*omit `--max-iters`")

    def test_commands_forward_max_iters_to_the_runtime_contract(self):
        rescue = _read("commands/rescue.md")
        as_claude = _read("commands/as-claude.md")

        self.assertIn("[--max-iters <N>]", rescue.splitlines()[2])
        self.assertIn("`max_iters: <N>` only if the user specified `--max-iters <N>`", rescue)
        self.assertIn("[--max-iters <N>]", as_claude.splitlines()[2])
        worktree = as_claude.split("5. **Worktree launch", 1)[1].split("6. **Direct launch", 1)[0]
        self.assertRegex(worktree, r"ollama_agent\.py[^\n]*\[--max-iters <N>\]")
        direct = as_claude.split("6. **Direct launch", 1)[1]
        self.assertRegex(direct, r"ollama launch claude[^\n]*\[--max-turns <N>\]")


class AsClaudeMaxItersToMaxTurnsTest(unittest.TestCase):
    def _capture_argv(self, **kwargs):
        task_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, task_dir, ignore_errors=True)
        task_file = os.path.join(task_dir, "task.txt")
        with open(task_file, "w", encoding="utf-8") as f:
            f.write("do a thing")
        seen = {}
        original = oa._launch_claude

        def fake_launch(argv, cwd, task_file, env, timeout=None):
            seen["argv"] = argv
            return {"is_error": False, "session_id": None, "result": "done"}

        oa._launch_claude = fake_launch
        try:
            oa.run_as_claude_in_worktree(_init_repo(self), task_file, **kwargs)
        finally:
            oa._launch_claude = original
        return seen["argv"]

    def test_max_iters_supplied_adds_max_turns_to_launch_argv(self):
        if shutil.which("git") is None:
            self.skipTest("git not on PATH")
        argv = self._capture_argv(max_iters=12)
        self.assertIn("--max-turns", argv)
        self.assertEqual(argv[argv.index("--max-turns") + 1], "12")

    def test_max_iters_omitted_leaves_no_max_turns(self):
        if shutil.which("git") is None:
            self.skipTest("git not on PATH")
        argv = self._capture_argv()
        self.assertNotIn("--max-turns", argv)


if __name__ == "__main__":
    unittest.main()
