"""Regression test for the pre-stage diff size caps.

Run: python -m unittest test_ollama_diff_caps (from this directory).
"""
import os
import shutil
import subprocess
import tempfile
import unittest

import ollama_agent as oa


class DiffCapTest(unittest.TestCase):
    def test_oversized_ignored_files_are_excluded_but_source_diff_survives(self):
        if shutil.which("git") is None:
            self.skipTest("git not on PATH")

        with tempfile.TemporaryDirectory() as repo:
            subprocess.run(["git", "init", "-q", repo], check=True, capture_output=True)
            subprocess.run(["git", "-C", repo, "config", "user.email", "t@example.com"],
                           check=True, capture_output=True)
            subprocess.run(["git", "-C", repo, "config", "user.name", "t"],
                           check=True, capture_output=True)

            os.makedirs(os.path.join(repo, "src"))
            source = os.path.join(repo, "src", "main.py")
            with open(os.path.join(repo, ".gitignore"), "w", encoding="utf-8") as f:
                f.write("Binaries/\n")
            with open(source, "w", encoding="utf-8") as f:
                f.write("BASE = 'before'\n" + "# source\n" * 30)
            subprocess.run(["git", "-C", repo, "add", "-A"], check=True, capture_output=True)
            subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "base"],
                           check=True, capture_output=True)

            with open(source, "w", encoding="utf-8") as f:
                f.write("CAP_TEST_MARKER = 'source survives'\n" + "# changed\n" * 30)
            binaries = os.path.join(repo, "Binaries")
            os.makedirs(binaries)

            def make_sized_file(name, size):
                with open(os.path.join(binaries, name), "wb") as f:
                    f.seek(size - 1)
                    f.write(b"\0")

            five_mib = 5 * 1024 * 1024
            make_sized_file("blob.bin", five_mib + 1)  # per-file cap
            for i in range(11):
                make_sized_file("chunk-%02d.bin" % i, five_mib)  # 55 MiB total

            report = oa._stage_limited(repo)
            excluded = {item["path"]: item for item in report["excluded_files"]}

            self.assertIn("Binaries/blob.bin", excluded)
            self.assertIn("file size", excluded["Binaries/blob.bin"]["reason"])
            self.assertTrue(any(path.startswith("Binaries/chunk-")
                                and "total budget" in item["reason"]
                                for path, item in excluded.items()))
            self.assertTrue(report["exclusion_reason"])

            staged = subprocess.run(
                ["git", "-C", repo, "diff", "--cached", "--name-only", "-z"],
                check=True, capture_output=True).stdout.decode("utf-8").split("\0")
            self.assertIn("src/main.py", staged)
            for path in excluded:
                self.assertNotIn(path, staged)

            source_diff = subprocess.run(
                ["git", "-C", repo, "diff", "--cached", "--", "src/main.py"],
                check=True, capture_output=True, text=True).stdout
            self.assertIn("diff --git a/src/main.py b/src/main.py", source_diff)
            self.assertIn("CAP_TEST_MARKER = 'source survives'", source_diff)


if __name__ == "__main__":
    unittest.main()
