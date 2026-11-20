"""stdin decoding of the two ollama entry points, exercised as real subprocesses.

Regression caught: reading stdin with the console code page (cp949 on Windows) mangled or crashed on
a UTF-8 packet. The API is replaced inside the child by a sitecustomize stub that records the request
bodies and never opens a socket, so no test here can reach the ollama daemon.
"""
import json
import os
import subprocess
import sys

import pytest

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
TEXT = "—…한글"

STUB = '''
import json, os, urllib.request

class _Resp:
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def read(self):
        return json.dumps({"message": {"role": "assistant", "content": "ok"}, "done": True,
                           "done_reason": "stop"}).encode("utf-8")

def _urlopen(req, timeout=None):
    with open(os.environ["STUB_CAPTURE"], "a", encoding="utf-8") as f:
        f.write(json.dumps(json.loads(req.data.decode("utf-8")), ensure_ascii=False) + "\\n")
    return _Resp()

urllib.request.urlopen = _urlopen
'''


def _run(tmp_path, script, args, stdin):
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "sitecustomize.py").write_text(STUB, encoding="utf-8")
    capture = tmp_path / "capture.jsonl"
    env = dict(os.environ,
               PYTHONIOENCODING="cp949",
               PYTHONPATH=str(stub_dir),
               PYTHONDONTWRITEBYTECODE="1",
               STUB_CAPTURE=str(capture),
               OLLAMA_CC_NUM_CTX="4096",
               OLLAMA_CC_HOST="http://127.0.0.1:9")
    proc = subprocess.run([sys.executable, os.path.join(SCRIPTS, script), *args],
                          input=stdin, capture_output=True, env=env, cwd=str(tmp_path), timeout=60)
    requests = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()] \
        if capture.exists() else []
    return proc, requests


ENTRY_POINTS = [
    pytest.param("ollama_companion.py", ["run"], id="companion"),
    pytest.param("ollama_agent.py", ["--root", ".", "--readonly"], id="agent"),
]


def _user_text(requests):
    assert requests, "the stub API saw no request"
    return [m for m in requests[0]["messages"] if m["role"] == "user"][-1]["content"]


@pytest.mark.parametrize("script,args", ENTRY_POINTS)
def test_utf8_stdin_reaches_the_api_unchanged_under_cp949(tmp_path, script, args):
    proc, requests = _run(tmp_path, script, args, TEXT.encode("utf-8"))
    assert _user_text(requests) == TEXT, proc.stderr.decode("utf-8", "replace")


@pytest.mark.parametrize("script,args", ENTRY_POINTS)
def test_utf8_bom_is_stripped_before_the_api(tmp_path, script, args):
    proc, requests = _run(tmp_path, script, args, b"\xef\xbb\xbf" + TEXT.encode("utf-8"))
    assert _user_text(requests) == TEXT, proc.stderr.decode("utf-8", "replace")


@pytest.mark.parametrize("script,args", ENTRY_POINTS)
def test_invalid_utf8_stdin_exits_with_one_line_error(tmp_path, script, args):
    proc, requests = _run(tmp_path, script, args, b"\xff\xfe\xfa")
    stderr = proc.stderr.decode("utf-8", "replace")
    assert proc.returncode != 0
    assert "stdin must be UTF-8" in stderr
    assert len(stderr.strip().splitlines()) == 1, stderr
    assert "Traceback" not in stderr
    assert requests == []
