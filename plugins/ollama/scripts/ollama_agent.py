#!/usr/bin/env python3
"""ollama-cc-plugin agentic tool loop. stdlib only.

Drives an ollama model over /api/chat with tools (read_file/write_file/list_dir/
grep_search, plus run_shell when --allow-shell), executing them locally inside a
JAILED root until the model stops calling tools or a bound trips. Returns a JSON
report (with a diff when run against a repo).

MODES:
- --repo <git repo>: the write-capable agent -- runs inside an isolated git
  worktree off HEAD, captures a diff, and requires a launch-gate token. The
  user's real tree is never touched; apply the diff after review.
- --root <dir>: runs directly on <dir> (no worktree). Pass --readonly to run it
  safely on a live repo (read tools only); without --readonly it writes directly,
  so use a scratch/throwaway root.
- --allow-shell: adds run_shell = uncontained RCE (opt-in; see the plan).

Cloud model default (glm-5.2:cloud); local/cloud is a config knob
(OLLAMA_CC_MODEL), not a gate.

CONTAINMENT (file tools): no tool path can escape --root. The boundary is realpath
(resolves symlinks/junctions/8.3/case) + commonpath (out-of-root -> denied) + a
':' ban (ADS/drive-relative) + a normalized `.git` deny. An IN-root symlink IS
followed (its target is still in-root, so contained). Residual: TOCTOU between the
realpath check and the later open()/makedirs -- assumes no concurrent FS mutation
(single-actor LLM model). A shell (--allow-shell) is NOT bounded by this jail.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error

# reuse the read-only companion's HTTP + config (same scripts/ dir)
from ollama_companion import (  # noqa: E402
    _post, DEFAULT_MODEL, is_cloud, _final_text, NUM_CTX, _resolve_num_ctx,
)


WRITE_CAP = 1024 * 1024       # 1 MiB per write_file: ample for source, caps a runaway model from filling the worktree disk before review
EGRESS_BUDGET = 8 * 1024 * 1024
MAX_ITERS = 15
TIMEOUT_TOTAL = 900           # seconds, whole run
TOOL_CALL_CAP = 40
MALFORMED_CAP = 3
LOOP_REPEAT_CAP = 3           # identical non-write call N times -> loop
# ponytail: char proxy for the token budget at ~2.75 chars/token, kept under NUM_CTX so the
# server never front-truncates our pinned system+task (client _truncate_history does it first).
CTX_CHAR_BUDGET = int(NUM_CTX * 2.75)
# Diff/staging budgets: prefilter before `git add -Af` so a multi-GB ignored build tree can never
# reach `git diff`, which has fatally malloc'd ~2.5 GB building the binary diff.
DIFF_MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MiB: files above this are almost certainly build artifacts or blobs
DIFF_MAX_TOTAL_BYTES = 50 * 1024 * 1024  # 50 MiB: keeps the patch reviewable and below git's allocator cliff
DIFF_MAX_FILES = 2000  # count guardrail: tens of thousands of build artifacts would bloat the index
TEMP_PATCH_TTL_SECONDS = 3600  # 1 hour: longer than any review/apply window; reclaims crashed-run orphans the same day


def _derive_read_cap(char_budget):
    """Bytes per read/tool result: ~1/3 of the context budget so one read never swamps the
    window (room for the system prompt, task, prior reads and the reply), clamped so it stays
    useful for a tiny model and bounded for a huge one."""
    return max(16 * 1024, min(96 * 1024, char_budget // 3))


READ_CAP = _derive_read_cap(CTX_CHAR_BUDGET)


class JailError(Exception):
    pass


class GitError(Exception):
    pass


# ---------------------------------------------------------------- path jail
def _suggest_relative(root_real, path):
    """What the caller should have typed instead of an absolute path."""
    try:
        # realpath BOTH sides, exactly as resolve_in_jail does: an 8.3 short root
        # (`C:\Users\ADMINI~1\...`, which is what tempfile hands out here) compared
        # against a long-form path yields `..` and a suggestion that is plain wrong.
        rel = os.path.relpath(os.path.realpath(str(path)), os.path.realpath(str(root_real)))
    except ValueError:  # a different drive -- relpath refuses outright
        return "a path relative to the root (this one is on another drive)"
    # `startswith(os.pardir)` alone also matches a real in-root file named `..foo`,
    # which would be reported as outside the root. Only a `..` component escapes.
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return "a path relative to the root (this one is outside it and unreachable)"
    return "'%s'" % rel.replace(os.sep, "/")


def resolve_in_jail(root_real, path):
    """Absolute real path if `path` resolves strictly inside root_real, else
    JailError. See module docstring for the containment boundary."""
    if path is None or not str(path).strip():
        raise JailError("empty path")
    path = str(path)
    # No legit in-jail relative path contains ':' -- bans NTFS alternate data
    # streams (a.txt:stream, .git:stream) and drive-relative (C:foo). On Windows
    # that also rejects every absolute path, INCLUDING one inside the root, so the
    # message has to name the root-relative form or the caller retypes the same
    # path with a different drive letter.
    if ":" in path:
        raise JailError(
            "':' not allowed in path: %s -- tool paths are relative to the "
            "working root (%s), so absolute paths are rejected even when they "
            "point inside it. Retry with %s"
            % (path, root_real, _suggest_relative(root_real, path)))
    # realpath the root too: normalizes 8.3 short names, symlinks, and case so the
    # root prefix actually matches the candidate's resolved prefix.
    root_real = os.path.realpath(root_real)
    candidate = os.path.realpath(os.path.join(root_real, path))
    r = os.path.normcase(root_real)
    c = os.path.normcase(candidate)
    try:
        common = os.path.commonpath([r, c])
    except ValueError:
        raise JailError("path on a different root/drive: %s" % path)
    if common != r:
        raise JailError("path escapes the working root: %s" % path)
    # `.git` deny, robust to NTFS trailing space/dot stripping (`.git ` -> `.git`).
    for part in os.path.relpath(c, r).split(os.sep):
        if part.rstrip(" .").lower() == ".git":
            raise JailError(".git is denied: %s" % path)
    return candidate


# ---------------------------------------------------------------- tools
def _count_newlines_before(path, offset):
    """Count newline bytes in the source prefix [0, offset) without loading the whole file."""
    if offset <= 0:
        return 0
    count = 0
    with open(path, "rb") as f:
        remaining = offset
        while remaining > 0:
            chunk = f.read(min(remaining, READ_CAP))
            if not chunk:
                break
            count += chunk.count(b"\n")
            remaining -= len(chunk)
    return count


def _format_numbered(text, start_line, mark_partial_last):
    """Return text with compact absolute line-number prefixes."""
    if not text:
        return text
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        return text
    last_line = start_line + len(lines) - 1
    width = len(str(last_line))
    out = []
    for i, line in enumerate(lines, start=start_line):
        suffix = " [partial]" if (mark_partial_last and i == last_line) else ""
        out.append("%*d: %s%s" % (width, i, line, suffix))
    return "\n".join(out)


def tool_read_file(root_real, args):
    p = resolve_in_jail(root_real, args.get("path"))
    if not os.path.isfile(p):
        raise JailError("not a file: %s" % args.get("path"))
    try:
        offset = max(0, int(args.get("offset") or 0))
    except (TypeError, ValueError):
        offset = 0
    with open(p, "rb") as f:                 # bounded read: never load the whole file
        if offset:
            f.seek(offset)
        raw = f.read(READ_CAP + 1)
    text = raw[:READ_CAP].decode("utf-8", "replace")
    truncated = len(raw) > READ_CAP
    if text or truncated:
        start_line = _count_newlines_before(p, offset) + 1
        text = _format_numbered(text, start_line, truncated)
    if truncated:
        text += ("\n[truncated at %d bytes; call read_file again with offset=%d for more]"
                 % (READ_CAP, offset + READ_CAP))
    return text


def _next_read_offset(requested, served_to, filesize):
    """Advance a re-read to the next unread chunk. A weaker model that loses track
    re-requests a file at an already-served offset (requested < served_to) instead of
    the next chunk; when the file has more to give, return the next unread offset so
    the read progresses (and its loop-key changes) instead of spinning on one range."""
    if requested < served_to < filesize:
        return served_to
    return requested


def tool_write_file(root_real, args):
    rel = args.get("path")
    p = resolve_in_jail(root_real, rel)      # p is in-root (contained)
    content = args.get("content")
    if content is None:
        raise JailError("write_file requires 'content'")
    nbytes = len(content.encode("utf-8"))
    if nbytes > WRITE_CAP:                    # bound each write so a runaway model can't fill the disk pre-review
        raise JailError("write_file content too large: %d bytes (cap %d)" % (nbytes, WRITE_CAP))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return "wrote %d bytes to %s" % (nbytes, rel)


def tool_list_dir(root_real, args):
    p = resolve_in_jail(root_real, args.get("path") or ".")
    if not os.path.isdir(p):
        raise JailError("not a directory: %s" % args.get("path"))
    entries = sorted(os.listdir(p))
    lines = [("%s/" % e if os.path.isdir(os.path.join(p, e)) else e) for e in entries[:500]]
    if len(entries) > 500:
        lines.append("[+%d more]" % (len(entries) - 500))
    return "\n".join(lines) if lines else "(empty)"


def tool_grep_search(root_real, args):
    pattern = args.get("pattern")
    if not pattern:
        raise JailError("grep_search requires 'pattern'")
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise JailError("invalid regex: %s" % e)
    base = resolve_in_jail(root_real, args.get("path") or ".")
    hits, scanned = [], 0
    # The schema documents `path` as "directory/file", and os.walk() on a file yields nothing --
    # which silently returned "(no matches)" for every pattern. Scan the one file instead.
    walk = ([(os.path.dirname(base), [], [os.path.basename(base)])] if os.path.isfile(base)
            else os.walk(base))
    for dirpath, dirnames, filenames in walk:
        dirnames[:] = [d for d in dirnames if d.lower() != ".git"]  # never descend into .git
        for fn in filenames:
            if scanned >= 5000:
                hits.append("[scan cap 5000 files reached; narrow the path]")
                return "\n".join(hits)
            scanned += 1
            fp = os.path.join(dirpath, fn)
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            hits.append("%s:%d: %s" % (os.path.relpath(fp, root_real), i, line.rstrip()[:200]))
                            if len(hits) >= 200:
                                hits.append("[+more matches; refine the pattern]")
                                return "\n".join(hits)
            except (OSError, ValueError):
                continue
    return "\n".join(hits) if hits else "(no matches)"


def _cap(text, limit=READ_CAP):
    b = text.encode("utf-8")
    if len(b) <= limit:
        return text
    return b[:limit].decode("utf-8", "replace") + "\n[truncated %d bytes]" % (len(b) - limit)


# Allowlisted OS/build env vars: enough for normal build/test commands, without
# forwarding the parent's other vars (so the shell can't trivially dump env-embedded
# secrets). Defense-in-depth on an RCE, not a boundary -- the shell can read files.
_SHELL_ENV_KEYS = ("PATH", "PATHEXT", "SYSTEMROOT", "SystemRoot", "WINDIR", "COMSPEC",
                   "TEMP", "TMP", "TMPDIR", "HOME", "HOMEDRIVE", "HOMEPATH", "USERPROFILE",
                   "APPDATA", "LOCALAPPDATA", "LANG", "LC_ALL", "OS", "PROCESSOR_ARCHITECTURE",
                   "NUMBER_OF_PROCESSORS", "USERNAME", "USER", "SHELL")


def _kill_tree(p):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        p.wait(timeout=5)
    except Exception:
        pass


def tool_run_shell(root_real, args):
    """Run a shell command with the worktree as cwd. OPT-IN (--allow-shell) and full
    RCE: a shell is NOT confined by the worktree/path-jail (it can cd/read/write/
    network anywhere on the host). Real exposure is accepted per the plan's safety
    section. On timeout the whole process tree is killed; a command that deliberately
    daemonizes a child can still outlive the run (a leaked worktree then surfaces as
    cleanup_error) -- an accepted residual of running an arbitrary shell."""
    cmd = args.get("cmd")
    if not cmd or not str(cmd).strip():
        raise JailError("run_shell requires 'cmd'")
    env = {k: os.environ[k] for k in _SHELL_ENV_KEYS if k in os.environ}
    # new process group/session so the whole tree can be killed on timeout; decode
    # child output as UTF-8, not the host cp949 codepage (else non-ASCII output crashes)
    if os.name == "nt":
        p = subprocess.Popen(str(cmd), shell=True, cwd=root_real, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             encoding="utf-8", errors="replace",
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        p = subprocess.Popen(str(cmd), shell=True, cwd=root_real, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             encoding="utf-8", errors="replace",
                             start_new_session=True)
    try:
        out, err = p.communicate(timeout=60)
        note = "\n[exit %d]" % p.returncode
    except subprocess.TimeoutExpired:
        _kill_tree(p)              # kill the whole tree, not just the shell parent
        try:
            out, err = p.communicate(timeout=5)   # drain whatever was buffered before the kill
        except Exception:
            out, err = "", ""
        note = "\n[killed: timed out after 60s]"
    parts = []
    if out:
        parts.append(_cap(out))
    if err:                        # cap stderr separately so a noisy stdout can't hide the failure
        parts.append("[stderr]\n" + _cap(err))
    return ("\n".join(parts) if parts else "(no output)") + note


_READ_TOOL = {"type": "function", "function": {
    "name": "read_file",
    "description": "Read a UTF-8 text file inside the working root. Path is relative to the working root. "
                   "Large files return the first chunk; pass offset (byte offset) to read further.",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string", "description": "path relative to the working root"},
                                  "offset": {"type": "integer", "description": "byte offset to start reading (default 0)"}},
                   "required": ["path"]}}}
_WRITE_TOOL = {"type": "function", "function": {
    "name": "write_file",
    "description": "Create or overwrite a UTF-8 text file inside the working root. Path is relative to the working root.",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                   "required": ["path", "content"]}}}
_SHELL_TOOL = {"type": "function", "function": {
    "name": "run_shell",
    "description": "Run a shell command with the working root as cwd. Use for builds, tests, and commands.",
    "parameters": {"type": "object",
                   "properties": {"cmd": {"type": "string", "description": "the shell command"}},
                   "required": ["cmd"]}}}
_LIST_TOOL = {"type": "function", "function": {
    "name": "list_dir",
    "description": "List entries of a directory inside the working root (directories end with '/'). Path is relative to the working root; defaults to '.'.",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string", "description": "directory path relative to the working root"}},
                   "required": []}}}
_GREP_TOOL = {"type": "function", "function": {
    "name": "grep_search",
    "description": "Search files under a path for a regex, returning file:line: matches. Cheaper than reading whole files.",
    "parameters": {"type": "object",
                   "properties": {"pattern": {"type": "string", "description": "a Python regular expression"},
                                  "path": {"type": "string", "description": "directory/file to search under (default '.')"}},
                   "required": ["pattern"]}}}


_READ_ONLY_TOOLS = (_READ_TOOL, _LIST_TOOL, _GREP_TOOL)
# single source of truth for the always-present read tools: _toolset's tool list
# and the readonly loop-guard exemption (_IDEMPOTENT_READS) both derive from this.


def _toolset(allow_write=True, allow_shell=False):
    # read tools (incl. the list/grep egress-reducers) are always available
    tools = list(_READ_ONLY_TOOLS)
    dispatch = {"read_file": tool_read_file, "list_dir": tool_list_dir, "grep_search": tool_grep_search}
    if allow_write:
        tools.append(_WRITE_TOOL)
        dispatch["write_file"] = tool_write_file
    if allow_shell:
        tools.append(_SHELL_TOOL)
        dispatch["run_shell"] = tool_run_shell
    return tools, dispatch


# ---------------------------------------------------------------- context mgmt
def _serialized_size(messages):
    return len(json.dumps(messages, ensure_ascii=False))


def _truncate_history(messages, pinned):
    """Pair-aware eviction: drop the oldest non-pinned assistant turn together
    with its following role:tool results. Never evicts messages[:pinned]
    (system + original task), and never leaves an orphan role:tool at the front."""
    while _serialized_size(messages) > CTX_CHAR_BUDGET and len(messages) > pinned + 1:
        del messages[pinned]
        while pinned < len(messages) and messages[pinned].get("role") == "tool":
            del messages[pinned]
    return messages


# Bounded stops where a forced no-tools synthesis call can salvage the work already
# in context. timeout/egress_budget are excluded: timeout leaves no time to make the
# call, and egress_budget must hold -- spending past the cap to salvage would defeat
# it. So a readonly runaway that hits egress/timeout before max_iters still returns
# empty; the readonly rescue only salvages when an iter/call cap binds first.
# api_error/malformed are excluded: the model is unreachable or producing junk.
_SYNTH_STOPS = {"loop_detected", "max_iters", "tool_call_cap"}
_SYNTH_INSTRUCTION = ("Stop calling tools. Using only the information already in "
                      "context, produce the final answer now.")
# readonly, side-effect-free reads: exempt from the hard loop abort so a readonly
# task isn't killed for a re-read. MAX_ITERS / TOOL_CALL_CAP / egress / timeout +
# the forced-synthesis fallback remain as runaway backstops.
_IDEMPOTENT_READS = tuple(t["function"]["name"] for t in _READ_ONLY_TOOLS)


def _chat_payload(model, messages, tools, think, num_ctx):
    """Build the /api/chat request body. Shared by the main loop and the fallback so
    the payload schema can't drift between them."""
    return {"model": model, "messages": messages, "tools": tools,
            "stream": False, "think": bool(think),
            "options": {"temperature": 0.2, "num_ctx": num_ctx}}


def _strip_incomplete_trailing_turn(messages):
    """Return a copy of messages with a trailing assistant turn whose tool_calls
    lack matching tool results removed. A bounded stop can leave such an orphan two
    ways: tool_call_cap aborts before any result is appended (last == assistant, 0
    results); loop_detected trips mid-turn on tc_k of N and appends only k results
    (last == tool, k < N). Feeding an orphan to a no-tools /api/chat makes ollama
    reject 'expected a tool result'."""
    if not messages:
        return list(messages)
    # walk back through the trailing run of role:tool results to its anchor assistant.
    have = 0
    i = len(messages) - 1
    while i >= 0 and messages[i].get("role") == "tool":
        have += 1
        i -= 1
    if i < 0:
        return list(messages)
    last = messages[i]
    if last.get("role") != "assistant" or not last.get("tool_calls"):
        return list(messages)
    if have >= len(last["tool_calls"]):
        return list(messages)
    # drop the incomplete assistant turn and any partial results after it.
    return list(messages[:i])


def _force_synthesis(messages, model, think, num_ctx, remaining_timeout, remaining_egress):
    """One no-tools /api/chat to salvage a final answer from context on a bounded
    stop. Returns (content, egress_spent). Degrades to ("", 0) when out of budget,
    the call fails, or the model returns empty -- never raises."""
    if remaining_timeout < 1 or remaining_egress <= 0:
        return "", 0
    msgs = _strip_incomplete_trailing_turn(messages)
    payload = _chat_payload(model, msgs + [{"role": "user", "content": _SYNTH_INSTRUCTION}],
                            [], think, num_ctx)
    projected = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    if projected > remaining_egress:
        return "", 0
    # charge on attempt, matching the main loop (which adds `projected` before _post).
    try:
        data = _post("/api/chat", payload, timeout=max(1, int(remaining_timeout)))
    except Exception:  # noqa: BLE001 - fallback must never crash the run
        return "", projected
    if not isinstance(data, dict) or data.get("error"):
        return "", projected
    msg = data.get("message")
    content = _final_text(msg) if isinstance(msg, dict) else ""
    return content, projected


# ---------------------------------------------------------------- the loop
def _system_prompt(root, allow_write=True):
    # The "must write, else FAILED" mandate applies ONLY when write_file exists. A --readonly
    # run has no write tool and a read-only completion is a valid, finished run -- telling it
    # the run FAILED and to call write_file would break every review-only task.
    if allow_write:
        how = ("Use read_file to inspect and write_file to make changes. You keep the full "
               "content of every file you have already read -- do NOT read the same file again. "
               "Once you understand the task, STOP reading and call write_file; a run that only "
               "reads and never writes has FAILED. ")
    else:
        how = ("Use read_file, list_dir and grep_search to inspect -- this run is READ-ONLY, "
               "there is no write tool. You keep the full content of every file you have already "
               "read -- do NOT read the same file again. Once you have enough context, STOP "
               "reading and give your final answer. ")
    return (
        "You are a coding agent working inside a working root you cannot escape: "
        "every path you pass to a tool is relative to that root and is confined to it. " + how +
        "Do only what the task asks. When the task is complete, reply with a short "
        "final summary and DO NOT call any more tools.\n"
        "Working root: %s" % root)


def _api_error(e):
    """Stop-reason text for a failed `/api/chat` call.

    `type(e).__name__` alone renders every rejection as a bare `HTTPError`, which says nothing
    about what the server refused -- context length, auth, a bad payload field all read alike.
    The status and the body are what make the next occurrence self-diagnosing, so keep both,
    bounded. A body that cannot be read must never mask the status.
    """
    if not isinstance(e, urllib.error.HTTPError):
        return "api_error:%s" % type(e).__name__
    try:
        raw = e.read(4096).decode("utf-8", "replace").strip()
    except Exception:  # noqa: BLE001 - consumed, closed, or absent body
        raw = ""
    try:
        detail = str(json.loads(raw).get("error", "")) or raw
    except Exception:  # noqa: BLE001 - HTML error pages and truncated JSON are still informative
        detail = raw
    return "api_error:HTTP %s%s" % (e.code, (" %s" % detail[:300]) if detail else "")


def _append_tool(messages, tool_call_id, name, content):
    m = {"role": "tool", "tool_name": name, "content": content}
    if tool_call_id:
        m["tool_call_id"] = tool_call_id
    messages.append(m)


def run_agent(task, root, model=None, think=False, allow_write=True, allow_shell=False,
              max_iters=MAX_ITERS, timeout_total=TIMEOUT_TOTAL, idle_gap=False):
    model = model or DEFAULT_MODEL
    tools, dispatch = _toolset(allow_write=allow_write, allow_shell=allow_shell)
    root_real = os.path.realpath(root)
    if not os.path.isdir(root_real):
        return {"stop_reason": "bad_root", "error": "root is not a directory: %s" % root}
    messages = [{"role": "system", "content": _system_prompt(root_real, allow_write=allow_write)},
                {"role": "user", "content": task}]
    pinned = 2
    iters = tool_calls_total = malformed = egress_bytes = 0
    seen = {}                 # loop-key -> count
    written = set()           # canonical paths written -> exempt a following re-read from loop-kill
    read_progress = {}        # canon path -> next unread byte offset (drives read auto-advance)
    actions = []
    start = time.monotonic()
    empty_retry_done = False

    def _req_timeout():
        # idle_gap (readonly adversarial-review): each call gets the full window as an idle
        # timeout -- matching /review's stream semantics, a steadily-progressing review is not
        # killed by a whole-run cap (max_iters/egress stay the runaway backstops). A write run
        # keeps a hard total budget that shrinks per call.
        if idle_gap:
            return max(1, int(timeout_total))   # floor a nonsensical --timeout <=0 like the write branch
        return max(1, int(timeout_total - (time.monotonic() - start)))

    def stop(reason, final=""):
        nonlocal egress_bytes
        if not final and reason in _SYNTH_STOPS:
            # trim to budget first: max_iters bails before _truncate_history, so the
            # in-context messages can be one turn over budget; without this ollama
            # front-truncates the pinned system+task (the context salvage needs).
            _truncate_history(messages, pinned)
            final, spent = _force_synthesis(
                messages, model, think, NUM_CTX,
                timeout_total if idle_gap else timeout_total - (time.monotonic() - start),
                EGRESS_BUDGET - egress_bytes)
            egress_bytes += spent
        partial_work = (reason != "done" and any(
            a.get("ok") and a.get("tool") in ("write_file", "run_shell") for a in actions))
        written_paths = [a["args"]["path"] for a in actions
                         if a.get("ok") and a.get("tool") == "write_file"
                         and isinstance(a.get("args"), dict) and "path" in a["args"]]
        return {"stop_reason": reason, "iterations": iters, "final": final,
                "actions": actions, "egress_bytes": egress_bytes,
                "tool_calls": tool_calls_total, "model": model,
                "partial_work": partial_work, "written_paths": written_paths}

    while True:
        iters += 1
        if iters > max_iters:
            return stop("max_iters")
        elapsed = time.monotonic() - start
        if not idle_gap and elapsed > timeout_total:
            return stop("timeout")
        _truncate_history(messages, pinned)
        payload = _chat_payload(model, messages, tools, think, NUM_CTX)
        projected = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        if egress_bytes + projected > EGRESS_BUDGET:   # check BEFORE it leaves the process
            return stop("egress_budget")
        egress_bytes += projected
        try:
            data = _post("/api/chat", payload, timeout=_req_timeout())
        except Exception as e:  # noqa: BLE001 - surface any transport/parse failure as a stop
            return stop(_api_error(e))
        if not isinstance(data, dict):
            return stop("api_error:non-dict-response")
        if data.get("error"):   # ollama can answer 200 with an {"error": ...} body
            return stop("api_error:%s" % str(data["error"])[:120])
        msg = data.get("message") or {}
        messages.append(msg)                       # assistant turn (may carry tool_calls)
        tcs = msg.get("tool_calls") or []
        if not tcs:
            final = _final_text(msg)
            if final or empty_retry_done:
                return stop("done", final=final)
            # Empty assistant turn: one same-iteration retry so iters is not incremented
            # and preflight is not re-run. Charge egress independently.
            empty_retry_done = True
            payload2 = _chat_payload(model, messages[:-1], tools, think, NUM_CTX)
            proj2 = len(json.dumps(payload2, ensure_ascii=False).encode("utf-8"))
            if egress_bytes + proj2 > EGRESS_BUDGET:
                return stop("egress_budget")
            egress_bytes += proj2
            try:
                data = _post("/api/chat", payload2, timeout=_req_timeout())
            except Exception as e:  # noqa: BLE001
                return stop(_api_error(e))
            if not isinstance(data, dict):
                return stop("api_error:non-dict-response")
            if data.get("error"):
                return stop("api_error:%s" % str(data["error"])[:120])
            msg = (data or {}).get("message") or {}
            messages[-1] = msg                  # replace the empty assistant turn
            tcs = msg.get("tool_calls") or []
            if not tcs:
                return stop("done", final=_final_text(msg))
            # retry produced tool_calls: fall through to the preflight/dispatch below
        # preflight the whole turn against the cap so we never partially apply it
        if tool_calls_total + len(tcs) > TOOL_CALL_CAP:
            return stop("tool_call_cap")
        for tc in tcs:
            tool_calls_total += 1
            fn = (tc.get("function") or {})
            name = fn.get("name")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = None
            tool_call_id = tc.get("id")
            if name not in dispatch or not isinstance(args, dict):
                malformed += 1
                _append_tool(messages, tool_call_id, name or "?",
                             "error: unknown or malformed tool call '%s'" % name)
                actions.append({"tool": name, "args": args, "ok": False})
                if malformed >= MALFORMED_CAP:
                    return stop("malformed")
                continue
            # canonicalize the file path so "out.txt"/"./out.txt" don't diverge in
            # the loop-key or written-set; keep offset etc. so paginated reads differ.
            canon = None
            if "path" in args:
                try:
                    canon = resolve_in_jail(root_real, args["path"])
                except JailError:
                    canon = None
            # Auto-advance a re-read to the next unread chunk so a model that re-requests
            # an already-served range progresses instead of spinning until the loop guard
            # aborts. Changing the offset also changes the loop-key below, so real progress
            # never counts as a loop.
            if name == "read_file" and canon and os.path.isfile(canon):
                try:
                    req_off = max(0, int(args.get("offset") or 0))
                except (TypeError, ValueError):
                    req_off = 0
                eff_off = _next_read_offset(req_off, read_progress.get(canon, 0),
                                            os.path.getsize(canon))
                if eff_off != req_off:
                    args = {**args, "offset": eff_off}
            key_args = dict(args)
            if canon:
                key_args["path"] = canon
            key = (name, json.dumps(key_args, sort_keys=True, ensure_ascii=False))
            is_reread_after_write = (name == "read_file" and canon is not None and canon in written)
            seen[key] = seen.get(key, 0) + 1
            readonly_idempotent = (not allow_write and name in _IDEMPOTENT_READS)
            if (seen[key] >= LOOP_REPEAT_CAP
                    and name != "write_file"
                    and not is_reread_after_write
                    and not readonly_idempotent):
                _append_tool(messages, tool_call_id, name, "error: repeated identical call (loop guard)")
                return stop("loop_detected")
            try:
                result = dispatch[name](root_real, args)
                ok = True
                if name == "write_file":
                    written.add(canon)
                    read_progress.pop(canon, None)   # file changed -> stale read offsets no longer valid (else a read-back auto-advances past the new content)
                elif name == "read_file":
                    written.discard(canon)
                    if canon and os.path.isfile(canon):
                        try:
                            eff_off = max(0, int(args.get("offset") or 0))
                        except (TypeError, ValueError):
                            eff_off = 0
                        read_progress[canon] = max(read_progress.get(canon, 0),
                                                   min(os.path.getsize(canon), eff_off + READ_CAP))
            except JailError as e:
                result, ok = "error: %s" % e, False
            except Exception as e:  # noqa: BLE001
                result, ok = "error: %s: %s" % (type(e).__name__, e), False
            actions.append({"tool": name, "args": args, "ok": ok})
            _append_tool(messages, tool_call_id, name, result)


# ---------------------------------------------------------------- git worktree (P2)
def _git(repo, *args, check=False):
    # Decode git output as UTF-8, not the host cp949 codepage, else a diff/status
    # carrying non-ASCII bytes crashes on a Korean-Windows host.
    # ponytail: errors=replace can't round-trip a non-UTF-8 blob byte-exact for a
    # later `git apply`; safe for UTF-8 repos, switch the diff capture to binary
    # mode if a non-UTF-8 repo ever needs it.
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise GitError("git %s failed: %s" % (args[0], (r.stderr or r.stdout).strip()))
    return r


def _porcelain_name(line):
    """The path out of one `git status --porcelain` line.

    Two shapes defeat a plain `line[3:]`: a rename/copy reads `R  old -> new`, where
    only the new path is what the agent would miss, and `core.quotepath` wraps a path
    containing spaces or non-ASCII in double quotes.
    """
    name = line[3:]
    if " -> " in name:
        name = name.split(" -> ", 1)[1]
    return name.strip('"')


def _git_state(repo):
    """Preconditions for a rescue run; raises GitError to refuse."""
    if _git(repo, "rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        raise GitError("not a git work tree: %s (non-git repos are refused)" % repo)
    head = _git(repo, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise GitError("no HEAD commit (unborn branch); commit something first")
    git_dir = _git(repo, "rev-parse", "--git-dir").stdout.strip()
    gd = git_dir if os.path.isabs(git_dir) else os.path.join(repo, git_dir)
    # in-progress sequencer ops leave the base ambiguous -> refuse. `sequencer`
    # covers multi cherry-pick/revert runs outside the conflicted-head window.
    for marker in ("rebase-merge", "rebase-apply", "MERGE_HEAD", "CHERRY_PICK_HEAD",
                   "REVERT_HEAD", "sequencer"):
        if os.path.exists(os.path.join(gd, marker)):
            raise GitError("a %s is in progress; finish or abort it first" % marker)
    porcelain = _git(repo, "status", "--porcelain").stdout.strip()
    if porcelain:
        # Both callers of this function go on to check out a worktree at the captured
        # HEAD, so uncommitted work is invisible to the agent by construction. The
        # `dirty_base` flag in the final report says the same thing -- but only after
        # the run has been paid for. Warn, never refuse: HEAD may be exactly what the
        # caller meant. One guard here covers both entry points.
        names = [_porcelain_name(line) for line in porcelain.splitlines()]
        shown = ", ".join(names[:10]) + (" ..." if len(names) > 10 else "")
        print("warning: %d uncommitted change(s) in %s will NOT be visible to the agent "
              "-- the worktree is checked out at HEAD. Commit them first if the agent "
              "needs them: %s" % (len(names), repo, shown), file=sys.stderr)
    return {"base_sha": head.stdout.strip(),
            "dirty": bool(porcelain),
            "detached": _git(repo, "symbolic-ref", "-q", "HEAD").returncode != 0}


def _make_worktree(repo, base_sha):
    # add at the CAPTURED base_sha (not live HEAD) so the reported base and the
    # diff base are one read -- a concurrent commit/checkout can't desync them.
    wt = tempfile.mkdtemp(prefix="ollama-wt-")
    os.rmdir(wt)  # `git worktree add` wants a non-existent path
    r = _git(repo, "worktree", "add", "--detach", wt, base_sha)
    if r.returncode != 0:
        _git(repo, "worktree", "prune")  # reap a partial admin entry (dir was rmdir'd)
        raise GitError("worktree add failed: %s" % (r.stderr or r.stdout).strip())
    return wt


def _remove_worktree(repo, wt):
    """Scoped removal (no global prune -> can't touch a concurrent run's worktree).
    Returns True iff the worktree dir is gone afterwards."""
    _git(repo, "worktree", "remove", "--force", wt)
    return not os.path.exists(wt)


def _stage_limited(wt):
    """Enumerate what `git add -Af` would stage, drop files that exceed the diff
    budget, then stage only the survivors. Returns {excluded_files, exclusion_reason}.
    Prefiltering before staging is what stops a multi-GB ignored build tree from
    reaching `git diff`, which has fatally malloc'd ~2.5 GB building the binary diff."""
    try:
        out = subprocess.run(["git", "-C", wt, "ls-files", "-z", "--modified",
                              "--deleted", "--others"],
                             capture_output=True, timeout=120)
    except subprocess.TimeoutExpired as e:
        raise GitError("git ls-files timed out: %s" % e)
    if out.returncode != 0:
        raise GitError("git ls-files failed: %s" % out.stderr.decode("utf-8", "replace").strip())
    candidates = []
    for raw in out.stdout.split(b"\0"):
        if not raw:
            continue
        path = raw.decode("utf-8", "replace")
        full = os.path.join(wt, path)
        try:
            size = 0 if not os.path.exists(full) else os.path.getsize(full)
        except OSError:
            continue
        candidates.append((path, size))
    # Prefer smaller files so the cumulative cap keeps the most reviewable work.
    candidates.sort(key=lambda x: x[1])
    accepted = []
    excluded = []
    total_bytes = 0
    for path, size in candidates:
        if size > DIFF_MAX_FILE_BYTES:
            excluded.append({"path": path, "size": size,
                             "reason": "file size %d exceeds %d bytes" % (size, DIFF_MAX_FILE_BYTES)})
            continue
        if len(accepted) >= DIFF_MAX_FILES or total_bytes + size > DIFF_MAX_TOTAL_BYTES:
            excluded.append({"path": path, "size": size,
                             "reason": "would exceed total budget (%d files or %d bytes)" % (DIFF_MAX_FILES, DIFF_MAX_TOTAL_BYTES)})
            continue
        accepted.append(path)
        total_bytes += size
    if accepted:
        batch = 100
        for i in range(0, len(accepted), batch):
            r = subprocess.run(["git", "-C", wt, "add", "-f", "--"] + accepted[i:i + batch],
                               capture_output=True, timeout=120)
            if r.returncode != 0:
                raise GitError("git add failed: %s" % r.stderr.decode("utf-8", "replace").strip())
    reason = None
    if excluded:
        reason = "%d file(s) excluded by diff budget (per-file > %d bytes, total <= %d bytes / %d files)" % (
            len(excluded), DIFF_MAX_FILE_BYTES, DIFF_MAX_TOTAL_BYTES, DIFF_MAX_FILES)
    return {"excluded_files": excluded, "exclusion_reason": reason}


def _claim_patch(path):
    """Write a sibling lease file so a concurrent sweep knows this patch may still be in use.
    A live caller renews the lease by touching it; a dead caller's lease expires after TTL."""
    try:
        with open(path + ".lease", "w", encoding="utf-8") as f:
            f.write("claimed\n")
    except OSError:
        pass


def run_agent_in_worktree(task, repo, **kw):
    """Run the agent inside an isolated worktree at the captured base_sha; the
    user's real tree is never touched. Returns the P1 report plus diff/base_sha;
    the worktree is force-removed on every exit path (done, bound, crash)."""
    repo = os.path.realpath(repo)
    try:
        state = _git_state(repo)
    except GitError as e:
        return {"stop_reason": "precondition", "error": str(e)}
    try:
        wt = _make_worktree(repo, state["base_sha"])
    except Exception as e:  # GitError, or OSError from tempfile/rmdir
        return {"stop_reason": "worktree_error", "error": str(e), "base_sha": state["base_sha"]}
    try:
        report = run_agent(task, wt, **kw)
        stage_meta = {"excluded_files": [], "exclusion_reason": None}
        try:
            # Prefilter before `git add -Af`: a multi-GB ignored build tree must not reach
            # git's diff allocator. _stage_limited still captures small ignored agent writes.
            stage_meta = _stage_limited(wt)
            # Capture the diff as raw BYTES via a direct subprocess. _git() decodes with
            # errors="replace", which corrupts a non-UTF-8 file's bytes to U+FFFD so the patch
            # no longer applies. report["diff"] keeps a lossy-decoded copy for display; diff_file
            # holds the byte-exact patch the apply gate uses. --no-textconv keeps the raw diff.
            dr = subprocess.run(["git", "-C", wt, "diff", "--cached", "--binary", "--no-textconv"],
                                capture_output=True, timeout=120)
            if dr.returncode != 0:
                raise GitError("git diff failed: %s" % dr.stderr.decode("utf-8", "replace").strip())
            diff_bytes = dr.stdout
            report["diff"] = diff_bytes.decode("utf-8", "replace")
            if diff_bytes.strip():
                # Write the patch to a file OUTSIDE the worktree so the command applies it by
                # path (`git apply <file>`) and never reconstructs untrusted patch text through
                # the shell. Binary write preserves the bytes; survives the cleanup below.
                fd, dpath = tempfile.mkstemp(prefix="ollama-diff-", suffix=".patch")
                with os.fdopen(fd, "wb") as f:
                    f.write(diff_bytes)
                report["diff_file"] = dpath
                _claim_patch(dpath)
        except (GitError, subprocess.TimeoutExpired) as e:
            # Capture failed/hung and the worktree is about to be removed -> the work is gone.
            # Fail loudly so the caller never reports "no changes"; offer nothing to apply.
            report["is_error"] = True
            report["stop_reason"] = "diff_error"
            report["diff_error"] = str(e) or "git capture timed out"
            report.pop("diff_file", None)
        report.update(stage_meta)
    finally:
        removed = _remove_worktree(repo, wt)
    if not removed:
        report["cleanup_error"] = "worktree not removed (may be locked): %s" % wt
    report.update({"base_sha": state["base_sha"], "dirty_base": state["dirty"],
                   "detached": state["detached"]})
    return report


# ---------------------------------------------------------------- launch-gate token (P3)
def _consume_gate_token(path):
    """Fail-closed interlock: the agentic --repo run proceeds only if a fresh,
    single-use token file exists (minted by the /ollama:rescue launch gate after
    its egress/RCE disclosure). Stops the write/shell runtime from running when
    the disclosure gate was skipped. NOT a barrier against a caller that mints its
    own token -- an anti-accidental-skip interlock (see plan safety section)."""
    if not path or not os.path.isfile(path):
        return False
    try:
        age = time.time() - os.path.getmtime(path)
        with open(path, "r", encoding="utf-8") as f:
            val = f.read().strip()
        os.remove(path)  # single-use: consume so it can't be replayed
    except OSError:
        return False
    return bool(val) and 0 <= age < 120


# ---------------------------------------------------------------- CLI
def _launch_claude(argv, cwd, task_file, env, timeout=TIMEOUT_TOTAL):
    """Launch `ollama launch claude` with a task on stdin, in its own process group so the
    WHOLE tree is killed on timeout (a plain subprocess timeout reaps only `ollama`, leaving
    the `claude` RCE descendant running after we give up). Returns an ALLOWLISTED dict
    {is_error, result, session_id, returncode} -- never the raw launch JSON, so a
    model-controlled `result` cannot smuggle gate-owned keys (diff_file, base_sha) into the
    caller's report. Fail-closed: a nonzero exit or unparseable output => is_error."""
    try:
        fh = open(task_file, "r", encoding="utf-8")
    except OSError as e:
        return {"is_error": True, "result": "cannot read task file: %s" % e,
                "session_id": None, "returncode": None}
    p = None
    try:
        try:
            if os.name == "nt":
                p = subprocess.Popen(argv, cwd=cwd, stdin=fh, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                     errors="replace", env=env,
                                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                p = subprocess.Popen(argv, cwd=cwd, stdin=fh, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                     errors="replace", env=env, start_new_session=True)
        except FileNotFoundError as e:
            return {"is_error": True, "result": "ollama binary not found on PATH: %s" % e,
                    "session_id": None, "returncode": None}
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(p)              # kill the whole tree, not just the `ollama` parent
            try:
                out, err = p.communicate(timeout=5)
            except Exception:
                out, err = "", ""
            return {"is_error": True, "session_id": None, "returncode": None,
                    "result": "ollama launch claude timed out after %ss (process tree killed)" % timeout}
    finally:
        fh.close()
        if p is not None and p.poll() is None:   # interrupted/errored mid-run -> don't leak the RCE tree
            _kill_tree(p)
    rc = p.returncode
    raw = out or ""
    data = None
    try:
        data = json.loads(raw)
    except ValueError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
            except ValueError:
                data = None
    if not isinstance(data, dict) or rc != 0:
        detail = (err or raw or "").strip()[:500] or "no output"
        sid = data.get("session_id") if isinstance(data, dict) else None
        return {"is_error": True, "session_id": sid, "returncode": rc,
                "result": "ollama launch failed (exit %s): %s" % (rc, detail)}
    # Carry the denial COUNT, not the list: under a non-bypass permission mode a headless run
    # denies tool calls and keeps going, so is_error stays false and an empty diff is otherwise
    # indistinguishable from "nothing needed changing". Coerced to int here so no untrusted
    # launch data reaches the report.
    denials = data.get("permission_denials")
    return {"is_error": bool(data.get("is_error")), "returncode": rc,
            "result": data.get("result") or "", "session_id": data.get("session_id"),
            "permission_denials": len(denials) if isinstance(denials, list) else 0}


PERMISSION_MODES = ("acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan")


def _permission_flags(mode):
    # bypassPermissions keeps the legacy flag: claude gates --permission-mode bypassPermissions
    # behind an availability check that --dangerously-skip-permissions bypasses outright.
    if mode == "bypassPermissions":
        return ["--dangerously-skip-permissions"]
    return ["--permission-mode", mode]


def run_as_claude_in_worktree(repo, task_file, model=None, timeout_total=TIMEOUT_TOTAL,
                              permission_mode="bypassPermissions", max_iters=None):
    """Launch a full Claude Code session inside an isolated worktree off HEAD, then capture
    its changes as a patch the caller can review and apply. The worktree is ALWAYS removed.
    The returned report is built here (not from the untrusted launch JSON -- only the result
    text, session id, and denial count are carried over); a diff is offered
    only when git capture actually produced one, independently of the launch's exit status
    (a session that exited nonzero may still have made edits worth reviewing)."""
    repo = os.path.realpath(repo)
    try:
        state = _git_state(repo)
    except GitError as e:
        return {"stop_reason": "precondition", "error": str(e)}
    try:
        wt = _make_worktree(repo, state["base_sha"])
    except Exception as e:
        return {"stop_reason": "worktree_error", "error": str(e), "base_sha": state["base_sha"]}
    # A fresh report we own end-to-end: only result/session_id are copied from the launch.
    report = {"is_error": False, "stop_reason": "done", "result": "", "session_id": None,
              "has_diff": False, "base_sha": state["base_sha"],
              "dirty_base": state["dirty"], "detached": state["detached"]}
    cleanup_error = None
    try:
        # Defense in depth: tell the launched session not to commit in the worktree. Even if
        # it does, the base_sha diff below still captures the change.
        with open(task_file, "r", encoding="utf-8") as f:
            original_task = f.read()
        fd, wrapped_task = tempfile.mkstemp(prefix="ollama-as-claude-task-", suffix=".txt")
        try:  # create+launch inside try/finally so a write failure can't leak wrapped_task
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write("You are working in a temporary git worktree. Do NOT run git commit here. "
                        "Make file edits, then finish; the user will review and apply your changes separately.\n\n")
                f.write(original_task)
            argv = (["ollama", "launch", "claude", "--model", model or DEFAULT_MODEL, "--", "-p"]
                    + _permission_flags(permission_mode)
                    + ["--output-format", "json", "--disallowed-tools", "Agent", "Task"])
            if max_iters is not None:
                argv += ["--max-turns", str(max_iters)]
            env = dict(os.environ)
            env["OLLAMA_AS_CLAUDE_ACTIVE"] = "1"
            launch = _launch_claude(argv, wt, wrapped_task, env, timeout=timeout_total)
        finally:
            try:
                os.remove(wrapped_task)
            except OSError:
                pass
        report["result"] = launch.get("result", "")
        report["session_id"] = launch.get("session_id")
        report["permission_denials"] = launch.get("permission_denials", 0)
        if launch.get("is_error"):
            report["is_error"] = True
            report["stop_reason"] = "launch_error"
        stage_meta = {"excluded_files": [], "exclusion_reason": None}
        try:
            # Prefilter before `git add -Af`: a multi-GB ignored build tree must not reach
            # git's diff allocator. _stage_limited still captures small ignored agent writes.
            stage_meta = _stage_limited(wt)
            dr = subprocess.run(["git", "-C", wt, "diff", "--binary", "--no-textconv", state["base_sha"]],
                                capture_output=True, timeout=120)
            if dr.returncode != 0:
                raise GitError("git diff failed: %s" % dr.stderr.decode("utf-8", "replace").strip())
            diff_bytes = dr.stdout
        except (GitError, subprocess.TimeoutExpired) as e:
            # Capture failed/hung and the worktree is about to be removed -> the work is gone.
            # Fail loudly so the caller never reports "no changes"; offer nothing to apply.
            report["is_error"] = True
            report["stop_reason"] = "diff_error"
            report["diff_error"] = str(e) or "git capture timed out"
            report.pop("diff_file", None)
        else:
            # Count real source files in the captured patch; exclude build artifacts only.
            _pyc_end = re.compile(rb"\.pyc(?:\s|\"|$)")
            source_files = 0
            for line in diff_bytes.splitlines():
                if line.startswith(b"diff --git "):
                    if (b"/__pycache__/" not in line
                            and b"/.pytest_cache/" not in line
                            and not _pyc_end.search(line)):
                        source_files += 1
            report["source_files"] = source_files
            if diff_bytes.strip():
                dpath = None
                try:
                    fd, dpath = tempfile.mkstemp(prefix="ollama-as-claude-diff-", suffix=".patch")
                    with os.fdopen(fd, "wb") as f:   # binary: preserve patch bytes verbatim
                        f.write(diff_bytes)
                    report["diff_file"] = dpath
                    report["has_diff"] = True
                    _claim_patch(dpath)
                    if source_files == 0:
                        report["is_error"] = True
                except OSError as e:                 # e.g. temp volume full: don't lose the work silently
                    if dpath and os.path.exists(dpath):
                        try:
                            os.remove(dpath)
                        except OSError:
                            pass
                    report["is_error"] = True
                    report["stop_reason"] = "diff_error"
                    report["diff_error"] = "could not write patch file: %s" % e
        report.update(stage_meta)
    finally:
        removed = _remove_worktree(repo, wt)
        if not removed:
            cleanup_error = "worktree not removed (may be locked): %s" % wt
    if cleanup_error:
        report["cleanup_error"] = cleanup_error
    return report


def _sweep_orphan_patches():
    """Delete abandoned diff patches from prior runs. A sibling .lease file that is
    still fresh means the patch is claimed by a live caller and must be left alone.
    If the caller died, the lease is never renewed and expires after TTL."""
    tmp = tempfile.gettempdir()
    ttl = TEMP_PATCH_TTL_SECONDS
    now = time.time()
    for name in os.listdir(tmp):
        if name.endswith(".lease"):
            continue
        if not (name.startswith("ollama-diff-") or name.startswith("ollama-as-claude-diff-")):
            continue
        path = os.path.join(tmp, name)
        lease = path + ".lease"
        try:
            if not os.path.isfile(path):
                continue
            if os.path.exists(lease):
                if now - os.path.getmtime(lease) < ttl:
                    continue
            elif now - os.path.getmtime(path) < ttl:
                # No lease yet: give a fresh patch a short race window before reclaiming.
                continue
            os.remove(path)
            if os.path.exists(lease):
                os.remove(lease)
        except OSError:
            pass


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    _sweep_orphan_patches()
    p = argparse.ArgumentParser(prog="ollama_agent")
    p.add_argument("task", nargs="?", default=None, help="task text (or via stdin)")
    where = p.add_mutually_exclusive_group(required=False)
    where.add_argument("--repo", help="git repo: run in an isolated worktree off HEAD, return a diff (P2)")
    where.add_argument("--root", help="jailed working root operated on directly (scratch/testing only)")
    p.add_argument("--gate-token", help="path to the single-use launch-gate token minted by /ollama:rescue")
    p.add_argument("--task-file", help="read the task from this file (avoids putting untrusted text on the command line)")
    p.add_argument("--model", default=None)
    p.add_argument("--as-claude", action="store_true",
                   help="as-claude mode: launch a real Claude session in a worktree (no gate token)")
    p.add_argument("--permission-mode", default="bypassPermissions", choices=PERMISSION_MODES,
                   help="permission mode of the launched Claude session (--as-claude only)")
    p.add_argument("--think", action="store_true")
    p.add_argument("--allow-shell", action="store_true",
                   help="DANGER: give the agent a run_shell tool -- full RCE, not contained by the worktree")
    p.add_argument("--readonly", action="store_true",
                   help="read-only tools only (read_file/list_dir/grep_search); no write_file/run_shell")
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--timeout", type=int, default=TIMEOUT_TOTAL)
    args = p.parse_args(argv)
    if args.as_claude:
        if not args.repo:
            p.error("--as-claude requires --repo")
        if not args.task_file:
            p.error("--as-claude requires --task-file")
    elif not args.repo and not args.root:
        p.error("one of --repo or --root is required")
    if args.permission_mode != "bypassPermissions" and not args.as_claude:
        p.error("--permission-mode is only used with --as-claude")
    if args.gate_token is not None:   # the single-use rescue token is only meaningful for a plain --repo run
        if args.as_claude:
            p.error("--gate-token is not used with --as-claude")
        if not args.repo:
            p.error("--gate-token requires --repo")
    # Size the context window (and the derived read cap) to the chosen model before running:
    # an explicit OLLAMA_CC_NUM_CTX wins, else a cloud model auto-sizes to its real window.
    global NUM_CTX, CTX_CHAR_BUDGET, READ_CAP
    NUM_CTX = _resolve_num_ctx(args.model or DEFAULT_MODEL)
    CTX_CHAR_BUDGET = int(NUM_CTX * 2.75)
    READ_CAP = _derive_read_cap(CTX_CHAR_BUDGET)
    if args.task_file:
        with open(args.task_file, "r", encoding="utf-8") as f:
            task = f.read()
    elif args.task is not None:
        task = args.task
    else:
        task = sys.stdin.read()
    if not task.strip():
        print("error: empty task", file=sys.stderr)
        return 2
    kw = dict(model=args.model, think=args.think,
              allow_write=not args.readonly, allow_shell=args.allow_shell and not args.readonly,
              max_iters=(args.max_iters if args.max_iters is not None else MAX_ITERS),
              timeout_total=args.timeout)
    if args.as_claude:
        report = run_as_claude_in_worktree(args.repo, args.task_file, model=args.model,
                                           timeout_total=args.timeout,
                                           permission_mode=args.permission_mode,
                                           max_iters=args.max_iters)
    elif args.repo:
        # the write-capable agent is fail-closed behind the launch gate's token
        if not _consume_gate_token(args.gate_token):
            print("error: refused -- no valid launch token. Start the write-capable agent via "
                  "/ollama:rescue (it discloses cloud egress / RCE and mints a fresh single-use "
                  "token). Direct --repo runs are gated.", file=sys.stderr)
            return 5
        report = run_agent_in_worktree(task.strip(), args.repo, **kw)
    else:
        # readonly (adversarial-review): idle-gap timeout, no whole-run cap. A --root write run
        # (scratch/testing) keeps the hard total cap.
        report = run_agent(task.strip(), args.root, idle_gap=args.readonly, **kw)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("stop_reason") == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
