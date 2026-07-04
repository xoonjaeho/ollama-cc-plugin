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

# reuse the read-only companion's HTTP + config (same scripts/ dir)
from ollama_companion import _post, DEFAULT_MODEL  # noqa: E402


def _int_env(name, default):
    """int() an env var, falling back to default on unset or unparseable value."""
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


READ_CAP = 16 * 1024          # bytes returned to the model per read/tool result
WRITE_CAP = 1024 * 1024       # 1 MiB per write_file: ample for source, caps a runaway model from filling the worktree disk before review
EGRESS_BUDGET = 8 * 1024 * 1024
MAX_ITERS = 15
TIMEOUT_TOTAL = 300           # seconds, whole run
TOOL_CALL_CAP = 40
MALFORMED_CAP = 3
LOOP_REPEAT_CAP = 3           # identical non-write call N times -> loop
NUM_CTX = _int_env("OLLAMA_CC_NUM_CTX", 32768)   # options.num_ctx; default assumes a >=32k model. Lower via OLLAMA_CC_NUM_CTX for a small local model.
# ponytail: char proxy for the token budget at ~2.75 chars/token, kept under NUM_CTX so the
# server never front-truncates our pinned system+task (client _truncate_history does it first).
CTX_CHAR_BUDGET = int(NUM_CTX * 2.75)


class JailError(Exception):
    pass


class GitError(Exception):
    pass


# ---------------------------------------------------------------- path jail
def resolve_in_jail(root_real, path):
    """Absolute real path if `path` resolves strictly inside root_real, else
    JailError. See module docstring for the containment boundary."""
    if path is None or not str(path).strip():
        raise JailError("empty path")
    path = str(path)
    # No legit in-jail relative path contains ':' -- bans NTFS alternate data
    # streams (a.txt:stream, .git:stream) and drive-relative (C:foo).
    if ":" in path:
        raise JailError("':' not allowed in path: %s" % path)
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
    if len(raw) > READ_CAP:
        text += ("\n[truncated at %d bytes; call read_file again with offset=%d for more]"
                 % (READ_CAP, offset + READ_CAP))
    return text


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
    for dirpath, dirnames, filenames in os.walk(base):
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
    # new process group/session so the whole tree can be killed on timeout
    if os.name == "nt":
        p = subprocess.Popen(str(cmd), shell=True, cwd=root_real, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        p = subprocess.Popen(str(cmd), shell=True, cwd=root_real, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
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
    content = msg.get("content", "") if isinstance(msg, dict) else ""
    return content or "", projected


# ---------------------------------------------------------------- the loop
def _system_prompt(root):
    return (
        "You are a coding agent working inside a working root you cannot escape: "
        "every path you pass to a tool is relative to that root and is confined to it. "
        "Use read_file to inspect and write_file to make changes. "
        "Do only what the task asks. When the task is complete, reply with a short "
        "final summary and DO NOT call any more tools.\n"
        "Working root: %s" % root)


def _append_tool(messages, tool_call_id, name, content):
    m = {"role": "tool", "tool_name": name, "content": content}
    if tool_call_id:
        m["tool_call_id"] = tool_call_id
    messages.append(m)


def run_agent(task, root, model=None, think=False, allow_write=True, allow_shell=False,
              max_iters=MAX_ITERS, timeout_total=TIMEOUT_TOTAL):
    model = model or DEFAULT_MODEL
    tools, dispatch = _toolset(allow_write=allow_write, allow_shell=allow_shell)
    root_real = os.path.realpath(root)
    if not os.path.isdir(root_real):
        return {"stop_reason": "bad_root", "error": "root is not a directory: %s" % root}
    messages = [{"role": "system", "content": _system_prompt(root_real)},
                {"role": "user", "content": task}]
    pinned = 2
    iters = tool_calls_total = malformed = egress_bytes = 0
    seen = {}                 # loop-key -> count
    written = set()           # canonical paths written -> exempt a following re-read from loop-kill
    actions = []
    start = time.monotonic()

    def stop(reason, final=""):
        nonlocal egress_bytes
        if not final and reason in _SYNTH_STOPS:
            # trim to budget first: max_iters bails before _truncate_history, so the
            # in-context messages can be one turn over budget; without this ollama
            # front-truncates the pinned system+task (the context salvage needs).
            _truncate_history(messages, pinned)
            final, spent = _force_synthesis(
                messages, model, think, NUM_CTX,
                timeout_total - (time.monotonic() - start),
                EGRESS_BUDGET - egress_bytes)
            egress_bytes += spent
        return {"stop_reason": reason, "iterations": iters, "final": final,
                "actions": actions, "egress_bytes": egress_bytes,
                "tool_calls": tool_calls_total, "model": model}

    while True:
        iters += 1
        if iters > max_iters:
            return stop("max_iters")
        elapsed = time.monotonic() - start
        if elapsed > timeout_total:
            return stop("timeout")
        _truncate_history(messages, pinned)
        payload = _chat_payload(model, messages, tools, think, NUM_CTX)
        projected = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        if egress_bytes + projected > EGRESS_BUDGET:   # check BEFORE it leaves the process
            return stop("egress_budget")
        egress_bytes += projected
        try:
            # per-request timeout bounded by the remaining total, so a late call
            # cannot block for another full window.
            data = _post("/api/chat", payload, timeout=max(1, int(timeout_total - elapsed)))
        except Exception as e:  # noqa: BLE001 - surface any transport/parse failure as a stop
            return stop("api_error:%s" % type(e).__name__)
        if not isinstance(data, dict):
            return stop("api_error:non-dict-response")
        if data.get("error"):   # ollama can answer 200 with an {"error": ...} body
            return stop("api_error:%s" % str(data["error"])[:120])
        msg = data.get("message") or {}
        messages.append(msg)                       # assistant turn (may carry tool_calls)
        tcs = msg.get("tool_calls") or []
        if not tcs:
            return stop("done", final=msg.get("content", ""))
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
                elif name == "read_file":
                    written.discard(canon)
            except JailError as e:
                result, ok = "error: %s" % e, False
            except Exception as e:  # noqa: BLE001
                result, ok = "error: %s: %s" % (type(e).__name__, e), False
            actions.append({"tool": name, "args": args, "ok": ok})
            _append_tool(messages, tool_call_id, name, result)


# ---------------------------------------------------------------- git worktree (P2)
def _git(repo, *args, check=False):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise GitError("git %s failed: %s" % (args[0], (r.stderr or r.stdout).strip()))
    return r


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
    return {"base_sha": head.stdout.strip(),
            "dirty": bool(_git(repo, "status", "--porcelain").stdout.strip()),
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
        try:
            # -Af: capture ALL agent writes, incl. paths under .gitignore, else the
            # agent's work silently vanishes from the diff. check=True on both so a
            # git failure surfaces instead of masquerading as "no changes".
            _git(wt, "add", "-Af", check=True)
            diff_text = _git(wt, "diff", "--cached", "--binary", check=True).stdout
            report["diff"] = diff_text
            if diff_text.strip():
                # Write the patch to a file OUTSIDE the worktree so the command applies
                # it by path (`git apply <file>`) and never reconstructs untrusted patch
                # text through the shell. Survives the worktree cleanup below.
                fd, dpath = tempfile.mkstemp(prefix="ollama-diff-", suffix=".patch")
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                    f.write(diff_text)
                report["diff_file"] = dpath
        except GitError as e:
            report["diff_error"] = str(e)
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
def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass
    p = argparse.ArgumentParser(prog="ollama_agent")
    p.add_argument("task", nargs="?", default=None, help="task text (or via stdin)")
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--repo", help="git repo: run in an isolated worktree off HEAD, return a diff (P2)")
    where.add_argument("--root", help="jailed working root operated on directly (scratch/testing only)")
    p.add_argument("--gate-token", help="path to the single-use launch-gate token minted by /ollama:rescue")
    p.add_argument("--task-file", help="read the task from this file (avoids putting untrusted text on the command line)")
    p.add_argument("--model", default=None)
    p.add_argument("--think", action="store_true")
    p.add_argument("--allow-shell", action="store_true",
                   help="DANGER: give the agent a run_shell tool -- full RCE, not contained by the worktree")
    p.add_argument("--readonly", action="store_true",
                   help="read-only tools only (read_file/list_dir/grep_search); no write_file/run_shell")
    p.add_argument("--max-iters", type=int, default=MAX_ITERS)
    p.add_argument("--timeout", type=int, default=TIMEOUT_TOTAL)
    args = p.parse_args(argv)
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
              max_iters=args.max_iters, timeout_total=args.timeout)
    if args.repo:
        # the write-capable agent is fail-closed behind the launch gate's token
        if not _consume_gate_token(args.gate_token):
            print("error: refused -- no valid launch token. Start the write-capable agent via "
                  "/ollama:rescue (it discloses cloud egress / RCE and mints a fresh single-use "
                  "token). Direct --repo runs are gated.", file=sys.stderr)
            return 5
        report = run_agent_in_worktree(task.strip(), args.repo, **kw)
    else:
        report = run_agent(task.strip(), args.root, **kw)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("stop_reason") == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
