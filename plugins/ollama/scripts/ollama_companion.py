#!/usr/bin/env python3
"""ollama-cc-plugin runtime. stdlib only. Subcommands: setup, run.

Talks to the local ollama daemon over its REST API (/api/version, /api/tags,
/api/chat). Ollama models are plain completions, so this is a stateless one-shot
HTTP call -- no app-server, session, or job control (see the plan for why).

Env overrides:
  OLLAMA_CC_MODEL  default model for `run` (default: glm-5.2:cloud)
  OLLAMA_CC_HOST   base URL of the daemon; falls back to ollama's own
                   OLLAMA_HOST (bare host:port is accepted), then localhost
"""
import argparse
import json
import os
import socket
import sys
import urllib.error
import urllib.request

DEFAULT_MODEL = os.environ.get("OLLAMA_CC_MODEL", "glm-5.2:cloud")
TIMEOUT = 600  # ponytail: default 600s; --timeout overrides for huge-diff reviews
PROMPT_WARN_CHARS = 100_000  # ~30k tokens; warn (never block) so an oversized diff can't silently overflow a ~32k-ctx model
EXIT_CONFIRM = 10  # pull/rm refuse to mutate without --yes; signals "confirm, then re-run with --yes"


def _int_env(name, default):
    """int() an env var, falling back to default on unset or unparseable value."""
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


NUM_CTX = _int_env("OLLAMA_CC_NUM_CTX", 32768)   # options.num_ctx; default assumes a >=32k model
NUM_CTX_CEILING = _int_env("OLLAMA_CC_NUM_CTX_MAX", 131072)  # cap for auto-detected num_ctx


def _resolve_host():
    # Prefer our own var; fall back to ollama's standard OLLAMA_HOST (often a
    # bare host:port with no scheme), then localhost.
    h = (os.environ.get("OLLAMA_CC_HOST") or os.environ.get("OLLAMA_HOST") or "").strip()
    h = h or "127.0.0.1:11434"
    if not h.startswith(("http://", "https://")):
        h = "http://" + h
    return h.rstrip("/")


HOST = _resolve_host()

# Failures that mean "daemon unreachable / unhealthy" -- includes a non-JSON
# body (JSONDecodeError) so a garbage response never escapes as a raw traceback.
_CONN_ERRORS = (urllib.error.URLError, socket.timeout, ConnectionError, OSError,
                json.JSONDecodeError)


def _url(path):
    return HOST + path


def _get(path, timeout=10):
    req = urllib.request.Request(_url(path), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(path, payload, timeout=TIMEOUT):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _url(path), data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post_stream(path, payload, timeout=TIMEOUT):
    """POST that yields one parsed JSON object per NDJSON line (e.g. /api/pull progress)."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _url(path), data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if line:
                yield json.loads(line.decode("utf-8"))


def _delete(path, payload, timeout=TIMEOUT):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _url(path), data=body,
        headers={"Content-Type": "application/json"}, method="DELETE",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _human_size(n):
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return "%d %s" % (n, unit) if unit == "B" else "%.1f %s" % (n, unit)
        n /= 1024


def _norm_tag(name):
    """A bare model name implies the :latest tag (ollama convention), so that
    `all-minilm` matches the installed `all-minilm:latest`."""
    return name if ":" in name else name + ":latest"


def is_cloud(name, entry=None):
    """Cloud model detection. The daemon's `remote_host` (present on /api/tags
    entries for cloud models) is authoritative; `cloud` anywhere in the name is
    the fallback -- catches e.g. `qwen3.5:397b-cloud` that lacks a `:cloud` tag."""
    if entry and entry.get("remote_host"):
        return True
    return bool(name) and "cloud" in name.lower()


def _final_text(msg):
    """Whitespace-safe content first, thinking fallback. Shared by companion and agent
    so all paths agree on what counts as an answer."""
    return (msg.get("content") or "").strip() or (msg.get("thinking") or "").strip()


def _detect_context_length(model):
    """The model's real context window from /api/show model_info, or None on any failure
    (best-effort; the caller falls back to the conservative default rather than crash)."""
    try:
        info = _post("/api/show", {"model": model}, timeout=10)
    except Exception:  # noqa: BLE001 - any transport/parse failure -> fall back, never crash the run
        return None
    mi = info.get("model_info") if isinstance(info, dict) else None
    if not isinstance(mi, dict):   # a valid but non-dict JSON body must not crash the best-effort probe
        return None
    for k, v in mi.items():
        if k.endswith(".context_length") and isinstance(v, int) and v > 0:
            return v
    return None


def _resolve_num_ctx(model):
    """Effective options.num_ctx for `model`. An explicit OLLAMA_CC_NUM_CTX always wins.
    Otherwise CLOUD models (no local GPU to OOM) auto-size to their real context clamped to
    NUM_CTX_CEILING; LOCAL models keep the conservative default so auto-detect never cranks a
    local KV cache into an OOM -- raise a local model per-run with OLLAMA_CC_NUM_CTX."""
    env = os.environ.get("OLLAMA_CC_NUM_CTX")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            return 32768   # set-but-unparseable -> the safe default; never fall through to an auto-crank
    if is_cloud(model):
        detected = _detect_context_length(model)
        if detected:
            return min(detected, max(1, NUM_CTX_CEILING))   # guard a mis-set (<=0) ceiling
    return 32768


def _daemon_down_msg():
    return ("ollama daemon not reachable at %s. Start it with `ollama serve` "
            "(or launch the Ollama app), then retry." % HOST)


def _emit(obj, as_json):
    if as_json:
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        return
    if not obj.get("daemon"):
        print(obj["message"])
        return
    print("ollama daemon OK (v%s)" % obj["version"])
    tail = "" if obj["default_model_installed"] else \
        "  (NOT installed -- run `ollama pull %s`)" % obj["default_model"]
    print("default model: %s%s" % (obj["default_model"], tail))
    if obj.get("models_error"):
        print("WARNING: %s" % obj["models_error"])
    if obj["models"]:
        print("installed models:")
        for m in obj["models"]:
            print("  - %s%s" % (m["name"], "  [cloud]" if m["cloud"] else ""))
    elif not obj.get("models_error"):
        print("no models installed. Pull one, e.g. `ollama pull glm-5.2:cloud`.")
    if obj.get("note_cloud_auth"):
        print(obj["note_cloud_auth"])


def cmd_setup(args):
    try:
        version = _get("/api/version").get("version", "unknown")
    except _CONN_ERRORS:
        _emit({"ok": False, "daemon": False, "reason": "daemon_unreachable",
               "message": _daemon_down_msg()}, args.json)
        return 3
    # Never swallow a /api/tags failure: the review egress gate reads this and
    # must fail closed (treat unknown as cloud) when models can't be listed.
    models_error = None
    tags = []
    try:
        tags = _get("/api/tags").get("models", [])
    except _CONN_ERRORS as e:
        models_error = "could not list models: %s" % (getattr(e, "reason", None) or e)
    models = [{"name": m.get("name"), "cloud": is_cloud(m.get("name"), m)} for m in tags]
    has_cloud = any(m["cloud"] for m in models)
    _emit({
        "ok": models_error is None, "daemon": True, "version": version,
        "default_model": DEFAULT_MODEL,
        "default_model_installed": any(m["name"] == DEFAULT_MODEL for m in models),
        "has_cloud_models": has_cloud,
        "models": models,
        "models_error": models_error,
        "note_cloud_auth": (
            "Cloud models route through ollama.com; if a cloud call returns an "
            "auth error, run `ollama signin`." if has_cloud else None),
    }, args.json)
    return 0


def _http_error(e, model):
    try:
        err = json.loads(e.read().decode("utf-8", "replace")).get("error", "")
    except Exception:
        err = str(e)
    low = str(err).lower()
    if e.code == 404 or "not found" in low:
        print("error: model '%s' not found. Pull it: `ollama pull %s`." % (model, model),
              file=sys.stderr)
        return 4
    if e.code in (401, 403) or "unauthor" in low or "sign in" in low or "signin" in low:
        print("error: cloud model '%s' requires sign-in. Run `ollama signin`." % model,
              file=sys.stderr)
        return 5
    print("error: ollama returned HTTP %s: %s" % (e.code, err or "unknown"), file=sys.stderr)
    return 1


def _run_stream(payload, model, args):
    """Stream /api/chat, writing content deltas as they arrive. With streaming, args.timeout is
    the socket read timeout = the max idle gap between chunks, so a long-but-progressing review
    is not killed by a total cap while a truly stalled stream still aborts. A stream that ends
    with no content at all is retried once; the buffered reasoning carries across both attempts."""
    think_buf = []
    done_seen = False
    for attempt in (0, 1):
        got = False
        done_seen = False
        try:
            for evt in _post_stream("/api/chat", payload, timeout=args.timeout):
                if evt.get("error"):
                    print("\nerror: ollama: %s" % evt["error"], file=sys.stderr)
                    return 1
                msg = evt.get("message") or {}
                chunk = msg.get("content") or ""
                if chunk:
                    if not got and args.show_thinking and think_buf:
                        # Reasoning streamed before the answer began: flush it now so
                        # --show-thinking still shows a thinking model's pre-answer reasoning.
                        sys.stderr.write("".join(think_buf))
                        sys.stderr.flush()
                    got = True
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                thinking = msg.get("thinking") or ""
                if thinking:
                    think_buf.append(thinking)
                    if args.show_thinking and got:
                        sys.stderr.write(thinking)
                        sys.stderr.flush()
                if evt.get("done"):
                    done_seen = True
                    break
        except urllib.error.HTTPError as e:
            return _http_error(e, model)
        except json.JSONDecodeError:
            print("\nerror: ollama returned a non-JSON stream line.", file=sys.stderr)
            return 1
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, socket.timeout) or "timed out" in str(reason).lower():
                print("\nerror: stream from '%s' idle >%ss (no new tokens)." % (model, args.timeout),
                      file=sys.stderr)
                return 6
            print("\nerror: " + _daemon_down_msg(), file=sys.stderr)
            return 3
        if got:
            sys.stdout.write("\n")
            sys.stdout.flush()
            # done_reason is unreliable (cloud models report "stop" even on a truncated reply),
            # but a missing terminal done:true event still flags a dropped stream.
            if not done_seen:
                print("warning: stream ended without a completion marker; the reply may be truncated.",
                      file=sys.stderr)
                return 1
            return 0
        # No content this attempt. A retry cannot corrupt output (nothing was flushed);
        # a second empty stream falls through to the reasoning-fallback below.
        if attempt == 0:
            continue
    if think_buf:
        # Promote the buffered reasoning as the answer. It was never teed to stderr (content
        # never arrived), so --show-thinking does not duplicate it.
        sys.stdout.write("[model produced only reasoning, no final answer]\n")
        sys.stdout.write("".join(think_buf))
        sys.stdout.write("\n")
        sys.stdout.flush()
        # A reasoning-only reply that also never saw done:true was truncated, not complete.
        if not done_seen:
            print("warning: stream ended without a completion marker; the reply may be truncated.",
                  file=sys.stderr)
            return 1
        return 0
    print("error: ollama returned an empty response (model '%s')." % model, file=sys.stderr)
    return 1


def _run_once(payload, model, args, _retried=False):
    """Single non-stream /api/chat request. Retry once on an empty-content response; fall back
    to the model's reasoning only after the retry also yields no content, so a transient empty
    gets a real answer before we settle for reasoning."""
    try:
        data = _post("/api/chat", payload, timeout=args.timeout)
    except urllib.error.HTTPError as e:
        return _http_error(e, model)
    except json.JSONDecodeError:
        print("error: ollama returned a non-JSON response from /api/chat.", file=sys.stderr)
        return 1
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
        reason = getattr(e, "reason", e)
        if isinstance(reason, socket.timeout) or "timed out" in str(reason).lower():
            print("error: request to '%s' timed out after %ss." % (model, args.timeout),
                  file=sys.stderr)
            return 6
        print("error: " + _daemon_down_msg(), file=sys.stderr)
        return 3
    # ollama can answer 200 with an {"error": ...} body (e.g. load failure).
    if data.get("error"):
        print("error: ollama: %s" % data["error"], file=sys.stderr)
        return 1
    message = data.get("message") or {}
    content = message.get("content") or ""
    # --show-thinking is a debug tee in the normal (content-arrived) case only, so a
    # reasoning-to-answer fallback is not echoed twice.
    if content.strip() and args.show_thinking and message.get("thinking"):
        sys.stderr.write(message["thinking"] + "\n")
    if not content.strip():
        # Retry on empty CONTENT even when reasoning is present, so A2 gets a real answer
        # a chance before we settle for the reasoning fallback.
        if not _retried:
            return _run_once(payload, model, args, _retried=True)
        thinking = (message.get("thinking") or "").strip()
        if thinking:
            print("[model produced only reasoning, no final answer]")
            print(thinking)
            return 0
        print("error: ollama returned an empty response (model '%s')." % model, file=sys.stderr)
        return 1
    print(content)
    return 0


def cmd_run(args):
    model = args.model or DEFAULT_MODEL
    raw = args.prompt if args.prompt is not None else sys.stdin.read()
    if not raw.strip():
        print("error: empty prompt (pass as an argument or via stdin).", file=sys.stderr)
        return 2
    if len(raw) > PROMPT_WARN_CHARS:
        print("warning: prompt is %d chars (~%dk tokens); may exceed the model's context and be "
              "silently truncated. Narrow the scope (e.g. --base, fewer files)."
              % (len(raw), len(raw) // 3000), file=sys.stderr)
    payload = {
        "model": model,
        # unstripped: preserve significant whitespace in a piped diff
        "messages": [{"role": "user", "content": raw}],
        "stream": bool(args.stream),
        "think": bool(args.think),
    }
    # Only cloud models get a num_ctx option: auto-cranking a local model's KV cache
    # can push a local GPU into OOM. Local models use their own defaults.
    if is_cloud(model):
        payload["options"] = {"num_ctx": _resolve_num_ctx(model)}
    if args.stream:
        return _run_stream(payload, model, args)
    return _run_once(payload, model, args)


def _model_size(name):
    """Local on-disk size (bytes) of an installed model, or None if absent/unlistable."""
    try:
        for m in _get("/api/tags").get("models", []):
            if _norm_tag(m.get("name", "")) == _norm_tag(name):
                return m.get("size")
    except _CONN_ERRORS:
        pass
    return None


def cmd_ps(args):
    try:
        models = _get("/api/ps").get("models", [])
    except urllib.error.HTTPError as e:
        return _http_error(e, "")
    except _CONN_ERRORS:
        print(_daemon_down_msg(), file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(models, ensure_ascii=False, indent=2))
        return 0
    if not models:
        print("no models are currently loaded in memory.")
        return 0
    print("running models:")
    for m in models:
        print("  - %s  size=%s vram=%s expires=%s" % (
            m.get("name", "?"), _human_size(m.get("size")),
            _human_size(m.get("size_vram")), m.get("expires_at", "?")))
    return 0


def cmd_show(args):
    try:
        data = _post("/api/show", {"model": args.model})
    except urllib.error.HTTPError as e:
        return _http_error(e, args.model)
    except _CONN_ERRORS:
        print(_daemon_down_msg(), file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    det = data.get("details") or {}
    size = _model_size(args.model)
    print("model: %s" % args.model)
    print("  family:       %s" % det.get("family", "?"))
    print("  parameters:   %s" % det.get("parameter_size", "?"))
    print("  quantization: %s" % det.get("quantization_level", "?"))
    if size is not None:
        print("  size:         %s" % _human_size(size))
    params = (data.get("parameters") or "").strip()
    if params:
        print("  default parameters:")
        for line in params.splitlines():
            print("    " + line)
    return 0


def cmd_pull(args):
    model = args.model
    if not args.yes:
        size = _model_size(model)
        if size is not None:
            print("'%s' is already installed (%s). Re-run with --yes to re-pull/update it."
                  % (model, _human_size(size)))
        else:
            print("About to download '%s' from the registry -- this can be several GB and "
                  "take a while. Re-run with --yes to proceed." % model)
        return EXIT_CONFIRM
    streamed = False
    ok = False
    try:
        for evt in _post_stream("/api/pull", {"model": model, "stream": True}, timeout=args.timeout):
            if evt.get("error"):
                if streamed:
                    sys.stdout.write("\n")
                print("error: %s" % evt["error"], file=sys.stderr)
                return 1
            total, done = evt.get("total"), evt.get("completed")
            status = evt.get("status", "")
            if status == "success":
                ok = True
            if total and done:
                sys.stdout.write("\r%s: %d%% (%s / %s)   " % (
                    status, done * 100 // total, _human_size(done), _human_size(total)))
                sys.stdout.flush()
                streamed = True
            else:
                if streamed:
                    sys.stdout.write("\n")
                    streamed = False
                print(status)
    except urllib.error.HTTPError as e:
        return _http_error(e, model)
    except _CONN_ERRORS:
        print(_daemon_down_msg(), file=sys.stderr)
        return 3
    if streamed:
        sys.stdout.write("\n")
    # A stream that ends without a terminal "success" was truncated/dropped:
    # reporting "pulled" then would be a false success on a partial download.
    if not ok:
        print("error: pull of '%s' ended without a success status (stream truncated?)."
              % model, file=sys.stderr)
        return 1
    print("pulled '%s'." % model)
    return 0


def cmd_rm(args):
    model = args.model
    try:
        tags = _get("/api/tags").get("models", [])
    except _CONN_ERRORS:
        print(_daemon_down_msg(), file=sys.stderr)
        return 3
    match = next((m for m in tags if _norm_tag(m.get("name", "")) == _norm_tag(model)), None)
    if match is None:
        print("'%s' is not installed -- nothing to delete." % model, file=sys.stderr)
        return 4
    full, size = match.get("name", model), match.get("size")
    if not args.yes:
        print("About to DELETE '%s' (%s). Re-run with --yes to confirm." % (full, _human_size(size)))
        return EXIT_CONFIRM
    try:
        _delete("/api/delete", {"model": full})
    except urllib.error.HTTPError as e:
        return _http_error(e, full)
    except _CONN_ERRORS:
        print(_daemon_down_msg(), file=sys.stderr)
        return 3
    print("deleted '%s' (%s freed)." % (full, _human_size(size)))
    return 0


def cmd_list(args):
    try:
        tags = _get("/api/tags").get("models", [])
    except _CONN_ERRORS:
        print(_daemon_down_msg(), file=sys.stderr)
        return 3
    models = [{"name": m.get("name"), "size": m.get("size"), "cloud": is_cloud(m.get("name"), m)}
              for m in tags]
    if args.json:
        print(json.dumps(models, ensure_ascii=False, indent=2))
        return 0
    if not models:
        print("no models installed. Pull one, e.g. `ollama pull glm-5.2:cloud`.")
        return 0
    print("available models:")
    for m in models:
        print("  - %s%s  %s" % (
            m["name"], "  [cloud]" if m["cloud"] else "", _human_size(m["size"])))
    return 0


def main(argv=None):
    # Windows consoles default to cp949 here; force utf-8 so non-ASCII model
    # output does not crash on print (verified failure mode).
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass
    p = argparse.ArgumentParser(prog="ollama_companion")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("setup", help="check daemon + list models")
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=cmd_setup)

    pr = sub.add_parser("run", help="one-shot chat completion (prompt from arg or stdin)")
    pr.add_argument("prompt", nargs="?", default=None)
    pr.add_argument("--model", default=None)
    pr.add_argument("--think", action="store_true", help="enable model reasoning")
    pr.add_argument("--show-thinking", action="store_true", help="print reasoning to stderr")
    pr.add_argument("--timeout", type=int, default=TIMEOUT)
    pr.add_argument("--stream", action="store_true",
                    help="stream the reply as generated; with --stream, --timeout is the max idle gap between chunks (a hang detector), not a total cap")
    pr.set_defaults(func=cmd_run)

    pls = sub.add_parser("list", help="list installed/available models")
    pls.add_argument("--json", action="store_true")
    pls.set_defaults(func=cmd_list)

    pps = sub.add_parser("ps", help="list running (in-memory) models")
    pps.add_argument("--json", action="store_true")
    pps.set_defaults(func=cmd_ps)

    psh = sub.add_parser("show", help="show model details (family, params, quantization, size)")
    psh.add_argument("model")
    psh.add_argument("--json", action="store_true")
    psh.set_defaults(func=cmd_show)

    ppl = sub.add_parser("pull", help="download/install a model (requires --yes to proceed)")
    ppl.add_argument("model")
    ppl.add_argument("--yes", action="store_true", help="confirm the download")
    ppl.add_argument("--timeout", type=int, default=TIMEOUT)
    ppl.set_defaults(func=cmd_pull)

    prm = sub.add_parser("rm", help="delete an installed model (requires --yes to proceed)")
    prm.add_argument("model")
    prm.add_argument("--yes", action="store_true", help="confirm the deletion")
    prm.set_defaults(func=cmd_rm)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
