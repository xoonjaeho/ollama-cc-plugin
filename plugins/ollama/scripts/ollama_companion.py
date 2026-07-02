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
TIMEOUT = 120  # ponytail: fixed 120s; --timeout overrides for huge-diff reviews


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


def is_cloud(name, entry=None):
    """Cloud model detection. The daemon's `remote_host` (present on /api/tags
    entries for cloud models) is authoritative; `cloud` anywhere in the name is
    the fallback -- catches e.g. `qwen3.5:397b-cloud` that lacks a `:cloud` tag."""
    if entry and entry.get("remote_host"):
        return True
    return bool(name) and "cloud" in name.lower()


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


def cmd_run(args):
    model = args.model or DEFAULT_MODEL
    raw = args.prompt if args.prompt is not None else sys.stdin.read()
    if not raw.strip():
        print("error: empty prompt (pass as an argument or via stdin).", file=sys.stderr)
        return 2
    payload = {
        "model": model,
        # unstripped: preserve significant whitespace in a piped diff
        "messages": [{"role": "user", "content": raw}],
        "stream": False,
        "think": bool(args.think),
    }
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
    if args.show_thinking and message.get("thinking"):
        sys.stderr.write(message["thinking"] + "\n")
    content = message.get("content", "")
    if not content.strip():
        print("error: ollama returned an empty response (model '%s')." % model, file=sys.stderr)
        return 1
    print(content)
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
    pr.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
