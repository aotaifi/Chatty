#!/usr/bin/env python3
"""Wake an existing ChatGPT conversation by reusing an already-open Chrome tab.

Key invariants:
- Never use cua.createBrowserTab(): it can exceed the NodeREPL per-call deadline.
- Never reuse AX numeric element IDs for input: they race with React re-renders.
- Bind an existing exact-thread tab and use Playwright DOM locators.
- Keep every CUA JS call short; long waiting happens in Python.
- A retry is idempotent: if the delivery marker is already visible, succeed without
  sending a duplicate.

No OpenAI API key is used.
"""
from __future__ import annotations

import json
import os
import plistlib
import re
import select
import subprocess
import sys
import time
import uuid
from pathlib import Path

APP = Path("/Applications/ChatGPT.app")
RES = APP / "Contents/Resources"
BASE = RES / "cua_node"
NODE = BASE / "bin/node"
SERVER = BASE / "lib/node_modules/@oai/cua-repl/bin/cua-repl.mjs"

CALL_TIMEOUT = float(os.environ.get("CHATTY_CUA_CALL_TIMEOUT", "25"))
IDLE_TIMEOUT = float(os.environ.get("CHATTY_WAKE_IDLE_TIMEOUT", "900"))
POLL_SECONDS = float(os.environ.get("CHATTY_WAKE_POLL_SECONDS", "2"))
DRY_RUN = os.environ.get("CHATTY_WAKE_DRY_RUN") == "1"

if len(sys.argv) < 3:
    raise SystemExit("usage: chatty-wake-existing.py THREAD_ID PROMPT...")
thread_id = sys.argv[1].strip()
prompt = " ".join(sys.argv[2:]).strip()
if not thread_id or not prompt:
    raise SystemExit("thread id and prompt must be non-empty")
if not SERVER.exists():
    raise SystemExit(f"ChatGPT CUA runtime not found: {SERVER}")

m = re.search(r"(?m)^job_id:\s*([^\s]+)", prompt)
delivery_marker = f"job_id: {m.group(1)}" if m else prompt[:120]
target_url = f"https://chatgpt.com/c/{thread_id}"

try:
    with (APP / "Contents/Info.plist").open("rb") as f:
        app_version = plistlib.load(f).get("CFBundleShortVersionString", "")
except Exception:
    app_version = ""

env = os.environ.copy()
env.update({
    "NODE_REPL_NATIVE_PIPE_CONNECT_TIMEOUT_MS": "1000",
    "NODE_REPL_NODE_MODULE_DIRS": str(BASE / "lib/node_modules"),
    "NODE_REPL_NODE_PATH": str(NODE),
    "NODE_REPL_TRUSTED_CODE_PATHS": os.path.expanduser("~/.codex") + ":" + str(BASE / "lib/node_modules"),
    "CODEX_HOME": os.path.expanduser("~/.codex"),
    "BROWSER_USE_AVAILABLE_BACKENDS": "chrome,iab",
    "BROWSER_USE_TINYSKY_ENABLED": "1",
    "NODE_REPL_INSTRUCTIONS_USE_CASE_BROWSER": "Control the in-app browser in conjunction with the Browser Plugin.",
    "NODE_REPL_INSTRUCTIONS_USE_CASE_CHROME": "Control the Chrome browser in conjunction with the Chrome Plugin.",
    "NODE_REPL_INSTRUCTIONS_USE_CASE_COMPUTER_USE": "Control desktop apps on macOS through Computer Use.",
    "BROWSER_USE_CODEX_APP_BUILD_FLAVOR": "prod",
    "BROWSER_USE_CODEX_APP_VERSION": app_version,
    "NODE_REPL_TRUSTED_SERVICES": '{"browser":"@oai/browser-desktop/service","sky":"@oai/sky/service"}',
    "SKY_CUA_SERVICE_PATH": os.path.expanduser("~/.codex/computer-use/Codex Computer Use.app"),
    "CODEX_CLI_PATH": str(RES / "codex"),
    "CUA_REPL_NODE_REPL_PATH": str(BASE / "bin/node_repl"),
    "CUA_REPL_ENABLED_SURFACES": "browser,computer",
})

proc = subprocess.Popen(
    [str(NODE), str(SERVER)],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
    env=env,
    start_new_session=True,
)
rid = 0


def raw_send(obj: dict) -> None:
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(obj, separators=(",", ":")) + "\n")
    proc.stdin.flush()


def send(method: str, params=None) -> int:
    global rid
    rid += 1
    msg = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        msg["params"] = params
    raw_send(msg)
    return rid


def notify(method: str, params=None) -> None:
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    raw_send(msg)


def wait_id(wanted: int, timeout: float):
    deadline = time.monotonic() + timeout
    assert proc.stdout is not None and proc.stderr is not None
    while time.monotonic() < deadline:
        ready, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.5)
        for stream in ready:
            line = stream.readline()
            if not line:
                continue
            if stream is proc.stderr:
                print("CUA:", line.rstrip(), file=sys.stderr, flush=True)
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("method") == "elicitation/create" and "id" in obj:
                params = obj.get("params") or {}
                meta = params.get("_meta") or {}
                allowed = (
                    meta.get("tool_name") == "access_browser_origin"
                    and meta.get("origin") == "https://chatgpt.com"
                )
                raw_send({
                    "jsonrpc": "2.0",
                    "id": obj["id"],
                    "result": {"action": "accept", "content": {}} if allowed else {"action": "cancel"},
                })
                continue
            if obj.get("id") == wanted:
                return obj
    raise TimeoutError(f"timeout waiting for MCP id {wanted}")


def tool_text(result: dict) -> str:
    if "error" in result:
        raise RuntimeError("MCP error: " + json.dumps(result["error"]))
    tr = result.get("result") or {}
    if tr.get("isError"):
        raise RuntimeError("CUA tool error: " + json.dumps(tr))
    return "\n".join(
        c.get("text", "") for c in tr.get("content", []) if c.get("type") == "text"
    )


def call_js(code: str, title: str, timeout: float = CALL_TIMEOUT) -> str:
    meta = {
        "x-codex-turn-metadata": json.dumps({
            "session_id": thread_id,
            "turn_id": "chatty-wake-" + uuid.uuid4().hex,
            "thread_id": thread_id,
            "model": "gpt-5.6-sol",
        })
    }
    call_id = send("tools/call", {
        "name": "js",
        "arguments": {"code": code, "title": title},
        "_meta": meta,
    })
    return tool_text(wait_id(call_id, timeout))


def marker_json(text: str, marker: str) -> dict:
    hits = re.findall(re.escape(marker) + r"\s+(\{[^\n]+\})", text)
    if not hits:
        raise RuntimeError(f"{marker} missing from CUA result: {text[-1500:]}")
    return json.loads(hits[-1])


def find_exact_tab() -> str | None:
    code = (
        "let ts=await cua.listTabs({browser:\"chrome\",emit:false});"
        "nodeRepl.write(\"CHATTY_TABS \"+JSON.stringify(ts));"
    )
    state = marker_json(call_js(code, "List existing ChatGPT tabs"), "CHATTY_TABS")
    # marker_json expects dict; list is handled below via direct extraction.
    return state


def discover_browser_id() -> str | None:
    out = call_js(
        'let bs=await cua.listBrowsers({emit:false});'
        'nodeRepl.write("CHATTY_BROWSERS "+JSON.stringify({browsers:bs}));',
        "Discover Chrome browser provider",
    )
    browsers = marker_json(out, "CHATTY_BROWSERS").get("browsers", [])
    for b in browsers:
        if b.get("family") == "chrome" or b.get("name") == "Chrome":
            bid = b.get("id")
            if bid is not None:
                return str(bid)
    return None


def list_tabs(browser_id: str) -> list[dict]:
    out = call_js(
        'let ts=await cua.listTabs({browser:' + json.dumps(browser_id) + ',emit:false});'
        'nodeRepl.write("CHATTY_TABS "+JSON.stringify({tabs:ts}));',
        "List existing ChatGPT tabs",
    )
    return marker_json(out, "CHATTY_TABS").get("tabs", [])


def choose_tab(tabs: list[dict]) -> dict | None:
    wanted = target_url.rstrip("/")
    exact = []
    for t in tabs:
        url = (t.get("url") or "").split("#", 1)[0].split("?", 1)[0].rstrip("/")
        if url == wanted:
            exact.append(t)
    return exact[0] if exact else None


def bind_tab(tab_id: str, browser_id: str) -> None:
    out = call_js(
        "globalThis.__chattyWakeTab=await cua.getTab("
        + json.dumps(str(tab_id))
        + ',{browser:"chrome"});'
        + 'nodeRepl.write("CHATTY_BOUND "+JSON.stringify({id:globalThis.__chattyWakeTab.id}));',
        "Bind existing target ChatGPT tab",
    )
    marker_json(out, "CHATTY_BOUND")


def readiness() -> dict:
    code = (
        'let p=globalThis.__chattyWakeTab.playwright.locator("#prompt-textarea");'
        'let n=await p.count();'
        'let visible=n>0?await p.first().isVisible():false;'
        'let enabled=n>0?await p.first().isEnabled():false;'
        "let seen=await globalThis.__chattyWakeTab.playwright.getByText("
        + json.dumps(delivery_marker)
        + ',{exact:false}).count();'
        'nodeRepl.write("CHATTY_READY "+JSON.stringify({n:n,visible:visible,enabled:enabled,seen:seen}));'
    )
    return marker_json(call_js(code, "Check ChatGPT composer readiness"), "CHATTY_READY")


def submit() -> None:
    code = (
        'let p=globalThis.__chattyWakeTab.playwright.locator("#prompt-textarea").first();'
        "await p.fill(" + json.dumps(prompt) + ",{timeoutMs:5000});"
        'await p.press("Enter",{timeoutMs:5000});'
        'nodeRepl.write("CHATTY_SENT "+JSON.stringify({ok:true}));'
    )
    state = marker_json(call_js(code, "Submit same-thread wake prompt"), "CHATTY_SENT")
    if not state.get("ok"):
        raise RuntimeError("submit did not return ok")


def verify_visible(deadline_s: float = 30) -> bool:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        code = (
            "let seen=await globalThis.__chattyWakeTab.playwright.getByText("
            + json.dumps(delivery_marker)
            + ',{exact:false}).count();'
            'nodeRepl.write("CHATTY_VERIFY "+JSON.stringify({seen:seen}));'
        )
        state = marker_json(call_js(code, "Verify submitted wake prompt"), "CHATTY_VERIFY")
        if state.get("seen", 0) > 0:
            return True
        time.sleep(1)
    return False


try:
    init_id = send("initialize", {
        "protocolVersion": "2025-06-18",
        "capabilities": {"elicitation": {}},
        "clientInfo": {"name": "chatty-existing-tab-waker", "version": "2.0"},
    })
    init = wait_id(init_id, 20)
    if "error" in init:
        raise RuntimeError("MCP initialize failed: " + json.dumps(init["error"]))
    notify("notifications/initialized", {})

    deadline = time.monotonic() + IDLE_TIMEOUT
    opened = False
    bound_id = None
    browser_id = None
    while time.monotonic() < deadline:
        try:
            if browser_id is None:
                browser_id = discover_browser_id()
                if browser_id is None:
                    if not opened:
                        subprocess.run(
                            ["/usr/bin/open", "-g", "-a", "Google Chrome", target_url],
                            check=False,
                            timeout=10,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        opened = True
                    print("WAIT Chrome provider unavailable; recovery tab requested", file=sys.stderr, flush=True)
                    time.sleep(max(3.0, POLL_SECONDS))
                    continue

            tabs = list_tabs(browser_id)
            chosen = choose_tab(tabs)
            if chosen is None:
                if not opened:
                    # Opening through macOS is cheap and does not invoke the flaky
                    # CUA createBrowserTab path. The next iteration discovers it.
                    subprocess.run(
                        ["/usr/bin/open", "-g", "-a", "Google Chrome", target_url],
                        check=False,
                        timeout=10,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    opened = True
                print("WAIT no exact target tab", file=sys.stderr, flush=True)
                time.sleep(POLL_SECONDS)
                continue

            print("CHOSEN", json.dumps(chosen), file=sys.stderr, flush=True)
            tab_id = str(chosen.get("providerTabId") or chosen["id"])
            if bound_id != tab_id:
                bind_tab(tab_id, browser_id)
                bound_id = tab_id
                print("BOUND", tab_id, target_url, file=sys.stderr, flush=True)

            state = readiness()
            print("STATE", json.dumps(state), file=sys.stderr, flush=True)
            if state.get("seen", 0) > 0:
                print("WAKE_ALREADY_PRESENT", thread_id, delivery_marker, flush=True)
                raise SystemExit(0)
            if state.get("n", 0) > 0 and state.get("visible") and state.get("enabled"):
                if DRY_RUN:
                    print("WAKE_DRY_RUN_READY", thread_id, browser_id, tab_id, flush=True)
                    raise SystemExit(0)
                submit()
                if verify_visible():
                    print("WAKE_SUBMITTED", thread_id, delivery_marker, flush=True)
                    raise SystemExit(0)
                raise RuntimeError("prompt submit returned but delivery marker was not visible")
        except (RuntimeError, TimeoutError) as exc:
            # Browser state can be transient while the active ChatGPT turn rerenders.
            # Drop the binding and retry. A hard MCP/kernel failure exits to durable
            # recovery rather than blindly duplicating a message.
            print("WAKE_RETRY_STATE", repr(exc), file=sys.stderr, flush=True)
            if "Browser is not available" in str(exc) or "provider unavailable" in str(exc):
                browser_id = None
            bound_id = None
        time.sleep(POLL_SECONDS)

    raise TimeoutError(f"target thread did not become idle within {IDLE_TIMEOUT}s")
finally:
    try:
        os.killpg(proc.pid, 15)
    except Exception:
        pass
    try:
        proc.wait(timeout=3)
    except Exception:
        try:
            os.killpg(proc.pid, 9)
        except Exception:
            pass
