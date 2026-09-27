#!/usr/bin/env python3
"""Submit a prompt into an existing ChatGPT conversation using ChatGPT.app's bundled browser CUA.

The CUA JS kernel has a hard ~30 s execution limit. Long waits therefore happen
in Python between short tools/call invocations; never spin/sleep in one JS call.

No OpenAI API key is used. Browser-origin access is approved only for
https://chatgpt.com.
"""
from __future__ import annotations

import json
import os
import plistlib
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

if len(sys.argv) < 3:
    raise SystemExit("usage: chatty-wake-browser.py THREAD_ID PROMPT...")

thread_id = sys.argv[1].strip()
prompt = " ".join(sys.argv[2:]).strip()
if not thread_id or not prompt:
    raise SystemExit("thread id and prompt must be non-empty")
if not SERVER.exists():
    raise SystemExit(f"ChatGPT CUA runtime not found: {SERVER}")

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
)
rid = 0


def send(method: str, params=None) -> int:
    global rid
    rid += 1
    msg = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        msg["params"] = params
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
    proc.stdin.flush()
    return rid


def notify(method: str, params=None) -> None:
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
    proc.stdin.flush()


def wait_id(wanted: int, timeout: int = 45):
    deadline = time.time() + timeout
    assert proc.stdout is not None and proc.stderr is not None
    while time.time() < deadline:
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
                result = {"action": "accept", "content": {}} if allowed else {"action": "cancel"}
                reply = {"jsonrpc": "2.0", "id": obj["id"], "result": result}
                assert proc.stdin is not None
                proc.stdin.write(json.dumps(reply, separators=(",", ":")) + "\n")
                proc.stdin.flush()
                continue
            if obj.get("id") == wanted:
                return obj
    raise TimeoutError(f"timeout waiting for MCP id {wanted}")


def tool_text(result: dict) -> str:
    if "error" in result:
        raise RuntimeError("MCP error: " + json.dumps(result["error"]))
    tool_result = result.get("result") or {}
    if tool_result.get("isError"):
        raise RuntimeError("CUA tool error: " + json.dumps(tool_result))
    return "\n".join(
        c.get("text", "")
        for c in tool_result.get("content", [])
        if c.get("type") == "text"
    )


def call_js(code: str, title: str, timeout: int = 45) -> str:
    """Run one bounded CUA step with an explicit node_repl timeout."""
    meta = {
        "session_id": thread_id,
        "turn_id": "chatty-wake-" + uuid.uuid4().hex,
        "thread_id": thread_id,
        "model": "gpt-5.6-sol",
    }
    # The bundled node_repl defaults to 30000 ms. Never rely on that default.
    tool_timeout_ms = max(1000, (timeout - 5) * 1000)
    call_id = send("tools/call", {
        "name": "js",
        "arguments": {
            "code": code,
            "title": title,
            "timeout_ms": tool_timeout_ms,
        },
        "_meta": {"x-codex-turn-metadata": json.dumps(meta)},
    })
    return tool_text(wait_id(call_id, timeout))

def extract_marker(text: str, prefix: str) -> str | None:
    # nodeRepl.write output may have wrapper prose around it.
    idx = text.rfind(prefix)
    if idx < 0:
        return None
    return text[idx:].splitlines()[0].strip()


try:
    init_id = send("initialize", {
        "protocolVersion": "2025-06-18",
        "capabilities": {"elicitation": {}},
        "clientInfo": {"name": "chatty-waker", "version": "1.2"},
    })
    init = wait_id(init_id, 30)
    if "error" in init:
        raise RuntimeError("MCP initialize failed: " + json.dumps(init["error"]))
    notify("notifications/initialized", {})

    target_url = "https://chatgpt.com/c/" + thread_id

    # Browser provider ids are ephemeral to this CUA process. Discover Chrome
    # first, then create the tab with that id in a separate short JS call.
    # This avoids the slow implicit provider discovery triggered by passing the
    # family name directly to createBrowserTab.
    browsers_code = (
        "globalThis.__chattyWakeBrowsers = await cua.listBrowsers({emit:false});"
        + "nodeRepl.write('CHATTY_BROWSERS ' + JSON.stringify(globalThis.__chattyWakeBrowsers));"
    )
    browsers_text = call_js(browsers_code, "Discover Chrome browser", timeout=45)
    marker = extract_marker(browsers_text, "CHATTY_BROWSERS ")
    if marker is None:
        raise RuntimeError("browser discovery marker missing: " + browsers_text[-2000:])
    browser_list = json.loads(marker[len("CHATTY_BROWSERS "):])
    chrome = next(
        (
            b for b in browser_list
            if b.get("family") == "chrome" or b.get("name") == "Chrome"
        ),
        None,
    )
    if chrome is None:
        raise RuntimeError("Chrome browser provider unavailable: " + repr(browser_list))
    browser_id = str(chrome["id"])
    print("WAKE_BROWSER", json.dumps(chrome, sort_keys=True), flush=True)

    # Important: no long waits inside JS. Keep the browser tab in globalThis so
    # short tools/call invocations can reuse it.
    create_code = (
        "globalThis.__chattyWakeTab = await cua.createBrowserTab("
        + json.dumps(browser_id) + ","
        + json.dumps(target_url)
        + ",{sessionName:" + json.dumps("Chatty wake") + "});"
        + "nodeRepl.write(" + json.dumps("CHATTY_TAB_CREATED") + ");"
    )
    created = call_js(create_code, "Open target ChatGPT thread", timeout=45)
    if "CHATTY_TAB_CREATED" not in created:
        raise RuntimeError("browser tab creation marker missing: " + created[-2000:])

    # Poll from Python. Each AX read gets its own <=30 s JS execution budget.
    ready = False
    deadline = time.time() + 15 * 60
    last_state = None
    last_state_log = 0.0
    while time.time() < deadline:
        state_code = r"""
const ax = await globalThis.__chattyWakeTab.getAXState({emit:false, disableDiffing:true});
const m = ax.match(/(\d+)\s+text entry area \(settable\) Description: (?:ChatGPT fragen|Ask ChatGPT)/);
const active = /\bbutton (?:Stoppen|Stop)\b/.test(ax);
const pending = ax.includes("pending-conversation-input");
const onTarget = ax.includes("/c/THREAD_ID");
const promptSeen = ax.includes(WAKE_PROMPT);
nodeRepl.write("CHATTY_STATE " + JSON.stringify({
  composer: m ? Number(m[1]) : null,
  active, pending, onTarget, promptSeen
}));
""".replace("THREAD_ID", thread_id).replace("WAKE_PROMPT", json.dumps(prompt))
        out = call_js(state_code, "Check target thread readiness")
        marker = extract_marker(out, "CHATTY_STATE ")
        if marker:
            state = json.loads(marker[len("CHATTY_STATE "):])
            now = time.time()
            if state != last_state or now - last_state_log >= 30:
                print("WAKE_STATE", json.dumps(state, sort_keys=True), flush=True)
                last_state = state
                last_state_log = now
            if state.get("promptSeen") and state.get("onTarget") and not state.get("active"):
                print("WAKE_ALREADY_PRESENT", thread_id, flush=True)
                print("WAKE_SUBMITTED", thread_id, flush=True)
                raise SystemExit(0)
            if state.get("composer") is not None and not state.get("active") and not state.get("pending") and state.get("onTarget"):
                ready = True
                break
        time.sleep(2)
    if not ready:
        raise RuntimeError("target thread did not become idle/ready within 15 minutes")

    paste_code = (
        "const ax = await globalThis.__chattyWakeTab.getAXState({emit:false,disableDiffing:true});"
        "const m=ax.match(/(\\d+)\\s+text entry area \\(settable\\) Description: (?:ChatGPT fragen|Ask ChatGPT)/);"
        "if(!m) throw new Error('composer missing before paste');"
        "await globalThis.__chattyWakeTab.paste(Number(m[1]),"
        + json.dumps(prompt)
        + ",{format:'text'});"
        "nodeRepl.write('CHATTY_PASTED');"
    )
    if "CHATTY_PASTED" not in call_js(paste_code, "Fill target thread composer"):
        raise RuntimeError("paste marker missing")

    verify_code = (
        "const ax=await globalThis.__chattyWakeTab.getAXState({emit:false,disableDiffing:true});"
        "nodeRepl.write('CHATTY_FILLED ' + "
        + json.dumps(prompt)
        + ".length + ' ' + (ax.includes("
        + json.dumps(prompt)
        + ") ? 'YES':'NO'));"
    )
    verified = call_js(verify_code, "Verify target thread composer")
    if "CHATTY_FILLED" not in verified or "YES" not in verified:
        raise RuntimeError("composer fill verification failed: " + verified[-2000:])

    send_code = r"""
const ax = await globalThis.__chattyWakeTab.getAXState({emit:false, disableDiffing:true});
const m = ax.match(/(\d+)\s+text entry area \(settable\) Description: (?:ChatGPT fragen|Ask ChatGPT)/);
if (!m) throw new Error("composer missing before submit");
await globalThis.__chattyWakeTab.pressKey(Number(m[1]), "Return");
nodeRepl.write("CHATTY_RETURN_SENT");
"""
    if "CHATTY_RETURN_SENT" not in call_js(send_code, "Submit wake prompt"):
        raise RuntimeError("submit marker missing")

    # Submission verification is also bounded and retried from Python.
    submitted = False
    for _ in range(15):
        out = call_js(
            "const ax=await globalThis.__chattyWakeTab.getAXState({emit:false,disableDiffing:true});"
            + "nodeRepl.write('CHATTY_SENT ' + (ax.includes("
            + json.dumps(prompt)
            + ") ? 'YES':'NO'));",
            "Verify wake submission",
        )
        if "CHATTY_SENT YES" in out:
            submitted = True
            break
        time.sleep(1)
    if not submitted:
        raise RuntimeError("submitted prompt was not visible in target thread")

    print("WAKE_SUBMITTED", thread_id, flush=True)
finally:
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
