#!/usr/bin/env python3
"""Submit a prompt into an existing ChatGPT conversation using ChatGPT.app's bundled browser CUA.

This intentionally only grants browser-origin access to https://chatgpt.com.
No OpenAI API key is used.
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
env.update(
    {
        "NODE_REPL_NATIVE_PIPE_CONNECT_TIMEOUT_MS": "1000",
        "NODE_REPL_NODE_MODULE_DIRS": str(BASE / "lib/node_modules"),
        "NODE_REPL_NODE_PATH": str(NODE),
        "NODE_REPL_TRUSTED_CODE_PATHS": os.path.expanduser("~/.codex")
        + ":"
        + str(BASE / "lib/node_modules"),
        "CODEX_HOME": os.path.expanduser("~/.codex"),
        "BROWSER_USE_AVAILABLE_BACKENDS": "chrome,iab",
        "BROWSER_USE_TINYSKY_ENABLED": "1",
        "NODE_REPL_INSTRUCTIONS_USE_CASE_BROWSER": "Control the in-app browser in conjunction with the Browser Plugin.",
        "NODE_REPL_INSTRUCTIONS_USE_CASE_CHROME": "Control the Chrome browser in conjunction with the Chrome Plugin.",
        "NODE_REPL_INSTRUCTIONS_USE_CASE_COMPUTER_USE": "Control desktop apps on macOS through Computer Use.",
        "BROWSER_USE_CODEX_APP_BUILD_FLAVOR": "prod",
        "BROWSER_USE_CODEX_APP_VERSION": app_version,
        "NODE_REPL_TRUSTED_SERVICES": '{"browser":"@oai/browser-desktop/service","sky":"@oai/sky/service"}',
        "SKY_CUA_SERVICE_PATH": os.path.expanduser(
            "~/.codex/computer-use/Codex Computer Use.app"
        ),
        "CODEX_CLI_PATH": str(RES / "codex"),
        "CUA_REPL_NODE_REPL_PATH": str(BASE / "bin/node_repl"),
        "CUA_REPL_ENABLED_SURFACES": "browser,computer",
    }
)

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


def wait_id(wanted: int, timeout: int = 180):
    deadline = time.time() + timeout
    assert proc.stdout is not None and proc.stderr is not None
    while time.time() < deadline:
        ready, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.5)
        for stream in ready:
            line = stream.readline()
            if not line:
                continue
            if stream is proc.stderr:
                print("CUA:", line.rstrip(), file=sys.stderr)
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # Least privilege: approve only the exact ChatGPT browser origin.
            if obj.get("method") == "elicitation/create" and "id" in obj:
                params = obj.get("params") or {}
                meta = params.get("_meta") or {}
                allowed = (
                    meta.get("tool_name") == "access_browser_origin"
                    and meta.get("origin") == "https://chatgpt.com"
                )
                result = (
                    {"action": "accept", "content": {}}
                    if allowed
                    else {"action": "cancel"}
                )
                reply = {"jsonrpc": "2.0", "id": obj["id"], "result": result}
                assert proc.stdin is not None
                proc.stdin.write(json.dumps(reply, separators=(",", ":")) + "\n")
                proc.stdin.flush()
                continue

            if obj.get("id") == wanted:
                return obj
    raise TimeoutError(f"timeout waiting for MCP id {wanted}")


try:
    init_id = send(
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {"elicitation": {}},
            "clientInfo": {"name": "chatty-waker", "version": "1.1"},
        },
    )
    init = wait_id(init_id, 30)
    if "error" in init:
        raise RuntimeError("MCP initialize failed: " + json.dumps(init["error"]))
    notify("notifications/initialized", {})

    target_url = "https://chatgpt.com/c/" + thread_id
    js = r"""
const targetUrl = TARGET_URL;
const wakePrompt = WAKE_PROMPT;
const sleep = ms => new Promise(r => setTimeout(r, ms));
let tab = await cua.createBrowserTab("chrome", "https://chatgpt.com", {sessionName:"Chatty wake"});
await tab.goto(targetUrl);
let composer = null;
for (let i = 0; i < 180; i++) {
  const ax = await tab.getAXState({emit:false, disableDiffing:true});
  const match = ax.match(/(\d+)\s+text entry area \(settable\) Description: (?:ChatGPT fragen|Ask ChatGPT)/);
  const active = /\bbutton (?:Stoppen|Stop)\b/.test(ax);
  const pending = ax.includes("pending-conversation-input");
  const onTarget = ax.includes("/c/" + targetUrl.split("/").pop());
  if (match && !active && !pending && onTarget) {
    composer = Number(match[1]);
    break;
  }
  await sleep(1000);
}
if (composer === null) throw new Error("target thread never became idle/ready");
await tab.paste(composer, wakePrompt, {format:"text"});
await sleep(300);
const filled = await tab.getAXState({emit:false, disableDiffing:true});
const filledMatch = filled.match(/(\d+)\s+text entry area \(settable\) Description: (?:ChatGPT fragen|Ask ChatGPT)/);
if (!filledMatch || !filled.includes(wakePrompt)) throw new Error("composer fill verification failed");
await tab.pressKey(Number(filledMatch[1]), "Return");
await sleep(1500);
const sent = await tab.getAXState({emit:false, disableDiffing:true});
if (!sent.includes(wakePrompt)) throw new Error("submitted prompt not visible in thread");
nodeRepl.write("WAKE_SUBMITTED " + targetUrl);
"""
    js = js.replace("TARGET_URL", json.dumps(target_url)).replace(
        "WAKE_PROMPT", json.dumps(prompt)
    )

    turn_meta = {
        "session_id": thread_id,
        "turn_id": "chatty-wake-" + uuid.uuid4().hex,
        "thread_id": thread_id,
        "model": "gpt-5.6-sol",
    }
    call_id = send(
        "tools/call",
        {
            "name": "js",
            "arguments": {"code": js, "title": "Wake ChatGPT thread"},
            "_meta": {"x-codex-turn-metadata": json.dumps(turn_meta)},
        },
    )
    result = wait_id(call_id, 240)
    if "error" in result:
        raise RuntimeError(json.dumps(result["error"]))
    tool_result = result.get("result") or {}
    if tool_result.get("isError"):
        raise RuntimeError(json.dumps(tool_result))
    text_parts = [
        c.get("text", "")
        for c in tool_result.get("content", [])
        if c.get("type") == "text"
    ]
    out = "\n".join(text_parts)
    marker = "WAKE_SUBMITTED " + target_url
    if marker not in out:
        raise RuntimeError("submission verification marker missing: " + out[-2000:])
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
