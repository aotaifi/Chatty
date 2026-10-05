#!/usr/bin/env python3
"""Persistent same-thread wake broker for Chatty research jobs.

Keeps ChatGPT.app's bundled CUA runtime warm and serves wake requests over a
local Unix socket. Delivery reuses an exact existing ChatGPT Chrome tab and uses
DOM locators; it never calls createBrowserTab and never reuses AX element IDs.
"""
from __future__ import annotations

import json
import os
import plistlib
import re
import select
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = Path.home() / "Chatty"
STATE = REPO / ".chatty"
SOCKET_PATH = STATE / "wake.sock"
APP = Path("/Applications/ChatGPT.app")
RES = APP / "Contents/Resources"
BASE = RES / "cua_node"
NODE = BASE / "bin/node"
SERVER = BASE / "lib/node_modules/@oai/cua-repl/bin/cua-repl.mjs"
CALL_TIMEOUT = 24.0
POLL_SECONDS = 2.0

STATE.mkdir(parents=True, exist_ok=True)

try:
    with (APP / "Contents/Info.plist").open("rb") as f:
        APP_VERSION = plistlib.load(f).get("CFBundleShortVersionString", "")
except Exception:
    APP_VERSION = ""


class CuaSession:
    def __init__(self):
        self.proc: subprocess.Popen[str] | None = None
        self.rid = 0
        self.thread_id = ""

    def stop(self):
        p = self.proc
        self.proc = None
        if p is None:
            return
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            p.wait(timeout=3)
        except Exception:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass

    def start(self, thread_id: str):
        self.stop()
        self.thread_id = thread_id
        self.rid = 0
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
            "BROWSER_USE_CODEX_APP_VERSION": APP_VERSION,
            "NODE_REPL_TRUSTED_SERVICES": '{"browser":"@oai/browser-desktop/service","sky":"@oai/sky/service"}',
            "SKY_CUA_SERVICE_PATH": os.path.expanduser("~/.codex/computer-use/Codex Computer Use.app"),
            "CODEX_CLI_PATH": str(RES / "codex"),
            "CUA_REPL_NODE_REPL_PATH": str(BASE / "bin/node_repl"),
            "CUA_REPL_ENABLED_SURFACES": "browser,computer",
        })
        self.proc = subprocess.Popen(
            [str(NODE), str(SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        init = self.rpc(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {"elicitation": {}},
                "clientInfo": {"name": "chatty-wake-daemon", "version": "1.0"},
            },
            timeout=20,
            include_meta=False,
        )
        if "error" in init:
            raise RuntimeError("MCP initialize failed: " + json.dumps(init["error"]))
        self.notify("notifications/initialized", {})

    def raw_send(self, obj: dict):
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("CUA process is not running")
        self.proc.stdin.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

    def notify(self, method: str, params=None):
        obj = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            obj["params"] = params
        self.raw_send(obj)

    def wait_id(self, wanted: int, timeout: float):
        if self.proc is None or self.proc.stdout is None or self.proc.stderr is None:
            raise RuntimeError("CUA process is not running")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"CUA process exited rc={self.proc.returncode}")
            ready, _, _ = select.select([self.proc.stdout, self.proc.stderr], [], [], 0.5)
            for stream in ready:
                line = stream.readline()
                if not line:
                    continue
                if stream is self.proc.stderr:
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
                    self.raw_send({
                        "jsonrpc": "2.0",
                        "id": obj["id"],
                        "result": {"action": "accept", "content": {}} if allowed else {"action": "cancel"},
                    })
                    continue
                if obj.get("id") == wanted:
                    return obj
        raise TimeoutError(f"CUA RPC {wanted} exceeded {timeout}s")

    def rpc(self, method: str, params=None, timeout: float = CALL_TIMEOUT, include_meta: bool = False):
        self.rid += 1
        obj = {"jsonrpc": "2.0", "id": self.rid, "method": method}
        if params is not None:
            obj["params"] = params
        if include_meta and params is not None:
            pass
        self.raw_send(obj)
        return self.wait_id(self.rid, timeout)

    def js(self, code: str, title: str, thread_id: str, timeout: float = CALL_TIMEOUT) -> str:
        self.rid += 1
        meta = {
            "x-codex-turn-metadata": json.dumps({
                "session_id": thread_id,
                "thread_id": thread_id,
                "turn_id": "chatty-wake-" + uuid.uuid4().hex,
                "model": "gpt-5.6-sol",
            })
        }
        obj = {
            "jsonrpc": "2.0",
            "id": self.rid,
            "method": "tools/call",
            "params": {
                "name": "js",
                "arguments": {"code": code, "title": title},
                "_meta": meta,
            },
        }
        self.raw_send(obj)
        result = self.wait_id(self.rid, timeout)
        if "error" in result:
            raise RuntimeError("MCP error: " + json.dumps(result["error"]))
        tr = result.get("result") or {}
        if tr.get("isError"):
            raise RuntimeError("CUA tool error: " + json.dumps(tr))
        return "\n".join(
            c.get("text", "") for c in tr.get("content", []) if c.get("type") == "text"
        )


def marker_json(text: str, marker: str) -> dict:
    hits = re.findall(re.escape(marker) + r"\s+(\{[^\n]+\})", text)
    if not hits:
        raise RuntimeError(f"{marker} missing: {text[-1200:]}")
    return json.loads(hits[-1])


def list_tabs(cua: CuaSession, thread_id: str) -> list[dict]:
    out = cua.js(
        'let ts=await cua.listTabs({browser:"chrome",emit:false});'
        'nodeRepl.write("CHATTY_TABS "+JSON.stringify({tabs:ts}));',
        "List existing ChatGPT tabs",
        thread_id,
    )
    return marker_json(out, "CHATTY_TABS").get("tabs", [])


def bind_tab(cua: CuaSession, thread_id: str, tab_id: str):
    out = cua.js(
        "globalThis.__chattyWakeTab=await cua.getTab("
        + json.dumps(str(tab_id))
        + ',{browser:"chrome"});'
        + 'nodeRepl.write("CHATTY_BOUND "+JSON.stringify({id:globalThis.__chattyWakeTab.id}));',
        "Bind target ChatGPT tab",
        thread_id,
    )
    marker_json(out, "CHATTY_BOUND")


def read_state(cua: CuaSession, thread_id: str, delivery_marker: str) -> dict:
    code = (
        'let p=globalThis.__chattyWakeTab.playwright.locator("#prompt-textarea");'
        'let n=await p.count();'
        'let visible=n>0?await p.first().isVisible():false;'
        'let enabled=n>0?await p.first().isEnabled():false;'
        "let seen=await globalThis.__chattyWakeTab.playwright.getByText("
        + json.dumps(delivery_marker)
        + ',{exact:false}).count();'
        'nodeRepl.write("CHATTY_STATE "+JSON.stringify({n:n,visible:visible,enabled:enabled,seen:seen}));'
    )
    return marker_json(cua.js(code, "Check wake readiness", thread_id), "CHATTY_STATE")


def send_prompt(cua: CuaSession, thread_id: str, prompt: str):
    code = (
        'let p=globalThis.__chattyWakeTab.playwright.locator("#prompt-textarea").first();'
        "await p.fill(" + json.dumps(prompt) + ",{timeoutMs:5000});"
        'await p.press("Enter",{timeoutMs:5000});'
        'nodeRepl.write("CHATTY_SENT "+JSON.stringify({ok:true}));'
    )
    state = marker_json(cua.js(code, "Submit same-thread wake", thread_id), "CHATTY_SENT")
    if not state.get("ok"):
        raise RuntimeError("submit did not return ok")


def verify(cua: CuaSession, thread_id: str, delivery_marker: str, seconds: float = 30) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        code = (
            "let seen=await globalThis.__chattyWakeTab.playwright.getByText("
            + json.dumps(delivery_marker)
            + ',{exact:false}).count();'
            'nodeRepl.write("CHATTY_VERIFY "+JSON.stringify({seen:seen}));'
        )
        state = marker_json(cua.js(code, "Verify wake delivery", thread_id), "CHATTY_VERIFY")
        if state.get("seen", 0) > 0:
            return True
        time.sleep(1)
    return False


def deliver(cua: CuaSession, thread_id: str, prompt: str, timeout: float) -> dict:
    target_url = f"https://chatgpt.com/c/{thread_id}"
    match = re.search(r"(?m)^job_id:\s*([^\s]+)", prompt)
    delivery_marker = f"job_id: {match.group(1)}" if match else prompt[:120]
    deadline = time.monotonic() + timeout
    opened = False
    bound_id: str | None = None
    last_error = ""

    while time.monotonic() < deadline:
        try:
            if cua.proc is None or cua.proc.poll() is not None:
                cua.start(thread_id)
            tabs = list_tabs(cua, thread_id)
            exact = [t for t in tabs if t.get("url") == target_url]
            if not exact:
                if not opened:
                    subprocess.run(
                        ["/usr/bin/open", "-a", "Google Chrome", target_url],
                        check=False,
                        timeout=10,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    opened = True
                print("WAIT no exact target tab", target_url, flush=True)
                time.sleep(POLL_SECONDS)
                continue

            tab_id = str(exact[0]["id"])
            if bound_id != tab_id:
                bind_tab(cua, thread_id, tab_id)
                bound_id = tab_id
                print("BOUND", tab_id, target_url, flush=True)

            state = read_state(cua, thread_id, delivery_marker)
            print("STATE", json.dumps(state), flush=True)
            if state.get("seen", 0) > 0:
                return {"ok": True, "already_present": True, "tab_id": tab_id}
            if state.get("n", 0) > 0 and state.get("visible") and state.get("enabled"):
                send_prompt(cua, thread_id, prompt)
                if verify(cua, thread_id, delivery_marker):
                    return {"ok": True, "already_present": False, "tab_id": tab_id}
                raise RuntimeError("submit returned but delivery marker not visible")

            time.sleep(POLL_SECONDS)
        except (TimeoutError, RuntimeError) as exc:
            last_error = repr(exc)
            print("CUA_RESTART", last_error, file=sys.stderr, flush=True)
            cua.stop()
            bound_id = None
            time.sleep(min(5.0, POLL_SECONDS))

    return {"ok": False, "error": last_error or "delivery deadline exceeded"}


def recv_line(conn: socket.socket, limit: int = 1_000_000) -> bytes:
    chunks = []
    total = 0
    while True:
        b = conn.recv(65536)
        if not b:
            break
        total += len(b)
        if total > limit:
            raise ValueError("request too large")
        chunks.append(b)
        if b"\n" in b:
            break
    return b"".join(chunks).split(b"\n", 1)[0]


def main():
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(SOCKET_PATH))
    os.chmod(SOCKET_PATH, 0o600)
    server.listen(8)
    print("WAKE_DAEMON_LISTENING", SOCKET_PATH, flush=True)

    cua = CuaSession()
    try:
        # Best-effort prewarm. Failure is not fatal; a request will restart it.
        try:
            cua.start("prewarm")
            list_tabs(cua, "prewarm")
            print("WAKE_DAEMON_PREWARM_OK", flush=True)
        except Exception as exc:
            print("WAKE_DAEMON_PREWARM_FAILED", repr(exc), file=sys.stderr, flush=True)
            cua.stop()

        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    request = json.loads(recv_line(conn).decode("utf-8"))
                    thread_id = str(request["thread_id"]).strip()
                    prompt = str(request["prompt"])
                    timeout = float(request.get("timeout", 900))
                    if not thread_id or not prompt:
                        raise ValueError("thread_id and prompt required")
                    result = deliver(cua, thread_id, prompt, timeout)
                except Exception as exc:
                    result = {"ok": False, "error": repr(exc)}
                try:
                    conn.sendall((json.dumps(result) + "\n").encode("utf-8"))
                except (BrokenPipeError, ConnectionResetError):
                    print("CLIENT_DISCONNECTED before wake result was read", file=sys.stderr, flush=True)
    finally:
        cua.stop()
        server.close()
        try:
            SOCKET_PATH.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
