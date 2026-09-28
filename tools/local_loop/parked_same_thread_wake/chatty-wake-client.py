#!/usr/bin/env python3
"""Client for the persistent Chatty wake daemon."""
from __future__ import annotations
import json, socket, sys
from pathlib import Path

SOCK = Path.home() / "Chatty/.chatty/wake.sock"
if len(sys.argv) < 3:
    raise SystemExit("usage: chatty-wake-client.py THREAD_ID PROMPT...")
thread_id = sys.argv[1].strip()
prompt = " ".join(sys.argv[2:]).strip()
req = {"thread_id": thread_id, "prompt": prompt, "timeout": 900}

s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(930)
try:
    s.connect(str(SOCK))
except OSError as exc:
    raise SystemExit(f"wake daemon unavailable at {SOCK}: {exc}")
s.sendall((json.dumps(req) + "\n").encode())
buf = bytearray()
while True:
    chunk = s.recv(65536)
    if not chunk:
        break
    buf.extend(chunk)
    if b"\n" in chunk:
        break
s.close()
if not buf:
    raise SystemExit("wake daemon closed without a response")
result = json.loads(bytes(buf).split(b"\n",1)[0])
print(json.dumps(result), flush=True)
raise SystemExit(0 if result.get("ok") else 1)
