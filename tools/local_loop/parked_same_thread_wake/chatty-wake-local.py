#!/usr/bin/env python3
"""Wake an existing ChatGPT conversation through the installed ChatGPT.app.

Uses the app's public codex://threads/<id>?prompt=... deep link, then submits the
prefilled composer with a local macOS Return keystroke. No OpenAI API key.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

if len(sys.argv) < 3:
    raise SystemExit("usage: chatty-wake-local.py THREAD_ID PROMPT...")

thread_id = sys.argv[1].strip()
prompt = " ".join(sys.argv[2:]).strip()
if not thread_id or not prompt:
    raise SystemExit("thread id and prompt must be non-empty")

DB = Path.home() / ".codex/sqlite/codex-dev.db"
SCOPE = Path.home() / "Library/Application Support/Codex/sentry/scope_v3.json"


def catalog_row():
    try:
        con = sqlite3.connect(DB)
        row = con.execute(
            "SELECT host_id, source_updated_at FROM local_thread_catalog WHERE thread_id=? "
            "ORDER BY source_updated_at DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        con.close()
        return row
    except Exception:
        return None


def catalog_updated() -> float | None:
    try:
        con = sqlite3.connect(DB)
        row = con.execute(
            "SELECT source_updated_at FROM local_thread_catalog WHERE thread_id=? "
            "ORDER BY source_updated_at DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        con.close()
        return float(row[0]) if row else None
    except Exception:
        return None


def recent_conversation_post(since_epoch: float) -> bool:
    try:
        data = json.loads(SCOPE.read_text())
        for b in reversed(data.get("scope", {}).get("breadcrumbs", [])):
            if b.get("category") != "electron.net":
                continue
            d = b.get("data") or {}
            if d.get("method") != "POST" or d.get("status_code") != 200:
                continue
            if "/backend-api/f/conversation" not in d.get("url", ""):
                continue
            ts = b.get("timestamp")
            if isinstance(ts, (int, float)):
                when = float(ts)
            elif isinstance(ts, str):
                when = dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            else:
                continue
            return when >= since_epoch - 1
    except Exception:
        pass
    return False


row = catalog_row()
host_id = row[0] if row else None
baseline = float(row[1]) if row else catalog_updated()
started = time.time()
params = {"prompt": prompt}
if host_id:
    params["hostId"] = host_id
url = (
    "codex://threads/"
    + urllib.parse.quote(thread_id, safe="")
    + "?"
    + urllib.parse.urlencode(params)
)

subprocess.run(["/usr/bin/open", url], check=True)
time.sleep(1.5)

script = r'''
tell application "ChatGPT" to activate
delay 0.6
tell application "System Events"
    keystroke return
end tell
'''
cp = subprocess.run(
    ["/usr/bin/osascript", "-e", script],
    text=True,
    capture_output=True,
    timeout=15,
)
if cp.returncode != 0:
    raise RuntimeError("osascript failed: " + (cp.stderr or cp.stdout).strip())

# Confirmation is deliberately backend-facing rather than just "keypress ran".
# A successful same-thread turn updates the local ChatGPT catalog; the network
# breadcrumb is a secondary corroboration.
deadline = time.time() + 30
while time.time() < deadline:
    after = catalog_updated()
    catalog_moved = (
        after is not None
        and (baseline is None or after > baseline + 1e-6)
    )
    if catalog_moved and recent_conversation_post(started):
        print(
            "WAKE_SUBMITTED",
            thread_id,
            "catalog_before=",
            baseline,
            "catalog_after=",
            after,
            flush=True,
        )
        raise SystemExit(0)
    time.sleep(1)

raise SystemExit(
    "wake submission could not be verified within 30 seconds; leave WAKE_PENDING for recovery"
)
