#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

args = sys.argv[1:]
if not args:
    raise SystemExit("usage: resolve_chat.py PROJECT_TAG [--json] [--no-map]")
tag = args[0].strip()
flags = set(args[1:])
if not flags.issubset({"--json", "--no-map"}):
    raise SystemExit("optional flags are --json and --no-map")
if not re.fullmatch(r"[A-Z0-9_-]+", tag):
    raise SystemExit("PROJECT_TAG must match [A-Z0-9_-]+")

HOME = Path.home()
HERE = Path(__file__).resolve().parent
DB = HOME / ".codex/sqlite/codex-dev.db"
MAPPING = HOME / "Chatty/.chatty/project_threads.json"
CURRENT = HERE / "ax_current_thread"

cp = subprocess.run([str(CURRENT)], text=True, capture_output=True, timeout=15, check=False)
if cp.returncode != 0:
    raise SystemExit("could not identify active normal ChatGPT thread: " + (cp.stderr or cp.stdout).strip())

fields = {}
for line in cp.stdout.splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        fields[k.strip()] = v.strip()
thread_id = fields.get("THREAD_ID", "")
visible_title = fields.get("TITLE", "")
if not thread_id:
    raise SystemExit("active normal ChatGPT WebArea has no thread id")

con = sqlite3.connect(DB)
try:
    row = con.execute(
        "SELECT display_title, source_kind, source_updated_at "
        "FROM local_thread_catalog WHERE thread_id=? "
        "ORDER BY source_updated_at DESC LIMIT 1",
        (thread_id,),
    ).fetchone()
finally:
    con.close()
if row is None:
    raise SystemExit(f"visible thread {thread_id} is absent from local ChatGPT catalog")
title, source_kind, updated = row
if source_kind != "chatgpt":
    raise SystemExit(f"visible thread {thread_id} is source_kind={source_kind!r}, not a normal ChatGPT chat")

record = {
    "thread_id": thread_id,
    "title": str(title or visible_title or ""),
    "visible_title": visible_title,
    "wake_mode": "chatgpt_normal_ax_thread_url",
    "bound_at": time.time(),
}
if "--no-map" not in flags:
    try:
        current = json.loads(MAPPING.read_text())
    except Exception:
        current = {}
    current[tag] = record
    tmp = MAPPING.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    tmp.replace(MAPPING)

if "--json" in flags:
    print(json.dumps(record, sort_keys=True))
else:
    print(f"BOUND {tag} {thread_id} {record['title']}")
