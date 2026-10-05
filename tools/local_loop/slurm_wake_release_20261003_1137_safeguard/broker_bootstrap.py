#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from common import ROOT

HERE = Path(__file__).resolve().parent
STATUS = ROOT / "ui-broker.json"
COMMAND = HERE / "ui_broker.command"

def alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        cp = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "command="],
            text=True,
            capture_output=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return False
    return cp.returncode == 0 and str(HERE / "ui_broker.py") in (cp.stdout or "")

try:
    state = json.loads(STATUS.read_text())
except Exception:
    state = {}
pid = int(state.get("pid") or 0)
if alive(pid):
    print(f"UI_BROKER_OK pid={pid}")
    raise SystemExit(0)

# Terminal.app already has user-granted Accessibility. Opening the broker
# command through Terminal gives its detached descendants that trusted user
# session lineage. -g keeps Terminal in the background; the command detaches
# the broker and exits immediately.
cp = subprocess.run(
    ["/usr/bin/open", "-g", "-a", "Terminal", str(COMMAND)],
    text=True,
    capture_output=True,
    timeout=10,
    check=False,
)
if cp.returncode != 0:
    raise SystemExit("UI_BROKER_BOOTSTRAP_FAILED: " + (cp.stderr or cp.stdout).strip())
print("UI_BROKER_BOOTSTRAP_REQUESTED")
