#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT, atomic_json, ui_context_trusted

HERE = Path(__file__).resolve().parent
RECOVER = HERE / "recover.py"
LOCK = ROOT / "ui-broker.lock"
STATUS = ROOT / "ui-broker.json"
INTERVAL = float(os.environ.get("CHATTY_UI_BROKER_INTERVAL", "10"))

def main() -> int:
    lock_fp = LOCK.open("a+")
    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0

    if not ui_context_trusted():
        atomic_json(STATUS, {
            "schema": 1,
            "pid": os.getpid(),
            "trusted": False,
            "status": "refused-untrusted-context",
            "updated_at": time.time(),
        })
        return 20

    atomic_json(STATUS, {
        "schema": 1,
        "pid": os.getpid(),
        "trusted": True,
        "status": "running",
        "started_at": time.time(),
        "updated_at": time.time(),
    })

    stopping = False
    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    while not stopping:
        started = time.time()
        cp = subprocess.run(
            [sys.executable, str(RECOVER)],
            text=True,
            capture_output=True,
            timeout=45,
            check=False,
            env={**os.environ, "CHATTY_UI_BROKER": "1"},
        )
        atomic_json(STATUS, {
            "schema": 1,
            "pid": os.getpid(),
            "trusted": True,
            "status": "running",
            "updated_at": time.time(),
            "last_recover_started_at": started,
            "last_recover_returncode": cp.returncode,
            "last_recover_stdout": (cp.stdout or "")[-4000:],
            "last_recover_stderr": (cp.stderr or "")[-4000:],
        })
        deadline = time.time() + INTERVAL
        while not stopping and time.time() < deadline:
            time.sleep(min(0.5, max(0.0, deadline - time.time())))

    atomic_json(STATUS, {
        "schema": 1,
        "pid": os.getpid(),
        "trusted": True,
        "status": "stopped",
        "updated_at": time.time(),
    })
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
