#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from common import (
    LOGS,
    STATES,
    TARGETS,
    read_json,
    refresh_chatgpt_turn_state,
    ui_context_trusted,
    update_state,
    watcher_alive,
)

HERE = Path(__file__).resolve().parent
DAEMON = HERE / "daemon.py"


def recover_one(state_file: Path, trusted_ui: bool) -> str:
    state = read_json(state_file)
    if not state.get("monitoring_enabled"):
        return "inactive"

    cluster = str(state.get("cluster") or "")
    job_id = str(state.get("job_id") or "")
    if cluster not in {"theorie", "paderborn"} or not job_id.isdigit():
        return "invalid-state"

    if state.get("wake_status") in {"delivered", "uncertain", "failed"}:
        return "terminal-local"

    # A terminal event waiting for ChatGPT delivery must not be spawned from
    # launchd's untrusted Accessibility context. The trusted broker calls this
    # same recovery entry point and will spawn it safely.
    if state.get("terminal_event") and not trusted_ui:
        return "ui-broker-required"

    retry_at = float(state.get("next_wake_retry_at") or 0)
    if retry_at > time.time():
        return "not-due"

    if watcher_alive(state, cluster, job_id, DAEMON):
        return "alive"

    target_file = TARGETS / f"{cluster}-{job_id}.json"
    if not target_file.exists():
        update_state(
            state_file,
            watch_status="recovery-target-missing",
            recovery_error=f"missing target {target_file}",
            recovery_last_attempt_at=time.time(),
        )
        return "target-missing"

    poll = float(state.get("poll_seconds") or 60.0)
    log = Path(state.get("watch_log") or (LOGS / f"{cluster}-{job_id}.log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    out = log.open("ab")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(DAEMON),
            "--cluster", cluster,
            "--job-id", job_id,
            "--target-file", str(target_file),
            "--state-file", str(state_file),
            "--poll", str(poll),
        ],
        stdin=subprocess.DEVNULL,
        stdout=out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    out.close()
    update_state(
        state_file,
        watcher_pid=proc.pid,
        recovery_spawn_pid=proc.pid,
        recovery_last_attempt_at=time.time(),
        watch_status="recovering",
    )
    return f"spawned:{proc.pid}"


def main() -> int:
    refresh_chatgpt_turn_state()
    trusted_ui = ui_context_trusted()
    STATES.mkdir(parents=True, exist_ok=True)
    changed = 0
    for state_file in sorted(STATES.glob("*.json")):
        result = recover_one(state_file, trusted_ui)
        if result.startswith("spawned:") or result in {"target-missing", "invalid-state"}:
            print(f"{state_file.name} {result}")
            changed += 1
    if changed == 0:
        print("RECOVERY_OK no-action")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
