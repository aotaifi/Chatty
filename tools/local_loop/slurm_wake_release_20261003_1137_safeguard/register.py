#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from common import (
    LOGS,
    ROOT,
    STATES,
    TARGETS,
    atomic_json,
    query_state,
    read_json,
    refresh_chatgpt_turn_state,
    update_state,
    watcher_alive,
)

HERE = Path(__file__).resolve().parent
RESOLVE = HERE / "resolve_chat.py"
DAEMON = HERE / "daemon.py"
VERSION = (HERE / "VERSION").read_text().strip()


def resolve_current(tag: str) -> dict:
    cp = subprocess.run(
        [sys.executable, str(RESOLVE), tag, "--json", "--no-map"],
        text=True,
        capture_output=True,
        timeout=25,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError((cp.stderr or cp.stdout).strip())
    try:
        return json.loads(cp.stdout.strip().splitlines()[-1])
    except Exception as exc:
        raise RuntimeError(f"invalid resolver output: {exc}: {cp.stdout!r}") from exc


def preflight(cluster: str, job_id: str) -> tuple[str, str | None, str]:
    last = (None, None, "missing")
    for attempt in range(3):
        last = query_state(cluster, job_id)
        state, exit_code, source = last
        if state is not None and not source.startswith("remote-error:"):
            return state, exit_code, source
        if attempt < 2:
            time.sleep(1)
    state, exit_code, source = last
    raise RuntimeError(
        f"Slurm job {cluster}:{job_id} is not verifiable "
        f"(state={state!r}, source={source!r})"
    )


def main() -> int:
    # Capture any active ChatGPT turn immediately at registration time, before
    # the app's short breadcrumb buffer can rotate.
    refresh_chatgpt_turn_state()
    p = argparse.ArgumentParser(
        description="Bind the active normal ChatGPT chat to one durable Slurm watcher."
    )
    p.add_argument("job_id")
    p.add_argument("description")
    p.add_argument("project_tag")
    p.add_argument("--cluster", choices=("theorie", "paderborn"), required=True)
    p.add_argument("--poll", type=float, default=60.0)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()

    if not re.fullmatch(r"\d+", args.job_id):
        raise SystemExit("job_id must be the numeric Slurm root job id")
    if not re.fullmatch(r"[A-Z0-9_-]+", args.project_tag):
        raise SystemExit("project_tag must match [A-Z0-9_-]+")
    if args.poll < 1:
        raise SystemExit("--poll must be >= 1 second")

    register_lock = ROOT / "register_locks" / f"{args.cluster}-{args.job_id}.lock"
    register_lock.parent.mkdir(parents=True, exist_ok=True)
    register_fp = register_lock.open("a+")
    fcntl.flock(register_fp.fileno(), fcntl.LOCK_EX)

    try:
        slurm_state, exit_code, source = preflight(args.cluster, args.job_id)
    except Exception as exc:
        raise SystemExit(f"PRECHECK_FAILED: {exc}")

    try:
        bound = resolve_current(args.project_tag)
    except Exception as exc:
        raise SystemExit(f"CHAT_BIND_FAILED: {exc}")

    target_file = TARGETS / f"{args.cluster}-{args.job_id}.json"
    state_file = STATES / f"{args.cluster}-{args.job_id}.json"
    target = {
        "schema": 1,
        "workflow_version": VERSION,
        "cluster": args.cluster,
        "job_id": args.job_id,
        "description": args.description,
        "project_tag": args.project_tag,
        "thread_id": bound["thread_id"],
        "title_at_registration": bound["title"],
        "wake_mode": "chatgpt_normal_ax",
        "created_at": time.time(),
    }

    immutable = ("cluster", "job_id", "project_tag", "thread_id", "description")

    if args.prepare_only:
        if target_file.exists():
            old = read_json(target_file)
            if any(str(old.get(k)) != str(target.get(k)) for k in immutable):
                print(
                    f"PREPARE_CONFLICT cluster={args.cluster} job={args.job_id} "
                    f"existing_target={target_file}"
                )
                return 3
        print(
            f"PREPARED cluster={args.cluster} job={args.job_id} "
            f"state={slurm_state} source={source} "
            f"thread={target['thread_id']} title={target['title_at_registration']!r} "
            f"would_target={target_file}"
        )
        return 0

    if target_file.exists():
        old = read_json(target_file)
        if any(str(old.get(k)) != str(target.get(k)) for k in immutable):
            raise SystemExit(f"REFUSE_RETARGET: existing immutable target differs: {target_file}")
        target = old
    else:
        atomic_json(target_file, target)

    state = read_json(state_file)
    if state.get("wake_status") == "delivered":
        print(
            f"ALREADY_DELIVERED cluster={args.cluster} job={args.job_id} "
            f"thread={target['thread_id']}"
        )
        return 0
    if state.get("wake_status") == "uncertain":
        print(
            f"WAKE_UNCERTAIN cluster={args.cluster} job={args.job_id} "
            f"state={state_file}; manual review required"
        )
        return 3
    if watcher_alive(state, args.cluster, args.job_id, DAEMON):
        print(
            f"ALREADY_WATCHING cluster={args.cluster} job={args.job_id} "
            f"watcher_pid={state.get('watcher_pid')} thread={target['thread_id']}"
        )
        return 0

    log = LOGS / f"{args.cluster}-{args.job_id}.log"
    update_state(
        state_file,
        schema=1,
        workflow_version=VERSION,
        cluster=args.cluster,
        job_id=args.job_id,
        monitoring_enabled=True,
        poll_seconds=args.poll,
        watch_log=str(log),
        registered_at=time.time(),
        registration_slurm_state=slurm_state,
        registration_slurm_source=source,
        registration_exit_code=exit_code,
        watch_status="starting",
    )

    out = log.open("ab")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(DAEMON),
            "--cluster", args.cluster,
            "--job-id", args.job_id,
            "--target-file", str(target_file),
            "--state-file", str(state_file),
            "--poll", str(args.poll),
        ],
        stdin=subprocess.DEVNULL,
        stdout=out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    out.close()
    update_state(
        state_file,
        registration_spawn_pid=proc.pid,
        watcher_pid=proc.pid,
        watch_status="starting",
        heartbeat_at=time.time(),
    )

    print(
        f"WATCHING cluster={args.cluster} job={args.job_id} watcher_pid={proc.pid} "
        f"thread={target['thread_id']} target={target_file} state={state_file} log={log}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
