#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from common import (
    TERMINAL_FAILURES,
    TERMINAL_SUCCESS,
    post_slack,
    query_state,
    read_json,
    refresh_chatgpt_turn_state,
    system_idle_seconds,
    ui_context_trusted,
    update_state,
)

HERE = Path(__file__).resolve().parent
WAKE = HERE / "wake.py"


def load_target(path: Path, cluster: str, job_id: str) -> dict:
    d = read_json(path)
    for key, expected in (("cluster", cluster), ("job_id", job_id)):
        if str(d.get(key)) != str(expected):
            raise RuntimeError(
                f"target mismatch for {key}: expected {expected!r}, got {d.get(key)!r}"
            )
    for key in ("project_tag", "thread_id", "description"):
        if not d.get(key):
            raise RuntimeError(f"target missing {key}")
    return d


def build_prompt(target: dict, event: str, code: str | None) -> str:
    tag = target["project_tag"]
    cluster = target["cluster"]
    job = target["job_id"]
    if event == "FAILED":
        return (
            f"{tag} cluster={cluster} job {job} FAILED exit={code or 'unknown'} — "
            "inspect cluster outputs and continue autonomously."
        )
    return (
        f"{tag} cluster={cluster} job {job} FINISHED — "
        "inspect cluster outputs and continue autonomously."
    )


def wake_once(thread_id: str, prompt: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [sys.executable, str(WAKE), thread_id, prompt],
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            [str(WAKE), thread_id, prompt],
            124,
            exc.stdout or "",
            ((exc.stderr or "") + "\nwake timed out").strip(),
        )

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cluster", choices=("theorie", "paderborn"), required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--target-file", type=Path, required=True)
    p.add_argument("--state-file", type=Path, required=True)
    p.add_argument("--poll", type=float, default=60.0)
    args = p.parse_args()

    if not args.job_id.isdigit():
        raise SystemExit("job id must be numeric")
    if args.poll < 1:
        raise SystemExit("poll must be >= 1 second")

    watch_lock = args.state_file.with_suffix(args.state_file.suffix + ".watch.lock")
    watch_lock.parent.mkdir(parents=True, exist_ok=True)
    watch_fp = watch_lock.open("a+")
    try:
        fcntl.flock(watch_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"DAEMON_ALREADY_RUNNING cluster={args.cluster} job={args.job_id}", flush=True)
        return 0

    target = load_target(args.target_file, args.cluster, args.job_id)
    refresh_chatgpt_turn_state()
    state = update_state(
        args.state_file,
        watcher_pid=os.getpid(),
        daemon_started_at=time.time(),
        watch_status="running",
        monitoring_enabled=True,
        poll_seconds=args.poll,
    )

    if state.get("wake_status") == "delivered":
        update_state(args.state_file, monitoring_enabled=False, watch_status="done", stopped_at=time.time())
        return 0
    if state.get("wake_status") == "uncertain":
        update_state(args.state_file, monitoring_enabled=False, watch_status="manual-review", stopped_at=time.time())
        return 3

    tag = target["project_tag"]
    desc = target["description"]

    if not state.get("slack_started_sent"):
        ok = post_slack(tag, args.cluster, "STARTED", args.job_id, desc, None)
        state = update_state(
            args.state_file,
            slack_started_sent=bool(ok),
            slack_started_last_attempt_at=time.time(),
        )

    terminal_event = state.get("terminal_event")
    slurm_state = state.get("slurm_state")
    exit_code = state.get("exit_code")
    missing_count = int(state.get("missing_count") or 0)
    missing_limit = int(os.environ.get("CHATTY_JOB_MISSING_LIMIT", "30"))

    while not terminal_event:
        refresh_chatgpt_turn_state()
        slurm_state, exit_code, source = query_state(args.cluster, args.job_id)
        if source.startswith("missing:"):
            missing_count += 1
        elif not source.startswith("remote-error:"):
            missing_count = 0

        state = update_state(
            args.state_file,
            last_polled_at=time.time(),
            last_slurm_state=slurm_state,
            last_slurm_source=source,
            missing_count=missing_count,
        )

        if slurm_state in TERMINAL_SUCCESS:
            terminal_event = "FINISHED"
            break
        if slurm_state in TERMINAL_FAILURES:
            terminal_event = "FAILED"
            break

        if source.startswith("missing:") and missing_count >= missing_limit:
            update_state(
                args.state_file,
                monitoring_enabled=False,
                watch_status="job-not-found",
                stopped_at=time.time(),
            )
            if not state.get("slack_watch_error_sent"):
                ok = post_slack(
                    tag, args.cluster, "WATCH_ERROR", args.job_id, desc, None
                )
                update_state(args.state_file, slack_watch_error_sent=bool(ok))
            print(
                f"JOB_NOT_FOUND cluster={args.cluster} job={args.job_id} "
                f"missing_count={missing_count}",
                file=sys.stderr,
                flush=True,
            )
            return 4

        if slurm_state is None:
            print(
                f"STATUS_UNAVAILABLE cluster={args.cluster} job={args.job_id} "
                f"source={source} missing_count={missing_count}",
                flush=True,
            )
        else:
            print(
                f"STATUS cluster={args.cluster} job={args.job_id} "
                f"state={slurm_state} source={source}",
                flush=True,
            )
        time.sleep(args.poll)

    code = (exit_code or "").split(":", 1)[0]
    state = update_state(
        args.state_file,
        terminal_event=terminal_event,
        slurm_state=slurm_state,
        exit_code=exit_code,
        terminal_detected_at=state.get("terminal_detected_at") or time.time(),
        wake_status=state.get("wake_status") or "pending",
        watch_status="terminal",
    )
    print(
        f"FINAL cluster={args.cluster} job={args.job_id} "
        f"state={slurm_state} exit={exit_code or ''}",
        flush=True,
    )

    if not state.get("slack_terminal_sent"):
        ok = post_slack(
            tag, args.cluster, terminal_event, args.job_id, desc, code or None
        )
        state = update_state(
            args.state_file,
            slack_terminal_sent=bool(ok),
            slack_terminal_last_attempt_at=time.time(),
        )

    if state.get("wake_status") == "delivered":
        update_state(args.state_file, monitoring_enabled=False, watch_status="done", stopped_at=time.time())
        return 0
    if state.get("wake_status") == "uncertain":
        update_state(args.state_file, monitoring_enabled=False, watch_status="manual-review", stopped_at=time.time())
        return 3

    prompt = build_prompt(target, terminal_event, code or None)
    retry_seconds = float(os.environ.get("CHATTY_WAKE_RETRY_SECONDS", "60"))

    # launchd is intentionally not Accessibility-trusted. It may detect Slurm
    # state, send Slack, and persist the durable terminal event, but only the
    # trusted user-session broker may touch ChatGPT's UI.
    if not ui_context_trusted():
        update_state(
            args.state_file,
            wake_status="retrying",
            watch_status="waiting-ui-broker",
            next_wake_retry_at=time.time() + 5,
            last_wake_error="WAKE_UI_BROKER_REQUIRED: untrusted UI context",
            stopped_at=time.time(),
        )
        print(
            f"WAKE_QUEUED_FOR_UI_BROKER cluster={args.cluster} project={tag} "
            f"job={args.job_id}",
            flush=True,
        )
        return 0
    min_idle = float(os.environ.get("CHATTY_WAKE_MIN_USER_IDLE", "12"))
    attempts = int(state.get("wake_attempts") or 0)

    while True:
        state = read_json(args.state_file)
        if state.get("wake_status") == "delivered":
            update_state(args.state_file, monitoring_enabled=False, watch_status="done", stopped_at=time.time())
            return 0
        if state.get("wake_status") == "uncertain":
            update_state(args.state_file, monitoring_enabled=False, watch_status="manual-review", stopped_at=time.time())
            return 3

        if not state.get("slack_terminal_sent"):
            ok = post_slack(
                tag, args.cluster, terminal_event, args.job_id, desc, code or None
            )
            update_state(
                args.state_file,
                slack_terminal_sent=bool(ok),
                slack_terminal_last_attempt_at=time.time(),
            )

        idle = system_idle_seconds()
        if idle is not None and idle < min_idle:
            update_state(
                args.state_file,
                wake_status="retrying",
                watch_status="waiting-user-idle",
                user_idle_seconds=idle,
                next_wake_retry_at=time.time() + max(15.0, min_idle - idle + 1.0),
                stopped_at=time.time(),
            )
            # Delivery is queue-driven. Exit and let launchd recovery retry later
            # instead of keeping a per-job daemon alive while the user is active.
            return 0

        refresh_chatgpt_turn_state()
        attempts += 1
        update_state(
            args.state_file,
            wake_status="attempting",
            wake_attempts=attempts,
            last_wake_attempt_at=time.time(),
            watch_status="waking",
        )
        cp = wake_once(str(target["thread_id"]), prompt)
        combined = ((cp.stdout or "") + "\n" + (cp.stderr or "")).strip()
        if cp.stdout:
            print(cp.stdout.rstrip(), flush=True)

        if cp.returncode == 0:
            update_state(
                args.state_file,
                monitoring_enabled=False,
                wake_status="delivered",
                wake_delivered_at=time.time(),
                last_wake_error=None,
                watch_status="done",
                stopped_at=time.time(),
            )
            print(
                f"WAKE_OK cluster={args.cluster} project={tag} "
                f"job={args.job_id} event={terminal_event}",
                flush=True,
            )
            return 0

        sent_unverified = (
            "WAKE_ALREADY_ATTEMPTED_UNVERIFIED" in combined
            or "WAKE_ATTEMPTED_NOT_VERIFIED" in combined
        )
        if sent_unverified:
            state = read_json(args.state_file)
            first = float(state.get("sent_unverified_since") or time.time())
            recovery_window = float(
                os.environ.get("CHATTY_WAKE_VERIFY_RECOVERY_WINDOW", "600")
            )
            if not state.get("sent_unverified_since"):
                update_state(args.state_file, sent_unverified_since=first)

            # Safe to call wake.py again: its own ledger is already in
            # "attempted", so it will only look for the exact transcript payload
            # and will never press Send a second time.
            if time.time() - first < recovery_window:
                update_state(
                    args.state_file,
                    wake_status="attempted-unverified",
                    last_wake_error=combined[-4000:],
                    next_wake_retry_at=time.time() + 15,
                    watch_status="verifying-send",
                    stopped_at=time.time(),
                )
                if combined:
                    print(combined, file=sys.stderr, flush=True)
                # Safe to defer: wake.py's ledger prevents a second Send.
                # Recovery will relaunch us to verify the exact transcript payload.
                return 0

            update_state(
                args.state_file,
                monitoring_enabled=False,
                wake_status="uncertain",
                last_wake_error=combined[-4000:],
                watch_status="manual-review",
                stopped_at=time.time(),
            )
            if combined:
                print(combined, file=sys.stderr, flush=True)
            print(
                f"WAKE_UNCERTAIN cluster={args.cluster} project={tag} "
                f"job={args.job_id}; exact payload not observed within "
                f"{recovery_window:g}s",
                file=sys.stderr,
                flush=True,
            )
            return 3

        permanent = (
            "not found in local ChatGPT catalog" in combined
            or "not a normal ChatGPT chat" in combined
            or "has no catalog title" in combined
            or "unsafe duplicate ChatGPT title" in combined
            or "possible duplicate chat title" in combined
        )
        if permanent:
            update_state(
                args.state_file,
                monitoring_enabled=False,
                wake_status="failed",
                wake_failed_at=time.time(),
                last_wake_error=combined[-4000:],
                watch_status="manual-review",
                stopped_at=time.time(),
            )
            post_slack(
                tag, args.cluster, "WAKE_FAILED", args.job_id, desc, None
            )
            print(
                f"WAKE_FAILED cluster={args.cluster} project={tag} "
                f"job={args.job_id} permanent={permanent}",
                file=sys.stderr,
                flush=True,
            )
            return 2

        update_state(
            args.state_file,
            wake_status="retrying",
            last_wake_error=combined[-4000:],
            next_wake_retry_at=time.time() + retry_seconds,
            watch_status="wake-retrying",
            stopped_at=time.time(),
        )
        if combined:
            print(combined, file=sys.stderr, flush=True)
        print(
            f"WAKE_DEFERRED cluster={args.cluster} project={tag} "
            f"job={args.job_id} retry_after={retry_seconds:g}s",
            file=sys.stderr,
            flush=True,
        )
        # Busy chat, draft present, navigation unavailable, etc. are not errors.
        # Persist pending state and let the launchd recovery dispatcher retry.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
