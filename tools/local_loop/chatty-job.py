#!/usr/bin/env python3
"""Durable long-run launcher/watcher for Chatty research.

This module records long-running work under ~/Chatty/jobs. It deliberately does
NOT wake ChatGPT, open browser tabs, inject prompts, or use Computer Use/CUA.

Notification policy:
- Agents must report running jobs in chat before ending their turn.
- Email notification is not enabled until a reliable noninteractive mail
  transport is configured and tested on this Mac.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

REPO = Path.home() / "Chatty"
JOBS = REPO / "jobs"


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def new_job(task: str, mode: str, extra: dict) -> tuple[Path, dict]:
    JOBS.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = f"{stamp}-{os.getpid()}"
    job_dir = JOBS / job_id
    job_dir.mkdir(parents=True)
    data = {
        "id": job_id,
        "mode": mode,
        "task": task,
        "cwd": os.getcwd(),
        "status": "running",
        "started_at": now_iso(),
        "notification": "chat_report_required",
        **extra,
    }
    atomic_json(job_dir / "job.json", data)
    return job_dir, data


def mark(job_dir: Path, data: dict, **updates) -> None:
    data.update(updates)
    atomic_json(job_dir / "job.json", data)


def launch_caffeinate(pid: int):
    if os.uname().sysname != "Darwin":
        return None
    try:
        return subprocess.Popen(
            ["/usr/bin/caffeinate", "-i", "-w", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None


def finish_without_auto_notification(job_dir: Path, data: dict) -> None:
    """Persist completion. Never trigger ChatGPT/browser/CUA."""
    (job_dir / "DONE").write_text(now_iso() + "\n")
    (job_dir / "NOTIFICATION_NOT_SENT").write_text(
        "Automatic email notification is not configured on this Mac.\n"
        "The launching agent must have reported this job in chat.\n"
    )
    mark(
        job_dir,
        data,
        notification="not_sent_email_transport_unconfigured",
        notification_finished_at=now_iso(),
    )


def cmd_run(args) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("chatty-run requires a command after --")

    job_dir, data = new_job(
        args.task,
        "run",
        {"command": command, "command_display": shlex.join(command)},
    )
    out = (job_dir / "stdout.log").open("wb")
    err = (job_dir / "stderr.log").open("wb")
    try:
        proc = subprocess.Popen(command, cwd=args.cwd, stdout=out, stderr=err)
        mark(job_dir, data, pid=proc.pid)
        caffeine = None if args.allow_sleep else launch_caffeinate(proc.pid)
        rc = proc.wait()
        if caffeine is not None:
            try:
                caffeine.wait(timeout=2)
            except Exception:
                pass
    except BaseException as exc:
        mark(
            job_dir,
            data,
            status="launcher_error",
            finished_at=now_iso(),
            launcher_error=repr(exc),
        )
        finish_without_auto_notification(job_dir, data)
        raise
    finally:
        out.close()
        err.close()

    mark(
        job_dir,
        data,
        status="completed" if rc == 0 else "failed",
        exit_code=rc,
        finished_at=now_iso(),
    )
    finish_without_auto_notification(job_dir, data)
    print(job_dir)
    return rc


def process_signature(pid: int) -> tuple[str, str] | None:
    try:
        cp = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    line = cp.stdout.strip()
    if cp.returncode != 0 or not line:
        return None
    parts = line.split(None, 5)
    if len(parts) < 6:
        return (line, "")
    return (" ".join(parts[:5]), parts[5])


def cmd_watch(args) -> int:
    sig = process_signature(args.pid)
    if sig is None:
        raise SystemExit(f"PID {args.pid} is not running")
    start_sig, command = sig
    job_dir, data = new_job(
        args.task,
        "watch",
        {
            "watched_pid": args.pid,
            "watched_process_start": start_sig,
            "watched_command": command,
        },
    )
    caffeine = None if args.allow_sleep else launch_caffeinate(args.pid)
    while True:
        time.sleep(args.poll)
        current = process_signature(args.pid)
        if current is None or current[0] != start_sig:
            break
    if caffeine is not None:
        try:
            caffeine.terminate()
        except Exception:
            pass

    mark(
        job_dir,
        data,
        status="exited",
        exit_code=None,
        finished_at=now_iso(),
    )
    finish_without_auto_notification(job_dir, data)
    print(job_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="sub", required=True)

    r = sub.add_parser("run")
    r.add_argument("--task", required=True)
    r.add_argument("--cwd", default=None)
    r.add_argument("--allow-sleep", action="store_true")
    r.add_argument("command", nargs=argparse.REMAINDER)
    r.set_defaults(func=cmd_run)

    w = sub.add_parser("watch")
    w.add_argument("--task", required=True)
    w.add_argument("--pid", required=True, type=int)
    w.add_argument("--poll", type=float, default=2.0)
    w.add_argument("--allow-sleep", action="store_true")
    w.set_defaults(func=cmd_watch)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))
