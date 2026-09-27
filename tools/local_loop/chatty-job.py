#!/usr/bin/env python3
"""Durable long-run launcher/watcher for Chatty research.

A finished run writes durable state under ~/Chatty/jobs and wakes the exact
ChatGPT conversation that launched/owns the work.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import fcntl
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path.home() / "Chatty"
TOOLS = REPO / "tools/local_loop"
JOBS = REPO / "jobs"
STATE = REPO / ".chatty"
DEFAULT_THREAD = STATE / "default_thread_id"
WAKE = TOOLS / "chatty-wake-browser.py"


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def resolve_thread(explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    env = os.environ.get("CHATTY_THREAD_ID", "").strip()
    if env:
        return env
    if DEFAULT_THREAD.exists():
        value = DEFAULT_THREAD.read_text().strip()
        if value:
            return value
    raise SystemExit(
        "No ChatGPT thread configured. Run: chatty-bind-thread <conversation-id> "
        "or pass --thread <conversation-id>."
    )


def new_job(task: str, thread_id: str, mode: str, extra: dict) -> tuple[Path, dict]:
    JOBS.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = f"{stamp}-{os.getpid()}"
    job_dir = JOBS / job_id
    job_dir.mkdir(parents=True)
    data = {
        "id": job_id,
        "mode": mode,
        "task": task,
        "thread_id": thread_id,
        "cwd": os.getcwd(),
        "status": "running",
        "started_at": now_iso(),
        **extra,
    }
    atomic_json(job_dir / "job.json", data)
    return job_dir, data


def mark(job_dir: Path, data: dict, **updates) -> None:
    data.update(updates)
    atomic_json(job_dir / "job.json", data)


def wake_message(job_dir: Path, data: dict) -> str:
    code = data.get("exit_code")
    exit_text = "unknown (attached watcher)" if code is None else str(code)
    return (
        "[CHATTY_JOB_DONE]\n"
        f"job_id: {data['id']}\n"
        f"job_dir: {job_dir}\n"
        f"task: {data['task']}\n"
        f"status: {data['status']}\n"
        f"exit_code: {exit_text}\n"
        "A long-running research job finished. Continue in THIS conversation. "
        "Use Remote Desktop Commander to read job.json, stdout.log/stderr.log "
        "and any result files in the job/project directories. Analyze the result, "
        "update durable project state when appropriate, and continue the research "
        "without waiting for the user to type 'go'. If another long run is needed, "
        "launch it with chatty-run or chatty-watch targeting this same thread."
    )


def wake(job_dir: Path, data: dict, retries: int = 1) -> bool:
    """Attempt same-thread delivery.

    WAKE_PENDING is durable intent to deliver. WAKE_FAILED means the most
    recent delivery attempt failed, not that the job is abandoned. A recovery
    worker can retry it later.
    """
    submitted = job_dir / "WAKE_SUBMITTED"
    pending = job_dir / "WAKE_PENDING"
    failed = job_dir / "WAKE_FAILED"
    if submitted.exists():
        pending.unlink(missing_ok=True)
        failed.unlink(missing_ok=True)
        return True

    pending.touch(exist_ok=True)
    prompt = wake_message(job_dir, data)
    wake_log = job_dir / "wake.log"
    lock_path = job_dir / "wake.lock"

    with lock_path.open("a+") as lockf:
        try:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False

        caffeine = launch_caffeinate(os.getpid())
        try:
            for attempt in range(1, retries + 1):
                if submitted.exists():
                    pending.unlink(missing_ok=True)
                    failed.unlink(missing_ok=True)
                    return True
                with wake_log.open("a") as log:
                    log.write(f"[{now_iso()}] wake attempt {attempt}\n")
                    log.flush()
                    try:
                        cp = subprocess.run(
                            [sys.executable, str(WAKE), data["thread_id"], prompt],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            # chatty-wake-browser waits across short CUA calls
                            # for up to 15 minutes for the target thread to idle.
                            timeout=1000,
                            check=False,
                        )
                        if cp.returncode == 0:
                            submitted.write_text(now_iso() + "\n")
                            pending.unlink(missing_ok=True)
                            failed.unlink(missing_ok=True)
                            return True
                        log.write(f"wake return code={cp.returncode}\n")
                    except Exception as exc:
                        log.write(f"wake exception={exc!r}\n")
                if attempt < retries:
                    time.sleep(min(60, 5 * attempt))
            failed.write_text(now_iso() + "\n")
            return False
        finally:
            if caffeine is not None:
                try:
                    caffeine.terminate()
                except Exception:
                    pass

def launch_caffeinate(pid: int):
    if sys.platform != "darwin":
        return None
    try:
        return subprocess.Popen(
            ["/usr/bin/caffeinate", "-i", "-w", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None


def cmd_bind(args) -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    DEFAULT_THREAD.write_text(args.thread.strip() + "\n")
    print(f"BOUND {args.thread.strip()}")
    return 0


def cmd_show_thread(args) -> int:
    try:
        print(resolve_thread(args.thread))
        return 0
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2


def cmd_run(args) -> int:
    thread = resolve_thread(args.thread)
    if not args.command:
        raise SystemExit("chatty-run requires a command after --")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("chatty-run requires a command after --")

    job_dir, data = new_job(
        args.task,
        thread,
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
        (job_dir / "DONE").write_text(now_iso() + "\n")
        wake(job_dir, data)
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
    (job_dir / "WAKE_PENDING").touch(exist_ok=True)
    (job_dir / "DONE").write_text(now_iso() + "\n")
    ok = wake(job_dir, data)
    mark(job_dir, data, wake_submitted=ok, wake_finished_at=now_iso())
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
    # lstart is 5 whitespace-delimited fields, remainder is command.
    parts = line.split(None, 5)
    if len(parts) < 6:
        return (line, "")
    return (" ".join(parts[:5]), parts[5])


def cmd_watch(args) -> int:
    thread = resolve_thread(args.thread)
    sig = process_signature(args.pid)
    if sig is None:
        raise SystemExit(f"PID {args.pid} is not running")
    start_sig, command = sig
    job_dir, data = new_job(
        args.task,
        thread,
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
    (job_dir / "WAKE_PENDING").touch(exist_ok=True)
    (job_dir / "DONE").write_text(now_iso() + "\n")
    ok = wake(job_dir, data)
    mark(job_dir, data, wake_submitted=ok, wake_finished_at=now_iso())
    print(job_dir)
    return 0




def cmd_retry(args) -> int:
    """Retry one durable job by id or path."""
    job_dir = Path(args.job)
    if not job_dir.is_absolute():
        job_dir = JOBS / args.job
    if not job_dir.is_dir():
        raise SystemExit(f"job not found: {job_dir}")
    data = json.loads((job_dir / "job.json").read_text())
    (job_dir / "WAKE_PENDING").touch(exist_ok=True)
    ok = wake(job_dir, data, retries=args.retries)
    mark(job_dir, data, wake_submitted=ok, wake_finished_at=now_iso())
    print(f"RETRY job={job_dir.name} delivered={int(ok)}")
    return 0


def cmd_recover(args) -> int:
    """Retry durable completed jobs whose wake was never confirmed."""
    JOBS.mkdir(parents=True, exist_ok=True)
    candidates = []
    for job_dir in sorted(JOBS.iterdir(), key=lambda p: p.name):
        if not job_dir.is_dir():
            continue
        if (job_dir / "WAKE_CANCELLED").exists() or (job_dir / "WAKE_SUBMITTED").exists():
            continue
        if not (job_dir / "WAKE_PENDING").exists():
            # Migrate jobs produced by the pre-WAKE_PENDING watcher.
            if (job_dir / "DONE").exists() and (job_dir / "WAKE_FAILED").exists():
                (job_dir / "WAKE_PENDING").touch()
            else:
                continue
        try:
            data = json.loads((job_dir / "job.json").read_text())
        except Exception:
            continue
        if not data.get("thread_id"):
            continue
        candidates.append((job_dir, data))

    delivered = 0
    attempted = 0
    for job_dir, data in candidates[-args.max_jobs:]:
        attempted += 1
        ok = wake(job_dir, data, retries=1)
        mark(job_dir, data, wake_submitted=ok, wake_finished_at=now_iso())
        delivered += int(ok)
    print(f"RECOVER attempted={attempted} delivered={delivered}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="sub", required=True)

    b = sub.add_parser("bind")
    b.add_argument("thread")
    b.set_defaults(func=cmd_bind)

    s = sub.add_parser("show-thread")
    s.add_argument("--thread")
    s.set_defaults(func=cmd_show_thread)

    r = sub.add_parser("run")
    r.add_argument("--thread")
    r.add_argument("--task", required=True)
    r.add_argument("--cwd", default=None)
    r.add_argument("--allow-sleep", action="store_true")
    r.add_argument("command", nargs=argparse.REMAINDER)
    r.set_defaults(func=cmd_run)

    w = sub.add_parser("watch")
    w.add_argument("--thread")
    w.add_argument("--task", required=True)
    w.add_argument("--pid", required=True, type=int)
    w.add_argument("--poll", type=float, default=2.0)
    w.add_argument("--allow-sleep", action="store_true")
    w.set_defaults(func=cmd_watch)

    q = sub.add_parser("recover")
    q.add_argument("--max-jobs", type=int, default=5)
    q.set_defaults(func=cmd_recover)

    y = sub.add_parser("retry")
    y.add_argument("job")
    y.add_argument("--retries", type=int, default=1)
    y.set_defaults(func=cmd_retry)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))
