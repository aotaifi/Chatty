from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

HOME = Path.home()
ROOT = HOME / "Chatty/.chatty/slurm_wake_release_20261003_1137_safeguard"
TARGETS = ROOT / "targets"
STATES = ROOT / "states"
LOGS = ROOT / "logs"
CHATGPT_SCOPE = HOME / "Library/Application Support/Codex/sentry/scope_v3.json"
TURN_STATE = ROOT / "chatgpt-turn-state.json"
ROOT.mkdir(parents=True, exist_ok=True)
TARGETS.mkdir(parents=True, exist_ok=True)
STATES.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

TERMINAL_SUCCESS = {"COMPLETED"}
TERMINAL_FAILURES = {
    "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
    "REVOKED", "SPECIAL_EXIT",
}
ACTIVE_STATES = {
    "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED",
    "REQUEUE_FED", "RESIZING", "SUSPENDED", "SIGNALING",
    "STAGE_OUT", "STOPPED",
}


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def update_state(path: Path, **updates) -> dict:
    lock = path.with_suffix(path.suffix + ".lock")
    with lock.open("a+") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        data = read_json(path)
        data.update(updates)
        atomic_json(path, data)
        return data

def _chatgpt_turn_events() -> list[tuple[str, float, str]]:
    """Return retained ChatGPT turn lifecycle events in breadcrumb order."""
    try:
        scope = json.loads(CHATGPT_SCOPE.read_text())
    except Exception:
        return []

    out: list[tuple[str, float, str]] = []
    for b in scope.get("scope", {}).get("breadcrumbs", []):
        if b.get("category") != "electron.net":
            continue
        d = b.get("data") or {}
        if str(d.get("method") or "").upper() != "POST":
            continue
        try:
            status = int(d.get("status_code") or 0)
            ts = float(b.get("timestamp") or 0)
        except Exception:
            continue
        if status < 200 or status >= 400:
            continue
        url = str(d.get("url") or "")
        if url.endswith("/backend-api/f/conversation/prepare"):
            kind = "prepare"
        elif url.endswith("/backend-api/f/conversation"):
            kind = "complete"
        else:
            continue
        key = f"{ts:.6f}|{kind}"
        out.append((key, ts, kind))
    return out


def refresh_chatgpt_turn_state(stale_seconds: float = 7200.0) -> dict:
    """Persist ChatGPT turn depth so short-lived breadcrumb rotation cannot lose it.

    This function is safe to call from register, daemon, recovery, and wake.
    Seen event fingerprints make repeated scans idempotent.
    """
    lock = TURN_STATE.with_suffix(".json.lock")
    now = time.time()
    with lock.open("a+") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        state = read_json(TURN_STATE)
        seen = list(state.get("seen_event_keys") or [])
        seen_set = set(str(x) for x in seen)
        pending = [
            float(x)
            for x in (state.get("pending_prepare_times") or [])
            if isinstance(x, (int, float))
        ]
        processed = 0

        for key, ts, kind in _chatgpt_turn_events():
            if key in seen_set:
                continue
            seen.append(key)
            seen_set.add(key)
            processed += 1
            if kind == "prepare":
                pending.append(ts)
            elif pending:
                pending.pop(0)

        expired = [ts for ts in pending if now - ts > stale_seconds]
        if expired:
            pending = [ts for ts in pending if now - ts <= stale_seconds]

        # Bounded history is enough because the source breadcrumb window is bounded.
        seen = seen[-1024:]
        state.update(
            {
                "schema": 1,
                "updated_at": now,
                "depth": len(pending),
                "pending_prepare_times": pending,
                "seen_event_keys": seen,
                "last_scan_event_count": processed,
                "expired_prepare_count": int(state.get("expired_prepare_count") or 0)
                + len(expired),
            }
        )
        atomic_json(TURN_STATE, state)
        return state


def chatgpt_turns_busy() -> tuple[bool, dict]:
    state = refresh_chatgpt_turn_state()
    return int(state.get("depth") or 0) > 0, state


def run(argv: list[str], timeout: float = 12) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv, text=True, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            argv, 124, exc.stdout or "",
            ((exc.stderr or "") + "\ncommand timed out").strip(),
        )


def ui_context_trusted() -> bool:
    """True only when this process lineage may use macOS Accessibility."""
    helper = Path(__file__).resolve().parent / "ax_trusted"
    try:
        cp = subprocess.run(
            [str(helper)],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return False
    return cp.returncode == 0 and "TRUSTED=true" in cp.stdout


def normalize_state(raw: str | None) -> str | None:
    if not raw:
        return None
    token = raw.strip().split(None, 1)[0].split("+", 1)[0]
    return token or None


def ssh_opts() -> list[str]:
    return [
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        "-o", "ConnectionAttempts=1",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=1",
    ]


def remote(
    cluster: str, command: str
) -> tuple[subprocess.CompletedProcess[str], str]:
    wrapped = f"/usr/bin/timeout 10s /bin/sh -lc {shlex.quote(command)}"
    opts = ssh_opts()

    if cluster == "theorie":
        last: subprocess.CompletedProcess[str] | None = None
        for host in ("ws3", "ws1"):
            cp = run(["/usr/bin/ssh", *opts, host, wrapped], timeout=14)
            # Scheduler rc=1 is still a successful SSH transport (for example,
            # a completed job no longer present in squeue). Only transport/
            # timeout failures should trigger host failover.
            if cp.returncode not in {124, 255}:
                return cp, host
            last = cp
        assert last is not None
        return last, "ws3/ws1"

    if cluster == "paderborn":
        nested = (
            "/usr/bin/timeout 14s /usr/bin/ssh "
            "-o BatchMode=yes -o ConnectTimeout=5 -o ConnectionAttempts=1 "
            "-o ServerAliveInterval=5 -o ServerAliveCountMax=1 "
            f"paderborn {shlex.quote(wrapped)}"
        )
        cp = run(["/usr/bin/ssh", *opts, "ws1", nested], timeout=18)
        return cp, "ws1->paderborn"

    raise ValueError(f"unsupported cluster {cluster!r}")


def parse_sacct(job_id: str, text: str) -> tuple[str | None, str | None]:
    rows: list[tuple[str, str, str | None]] = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue
        jid = parts[0].strip()
        state = normalize_state(parts[1])
        exit_code = parts[2].strip() if len(parts) > 2 else None
        if jid and state:
            rows.append((jid, state, exit_code))

    exact = [r for r in rows if r[0] == job_id]
    if exact:
        return exact[0][1], exact[0][2]

    alloc = [
        r for r in rows
        if "." not in r[0]
        and (r[0].startswith(job_id + "_") or r[0].startswith(job_id + "["))
    ]
    if not alloc:
        return None, None

    for _, state, exit_code in alloc:
        if state in ACTIVE_STATES:
            return state, exit_code
    for _, state, exit_code in alloc:
        if state in TERMINAL_FAILURES:
            return state, exit_code
    if all(state in TERMINAL_SUCCESS for _, state, _ in alloc):
        return "COMPLETED", "0:0"
    return alloc[0][1], alloc[0][2]

def query_state(cluster: str, job_id: str) -> tuple[str | None, str | None, str]:
    q, qvia = remote(cluster, f"squeue -h -j {shlex.quote(job_id)} -o %T")
    if q.returncode == 0 and q.stdout.strip():
        states = [normalize_state(x) for x in q.stdout.splitlines() if x.strip()]
        states = [s for s in states if s]
        if states:
            state = next((s for s in states if s == "RUNNING"), states[0])
            return state, None, "squeue:" + qvia

    # A scheduler-level rc=1 from squeue commonly means the completed job has
    # fallen out of the live queue. That is not a transport failure: consult
    # accounting before deciding the job is missing.
    if q.returncode in {124, 255}:
        return None, None, "remote-error:" + qvia

    a, avia = remote(
        cluster,
        f"sacct -X -j {shlex.quote(job_id)} "
        "--format=JobIDRaw,State,ExitCode -n -P",
    )
    if a.returncode in {124, 255}:
        return None, None, "remote-error:" + avia
    if a.returncode != 0 or not a.stdout.strip():
        return None, None, "missing:" + avia

    state, exit_code = parse_sacct(job_id, a.stdout)
    if state is None:
        return None, None, "missing:" + avia
    return state, exit_code, "sacct:" + avia


def post_slack(
    tag: str,
    cluster: str,
    event: str,
    job_id: str,
    description: str,
    exit_code: str | None,
) -> bool:
    if os.environ.get("CHATTY_DISABLE_SLACK") == "1":
        return True
    msg = f"[Chatty][{tag}] {event} {cluster}:{job_id} — {description}"
    if event == "FAILED" and exit_code:
        msg += f" exit={exit_code.split(':', 1)[0]}"
    payload_hex = json.dumps({"text": msg}).encode().hex()
    remote_shell = (
        "tmp=$(mktemp /tmp/chatty-slack.XXXXXX); "
        f"printf '%s' {shlex.quote(payload_hex)} | /usr/bin/xxd -r -p > \"$tmp\"; "
        "/usr/bin/curl -fsS -X POST -H 'Content-type: application/json' "
        "--data-binary @\"$tmp\" \"$(cat $HOME/.config/chatty/slack_webhook)\" >/dev/null; "
        "rc=$?; rm -f \"$tmp\"; exit $rc"
    )
    opts = [
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=6",
        "-o", "ConnectionAttempts=1",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=1",
    ]
    for host in ("ws1", "ws3"):
        cp = run(["/usr/bin/ssh", *opts, host, remote_shell])
        if cp.returncode == 0:
            return True
    return False


def watcher_alive(state: dict, cluster: str, job_id: str, daemon_path: Path) -> bool:
    try:
        pid = int(state.get("watcher_pid"))
    except (TypeError, ValueError):
        return False
    cp = run(["/bin/ps", "-p", str(pid), "-o", "command="], timeout=5)
    cmd = cp.stdout.strip()
    return (
        cp.returncode == 0
        and str(daemon_path) in cmd
        and f"--cluster {cluster}" in cmd
        and f"--job-id {job_id}" in cmd
    )


def system_idle_seconds() -> float | None:
    cp = run(["/usr/sbin/ioreg", "-c", "IOHIDSystem", "-d", "4", "-r"], timeout=5)
    if cp.returncode != 0:
        return None
    m = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', cp.stdout)
    if not m:
        return None
    return int(m.group(1)) / 1_000_000_000.0
