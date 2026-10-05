"""Wake an exact normal ChatGPT conversation through ChatGPT.app.

Navigation is by the normal-chat URL https://chatgpt.com/c/<thread-id>.
Submission is performed through macOS Accessibility against the composer whose
ancestor AXWebArea URL contains the exact immutable thread id. Titles are
metadata only. No Chrome tab search, no codex:// local/Work thread route, and
no blind Return key.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from common import ui_context_trusted

if len(sys.argv) < 3:
    raise SystemExit("usage: chatty-wake-local.py THREAD_ID PROMPT...")

thread_id = sys.argv[1].strip()
prompt = " ".join(sys.argv[2:]).strip()
if not thread_id or not prompt:
    raise SystemExit("thread id and prompt must be non-empty")

HOME = Path.home()
HERE = Path(__file__).resolve().parent
DB = HOME / ".codex/sqlite/codex-dev.db"
ROOT = HOME / "Chatty/.chatty"
STATE = ROOT / "slurm_wake_release_20261003_1137_safeguard"
STATE.mkdir(parents=True, exist_ok=True)
LOCK = ROOT / "wake-app.lock"
LEDGER = STATE / "wake-ledger.json"
AX = HERE / "ax_target"
AX_PAGE = HERE / "ax_page_ready"
AX_CURRENT_THREAD = HERE / "ax_current_thread"

# Serialize all projects through the one ChatGPT desktop UI.
_lock_fp = LOCK.open("a+")
fcntl.flock(_lock_fp.fileno(), fcntl.LOCK_EX)

# Defense in depth: launchd itself is not Accessibility-trusted on macOS.
# A terminal event remains durable and must be retried by the trusted broker.
if not ui_context_trusted():
    raise SystemExit("WAKE_UI_CONTEXT_UNTRUSTED: trusted user-session broker required")


def catalog_entry() -> tuple[str, str, float | None]:
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
        raise RuntimeError(f"thread {thread_id} not found in local ChatGPT catalog")
    title, source_kind, updated = row
    if source_kind != "chatgpt":
        raise RuntimeError(
            f"thread {thread_id} is source_kind={source_kind!r}, not a normal ChatGPT chat"
        )
    if not title:
        raise RuntimeError(f"thread {thread_id} has no catalog title")

    return str(title), str(source_kind), float(updated) if updated is not None else None




def load_ledger() -> dict:
    try:
        return json.loads(LEDGER.read_text())
    except Exception:
        return {}


def save_ledger(data: dict) -> None:
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, LEDGER)


def run_ax(*args: str, timeout: float = 12) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(AX), *args],
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def parse_state(cp: subprocess.CompletedProcess[str]) -> dict | None:
    if cp.returncode != 0:
        return None
    out: dict[str, str | bool] = {}
    for line in cp.stdout.splitlines():
        if line.startswith("VALUE="):
            out["value"] = line[6:]
        elif line.startswith("SEND="):
            out["send"] = line[5:].strip().lower() == "true"
        elif line.startswith("STOP="):
            out["stop"] = line[5:].strip().lower() == "true"
    if "value" not in out:
        return None
    return out


def get_state() -> dict | None:
    return parse_state(run_ax("get", thread_id))


def page_ready() -> bool:
    for _ in range(5):
        cp = subprocess.run(
            [str(AX_PAGE), thread_id],
            text=True,
            capture_output=True,
            timeout=8,
        )
        if cp.returncode == 0 and "READY" in cp.stdout:
            return True
        time.sleep(0.2)
    return False


def current_thread_id() -> str | None:
    cp = subprocess.run(
        [str(AX_CURRENT_THREAD)], text=True, capture_output=True, timeout=8, check=False
    )
    if cp.returncode != 0:
        return None
    for line in cp.stdout.splitlines():
        if line.startswith("THREAD_ID="):
            return line.split("=", 1)[1].strip() or None
    return None


def navigate_exact() -> dict | None:
    if page_ready():
        return get_state()

    current_id = current_thread_id()
    if current_id and current_id != thread_id:
        current = parse_state(run_ax("get", current_id))
        if current is None or current.get("stop", False):
            raise SystemExit(
                "WAKE_UI_BUSY_OTHER_TURN: refusing to navigate away from an active ChatGPT turn"
            )
        current_value = str(current.get("value", ""))
        if not is_empty_composer(current_value):
            raise SystemExit(
                "WAKE_UI_DRAFT_OTHER_CHAT: refusing to navigate away from an unsent user draft"
            )

    subprocess.run(
        ["/usr/bin/open", "-a", "ChatGPT", f"https://chatgpt.com/c/{thread_id}"],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        if page_ready():
            return get_state()
        time.sleep(0.25)
    raise RuntimeError(
        f"exact normal-ChatGPT navigation failed: thread URL {thread_id} did not verify"
    )


def message_present() -> bool:
    cp = run_ax("contains", thread_id, prompt)
    return cp.returncode == 0 and "FOUND=true" in cp.stdout

def is_empty_composer(value: object) -> bool:
    text = str(value or "").strip()
    return text == "" or text == "Ask ChatGPT"


def mark_delivered(ledger: dict, key: str, title: str, recovered: bool = False) -> None:
    entry = ledger.setdefault(key, {})
    entry.update(
        {
            "state": "delivered",
            "thread_id": thread_id,
            "title": title,
            "prompt": prompt,
            "verified_at": time.time(),
        }
    )
    if recovered:
        entry["recovered"] = True
    save_ledger(ledger)


title, _, catalog_before = catalog_entry()
delivery_key = hashlib.sha256((thread_id + "\0" + prompt).encode()).hexdigest()
ledger = load_ledger()
previous = ledger.get(delivery_key)

if previous and previous.get("state") == "delivered":
    print("WAKE_ALREADY_RECORDED", thread_id, delivery_key, flush=True)
    raise SystemExit(0)

# Fast-path for the exact scenario this workflow must tolerate:
# if the target conversation is already visible but its composer is absent,
# ChatGPT is still answering. Do not open Search or disturb the active turn.
if page_ready():
    visible_state = get_state()
    if visible_state is None or visible_state.get("stop", False):
        raise SystemExit("WAKE_TARGET_BUSY_NOW: target chat is still in an active turn")

state = navigate_exact()
if state is None or state.get("stop", False):
    raise SystemExit("WAKE_TARGET_BUSY_NOW: exact target chat is still in an active turn")

if previous and previous.get("state") == "attempted":
    if message_present():
        mark_delivered(ledger, delivery_key, title, recovered=True)
        print("WAKE_RECOVERED_AS_DELIVERED", thread_id, delivery_key, flush=True)
        raise SystemExit(0)
    raise SystemExit(
        "WAKE_ALREADY_ATTEMPTED_UNVERIFIED: refusing to resend an already-submitted wake"
    )

# Durable recovery before Send: verify the actual machine-readable wake text in this exact chat.
if message_present():
    mark_delivered(ledger, delivery_key, title, recovered=True)
    print("WAKE_RECOVERED_AS_DELIVERED", thread_id, delivery_key, flush=True)
    raise SystemExit(0)

ledger[delivery_key] = {
    "state": "pending",
    "thread_id": thread_id,
    "title": title,
    "prompt": prompt,
    "started": previous.get("started", time.time()) if previous else time.time(),
    "catalog_before": catalog_before,
}
save_ledger(ledger)

# Never overwrite a person's draft. If the thread is answering, wait for its
# normal ChatGPT composer to reappear rather than interrupting the active turn.
idle_timeout = float(os.environ.get("CHATTY_WAKE_IDLE_TIMEOUT", "30"))
idle_deadline = time.time() + idle_timeout
while True:
    if not page_ready():
        navigate_exact()
    state = get_state()
    if state is None:
        if time.time() >= idle_deadline:
            raise SystemExit(
                f"WAKE_TARGET_BUSY_TIMEOUT: target chat composer unavailable for {idle_timeout:g}s"
            )
        time.sleep(1)
        continue

    value = str(state.get("value", ""))
    if not is_empty_composer(value) and value.strip() != prompt:
        raise SystemExit(
            "WAKE_BLOCKED_EXISTING_DRAFT: refusing to overwrite target ChatGPT composer"
        )

    if not state.get("stop", False):
        break
    if time.time() >= idle_deadline:
        raise SystemExit(f"WAKE_TARGET_BUSY_TIMEOUT: target chat remained busy for {idle_timeout:g}s")
    time.sleep(1)

# Multi-device guard: require the exact target chat to remain visibly idle
# (empty composer, no active response) continuously before staging anything.
# This gives mobile/web activity time to synchronize into the desktop app.
quiet_seconds = float(os.environ.get("CHATTY_WAKE_CHAT_QUIET_SECONDS", "6"))
quiet_since = time.time()
while time.time() - quiet_since < quiet_seconds:
    check = get_state()
    if check is None:
        quiet_since = time.time()
        time.sleep(0.5)
        continue
    check_value = str(check.get("value", ""))
    if check.get("stop", False) or (
        not is_empty_composer(check_value) and check_value.strip() != prompt
    ):
        if not is_empty_composer(check_value) and check_value.strip() != prompt:
            raise SystemExit(
                "WAKE_BLOCKED_EXISTING_DRAFT: refusing to overwrite target ChatGPT composer"
            )
        quiet_since = time.time()
    time.sleep(0.5)

state = get_state()
if state is None or state.get("stop", False):
    raise SystemExit("WAKE_TARGET_CHANGED_DURING_QUIET_WINDOW")

# If a previous attempt already staged this exact prompt, reuse it. Otherwise stage it.
if str(state.get("value", "")).strip() != prompt:
    cp = run_ax("set", thread_id, prompt)
    if cp.returncode != 0:
        raise SystemExit("WAKE_COMPOSER_SET_FAILED: " + (cp.stderr or cp.stdout).strip())

# AX value changes are asynchronous. Require exact read-back and a target-context Send button.
ready_deadline = time.time() + 8
ready = None
while time.time() < ready_deadline:
    ready = get_state()
    if (
        ready is not None
        and str(ready.get("value", "")).strip() == prompt
        and ready.get("send", False)
        and not ready.get("stop", False)
    ):
        break
    time.sleep(0.25)
else:
    # Only clear if it is still exactly our own unsent prompt.
    current = get_state()
    if current is not None and str(current.get("value", "")).strip() == prompt:
        run_ax("clear-if", thread_id, prompt)
    raise SystemExit("WAKE_COMPOSER_VERIFY_FAILED: exact prompt/send state not reached")

# Final synchronization guard: after staging, require the exact prompt/send
# state to stay unchanged briefly. If a mobile/web turn starts during this
# window, Stop appears or the composer state changes and we abort without Send.
staged_quiet = float(os.environ.get("CHATTY_WAKE_STAGED_QUIET_SECONDS", "2"))
stable_since = time.time()
while time.time() - stable_since < staged_quiet:
    current = get_state()
    if (
        current is None
        or str(current.get("value", "")).strip() != prompt
        or not current.get("send", False)
        or current.get("stop", False)
    ):
        if (
            current is not None
            and str(current.get("value", "")).strip() == prompt
        ):
            run_ax("clear-if", thread_id, prompt)
        raise SystemExit("WAKE_TARGET_CHANGED_BEFORE_SEND")
    time.sleep(0.25)

cp = run_ax("press-send", thread_id, prompt)
if cp.returncode != 0:
    raise SystemExit("WAKE_SEND_FAILED: " + (cp.stderr or cp.stdout).strip())

submitted_at = time.time()
ledger = load_ledger()
entry = ledger.setdefault(delivery_key, {})
entry.update(
    {
        "state": "attempted",
        "thread_id": thread_id,
        "title": title,
        "prompt": prompt,
        "submitted_at": submitted_at,
    }
)
save_ledger(ledger)

# Strong verification: only the exact injected payload appearing in the
# target transcript counts as delivered. A catalog timestamp is not sufficient:
# an unrelated manual message in the same thread could move that timestamp.
verify_deadline = time.time() + 35
while time.time() < verify_deadline:
    if message_present():
        mark_delivered(ledger, delivery_key, title)
        print("WAKE_SUBMITTED", thread_id, delivery_key, "content=true", flush=True)
        raise SystemExit(0)
    time.sleep(0.5)

# Keep "attempted" permanently if verification is unavailable. A future invocation may
# recover it as delivered, but will never resend the same wake automatically.
raise SystemExit(
    "WAKE_ATTEMPTED_NOT_VERIFIED: send was pressed; refusing automatic resend"
)
