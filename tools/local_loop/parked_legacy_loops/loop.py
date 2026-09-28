from pathlib import Path
import time, datetime

base = Path.home() / "Chatty/tools/local_loop"
inbox = base / "inbox.txt"
outbox = base / "outbox.txt"
seen = None

while True:
    try:
        if inbox.exists():
            msg = inbox.read_text(errors="ignore")
            if msg and msg != seen:
                seen = msg
                ts = datetime.datetime.now().isoformat(timespec="seconds")
                outbox.write_text(f"[{ts}] local-loop received: {msg.strip()}\n")
        time.sleep(0.2)
    except Exception as e:
        outbox.write_text(f"ERROR: {e}\n")
        time.sleep(1)
