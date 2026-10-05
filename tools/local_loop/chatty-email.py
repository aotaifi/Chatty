#!/usr/bin/env python3
"""Send Chatty watcher notifications through macOS Mail via Automator.

This uses Apple's built-in authenticated Mail account. It never opens ChatGPT,
Chrome, or CUA, and it does not store mail credentials.
"""
from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "mail_template.workflow"
RECIPIENT_FILE = Path.home() / "Chatty/.chatty/notify_email"


def recipient() -> str:
    value = RECIPIENT_FILE.read_text().strip()
    if not value:
        raise RuntimeError(f"empty notification address: {RECIPIENT_FILE}")
    return value


def send(subject: str, body: str) -> None:
    if not TEMPLATE.exists():
        raise RuntimeError(f"missing mail workflow template: {TEMPLATE}")
    with tempfile.TemporaryDirectory(prefix="chatty-mail-") as td:
        wf = Path(td) / "Chatty Notify.workflow"
        shutil.copytree(TEMPLATE, wf)
        doc = wf / "Contents/document.wflow"
        with doc.open("rb") as f:
            p = plistlib.load(f)
        params = p["actions"][0]["action"]["ActionParameters"]
        params["account"] = "defaultAccount"
        params["toAddresses"] = recipient()
        params["subject"] = subject
        params["message"] = body
        params["ccAddresses"] = ""
        params["bccAddresses"] = ""
        with doc.open("wb") as f:
            plistlib.dump(p, f, fmt=plistlib.FMT_XML, sort_keys=False)

        cp = subprocess.run(
            ["/usr/bin/automator", str(wf)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
        if cp.returncode != 0:
            raise RuntimeError(
                f"Automator mail failed rc={cp.returncode}: {cp.stdout.strip()}"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--body", required=True)
    args = ap.parse_args()
    send(args.subject, args.body)
    print("EMAIL_SENT", recipient())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
