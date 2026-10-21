---
name: ssh-waiter-loops
description: "Never poll ws1 with `until ssh ... pgrep -f \"...\"` loops - remote pgrep -f matches its own command line, loop never ends and hammers ws1"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 538ff1fa-50fd-457b-aa22-c58f295c8d22
  modified: 2026-10-19T18:02:32.757Z
---

Background waiters of the form `until ssh ws1 '! pgrep -f "codex exec"'; do sleep 30; done` never terminate: the remote `pgrep -f` matches the ssh-spawned shell whose command line contains the pattern. Three such loops ran ~6 days (2026-10-13..19), retrying every 30 s, and likely got ws1 SSH refused.

**Why:** ws1 started closing every SSH connection; all cluster work (bond-field verdict, 12x12 pilot) stalled.

**How to apply:** wait on a marker file written by the remote job (`test -s done.flag`), or use `pgrep -x`/a pidfile; always add a max iteration count or timeout; stop waiters as soon as their output is fetched. Related: [[cluster-cleanup-rule]].
