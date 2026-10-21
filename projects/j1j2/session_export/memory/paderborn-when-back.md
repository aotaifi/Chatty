---
name: paderborn-when-back
description: Move J1-J2 GPU work to Paderborn H100s once its maintenance ends (ws1 A40s are slow)
metadata:
  node_type: memory
  type: project
  originSessionId: 538ff1fa-50fd-457b-aa22-c58f295c8d22
  modified: 2026-10-07T16:08:43.957Z
---

Paderborn GPUs were down for maintenance from 2026-10-05 (expected back ~Oct 9-12). Once back, start running GPU-heavy J1-J2 jobs there instead of the slow, congested ws1 A40s (PI request 2026-10-07).

**Why:** saves time; A40 queues on ws1 stalled jobs for hours.
**How to apply:** at session start after ~Oct 9, check Paderborn (`ssh ws1`, then `ssh -o HostName=login1 -o HostKeyAlias=login2 paderborn sinfo` if login2 is still down; VPN runs on ws1). Note also in projects/j1j2/RESEARCH_MAP.md section 5b.
