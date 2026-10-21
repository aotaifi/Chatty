---
name: cluster-cleanup-rule
description: When Claude may delete old/stale run data on ws1 (quota-critical only) and when to ask the owning session instead
metadata:
  node_type: memory
  type: feedback
  originSessionId: 538ff1fa-50fd-457b-aa22-c58f295c8d22
  modified: 2026-10-06T16:32:09.718Z
---

On ws1 (LMU theorie), delete old/stale run data only when the quota is exceeded or critical enough that our runs would fail (lfs quota on /project/theorie: files or GB near the soft limit 475k / 475 GB; hard 500k / 500 GB). Then remove our own campaign's stale intermediate runs first, keeping anything a result depends on. If the space belongs to another campaign (e.g. the Ruby campaign), ask that session (ruby_runs) to clean up instead of deleting it.

**Why:** user's standing instruction (2026-10-06); deleting run data is otherwise irreversible and off-limits during /takeover.
**How to apply:** check `lfs quota -h -u $USER /project/theorie` at milestones; below critical, never delete — message the campaign owner if cleanup is wanted.
