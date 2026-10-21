---
name: always-push-chatty
description: User wants commits to aotaifi/Chatty pushed to GitHub main right away, not held locally
metadata:
  type: feedback
---

Push commits for the Chatty repo (aotaifi/Chatty) to GitHub `main` after committing; don't wait for review.

**Why:** the user said "yes pls always push actually" (2026-10-05); GitHub `main` is what Overleaf syncs from, so unpushed work is invisible to them.
**How to apply:** commit in the worktree ~/Chatty-organize (branch j1j2-organize), `git push origin HEAD:main` (fast-forward only; fetch and rebase first if main moved), then `git merge --ff-only origin/main` in ~/Chatty without touching the other agent's uncommitted edits. Never force-push. Keep files > 5 MB out of commits. See [[j1j2-paper-on-overleaf]].
