---
name: j1j2-paper-on-overleaf
description: J1-J2 paper is edited on Overleaf (git remote), the Chatty repo only holds data and scripts
metadata:
  type: project
---

Since 2026-10-05 the J1-J2 FN/Krylov paper is edited on Overleaf, git remote
https://git.overleaf.com/6ac3b1e870c49911df02fa6a (the user imported it there).
The Chatty repo (projects/j1j2, branch j1j2-organize) stands for data and scripts only;
paper/ in the repo is no longer the editing copy.

**Why:** the user decided to write the paper themselves on Overleaf.
**How to apply:** make paper edits in a local clone of the Overleaf repo (e.g. ~/overleaf-j1j2),
pull before editing, push after; regenerate figures from the repo scripts and copy them over.
Overleaf git needs an Overleaf git token — never enter it; ask the user to set up the credential.
