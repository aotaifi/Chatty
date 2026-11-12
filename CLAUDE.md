# CLAUDE.md — working rules for Claude on the Chatty repository

The J1-J2 project's rules and state: read projects/j1j2/CLAUDE.md (the PI's agreement; it also holds the research rules
that used to stand here, folded in 2026-11-12) and projects/j1j2/PLAN.md first. Rules for all the PI's projects:
/project/theorie/a/A.Otaifi/CLAUDE.md. AGENTS.md is the ChatGPT-loop agent's file; only its section "Where everything lives now"
applies to Claude.

## Operations
- Work in the git worktree ~/Chatty-organize (branch j1j2-organize); never edit ~/Chatty directly (another agent's uncommitted state). Commit, `git fetch && git merge --ff-only origin/main`, `git push origin HEAD:main`, `git -C ~/Chatty merge --ff-only origin/main`. Always push. No files > 5 MB in git.
- Cluster ws1 (LMU theorie; if ws1 = th-ws-7010m52 refuses SSH, use th-ws-7010m53, m47 or m42: same home, /project and Slurm; poll sparingly, never `until ssh ... pgrep -f` loops): full A40 `--gres=gpu:a40:1` (cip,inter) or RTX 2080 Ti `--partition=inter --gres=gpu:rtx2080ti:1`; V100 fails; a40-NNgb vGPU slices work only with the jax 0.8.2 venv (projects/j1j2/cluster/A40_SLICES_HOWTO.md; often idle); ViT at full fp32. Large files on /project, not /home. GPU placement: the card that finishes first (cluster-jobs skill; Paderborn login1 if login2 is down).
- Never run chatty-slurm-watch or tools/local_loop watchers; whoever submits a job owns it and waits without polling
  (/project/theorie/a/A.Otaifi/WORKER.md, root CLAUDE.md §5). Never print credentials.
- Delete run data on ws1 only if the quota is critical; other campaigns' data: ask their owner session (e.g. ruby_runs).

## Directory layout changed 2026-10-24
Read the section "Where everything lives now" in AGENTS.md first. Short form: all former `chatty_*` dirs are `Chatty/runs/<name>/`,
`~/chatty_admin` is `Chatty/admin/` (the "scripts in ws1:~/chatty_admin" above means that directory now), and this worktree lives at
`Chatty/organize/` with `~/Chatty-organize` kept as a symlink. Old-to-new paths: `Chatty/runs/PATHS_MOVED.tsv`.
