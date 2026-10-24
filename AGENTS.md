# Chatty agent instructions

This repository is the durable source of truth for long-running research work.

## Long-running research jobs

Do not keep a ChatGPT turn open merely to wait for a simulation, benchmark,
download, or other long-running Mac process.

1. For a new long process, launch it through
   `~/Chatty/tools/local_loop/chatty-run`.
2. For a process that is already running, attach
   `~/Chatty/tools/local_loop/chatty-watch --pid <pid>`.
3. Launch the wrapper detached (`nohup ... &`) so it survives the current
   ChatGPT turn.
4. The watcher sends an email immediately when monitoring starts and another
   when the process exits.
5. Before ending the turn, report in chat what is running: PID, task, job
   directory, output/result paths, and what should be checked after completion.
6. Then end the turn. Do NOT poll or wait for the process from ChatGPT.
7. The user will receive the completion email and return to the research chat to
   tell the agent to proceed. At that point read the durable files and continue.
8. Watchers must NOT wake ChatGPT, open ChatGPT/Chrome tabs, inject prompts, use
   Computer Use/CUA, or continue the conversation automatically.

See `tools/local_loop/README.md` for the watcher interface.

## J1-J2 project

1. Read `projects/j1j2/STATUS.md` first.
2. Read `projects/j1j2/FAILED_ROUTES.md` before reviving an old idea.
3. Treat files on this Mac as persistent working state; do not rely on a chat sandbox.
4. Keep decisive numerical outputs under `projects/j1j2/results/`.
5. Keep reproducible tests under `projects/j1j2/experiments/` or `scripts/`.
6. After a decisive result, update STATUS.md before ending the session.
7. Record why a route failed, not just that it failed.
8. Do not overwrite established results without evidence.
9. Cluster access is intentionally not configured here yet.

## Where everything lives now (changed 2026-10-24, PI decision: one Chatty, no clutter)

All Chatty material is under ONE directory, `/project/theorie/a/A.Otaifi/Chatty/`:
- `Chatty/` itself: this git repository (GitHub `aotaifi/Chatty`). Only tracked files go to GitHub.
- `Chatty/runs/<name>/`: every former compute directory `chatty_<name>` (from the project root AND from `/home/a/A.Otaifi`),
  e.g. `/home/a/A.Otaifi/chatty_ll6` is now `Chatty/runs/ll6`, `chatty_compress6` is `Chatty/runs/compress6`. Excluded from git
  locally (`.git/info/exclude`), so they are never pushed. `ChattyRun` is `runs/ChattyRun`, `chatty_slurm_wake_tests` is
  `runs/slurm_wake_tests`.
- `Chatty/admin/`: former `~/chatty_admin` (workspace scripts `ws_*.py`, design and consult notes).
- `Chatty/organize/`: the git worktree for branch `j1j2-organize` (former `Chatty-organize`). `~/Chatty-organize` is kept as a
  symlink to it, so the commands in CLAUDE.md still work.
- Old-path to new-path table for every moved directory: `Chatty/runs/PATHS_MOVED.tsv`. Copy log: `Chatty/runs/MERGE_LOG_20261024.txt`.

Consequences you must handle:
1. The docs under `projects/j1j2/` (about 1000 references) and sbatch/launch scripts inside the run dirs still cite the OLD
   absolute paths such as `/home/a/A.Otaifi/chatty_ll6/...` or `/project/theorie/a/A.Otaifi/chatty_compress6/...`. No symlinks were left at
   the old places. Translate with PATHS_MOVED.tsv when you read an old log; use the new paths in everything you write from now on.
   Resubmitting an old launcher needs its `cd` and file paths edited first.
2. `chatty_signlearn` existed twice (home and `chatty_archive/chatty_signlearn`), byte-identical (checked with `diff -rq`); only the copy
   `Chatty/runs/archive/chatty_signlearn` remains.
3. Seven names existed in both places (compress6, itfit, locality8, rbmpp6, stall6, writeback, writeback_tail) with NO common file paths;
   their contents were merged into one `runs/<name>/`.
4. Nothing under `runs/` or `admin/` is in git. Keep decisive numbers in `projects/j1j2/results/` as before.

### Addendum (2026-10-24, same day)
- Also moved into `Chatty/`: `a40slice` -> `Chatty/runs/a40slice` (A40 GPU smoke-test setup cited by `loop_8x8` and `signs_8x8` launchers),
  and the two older J1-J2 projects `sign_prob` -> `Chatty/sign_prob` (NQS-guided fixed-node benchmark, active until 2026-10-23) and
  `J1J2_codex` -> `Chatty/J1J2_codex` (git repo, GitLab `A.Otaifi/j1j2_sign`). All three are excluded from the Chatty git locally.
  Sessions started inside `Chatty/sign_prob` now also load Chatty's CLAUDE.md from the parent directory.
- Symlinks inside the moved run directories that pointed at the old absolute paths (1,368 of 2,695) were rewritten to the new
  locations (2026-10-24, script run once; targets verified to exist). Symlinks and file contents were the only things touched.
- Log files, sbatch scripts and notes still contain the OLD paths as text; translate with `Chatty/runs/PATHS_MOVED.tsv`.
