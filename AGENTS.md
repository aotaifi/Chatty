# Chatty agent instructions

This repository is the durable source of truth for long-running research work.

## Long-running research jobs

Do not keep a ChatGPT turn open merely to wait for a simulation, benchmark, download, or other long-running Mac process.

1. Bind the current ChatGPT conversation once with:
   `~/Chatty/tools/local_loop/chatty-bind-thread <conversation-id>`
2. For a new process, launch it through `~/Chatty/tools/local_loop/chatty-run`.
3. For a process that is already running, attach `~/Chatty/tools/local_loop/chatty-watch --pid <pid>`.
4. Launch the wrapper detached (`nohup ... &`) so it survives the current ChatGPT turn.
5. Give `--task` a concrete continuation instruction: what output to inspect, what decision/test to make next, and what durable files should be updated.
6. On completion the watcher writes `WAKE_PENDING` under `~/Chatty/jobs/<job-id>/` and attempts to wake the SAME ChatGPT conversation through ChatGPT.app. The fresh turn should inspect the job and continue without asking the user to type "go".
7. `WAKE_SUBMITTED` means delivery succeeded. `WAKE_FAILED` only means the latest attempt failed; while `WAKE_PENDING` remains, the recovery worker must keep retrying.
8. Ensure the recovery worker is installed with `~/Chatty/tools/local_loop/chatty-install-recovery`. Use `chatty-retry-wake <job-id>` for an immediate targeted retry.
9. Never put a long wait loop inside one CUA JavaScript call: node_repl has a ~30 s per-call execution limit. Poll from the outer Python process using short CUA calls.

See `tools/local_loop/README.md` for usage and details.

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
