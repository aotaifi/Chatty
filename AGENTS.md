# Chatty agent instructions

This repository is the durable source of truth for long-running research work.

## Long-running research jobs

Do not keep a ChatGPT turn open merely to wait for a simulation, benchmark, download, or other long-running Mac process.

1. If you leave a process running, report it in the chat before ending the turn. Include the PID, what is running, where stdout/stderr/results are written, and what should be checked when it finishes.
2. For a new process, you may launch it through `~/Chatty/tools/local_loop/chatty-run`.
3. For a process that is already running, you may attach `~/Chatty/tools/local_loop/chatty-watch --pid <pid>`.
4. Launch the wrapper detached (`nohup ... &`) so it survives the current ChatGPT turn.
5. A watcher may notify the user by email when the run finishes.
6. A watcher must NOT wake ChatGPT, open ChatGPT/Chrome tabs, inject prompts, use Computer Use/CUA, or try to continue the conversation automatically.
7. The next research agent should resume from the durable files on the Mac after the user returns to ChatGPT.

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
