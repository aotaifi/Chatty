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
