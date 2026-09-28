# Chatty long-run jobs

This directory contains the Mac-side helpers for durable research jobs.

## Current policy

The previous same-thread ChatGPT wake bridge is **parked**. Active research
agents must not wake ChatGPT, open ChatGPT/Chrome tabs, inject prompts, or use
CUA/Computer Use when a run finishes.

Instead:

1. The agent launches or attaches a durable watcher.
2. Before ending its turn, the agent reports the PID, task, output paths, and
   what should be checked after completion.
3. The watcher records completion under `~/Chatty/jobs/<job-id>/`.
4. The user returns to ChatGPT when convenient; the next agent resumes from the
   durable files.

## Launch a new long run

```bash
nohup ~/Chatty/tools/local_loop/chatty-run \
  --task "Analyze the finite-size scaling and update STATUS.md" \
  --cwd ~/Chatty/projects/j1j2 \
  -- python3 scripts/benchmark.py \
  > /tmp/chatty-launch.log 2>&1 &
```

## Attach to an existing process

```bash
nohup ~/Chatty/tools/local_loop/chatty-watch \
  --pid 12345 \
  --task "Inspect the result and continue the scaling analysis" \
  > /tmp/chatty-watch.log 2>&1 &
```

An attached watcher cannot know the target process's exit code because it is not
the parent; it records `exit_code: null`.

## Job layout

```
~/Chatty/jobs/<job-id>/
  job.json
  stdout.log              # chatty-run only
  stderr.log              # chatty-run only
  DONE
  NOTIFICATION_NOT_SENT
```

## Email notification status

Automatic email is intentionally **not enabled yet**. Tests on 2026-09-28 found:

- macOS Postfix/sendmail: message was rejected by the destination server
  because the Mac has no valid reverse DNS;
- Apple Mail scripting: hangs even for basic outgoing-message operations;
- Mail UI/key injection: did not produce a delivered message;
- no noninteractive SMTP/Google CLI credential is exposed to the watcher.

Do not claim email delivery works until a real completion email is verified in
the recipient mailbox. A dedicated authenticated relay/app password or another
noninteractive notification transport is still required.

## Parked wake implementation

The old same-thread wake/CUA experiments are archived under
`tools/local_loop/parked_same_thread_wake/` for reference only. They are not
part of the active agent workflow.
