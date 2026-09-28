# Chatty long-run jobs

This directory contains the Mac-side helpers for durable research jobs.

## Current policy

The previous same-thread ChatGPT wake bridge is **parked**. Active research
agents must not wake ChatGPT, open ChatGPT/Chrome tabs, inject prompts, or use
CUA/Computer Use when a run finishes.

Instead:

1. The agent launches a long job through `chatty-run` or attaches `chatty-watch`.
2. The watcher immediately emails `alyotaifi@gmail.com` that the job started,
   including PID, task, command, and job/result paths.
3. The ChatGPT turn ends. It does not poll or wait for the job.
4. The detached watcher keeps running independently on the Mac.
5. When the process exits, the watcher writes durable completion state and sends
   a second email with status and result/log locations.
6. The user returns to the research chat and says to proceed. The next agent
   reads the durable job/project files and continues.

The mail transport uses Apple's built-in Automator Mail actions through the
already-authenticated macOS Mail account. It does not use ChatGPT, Chrome, CUA,
Postfix, an OpenAI API key, or a stored Gmail password.

## Launch a new long run

```bash
nohup ~/Chatty/tools/local_loop/chatty-run \
  --task "Analyze the finite-size scaling and update STATUS.md" \
  --cwd ~/Chatty/projects/j1j2 \
  -- python3 scripts/benchmark.py \
  > /tmp/chatty-launch.log 2>&1 &
```

The launching agent should report the job PID/task/paths in chat and then end its
turn rather than monitoring it.

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
  email.log
  EMAIL_START_SENT
  EMAIL_START_FAILED      # only if start notification failed
  EMAIL_FINISH_SENT
  EMAIL_FINISH_FAILED     # only if completion notification failed
  DONE
```

## Email transport

Recipient:

```
~/Chatty/.chatty/notify_email
```

Current value: `alyotaifi@gmail.com`.

`chatty-email.py` copies the tested `mail_template.workflow`, fills its
recipient/subject/body for that notification, and runs it with `/usr/bin/automator`.
The template contains only Apple's built-in `New Mail Message` and
`Send Outgoing Messages` actions.

A real delivery test on 2026-09-28 was verified in Gmail.

## Parked wake implementation

The old same-thread wake/CUA experiments are retained only for historical
reference and are not part of the active agent workflow.
