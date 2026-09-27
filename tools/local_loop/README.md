# Durable long-run wake loop

This directory contains the Mac-side bridge used for long-running research jobs.

## What it does

A run can outlive the ChatGPT turn that launched it:

```
ChatGPT research turn
  -> Mac simulation / benchmark
  -> durable job state under ~/Chatty/jobs/<job-id>/
  -> run exits
  -> WAKE_PENDING is written
  -> ChatGPT.app bundled browser CUA
  -> prompt submitted into the SAME ChatGPT conversation
  -> a fresh model turn continues the research
```

No OpenAI API key is used.

## One-time binding for a conversation

Bind the exact ChatGPT conversation that should be woken:

```bash
~/Chatty/tools/local_loop/chatty-bind-thread <conversation-id>
```

The binding lives in `~/Chatty/.chatty/default_thread_id` and is intentionally not committed. A run can override it with `--thread <id>`.

## Install durable recovery once

```bash
~/Chatty/tools/local_loop/chatty-install-recovery
```

This installs `~/Library/LaunchAgents/com.chatty.recover.plist`. Once per minute it checks for jobs that still contain `WAKE_PENDING` and have no `WAKE_SUBMITTED`, and retries delivery. Per-job file locking prevents two recovery attempts from delivering simultaneously.

## Launch a new long run

```bash
nohup ~/Chatty/tools/local_loop/chatty-run \
  --task "Analyze the finite-size scaling and update STATUS.md" \
  --cwd ~/Chatty/projects/j1j2 \
  -- python3 scripts/benchmark.py \
  > /tmp/chatty-launch.log 2>&1 &
```

`chatty-run` records the command, PID, timestamps, exit code, stdout and stderr. It uses `caffeinate -i` while the child is alive and again while a wake delivery is in progress.

## Attach to a run that already exists

```bash
nohup ~/Chatty/tools/local_loop/chatty-watch \
  --pid 12345 \
  --task "When this run ends, inspect its output and continue the scaling analysis" \
  > /tmp/chatty-watch.log 2>&1 &
```

An attached watcher cannot recover the target process's exit code because it is not the parent; it records `exit_code: null` and wakes the conversation when the original PID disappears (or is reused).

## Durable job layout

```
~/Chatty/jobs/<job-id>/
  job.json
  stdout.log          # chatty-run
  stderr.log          # chatty-run
  DONE
  WAKE_PENDING        # delivery still required
  wake.log
  WAKE_SUBMITTED      # delivery confirmed
  WAKE_FAILED         # latest attempt failed; recovery still retries
  wake.lock
```

`WAKE_FAILED` is not terminal. `WAKE_PENDING` is the durable source of truth that a continuation still needs delivery.

## Wake implementation

The CUA node runtime has a roughly 30-second deadline per JavaScript request. Therefore the wake helper never waits in one long JS call. It keeps one CUA session alive and uses short calls for:

1. creating a browser tab;
2. navigating to the target conversation;
3. polling thread readiness from Python with repeated AX-state calls;
4. setting the composer value;
5. submitting Return.

The thread may remain busy for up to 15 minutes without hitting the per-call CUA timeout. If delivery still fails, launchd recovery tries again later.

## Requirements / failure behavior

- ChatGPT.app must remain installed and logged in.
- The Mac needs network access when delivery eventually occurs.
- Browser-origin permission is auto-approved only for exactly `https://chatgpt.com`.
- The ChatGPT app version is detected from its Info.plist at runtime.
- If delivery fails, inspect `wake.log`; do not remove `WAKE_PENDING` unless the continuation is intentionally abandoned.

## Recovery / manual retry

Retry one job immediately:

```bash
~/Chatty/tools/local_loop/chatty-retry-wake <job-id>
```

Scan pending jobs now:

```bash
~/Chatty/tools/local_loop/chatty-recover --max-jobs 10
```

## Manual wake test

```bash
~/Chatty/tools/local_loop/chatty-wake-browser.py \
  "$(~/Chatty/tools/local_loop/chatty-thread)" \
  "[CHATTY_MANUAL_WAKE_TEST] Reply exactly CHATTY_WAKE_OK"
```
