# Durable long-run wake loop

This directory contains the Mac-side bridge used for long-running research jobs.

## What it does

A run can outlive the ChatGPT turn that launched it:

```
ChatGPT research turn
  -> Mac simulation / benchmark
  -> durable job state under ~/Chatty/jobs/<job-id>/
  -> run exits
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

The value is stored locally in `~/Chatty/.chatty/default_thread_id` and is intentionally not committed. A run can always override it with `--thread <id>`.

## Launch a new long run

```bash
nohup ~/Chatty/tools/local_loop/chatty-run \
  --task "Analyze the finite-size scaling and update STATUS.md" \
  --cwd ~/Chatty/projects/j1j2 \
  -- python3 scripts/benchmark.py \
  > /tmp/chatty-launch.log 2>&1 &
```

`chatty-run` records the command, PID, timestamps, exit code, stdout and stderr. By default it uses `caffeinate -i` while the child is alive so an idle Mac does not sleep.

## Attach to a run that already exists

```bash
nohup ~/Chatty/tools/local_loop/chatty-watch \
  --pid 12345 \
  --task "When this run ends, inspect its output and continue the scaling analysis" \
  > /tmp/chatty-watch.log 2>&1 &
```

An attached watcher cannot recover the target process's exit code because it is not the parent; it records `exit_code: null` and still wakes the conversation when the original PID disappears (or is reused).

## Durable job layout

```
~/Chatty/jobs/<job-id>/
  job.json
  stdout.log          # chatty-run
  stderr.log          # chatty-run
  DONE
  wake.log
  WAKE_SUBMITTED      # success
  WAKE_FAILED         # after retry exhaustion
```

The wake prompt tells the next ChatGPT turn to inspect the job and continue without waiting for the user to type "go".

## Requirements / failure behavior

- ChatGPT.app must remain installed and logged in.
- The Mac needs network access and must be awake when the wake is submitted.
- Browser-origin permission is auto-approved only for exactly `https://chatgpt.com`.
- The wake helper retries three times and leaves `WAKE_FAILED` plus `wake.log` if all attempts fail.
- The ChatGPT app version is detected from its Info.plist at runtime; no build number is hard-coded.

## Manual wake test

```bash
~/Chatty/tools/local_loop/chatty-wake-browser.py \
  "$(~/Chatty/tools/local_loop/chatty-thread)" \
  "[CHATTY_MANUAL_WAKE_TEST] Reply exactly CHATTY_WAKE_OK"
```
