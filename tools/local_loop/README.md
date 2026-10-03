# Chatty durable Slurm wake workflow

Use this for long-running Slurm jobs that should resume the exact originating
normal ChatGPT conversation when the job finishes.

## Supported clusters

Exactly two profiles are supported:

- `theorie`: the LMU Theorie Slurm cluster. `ws1` and `ws3` are equivalent
  gateways; polling fails over between them.
- `paderborn`: PC2/Paderborn through `ws1 -> paderborn`. The rootless PC2
  VPN is expected on `ws1`.

## One command after sbatch

After `sbatch` returns a numeric root job ID, run:

```bash
~/Chatty/tools/local_loop/chatty-slurm-watch \
  <job_id> "<short description>" <PROJECT_TAG> \
  --cluster theorie
```

or:

```bash
~/Chatty/tools/local_loop/chatty-slurm-watch \
  <job_id> "<short description>" <PROJECT_TAG> \
  --cluster paderborn
```

Then report the job ID/output locations in chat and end the turn. Do not keep
ChatGPT polling.

## What registration does

The public wrapper:

1. Verifies that the Slurm job really exists.
2. Identifies the currently visible normal ChatGPT conversation and captures
   its immutable thread ID from the loaded ChatGPT WebArea.
3. Treats the title as metadata only; routing and post-navigation verification
   use the thread ID.
4. Freezes cluster, job ID, project tag, thread ID and description into one
   immutable per-job target.
5. Starts exactly one detached watcher.
6. Sends Slack STARTED.
7. On FINISHED/FAILED, persists terminal state, sends Slack, and wakes the exact
   frozen ChatGPT thread.

A later chat cannot retarget an already registered job.

## Delivery safety

The wake path is deliberately fail-closed:

- one global ChatGPT UI lock serializes different job completions;
- one per-job lock prevents duplicate watcher processes;
- the wake ledger prevents the same exact payload from being knowingly sent
  twice;
- wake prompts include `cluster=<name>`, so equal numeric job IDs on Theorie
  and Paderborn cannot collide;
- the target thread ID is verified against ChatGPT's own loaded conversation ID
  after navigation;
- existing user drafts are never overwritten;
- FINISHED/FAILED is first persisted as durable terminal state; delivery is a
  separate queue step;
- if the target chat is answering, has a draft, is temporarily inaccessible, or
  changes during the safety window, the wake stays `pending` and the daemon
  exits cleanly;
- the launchd recovery dispatcher retries pending wakes every 60 seconds and
  after login/reboot, so a long active turn cannot lose the event;
- cross-device activity gets a continuous idle window before staging and a
  second stability window after staging before Send;
- if Send may have happened but cannot be verified, the workflow enters
  verification/manual-review mode and never blindly presses Send again;
- temporary ChatGPT/Accessibility/SSH/VPN failures retry rather than being
  misclassified as simulation failure;
- permanent thread identity failures stop safely and Slack remains the backup.

### Multi-device race limit

A mobile/web turn normally synchronizes to the Mac as an unavailable composer
or active Stop state, so the watcher waits. The release also requires several
seconds of stable target state before Send.

There is still a very small unavoidable race if another device submits a new
message in the final moments before our Send and that remote activity has not
yet synchronized to the Mac. Eliminating that last window would require a
server-side transactional "send only if conversation version is unchanged"
primitive, which this local workflow does not have.

## Arrays and accounting

The watcher handles Slurm array jobs. While any allocation/task remains in
`squeue`, the job is considered active. After it leaves the live queue, the
watcher checks `sacct` and aggregates array-task state when needed.

A scheduler-level `squeue` "invalid job id" for an already completed job is
not treated as a transport error; the watcher proceeds to accounting.

## Crash, sleep and reboot recovery

Recovery is automatic through the macOS LaunchAgent:

`com.chatty.slurm-recover`

It scans registered unfinished jobs every 60 seconds and after login, and
restarts missing watchers. Per-job locks make repeated recovery scans safe.

Manual recovery is also available:

```bash
~/Chatty/tools/local_loop/chatty-slurm-recover
```

Transient UI failures do not expire merely because the Mac slept overnight.

## Diagnostics

To verify the real job and current chat without persisting a target or launching
a watcher:

```bash
~/Chatty/tools/local_loop/chatty-slurm-watch \
  <real_job_id> "diagnostic" TEST_TAG \
  --cluster theorie --prepare-only
```

Use `--cluster paderborn` for PC2.

## Public commands

Future chats should use only:

- `chatty-slurm-watch` — registration
- `chatty-slurm-recover` — manual recovery/diagnostic recovery scan

Do not call legacy `watch_paderborn_slurm.sh` scripts for new jobs, manually
copy thread IDs, edit `project_threads.json` for a new job, or rebuild the
same-thread wake path per conversation.

The public commands are pinned to a tested release directory. Internal release
files are implementation detail and should not be edited by ordinary research
chats.

## Local non-Slurm processes

For local Mac jobs, continue using `chatty-run` / `chatty-watch`. Those use
separate durable state and email notifications.

## Native plugin route review (2026-10-03)

OpenAI now exposes MCP Events for webhook-driven plugin subscriptions, which is
the clean server-side architecture we would prefer. Today, however, MCP Events
are documented for Work chats (and dots), not arbitrary normal ChatGPT chats.
Slack event-triggered tasks have the same Work requirement.

Plugin UI also exposes `sendFollowUpMessage`, but that is a runtime API for a
loaded component. It is not a durable background webhook that can wake an
arbitrary closed/inactive normal conversation after ChatGPT restarts.

Therefore the normal-chat production path remains the local trusted UI broker.
If MCP Events becomes available for normal chats, replace the AX delivery layer
with MCP Events and keep the same immutable per-job binding/idempotency model.

## Proven baseline

Release `1.2.2-release-20261003-1137-safeguard` passed fresh end-to-end terminal
wake tests on both supported backends into the same exact normal ChatGPT thread:

- Theorie job `16805444`: `COMPLETED 0:0`, exact payload verified in transcript,
  ledger `delivered`, one wake attempt.
- Paderborn job `3554875`: `COMPLETED 0:0`; first delivery attempt correctly
  deferred while the target chat was in an active turn, recovery retried, exact
  payload was then verified in transcript, ledger `delivered`, two attempts.
- Re-registering the Paderborn job with changed immutable metadata was refused.
  Re-registering the exact original binding returned `ALREADY_DELIVERED` and did
  not send a duplicate.
