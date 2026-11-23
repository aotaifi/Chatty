# Inner voice: a Stop hook that sends the j1j2 lead back to work (PI 2026-11-23)

It only checks: at each stop of an enabled lead session it reads the closing message and the turn's tool calls, and when the
lead is ending on work it could do now, it blocks the stop with one instruction. Script: `~/.claude/hooks/inner_voice.py`.

## Evidence (lead transcript 00dd5079, 118 turns, 11-19 07:44 to 11-23 16:19)
- Lazy stops all ended on a deferral, an offer, or "next I'll" with nothing launched: 11-23 00:59 "I'm now working out on paper"
  (idle 4 h 47), 05:46 "I'll write that down", 06:21 "Next I'm deriving" (idle 2 h 57, then done in 2 min after "Great"),
  12:59 "next I'll try ... (6 h CPU fit)" (not launched), 16:14 "I can check that ... ~1 h CPU", 11-19 "Shall I run the cheap check?".
- Idle beside workers: a worker result logged, then "." with nothing new (11-23 09:56: 3 h; 11-19 17:37 KILL: 13.5 h);
  a launch, then 2 h 40 of sitting while holding an untested idea (11-23 13:23). Lead's diagnosis: it read root §3 ("a worker
  report ends with '.'") as "end the turn"; it only means "say nothing to the PI". There was a free step both times.
- Legitimate stops: "." on duplicate or interim notifications ("waiting on its own background work", "no results yet"), status
  answers after a real launch, the overnight wait on a worker the lead had launched with its next step named.

## Policy (each rule is a pattern in the evidence; the lead agreed in interview, 11-23)
| rule | fires when | why |
|---|---|---|
| OFFER | closing offers work: "I can check/run ...", "Shall I", "want me to" | rung 1-2 are the lead's (lead Q1); a run needs its prediction from an earlier turn and "now test it" on record, else the missing paper step is the action |
| DEFER | closing promises paper work in future/present tense: "I'll derive", "I'm now working out on paper" | nothing continues after a stop (lead Q3); exempt when tied to a pending result ("once X lands") |
| ANNOUNCE | "next I'll / I'm now run/test/fit/check ..." and the turn launched nothing | the 12:59 case; a launch or a message to a worker satisfies it |
| RESULT | a worker result arrived (hand-back, VERDICT wait, Astra), nothing launched, no `[wait]`/`[parallel]` logged | 09:56 and 17:37 cases |
| LAUNCH | a turn not started by the PI launched new work with no `[wait]`/`[parallel]` line | the 13:23 case: launch, then sit |
| NOFREE | `[wait] no free step` written without loading `step-back` | the lead's own warning: that line is where a false wait hides (Q4) |
- Text rules (OFFER, DEFER, ANNOUNCE) fire on every stop, including mid-conversation with the PI (lead Q5: he wants action, not quiet).
  LAUNCH does not fire after a PI prompt (Q5: a plain status answer after a real launch is fine); the next result turn is checked.
- Never pushes into a run the rules forbid: the nag text names the ladder (point 9). The lead reads the PI's 11-22 "we need this
  test definitively" as "now test it" for CPU fits on the copy question, but no such line is in PLAN.md: write it there, or the
  hook and the project file disagree.
- Frequency: one nag per stop at most, at most 2 blocks in a row (then the stop is allowed); after a nag only the closing words are
  re-judged. Backtest on the history: 24 nags in 4.4 days (9 text, 6 RESULT, 9 LAUNCH), about 5 a day, and all 9 text hits are cases
  the lead calls lazy. A lead that writes `[parallel] <step>` or `[wait] <step> needs <job>: <why>` in delta_today.md at each launch and
  each result gets no RESULT/LAUNCH nags at all.
- Every decision is logged to `~/.claude/hooks/inner_voice_log.jsonl` (time, rule, closing line): the PI's count. Review it weekly:
  a nag the lead had to argue against is a false positive; drop or narrow the phrase in the script.

## Lead's side of the contract
- A closing line in the future tense is backed by a tool call made in the same turn (a launch, or the STORAGE block it names).
- At each launch and each result: one delta line, `[parallel] <paper step I am doing now>` or `[wait] <step> needs <job>: <why>`.
- After acting on a nag, close with one line or "." (the two-line rule still holds).

## Install (the lead does this; the hook stays silent until a session is enabled)
1. Add to `~/.claude/settings.json` under `"hooks"` (next to PreToolUse/PostToolUse):
```json
"Stop": [
  { "hooks": [ { "type": "command",
                 "command": "/usr/bin/python3 /home/a/A.Otaifi/.claude/hooks/inner_voice.py",
                 "timeout": 20 } ] }
]
```
2. Enable this session: `python3 ~/.claude/hooks/inner_voice.py --enable "<a unique phrase from this turn's own command>"`
   (it finds the main transcript modified in the last 15 min that contains the phrase and adds its session id to
   `inner_voice_sessions.txt`). Each fresh session after /clear runs this again: put the line in the migrate prompt.
3. Silence it without touching settings: `touch ~/.claude/hooks/inner_voice.off`. Replay any transcript:
   `python3 ~/.claude/hooks/inner_voice.py --backtest <transcript.jsonl>`.

## Verified / not verified
- Verified here: the backtest above; an end-to-end run on fake input (OFFER, exit 2 with the reason; a chained stop with a
  launch, exit 0; a session not enabled, exit 0); `--enable` found transcript 00dd5079.
- From the hook docs as relayed (code.claude.com/docs/en/hooks.md): Stop input carries `last_assistant_message`, `stop_hook_active`,
  `background_tasks`; Claude Code forces the stop after 8 blocks in a row. Not re-fetched by me: that exit code 2 on a Stop command
  hook blocks and passes stderr to the model. This is the documented behaviour, and the existing PreToolUse gate relies on the same
  convention. Check it in the first live nag.
- An Opus prompt or agent hook was considered and not used. Stop hooks cannot be scoped to one session, and every lead under
  /project/theorie/a/A.Otaifi reads the same settings, so it would judge the z2_hunt and ParaToric leads under rules that differ.
  A prompt hook also cannot see the turn's tool calls, and its decisions cannot be backtested. Revisit if the log shows misses the
  patterns cannot catch: an `agent` hook that first greps `inner_voice_sessions.txt` for the session id.
