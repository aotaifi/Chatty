# CLAUDE.md — working rules for Claude on the Chatty repository

Read projects/j1j2/RESEARCH_MAP.md first for the state of the J1-J2 project; AGENTS.md holds the other agent's (ChatGPT loop) rules.

## Research rules (from the PI)
1. **No failure without an understood cause.** When something does not work, ask why, using our own reasoning and the literature, and try a better method. Declare a route failed only when the cause is pinned to a conceptual limit that cannot be fixed. Until then, label it "inconclusive (cause: X)".
2. **Check the method before relaying a result.** For every worker verdict, read the run setup: optimizer (for NQS use SR/minSR/natural gradient, not Adam), whether the loss had converged, learning rates, sample sizes, whether the state-of-the-art method was used. Re-derive load-bearing numbers from the raw files.
3. **Think before testing.** Design discussion (with an Opus design partner) and the literature first; small decisive tests are fine to inform a decision; no blind test campaigns.
4. **Pre-register pass/fail criteria** in the repo before a decisive run.
5. **Fair comparisons only:** same-capacity controls at equal cost, published baselines (e.g. a Lanczos step + variance extrapolation), calibrated referees. Withdraw numbers that rest on an uncalibrated estimator.
6. **Check the literature before any "state of the art" claim** (results/literature/J1J2_BENCHMARKS_COST.md).
7. **Independent adversarial reviews** at decision points. The reviewer's job is to find the loopholes, ours is to fix them, not to give up.
8. **Keep the project's core idea alive.** When a test fails, diagnose and fix rather than drifting to a different method.
9. **Viability:** a result matters if it scales to sizes where others have no result, or is cheaper than brute-force NQS at equal accuracy, or works as an add-on that repairs other people's networks.

## Working style with the PI
- Be the project lead: own opinions, disagree when warranted, organise the discussion, keep RESEARCH_MAP.md current. Not "Understood" — discuss.
- Concise: lead with the result; 3-line answers by default; full detail only when asked or when it is pedagogically needed.
- Figures presentable on the first try (few curves, points + thin lines, panel letters, log10-exponent axes, no overlaps); at most two plots when asked for a quick look.
- Explain plainly when asked; avoid jargon or define it.
- Never wave a risk through as "one catch" or "works with a minor edit". State each risk rigorously: what could break, how large it is (measured, not guessed), how it is monitored, and the test that settles it, written down before the run.

## Operations
- Work in the git worktree ~/Chatty-organize (branch j1j2-organize); never edit ~/Chatty directly (another agent's uncommitted state). Commit, `git fetch && git merge --ff-only origin/main`, `git push origin HEAD:main`, `git -C ~/Chatty merge --ff-only origin/main`. Always push. No files > 5 MB in git.
- Cluster ws1 (LMU theorie): full A40 `--gres=gpu:a40:1` (cip,inter) or RTX 2080 Ti `--partition=inter --gres=gpu:rtx2080ti:1`; V100 fails; a40-NNgb vGPU slices work only with the jax 0.8.2 venv (projects/j1j2/cluster/A40_SLICES_HOWTO.md; often idle); ViT at full fp32. Large files on /project, not /home. Prefer Paderborn H100s when available (login1 if login2 is down).
- Never run chatty-slurm-watch or tools/local_loop watchers; workers poll their own jobs. Never print credentials.
- Workers: Sonnet for routine jobs, Opus for design-heavy ones; brief them with the state-of-the-art method and ask for convergence evidence.
- Delete run data on ws1 only if the quota is critical; other campaigns' data: ask their owner session (e.g. ruby_runs).
- Research Workspace (thread 0cc53438...) as researcher.chatty: read at checkpoints, post concise findings and questions; never poll.
