# Next step after the 6x6 loop (for the PI to choose; lead, 2026-11-01). Nothing here has run.

Where we are (exact FN, 6x6, per site above E0): one-level augmented loop guide 4.37e-6; network + one signed Lanczos step 5.57e-6;
the loop stalls after ~3 rounds with a complex PP (round 4 drop 0.34e-6). Deployable only with a stored sign (sign angle V1 decides).

| option | question it answers | cost (derived, unvalidated) | what kills it |
|---|---|---|---|
| A. 10x10 add-on: trained guide + one step + FN-GFMC, two chains | does the add-on beat the best published bound -0.4976921(4) (Chen-Heyl) | trained 10x10 RBM+PP checkpoint exists (rbL10_prod/backup_paderborn, unfinished); finish training ~5-10 H100-h; FN p=1 ~(100/64)^2-3 x 7.9 H100-h = 19-30, halved by the fast ratio path (2.04x at 10x10) -> ~10-15 H100-h | FN p=1 above -0.49769 with error bars |
| B. 8x8 loop with sampled fits and a stored sign | does the loop survive sampling and a stored sign where exact tables do not exist | per round: label-free fit with flattened sampling (untested), sign net (S2, <= 3 H100-h), evaluation by FN-GFMC with one Jacobi level (~5.4x plain; p=0 was 0.4 H100-h) -> ~3-6 H100-h per round | augmented FN at 8x8 not below network + one step (8x8 p=1 FN -0.498946(5)) after 2 rounds |
| C. richer class for the loop at 6x6 (Pfaffian angle + Astra, on paper now) | can the stall be lifted at one-level cost | <= 2 H100-h per candidate round | per its proposal |

Lead's recommendation: A first. It needs no new method (the add-on is tested at 8x8: -0.498946(5), 6.2e-5 below the best published
bound there), it answers the publishable question, and the loop's augmented guide can join as a second arm once B shows sampling works.
Confidence ~50% that A beats -0.49769 (the 8x8 gain fades with N: one step removes 84% of the error at 6x6, 47% at 8x8).
B is the loop's real scale test but has two untested pieces (sampled near-node fits, stored composite sign). C only matters if B works.

## Astra review (cj-782e689682e8, blind to the recommendation; /scratch/a/A.Otaifi/codex_jobs/cj-782e689682e8/result.md)
- Ranking A > B > C (~70%). A: 80% decision-relevant, 15-25% publishable positive.
- Corrections to A's cost: the pilot sizing model (experiments/rbmpp_6x6/m13_LxL/sizing.py) gives 66 h at 5e-6 error, 32 h with
  the 2.04x kernel speed-up (itself measured with random 10x10 params): 2-3x my estimate, which matched 1e-5 error.
- Corrections to A's start: our 10x10 checkpoint stopped at training E/N -0.497272 (delta_today 2026-10-26 19:15), 3.6e-4 above
  Nomura's -0.497629; "finish in 5-10 H100-h" is unsupported. From -0.497629 the bar needs 73% recovery of the gap to Chen-Heyl's
  extrapolation (35% against Hu's); one step gave 84% at 6x6, 47% at 8x8 (+2.3e-5 from FN after the step).
- Cheapest kill for A: evaluate the saved checkpoint independently + a short equilibrated two-chain p=1 pilot.
- B: hinge = gain left after sampling and sign-storage error (sign tax 0.36e-6 could eat the round-2 margin). Cheapest kill:
  6x6 replay with finite-sample fits and a deployable stored sign, exact tables only for grading (CPU).

## Lead, revised after the review
A stays first, but as A0 (gate, ~2-3 H100-h): VMC + short two-chain p=1 pilot from a STRONG 10x10 guide, then size A from the
pilot. Our own checkpoint is not strong enough. Candidate guide (hypothesis, unchecked): the published ViT checkpoint
huggingface nqs-models/j1j2_square_10x10_05 (-0.497676, results/literature/J1J2_BENCHMARKS_COST.md l.68): the add-on applied
to someone else's network is exactly the project's viability case 3. Its walker cost per hop is a full ViT pass (no fast
update), so start with FN p=0 (no Lanczos step): at 8x8 FN p=0 alone gained 8.3e-5 over VMC p=0, which here would land below
-0.4976921 if even a third of it survives. Kill: FN p=0 from the ViT not below -0.497692 at two-chain error <= 1e-5.
