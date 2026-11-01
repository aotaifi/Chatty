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
