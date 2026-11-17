# Why the sign network copies only about half of the C-free flips (2026-11-17, sign angle, takeover night)

## What was built
CPU checks of every remaining copy fix, all in `runs/j1j2/sign_bad/step6/` and `step7/`, all recorded in PLAN_sign.md
(STEP 6, STEP 7, STEP-BACK, 7b-7e). Astra reviewed step 6 (`/scratch/a/A.Otaifi/astra/step6_review/ASTRA_step6.md`).

## Results
- The C-free step is a zero-temperature Ising majority update in the Marshall frame (J1 links ferro, J2 links antiferro) [exact].
- The network copies growth of anti-Marshall domains about twice as often as shrinkage: 0.48 vs 0.29 for CF [measured].
- The copy is worth having. The measured FN gain g_3 = 5.1(13)e-5 per site against 6.4e-5 at first order, so a round realises about 0.8 of
  its first-order gain. A perfect copy would add about 1.1e-4 per site per round at first order.
- Every cheap fix failed its pre-registered kill:
  - Exact sign where the network is unsure: the gain needs threshold >= 0.5, the cost needs <= 0.25.
  - A logit offset that depends on the old sign: at most ~1e-5, and the best direction is opposite to the mechanism.
  - Amplitude RBM features: adds nothing.
- The label is robust to independent input noise: 50% ln-ratio noise changes 19% of flip labels. Yet CF misses about 50%.
  So the network does not compute a noisy field. It predicts from proxies.
- The field's inputs are nonlocal:
  - A 7x7 linear window explains 41% of ln(A(z^b)/A(z)).
  - The neighbour's old sign has no local linear signal.
  - Local ratios plus the exact old signs still copy only 0.40 of flips.

## Rejected and why
- Confidence gate:
  - Cost 1 + 0.8 x (gated neighbours per E_L), anchored on the G(-5) timing.
  - Requests are deep: median decade -6 for every walker decade.
- Per-bond gating: the sign of z would depend on which walker asks, so psi_T would not be single-valued.
- Independent-set colouring: cross terms are only 0.2% of the linear term on the panel.

## Three things to understand
1. The label at z is the sign of the amplitude's own off-diagonal local energy at z. Copying it means emulating the amplitude network's swap
   ratios and the previous sign net, from spins alone.
2. Cost is counted per neighbour evaluation, not per importance-weighted request. Every local energy touches about 142 neighbours.
3. First order is a good proxy here, with cross terms at 0.2% and FN/first-order about 0.8. The open risk is the FN mapping in general
   (Astra finding 19: a counterexample exists).
