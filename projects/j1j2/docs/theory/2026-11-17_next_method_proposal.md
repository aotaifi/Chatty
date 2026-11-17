# Next method for the storage step (lead, 2026-11-17; theorist mode; DRAFT until rounds 2-3 and Astra's review are in)

Organising question: how to carry one round's fixed-node improvement into the next round at one-level walker cost.

## 1. What failed, and why (established today)
- Learned small heads: an RBM factor holds 2-3% of the round's correction in the energy norm at first order (rung 2b, exact
  full 6x6 table), below a random vector; the records already said so (M-R KILL 10-28: ~6% stored; Test L: two-body features leave 97%).
- Log-loss refit of the whole network (R1', 8x8): the near-node rows are small because the 256 symmetry copies cancel
  (kappa ~ 0.005); |grad ln B| ~ 1/kappa (corr -0.93) puts those rows outside the linear region of any bulk-sized step.

## 2. What the data say the correction IS (rung 2, exact 6x6, design B round 1)
With the Lanczos guide G = A |1 + alpha(E_L - E)| (sign s2, alpha -0.10) and p its fixed-node state, u = ln(p/A):
    u  ~  l + c f_J[G],   l = ln|1 + alpha(E_L - E)|,   f_J[G] = ln[K_G / ((H_xx - E') G + V_s,G)],   c = 1.31   (a = 1.01 on l)
leaves eps^2 = 0.0033 (5-fold CV 0.0034) in the blocked-pair D-norm; storing exactly this gives 98.5% of the fixed-node gain
(E_FN 6.511e-6 vs 6.285e-6 /site for the perfect copy). f_J alone: 0.85. [measured, existing data]
Reading [own]: keep the Lanczos amplitude factor as it is, and add about 1.3 Jacobi steps of the FN equation on that guide.
Nothing needs to be learned inside a round.

## 3. The real problem is cost
A_{k+1}(x) = A_k(x) |1 + alpha_k(E_L(x) - E)| exp(c f_J[G_k](x)) needs A_k at two hops (E_L at x: one hop; f_J[G_k] needs G_k at
neighbours: two hops). Evaluated directly, k rounds need 2k hops: cost ~ d^{2k} per amplitude. [exact counting]

## 4. Proposal: explicit round + consolidation
 (a) Round k: the explicit formula above is evaluated only as the TARGET of the consolidation, on a fixed set of configurations, at
     two hops of the current base B^(k) (two-swap ratios; PLAN l.512: a constant factor, 5.4x for the PP at 6x6). Walkers never evaluate
     it: they use B^(k+1) after consolidation, plus design B's one Lanczos step.
 (b) Consolidation after each round (or every second round): fit a FULL network B^(k+1) (same class as B_0) to the explicit state
     A_{k+1} on fixed configurations, with a loss whose 1/B is a fixed weight: relative square ((|B| - T)/T)^2 or the frozen-H_FN
     Rayleigh quotient. Derived 11-17: the relative-square residual is linear in |B|, so the 1/kappa of cancelling rows enters as a fixed
     weight, not as the log's curvature that made R1' overshoot. Then the next round starts from B^(k+1) at one hop.
 Why not learned heads: section 1. Why not nesting: section 3.

## 5. What is assumed, and how each would be killed
 A1. The two-term formula stays accurate in later rounds (stable c). Kill: rounds 2-3 on 6x6 (running, rung 2) give eps^2 > 0.05 or c
     drifting > 30%.
 A2. Consolidation with a fixed-weight loss stores A_{k+1} without the R1' failure. Kill (needs a run, the PI's 'now test it'): on 6x6,
     consolidation keeps < 70% of the round's gain, or the near-node rows dominate the fit (weights ~ 1/T^2) so the bulk does not move.
 A3. The fitted-power-iteration loop stays stable under consolidation error: it improves while the consolidation error per round is below
     that round's gain (eps_k <= Delta_k; theorist note 2026-11-17).
 A4. 8x8 transfer: the 8x8 Lanczos factor is tame under |B_0|^2 (rung-2 8x8: 0 of 1e5 rows with f < e^-4.88), so the formula's
     dangerous rows are rarer there than at 6x6; the near-node cancellation diagnosis (kappa) is 8x8's own.

## 6. Literature anchors (from today's reviews)
 c f_J is a Davidson-type diagonal correction; the explicit round is a projector step; consolidation is fitted power iteration (needs a
 fit that does not amplify errors: Gordon 1995 'averagers'; arXiv 2303.08902); refitting to the FN state: self-healing DMC
 (0808.3417), with a biased-node warning (2609.01301).
