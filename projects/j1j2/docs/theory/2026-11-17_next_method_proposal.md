# Next method for the storage step (lead, 2026-11-17; theorist mode; v2 after Astra's review, ready for the PI)

Organising question: how to carry one round's fixed-node improvement into the next round at one-level walker cost.
Review: /scratch/a/A.Otaifi/astra/next_method_review/REVIEW.md (gpt-6-astra, cj-b0203391dc9d); adjudication in STORAGE_PROBLEM.md
'NEXT-METHOD REVIEW'. v1's retention law (tax/gain = eps^2 in the D-norm) was wrong and is removed.

## 1. What failed (established today)
- Small learned heads (RBM factor): 2-3% of a round's correction stored (rung 2b, exact 6x6); records agree (M-R KILL 10-28, Test L).
- Log-loss refit of the whole network (R1', 8x8): near-node rows are small because the 256 symmetry images cancel (kappa ~ 0.005);
  |d ln B / d theta| ~ 1/kappa puts them outside the linear region of any bulk-sized step.
- Sampled frozen-H_FN Rayleigh on the head (B10 pilot): population M_req 5.2e9 vs 1.5e4 estimated from 4096 draws (missed heavy tail).

## 2. What the data say the round's correction is [measured, exact 6x6, fixed oracle sign s2]
G = A |1 + alpha(E_L - E)| (Lanczos guide), p = FN state of G, u = ln(p/A), l = ln|1 + alpha(E_L - E)|,
f_J[G] = ln[K_G / ((H_xx - E') G + V_s,G)] (one Jacobi step of the FN equation, E' = E_FN[G]).
    explicit teacher:  A_{k+1} = A_k |1 + alpha_k (E_L - E_k)| e^{c f_J[G_k]},  c = 1.307
Separate FN solves: this teacher keeps 98.07% (round 1) and 96.27% (round 2) of each round's FN gain, E_FN 2.09e-5 -> 6.57e-6 -> 3.32e-6
/site (ceilings 6.29e-6, 3.19e-6). Refitted c: 1.31, 1.38, 1.58 (rising). The scalar c f_J captures 86-88% of the post-Lanczos correction.
Certificate (section 4c) for the teacher itself: E_var[A_{k+1}] < E_FN[G_k] by 7.45e-7 and 1.70e-7 /site = 5.1% and 5.0% of the round's gain.
Not shown: moving Lanczos signs (design B step 3), any network storing it, any size other than 6x6.

## 3. Cost
A_{k+1} at x needs A_k on the two-hop neighbourhood of x; k nested rounds need radius 2k (a naive recursion ~ d^{2k} work).
So the teacher must be consolidated into a network, not nested.

## 4. Proposal (corrected)
 (a) Frozen teacher. At round k freeze the stored network snapshot B_k (state psi_k = S_k |B_k|), build the Lanczos guide g_k and evaluate
     tau_k = F_k[B_k] (the formula of section 2) on a fixed configuration set, at two hops of B_k. alpha_k AND c_k are chosen by 1-D energy
     searches, never by fitting to p (p is not available outside the 6x6 table). This defines a fitted iteration B_{k+1} = Pi_k F_k[B_k];
     its errors compound (z_{k+1} <= L_k z_k + eta_k), so acceptance is per round (c), not by a contraction claim.
 (b) Consolidation: fit a full network chi_k (same class as B_0, normalised) to tau_k with the frozen-relative pair loss
         L = sum_{blocked b = (x,y)} h_xy T_x T_y (t_x / T_x - t_y / T_y)^2,   t = |B_theta|, T = tau_k (frozen)
     plus a normalisation term (t = 0 is otherwise a minimiser). No moving 1/B in its gradient. Each blocked edge is drawn from its
     bulk end (proposal ~ h_xy (T_x^2 + T_y^2)); then a matched near-node edge contributes O(delta^2) to the gradient's second moment,
     not O(delta^2/kappa^2) as with Born row sampling. Prior art for (a)+(b): SWO (1811.12423), Ledinauskas-Anisimovas 2307.15521.
 (c) Acceptance certificate (Astra eq. 10, exact, sufficient): accept chi_k only if E_H[chi_k] < E_FN[g_k]. Then the next Lanczos + FN step
     descends. On existing 6x6 data the teacher leaves consolidation ~5% of the round's gain as variational-energy budget.

## 5. Assumptions and kills
 A1. Teacher quality holds with energy-searched c and moving signs. Kill (paper + 6x6 tables, CPU, rung 2): with c from a 1-D E_FN
     search and S_{k+1} = S_k sgn[1 + alpha(E_L - E)], retention < 90% or the certificate margin < 0 in any of rounds 1-3.
 A2. A full network realises the teacher within the ~5% budget under loss (b) with an affordable gradient. Open; needs a run (PI's 'now
     test it'). Kill: on 6x6, E_H[chi_1] >= E_FN[g_0] after the declared budget, or M_25% (Astra eq. 23) of the pair-loss gradient along the
     SR direction exceeds the sample budget. Number to beat: the M-R heads kept ~6% of a gain; the certificate needs ~95%.
 A3. c stays bounded. Reading [own, conj 55%]: one Jacobi step undershoots slow modes, so c ~ 1/(1 - rho) on the current residual; conditional
     (fixed-P single mode), not a measurement. Kill: energy-searched c grows without bound or the frozen-mode factor |1 - c(1 - rho)| >= 1.
 A4. 8x8 transfer: no argument yet (a rare Born tail does not bound the gradient's second moment). Needs its own estimate before any 8x8 run.
 Known failure mode from the literature: repeated projection into an imperfect class can stall at an error floor or converge to a wrong node
 (self-healing DMC, 2609.01301); the per-round certificate is the guard.
