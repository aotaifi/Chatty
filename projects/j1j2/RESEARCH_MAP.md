# J1-J2 FN/Krylov project: research map (living document, updated 2026-10-07)

One page that holds the thread of our discussions. Details live in CLAUDE_TAKE.md (log) and results/*/README.md.

## 1. What would make this worth publishing (PI's criteria)
1. **Scale:** new results where other methods have none. Survey (results/literature/J1J2_BENCHMARKS_COST.md):
   - 8x8: an energy below -0.49889 (RBM+PP -0.498886) would be new.
   - 12x12, 14x14, 18x18: no deep-NQS results at all (the clearest gaps).
   - 10x10: saturated (record -0.4976921); only a cost win or a controlled extrapolation counts.
2. **Cost:** cheaper than brute-force NQS at equal accuracy. Bar: the 10x10 ViT took ~1.9e3-3.8e3 A100-h.
3. **Separate pitch:** FN/Krylov repairs the signs of bad or imperfect NQS guides ("enhance your NQS").

## 2. Established (with the file that holds the evidence)
| Result | Where |
|---|---|
| Krylov sign step (label-free, energy-optimal T): exact signs in 3-6 steps given the exact amplitude; depth grows slowly with N (16 -> 36 sites) | paper, sign_design_tests |
| Exact FN/Krylov loop reaches the GS; iteration count flat 16 -> 36 sites; Anderson ~3x faster | paper |
| Bad starts: any in-sector GS content -> GS; exact eigenstates are fixed points; other symmetry sectors trap; Anderson unsafe from bad starts | bad_start/ |
| Stored signs: composite sign net + ONE exact hop = recursion at fixed cost; train on tempered samples + one-hop neighbours (self-sealing); Krylov step factors are unlearnable, composite signs are learnable | stored_signs*/, sign_learnability/ |
| 6x6 signs: composite net + hop w_s 2e-5 vs ViT 1.3e-4 | amp_design/T3 |
| FN-surrogate amplitude objective = one majorise-minimise step on fixed-sign VMC; second-order fixed-sign VMC beats it 4-10x | amp_design/DESIGN_MEMO.md |
| 4x4 from a trained net: Krylov sign + second-order fixed-sign VMC beats the best tuned VMC (signs+amplitude) 3.3-7.2x at equal CPU-h | amp_design/vit_t1t2* |
| 6x6 from the ViT: energy tie; near the ViT the remaining FN error is the amplitude, not the signs | learned_loop_6x6_it2, amp_design/T3 |
| Corrections: our ViT is NOT state of the art at 6x6/8x8 (RBM+PP is better) | CLAUDE_TAKE 2026-10-07 |

## 3. Open questions, in the order we attack them
1. ~~Why does the loop stall at the ViT level on 6x6?~~ **Answered 2026-10-07 (stall_6x6/README.md):** not information (ideal exact loop from the symmetrised ViT passes RBM+PP in 3 iterations, 8.8e-6 at 12) and not signs; the network cannot absorb the FN update accurately enough: energy cost 0.33 sigma^2/site, so each update must be written back accurately (0.004 rms is the white-noise case; the real criterion is the edge-difference energy form Q(e) <~ 5e-6, review finding 4). Symmetrisation is worth 2.4e-5 for free. **New open question 1: an amplitude representation/optimiser that reaches 0.004 rms.**
2. ~~Do large-sample amplitude steps beat their own control?~~ **Answered 2026-10-07: no.** Large N fixes generalisation (steps verify at z 2-8), but at the ViT amplitude the fixed-sign optimum is within ~1e-5/site; loop arm ties, control D gains -0.35(0.31)e-5 (amp_design/AMP6_table.md).
3. **Cost race from scratch:** Marshall + amplitude-only net + Krylov signs vs training a full NQS, total GPU-h to a target energy (6x6 validation -> 8x8 target < -0.49889). Waits for 1 (architecture choice).
4. **Add-on pitch at scale:** one Krylov sign step on the public 10x10 ViT checkpoint (nqs-models); cheap check.
5. **Baseline to beat for any energy claim:** Lanczos step + variance extrapolation (Hu et al. 2013).

## 3b. Independent review 2026-10-07 (results/review/REVIEW_2026-10-07.md) — accepted points
- Critical: at a VMC-converged network the FN amplitude step has zero first-order gain (majorisation); every amplitude test needs a same-capacity VMC control.
- Critical: 4x4 "3.3-7.2x at equal CPU-h" is not fair/transferable (exact sign table on 4x4; CPU-h implementation-dependent; 1.2-1.8x at equal step count). Do not claim it as is.
- Critical: one Lanczos step is the untested baseline at equal hop cost. Exact 6x6 comparison running (results/lanczos_baseline_6x6/).
- Major: FN/DMC referee biased by ~1e-5 at M=128; calibrate vs exact 6x6 FN before any 1e-5 claim.
- Major: symmetrisation gain 2.4e-5 is ~1 sigma vs the ViT's VMC error; costs 16x per evaluation.
- Reviewer's viability verdict: not viable as a ground-state solver at 8x8-10x10; narrow add-on survives only if it beats a Lanczos step at equal hop cost.

## 3c. Pre-registered verdict thresholds (reviewer, 2026-10-07, set BEFORE the results)
- Lanczos vs Krylov sign (exact 6x6, equal hop cost): toward viable if sign-only recovers >= 50% of the Lanczos gain, or Krylov+Lanczos beats Lanczos alone by >= 2e-5; kill the sign-only add-on if Lanczos wins >= 5x.
- Write-back with same-capacity VMC control: upgrade only if the FN-target arm captures >= 0.5 of the gain AND beats plain VMC >= 1.5x at equal samples; tie -> drop the FN target (sign step and FN bound survive); < 0.1 -> cost case worsens.
- Referee: DMC - exact FN bias < 3e-6 at fixed protocol (M >= 512) and shrinking ~1/M -> FN bound usable at 1e-5; guide-dependent or non-shrinking -> withdraw DMC-based 6x6/8x8 differences.
- Surviving regimes and the evidence required: sign-poor guides (8x8+, >= 3x cheaper than complex VMC to the same energy, all costs, >= 3 seeds, calibrated referee); Krylov+Lanczos must beat two Lanczos steps; FN as a calibrated bound on a strong hop-free guide (8x8 below -0.49889 or first error-barred 12x12/14x14).
- Killed now: 4x4 equal-CPU claim; more write-back into the converged ViT; from-scratch cost race at 8x8/10x10 until the write-back + VMC-control test passes; histogram/MLE routes; "beat the ViT" and w_s as figures of merit.

## 3d. Lanczos baseline verdict (exact 6x6, results/lanczos_baseline_6x6/, 2026-10-07)
- From the symmetrised ViT psi_P (1.32e-4): one Lanczos step 2.55e-5, two steps 8.6e-6; one Krylov sign step 1.27e-4. Lanczos wins 20x in gain -> per the pre-registered threshold the sign-only add-on is KILLED. Krylov+Lanczos (3.6e-5) is worse than Lanczos alone.
- FN with the Lanczos-p1 guide: 1.89e-5 (below RBM+PP 4.5e-5). Candidate route: strong NQS guide + Lanczos step(s) + calibrated FN at 8x8+ (check novelty: FN with Lanczos-step guides exists for Gutzwiller states, Sorella 2001 / Becca et al.).

- PI caveat (2026-10-07): the 20x is one size. A fixed number of Lanczos steps is a global operation whose per-site gain is expected to shrink with N; the Krylov sign step acts per configuration and may stay size-intensive. Exact scaling test N = 16..36 running (results/lanczos_vs_krylov_scaling/). Result (results/lanczos_vs_krylov_scaling/, f3d10a9): up to N=36 one Lanczos step removes a fixed fraction (0.73-0.92) of the start error at every N, no shrinkage; the Krylov step removes only the sign part of the error (0 for sign-exact guides, 0.84-1.0 for Marshall-type starts, where it beats Lanczos by ~1.2x). No evidence of Krylov overtaking; Lanczos shrinkage expected only for N*dE > gap (N ~ 1e2-1e3), untested. Kill stands for good guides; the sign step matters only for sign-dominated guides.

## 3e. Write-back verdict (results/writeback/DESIGN.md, 2026-10-07) — FAILED ROUTE
- The FN amplitude update is white-noise-rough in the energy metric (0.97 of white noise) and tail-concentrated (81% of the gain on 7.6% of the weight). Exact oracle data: energy-metric residual net keeps 17% of one FN iteration, infidelity 8%; with realistic VMC samples < 1%, tying plain fixed-sign VMC.
- Decision: stop writing FN amplitude updates back (no loop, no 8x8 write-back). Amplitude from ordinary VMC; FN/Krylov for signs and as referee. Reopen only if a non-local pair-product residual or stronger tail tempering passes 50% on the same oracle harness.
- Best remaining route: strong (symmetrised) NQS + Lanczos step(s) + calibrated FN at 8x8+ (3d), pending referee calibration.
- 2026-10-08 REOPENED (PI + Research Workspace seq100): the loop's novel part is the alternating construction (Krylov sign -> flatten composite sign -> FN amplitude refinement, calibrated referee); NQS+Lanczos and NQS+FN alone are known (Chen et al. NeurIPS 2022; Wang/He/Lu PRB 113, 085120 (2026); NQS->FN-GFMC on Hubbard/t-J). Write-back failed with |psi|^2-type sampling; new test samples the ENERGY METRIC (tail proposal ~ violating/kept-edge weight) and fits edge differences with FN Laplacian weights. Reopen rule: >= 50% of one FN iteration on oracle data (results/writeback_tail/).

## 4. Decisions taken (and why)
- FN = signs + referee/bound; amplitude objective = fixed-sign VMC energy with a second-order optimizer (test A, 2026-10-06).
- Signs are stored as composite nets, always used with one exact hop (angle 2).
- FN referee protocol (calibrated 2026-10-07, results/fn_calibration_6x6/): tau_max 0.025, M=512, beta window 0.8-2.4, BETA-TIME-AVERAGED mean over >= 500 populations (bias <= 3e-6, SE ~3e-6). The old step-averaged estimator is biased by -11..+25e-6 (guide-dependent) and medians are biased: do NOT use either. The it2_fn DMC numbers (oracles, G2, K3vit, K3a1, ViT guide) are WITHDRAWN until re-run; ll6_fn runs can be re-evaluated from saved per-step curves.
- "Beat the ViT" is not a result by itself; 6x6 is a lab with exact references.

## 5. Parked ideas (not dropped)
- Walker-histogram / MLE amplitude fitting (no zero-variance property; ledger).
- 8x8 sign machinery alone (cheap; useful for the add-on pitch).
- Separate paper repo for Overleaf; paper rewrite after the angles settle.
- Possible NQS collaborators (ViT authors, CQSL).

## 5b. Compute note
- When Paderborn is back from maintenance (expected ~Oct 9-12), move GPU-heavy work there (H100s; ws1 A40s are slow and the queue is congested). Access via ws1 VPN; if login2 is down use `ssh -o HostName=login1 -o HostKeyAlias=login2 paderborn` (same host key). Check `sinfo` there first.

## 6. Running now
| Work | Owner | Budget |
|---|---|---|
| (nothing running) | | |
