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

## 3e. Write-back verdict (results/writeback/DESIGN.md, 2026-10-07) — INCONCLUSIVE (Adam optimizer, not converged; re-test with SR/minSR/Gauss-Newton running)
- The FN amplitude update is white-noise-rough in the energy metric (0.97 of white noise) and tail-concentrated (81% of the gain on 7.6% of the weight). Exact oracle data: energy-metric residual net keeps 17% of one FN iteration, infidelity 8%; with realistic VMC samples < 1%, tying plain fixed-sign VMC.
- Decision: stop writing FN amplitude updates back (no loop, no 8x8 write-back). Amplitude from ordinary VMC; FN/Krylov for signs and as referee. Reopen only if a non-local pair-product residual or stronger tail tempering passes 50% on the same oracle harness.
- Best remaining route: strong (symmetrised) NQS + Lanczos step(s) + calibrated FN at 8x8+ (3d), pending referee calibration.
- 2026-10-08 REOPENED (PI + Research Workspace seq100): the loop's novel part is the alternating construction (Krylov sign -> flatten composite sign -> FN amplitude refinement, calibrated referee); NQS+Lanczos and NQS+FN alone are known (Chen et al. NeurIPS 2022; Wang/He/Lu PRB 113, 085120 (2026); NQS->FN-GFMC on Hubbard/t-J). Write-back failed with |psi|^2-type sampling; new test samples the ENERGY METRIC (tail proposal ~ violating/kept-edge weight) and fits edge differences with FN Laplacian weights. Reopen rule: >= 50% of one FN iteration on oracle data (results/writeback_tail/).

## 3f. Tail concentration vs N (results/tail_vs_N/, 2026-10-08)
- For loop guides the weight fraction carrying half the FN gain is flat in N (12-20%, 16..36 sites); it is NOT a size trend. But the number of configurations to be corrected explodes (82 orbits at 16 -> 1.6e7 orbits at 36), so the 4x4 sampled success was memorisation of the whole support; at 6x6+ the tail must be learned by generalisation.
- The better the guide, the more tail-heavy the gain (k=1,3,8 at 36: 19.9 -> 11.6 -> 7.1%; the symmetrised ViT 1.0% carries 50%). Write-back is hardest near convergence, easiest from mediocre guides.
- Decisive next: the learnability test (held-out tail orbits vs shuffled-target control), running in writeback_tail.

## 3g. Imaginary-time fitting with H_FN (results/itfit_6x6/, 2026-10-08) — INCONCLUSIVE, cause identified
- Stiffness SOLVED: lambda_max(F) = 1.6e6; explicit steps hopeless, but a semi-implicit step (wall implicit, kept hops explicit; local, positive) captures 87% of an FN iteration in one step, 97% in two; the unprojected loop with it tracks the ideal loop within 5%.
- Fitting a zero-hop residual CNN: <= 3.7% (oracle 8.3%), not converged. Cause: the FN target is an explicit function of ONE-HOP quantities of the guide (kept weight W, wall V, H_xx); a zero-hop net cannot represent it.
- Net + one exact semi-implicit hop recovers 82% of iteration 1 (<H> 1.32e-4 -> 8.9e-5, E_FN 1.02e-4 -> 7.10e-5), then stalls (the stored net cannot carry later changes).
- Next: give the network one-hop inputs from the frozen base (f = g(x, log a_P, W_P, V_P)): same 'net + one hop' cost as the signs, no recursion. Exact pre-test (>= 50% per iteration) approved and running.

## 3h. Learnability of the tail correction (results/writeback_tail/, wtKA1a vs wtKB, 2026-10-08) — POSITIVE
- 20% of orbits held out with all their edges removed: held-out tail capture 20.7% = training tail 20.9%; shuffled-target control 0.4%. The tail correction is STRUCTURED and the net GENERALISES; no memorisation gap.
- Limit = representation precision (~a quarter of each weight decade, falling with depth), not generalisation, not noise. Next: one-hop guide inputs (wtKA2, running) and the tail expert (wtEXP, running); itfit predicts the inputs W, V, H_xx are what is missing.

## 3i. WRITE-BACK SOLVED on oracle data (results/writeback_tail/, wtKA2, 2026-10-08)
- A 37k net with three guide-local one-hop inputs (log a, V = FN wall term, W = kept-edge weight) captures 88.7% of one exact FN iteration (quadratic 89.2%), 93% of the <H> gain (1.319e-4 -> 8.86e-5; exact FN step 8.41e-5), ~90% in every weight decade down to 1e-14, held-out tail 89.7% = training 89.8%. Copy error 0.0027 rms (< 0.004). Spin-only nets: 20-31%; tail expert and edge sampling did not help.
- Cause of all earlier failures: the FN correction is a function of one-hop guide quantities; spin-only nets cannot compute them. Same 'net + one hop' structure as the stored signs.
- itfit route (results/itfit_6x6/, e8e72c2): realistic tempered samples, 5k net on frozen-base one-hop features, semi-implicit FN target: 95% / 80% / 69% per iteration; E_FN 1.02e-4 -> 4.74e-5 in 3 iterations (ideal 3.81e-5); with one exact hop <H> 4.53e-5 vs RBM+PP 4.47e-5. Same-model VMC control NOT yet run (open condition).
- Next: pre-registered Step 2 (realistic samples, no phi_FN; same-input fixed-sign VMC control; pass >= 0.5 and >= 1.5x control; guide-evaluation cost per sample -> 8x8), then 3-iteration jump and the 6x6 loop vs RBM+PP. itfit route merged into this.

## 3j. Step 2 PASSED (realistic data, results/writeback_tail/ section 6b, 284e623, 2026-10-09)
- No phi_FN anywhere: semi-implicit FN target T = log(1+W) - log(1+(H_xx+V-E_a)) from the guide only, edge least squares, Adam: 73.4% of one FN iteration (<H> 1.319e-4 -> 1.032e-4), held-out tail = training tail (75.4%).
- Same-input VMC controls (frozen-FN VMC and fixed-sign <H> VMC, Adam and minSR) LOSE energy (-15%) or diverge; itfit's same-model VMC control also fails. Cause: a realistic VMC energy estimate scatters ~200x the per-iteration gain, while the FN step gives a zero-variance per-configuration target. This is the loop's genuine edge over VMC.
- Open caveats (monitored): smaller-step VMC controls (wtS2c) running; minSR destroys the bulk on the regression (Adam works); itfit loop vs RBM+PP: 4.53e-5 vs 4.47e-5. Next: multi-iteration loop with the FEAT net (queued), then 8x8 cost/feasibility.
- 6x6 LOOP with realistic write-back (6c, 6f3e6d4): after 5 iterations exact E_FN 3.96e-5 < RBM+PP 4.47e-5; <H> 4.64e-5 (RBM+PP 4.47e-5). Needs ~5 iterations where the ideal loop needs ~3. CAVEAT for scaling: here each new guide is stored as an exact 6x6 table la_{k+1} = la_k + f; with current-guide features a real callable chain recurses (k hops after k iterations) -> amortising the guide stack is the next engineering problem (frozen-base features avoid recursion at 0.95/0.80/0.69). 8-iteration run (wtLOOP8, f37f810): <H> passes RBM+PP at iteration 6 (4.03e-5); after 8 iterations <H> 3.16e-5, E_FN 2.75e-5 (exact, 6x6). Loop runs at ~half the ideal speed. 8x8 plan: results/writeback_tail/SCALING_PLAN.md (~40-60 H100-h; options B frozen-base features / C re-basing by distillation / D caching; 6x6 pre-tests first).

## 3k. Review section E (dec53f5) — accepted points
- Critical: one exact Lanczos step on psi_P (callable, one hop, no training) gives <H_ph> 2.55e-5 / E_FN 1.89e-5, better than the table loop after 8 iterations (3.16e-5 / 2.75e-5). Quote all loop numbers against Lanczos at equal hop count; the fair test is 'loop guide + Lanczos' vs 'ViT + Lanczos'.
- 'Passes RBM+PP' is a TABLE result; the only callable variant (option B) stops at 5.6-6.2e-5.
- Needed: >= 3 seeds; amplitude-only trajectory (no exact Krylov sign); supervised-Lanczos control (same net regressed on log(1+alpha(E_L-E))).
- Scaling: distillation not ruled out in principle (target is a function of x alone; the obstacle is log-ratio precision ~3e-3 on ~135 bonds). Avoid message passing on the spin-swap graph and the warm ViT; prefer a bond-output ratio net or input-gradient features. Pre-tests: V/W sensitivity, ratio fidelity per decade, end-to-end callable 3/6 iterations with 3 seeds (pass: >= 70% of table gain, <= 3 base passes per configuration, callable + Lanczos >= 2x below 2.55e-5).
- Running now: Lanczos-on-top, supervised-Lanczos control, seeds, w_s along the loop.

## 3l. Compression pre-tests (results/compress_6x6/, 8e72a04, 2026-10-09) — INCONCLUSIVE (cause: optimisation + precision)
- P1 precision target: one write-back step keeps >= 90% only if bond log-ratios are accurate to 0.003 rms (V, W inputs to 0.001). psi_P itself is 2-100x outside this tolerance against la_3 (0.005 at 1e-7, 0.3 at 1e-14).
- P2: no student (warm ViT with Adam or damped Gauss-Newton, scratch ViT, bond-ratio net) moved toward la_k: <= 10% kept; held-out = train (generalisation fine), precision is the limit.
- Lanczos context (6x6): table stack k=6 + 1 Lanczos 1.55e-5 vs psi_P + 1 Lanczos 2.55e-5, but psi_P + 2 Lanczos 8.6e-6. At equal hop count Lanczos still wins on 6x6; the scaling case rests on Lanczos gains shrinking at larger N (untested <= 36 sites) and on the 8x8 test.
- Options: (a) cap depth (2 route-B iterations + Lanczos + calibrated FN, reviewer E6); (b) accept k=2-3 hops with caching on H100s; (c) input-gradient features (reviewer E4) — first check cheaply whether linearised log-ratios can reach the 0.003 tolerance at all.

## 3m. Compression round 2 (results/compress_6x6/, ad75e84, 2026-10-11) — INCONCLUSIVE, close to the bar
- Constant-hop loop (re-base every iteration into a frozen-base-feature net; cost per sample independent of k): <H_ph> 7.0e-5 after 6 iterations (2 seeds), 67% of the table loop's gain (bar 70%), then plateau. Write-back fine (0.62-0.78 per iteration); the re-base loses the new gain (0.95 -> ~0 per iteration) while keeping 80-84% of the total. Loss floors at ~5e-6 independent of size/steps -> limit = information in the base features (sums V_P, W_P).
- Structured errors are 1.5-2x more tolerable than white noise (students); re-base errors less tolerable (rare large errors).
- Amortised V/W from base: 0.894 at k=3, 0.48 at k=6.
- Equal-hop Lanczos on 6x6 still not beaten (guide + 1 Lanczos 2.5e-5 = psi_P + 1 Lanczos).
- Next (approved): per-bond log-ratio inputs (144 bonds, bond-equivariant layer) at unchanged one-hop cost; rolling base as fallback.

## 3n. First 8x8 test (results/loop_8x8/, f030824, 2026-10-11) — INCONCLUSIVE (control unresolved)
- Projected (D4 x flip) 8x8 ViT guide -0.498856(29) (paper ViT -0.498823(35)). One FN write-back (route B, realistic): -0.498884(24), Delta -2.84(0.59)e-5 (4.8 sigma) = RBM+PP -0.498886(1) within errors. Best same-sample VMC control (single-pass minSR) -3.87(2.21)e-5: not separable; Adam VMC controls gain ~0.
- One Lanczos step on the projected ViT: -0.498962(29), 2.6 sigma BELOW RBM+PP (sampled-moment estimate; confirming by direct VMC). Lanczos gains 3.7x more than one write-back at equal hop cost.
- Cost 12.2 GPU-h on A40 slices; calibrated FN-DMC unaffordable here (~6 GPU-h per population on the written-back guide).
- Next (running): confirm the Lanczos lead by direct VMC; loop guide + Lanczos vs ViT + Lanczos (paired); resolve the control.

## 3o. Review-E validations on 6x6 (results/writeback_tail/ 6c/6e, b03d683, 2026-10-11)
- Sign error along the loop: w_s 1.11e-4 -> 1.98e-5 (it 1) -> 4.50e-6 (it 8), 1.2-2.6x above the ideal loop.
- Lanczos on top: loop guide k + 1 Lanczos step: 3.03e-5 (k=1) -> 1.26e-5 (k=8); beats psi_P + 1 Lanczos (2.55e-5) from k=3. At EQUAL hops Lanczos on psi_P still wins (table result).
- Seeds (5): <H_ph> after 3 iterations 6.54(30)e-5; capture spread up to 13 points.
- Amplitude-only (no Krylov sign step): 7.34e-5 vs 6.15e-5 after 3 iterations -> the sign step is worth ~17% of the gain.
- Supervised-Lanczos control diverges: the one-hop Lanczos factor |1 + alpha E_L| is unbounded on deep-tail configurations, so it is unusable as a write-back target; the bounded semi-implicit FN step works (structural advantage of the FN target; bounded resolvent-form control not run).

## 3p. Compression round 3 (results/compress_6x6/, 60a9bbe, 2026-10-12) — CONSTANT-HOP LOOP PASSES
- Residual re-base G_{k+1} = G_k + h_k (fit only the new increment, psi_P one-hop scalars, increment-weighted sampling): at iteration 6 the callable guide keeps 82% / 85% of the table loop's gain (2 seeds; bar 70%). <H_ph> 5.68e-5 / 5.45e-5 at it 6, 5.53e-5 / 5.04e-5 at it 8 (table 3.16e-5); gap grows late (bad re-base fits).
- Cost per sample independent of k for hops (1 hop of psi_P for the guide, 2 for V/W, 3 per training sample), plus k small nets.
- Guide + 1 Lanczos = 2.0e-5 at it 8 (< psi_P + 1 Lanczos 2.55e-5); at equal hops psi_P + 2 Lanczos (8.6e-6) still wins on 6x6.
- Approved per-bond and rolling-base arms failed their pre-tests (0.08-0.21): the limit was the fit of the whole accumulated correction, not missing information.
- (done) robust residual re-base (re-seed dead fits, held-out acceptance, 40k steps), 3 seeds; then 8x8 with the residual-re-base loop + Lanczos comparison.
- Round 4 (2026-10-12, robust residual re-base, 3 seeds, PASS): 85% / 86% of the table loop's gain at it 6 / it 8. <H_ph> at it 8: 5.10 / 4.23 / 4.40e-5 (seeds 1-2 below RBM+PP 4.47e-5). Robustness: clipping removed dead fits; 2 stalls re-seeded; worst re-base retention 0.18 (round 3: -0.63). Open weakness: acceptance only requires 'better than nothing' (one accepted write-back captured 0.01).
- Equal-hop comparison (guide = 1 hop of psi_P; each Lanczos step +1 hop): 2 hops: guide+1L 1.8-2.0e-5 vs psi_P+2L 8.6e-6; 3 hops: 7.0-8.1e-6 vs 3.8e-6 -> at 6x6 plain Lanczos wins ~2x at equal hops. Limit of the loop: late re-bases keep only 0.2-0.7 of their iteration's gain.
- 8x8 follow-up (2026-10-12, results/loop_8x8 FU): psi_P + 1 Lanczos step = -0.4989530(164) combined, direct VMC of psi_1 -0.4989476(231) (2.7 sigma below RBM+PP -0.498886; needs independent review). One write-back adds no resolvable gain beyond a Lanczos step (-0.6(3.3)e-6): the FN target is 97% collinear with the Lanczos direction, T ~ -(E_L - E)/(1+W) = a Lanczos step with per-configuration step 1/(1+W). FN vs best VMC control: +0.9(1.5)e-5, unresolved (FN 40x more reproducible). Consequence: the loop's value must come from compounding iterations at constant hop cost (k loop iterations vs p Lanczos steps), not from one write-back.
- Step-size test stage 1 (2026-10-12, cx_step, exact tables): a clipped-Lanczos target d = log max(|1+alpha(E_L-E)|, 0.3), sign sgn(f), gives per iteration what a Lanczos step gives: 2.62e-5, 1.16e-5, 7.06e-6, 4.91e-6 after k = 1..4 (vs current SI target 9.2e-5 ... 4.4e-5; table loop 8 its 3.16e-5). Pre-registered bars: k=1,2 pass, k=3 fails because the bar used the joint 3-step polynomial (exact iterated Lanczos also fails it). Roughness proxy fails only at k=1 (first step flips 24% of configs). The SI target is over-damped by a factor that grows with N (log difference of extensive quantities). Next: stage 2, network loop with the clipped-Lanczos target.
- Long table run (cx_steplong, 16 exact iterations, no networks): new target (clipped Lanczos) eps = 1.2e-6 relative after 16 iterations (6.1e-7/site), still falling ~12%/iteration, no stall; exact one-step Lanczos 1.17e-6; old target 1.5e-5. For scale: DMRG(8192) 1e-5, RBM+PP 9e-5. This is the ideal (exact tables); the network loop must follow it (stage 2 running).
- 8x8 stored sign net + one exact hop (signs_8x8, c5b7eae): from Marshall PASS (net+hop beats exact K1 by -5.2(1.1)e-4/site, closes 98.5% of the Marshall->ViT-sign gap vs 88% for K1; power clause missed, bound clears the line 30x); from the ViT sign inconclusive (cause: one rare high-E_L sample dominates the SE; without it +0.9(2.5)e-6). Storage of a Krylov/Lanczos-type sign step at one hop per step works at 8x8 (K2 exact would cost ~1e4 ViT calls/config). The global Krylov step on the ViT sign is null at 8x8.
- Local vs global Krylov (local_krylov, 63c1172): (a0) decoupled 4x4 copies (Marshall sign on |psi0|, exact): global-rule gain / one-copy gain 0.95 (N=64), 0.87 (128), 0.69 (256), 0.47 (384), 0.24 (512), 0.01 (768): N_1/2 = 368; local rule keeps the full gain at every N. (a8) 8x8 two-hop data: from Marshall the global rule beats C_4 (-3.96(24)e-3 vs -2.84(32)e-3/site); from the ViT sign all null. Cause: window-energy noise (sd 1.1 per 4x4 window vs 0.27 for the total in a good state) forces a high local threshold. Reading: size-consistency dilution is real but sets in at N ~ 300-1000 (17x17-32x32) for this defect density, later for better signs; at 8x8-14x14 the global step is not yet diluted. The local rule needs a lower-noise window statistic before ~16x16.
- 6x6 RBM+PP guides (rbmpp_6x6): own JAX RBM+PP (SR), converged 4.74e-5/site (published projection target missed by 1.8e-5, optimiser-limited); PP alone 6.3e-5 (2x better than our ViT, ~0.2 GPU-h to plateau). Table-level Lc_0.3 loop: 16 its from RBM+PP 2.98e-7/site; one Lc iteration = one Lanczos step from every guide; equal-hop joint Lanczos still 3-5x better. Loop speed depends on the guide, not only on its energy (early checkpoint B2 starts like the ViT, ends 5x higher).
- Residual locality + adaptive depth (2026-10-13, 9085e06, c2261bf): residual log(psi0/base) relative to PP, RBM+PP, ViT removes ~3x of the short-range non-additivity but stays non-local at distance >= 2.8 (rms 0.26-0.63, heavy tail from configurations near the base's nodes). Adaptive depth (second hop only at near-node flagged configurations): 2% of the weight recovers share 0.76-0.80 at iteration 2, but the flagged region branches with r ~ 50 neighbours per flagged configuration (bar < 0.5): sparse in weight, dense and connected in count. Overall verdict at J2 = 0.5: every constant-cost fold-back fails for one measured physical reason (non-additive amplitude/sign structure, also of the correction relative to the best bases, concentrated in a connected near-node region); each loop iteration costs one hop like a Lanczos step, and at equal hops Lanczos is better. Locality returns only at weak frustration (J2 <~ 0.3).
- Route B + locality vs J2 (2026-10-13, 7c657f3): spin-only tail net on top of the re-anchored ViT learns 4-5% of the normalised tail residual (share -0.23/-0.25; same net with one exact hop of inputs 0.96); input-gradient (zero-hop) features useless (corr with exact E_L <= 0.11). Verdict: inconclusive (cause: spin-only inputs cannot represent the one-hop structure of the rare-config residual; capacity/optimiser ruled out). Locality vs J2 (exact 4x6 ground states): double-swap interaction rms at distance >= 2 is 6-13x larger at J2 = 0.5 than at J2 = 0; local at J2 = 0, nearly at 0.3, non-local at 0.4-0.5, smaller again at 0.6. Statement: fold-back locality tracks frustration; constant-hop loop plausible only deep in the Neel regime.
- Constant-hop fold-back, 2026-10-13 (design partner, exact 6x6 + 8x8 data; results/compress_6x6 README sections D1/D2, re-anchor, sign channel, hybrid, wrapper W0; results/locality_8x8): every constant-cost storage of a loop step fails from iteration 2. Re-base on psi_P one-hop features 0.02 (0.99 with the guide's own E_L = a second hop); rolling ViT re-anchor -0.1 (precise on common configs, wrong on a^2 < 1e-9, where the gain sits); hybrid ViT + one-hop correction 0.96 then 0.23 (0.92 with the second hop); Lanczos-wrapped network: not sound (rare-tail errors of earlier steps must still be stored). Locality: the double-swap interaction I(b,b') does not decay - rms 0.5-0.7 at distance 3-5 on 8x8 (same as the exact 6x6 ground state), weighted 0.16-0.40, sign non-additivity 4%; light cone exact to distance 4 still 5x above tolerance. Verdict: constant-hop compression with fixed-radius features FAILS at J2 = 0.5 (cause: non-local amplitude/sign structure of the near-critical state, measured); spin-only representation INCONCLUSIVE (cause: precision on rare configurations). Fallback in force: guide <= 2 hops + Lanczos polish; at equal hops Lanczos still wins at 6x6. Decision for the PI: (B) dedicated tail network or (C) ViT + 1-2 Lanczos + calibrated FN at 8x8-14x14 as the deliverable.
- AMENDMENT to the 2026-10-13 decision (local_krylov F1, 51fcbbf; exact decoupled 4x4 copies, 6 guide qualities): for ViT-quality guides one global step is NOT diluted through 14x14 (r(192) = 0.95-1.00 for Krylov, Lanczos, clipped Lanczos); halving sizes N_1/2: Krylov ~0.24/(w_s per site) (6e3-2.5e4 for ViT-like signs, 364 for Marshall); Lanczos N_1/2 ~ 2 kappa3^2/kappa2^3 (per-site energy cumulants; 4e3 to >6.5e4 for ViT-like guides, ~1e3 for Marshall); 8x8 ViT estimate ~6e4 (order of magnitude, heavy-tailed kappa3). So size consistency binds only at ~1e3-1e4 sites (~400 for sign-poor guides). Up to 14x14 the global Lanczos/Krylov-type engine is adequate; the bottleneck there is representing the amplitude (fold-back; rare near-node configurations, which also control the lattice E_FN via the wall term).
- DECISION 2026-10-13 (PI + lead, conceptual, size consistency): the loop's engine must act block by block (product over independent blocks), else its per-site gain fades with N. Global-alpha Lanczos / clipped Lanczos: fails -> final polish only, not the engine (6x6/8x8 results with it are diagnostics of the write-back machinery, not the method). Local log step e^{-tau(E_L-E)} + fit = SR-VMC, no edge. Real FN projection: size-consistent and reaches tau -> infinity for the given signs in one shot = the loop's edge; the one-hop SI target is a non-size-consistent stand-in (1/(1+W)) -> replace by fitting the network to FN walkers. Krylov sign step with one global threshold on E_L fails (local sign signal O(1) buried under O(sqrt N) fluctuations). Open design problem: a local (patch-wise) sign update. Next: design with Opus partner + consultants before any run.
- Step-size stage 2 (2026-10-13, cx_chop3, 2 seeds x 4 its, results/compress_6x6 'Step size, stage 2') - FAIL at the
  pre-registered bar (share of the exact clipped-Lanczos step kept >= 0.7 every iteration): 0.84 / 0.66 / 0.31 / 0.09.
  - k = 1 works: the callable guide reaches <H> 4.3e-5 in ONE iteration (round 4 needed 8). This needs asinh(E_L) as
    a feature in both nets (write-back capture 0.25 -> 0.84).
  - Plateau ~2.0e-5 at k = 4. At 1 hop it beats psi_P + 1 Lanczos step (2.55e-5); at 2-3 hops it ties or is 1.25x worse
    than psi_P + 2-3 Lanczos steps.
  - Cause: the write-back is near-exact from k = 2 (capture ~1.0). The one-hop-of-psi_P re-base has a fixed absolute
    error (~0.013 bond-rms) while the increments shrink, so retention falls to 0.09.
  - Sign-matched re-base features: refuted (0.55 at k = 2). The increments need two-hop information of psi_P (the
    guide's own neighbour ratios).
  - Next: two-hop fixed anchor (G_1's V, W, E_L) for the re-base, or k <= 2 hop guides; the decisive scaling test is
    k = 1 at 8x8.
- Lanczos-polish bump after loop iteration 1 (cx_l1bump, exact, table-loop it 1): the hard Krylov sign flip (18% of orbit reps flipped) raises the energy variance and makes the 1-step polish WORSE (ViT amp + Krylov sign: <H> 1.27e-4, +L1 3.6e-5 vs ViT+L1 2.55e-5), while the written-back amplitude alone helps (new amp + old sign: +L1 2.22e-5). Combined (it 1): +L1 3.03e-5. Later write-backs shrink amplitudes near the new nodes and recover. Idea to test: damped flip (scale down the amplitude of flipped configurations).
- Next: (a) raise re-base retention (the bottleneck), tighten acceptance to a fraction of the ideal step; (b) 8x8 with the loop + Lanczos at equal hops, where the hop count, not k, sets the cost.

## 3q. RBM+PP as the guide (results/rbmpp_6x6/, 2026-10-13)
- Own JAX RBM+PP (published ansatz, SR). Exact tables (full projection): converged RBM+PP dE 4.74e-5 (= published
  level; with the published simple projection only -0.503727 VMC, target -0.503745 missed by 1.8e-5, optimiser-limited);
  PP alone 6.3e-5 (2x better than our ViT, ~0.2-1.3 GPU-h on a 2080 Ti); early checkpoints B2 1.54e-4, B1 3.87e-4.
- Lc_0.3 table loop (cx_step, 16 it): one iteration = one Lanczos step from every guide. From RBM+PP: 4.74e-5 ->
  2.98e-7 (rel 5.9e-7) after 16; ViT 6.1e-7; PP 8.7e-7; B2 2.9e-6 (same start as the ViT, 5x slower: guide-dependent
  rate, cause open). Equal hops: joint Lanczos still 3-5x better at 4 hops. Network loop on these guides not run.
- Cost 25.4 GPU-h (brief 15): 6.5 h on two plateaued RBM+PP runs that located the optimiser cause.

## 3r. Deliverable C round 1 (results/deliverable_C/, 2026-10-19) — known method, new 8x8 numbers
- Literature: NQS + Lanczos published only at 6x6/10x10 (Chen 2022, Wang-He-Lu 2026); no NQS-guided FN/GFMC for J1-J2.
- psi_P + 1 Lanczos step, direct VMC (11264 samples): -0.498971(12) (late-time only -0.498958(22)), 8.5e-5 below RBM+PP.
- Variance extrapolation p = 0, 1 (exact-6x6-calibrated, sys 8e-6): E0(8x8) = -0.498992(16)(8); in 3.4-sigma tension with
  Hu et al.'s -0.49906(1) (late-only 2.9 sigma). Heavy tails checked (trim, second estimator, late-time run).
- FN-DMC with the psi_P guide (new cached-ViT referee; 4x4 gate passes after the 1/M population-bias extrapolation):
  -0.498889(11), 8 populations; FN gain -4.2(1.5)e-5 over psi_P; equals RBM+PP, above psi_1.
- p = 2: three-hop E_L costs 86 s/sample on a slice; 3x3 Ritz not estimable from a 64-sample pilot -> repeated step f^2
  (keeps 78% of the optimal p2 gain at 6x6). Production ~130 H100-h for paired SE 5e-6 (Paderborn draft).
- 10x10: our ViT code loads the HF architecture (434,760 params = card); weights need the user's OK to download.
- 12x12/14x14 plan (PLAN.md): RBM+PP guide (own JAX code) + Lanczos + FN as the cheapest decisive route; ViT as add-on.
- H100 NVL exists on ws1 inter (2.36e4 projected evals/s, 4.5x an A40 slice); using it needs the PI's OK.

## 4. Decisions taken (and why)
- RULE (PI, 2026-10-08): a route is declared failed only when the failure is understood and pinned to a conceptual limit that cannot be fixed. Before that: diagnose (optimizer, convergence, sampling, literature) and try the better method. Section 3e write-back verdict is downgraded to INCONCLUSIVE (cause: Adam, not converged).
- FN = signs + referee/bound; amplitude objective = fixed-sign VMC energy with a second-order optimizer (test A, 2026-10-06).
- Signs are stored as composite nets, always used with one exact hop (angle 2).
- FN referee protocol (calibrated 2026-10-07, results/fn_calibration_6x6/): tau_max 0.025, M=512, beta window 0.8-2.4, BETA-TIME-AVERAGED mean over >= 500 populations (bias <= 3e-6, SE ~3e-6). The old step-averaged estimator is biased by -11..+25e-6 (guide-dependent) and medians are biased: do NOT use either. The it2_fn DMC numbers (oracles, G2, K3vit, K3a1, ViT guide) are WITHDRAWN until re-run; ll6_fn runs can be re-evaluated from saved per-step curves.
- "Beat the ViT" is not a result by itself; 6x6 is a lab with exact references.

## 5. Parked ideas (not dropped)
- Tail-steered FN walkers (importance function flatter than the guide, e.g. a^beta or a mixture, reweighted): needed at 8x8+ for tail data; queued AFTER the 6x6 oracle tests show the tail is learnable (>= 50%). Risk: weight fluctuations grow with N (ESS 5% at beta=1/2 on 6x6) -> use a mixture importance function.
  Recipe (Chatty, seq102): freeze F = H_FN[a,s] INCLUDING the violating-edge diagonal sum_bad K_xy a(y)/a(x); introduce a separate positive importance g only for kept-edge rates -F_xy g(y)/g(x) and E_loc^g = F_xx + sum_kept F_xy g(y)/g(x) (branching, adaptive tau). Never put g into the bad-edge diagonal (it changes the FN Hamiltonian). Walkers then sample g*phi_FN; old a*phi_FN samples cannot be reweighted into missing tail support. First test on an exact 4x4 frozen operator vs ED (energies, tail histograms, ESS/diversity), then 8x8. Current K1FNEngine does not support it.
- Walker-histogram / MLE amplitude fitting (no zero-variance property; ledger).
- 8x8 sign machinery alone (cheap; useful for the add-on pitch).
- Separate paper repo for Overleaf; paper rewrite after the angles settle.
- Possible NQS collaborators (ViT authors, CQSL).

## 5b. Compute note
- When Paderborn is back from maintenance (expected ~Oct 9-12), move GPU-heavy work there (H100s; ws1 A40s are slow and the queue is congested). Access via ws1 VPN; if login2 is down use `ssh -o HostName=login1 -o HostKeyAlias=login2 paderborn` (same host key). Check `sinfo` there first.

## 6. Running now (2026-10-09)
| Work | Owner | Budget |
|---|---|---|
| Validations from review E: Lanczos on top of loop guides, supervised-Lanczos control, 3 seeds, amplitude-only loop, w_s along the loop (results/writeback_tail/) | Opus | <= 4.5 GPU-h |
| Compression pre-tests: V/W sensitivity; distil wtLOOP8 stack into ViT (fresh/warm), bond-ratio net, one-layer edge aggregation (results/compress_6x6/) | Opus | <= 8 GPU-h |
| 8x8 one-iteration route-B test with same-sample VMC control, calibrated referee (results/loop_8x8/) — PI go 2026-10-09 | Opus | <= 20 GPU-h |

## 7. Pending at 2026-10-08 evening (usage limit hit)
- Queued on ws1 (writeback_tail, results land in /project/theorie/a/A.Otaifi/chatty_writeback_tail/runs/<job>_<id>/wt.json; expected starts 20:45 -> 05:40): wtC48 16948146 (63k), wtC96 16998622 (250k), wtEXP 16961949 (tail expert), wtKA 16976428 (learnability, real target, neighbourhood inputs, edge sampling), wtKB 17009145 (shuffled control), wtKGN 16976430 (global Gauss-Newton). Nobody folds them in: resume the writeback_tail worker or read the wt.json files directly. Low fair share because other campaigns hold ~730 running jobs; move to Paderborn when back.
- itfit_6x6 (imaginary-time fitting with H_FN) worker was running; check results/itfit_6x6/.
