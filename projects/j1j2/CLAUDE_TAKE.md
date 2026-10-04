# Claude's take on the FN–Krylov loop (2026-10-03)

Working notes from the paper-restructuring session. Opinion, not established results.

## Where the evidence is encouraging
- **Sign half is solved given a good amplitude.** Exact 4x4 amplitude + Marshall: 3–4 label-free Krylov steps reach the exact signs for every J2/J1 in 0.4–1.0 (paper Fig. 1). On 6x6 one step matches the ViT signs energetically (Fig. 3).
- **Amplitude half has a guarantee.** Lattice FN obeys E0 <= E_FN <= <H>_guide (ten Haaf et al.); the exact FN/Krylov loop reaches the exact ground state on 4x4 and the 20-site torus, from a start (J2=0 amplitude) where Krylov alone is stuck.
- **One half-iteration already scales.** 8x8, ViT amplitude + Krylov signs + FN: E/N = -0.49821, among the best published same-geometry values, and replica-stable.

## Worries
1. **The exact loop is slow (~100 iterations, power-law-like decay).** Cause: lattice FN moves forbidden elements to the diagonal weighted by guide amplitude ratios a(y)/a(x) (see `build_fixed_node` in `krylov_sign_structure/experiments/closed_fn_krylov_exact4x4.py`). So E_FN is exact only if the guide amplitude is already exact, and a_{k+1} = phi_FN[a_k, s_k] is a slowly contracting fixed-point map. Evidence: on 4x4 the sign error is flat at 1e-6 from iteration 20 to 99 while the energy keeps decreasing — only the amplitude is moving.
   (Correction to an earlier remark in the session: FN does NOT depend on the signs only.)
2. **Amplitude noise feeds into the signs.** With an approximate amplitude, Krylov returns the energy-optimal signs *for that amplitude*, not the true ones; with a poor amplitude the sign error can even increase (Fig. 2, amplitude from J2=0.6).
3. **8x8 iterations so far fail.** The iteration-1 FN gain was not reproduced by an independent rebuild; iteration 2 was worse than the start.
4. **Sampling.** An older sampled 4x4 run stalled near 1e-3 energy error; a clean rerun (histogram amplitude, energy-optimal threshold) is in progress.
5. **FN systematics not checked at 8x8.** Only M=128, beta=1.2, two replicas; population-control bias and projection-length dependence are unknown.

## Ideas
- **Accelerate the amplitude fixed point** (Anderson/DIIS mixing on log a over the last few iterates). Quick exact 4x4 test: iterations to reach the exact signs / a given energy error vs plain iteration.
- **Use the loop as a correction to a good network**, not as a solver from scratch: one or a few iterations from a strong NQS. The critical question becomes whether a learned amplitude can beat the ViT amplitude by a small margin so that one more iteration helps.
- **Track cost explicitly**: total = iterations x (FN walkers x projection length + learning samples). Measure each on 4x4 -> 6x6 -> 8x8 to get scaling, not single points.
- **Fair III.D comparison**: same ViT amplitude, guides Marshall / Krylov / ViT signs, energies before and after FN at matched M, beta, replicas.

## Update — end of 2026-10-03 session

New results (all in the paper branch `j1j2-organize`):
- **Repeated Krylov, exact 4x4 amplitude**: exact signs in 3-4 steps for all J2/J1 in 0.4-1.0 (Fig. 2).
- **Approximate amplitude (4x4, amplitude of another J2)**: energy still drops but plateaus; Krylov signs are the best signs for that amplitude, sometimes further from the true ones (Fig. 3).
- **Repeated Krylov, 6x6 ViT amplitude** (job 3576208, Paderborn CPU): step 2 does not lower the energy (+1.7(1.8)e-4/site), though it halves the sign disagreement with the ViT. One step already sits at an energy-stationary point of the threshold family. Step 3 not run (needs ~12-15 CPU node-h).
- **Anderson acceleration** of the exact loop: 3-4x fewer iterations (4x4: 100 -> 31; 20-site: 224 -> 59).
- **Geometric-mean update** a <- sqrt(a phi_FN) (what walker MLE gives): converges exactly 2x slower than a <- phi_FN.
- **Sampled 4x4 loop** (phi_FN from counts): floor in energy error ~1/N (1.3e-2, 1.7e-3, 2.2e-4 for 1e6, 1e7, 1e8 samples/iteration); real walkers worth ~3x fewer samples than i.i.d.
- **III.D matched comparison (M=128, beta=1.2)**: FN with Krylov signs is ~6e-4/site ABOVE the ViT itself on 6x6 and 8x8; FN with the full ViT guide reproduces the ViT energy. The ViT alone (-0.498823 on 8x8) beats every 8x8 literature value we compare to.
- **Possible FN bound violation on 6x6**: E_FN[Krylov guide] = -0.502984 lies above the guide's own variational energy (~ -0.5033..-0.5035). Either the two use different thresholds T or the FN runs are biased (beta / population). Check running (fn_bound_check_6x6).

Key insight for the learning step: guaranteed improvement needs <H>_{new guide} <= E_FN[old], not just <H>_new <= <H>_old. Train on the frozen <H_FN> (Rayleigh quotient of H_FN[a_k, s_k]); use the FN run as referee (gap <H_FN>_theta - E_FN) and as the reported energy. Ledger of all past attempts: AMPLITUDE_LEARNING_LEDGER.md.

Running at end of session: 6x6 symmetric ED (Opus), 4x4 learning-ladder rung 2 with frozen <H_FN> + SR from samples (Opus), 6x6 FN bound check (Sonnet), related-work literature check (Sonnet).

## Update — 2026-10-05
- **Loop iterations vs N (exact, 16..36 sites)**: iterations to eps<1e-6 ~82-104 plain, 32-57 Anderson; to w_s<1e-8 ~89-125 plain, 28-70 Anderson. No visible growth with N. Only sign errors on configurations of weight <1e-12 take longer at larger N. (krylov_sign_structure/results/closed_fn_krylov_6x6/)
- **6x6 baseline corrected again**: Krylov-guide FN = -0.503100(40) from 76 populations (8-population -0.503288 was low). FN lowers the guide by 2.6(6)e-4; 6e-4 above the ViT.
- **6x6 learned loop, iteration 1**: E_FN = -0.503667(63), equal to ViT within errors; true sign error 8e-5 < ViT 1.3e-4. Control: second Krylov step alone -0.503596(52) -> most gain from the sign step.
- **Real scaling problem found**: recursive sign definition — each Krylov step adds a neighbour hop (1.3e3, 5e4, 1.2e6 configs per state), iteration 2 would cost >50 GPU-h. Fix to try: store/distill signs after each iteration (sign network or per-iteration sign table on samples) so evaluation cost stays at one hop.
- FN at M=128 is noisy (single populations scatter ~3e-4/site, one fell below E0): need many populations or larger M for any claim.
