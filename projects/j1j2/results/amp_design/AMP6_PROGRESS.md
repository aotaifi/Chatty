# AMP6: learning the 6x6 amplitude net at large N (progress log)

Goal (PI): unblock amplitude learning on 6x6. Plan from the coordinator, 2026-10-07:
- A. Zero-hop composite sign net for the amplitude phase.
- B. Amplitude steps at N = 1e5-5e5.
- C. Subspace restriction if needed.
- D. Same-optimizer control with the ViT's own signs.

Gate: one large-N amplitude iteration must give a verified gain > 2 sigma. Budget <= 25 GPU-h.

## Critique of the diagnosis (07:30)
**Agree.** The cost driver is the hop sign. Local data with a hop needs the ViT on the 2-hop shell, about 3e3-7e3 evaluations per sample. A zero-hop stored sign needs about 83 ViT plus 83 CNN evaluations per sample, so N can grow 40-80x at the same cost.

**Three caveats.**
1. **Large N rules out sample-space minSR.** At N = 1e5 the Jacobian is 62 GB and K is N x N, so neither can be stored. Directions must be built matrix-free in parameter space:
   - the gradient (one vjp);
   - SR by CG with S v = J^T W J v (jvp + vjp per matvec);
   - RGN with neighbour jvp/vjp (about 80x the sample cost per matvec).
   T3 already showed that only gradient-like / heavily shifted directions generalise. With N >= P, unshifted SR should start to generalise too. That is the falsifiable test of (ii).
2. **(i) is not only a sample-size problem.** In iteration 2 the gradient norm fell from 4.5 to 0.9 and every candidate measured <= 1.3e-6/site. Larger N resolves gains of 1e-6 (SE ~ N^-1/2: N = 4e5 -> ~5e-7). But a 2-sigma energy win over the ViT needs ~4e-5 cumulative, i.e. tens of verified steps, if the gain exists at this capacity.
   - That is why D matters: the ViT was trained with SR at much smaller N per step, so the same large-N optimiser with the ViT's OWN signs may gain comparably. Any energy gain must be split into sign and optimiser parts.
3. **A is the riskiest part.** Angle 2 shows a composite net alone carries 10-30x more error than the same net plus one hop. Our composite net N1 alone had w_s 1.1e-4, and with a hop 2e-5.
   - Reaching w_s <= 4e-5 without a hop needs more capacity and longer training: 64x6 (~186k) and 128x6 (~0.74M) run in parallel. Sign errors sit at low |psi|, so the paired <H> criterion (within 1e-5) is the operative one.
   - If A fails, B can still run with the composite net as a fixed sign (w_s ~1e-4). That costs the sign gain but tests (i)-(ii) cleanly.

## Log
- 07:40 **A:** composite distillation of the best sign s2 = N1 + hop(a1) into one net.
  - Training data: beta 0.5 samples of a1 plus all neighbours (2.66e6 states), 8e4 steps.
  - Two capacities: zM = 64x6 (16853971), zXL = 128x6 (16853972), both on 2080Ti.
  - Checks: ED w_s, plus paired <H>_(a1, net) - <H>_(a1, hop) and both vs ViT on 16384 a1^2 samples.
- 08:00 **B code** (`amp6_large.py`): zero-hop sign (stored net, or the ViT's own phase for control D), large N, everything matrix-free in parameter space.
  - Directions: the gradient (one vjp) and SR by CG with S v = J^T W J v (jvp+vjp).
  - Candidates are measured on independent fresh tempered samples, with paired SE. The train-vs-fresh RMS(delta log a) ratio is logged.
- 09:02 **Smoke test** (A40, N = 8192, sign = N1 zero-hop):
  - **Local data 24 s for 2 x 8192 samples, vs 870 s with the hop sign (36x faster).** The cost driver is confirmed.
  - The gradient candidate was accepted at -8.7(2.8)e-6/site (3.1 sigma). Fresh/train RMS ratio = 1.17 for the gradient, 1.27 for SR (eps 1).
- 09:05 **Gate runs** (A40, N = Nv = 1e5, 3 steps, from the ViT; certificates at 65536 and 131072 fresh samples):
  - B with the N1 sign (zero-hop composite net, w_s 1.1e-4): 16854230.
  - D control with the ViT's own sign: 16854231.
- 09:45 **Cancelled zXL** (128x6): 12.5 min/epoch on the 2080Ti, so 31 epochs would exceed the 6 h job limit (~5 GPU-h).
  - zM (64x6) reached validation error 0.39% (flips 1.3%) at epoch 18, vs 0.9% for the 48x6 N1.
- **Gate runs, step 1 (N = 1e5, A40, ~20 min/step):**
  - **Generalisation is solved at N = 1e5.** Fresh/train RMS(delta log a) = 0.94-1.08 for the gradient and 0.5-0.85 for SR (T3 at N = 8-16k: 4-11).
  - **B (N1 sign):** starts at <H> - E_ViT = +7.0(1.6)e-5. The N1 sign alone costs energy at the ViT amplitude.
    - Gradient candidates: -3.2(0.4)e-6 (z 8.2), -6.7(1.0)e-6 (z 7.1), **-1.02(0.22)e-5 (accepted)**.
    - SR candidates: -2.8 to -5.0e-6 (z 2.6-3.8).
  - **D control (ViT's own sign):**
    - Gradient: nothing to gain (-1.6(2.7)e-7; a larger step raises E).
    - SR finds gains: -2.7(0.7)e-6 (z 3.8) and **-4.85(1.85)e-6 (accepted)**. The ViT is not at the large-N SR optimum of its own fixed-sign energy, so D's gains must be subtracted from any loop gain.
- 10:15 **Gate runs finished** (1.20 + 1.12 A40-h):
  - **B (N1 sign):** steps 1 and 3 accepted (-1.02(0.22)e-5 and -0.46(0.20)e-5); step 2 none.
    - Certificate: <H> - E_ViT went from +7.0(1.6)e-5 to **+3.2(2.4)e-5**. Still above the ViT; the N1 zero-hop sign is too lossy.
  - **D (ViT sign):** steps 1 and 3 accepted (SR, -4.9(1.9)e-6 and -3.5(1.7)e-6).
    - Certificate: **-0.35(0.31)e-5**. The same optimiser improves the ViT itself by a few 1e-6.
  - After one good step, the gradient norm falls by 10-20x in both arms. Fixed-sign gains per step are then 1e-6..5e-6/site: resolvable at N = 1e5 (SE 4e-7 to 2e-6), but small.
- 10:39 **A accepted: zM (64x6 composite net, zero hop)** (2.53 GPU-h on 2080Ti; training 1.68).
  - ED w_s = 6(1.7)e-5 vs hop sign 2(1)e-5 (paired +4(1.7)e-5). This is marginal against the 4e-5 target.
  - **Paired <H>_(a1, zM) - <H>_(a1, hop) = -0.31(0.65)e-5/site: passes the energy criterion.** vs ViT: zM +0.49(0.75)e-5, hop +0.80(0.66)e-5.
  - Held-out disagreement with the hop sign: 0.44% at beta 0.5, 0 at beta 1.
  - Submitted B with zM from the ViT amplitude (same settings as D; step sizes 0.001-0.005 because the gains sit at small steps).
- 11:30 **Correction to the T3 "poor generalisation" numbers.**
  - The T3 ratio compared the *weighted* training RMS (the step target) with the *unweighted* RMS on verification states, so it was not a like-for-like measure.
  - With matched unweighted RMS, the N = 8192 smoke test gives fresh/train = 1.17 (gradient) and 1.27 (SR, eps 1).
  - The real T3 evidence of overfitting stands: minSR with a tiny shift (1e-3) *raised* the fresh-sample energy, by +1.3e-5 to +3.3e-3/site, while gradient and shifted-SR steps lowered it.
  - At N = 1e5, SR with shift 0.01 and with shift 1 give the same verified gains, so the shift no longer matters.
- Waiting for an A40 / 2080Ti slot for B_zM (16855760 / 16856048 duplicates; the first to start wins).
- 13:06 **B with zM, from the ViT amplitude** (16855760, 1.30 A40-h):
  - Certificate start: +1.11(0.69)e-5. Certificate end: **+0.67(0.90)e-5**.
  - Verified steps: -3.4(0.9)e-6 (z 4.0), -1.4(0.7)e-6 (z 2.1), then nothing (-0.4(0.4)e-6). The gradient norm collapses after step 1 (7.6 -> 0.4).

## GATE (13:10): PASSED technically, not in physics
- **What the gate asked for works.** Every arm has verified gains above 2 sigma at N = 1e5 (z up to 8). The fresh/train RMS ratio of the steps is ~1 (gradient 0.9-1.2, SR 0.3-1.0); the step cost is ~20 min on an A40.
  - **Learning the 6x6 amplitude is no longer limited by sampling or overfitting.**
- **What it shows.** At fixed sign the ViT amplitude sits within ~1e-5/site of its fixed-sign optimum. After one step the remaining gain per step is ~1e-6/site.
  - The control D (ViT's own sign) gains -0.35(0.31)e-5 with the same optimiser.
  - With the zero-hop zM sign the guide ends at +0.67(0.90)e-5. Its sign errors (w_s 6e-5) cost ~1e-5 at the ViT amplitude, and the amplitude recovers only part of that.
  - **The sign advantage is too small at this amplitude to show up in <H>.**
- **Compute:** 7.8 GPU-h this phase (distillation zM 2.5, zXL 1.6 cancelled, smoke 0.05, B_N1 1.2, D 1.1, B_zM 1.3), of 25.
