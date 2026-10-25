# PLAN (J1-J2 FN/Krylov loop) - current state only; history in RESEARCH_MAP.md, delta_today.md, experiments/compress_6x6/README.md
Goal: a loop (FN amplitude <-> sign flip, stored in networks) that improves a trained network's ground state at
constant cost per round and beats brute-force NQS in GPU-h at 8x8-14x14.

## Where we stand (2026-10-22, exact 6x6; all numbers E_FN per site above E0, sign s_2 unless noted)
- G1 (g1_17426232): PP + 3x3 backflow trained on the tail (decade-stratified fit to the oracle phi*(s_2)) + one hop
  |a||1+alpha(E_L-E)| gives 7.06e-6, against a bar of 1e-5. The same-sign, same-hop VMC RBM+PP gives 1.29e-5. This
  is representability only: oracle sign s_2, oracle target, one seed.
- O1 (o1_17430543): alpha chosen by the VMC energy of the one-hop state gives 7.11e-6. The alpha oracle is removed.
- O2a (o2a_17430739): the fit-free hop+FN chain from the VMC PP goes 2.37e-5 -> 8.1e-6 -> 4.5e-6 -> 2.9e-6 -> 2.1e-6
  -> 1.7e-6, c = 0.32-0.64. The loop's map is fast and there is no wrong-node signature.
- O2b (o2b_17431886): fitted loop 2.37e-5 -> 9.26e-6 at round 3 -> 8.42e-6 at round 5; PASS, viability FAIL (refit tax ~6e-6 every round).
- S1: oracle-free signs allow 2.5e-6 (s_1) and <= 7e-6 (PP own); the refit, not the sign, binds at 6x6.
- B0 closed (README 'Results: B0 pilots'): no resolvable VMC gain from 3x3 backflow.
- CORRECTION 2026-10-23: the fair competitor is VMC RBM+PP + the same Lanczos steps + FN with its OWN sign: one step 5.57e-6, two steps <H> 2.49e-6, three 1.07e-6 (results/rbmpp_6x6/runs/table_A.json). It beats the fitted loop (8.4e-6). The earlier '1.5x edge' paired RBM+PP with the mismatched s_2 and is withdrawn. The loop's accuracy edge at 6x6 is NOT demonstrated.
- F1 (Paderborn 3811214/6): the refit tax is not under-training (13x steps: -6%) and only partly capacity (5x5: -18%); the log-amplitude objective misses what FN uses.
- F2: fitting the violating-bond ratios does not generalise (top 0.1% of bonds carry 55% of the error; held-out L_v flat). Step back: compute the tail at use time.
- H1 (Paderborn 3818121): a second hop cuts the refit tax to 0.32x (1.90e-6 on O2b's table); derived two-hop loop floor ~2.5e-6.
- N0 (Paderborn 3818482): yes, the tail is the near-node cancellation region (median c 0.056 vs 0.583).
- NEXT MAJOR CALCULATION approved 2026-10-23 17:40 (PI): M1 + gate, proposals/2026-10-23_next_major_calc_v2_final.md. Takeover by the lead from 17:45.
- Gate (2026-10-23): hop is tail-specific (G-A2), N0 not by construction, fit-free signed chain 1.95e-6 -> 3.0e-9 in 6 rounds, but RBM+PP two-hop refit tax 1.66e-6 = 1.02 x the round gain: GATE SHUT, loop C not run. Storage of the node region is the blocker (ansatz angle C2-L).
- M1.0: no Lanczos+FN, no FN/GFMC and no NQS+Lanczos at 8x8 published; M1.3 is new.
- M1.3 (8x8, translation RBM+PP guide): FN p=0 -0.4987565(28) quotable; FN p=1 -0.498946(5) QUOTABLE (2026-10-25, 2 chains x 1080 blocks); VMC p=1 -0.498923(17). p=1 FN would beat published full-projection RBM+PP by 6e-5, 2.5e-4 above DMRG. And sign labels without the oracle (O3)?

## Step back 2026-10-23 afternoon (trigger: the PI; the tail finding had not moved in weeks)
1. Verdict: reframe. The tail is the neighbourhood of the nodal surface; its amplitude and the sign are one object (the node). We store them as two (log|a| network + discrete sign), and log|psi| is singular at a node, so the tail cannot be learned that way.
2. Thought: S1 (own PP sign beats oracle s_2 on |PP|: 1.87e-5 vs 2.37e-5 with 19x more wrong weight) says sign-amplitude consistency matters more than sign correctness; F2 (tail ratios do not generalise in log space) and B0 (VMC damages the tail freely) fit the same picture. Our edge: FN computes the near-node amplitude given a sign, and the hop computes the near-node cancellation sums; neither has to be learned.
3. Next: N0 diagnostic (are the top-0.1% violating bonds near-node, i.e. strong cancellation c(x) = |sum_y H_xy phi(y)| / sum_y |H_xy phi(y)| small?), minutes on existing tables; then a signed guide (network's own sign x sign of the hop factor; the conceptual angle's hop-sign check) instead of more refit work. Hypothesis, ~50%.

## Step back 2026-10-23 (trigger: second failed refit fix)
1. Verdict: stop improving the stored network on the tail; compute it at use time (more hops).
2. Thought: the FN-relevant error sits on ~0.1% of violating bonds whose ratios are cancellations; networks do not generalise there, a hop evaluates them exactly.
3. Next: H1 (tax with two hops on existing tables), minutes, no training.

## Step back 2026-10-22 evening (trigger: second failed version of B0 training)
1. Verdict: stop sharpening B0; take the middle branch (modest edge, ~1.5x, from RBM+PP one hop 1.29e-5 vs loop 8.4e-6).
2. Thought: the decision does not need the PP+bf VMC number; 400 SR steps show no resolvable gain over the PP (+2e-6 +- 1.1e-5).
3. Next: the fit-floor test F1 (bigger backflow fitted to the saved O2b round targets, tax vs 5.9e-6), ~1-2 H100-h.

## Step back 2026-10-22 (trigger: T1s then T1s-2 both missed 1e-5 at zero hop)
1. Verdict: switch from training harder to one hop. Done (G1 PASS).
2. Thought: a Pfaffian makes neighbour ratios cheap, so one hop is constant cost.
3. Next: remove the oracles one at a time (README "De-oracling the G1 guide"): O1 done, O2a done, O2b running, then
   O3 (sign), then O4 (walkers).

## Running
| job | run dir | next step per verdict |
|---|---|---|
| 10x10 / 12x12 RBM+PP training, Paderborn 3921516 / 3921519 (cont, 4096 chains; watchers -> VERDICT in delta_today.md) | Paderborn /scratch/hpc-prf-ppdmrg/hpcalot/rbmppL/runs/rbL{10,12}_prod; local /scratch/a/A.Otaifi/rbmppL/prod | converged -> VMC p=0/1 + FN p=0 (+ FN p=1 at 10x10) via gfmc_fn_lattice/m13b_LxL; drifting -> resume |
| Test L8 (one loop round at 8x8, walkers): worker building | tbd | README 'Test L8' branches |
| M1.3b 8x8: V 3832999 (VMC p=0/1), A 3833000/01 (GFMC p=0), B 3833002/03 (GFMC p=1), X 3833004-08 (bias scans); Paderborn express, afterok:V; setsid watchers | Paderborn /scratch/hpc-prf-ppdmrg/hpcalot/gfmc8/runs/rb8; local /scratch/a/A.Otaifi/gfmc_m13b/prod/rb8 | gfmc8_analyze.py: E_FN p=0, p=1 at 8x8 vs VMC p=0/1 and the published bars, GPU-h per number; seeds disagree -> continue chains before any fit |

## Angles
- ansatz-pioneer (angle lead session, started 2026-10-23 at the PI's request): find a genuinely new ansatz for frustrated magnets (landscape map, >= 3 candidates with novelty checks, cheap 4x4/6x6 tests); plan file PLAN_ansatz.md, deliverable ansatz/ANSATZ_MEMO.md; normal QoS on Paderborn.
- conceptual (angle lead session, started 2026-10-22 at the PI's request): physics of the loop's blocks with the PI; plan file PLAN_conceptual.md, deliverable conceptual/MEMO.md (ranked hypotheses with kill tests). Acted on when closed or when it changes this plan (O3 design especially).

## Open questions
| question | budget | scope line |
|---|---|---|
| O3: KKT flip labels from the network's own FN solution (sign s_1 -> s_2' without the oracle) | pre-register after O2b | previous fitted-ViT labels recalled 5e-7; retry with PP+backflow |
| O4 / T2: walkers instead of exact solves; cost at 8x8 with Woodbury updates; edge over plain SR VMC of PP+backflow at equal GPU-h | design after O3 | the PI's T2 spec; the tail-steered sampling idea is in RESEARCH_MAP section 5 |

## Dead ends
- Fitting the tail better, any objective (closed 2026-10-23): log-amplitude L2 (refresh, 9.2e-5), decade-stratified (T1s/T1s-2, r(-11) plateau 0.07; 4x samples +5-12%), more steps/capacity (F1: -6% / -18% tax), the exact FN-relevant bond-ratio objective (F2-1: fits shown bonds, held-out flat). The tail is not learnable by these networks; route around it (hops), do not refit harder.
- B0 PP+3x3bf VMC: SR from the converged PP gives no gain in 400 steps (lr 0.01: +2e-6 +- 1.1e-5; Paderborn 3806793); gradient not unit-tested.
- (not dead, settled) S1 s1_17446338: oracle-free PP sign ceiling <= 7.0e-6 (3-6e-6 extrapolated); oracle-free s_1 ceiling 2.51e-6; oracle s_2 9.15e-7. Sign is not the binding constraint at 6x6; the refit tax (~6e-6) is. (number, run)
- gamma-FN to free the energy from the tail: 26% of the gap removed, bar 50% (g0_17411369).
- ViT refit to phi*(s_2): 9.2e-5 (refresh_17409419, refresh2_17409866; the fit had stalled).
- Tail-trained ViT: decade -11 error 0.113, E_FN 6.89e-5; + hop 1.52e-5 (t1s2_vit_17418858, g1).
- More samples on the decade-stratified fit: 4x samples moved r(-11) by 5-12% (t1s2_*).
- Fixed VMC-trained pairing as a tail basis: no advantage at equal L2 (t0, t0b). This is an objective artefact per
  the Fable audit, not a representability limit.
- Constant-cost storage of loop steps: 7 designs (RESEARCH_MAP 2026-10-13). Learned one-hop bond field: 1.92e-5
  (RESEARCH_MAP 2026-10-20).
