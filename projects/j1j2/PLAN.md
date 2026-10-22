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
- Open: does a network carry each round without losing the gain (O2b)? And sign labels without the oracle (O3)?

## Step back 2026-10-22 (trigger: T1s then T1s-2 both missed 1e-5 at zero hop)
1. Verdict: switch from training harder to one hop. Done (G1 PASS).
2. Thought: a Pfaffian makes neighbour ratios cheap, so one hop is constant cost.
3. Next: remove the oracles one at a time (README "De-oracling the G1 guide"): O1 done, O2a done, O2b running, then
   O3 (sign), then O4 (walkers).

## Running
| job | run dir | next step per verdict |
|---|---|---|
| O2b, fitted loop, job 17431886 (one job, wall 19:15, setsid watcher pid 145313 -> VERDICT line in delta_today.md) | chatty_compress6/runs/o2b_17431886 (round_k checkpoints, resumable with resume_from; ANALYSIS_BRIEF.md) | Pre-registered: README O2 + Amendments O2-1, O2b-1. PASS (<= 1e-5 within 4 rounds) and viability (fit tax < 0.5 x the O2a gain in rounds 1-3) -> O3. Viability FAIL with a floor near 7e-6 (the lead's prediction, written before the run) -> the lever is the fit floor: a 5x5 backflow or hidden-fermion rows, then retry. STALL or RISE -> diagnose with O2-1 item 6. |

## Angles
- conceptual (angle lead session, started 2026-10-22 at the PI's request): physics of the loop's blocks with the PI; plan file PLAN_conceptual.md, deliverable conceptual/MEMO.md (ranked hypotheses with kill tests). Acted on when closed or when it changes this plan (O3 design especially).

## Open questions
| question | budget | scope line |
|---|---|---|
| B0: same network (PP+3x3 bf) trained by plain VMC + one hop + FN; decides whether the loop has an accuracy edge or only a cost edge (README Test B0) | <= 8 A40-h; worker building code 2026-10-22, launcher to be read before sbatch | MC SR, rbmpp_train recipe; exact-gradient VMC on the 15.8M table rejected (41 min per forward pass) |
| O2b: the fitted loop | ~14 A40-h | PP + 3x3 backflow, cx_t1s fits; ViT dropped (tail-trained ViT + hop only 1.52e-5) |
| O3: KKT flip labels from the network's own FN solution (sign s_1 -> s_2' without the oracle) | pre-register after O2b | previous fitted-ViT labels recalled 5e-7; retry with PP+backflow |
| O4 / T2: walkers instead of exact solves; cost at 8x8 with Woodbury updates; edge over plain SR VMC of PP+backflow at equal GPU-h | design after O3 | the PI's T2 spec; the tail-steered sampling idea is in RESEARCH_MAP section 5 |

## Dead ends
- (not dead, settled) S1 s1_17446338: oracle-free PP sign ceiling <= 7.0e-6 (3-6e-6 extrapolated); oracle-free s_1 ceiling 2.51e-6; oracle s_2 9.15e-7. Sign is not the binding constraint at 6x6; the refit tax (~6e-6) is. (number, run)
- gamma-FN to free the energy from the tail: 26% of the gap removed, bar 50% (g0_17411369).
- ViT refit to phi*(s_2): 9.2e-5 (refresh_17409419, refresh2_17409866; the fit had stalled).
- Tail-trained ViT: decade -11 error 0.113, E_FN 6.89e-5; + hop 1.52e-5 (t1s2_vit_17418858, g1).
- More samples on the decade-stratified fit: 4x samples moved r(-11) by 5-12% (t1s2_*).
- Fixed VMC-trained pairing as a tail basis: no advantage at equal L2 (t0, t0b). This is an objective artefact per
  the Fable audit, not a representability limit.
- Constant-cost storage of loop steps: 7 designs (RESEARCH_MAP 2026-10-13). Learned one-hop bond field: 1.92e-5
  (RESEARCH_MAP 2026-10-20).
