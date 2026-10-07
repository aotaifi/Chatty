# Write-back test (exact 6x6): residual factor + energy metric vs infidelity

Design and verdict: `results/writeback/DESIGN.md`. Code: `wb_run.py` (+ `../stall_6x6/st6_sector.py`), `wb.sbatch`, `specs/`.
ws1: code `~/chatty_writeback/code`, runs `/project/theorie/a/A.Otaifi/chatty_writeback/runs/`.

## Pre-registration (written 2026-10-07 before any training result)
Start: guide (|psi_P|, s_P) = symmetrised ViT; exact phi_FN of H_FN[psi_P, s_P]. Frozen-FN gain of the ideal
iteration G0 = E_f[psi_P] - E_FN (3.0e-5/site expected).
Model: b = |psi_P| exp(f), f = D4 x flip averaged residual CNN (translation invariant, zero-initialised head).
Arms (same net, same samples x ~ phi^2, Adam, same steps):
- **E-M** (candidate): loss = frozen-FN energy (exact edge-ratio identity), CNN 32 ch x 4 layers (~28k params).
- **I-M** (control): loss = infidelity to phi_FN (pointwise phi^2-weighted L2 to second order), same net.
- **E-L** (capacity): loss = frozen-FN energy, CNN 64 ch x 4 layers (~111k params).
Primary metric (exact): fraction of the frozen-FN gain captured, frac = (E_f[psi_P] - E_f[b]) / G0.
Secondary (exact): <H>(b, s_P) vs <H>(psi_P, s_P) = 1.319e-4 and <H>(phi_FN, s_P) = 8.41e-5; rms(log b - log phi);
after a Krylov step: <H> and E_FN of b vs the exact-loop guide (7.84e-5 / 6.5e-5).

Stop rule (decided in advance):
- **PASS** if the best E arm reaches frac >= 0.5 and <H>(b, s_P) <= 1.319e-4 - 2.4e-5 (half the exact step's gain):
  write-back feasible; next = cumulative 3-iteration loop with one warm-started f, then the sampled version.
- **PARTIAL** if 0.1 <= frac < 0.5: representation-limited; at most one further capacity step, no loop.
- **FAIL** if frac < 0.1 (no better than the 7 one-hop features, 7.7%): the residual-CNN route is closed at 6x6;
  move to non-local factors (pair-product) or hop-based representations.
- **Metric check:** if I-M reaches a lower frozen-FN energy than E-M, the energy-metric diagnosis is wrong.
- Budget: <= 6 GPU-h in total (smoke + arms); an arm stops at its step budget or max_sec, whichever first.

## Amendment (2026-10-07 ~18:30, before any training result; review `results/review/REVIEW_2026-10-07.md` + PI)
- **This is a representability test**, not a loop test: the E and I losses use the exact phi_FN on x and on all
  sampled neighbours and exact sampling from the sector table. The sampled (8x8) version would need instead:
  samples from b^2 (VMC) or FN walkers, a_k and b on the one-hop neighbours, no phi; see DESIGN.md section 2.
- **Same-capacity control arm V-M added:** the same residual net trained on the fixed-sign <H>(b, s_P) itself
  (fixed-sign VMC estimator: local energies from K = 8 sampled bonds, reference samples x ~ phi^(2 beta) with
  self-normalised importance weights), same data budget. Without it, a PASS of E-M would only say "representable".
- **Criterion restated in the energy metric:** the exact identity gives Q_FN(e) := E_f[b] - E_FN =
  1/2 sum_allowed |H_xy| phi_x phi_y (g_x - g_y)^2 / sum phi^2 g^2 (g = b/phi). PASS requires
  Q_FN(e) <= 0.5 G0 = 1.5e-5/site (= frac >= 0.5) and <H>(b, s_P) <= 1.079e-4; FAIL is Q_FN(e) > 0.9 G0.
  The 0.004-rms figure is only the white-noise translation of this (kappa_FN = 0.42).
- All arms sample x ~ phi^(2 beta) with beta = 0.5 (smoke check: 9x smaller standard error than beta = 1 for the
  same batch; 81% of the gain sits in configurations holding 7.6% of the phi^2 weight).
- Training uses one random D4 x flip image per x (shared with its neighbours); validation, model selection and all
  exact evaluations use the exactly symmetrised f. Selection: best validation value of the arm's own objective.
- Reading: E passes and V fails -> the FN target carries information the residual VMC does not reach; E and V both
  pass -> representable, and plain fixed-sign VMC with a residual factor suffices; E fails -> representation-limited.

## Amendment 2 (2026-10-07 19:00, after smoke runs, before main results)
- Smoke (A40, `wbsmoke2A`): E arm at lr 3e-3 captured 2.8% of the gain in 2000 steps (exact), still falling ~0.7%/500
  steps; lr 1e-3 was ~3x slower. Estimator checks: the realistic VMC estimators (frozen H_FN and fixed-sign H, full
  local energies, samples from psi_P^(2 beta), no phi_FN) agree with the exact values within 0.6 SE; their SE is
  2.3e-5/site per 4096 samples (oracle edge identity: 4.8e-7 per 1024).
- Main arms (all C32x4 unless stated, 20k Adam steps, beta = 0.5, equal network evaluations per step):
  E-M (oracle frozen-FN identity, lr 3e-3), I-M (oracle infidelity, lr 3e-3),
  Vfn-M (realistic VMC on frozen H_FN, B = 64 x all 144 bonds, lr 1e-3), Vh-M (realistic fixed-sign <H> VMC, same),
  E-L (C64x4, lr 3e-3, capped at 1 h), E-M-lr1e-2 (optimisation-speed probe).
  The VMC arms use lr 1e-3 because at 3e-3 their noisy gradients raised the exact frozen-FN energy in the smoke run.
- The stop rule is unchanged and applies to the best E arm.

## Result (2026-10-07 22:20)
PARTIAL by the stop rule: E-M 17% of the frozen-FN gain (I-M 8.4%, E-L 12.7%, lr 1e-2 3.9%); realistic VMC arms
Vfn-M 0.8%, Vh-M 0.9% (tie -> FN amplitude target dropped per the reviewer's rule). Details, table and figure:
`results/writeback/DESIGN.md`, `results/writeback/writeback_6x6.png`, run JSONs in `results/writeback/runs/`.
