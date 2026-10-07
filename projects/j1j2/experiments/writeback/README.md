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
