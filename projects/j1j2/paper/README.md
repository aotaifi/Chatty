# J1-J2 sign/FN paper

PRB-style manuscript on reconstructing frustrated signs from a positive
amplitude and closing the amplitude/sign feedback with lattice fixed node.

## Rebuild

```bash
cd /Users/aliotaifi/Chatty/projects/j1j2/paper
make figures
make pdf
```

The build uses `tectonic`. Figure generation reads authoritative numerical
outputs directly from `../results/` and
`../krylov_sign_structure/results/`. Remote SR diagnostics copied for the
paper retain their Slurm job IDs in `data/`.

## Structure

1. Introduction
2. Methods: model and error measures, one-step Krylov sign update, fixed-node
   amplitude update, learning the amplitude from walkers (MLE + SR), setup.
3. Results, in order of increasing difficulty:
   A. Krylov sign update — exact 4x4 ED amplitude with Marshall signs, one and
      then repeated Krylov sign steps, compared with ED (plus N=20);
   B. exact FN/Krylov feedback loop (4x4, N=20);
   C. 6x6 with fixed ViT amplitude, and 6x6 FN;
   D. 8x8 FN vs literature;
   E. open problem: learning the FN amplitude from walkers.
4. Discussion; appendices on the second Krylov step and statistics.

## Result policy

Read `RESULT_REGISTRY.md` before promoting a new claim.

- **STABLE**: may appear as a result in the main text.
- **PROVISIONAL / LIVE**: may appear only with an explicit qualification.
- **SUPERSEDED**: must not be used in manuscript figures or claims.

In particular, the 8x8 SR cross-population likelihood result is stable, but a
repeated learned 8x8 FN/Krylov loop is not claimed to be converged.

## Main figure provenance

All figure inputs live in `data/` or in the tracked
`../krylov_sign_structure/results/`, so `make pdf` works from a fresh clone.

- **Fig. 1 — Krylov sign update:** (a) `data/k1_threshold_4x4_J2p5.npz`
  from `scripts/k1_exact_4x4.py` (recomputes the 4x4 ED and the threshold,
  about 10 s); (b,c) `data/k1_iterated_4x4.json` from the same script (repeated steps, J2/J1 = 0.4-1.0); (d) adds
  `groundstate_k1_20site_exact_J2p5.json`.
- **Fig. 2 — exact FN/Krylov feedback:** `closed_fn_krylov_4x4_J2p5_J2zero_init_100.json`
  and `closed_fn_krylov_20site_J2p5_J2zero_init.json`.
- **Fig. 3 — 6x6 fixed ViT amplitude:** `data/mechanism_6x6.json` and
  `data/energy_krylov_vs_vit_6x6_3479622.npz` (copy of
  `../results/a1_node_audit_3479622/energy_krylov_vs_vit_6x6_indep.npz`).
- **Fig. 4 — 8x8 benchmark:** `data/fn8_krylov_M128_3471544.npz`,
  `data/fn8_marshall_M128_3471545.npz` (copies from `../results/8x8_*`) and
  `data/literature_8x8_pbc_J2p5.json`.
- **Fig. 5 — amplitude learning:** `data/fn_mle_halfstep_exact4x4.out` and
  `data/fnmle8_sr_lambda1_linesearch_3506041.json`.

Numbers in the text: 6x6 paired differences are recomputed from the Fig. 3 npz
with 64-chain standard errors; 6x6 FN energies come from
`../results/krylov_fn_6x6_matched_verdict_2026-09-28.md` and
`../results/fn_krylov_closedloop_6x6_verdict_2026-09-29.md`.
