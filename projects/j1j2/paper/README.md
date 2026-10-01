# J1-J2 sign/FN paper

PRB-style write-up of the projected-Krylov sign reconstruction and
fixed-node amplitude-feedback program.

## Rebuild

```bash
cd /Users/aliotaifi/Chatty/projects/j1j2/paper
make figures
make pdf
```

The plotting script reads authoritative numerical outputs directly from
`../results/` and `../krylov_sign_structure/results/`.
Copied remote SR diagnostics live under `data/` with their Slurm job IDs
in the filenames.

## Result policy

Read `RESULT_REGISTRY.md` before promoting a new claim into the paper.

- **STABLE**: may appear unqualified in main text.
- **PROVISIONAL / LIVE**: may appear only with explicit caveat.
- **SUPERSEDED**: must not be used in manuscript figures or claims.

The current 8x8 SR energy-lowering claim is intentionally provisional because
an independent near-identical SR checkpoint produced substantially different
FN energies. The SR cross-replica likelihood result itself is stable.

## Figure provenance

- Fig. 1: exact 4x4 energy-optimized K1 sweep.
- Fig. 2a: 6x6 same-amplitude Marshall/K1/ViT comparison.
- Fig. 2b: 8x8 PBC FN benchmark plus Qian-Qin literature Table I.
- Fig. 3: exact current-sign 4x4 FN/K1 loop.
- Fig. 4: matrix-free SR cross-replica line searches.
- Fig. 5: 8x8 physical-sample local-field separation.

Literature benchmark values are stored in
`data/literature_8x8_pbc_J2p5.json` with DOI and table provenance.

## Live-result integration

New production results should first be added to `RESULT_REGISTRY.md`.
If they replace a provisional value, update the source artifact and plotting
script rather than hard-coding a new number into the figure.

The manuscript is deliberately written so that the live 8x8 convergence
section can change without altering the established sign-reconstruction and
amplitude-handoff story.
