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

## Scientific organization

The manuscript is written for a reader with no knowledge of the project
history:

1. define the J1-J2 Hamiltonian and the decomposition `psi=s*a`;
2. derive the one-step Krylov sign rule from the local energy;
3. define the lattice fixed-node amplitude problem and the walker mixed
   distribution;
4. benchmark signs with ED and a same-amplitude ViT comparison;
5. close the exact 4x4 FN/Krylov loop;
6. scale the sign guide to 8x8 and compare with same-geometry literature;
7. isolate the remaining amplitude-learning problem and test SR transfer
   across independent FN populations.

Failed implementation branches and unresolved learned-loop energy claims are
not used as the narrative spine.

## Result policy

Read `RESULT_REGISTRY.md` before promoting a new claim.

- **STABLE**: may appear as a result in the main text.
- **PROVISIONAL / LIVE**: may appear only with an explicit qualification.
- **SUPERSEDED**: must not be used in manuscript figures or claims.

In particular, the 8x8 SR cross-population likelihood result is stable, but a
repeated learned 8x8 FN/Krylov loop is not claimed to be converged.

## Main figure provenance

- **Fig. 1 — one-step Krylov benchmark:** exact 4x4 sweep from
  `square_exact_energyopt.csv`, plus the independent exact N=20 skew-torus
  check from `groundstate_k1_20site_exact_J2p5.json`.
- **Fig. 2 — exact fixed-node/Krylov feedback:** N=16 history from
  `closed_fn_krylov_4x4_J2p5_J2zero_init_100.json` and N=20 history from
  `closed_fn_krylov_20site_J2p5_J2zero_init.json`.
- **Fig. 3 — larger-system Krylov validation:** 6x6 same-amplitude
  Marshall/Krylov/ViT energies and independent ViT-weighted sign diagnostics.
- **Fig. 4 — 8x8 benchmark:** Krylov-sign and Marshall-sign fixed-node
  replicas plus the same-geometry literature benchmarks.
- **Fig. 5 — amplitude learning:** exact 4x4 amplitude-update test and the
  8x8 stochastic-reconfiguration cross-sample line search.

The 8x8 fixed-node bars display the two individual populations and use their
half-difference only as a reproducibility scale, not a precision asymptotic
uncertainty.
