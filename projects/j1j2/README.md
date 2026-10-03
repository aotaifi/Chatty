# J1-J2 sign/amplitude loop — project index

Square-lattice spin-1/2 J1-J2 model at J2/J1 = 0.5. The wave function is split as
psi(x) = s(x) a(x). Signs are updated by one shifted Krylov step,
s'(x) = s(x) sgn[T - r(x)] with r = (H psi)/psi; amplitudes are updated by lattice
fixed node (FN). The loop is closed exactly on 4x4 / N=20 and tested on 6x6 and 8x8.
The open problem is learning the FN amplitude from walkers at 8x8.

Start here:

| File | What it is |
|---|---|
| `paper/` | Manuscript, figure script and frozen figure inputs. See `paper/README.md`. |
| `paper/RESULT_REGISTRY.md` | Which results are stable, provisional or superseded. |
| `STATUS.md` | Chronological research log (append-only, newest entries mostly at the bottom). |
| `FAILED_ROUTES.md` | Routes that failed and why. Read before reviving an idea. |
| `OPEN_ANGLES.md` | Ideas not yet pursued. |
| `AMPLITUDE_HANDOFF_API.md` | Interface for a callable amplitude (partly superseded: the sampled threshold is now current-sign, energy-optimal). |

Files are not moved between folders: many scripts hardcode absolute
`/Users/...` or `/pc2/...` paths and running jobs write into `results/`.

## Folders

| Folder | Contents |
|---|---|
| `krylov_sign_structure/` | Exact sign-structure subproject. `experiments/` holds the 4x4 / N=20 exact benchmarks (`square_exact_energyopt.py` = paper Fig. 1, `groundstate_k1_20site_exact.py`, `closed_fn_krylov_*.py` = Fig. 2), triangular/symmetry checks, the finite-tau CTQMC side angle (`finite_tau_*`, parked) and second-step tests (`k2_*`, parked). `results/` holds their outputs. |
| `experiments/` | Larger-lattice code: 6x6/8x8 GFMC (`gfmc_*`), 6x6 fixed-amplitude energies (`energy_krylov_vs_vit_6x6*`), amplitude learning (`fn_mle_*`, `fnmle8_sr_*`, `frozen_hfn_*`, `fixedsign_energy_*`, `mixed_efn_force_*`). |
| `cluster/paderborn/` | Slurm scripts for the Paderborn GPUs; a40/h100 pairs differ only in resources. |
| `results/` | Run outputs and dated verdict notes (`*_YYYY-MM-DD.md`). Run folders carry the Slurm job ID (e.g. `8x8_krylov_3471544/`). |

## Research threads and where they live

| Thread | Status | Main files |
|---|---|---|
| Exact one-step Krylov sign benchmark (4x4 sweep, N=20) | Stable, paper Fig. 1 | `krylov_sign_structure/experiments/square_exact_energyopt.py`, `groundstate_k1_20site_exact.py`; `paper/scripts/k1_exact_4x4.py` |
| Exact FN/Krylov feedback loop | Stable, paper Fig. 2 | `krylov_sign_structure/experiments/closed_fn_krylov_*.py` |
| 6x6 fixed-amplitude Marshall vs Krylov vs ViT | Stable, paper Fig. 3 | `experiments/energy_krylov_vs_vit_6x6_indep.py`, `results/a1_node_audit_3479622/` |
| 6x6 Krylov-FN vs Marshall-FN, 6x6 fixed point | Stable, paper Sec. III C | `experiments/gfmc_6x6_*`, `results/krylov_fn_6x6_matched_verdict_2026-09-28.md`, `results/fn_krylov_closedloop_6x6_verdict_2026-09-29.md` |
| 8x8 FN with Krylov vs Marshall signs | Stable, paper Fig. 4 | `results/8x8_krylov_3471544/`, `results/8x8_marshall_3471545/` |
| Exact 4x4 MLE half-step | Stable, paper Fig. 5(a) | `experiments/fn_mle_halfstep_exact4x4*.py` |
| 8x8 MLE + SR amplitude learning | Live / open; iteration 1 not reproduced, iteration 2 worse | `experiments/fnmle8_sr_*`, `results/fnmle_sr_closedloop_8x8_iter1_2026-10-01.md` |
| Frozen-H_FN factorized SR | Live (current route) | `experiments/frozen_hfn_*`, `results/frozen_hfn_factorized_sr_exact4x4_2026-10-03.md` |
| Second Krylov step (K2) | Failed on 6x6, paper App. A | `experiments/k2_amplitude_distortion_*`, `results/true_k2_6x6_3471990/` |
| Global sign models, local cavities/boundaries, CNN classifier, warm-start ViT gates, Adam MLE | Failed (see FAILED_ROUTES.md) | `experiments/global_*`, `bench_*`, `learned_fn8_*`, `warmstart_vit_*`, `train_fn_mle_vit8.py` |
| Finite-tau CTQMC | Parked | `krylov_sign_structure/experiments/finite_tau_*`, `results/finite_tau_*` |

## Known housekeeping issues (left in place on purpose)

- `results/paderborn_*` folders are empty duplicates of the run folders without the prefix.
- `results/a1-node-audit-3479622.err/.out` belong to `results/a1_node_audit_3479622/`.
- `krylov_sign_structure/experiments/closed_fn_krylov_exact20.py` and `closed_fn_krylov_exact20site.py` both write `results/closed_fn_krylov_20site_J2p5_J2zero_init.json`.
- Several scripts load code from `~/j1j2_vit_bench` (outside the repo); ViT checkpoints are symlinked from there and not committed.
- Files above 5 MB are kept local only (see `.gitignore`).
