# Pair interaction (locality) of psi_P at 8x8, 2026-10-13
Pre-registration and amendment L-1: `experiments/locality_8x8/README.md`. Code: `s8_pairint.py`, `s8_elcone.py`.
Raw files: `s8_pairint_RB1.json`, `s8_elcone_RB1.json`.
- Data: 3072 samples of |psi_P|^2 (signs_8x8 referee run s8RB1_17160055). The projected 8x8 ViT was already stored
  on all configurations within two exchange moves, so no new ViT evaluation was needed.
- Compute: CPU only, about 1 min.

## What was run
1. For every sample and every pair of valid, site-disjoint bonds (b, b'), I computed the departure from additivity:
   - log-amplitude: I = la(x^{bb'}) + la(x) - la(x^b) - la(x^{b'});
   - sign: whether s(x^{bb'}) s(x) s(x^b) s(x^{b'}) = -1.
   About 30M pairs, binned by bond distance.
2. Amendment L-1: the resulting error of psi_P's local energy at each neighbour x^b, when psi_P at x^{bb'} is exact
   within radius rho of b and additive beyond. This is the light-cone construction.

## Result
**One line: NOT LOCAL (pre-registered). At bond distance >= 3 the rms interaction is 0.25-0.72 (bar 0.10) and the
sign non-additivity 0.6-4.8% (bar 0.3%). There is no faster decay than at 6x6: at equal distance the 8x8 values
are within ~0.2 of the 6x6 ones, and they level off at about 0.5 out to the largest distance.**

| bond distance | 1 | 1.41 | 2 | 2.24 | 2.83 | 3 | 3.16 | 3.61 | 4 | 4.12 | 4.24 | 4.47 | 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rms I, 8x8 psi_P | 1.54 | 1.12 | 0.83 | 0.81 | 0.71 | 0.72 | 0.64 | 0.59 | 0.37 | 0.28 | 0.55 | 0.25 | 0.57 |
| rms I, 6x6 psi_P | 1.68 | 1.28 | 1.02 | 0.95 | 0.82 | 0.61 | 0.43 | 0.75 | | | | | |
| rms I, 6x6 exact psi0 | 1.69 | 1.29 | 1.02 | 0.94 | 0.80 | 0.61 | 0.42 | 0.72 | | | | | |
| weighted rms I, 8x8 | 1.35 | 0.68 | 0.52 | 0.38 | 0.28 | 0.40 | 0.27 | 0.21 | 0.23 | 0.12 | 0.18 | 0.11 | 0.16 |
| sign non-additivity, 8x8 | 17% | 12% | 6.9% | 6.9% | 5.6% | 4.8% | 4.3% | 3.8% | 1.3% | 0.8% | 3.4% | 0.6% | 3.9% |
| sign non-additivity, 6x6 psi0 | 19% | 13% | 9.8% | 8.5% | 6.3% | 2.6% | 1.6% | 5.4% | | | | | |

Light-cone error of E_L(x^b) at 8x8 (amendment L-1). For scale: the sample sd of E_L(x) is 0.32, and the step
target needs E_L to about 0.03 (alpha ~ 0.1 maps it to the 0.003 log-amplitude tolerance).

| exact within rho | 0 | 1 | 2 | 3 | 4 | 5 (= the full second hop) |
|---|---|---|---|---|---|---|
| rms error (all neighbours) | 63 | 40 | 33 | 13 | 3.5 | 0 |
| edge-weighted rms error | 1.6 | 1.0 | 0.75 | 0.53 | 0.16 | 0 |

## What this implies for a light-cone fold-back at 8x8-14x14
- **8x8: no saving.**
  - Even with every bond within distance 4 exact (nearly the whole 8x8 lattice), the local energy at the neighbours
    is off by 0.16 (edge-weighted). That is 5x the precision the step needs and half the natural spread of E_L.
  - Only the full second hop (rho = 5) is exact, so the light cone costs the same as the exact two-hop features.
- **10x10-14x14: worse, not better.** At fixed distance the interaction is not smaller at 8x8 than at 6x6, and it
  levels off at about 0.5 out to distance 5.
  - The number of far partners per bond grows with N, so the summed light-cone error grows with system size.
  - A light cone pays off only if the interactions decay beyond distance 5. Nothing in the 6x6 or 8x8 data
    suggests that near J2 = 0.5.
- This is a property of the state, not only of the network: at 6x6 the exact ground state shows the same values as
  psi_P.
- So the constant-hop fold-back via a light cone is not viable at these sizes. The pre-registered fallback stays:
  cap the depth (<= 2 hops + Lanczos polish), and pay one hop per loop iteration where more depth is wanted.
