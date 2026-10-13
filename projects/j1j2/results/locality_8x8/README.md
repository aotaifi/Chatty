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

# Locality vs J2 (4x6 exact ground states, 2026-10-13)
Pre-registration: `experiments/locality_8x8/README.md` ("Locality vs J2"). Code `ed_pairint.py`. Raw file:
`ed_pairint_4x6.json`. CPU 2 min.

## What was run
- Exact ground state of the 4x6 torus (Sz = 0, 2.7e6 states) at J2 = 0, 0.3, 0.4, 0.5, 0.6.
- For each: 2000 samples of |psi0|^2, all valid site-disjoint bond pairs, rms I and sign non-additivity by bond
  distance.

## Result
**One line: the double-swap interaction is strongly J2-driven. Locality (pre-registered bar: rms I <= 0.10 and sign
non-additivity <= 0.3% at distance >= 2.8) holds at J2 = 0 only. At J2 = 0.3 it is close (0.21 at distance 2.83,
0.07-0.10 beyond), and it fails at J2 = 0.4-0.6, worst at 0.5.**

| J2 | E0/N | d = 1 | 1.41 | 2 | 2.24 | 2.83 | 3 | 3.16 | sign non-additivity at d >= 2.8 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | -0.6897 | 0.61 | 0.17 | 0.13 | 0.11 | **0.089** | **0.072** | **0.041** | 0 (Marshall) |
| 0.3 | -0.5741 | 0.96 | 0.43 | 0.26 | 0.22 | 0.21 | 0.095 | 0.069 | 0.02% |
| 0.4 | -0.5435 | 1.79 | 1.13 | 0.65 | 0.70 | 0.73 | 0.22 | 0.21 | 0.3-4.9% |
| 0.5 | -0.5225 | 2.19 | 1.64 | 0.79 | 0.94 | 1.15 | 0.32 | 0.29 | 0.7-8.8% |
| 0.6 | -0.5221 | 1.54 | 0.90 | 0.58 | 0.47 | 0.54 | 0.061 | 0.077 | 0.01-1.6% |
| 6x6 exact, J2 = 0.5 (reference) | | 1.69 | 1.29 | 1.02 | 0.94 | 0.80 | 0.61 | 0.42 | 2.6-6.3% |

Readings, per the pre-registration:
- **"Locality returns at J2 <= 0":** J2 = 0 passes the bar. J2 = 0.3 misses at distance 2.83 (0.21).
- **"Locality is J2-dependent":** at distance >= 2, J2 = 0 is 6-13x below J2 = 0.5.
- The interaction peaks in the frustrated window 0.4-0.5 and falls again at 0.6.
- Caveats:
  - 4x6 is small and anisotropic; a distance of 3 is antipodal along the 6-direction.
  - This compares J2 at equal distance; it is not an asymptotic decay.
- **Implication.**
  - A light-cone fold-back is plausible only deep in the Neel phase (J2 <~ 0.3). There the sign problem is weak
    (J2 = 0 is Marshall-exact, sign-problem-free and solvable by QMC).
  - It is not plausible in the frustrated regime J2 ~ 0.4-0.6, where the method would be needed.
  - The non-locality is a physical property of the frustrated ground state, not a network artefact.

# Locality of the residual relative to a pair-product base (exact 6x6, 2026-10-13)
Pre-registration: `experiments/locality_8x8/README.md` ("Locality of the RESIDUAL"). Code
`compress_6x6/cx_pairint.py` (residual guides). Raw file: `cx_pairint_residuals_6x6.json`. GPU 1 min.

## What was run
- The same 1024 samples of |psi0|^2 for every case.
- Pair interaction I and sign non-additivity by bond distance for:
  - log|psi0|;
  - the residual log|psi0| - log|base| with relative sign s0 s_base, for three bases: PP alone, converged RBM+PP
    (A), and the symmetrised ViT.
- All bases are the fully projected exact tables.

## Result
**One line: NOT LOCAL. The bases remove most of the short-range non-additivity (rms I at distance 1: 1.69 -> 0.47-0.56),
but the residual keeps a distance-independent part: rms I about 0.5, sign non-additivity 1-3% from distance 1 out to
3.6. The pre-registered bar (rms <= 0.10, sign <= 0.3% at distance >= 2.8) fails for every base.**

| bond distance | 1 | 1.41 | 2 | 2.24 | 2.83 | 3 | 3.16 | 3.61 |
|---|---|---|---|---|---|---|---|---|
| log|psi0|: rms I / sign % | 1.69 / 19 | 1.29 / 13 | 1.02 / 9.8 | 0.94 / 8.5 | 0.80 / 6.3 | 0.61 / 2.6 | 0.42 / 1.6 | 0.72 / 5.4 |
| residual to PP | 0.54 / 3.0 | 0.58 / 3.4 | 0.51 / 2.6 | 0.56 / 3.2 | **0.55 / 3.1** | **0.29 / 0.9** | **0.30 / 1.0** | **0.49 / 2.6** |
| residual to RBM+PP (A) | 0.47 / 2.4 | 0.52 / 2.8 | 0.45 / 2.2 | 0.52 / 2.7 | **0.51 / 2.7** | **0.26 / 0.7** | **0.28 / 0.8** | **0.46 / 2.1** |
| residual to ViT | 0.56 / 3.4 | 0.63 / 4.5 | 0.55 / 3.3 | 0.62 / 4.4 | **0.63 / 4.5** | **0.31 / 1.0** | **0.34 / 1.1** | **0.59 / 3.7** |

- Mean |I| of the residuals is 0.05-0.25, so the rms is driven by a heavy tail.
- The residual's non-additivity does not decay with distance. A large I occurs whenever one of x^b, x^{b'}, x^{bb'}
  lands where the base is wrong (near its nodes, or where its sign differs from psi0), wherever b and b' are.

**Design consequence (as asked).**
- If the residual had been local, the architecture would have been: a PP base (exact, cheap multi-swap ratios via
  fast Pfaffian/determinant updates, no network hops) plus a local correction network, folded back through a
  light cone of fixed radius.
- With this result that architecture does not work as it stands. The correction relative to the best available
  bases (PP 6.3e-5, RBM+PP 4.7e-5, ViT 1.3e-4) is as non-local as psi0 itself at distance >= 2.8.
- Only the short-range part is removed (3x at distance 1).
- What would be needed: a base whose errors are themselves local. These residual errors are concentrated near
  the base's nodes, i.e. the same rare-configuration / sign problem as before.
