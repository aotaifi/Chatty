# Mechanism verdict — 2026-09-28

## Core result
The one-step Krylov sign miracle is **not a generic property of frustration**.
It is a **good-gauge / large-basin phenomenon**: when the baseline sign gauge is aligned with the dominant antiferromagnetic structure, the scalar
[
r_0(x)=rac{(H,a s_0)(x)}{a(x)s_0(x)}
]
orders the important residual sign defects almost one-dimensionally, so a single threshold update can nearly solve them. When no such two-color gauge removes the dominant frustration, the mechanism fails.

For the mechanism sweep we used exact ground-state amplitudes on a periodic 4x4 square lattice in the S^z=0 sector. The threshold t was chosen **without hidden signs** by minimizing the fixed-amplitude variational energy over the one-parameter family
[
s_t(x)=s_0(x),mathrm{sign}[t-r_0(x)].
]
An oracle hidden-sign threshold was recorded only as a diagnostic of the capacity of the scalar coordinate.

## Square-lattice J1-J2 sweep
| J2/J1 | baseline | O_S^(0) | O_S^(1) | fixed-amp error/site: before -> after | P_stable^(1) |
|---:|---|---:|---:|---:|---:|
| 0.0 | Marshall | 1.000000 | 1.000000 | ~0 -> ~0 | 1.000000 |
| 0.2 | Marshall | 1.000000 | 1.000000 | ~0 -> ~0 | 1.000000 |
| 0.4 | Marshall | 0.999128 | 0.999983 | 6.05e-4 -> 1.18e-5 | 0.999992 |
| 0.5 | Marshall | 0.974538 | 0.999319 | 1.156e-2 -> 5.51e-4 | 0.999658 |
| 0.6 | Marshall | 0.795572 | 0.957131 | 7.570e-2 -> 3.714e-2 | 0.976555 |
| 0.8 | Marshall | 0.123664 | 0.061971 | 5.897e-1 -> 4.243e-1 | 0.584007 |
| 1.0 | Marshall | 0.064215 | 0.171346 | 8.700e-1 -> 6.135e-1 | 0.494345 |

The oracle threshold on the *same* r0 coordinate also collapses at large J2 with Marshall: O_S^(1,oracle)=0.443 at J2=0.8 and 0.285 at J2=1.0. Therefore this is not merely a bad threshold-selection algorithm; the Marshall r0 coordinate itself has lost the relevant sign information.

## J2-adapted baseline
Using the stripe bipartition appropriate to the dominant J2 antiferromagnet restores the one-step structure:

| J2/J1 | baseline | O_S^(0) | O_S^(1) | fixed-amp error/site: before -> after | P_stable^(1) |
|---:|---|---:|---:|---:|---:|
| 0.8 | stripe | 0.899845 | 0.975927 | 5.009e-2 -> 2.401e-2 | 0.986240 |
| 1.0 | stripe | 0.955819 | 0.994884 | 2.649e-2 -> 5.776e-3 | 0.997445 |

The x- and y-stripe gauges give the same numbers on the 4x4 torus. In both cases the label-free energy-optimal threshold reaches the same O_S^(1) as the oracle overlap threshold to numerical precision.

## Triangular-lattice falsification
After the internal square-lattice sweep, we tested the nearest-neighbor triangular Heisenberg antiferromagnet on a periodic 6x3 triangular Bravais torus (18 spins, S^z=0, exact diagonalization). We compared simple two-color gauges and optimized t by the same fixed-amplitude energy criterion.

For the best tested baseline:
- O_S^(0) ~ 0;
- O_S^(1) ~ 0;
- even the **oracle** threshold on r0 reaches only O_S^(1,oracle) <= 0.0604 across the tested gauges;
- best fixed-amplitude energy error improves only from about 0.2222/site to 0.1972/site;
- nevertheless P_stable^(1) is high, about 0.947 for that baseline.

This is the decisive failure mode: the post-update state can be locally stable while being globally in the wrong sign sector. Local-field stability alone is therefore not evidence that the sign problem has become simple.

## Structural criterion suggested by the data
ACCEPT:
1. A useful baseline gauge must place the state inside the correct broad sign basin (Marshall at small/intermediate J2, stripe at large J2).
2. Within that basin, the residual important sign corrections must be approximately monotone in the single scalar r0.
3. A label-free physical objective (fixed-amplitude energy) must select essentially the same threshold as the hidden-sign oracle.

REJECT:
- "One Krylov step generically resolves frustrated signs."
- "High P_stable^(1) implies the reconstructed sign sector is correct."
- A universal Marshall-based rule across the whole J1-J2 phase diagram.

INTERPRETATION:
The miracle is best understood as a near-unfrustrated-basin phenomenon in the effective fixed-amplitude sign-Ising problem. Intrinsic non-bipartite frustration produces many locally stable basins and destroys the one-dimensional ordering by r0.

## Reproducibility
Primary scripts:
- experiments/square_exact_energyopt.py
- experiments/triangular_exact_test.py

Raw outputs:
- results/square_exact_energyopt.json
- results/square_exact_energyopt.csv
- results/triangular_exact_test.json

Diagnostic showing why forced two-cluster thresholding is not universal:
- experiments/square_exact_sweep.py
- results/square_exact_sweep.json
