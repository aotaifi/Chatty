# Iterated Krylov sign convergence — 2026-09-28

## Question
The one-step tests suggested that the Krylov/local-field rule works when the initial sign gauge is already in the correct broad basin. Does recomputing the local-energy coordinate after each sign update then contract all the way to the exact ground-state sign structure?

## Controlled test
Periodic 4x4 square-lattice spin-1/2 J1-J2 Heisenberg model in S^z=0, exact diagonalization.

Hold the exact ground-state amplitudes a(x)=|psi_*(x)| fixed throughout and update only signs:

r_k(x) = H[a s_k](x) / [a(x) s_k(x)]

s_(k+1)(x) = s_k(x) sign[t_k-r_k(x)]

The threshold t_k is chosen without ground-state sign labels by exactly minimizing the fixed-amplitude variational energy over every realizable threshold of r_k. Equal or near-equal r_k values are flipped as one group, so no artificial splitting of symmetry-degenerate values is allowed.

Exact signs are used only afterward to score O_S and to record the oracle-best threshold on the same scalar coordinate.

## Results

### J2/J1 = 0.5, Marshall start
O_S:
0.974538 -> 0.9993189 -> 0.99999823 -> 1.000000

Energy error/site:
1.1563e-2 -> 5.5138e-4 -> 1.3188e-6 -> machine zero

The exact sign structure is reached after **3 Krylov sign updates**.

### J2/J1 = 0.6, Marshall start
O_S:
0.795572 -> 0.957131 -> 0.991267 -> 0.999444 -> 1.000000

Energy error/site:
7.5696e-2 -> 3.7139e-2 -> 8.0092e-3 -> 5.3771e-4 -> machine zero

The exact sign structure is reached after **4 updates**.

### J2/J1 = 1.0, stripe/J2-adapted start
O_S:
0.955819 -> 0.994884 -> 0.9996918 -> 0.99999944 -> 1.000000

Energy error/site:
2.6492e-2 -> 5.7764e-3 -> 4.0848e-4 -> 7.9304e-7 -> machine zero

The exact sign structure is reached after **4 updates**.

## Strong diagnostic
At every nontrivial iteration in all three cases, the sign overlap reached by the **label-free energy-minimizing threshold** equals the overlap of the **hidden-sign oracle-best threshold** on the same r_k coordinate to numerical precision.

Thus the observed convergence is not being limited by threshold selection in these tests: once inside the appropriate basin, the scalar Krylov coordinate itself exposes the remaining sign defects in the correct order.

## Verdict
**PASS — basin contraction on the tested exact cluster.**

This file records the first representative iteration test. A later wrong-gauge test showed that the attraction basin on the 4x4 square cluster is substantially larger than inferred here: even Marshall starts at J2/J1=0.8 and 1.0 eventually reach the exact signs. Thus “good sign basin” is sufficient for rapid convergence but is not required for eventual convergence in the tested square cases.

See `BASIN_DYNAMICS_VERDICT_2026-09-28.md` for the current interpretation.

Two steps are not literally exact:
- at J2/J1=0.5, step 2 gives O_S=0.99999823;
- at 0.6, step 2 gives O_S=0.991267;
- at 1.0 stripe, step 2 gives O_S=0.9996918.

But a few iterations reach the exact ground-state signs and exact fixed-amplitude energy.

## Scope / caveat
This is a **mechanism test**, not yet a scalable ground-state algorithm, because the amplitudes were fixed to the exact ground-state amplitudes and the threshold energy was evaluated exactly on the finite Hilbert space. What is established is contraction of the sign map once the amplitudes and broad sign basin are good; approximate/scalable amplitude and threshold handoffs remain separate questions.

## Reproducibility
- experiments/iterated_krylov_exact.py
- results/iterated_krylov_exact.json
- results/iterated_krylov_exact.csv
