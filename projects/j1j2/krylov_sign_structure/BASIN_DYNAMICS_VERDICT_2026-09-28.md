# Basin dynamics verdict — 2026-09-28

## Core mechanism

Fix positive amplitudes a(x) and signs s(x), and write psi_s(x)=a(x)s(x). Define

r_s(x) = (H psi_s)(x) / psi_s(x).

A Krylov step gives

[(t-H) psi_s](x) = [t-r_s(x)] psi_s(x),

so its sign is exactly

T_t[s](x) = s(x) sign[t-r_s(x)].

Our nonlinear sign iteration chooses t label-free by minimizing the fixed-amplitude energy E[T_t[s]] over all realizable thresholds, then restores the amplitudes a(x) and repeats.

## Exact monotonicity fact

The threshold family always contains the current sign pattern: for t > max_x r_s(x), T_t[s]=s. Therefore the energy-minimizing update obeys

E[T(s)] <= E[s].

With exact ground-state amplitudes a=|psi_*|, the exact ground-state signs s_* are a global minimum of E[s] and a fixed point because r_*(x)=E_0 for every x with nonzero amplitude.

Thus the iteration is best viewed as a **projected Krylov / one-dimensional global sign-descent map**. It is guaranteed not to raise the fixed-amplitude energy, but it is not guaranteed to reach the global minimum: other threshold-stable fixed points can exist.

## Why one step looked miraculous

At J2/J1=0.5 with Marshall signs, the starting pattern already lies in the attraction basin of s_* and the remaining defects are almost perfectly ordered by r_0. Hence the first threshold move removes nearly all important wrong signs:

O_S: 0.974538 -> 0.9993189 -> 0.99999823 -> 1.

The first step is therefore not a unique miracle; it is the first, exceptionally strong contraction step of an iterative descent.

## Square-lattice repeated dynamics

With exact amplitudes on the 4x4 square J1-J2 model:

- J2/J1=0.5, Marshall: exact signs after 3 updates.
- J2/J1=0.6, Marshall: exact after 4.
- J2/J1=0.8, **Marshall despite its one-step failure**:
  0.123664 -> 0.061971 -> 0.098225 -> 0.219270 -> 0.469895 -> 0.507588 -> 0.848923 -> 0.962882 -> 0.991197 -> 0.999846 -> 1,
  exact after 10 updates.
- J2/J1=1.0, **Marshall despite its one-step failure**:
  0.064215 -> 0.171346 -> 0.290203 -> 0.339260 -> 0.910391 -> 0.990778 -> 0.998562 -> 0.999975 -> 0.99999941 -> 1,
  exact after 9 updates.
- J2/J1=0.8 and 1.0 stripe starts reach exact signs in 4 updates.

Therefore a good gauge is **not required for eventual convergence on the tested square cluster**. It mainly makes the first step spectacular and shortens the path.

## Random-corruption basin profile on square lattice

Starting from the exact signs and randomly flipping configurations until a target physical probability weight q is wrong, with exact amplitudes fixed:

- all 32/32 sampled starts converged for q <= 0.42 in all three tested square cases (J2/J1=0.5, 0.6, and 1.0);
- at q=0.44 success was 30/32, 32/32, and 29/32 respectively;
- success then becomes model- and pattern-dependent as q approaches 0.5.

Since O_initial approximately equals |1-2q|, q=0.42 corresponds to only O_initial about 0.16.

This is **not a rigorous basin radius**: it is a random-corruption success profile. The important conclusion is that the ground-state basin is very large in physical-weight Hamming distance, while coherent structured errors can behave differently from random errors.

## Triangular-lattice control

On the periodic 6x3 triangular Heisenberg antiferromagnet with exact amplitudes, repeated updates from three simple two-color gauges do **not** approach the exact signs:

- maxcut_x reaches a wrong fixed point after 13 updates with O_S about 2e-16;
- parity_y reaches a wrong fixed point after 18 updates with O_S about 1e-15;
- parity_xy reaches a wrong fixed point after 18 updates with O_S about 2e-15.

However, the exact-sign basin itself is large. Random corruptions of the exact triangular signs recovered in all 8/8 trials through q=0.45 (O_initial about 0.10), and 4/8 trials recovered even at q=0.49.

Therefore the triangular failure is **not** that the projected Krylov map lacks a ground-state attractor. Rather, the simple structured gauges lie in other attraction basins / invariant structured sectors and flow to wrong threshold-stable fixed points.

### Correction to the earlier triangular one-step diagnostic

The original `triangular_exact_test.py` swept configurations one at a time even when r values were exactly/near-degenerate. Such an ordering can split a degenerate r group and therefore need not correspond to any realizable single threshold t.

The corrected grouped-threshold implementation treats equal/near-equal r values as inseparable. Under this valid threshold family the tested triangular two-color gauges retain essentially zero sign overlap throughout their flow. The earlier quoted oracle value <=0.0604 should therefore be treated as a legacy upper-bound artifact, not the current decisive result.

## Current conceptual verdict

**ACCEPT**
1. The update is exactly the sign of a Krylov vector (t-H)psi followed by a projection back to fixed amplitudes.
2. Choosing t by fixed-amplitude energy minimization makes the map energy non-increasing.
3. Exact ground-state signs are an attractive fixed point with a very large empirical basin under random sign corruption.
4. The one-step miracle at J2/J1=0.5 occurs because the Marshall state is positioned so that one threshold already removes almost all residual defects.

**REJECT / CORRECT**
- Repeated convergence on the square lattice does not require the J2-adapted gauge; even Marshall at J2/J1=0.8 and 1.0 converges.
- Basin membership is not determined by sign overlap alone.
- Triangular frustration does not destroy local contraction around the exact signs; it creates/permits competing structured basins that trap simple gauges.
- High local-field stability alone still does not identify the correct basin.

## Remaining bottleneck

The empirical mechanism is now clear. The next genuinely new theoretical question is to characterize the competing basins: what symmetry, sign-flux, or weighted-frustration quantity distinguishes a start that flows to s_* from one that flows to a wrong threshold-stable fixed point?

For algorithmic use, a separate practical question remains: how much of this survives when a(x) and the threshold-energy objective are only sampled/approximate rather than exact?
