# J1-J2 Krylov/FN next-step decision — updated 2026-09-27

## What is already established
1. The label-free one-step Krylov sign rule is extremely efficient:
   s_t(x)=s_M(x) sign[t-r_M(x)], r_M=(H a s_M)/(a s_M).
2. 6x6 and 8x8 flattened stress tests remain at roughly the 10^-3 overlap-deficit scale.
3. 8x8 physical alpha=2 evaluation with a frozen alpha=1.2 threshold gives O=0.9995117 on 4096 raw samples; MCMC autocorrelation reduces effective independent information to roughly 1100-1300 samples, so rare-error scaling is not yet closed.
4. Clean 6x6 fixed-amplitude energy test:
   E_K-E_ViT=+0.00420 +/- 0.01035 total energy (0.41 sigma, indistinguishable);
   E_K-E_M=-0.17247 +/- 0.04489 (3.84 sigma improvement).
   Thus Krylov signs recover essentially all physically important ViT sign structure without sign labels.
5. Exact 4x4 FN->amplitude->Krylov iteration improves strongly but is not exact: Marshall O=0.97454 -> first update O=0.99812 -> settled O~0.99791, residual energy bias ~3.39e-4/site.
6. Naive FN-walker histogram -> amplitudes is not scalable: ~1e6 ideal independent walkers give sign overlap ~0.99747 and 3e6 give ~0.99876, versus exact-amplitude one-step ceiling ~0.99926 on 4x4.

## Project priority
Do not spend the next effort on more standalone sign-overlap scaling and do not attempt a raw-histogram 6x6 FN loop.

The decisive missing component is:
FN walkers -> compact callable amplitude-ratio representation -> Krylov sign update.

The Krylov rule needs a(y)/a(x) for Hamiltonian neighbors. A walker cloud alone is not a callable amplitude function and its raw histogram scales badly.

## Best next construction: density-ratio amplitude correction
Use the previous callable guide amplitude a_k as the reference. Importance-sampled FN walkers target
f_k(x) ∝ a_k(x) a_FN,k(x),
while we can sample the reference distribution
q_k(x) ∝ a_k(x)^2.
Therefore
f_k(x)/q_k(x) ∝ a_FN,k(x)/a_k(x).

Fit only the log density ratio
g_k(x) ≈ log[f_k(x)/q_k(x)].
Then obtain a callable updated amplitude
a_{k+1}(x) ∝ a_k(x) exp(g_k(x)),
and, crucially,
a_{k+1}(y)/a_{k+1}(x)
= [a_k(y)/a_k(x)] exp[g_k(y)-g_k(x)].

This avoids reconstructing a full exponential histogram. It also means we only learn a positive amplitude correction, not the hard sign structure.

## Falsification order
A. 4x4 controlled test first.
   Generate exact FN mixed-distribution samples f and reference samples q.
   Fit progressively simple density-ratio models: scalar/linear physical features first, then a small compact model only if needed.
   Compare recovered neighbor amplitude ratios, resulting Krylov signs, and energy to the exact FN-amplitude result.
   Measure sample/model size required for a fixed error.

B. If A passes with modest resources, do the real 6x6 loop:
   Marshall/initial guide -> FN walkers -> density-ratio amplitude correction -> Krylov signs -> FN again.
   Track energy/site, sign-change physical weight, local-energy variance, walker count, projection length, and model size.

C. Only if 6x6 closes stably, repeat at 8x8 and perform true scaling.

## Interpretation
The attractive hypothesis remains: the difficult sign map may be generated almost for free by one Krylov application; the only remaining hard object is a positive amplitude update. The project succeeds only if that positive amplitude handoff can also be represented and learned with polynomial resources.
