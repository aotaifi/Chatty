# 6x6 FN -> amplitude -> Krylov closed-loop verdict — 2026-09-29

## Production run
Paderborn Slurm job 3466255, 1x H100, 16 CPUs, M=128, two independent replicas.
Wall time: 29m41s. Exit code 0.

## Iteration-2 construction
Starting stable first handoff:
g0 = [-0.0409143292, 0.0032384247, 0.0254417125, 0.0120598933]
with eta=0.5 in the static r0 four-bin coordinate.

The corrected amplitude inferred a new label-free Krylov threshold
T1 = -14.651543935454582
from T0 = -14.985799779143964.

## Decisive fixed-point diagnostics
Physical-weighted sign change between the old and reconstructed guide:
sign_change = 0.0

Reweighted guide energy:
E_guide = -18.1165515702

Two independent FN replicas:
- seed 9701: Emean = -18.1165930405, tail8 = -18.1538559140
- seed 9702: Emean = -18.1093514526, tail8 = -18.1228984960

Mean of replica Emeans:
-18.1129722465
between-replica SE:
0.0036207939

Second density-ratio corrections:
replica 1 = [ 0.01220270, -0.05973187, -0.02570743,  0.07320811]
replica 2 = [-0.00289623,  0.04042602,  0.03379327, -0.07122193]

Combined:
g2 = [0.00378861, -0.01087963, 0.00257648, 0.00450005]

Pairwise noise estimate |delta g|/sqrt(2):
[0.01067656, 0.07082232, 0.04207335, 0.10212746]

Thus every |g2_i| is <0.36 sigma_pair; RMS(g2)=0.00632 while RMS(noise)=0.06582.
With eta=0.5, the largest implied next log-amplitude correction is only 0.00544.

## Verdict
Within the tested M=128 / beta=1.2 regime, the 6x6 loop has reached a practical fixed point:
1. the Krylov sign guide is unchanged on the physical diagnostic measure after the first amplitude handoff;
2. the next amplitude correction is statistically consistent with zero and an order of magnitude below replica noise;
3. FN energies are stable between replicas.

Do NOT launch iteration 3 by applying the noisy g2 estimate; that would inject sampling noise rather than resolve a measurable drift.

Next scientific step: carry the converged label-free FN/Krylov protocol to 8x8 and test whether the same fixed-point/scaling behavior survives.
