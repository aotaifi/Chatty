# Finite-T CTQMC adaptive FN ↔ K1 checkpoint — 2026-10-01

## Goal
Construct a genuine continuous-time sampled finite-T algorithm, not purification and not a ground-state/projector reduction.

## Residual-defect falsifier
On the 20-site skew torus (D=184756), in the endpoint matching gauge, the Dyson residual sectors are dilute only on short blocks.
For y=63661:
- dt=.05: p0=.97672, p1=.02258, p2=.000683, p(m>=3)=1.95e-5, residual sign=.95481.
- dt=.10: p0=.91544, p1=.07778, p2=.00630, p(m>=3)=4.86e-4, residual sign=.84353.
- dt=.25: p(m>=3)=.02362, residual sign=.43391.
- dt=.50: p(m>=3)=.21977, residual sign=.09034.
Hard column y=59279 gives the same conclusion.
Thus dt~.05-.1 is a controlled local-defect regime; .5 is not.

## Bare released-residual walkers
Rejected as the cure. One-block residual CT sampling is correct, but repeated signed residual propagation plus endpoint annihilation develops a finite-population cancellation catastrophe. This motivates the positive FN backbone.

## FN residual target
With gauged H~=F[g]+C[g], the FN correction obeys C[g]g=0.
The local error is
epsilon_i = sum_{j bad(i)} h_ij (a_j/a_i - g_j/g_i).
Exact 20-site diagnostics confirm that reducing this residual orders next-block FN fidelity correctly. At tau=.5 on the hard column:
- uniform guide: ||Ca||=1.129, next Euler fidelity=.997642
- one-block-lag guide: ||Ca||=.529, fidelity=.999390
- exact guide: Ca~0, fidelity=.999758.
So local amplitude ratios, not global amplitude fidelity, are the right amplitude-side target.

## K1 sign refresh
Exact transient amplitude + recursive K1 is extremely accurate:
at tau=1, fixed endpoint-distance sign mismatch=6.22e-4 while recursive K1 mismatch=4.55e-9.

Using the actual approximate FN amplitude, not the oracle, also passes through tau=.5 on hard y=59279:
- fixed-sign adaptive FN baseline at .5: sign mismatch=2.6245e-5; full fidelity=.9998929.
- FN -> K1 self-fed loop at .5: carried sign mismatch=2.254e-6; full fidelity=.9999164.
Thus the FN↔K1 architecture itself is internally consistent.

## Finite-walker recursive loop
Uniform-importance positive FN walkers + sampled amplitude reconstruction + recursive K1 remains beneficial:
- M=8192 replicas: tau=.5 sign mismatch ~6.7e-6 to 7.7e-6.
- M=32768 replicas: ~6.0e-6 to 6.4e-6.
This beats the no-refresh fixed-distance value 2.62e-5, but increasing M does not remove the floor; representation/ratio bias dominates.

## Crucial learning-objective result
A raw-configuration MLP trained with ORACLE local edge targets
log a_j - log a_i at tau=.5 had poor global amplitude fidelity (0.6453) but K1 next-sign error only 1.47e-7.
Therefore global amplitude fidelity is not the relevant learning metric; local edge ratios can be sufficient for sign reconstruction.

Walker-only raw count-ratio training failed because the ratio labels are noisy:
cross-replica ratio RMSE ~0.80; K1 next error ~1.41e-5.

Raw-config walker pseudolikelihood is the correct sample-only route. After fixing the required degree correction deg(x)/K and early stopping:
- M=32768, 131072 local pairs
- independent-replica validation pseudolikelihood = .4823
- model amplitude fidelity = .9743
- current sign error = 5.29e-6
- carry-to-next sign error = 1.2299e-5
- K1 using learned ratios = 1.1999e-5
It is statistically alive but not yet accurate enough to approach the oracle local-ratio K1 result.

## Sign gating diagnostic
Two-replica consensus was tested and is too conservative. Oracle decomposition shows missed true crossings dominate false accepted flips by orders of magnitude.
At the late blocks, false-consensus flips are tiny while most true flip weight is missed.
Union-of-replica flips improves over consensus but still misses most late-tau crossings.
Conclusion: gating is secondary; local-ratio quality is the real bottleneck.

## Current verdict
ACCEPT: adaptive positive CT/FN -> K1 sign refresh as the algorithmic architecture.
ACCEPT: local bad-edge amplitude ratios / epsilon_i as the correct scalable learning target.
REJECT: bare released-residual signed walkers.
REJECT: global amplitude fidelity or classification AUC as the main training objective.
REJECT: raw walker count ratios as labels.
OPEN: sample-efficient local-ratio learning from positive CT walkers.

Next highest-ROI test: train one shared raw-configuration local-pseudolikelihood model across several adjacent short-time blocks (rather than only tau=.5), using independent replicas for validation, then insert that ratio model into the recursive FN->K1 loop. Do not scale system size before this closes.

## Steps 3–4: crossing-weighted learning and recursive replay
A fully walker-accessible proxy margin was built from the current sampled ratio model:
mhat_i = s_i ahat_i - dt (H s ahat)_i,
with relative margin |mhat_i|/(|ahat_i| + dt |H| ahat_i).
No oracle information entered the training weights.

Low-margin samples were upweighted by
w -> w [1 + 8 exp(-(mhat/.12)^2)].
However, the positive walkers essentially never visit the true crossing region:
fraction of training edge examples with proxy margin < .12 = 8.54e-5;
independent validation replica = 1.34e-4.
The 10th percentile walker margin at tau=.5 remains ~.66, while actual wrong-K1 states have exact median relative margin ~.033.

Critical-1% ratio RMSE, baseline -> margin weighted:
- tau=.25: 1.168 -> 1.195 (worse)
- tau=.40: 2.219 -> 2.127
- tau=.50: 2.308 -> 2.270
Thus reweighting cannot repair a region absent from the sampling measure.

Recursive sampled replay improves only slightly:
at tau=.5 unweighted shared model had current sign error 2.334e-5 and K1-next 4.184e-5;
margin-weighted model gives current 2.177e-5 and K1-next 3.928e-5.
Conclusion: reject ordinary importance reweighting as sufficient. The next problem is support acquisition / targeted sampling of the nodal-crossing boundary, not loss weighting.
