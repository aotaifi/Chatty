# 8x8 learned density-ratio amplitude verdict — 2026-09-30

## Run
Paderborn A40 Slurm 3494725, two independent M=128 replicas, seeds 12001 and 12002.
The Slurm job is marked FAILED only because the final summary script compared the old
4096-state threshold array to the new 6144-state train+validation threshold array.
Both projector replicas themselves completed and wrote valid NPZ outputs.

## Learned amplitude
The tested callable amplitude was
[
log a_{\rm learn}(x)=log a_0(x)+\eta g_\theta(x),qquad \eta=0.35966757,
]
where (g_\theta) is the cross-replica-validated periodic CNN density-ratio residual.

## Measured nodal change
Original K1 threshold:
- T0 = -28.3710787629.

Learned-amplitude threshold:
- T1 = -30.8572719392.

On the original 4096-state train set:
- 649 / 4096 states change K1 sign = 15.8447% raw;
- changed mass under old physical weights = 0.177499;
- changed mass under learned-amplitude reweighting = 0.447803.

This is not a small iterative correction.

## FN energies
Replica 12001:
- Emean = -27.00354306;
- tail8 = -26.91105698;
- 35,619,624 amplitude evaluations.

Replica 12002:
- Emean = -27.84396428;
- tail8 = -27.98289864;
- 33,905,734 amplitude evaluations.

Pair mean:
- -27.42375367;
- between-replica SE proxy = 0.42021061.

Established original 8x8 K1-FN baseline:
- E = -31.88542411;
- between-replica SE proxy = 0.00159161.

Thus the learned classifier update worsens the FN energy by about +4.46167 and destroys
replica stability.

## Interpretation
The FN propagation kernel in callable_fn_loop.py matches the successful original 8x8
GFMC kernel in its fixed-node construction, branching step, and population resampling.
The failure is therefore not explained by an obvious rewrite of the projector.

Treat eta=0.3597 classifier residual as falsified for production. This does not falsify
FN -> amplitude learning itself. The remaining question is whether the cross-replica
ratio signal is a useful infinitesimal direction under strong damping.

Next diagnostic:
- eta trust-region scan, Slurm 3495590;
- measure fresh K1 threshold and physical-weighted node-change mass for
  eta in {0,.01,.02,.05,.10,.20,.3597};
- if no small eta gives a controlled nodal move, close the classifier-residual route
  and pivot to full-network walker maximum-likelihood training.
