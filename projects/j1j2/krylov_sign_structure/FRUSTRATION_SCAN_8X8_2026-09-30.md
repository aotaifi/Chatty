# Frustration dependence / K1 reconstructibility — 8x8 scan

Opened 2026-09-30 as active Angle 3.

## Question
Is the near-perfect one-step Krylov sign reconstruction at J2/J1=0.5 a special accident, or does it follow a structural rule tied to the actual frustrating coupling?

## Decomposition
For each 8x8 ViT checkpoint, decompose the Marshall local field into

r_M = D1 + J2 D2_bare + F1 + J2 F2_bare,

where F1 is the amplitude-weighted NN off-diagonal Marshall field and F2_bare is the corresponding NNN field before multiplying by J2.

The structural classifier family is

F_lambda = F1 + lambda F2_bare.

The physical Hamiltonian chooses lambda = J2.

## First sweep
J2 = 0.00, 0.20, 0.40, 0.50, 0.60, 0.80, 1.00.

Existing 8x8 ViT checkpoints only; no FN and no retraining.
Per point: 1024 train + 512 independent validation samples from |a|^1.2, reweighted to |a|^2.

## Measurements
1. Marshall physical wrong-sign mass.
2. Label-free robust-kmeans K1 threshold and residual wrong-sign mass.
3. Fraction of Marshall wrong-sign mass removed by K1.
4. Hidden-sign diagnostic only: train-optimal lambda* for F1 + lambda F2_bare.
5. Apply the train-selected lambda*/threshold to validation.
6. Compare lambda* to the physical J2.

## Decision logic
High-value positive result:
- physical lambda=J2 lies near lambda* across the frustrated regime,
- label-free K1 removes most physical wrong-sign mass,
- degradation correlates with a measurable loss of separability/margin as J2 changes.

This would support a general criterion: one-step sign reconstructibility occurs when the Hamiltonian's amplitude-weighted competing hopping field is itself close to the optimal separator of the sign sectors.

Negative/falsifying result:
- lambda* varies independently of J2 or overfits train and fails validation,
- or K1 quality has no systematic connection to physical off-diagonal-field separability.

Then the J2=0.5 success is more model/point-specific and Angle 3 should not be generalized without another mechanism.

## Production
Paderborn Slurm H100 array: 3492472
Array tasks 0-6 map to J2 = [0, .2, .4, .5, .6, .8, 1.0].
Watcher/email attached through Chatty local loop.
