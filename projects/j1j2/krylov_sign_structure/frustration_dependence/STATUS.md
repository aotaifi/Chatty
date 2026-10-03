# Frustration-dependence subproject

## Goal
Determine when and why a single projected Krylov sign step reconstructs the sign structure across frustration in the 8x8 square-lattice J1-J2 model.

## Core hypothesis
The useful K1 signal is the amplitude-weighted off-diagonal competition field
F = O1 + O2,
where O1 is Marshall-favored J1 hopping and O2 is frustrating J2 hopping.
At J2/J1=0.5 the physical combination is already near the best linear separator of hidden sign defects.

## First decisive sweep
Existing 8x8 ViT checkpoints:
J2 = 0.00, 0.20, 0.40, 0.50, 0.60, 0.80, 1.00.

For each point:
1. sample independent alpha=1.2 train/validation configurations;
2. compute physical |a|^2 importance weights;
3. measure Marshall wrong-sign mass;
4. construct label-free one-step K1 threshold from the full Marshall local energy;
5. measure K1 residual wrong-sign mass and fraction of Marshall error removed;
6. decompose r_M = D1 + D2 + O1 + O2;
7. diagnostic-only oracle scan F_lambda = O1 + lambda O2, choosing threshold on train and evaluating on validation;
8. compare physical lambda=1 to the best lambda and track whether separability degrades with frustration.

Hidden ViT signs are diagnostics only and never choose the production K1 threshold.

## Current job
Paderborn A40 Slurm job 3492470 (started immediately; pending H100 duplicate 3492469 was cancelled before use).
Script: frustration_sweep_8x8.py
Expected output: results/frustration_sweep_8x8_3492470/frustration_sweep_8x8.json
Partial JSON is written after each J2 point.

## Decision
If physical lambda=1 stays near-optimal while K1 removes most Marshall wrong-sign mass, infer a robust frustration-field mechanism.
If optimal lambda drifts strongly or K1 removal collapses as J2 increases, identify the crossover and inspect what extra sign structure appears there.
