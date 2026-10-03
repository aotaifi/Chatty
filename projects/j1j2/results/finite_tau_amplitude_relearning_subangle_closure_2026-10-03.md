# Finite-T amplitude relearning subangle — PARKED (2026-10-03)

## Scope
Finite-temperature J1-J2 propagator-column / CTQMC branch. Question: after a short analytic K1/K2 propagation block, can the updated positive amplitude be relearned/compressed into a directly callable NQS accurately enough to continue to high beta?

## What is already positive
- Exact short-time sign/amplitude propagation is strong and self-correcting.
- From the reconstructed 20-site guide at tau=.5 and dt=.0125, repeated exact K1/K2 propagation improves toward the exact state through beta=3.
- Nonlinear K1 factor log|1-dt E1| captures about 94% of the full K2 Hamiltonian-edge correction.
- Therefore the main unresolved issue is not the local sign-update rule itself.

## Computational obstruction
Exact recursive callable propagation expands rapidly in Hamiltonian shells:
- K1 representative state counts: about 56-58, 1.2k, 12k, 61-64k for recursion depths 1-4.
- Exact K2 recursion grows much faster and becomes prohibitive.
Thus long-beta work requires periodic amplitude compression/refresh.

## Closed unsuccessful compression families
State-only SR/TDVP, hybrid/closure SR variants, PairGNN state and edge objectives, local-potential GNN, simple normalized autoregressive GRU MLE, and MLE+edge supervision all failed to preserve Hamiltonian-neighbour ratios accurately enough under recursion.

## Final discriminator: factorized nonlinear overlap-SR
Paderborn job 3565142, completed successfully.
Setup:
- 20-site exact benchmark
- reconstructed M=100k guide, y=59279
- tau=.5, dt=.0125
- factorized residual PairGNN, 48 channels / 4 layers
- 80,961 parameters
- 20 nonlinear SR steps
- ns=1024, trust=.002, shift=.1
- overlap-SR objective toward exact one-block K2 amplitude target
- blind Hamiltonian-edge ratio audit

Reference:
- original physical fidelity at tau=.5 = 0.849313306956
- exact K2 one-block physical fidelity at tau=.5125 = 0.859770401498
- using the K2-updated sign but unchanged old amplitude already gives physical fidelity = 0.855530873703
- initial amplitude-target fidelity = 0.999683163532
- initial blind edge RMSE = 0.186851119241

Best early joint amplitude/edge point (step 3):
- target fidelity = 0.999701278680, improvement only +1.81e-5
- blind edge RMSE = 0.186353186726, improvement only 0.27%
- physical fidelity = 0.854144077238, already below the unchanged-amplitude baseline 0.855530873703

After 20 SR steps:
- target fidelity = 0.999648631359, worse than start
- blind edge RMSE = 0.185711321330, only ~0.61% improvement
- physical fidelity = 0.851821939587, substantially worse than the sign-only/unchanged-amplitude baseline

## Verdict
FAIL promotion. The factorized nonlinear overlap-SR update carries a small edge-ratio signal but does not materially learn the one-block amplitude refresh and degrades the physically relevant state. This is not enough to justify repeated learned-refresh recursion or finer CTQMC integration.

This result does NOT falsify the adaptive sign-repair/K1-K2 mechanism. It isolates the remaining bottleneck as callable positive-amplitude relearning with accurate Hamiltonian-neighbour ratios.

## Parking decision
- Park the finite-T/CTQMC subangle here.
- Do not implement a finer CTQMC framework yet.
- Do not launch another independent finite-T architecture/optimizer search.
- Generic callable-amplitude learning belongs to the shared ground-state FN amplitude project.
- Reopen finite-T only when that shared amplitude engine demonstrates a materially better scalable handoff, then adapt its objective to finite-dt propagation.
- When reopened, first test one-block neighbour-ratio preservation, then repeated exact-block -> learned-refresh recursion beyond beta=3, and only afterward replace exact expectations by QMC/walker estimates.

No active finite-T job remains after 3565142.
