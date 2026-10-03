# 6x6 direct-ViT Krylov-FN pilot — 2026-09-28

## Purpose
Test whether the label-free one-step Krylov signs improve the actual lattice fixed-node projector energy at 6x6 when using the public ViT amplitudes directly, avoiding any Jastrow or learned amplitude surrogate.

## Projector validation
Before 6x6, the same fixed-population importance-sampled GFMC implementation was checked on exact 4x4 Marshall-FN:
- exact FN energy: -8.34152411548
- GFMC estimate: -8.34172756 +/- 0.0005366 (snapshot naive SE)
This validates the projector convention and implementation at the needed precision.

## 6x6 Krylov-FN pilot
Guide amplitude: public 6x6 ViT, 154,980 parameters.
Guide signs:
  s_K(x)=s_M(x) sign[t-r_M(x)]
with label-free frozen threshold
  t=-14.9857997791
from the alpha=1.2 Otsu 5-95 construction.
No hidden signs enter the FN projector.

Pilot controls:
- walkers M=32
- beta_target=0.9
- burn_beta=0.3
- tau_max=0.025
- initial states drawn from the same fixed physical sample pool used for the prior energy benchmark.

Result:
- Emean = -18.1279816529
- tail estimate = -18.1233878449
- naive snapshot SE = 0.00594775
- 13 stored projected energy snapshots
- 416 stored mixed-distribution walker states, 268 unique
- final run wall time ~650.6 s
- ~1,170,348 unique/cached ViT amplitude evaluations were required.

Early projected values:
beta=0.025: -18.1147
beta=0.125: -18.1159
beta=0.25:  -18.1891
beta=0.50:  -18.1368
beta=0.75:  -18.1346
beta=0.875: -18.1342

## Interpretation
The Krylov-FN energy is encouraging and appears lower than the earlier unmatched Marshall pilot (~ -18.00 late values), but this is not yet a controlled verdict because walker number/projection window/initial pool differed.

The important computational observation is also quantitative: exact on-the-fly Krylov signs inside FN require evaluating r_M on Hamiltonian neighbors, producing a second-shell cost. Even M=32 required ~1.17e6 ViT amplitude evaluations. This is polynomial but has a large prefactor.

## Matched control now running
A matched Marshall-FN control was launched through Chatty durable jobs with:
- M=32
- beta_target=0.9
- burn_beta=0.3
- tau_max=0.025
- identical initial pool / seed convention

Job:
- job id: 20260928-084549-47862
- target PID at launch: 47864
- job dir: /Users/aliotaifi/Chatty/jobs/20260928-084549-47862
- expected result: /Users/aliotaifi/j1j2_vit_bench/gfmc_6x6_marshall_matched32.npz

After completion, compare its projected window directly with:
  /Users/aliotaifi/j1j2_vit_bench/gfmc_6x6_krylov_vit_pilot.npz

If the matched Krylov-FN energy is materially lower, proceed to the four-bin density-ratio amplitude handoff at 6x6. If not, reassess before spending on the handoff.
