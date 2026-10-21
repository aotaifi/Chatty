# Claude agent handoff — AVSL-12 (2026-10-21)

**Project goal:** scalable and *repeatedly improving* fixed-node (FN) -> Krylov sign update -> FN loop for frustrated spin-1/2 J1–J2 at J2/J1=.5, with honest all-in computational cost. Real bottlenecks: (a) FN suppresses important wrong-sign configurations, (b) the next Krylov sign function must be cheaply callable *anywhere*, and (c) successive independently projected FN energies must improve beyond first K1. Please DO NOT mistake good classifier metrics or single-flip frozen-amplitude scores for loop success.

## What ChatGPT did locally — Mac unavailable

Reused archived tilted **20-site** post-K1 sign+approximate amplitude table `sign20_training_data.npz` and AVSL10 reconstructed **M=128 FN branching** populations; no Mac/HPC. Reconstructed 4,654 translation/global-spin-inversion symmetry orbits, generated 6 independently seeded FN populations, and selected **1,500 symmetry-distinct orbits** from the distance-1/distance-2 population frontier for a **simulated Krylov teacher-label budget**. Labels are whether the next fixed-amplitude Krylov sign-flip condition `s(x) * Σ_y H_xy s(y)a(y) > 0` holds on the archived 20-site reference; **98/1,500 labels are positive**. 1,250 labels fit, 250 internal model-selection; 699 held-out orbits (29 positive) are excluded from fit/selection. Spin features + current post-K1 sign only; neither exact amplitudes nor ED ground-state labels are model inputs. Training uses orbit-disjoint splits and random lattice-translation/spin-inversion augmentation; class balancing by oversampling within training only.

Architectures of roughly comparable modest size: plain MLP, lattice-topology graph network (separate J1/J2 message passing), and vanilla 2-layer 4-head transformer with free absolute position embeddings. Their purpose is predicting verified *Krylov* signs in previously unseen candidate configurations, not learning ground-state phase or a stand-alone NQS.

**Initial held-out test, 60 top-ranked candidate verifications among 699 unseen symmetry orbits:** MLP seed 11/21/31: **6,7,4** true corrections; graph NN: **16,14,18**; vanilla transformer: **0,1,2**. Random expectation 60*29/699 = **2.49**. GNN held-out average precision .298/.336/.339; transformer .031/.031/.056, MLP .112/.158/.078. Extra transformer seeds 41 yield **2/60**; thus vanilla attention training has not rescued it. GNN is strongest in this *offline ranking* test.

**Crucial cautions:** (1) 20-site exact callable amplitude/sign tables are used to generate teacher labels, not a physically realized scalable verifier; (2) full orbit enumeration and exact distances used for study design, cannot be taken as free in 6x6; (3) model uses exact stored current s(x), which is not yet globally cheap for later FN–Krylov iterations; (4) varying architectures do not have matched inference FLOPs or training compute, only simulated verification label budget; (5) held-out orbits are not necessarily different frustrated sign chambers; (6) rare configurations are from FN-support graph distance, not a validated independent long-length-scale test; (7) **no independent next FN reprojection or repeated loop success** shown.

## What is being tested now (NOT yet a conclusion)

1. Verify failure of vanilla transformer isn't just a faulty positional embedding: test **lattice-aware self-attention** with topology-aware J1/J2 relation features and symmetry-consistent pooling/augmentation, using exactly the same 1,250 fit/250 validation and 699 held-out orbits. Compare to the same GNN, not against a weaker straw man; report seed variability, top-k hits and query/inference costs.
2. Check the all-in advantage against simple physics/locality baselines and random selection; **pre-verification ranked corrections are hypotheses**, not accepted flips. Retain actual Krylov checks as mandatory.
3. Run a **single fully callable sign-update -> independently reoptimized FN step** for a small candidate subset and report changes in physical FN energy and runtime, if possible. Do not silently use ground-truth held-out labels to construct a deployed classifier; a classifier can propose checks but cannot be equated with verified signs.
4. Check symmetry consistency and possibility of high false-positive error if predictions are directly applied as global sign function. If no robust improvement, stop architecture tuning and investigate off-support FN amplitude-ratio oracle and economical Krylov verification.

### Required response from Claude

Please independently critique the above as a scaling experiment. In particular, is the test label/candidate sampling distribution realistic for M128 6x6? Are orbit features or exact current s implicitly leaking scalability assumptions? Propose ONE fair next full-loop test and explicit stop criteria. Avoid suggesting we simply repeat already-completed Krylov-first-step or another frozen-sign toy search.

### Artifacts

The local interactive ChatGPT runtime has `/mnt/data/avsl12/{prepare.py,select.py,train_models.py,evaluate.py,orbit_base.npz,experiment_split.npz,allrank_results.json,*.pt}`. The Mac does not have these local runtime paths until transferred. GitHub branch: `avsl/experiment-05-onpolicy-costed-2026-10-19`. Other project reports: `EXPERIMENT_09_LOCAL_J2_TIERED_2026-10-20.md`, `EXPERIMENT_11_NODAL_RELEASE_2026-10-21.md`.

**Status of current new experiments at note creation:** 3 architecture comparisons plus 1 extra transformer seed complete; geometry-aware transformer and independently FN-reprojected joint update still being evaluated, not demonstrated.
