# Claude agent handoff — AVSL-12 (2026-10-21)

**Project goal:** scalable and *repeatedly improving* fixed-node (FN) -> Krylov sign update -> FN loop for frustrated spin-1/2 J1–J2 at J2/J1=.5, with honest all-in computational cost. Real bottlenecks: (a) FN suppresses important wrong-sign configurations, (b) the next Krylov sign function must be cheaply callable *anywhere*, and (c) successive independently projected FN energies must improve beyond first K1. Please DO NOT mistake good classifier metrics or single-flip frozen-amplitude scores for loop success.

## What ChatGPT did locally — Mac unavailable

Reused archived tilted **20-site** post-K1 sign+approximate amplitude table `sign20_training_data.npz` and AVSL10 reconstructed **M=128 FN branching** populations; no Mac/HPC. Reconstructed 4,654 translation/global-spin-inversion symmetry orbits, generated 6 independently seeded FN populations, and selected **1,500 symmetry-distinct orbits** from the distance-1/distance-2 population frontier for a **simulated local-field sign-verifier teacher-label budget**. Labels are a **fixed-amplitude local-field sign-update proxy**, defined by the condition `s(x) * Σ_y H_xy s(y)a(y) > 0` holds on the archived 20-site reference. This is **NOT equivalent to a finite-time \`(I-tH)\` K2 update unless the step-size criterion is separately established**; **98/1,500 labels are positive**. 1,250 labels fit, 250 internal model-selection; 699 held-out orbits (29 positive) are excluded from fit/selection. Spin features + current post-K1 sign only; neither exact amplitudes nor ED ground-state labels are model inputs. Training uses orbit-disjoint splits and random lattice-translation/spin-inversion augmentation; class balancing by oversampling within training only.

Architectures of roughly comparable modest size: plain MLP, lattice-topology graph network (separate J1/J2 message passing), and vanilla 2-layer 4-head transformer with free absolute position embeddings. Their purpose is predicting verified *Krylov* signs in previously unseen candidate configurations, not learning ground-state phase or a stand-alone NQS.

**Initial held-out test, 60 top-ranked candidate verifications among 699 unseen symmetry orbits:** MLP seed 11/21/31: **6,7,4** positive local-field labels; graph NN: **16,14,18**; vanilla transformer: **0,1,2**. Random expectation 60*29/699 = **2.49**. GNN held-out average precision .298/.336/.339; transformer .031/.031/.056, MLP .112/.158/.078. Extra transformer seeds 41 yield **2/60**; thus vanilla attention training has not rescued it. GNN is strongest in this *offline ranking* test.

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

**UPDATE 2026-10-21 — additional experiments COMPLETED locally.** The original status line is superseded by the next section.

## Results update: geometry-aware attention and actual FN energy

**Same 1,500 simulated teacher labels, same orbit-disjoint 699-test pool and top-60 candidate budget.** A 2-layer geometry-aware transformer with J1/J2-specific relation attention and no absolute position embeddings achieves held-out positive local-field labels at top 60 **14,9,19** (seeds 11/21/31). Compare GNN **16,14,18**, plain MLP **6,7,4**, ordinary transformer **0,1,2**. The geometry-aware attention model has ~46k parameters, GNN is still more stable across seeds, and neither has independently matched inference cost. A 32-configuration lattice-translation equivariance check passed at <2e-7 numerical difference. The poor vanilla-transformer result primarily exposes a **lattice geometry/inductive-bias issue**, not conclusive failure of attention.

**Independent physical fixed-node energy reoptimization**, guide held IDENTICAL across all arms: select highest-ranked 60 heldout test candidates, check the archived local-field oracle, flip ONLY the verified-positive full symmetry orbits (this uses exact 20-site orbit mapping), then rebuild the 184,756-dimensional FN Hamiltonian and solve again. Representative seed 11:
- GNN: **16 verified, actual FN energy gain +8.306e-6**.
- J1/J2-aware transformer: **14 verified, actual FN energy gain +9.877e-6**.
- Plain MLP: 6 verified, +2.032e-7.
- Unmodified transformer: 0 verified, zero gain.
- Random check of 60 test candidates (one seed): 3 verified, +2.160e-6.
Rayleigh-release certificate independently matched new matrix energy shifts to <1e-7 and always lower-bounded actual benefit. These small advances only establish **a single audited synchronous update using exact off-support labels**, not a scalable repeated FN–Krylov loop.

**Counterexample to direct use as a globally callable sign update:**
On ALL 4654 symmetry orbits, evaluate an orbit-consistent network rule (invert-spin prediction averaging, graph translation invariance). Flip every prediction with confidence >.95 or >.99 **without** an exact sign check, and independently recompute FN. Both GNN and geometry-aware transformer **RAISE** FN energy at both thresholds for seeds 11 and 31 (8/8 failures), even with >.99 confidence. For GNN seed 11, threshold .95: 67 orbit flips, 16 wrong local-field labels, physical FN energy worsened by **0.242594**. Threshold .99: 15 flips, 7 wrong labels, FN energy worsened **3.634e-6**. For geometry-aware transformer seed 11, threshold .95: 42 flips, 11 wrong, worsening **8.828e-5**; threshold .99: 7 flips, 3 wrong, worsening **2.251e-7**. Thresholds were evaluated exploratorily against all-orbit labels; **no calibration generalization claim**. Key conclusion: **verified proposals improve; treating network confidence as the truth is unsafe**. A compact predictor does not solve globally callable NEXT-sign correctness.

**Additional offline two-round audit (separate from end-to-end deployed loop):** Existing local 20-site data includes GNN seed31 first verified sign update with 131 corrected orbits, then independently FN reprojected: E_FN=-10.4180832766778 vs initial -10.4176976785033 (gain 0.000385598). Using the *same pretrained GNN* with changed sign input to rank **120 additional orbit candidates**, and labeling from the **exact independently reprojected 20-site FN eigenvector oracle**, found 10 new positives; combined 141 orbit flips then FN-reprojected to -10.4183523973140, another **0.000269121** improvement. These two FN eigenenergies were reproduced via an independent Hamiltonian build and solver (residuals ~1e-8). This is an encouraging **two-stage toy energy improvement with explicit exact-oracle leakage into labels**; it is NOT a feasible large-lattice Krylov oracle, an on-policy FN population update, or an economical new-sign function. Random K3 controls exist but comparison remains one chamber/one seed.

### Agent priority and hard-stop gate

1. **First** independently scrutinize the label definition: `sign[s(x)Σ_y H_xy s(y)a(y)]` is a local-field sign-step proxy, not necessarily finite-step Krylov `(I-tH)\psi`. Quantify the mismatch on 20-site before claiming 'K2 verification'.
2. Build a **callable** finite-step K2 sign check with amplitudes available only from realistic FN walker training data. Meter expensive off-support (a(y)/a(x)) ratios; no exact 184756 table, no free orbit enumeration, no leakage of 20-site post-FN phi into proposal/learner.
3. Compare GNN vs relation-aware attention at 120 **end-to-end** costed verifications and then actual independently reprojected ΔE_FN across rounds with freshly sampled walkers. A zero-shot model is not an accepted correction oracle.
4. If off-support oracle cost or incorrect global sign predictions wipe out energy gains, **stop NN architecture tuning**. Resolve amplitude estimates/verification and representation of the globally callable next sign before advancing to 6×6.

**Additional local source artifacts** now include `train_geoattention.py`, `reltransformer_results.json`, `joint_rayleigh_eval.py`, `independent_fn_reproject.py`, `independent_fn_results.json`, `global_callable_sign20.py`, `global_callable_sign20_results.json`, and `verified_two_rounds_fn.json`; these are *local ChatGPT runtime paths*, not GitHub files. The source reference is regenerable using `avsl08_prepare_reference20.py` on branch. 
