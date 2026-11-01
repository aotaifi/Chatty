# AVSL-12: Can spatial attention / RL repair rare FN signs? Local proof-of-concept
2026-10-21, tilted 20-site J1-J2, J2/J1=.5, local CPU only.

## Goal
Learn to focus expensive Krylov checks on **suppressed wrong-sign configurations** without exponential enumeration, make real fixed-node (FN) energy improve over repeated iterations, and eventually scale past 20 sites.

## Controlled architecture experiment
- Genuine reconstructed M128 finite-beta FN walkers, six independent populations, provide 1/2-hop candidate neighborhoods. Archived exact 20-site amplitudes/signs are oracle data, not neural-network input.
- 1,500 symmetry-distinct training/validation Krylov labels, **same labels and budget for each architecture**. 1,250 fit + 250 model selection, 699 held-out orbit test (29 positive K2 labels), all separated by translation/global-inversion symmetry orbit. Inputs are spin configuration and currently assigned sign. Training augments lattice translation and global spin inversion.
- Architecture mean held-out AP over three initialization seeds: random .0415; MLP .116; ordinary 2-layer transformer .0395; physics J1/J2 graph network **.324**; bond-biased relative-attention transformer **.299** (best seed .435). A generic transformer fails; physical bond-based attention can succeed. **RL was not run**, since direct supervised teacher labels already discriminate approaches.
- On the SAME 5,000 locally generated one/two-hop candidate attempts from an independent actual M128 FN run, 120 teacher checks per arm: beta=.38: GNN 12/12/15 discoveries across seeds 11/21/31, bond-aware transformer 9/8/13, random mean ~2.7; beta=3.0: GNN 9/12/12, bond-aware transformer 8/7/12, random mean ~2.6.

## Independent fixed-node energy after accepted verification
Every updated sign was accepted only after the archived K2 or freshly computed K3 Krylov local-field verification. Same positive guide **a** was held fixed across FN operators, followed by independent large sparse FN eigensolves (residual ~7e-8).
A **fully chained walker-generated** two-round sequence for GNN31:
| Stage | FN ground energy | Improvement from preceding stage |
| --- | ---: | ---: |
| Starting post-K1 signs | -10.417697678503 | — |
| 98 verified K2 labels obtained during 1,500 training checks | -10.417915118494 | +.000217440 |
| 5,000 walker-local proposals, 120 checks, 15 newly verified K2 flips, independent FN update | -10.418348017727 | +.000432899 |
| NEW on-policy M128 FN population; next 5,000 proposals, 120 checks, same pretrained GNN, 3 newly verified K3 flips, independent FN update | **-10.418364701445** | **+.000016684** |

Second iteration selection is **not convincingly superior to random**: 3 verified K3 flips per 120 checks vs random mean 2.27 from the same next-chamber pool. The exact old-FN Rayleigh certificate of those 3 newly selected flips was +1.40e-5, consistent with the independently recomputed gain. In a non-scalable global-orbit ranking variant, results were larger, but **the global enumeration must not be confused with an implementable algorithm**.

On the first β=.38 FN candidate pool, same 98 training corrections + 120 subsequent *verified* ranker suggestions produced GNN FN energy -10.4183360/-10.4183406/-10.4183480 vs physics relative-transformer -10.4183247/-10.4183267/-10.4183378 (three seeds). Both improve energy. These comparisons are exploratory and the 1,500 training teacher calls MUST be included in end-to-end cost.

## Hard negative result: no direct sign predictor yet
The trained GNN31's unchecked top 120 candidate predictions contain 33 true K2 labels and 87 wrong flips; applying them as a global update and reoptimizing FN raises energy from -10.4176977 to -10.0958124. High-confidence unchecked updates for GNN and bond-aware transformer also failed direct FN-energy tests. **The neural model may rank checks, but cannot replace expensive verification or serve as a reliable globally callable sign correction rule.**

## Scaling verdict
A compact physics-aware ranker enriches useful rare sign candidates and permits two improving local FN updates at 20 sites; the advantage at the second verified iteration is weak. The full reference K2 labels, fresh K3 labels, FN walker rates, 20-site canonical orbit indexing and current-sign input still rely on **full 20-site oracle tables**, not demonstrated scalable procedures. Approximate finite-β walker mixing is unvalidated. A compact callable next-sign representation that is accurate off support remains an OPEN PROBLEM. **Do not start RL yet**. Gate next work on a costed callable verifier/sign representation and complete loop at 6x6 and 8x8; compare actual energy per *total* cost to simple heuristics.

## Reproducibility
Full scripts, 20-site reference data, trained model state_dicts (3 seeds each for ordinary transformer/GNN/MLP/relative transformer), orbit-level train/test split, raw comparisons, independent FN eigensolve logs, the more detailed 10 KB report and six passing integrity tests were produced locally as a ChatGPT conversation artifact: `AVSL12_Transformer_vs_GNN_20site_results.zip`. Source paths in some scripts assume `/mnt/data/avsl12` and should be adjusted in a repository checkout. No Mac or HPC connection was used.
