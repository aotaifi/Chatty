# PLAN_tooling (angle lead "tooling", session local_e7c8b147, started 2026-11-03; reports to the FN-loop lead)

## Question (PI, via the lead)
Our pipeline is custom JAX. Should we reuse anything from NetKet, Quantax, jVMC or similar, and would it improve training
or sampling? Ending test: a ranked table (item, what it replaces/adds, gain with evidence, effort, risk) and, for the top two
items, a CPU micro-benchmark or <= 1 A40-h benchmark against our implementation on the same inputs.
Budget: CPU + <= 2 A40-h (theorie). Rules: libraries only in /scratch/a/A.Otaifi/venvs/tooling/, copies only, no deletions.
Scope: speed and capability of what we run now (6x6 exact-table fits, sign classifiers, lattice FN-GFMC with ViT / PP /
two-PP + Jacobi guides). Not in scope: switching optimisers (PI fixed SR/minSR), retraining published networks.

## Where our time goes (measured, before any library)
- 10x10 ViT FN p=0 (A0.4 smoke, A40, f32h + f64 recheck): 3.5 s/walker-tau; the cost is ~226 one-swap neighbour evaluations
  of a 64-image symmetrised ViT per hop (symm f64 200 evals/s, f32h 3000/s; experiments/addon_10x10/README.md,
  delta_today 2026-11-02 18:10). No fast update exists for a ViT (attention mixes all patches after layer 1), so no library
  can change this term; only precision (done) and the image count can.
- 10x10 PP walker step, one core, 1600 images, inverse recomputed per image per step (T2 convention;
  /scratch/a/A.Otaifi/ansatz_runs/pfaffian_cost/run/walker_cost_L10_time.json, components_conf0):
  plain P 0.344 s = prep (inverse + T = f_DD G) 0.294 + one-swap kernel 0.051; J (one Jacobi level) 1.89 s, of which the
  two-swap (k=2) kernel 1.34 s and prep_full 0.39 s; D (two PP + J) 3.75 s.
  => a persistent inverse (low-rank updated along the walker path) can remove most of prep: P up to ~3x, J and D only ~1.2x
  (derived from the split, unvalidated; the k=2 kernel is unchanged by persistence).
- 8x8 M1 GPU walker (gpu_profile/PROFILE.md, DECISION.md): bond reductions were 49% (fixed by fastratio, H100 2.95x/2.04x);
  LU/inverse ~8%. The p=1 guide recomputes a fresh LU per neighbour per image (147 x 128 per hop).
- 6x6 fits (cx_r1fit.py): custom least-squares loss (value rows + bond-difference rows, decade-stratified), damped GN in
  sample space with a gather-built Gram matrix (solve_dir). Library SR/minSR drivers are built for the energy loss.

## Candidates (filled from angles/tooling/INVENTORY.md, worker, running)
lrux (Chen & Roth arXiv:2602.05255, JAX low-rank det/Pfaffian updates, delayed updates); Quantax (Chen, JAX + Equinox,
v0.4.0, MinSR examples, local updates, fermion mean field; whether it uses lrux: to check); NetKet (GCNN, ViT?, samplers,
VMC_SR/minSR, FullSumState); jVMC; mVMC (C, Pfaffian fast updates, arXiv:2105.13098; reference only).

## Ranked table (rank = expected value to the open questions A (10x10 ViT FN) and C (two-PP + Jacobi loop guide))
Versions: lrux 0.2.0, quantax 0.4.0, NetKet 3.23.0, jVMC 1.5.8 (angles/tooling/INVENTORY.md, file:line evidence).
| # | item | replaces / adds in our code | expected gain (evidence) | effort | risk |
|---|---|---|---|---|---|
| 1 | lrux `det_lru` (+ `det_lru_delayed`), persistent inverse along the walker path | PP walker "prep" (fresh inverse per image per step): fastratio GPU path, walker_cost C kernel | REJECTED by B1. Best form (H: lrux update + our table) 0.62-0.69 x our time on A40 (n=50), but near-node drift needs a refresh every 2 steps at worst (median 8-26); with refresh 0.69-0.87 x (<= 1.45x) on the plain step only, so <= ~1.1x on the Jacobi / two-PP guides. Delayed updates 4-6x slower | 2-3 d | drift near nodes (kappa up to 2e9 at 8x8; lrux has no refresh or condition check, det_lru.py; caller must re-invert); memory n^2 x images per walker (1600 images, n=50: 64 MB/walker, 16 GB at M=256); MIT |
| 2 | quantax `SingletPair` E_L path (inverse once per sample, one lrux ratio per connection; operator.py:314-327) | our fastratio all-bond table | REJECTED by B1: 1.68-1.82 x SLOWER than ours on A40 (n=50, B 256-4096), 2.7-4.0x on CPU. Cause: u^T Ainv per bond = (d x n)(n x n) product with d ~ 4n antiparallel bonds; it can only win if d < ~3n. Also a sign gauge (-1)^(p-q) differs from ours | 1 d | same as 1; ties us to quantax's Sites/operator API; MIT |
| 3 | lrux in the VMC Metropolis sampler (one ratio per proposal, update on accept) | rbmpp_lib.make_sampler: full log psi (all images, fresh slogdet) per proposal | derived, unvalidated: per proposal O(n^3) -> O(n^2) per image; sampling is ~20% of an 8x8 SR step by my op-count estimate (train.json t_sample is an async timer, not usable), so training ~1.1-1.3x. Only matters when we train a 10x10 PP guide (not now) | 2 d | as 1; acceptance in f32 today: persistent f32 inverse drifts faster |
| 4 | NetKet `ParallelTemperingSampler` / `ParallelTemperingExchange` (netket/sampler/parallel_tempering.py) | adds PT to VMC sampling | capability only; no sticking seen: 10x10 ViT VMC chains agree (chi2/dof 2.41, p 0.12, vit10_runs/a03); cost x n_replicas (default 32) ViT evaluations | 1-2 d | low; Apache-2.0 |
| 5 | quantax `solve_equation` (optimizer/qngd.py:150; Cholesky MinSR/SR by shape, f64 inside) or NetKet `_compute_srt_update` (private, _src/ngd/srt.py:22) | cx_r1fit.solve_dir | none: same damped Cholesky in sample space; ours builds the Gram matrix by gathers without materialising the (1+K)M bond-row Jacobian (code read); quantax centres the Jacobian by default (our offset column does the same job) | 0.5 d | private API (NetKet); no gain to measure |
| 6 | NetKet GCNN / DenseSymm, quantax symmetry module (C4v x translations x spin flip) | adds symmetric architectures | none now: guides are a published ViT and our PP; GCNN is not size-portable (TRIED_IDEAS K8: Marshall overlap 0.975 -> 0.322); projection we already have | - | - |
| 7 | NetKet `FullSumState` | exact-table training | none: no symmetry-reduced Hilbert space (vqs/full_summ/state.py), int32 index limit 2.1e9 < 6x6 Sz=0 9.1e9 states; our orbit tables stay | - | - |
| 8 | jVMC 1.5.8 | - | nothing NetKet/quantax lacks; pins jax <= 0.7.2 (we run 0.8.2) | - | version conflict |
| 9 | FN-GFMC / lattice projector code | - | none exists in any of the four (grep: 0 hits; 19 hits are TDVP imaginary-time and lanczos_ed); literature: PEPS + GFMC (arXiv:2406.12207) has no public library | - | - |
| 10 | mVMC (C/MPI, Pfaffian fast updates, arXiv:2105.13098) | reference only | cross-check of RBM+PP VMC energies if ever needed; not JAX | - | - |
Not library items but the larger levers for C at 10x10 (from the same split, lead's view): the image count (1600 = C4v x
translations x flip; translations x flip = 200 would cut every arm 8x if the energy cost is small) and the two-swap (k=2)
kernel, 71% of the Jacobi step.

## Benchmarks
### B1 (pre-registered 2026-11-03, before any timing): all one-swap ratios of one pair-product determinant F = f[U,D]
Facts read by the lead (not the worker's summary): lrux det_lru (lrux/det_lru.py:78-129) forms R = I + u^T Ainv v; for the
up-down exchange quantax.SingletPair passes u = (dense new-row diff, col idx), v = (dense new-col diff, row idx)
(quantax/model/fermion_mf.py:960-976), so each ratio costs u^T Ainv = O(n^2) (a matrix-vector product), i.e. O(d n^2) ~ 4 n^3
per configuration for all d ~ 4n neighbours. Ours (fastratio, rbmpp_lib.py img_quant_k fast) is inverse + three n x n GEMMs
(~5 n^3, GEMM-efficient). quantax's E_L builds the inverse once per sample (operator.py:314) like ours: no persistence.
Arms, same configurations, same f: R = ours (fresh inverse + 3 GEMMs + gathers); Q = quantax/lrux style (fresh inverse + one
lrux ratio per antiparallel bond); L = persistent inverse (lrux rank-2 update for one accepted exchange per step) + Q's ratios;
H = persistent inverse (lrux update) + R's GEMM table (removes only the inverse).
Predictions (time per configuration-image per step, n = 50, R = 1): CPU 1 thread: Q 1.0-3.0, L 0.6-1.5, H 0.6-0.85;
A40 f64: Q 1.5-5, L 0.8-3, H 0.6-0.9. Mechanism: Q and L replace GEMMs by d matrix-vector products (memory-bound on GPU);
H removes the O(n^3) inverse, ~35-45% of R's flops. The term that would be wrong if H misses: the inverse's share of R's time.
Decision rule: adopt a persistent inverse (L or H) in the PP walker only if A40 time <= 0.67 x R at n = 50 AND on a walk from
near-node configurations (8x8 production f) the ratio error vs a fresh inverse stays <= 1e-8 with a refresh interval K >= 20.
Adopt the quantax-style E_L path (Q) only if <= 0.67 x R. Otherwise: keep ours, record why.
### B1 RESULT (2026-11-03; worker, verified by the angle lead from /scratch/a/A.Otaifi/tooling_bench/b1/{time_a40,drift}.json; RESULT.md)
Jobs: A40 17691395 (0.133 A40-h, cip-cl-nv01), CPU 17691326 / 17691351, drift 17691422. Code angles/tooling/bench/b1_lib.py.
| time / ours (R), n = 50, complex128 | CPU 1 thread B=64 | A40 B=256 | A40 B=1024 | A40 B=4096 |
|---|---|---|---|---|
| Q quantax/lrux E_L | 2.66 | 1.68 | 1.79 | 1.82 |
| L persistent lrux + lrux ratios | 2.25 | 1.37 | 1.50 | 1.52 |
| H persistent lrux + our table | 0.58 | 0.62 | 0.68 | 0.69 |
R = 22.8 us per configuration per step on A40 (IQR < 1%). Fresh inverse alone = 0.39 of R (predicted 35-45% of R: held).
Predictions vs measured: A40 Q 1.5-5 / 1.68-1.82, L 0.8-3 / 1.37-1.52, H 0.6-0.9 / 0.62-0.69 (all inside); CPU Q 1.0-3.0 / 2.66
(inside), L 0.6-1.5 / 2.25 (MISSED high: under vmap u^T G for all bonds is one (d x n)(n x n) product, d ~ 4n > 3n), H 0.6-0.85 / 0.58.
Drift (8x8 production f, 50 near-node starts; first accepted step with ratio error > 1e-8 vs a fresh inverse): uniform walk
elementwise median 8, min 0-2, max 32; normwise median 26, min 2; |psi|^2 Metropolis walk normwise median 12, min 2 (37 of 50
never exceed). cond(F) along the walk median 4.5e3, max 1.2e8. The same growth in plain numpy Woodbury: the cause is cond(F), not lrux.
Correctness: 0 sign mismatches in all arms; broken control (u/v indices exchanged) fails by 1e5-1e6. Elementwise 1e-10 is not met
even by R at n=32 (1.6e-8; exact-rational check: eps x cond(F) of any inverse-based formula, shared by R and Q).
VERDICT by the pre-registered rule: keep ours. H passes the time test only at B=256 and fails K >= 20 on drift; L and Q fail time.
Not checked: drift at n=50, complex f, symmetry images, RBM factor, a full walker.
### B2 = the Q arm of B1 (quantax E_L path, item 2): rejected above. No solver benchmark (item 5: same Cholesky; nothing to gain).

## Answer (2026-11-03)
Reuse nothing in production. None of NetKet 3.23, quantax 0.4, lrux 0.2 or jVMC 1.5.8 has lattice FN-GFMC, a fast update for a
ViT, exact-table training with symmetry-reduced sectors, or a custom-loss minSR better than ours. The one candidate with a
mechanism (lrux persistent inverse) loses its 1.45-1.6x on the plain PP step to refreshes forced by ill-conditioned F near
nodes, and the Jacobi / two-PP guides spend 71%+ in the two-swap kernel it does not touch. Keep the venv
(/scratch/a/A.Otaifi/venvs/tooling) for cross-checks; NetKet's parallel tempering is the first thing to try if 10x10 VMC chains
ever stick. Angle CLOSED.
