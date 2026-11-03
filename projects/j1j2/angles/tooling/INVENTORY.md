# NQS tooling inventory (read from installed sources, no benchmarks)

Venv: /scratch/a/A.Otaifi/venvs/tooling (Python 3.11.5, /software/opt/el_9/x86_64/python/3.11-2023.09). Freeze: .../tooling/FREEZE.txt.
SP = /scratch/a/A.Otaifi/venvs/tooling/lib/python3.11/site-packages. All paths below are relative to SP.
Versions: jax/jaxlib 0.8.2 (+ jax-cuda12-plugin/pjrt 0.8.2, nvidia cu12.9 wheels), NetKet 3.23.0, quantax 0.4.0 (plain, NOT `[full]`),
lrux 0.2.0, fermix 0.1.0, jVMC 1.5.8, flax 0.12.6, equinox 0.13.8, optax 0.2.8, numpy 2.4.6, mpi4py 4.1.2.
Install: wheels only (pip download --only-binary into /scratch/a/A.Otaifi/venvs/tooling_dl, then pip install --no-index). No setup script ran.
- jVMC: the resolver first picked jVMC 0.1.0 (imports `flax.nn`, removed -> ImportError). I replaced it by jVMC 1.5.8
  (wheel in tooling_dl_jvmc_whl/) installed with `--no-deps`: its METADATA pins `jax<=0.7.2`, `openfermion`, `matplotlib`, which conflict with
  jax 0.8.2 / are missing. `import jVMC` works on CPU with jax 0.8.2; nothing of jVMC was run beyond import. openfermion/matplotlib absent.
- `python -I -c "import jax, netket, quantax, lrux, jVMC"` with JAX_PLATFORMS=cpu, x64 on: OK, `jax.devices()` = [CpuDevice(id=0)]. CUDA not testable on login node.
- Only lrux was executed (my test /scratch/a/A.Otaifi/tmp/toolwork/t_lrux.py, CPU, complex128): rank-2 row+col update under jit+vmap
  matches brute-force det ratio to 1.1e-14, new inverse to 2.6e-15. Everything else below is from reading code, not from running it.

## a. lrux 0.2.0 (MIT, "Copyright (c) 2025 Chen Ao", lrux-0.2.0.dist-info/licenses/LICENSE)
Public API (lrux/__init__.py:1-15): det_lru, DetCarrier, init_det_carrier, merge_det_delays, det_lru_delayed; skew_eye, pf_lru, PfCarrier,
init_pf_carrier, merge_pf_delays, pf_lru_delayed.
- det low-rank update: yes, `lrux/det_lru.py:159 det_lru(Ainv, u, v, return_update=False)`. A1 = A0 + v u^T, ratio = det(I + u^T Ainv v) (det_lru.py:78-96, 103).
  Needs the inverse Ainv (n,n) as input. u,v can be dense (n,k), one-hot integer indices, or a tuple (dense, one-hot) (det_lru.py:24-66, docstring 194-300).
- Ratio only: yes, return_update=False -> `_det_lru_no_update` (det_lru.py:99-104), cost O(n k) for dense, O(k^2) for pure one-hot u,v (gather Ainv[idx,idx]); returns no new inverse.
- Inverse update: yes, return_update=True -> (ratio, Ainv_new) via Woodbury, `_det_lru_update` det_lru.py:107-129 (one fused det+inverse of the k x k matrix R, `det_inv`).
- Delayed updates: yes. `init_det_carrier(A, max_delay, max_rank=1)` det_lru.py:400-433 (computes Ainv=jnp.linalg.inv(A) at :430, a,b of shape (T,n,K) at :431-432);
  `det_lru_delayed(carrier,u,v,return_update,current_delay)` det_lru.py:576 (formula for R_tau in docstring ~:600-630, impl :501-543); `merge_det_delays` det_lru.py:436-466 folds a,b into Ainv
  (the caller must call it when current_delay == max_delay-1; not automatic).
- Pfaffian: `pf_lru(Ainv,u,return_update)` pf_lru.py:63 (pf(A0 - u J u^T), rank 2k), `init_pf_carrier` :242, `merge_pf_delays` :278, `pf_lru_delayed` :397. Skew-symmetrises Ainv after each update (pf_lru.py:230, 272).
- Batched/vmap: yes, docstring "compatible with jax.jit and jax.vmap" (det_lru.py ~:180-195, recommends `vmap(det_lru, in_axes=(0,0,0,None))`, `jit(..., static_argnums=3, donate_argnums=0)`). Tested by me (above).
  Shared Ainv with vmap over many (i,j) index pairs also traces and runs (tested for 64 pairs, shape only, not timed).
- complex128: yes, tested by me; _small_inv.py:63 dtype set = float32, float64, complex64, complex128. Triton/Pallas fused k x k kernel only on CUDA GPU, k <= 32 (_small_inv.py:12-20, :43 SMALL_MAX=32); elsewhere LU.
- Rank-2 simultaneous row+column replacement (up-down exchange in det F[U,D]): yes, documented example "Simultaneous update of row and column" det_lru.py:~290-310:
  `u=(row_vec, col_idx)`, `v=(col_vec, row_idx)`, where row_vec = new row - old row and col_vec = new column - old column (convention verified against brute force by my test).
- Memory per state: plain: Ainv (n,n) only. Delayed: Ainv + a + b = n^2 + 2 T n K entries (det_lru.py:394-398, 431-432); Pf carrier: Ainv, a, Rinv (pf_lru.py:236-240). 
  n=60 complex128 plain: 60*60*16 B = 57.6 kB per walker.
- Numerical safeguards: only (i) singular R (zero pivot / non-finite inverse) -> ratio 0 and zero inverse (_small_inv.py:22-24); (ii) Pf Ainv re-skew-symmetrised. No condition number check, no
  automatic recompute from scratch; the caller must re-invert (init_det_carrier) every k steps itself.

## b. quantax 0.4.0 (MIT, "Copyright (c) 2024 Chen Ao", quantax-0.4.0.dist-info/licenses/LICENSE; requires jax>=0.8.0, lrux>=0.2.0)
- Uses lrux: yes. quantax/model/fermion_mf.py:10 (import), :266/:271 (GeneralDet), :744/:749 (GeneralPf), :971/:976 (SingletPair); model/backflow.py:187-197 (DetBackflow), :404-413 (PfBackflow).
- Models with fast low-rank ref_forward (class RefModel, nn/modules.py:90): GeneralDet fermion_mf.py:133, GeneralPf :597, SingletPair :774, DetBackflow backflow.py:27, PfBackflow :202,
  GeneralJastrow jastrow.py:229 (the mean-field part uses lrux; network factor recomputed from scratch, jastrow.py:362-363, :375).
  Models WITHOUT it (plain eqx.Module): RestrictedDet :276, UnrestrictedDet :355, MultiDet :445, MultiPf :981, PartialPair :1051, Jastrow jastrow.py:49, ResConv conv_nets.py:72, SingleDense/RBM_* shallow_nets.py.
- "Local updates" for CNN/ResConv (light cone caching): NOT found in 0.4.0. grep for "light cone|lightcone|cache" in quantax/: no model code. The only RefModel implementers are the six above. 
  I did not read the online tutorial; it may refer to a newer version or to a different mechanism.
- SingletPair (fermion_mf.py:774-): psi(n)=det(n_up * F * n_dn), docstring :776-779; init_internal :862-875 stores (idx_up,idx_dn), inv=inv(F[idx_up,idx_dn]), psi (MF_Internal, :38-42).
  ref_forward :917-978: up hop = row update, down hop = column update, combined into ONE lrux.det_lru call with u=(row_update.T, col_update_idx), v=(col_update,row_idx) (:965-971) = the rank-2 row+column case.
  For spin sites (J1-J2) a spin exchange has nflips=2 -> nhops_up=nhops_dn=1 (:934-936): exactly the up<->down exchange as rank-2 update. Requires equal Nup = Ndn (docstring :778).
  Ratios use return_update=False (:976) i.e. ratio-only on the stored Ainv.
- Local energy path: operator/operator.py `_Oloc` :288-348: init_internal(s) once per sample (:314, O(n^3) inverse), then for every connected configuration (grouped by update mode)
  `state.segment_ref_forward` (:327) = ratio-only lrux call. State side: state/variational.py:400-420: `internal = filter_tree_map(lambda x: x[idx_segment], internal)` (:405) gathers the
  (n,n) inverse once per connected config under vmap (so the n^2 buffer is replicated per connection inside a chunk unless XLA fuses the gather; not measured). chunking by ref_chunk (:189-194).
  Update-mode filters: operator/update_mode_filters.py (nflips_filter, nflips_up_dn_filter :55); requirement check operator.py:188-193.
- MCMC with fast updates: Metropolis.__init__ metropolis.py:95-105 (use_ref if sampler update_mode provides state.required_update_modes, else warning "fast local updates are not utilized");
  SpinExchange (common_samplers.py:102, update_mode {"nflips":2} :172), ParticleHop (:182, :266), SiteExchange :324, SiteFlip :409, LocalFlip :24, MixSampler metropolis.py:538.
  The sampler does not return psi for ref states; it is recomputed with init_internal (metropolis.py:107-112).
- Samplers: Metropolis family above + ExactSampler sampler.py:81 + RandomSampler :166. No parallel tempering (grep "temper": 0 hits). "Every sample is an independent Markov chain, chains distributed over devices" (metropolis.py:31-33), shard_batch in jit_chunk_vmap -> multi-GPU capable.
  SpinExchange needs `Sites(..., Nparticles=(Nup,Ndn))` (common_samplers.py:104-106, sites/sites.py:25) = fixed Sz sector.
- Symmetry: Symmetry class symmetry/symmetry.py:111 (generator permutations, sector, character, Z2_inversion, perm_sign); Translation translation.py:8, TransND :90; Rotation common_symmetries.py:170, C4v :212, D6 :247,
  Z2Inversion :21, SpinInverse :42 (spin flip, eigval +-1), ParticleHole :71, Flip :147, LinearTransform :101, `@` product symmetry.py:445. Projection applied inside Variational: `symm.symmetrize(psi, s)` (variational.py:376-379).
  Sz sector via Sites Nparticles (above); the non-trivial point-group irreps via `repr` argument of C4v/D6.
  Cost note: symmetrised psi evaluates the model on all symmetry images (get_symm_spins) -> ref_forward vmapped over the group (variational.py:371-373); so low-rank updates are used per group element.
  ExactQNGD/basis uses QuSpin basis (symmetry.py:297-, utils/basis.py) - quspin not installed here.
- SR/MinSR: optimizer/sr.py: SR :20 (auto SR vs MinSR by shape), SPRING :62, MARCH :113, AdamSR :177, ER :234 (exact), TimeEvol :280. Base optimizer/qngd.py: QNGD :39, StochasticQNGD :219, ExactQNGD :269.
  Solvers optimizer/solver.py: auto_shift_eig :419 (A.shape[0]<A.shape[1] -> MinSR `minnorm_shift_eig` :315 else `lstsq_shift_eig` :370), both Cholesky via jax.scipy solve(assume_a="pos") (:360, :411)
  with diagonal shift eps = Tr(M)*rshift/sqrt(n)+ashift (rshift 1e-12 for f64, 1e-6 for f32), internal dtype default float64 even if model is float32 (mixed precision: dtype argument, :317, :421). Others: lsmr :208, minnorm_cg :131, lstsq_cg :56,
  minnorm_pinv_eig :520 / lstsq_pinv_eig :581 / auto_pinv_eig :636 (eigh pinv), block_pinv_eig :681, sgd_solver :745.
- Custom least-squares loss: yes, by design. QNGD solves  Obar * thetadot = ebar  with Obar centered Jacobian and ebar from any "gradient source" object (qngd.py:39-60; EnergyGrad gradient.py:15, OverlapGrad :125,
  used by Supervised supervised.py:12). A custom source needs `.ebar(state, samples)` (qngd.py:~260 get_Ebar). Even simpler: `StochasticQNGD.solve(Obar, Ebar, buffers)` / `solve_equation(Obar,Ebar)` (qngd.py:134-185)
  accept ARBITRARY Jacobian and residual arrays; `state.jacobian(spins)` (variational.py:625) gives per-sample dlogpsi/dtheta. Real/complex handling for non-holomorphic states stacks Re/Im (qngd.py:164-170).
  Caveat: Obar is the mean-centered Jacobian (qngd.py:243-256, `_Omat_center`), so a residual that is not of centered-log-derivative form needs the centering bypassed by calling solve_equation directly with your own matrix.
- f64: yes. `set_default_dtype(jnp.float64 / complex128)` (global_defs.py:11-28) turns on jax x64; docstring says default float32 and "doesn't alter the computation inside quantax.model" (:16-17). Low-precision/TF32 awareness: utils/precision.py.
- Licence: MIT (above).

## c. NetKet 3.23.0 (Apache License 2.0, netket-3.23.0.dist-info/licenses/LICENSE)
- Symmetric models: GCNN() models/equivariant.py:593 (-> GCNN_FFT :47, GCNN_Irrep :153, GCNN_Parity_FFT :272, GCNN_Parity_Irrep :425), DenseSymm nn/symmetric_linear.py:662, DenseEquivariant :742, RBMSymm models/rbm.py:205.
  No ViT (grep Vision/ViT: nothing in this version). Also Jastrow models/jastrow.py:22, MLP, autoregressive models, tensor_networks, deepset.
- Fermion models: Slater2nd models/slater.py:30, MultiSlater2nd :342. No Pfaffian / pair-product model in models/. Evaluation is a full `nkjax.logdet_cmplx` per call (slater.py:206): NO fast updates, no lrux.
- Samplers: MetropolisSampler/MetropolisLocal/MetropolisExchange (metropolis.py:573), MetropolisHamiltonian, MetropolisGaussian, MetropolisAdjustedLangevin, MetropolisFermionHop (sampler/__init__.py:22-33);
  rules: ExchangeRule, FermionHopRule (rules/fermion_2nd.py:9), GlobalSpinFlipRule, MultipleRules, TensorRule, HamiltonianRule (rules/__init__.py:15-31). Parallel tempering is in netket.sampler (NOT experimental) in this version:
  ParallelTemperingSampler parallel_tempering.py:133 (n_replicas default 32 :~141, total chains n_chains*n_replicas), ParallelTemperingExchange :561, Local, Hamiltonian. ExactSampler, ARDirectSampler, Numpy samplers (CPU).
  GPU: jax samplers run n_chains vectorised (metropolis.py:179-195 round chains to multiples of device count; sharding over devices). Number of chains is a free parameter; I did not measure scaling.
- SR variants: VMC_SR (netket/_src/driver/vmc_sr.py:117; exported driver/__init__.py), default solver Cholesky `netket.optimizer.solver.cholesky` (solvers.py:248), `cholesky_with_fallback` imported :11 (solvers.py:310), also pinv_smooth :39, svd :213, LU :280, solve :351.
  SR vs SRt (kernel trick = MinSR) selected inside VMC_SR; the SRt math: `_src/ngd/srt.py:22 _compute_srt_update(O_L, dv, diag_shift, solver_fn, mode, proj_reg, momentum, old_updates)` -> solves (O O^T + shift) x = dv, update = O^T x (:50-85).
  SPRING momentum supported (vmc_sr.py docstring ~:180-205). on_the_fly Jacobian contraction default (docstring ~:219). Old-style SR preconditioner optimizer/sr.py:62 with qgt/ (qgt_jacobian*, qgt_onthefly).
- Custom loss: `_compute_srt_update` is private (`_src`), but takes an arbitrary Jacobian O_L and arbitrary residual dv, so a custom Gauss-Newton/minSR step can call it directly. A public custom-loss driver exists only for fidelity:
  Infidelity_SR (_src/driver/infidelity_sr.py:43). VMC_SR itself builds the energy residual internally.
- FullSumState vqs/full_summ/state.py:64: sums over the whole Hilbert space; NO symmetry-reduced Hilbert space (grep "symm" in the file: none); netket.symmetry has Representation/translation/spin_flip representations (symmetry/__init__.py) used for models/operators, not for a reduced basis here.
  Max size: hilbert indexing limited to np.iinfo(int32).max = 2147483647 states (hilbert/index/base.py:20-34), practically memory-limited. lanczos_ed (exact.py:24, scipy eigsh) for ED.
- TDVP: netket/experimental/driver/tdvp.py:39 (real and "imag" propagation, `tdvp_common.py:108`), tdvp_schmitt.py.

## d. jVMC 1.5.8 (MIT, "Copyright (c) 2021 Markus Schmitt", jvmc-1.5.8.dist-info/licenses/LICENSE; imports OK on jax 0.8.2 although METADATA pins jax<=0.7.2; untested beyond import)
- Models (jVMC/nets): CpxRBM rbm.py:17, RBM :67, CpxRBM_Nospinflip :43, FFN ffn.py:13, CNN/CpxCNN cnn.py:18/86, CpxCNNSym :155, RNN1DGeneral rnn1d_general.py:58, RNN2DGeneral rnn2d_general.py:64, TwoNets two_nets_wrapper.py:8, SymNet sym_wrapper.py:22. No determinant/Pfaffian models.
- Samplers: MCSampler sampler.py:67 (numChains per device, mu in [0,2] importance sampling :129-130, MPI+local devices :80-81), proposers propose_spin_flip :15, _Z2 :30, _zeroMag :42 (Sz=0 exchange), ExactSampler :385. No exchange-on-lattice neighbours rule, no parallel tempering.
- TDVP util/tdvp.py:26 (SVD/eigh pseudo-inverse of S: jnp.linalg.eigh(S) :157, makeReal real/imag, imaginary-time via rhsPrefactor), MinSR util/minsr.py:16 (jnp.linalg.pinv(T,hermitian=True) of the sample kernel :61; `solve(eloc, gradients, holomorphic)` :53
  takes the vector `eloc`, but `gradients` must be a jVMC SampledObs). steppers util/stepper.py (Euler :6, Heun :45, AdaptiveHeun :98). ground_state_search util/util.py:160.
- Symmetry: LatticeSymmetry util/symmetries.py:9, orbits for 1D/2D square (translation, reflection, rotation, spin flip; get_orbit_2D_square :114), applied by SymNet averaging (sym_wrapper.py:7-20 avgFun_*). No Sz projection except zeroMag proposer.
- f64/complex128 fixed: global_defs.py:4-6 (tCpx=complex128, tReal=float64). GPU via jax; optional extra pins jax[cuda12]<=0.5.0.

## e. Grep: fixed.node|fixed_node|GFMC|green.?s function|diffusion monte|projector monte|lanczos|power method|imaginary.time (netket, quantax, lrux, jVMC; .py/.md/.txt)
19 hits total, none is a fixed-node / GFMC / projector / power-method implementation:
- netket/exact.py:24 `lanczos_ed` (ED via scipy eigsh), :33, :61, :65, :85 (same function); _src/operator/pyscf_api.py:94 (docstring using lanczos_ed).
- netket/experimental/driver/tdvp.py:79, :107; tdvp_common.py:108, :399; tdvp_schmitt.py:141, :171: TDVP with "imag" = imaginary-time SE (stochastic variational imaginary-time evolution, not projector MC); density-matrix imaginary time marked TODO.
- quantax/optimizer/qngd.py:68, :112, :296; sr.py:45, :92, :153, :210, :256: `imag_time` flag of SR/SPRING/MARCH/AdamSR/ER (variational imaginary-time step, SR = imag_time TDVP).
- jVMC/operator/povm.py:139, :286: POVM "imaginary time operator" (open-system POVM formalism, unrelated).
- lrux: 0 hits.
