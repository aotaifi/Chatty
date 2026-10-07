# Stall diagnosis 6x6 (J1-J2, J2/J1 = 0.5, periodic): progress log

Question: why does the FN/Krylov loop stall at the ViT level on 6x6? Compression (ViT cannot represent a better
amplitude), information (FN information per iteration), or optimizer/noise (sampled updates)?
Code: `experiments/stall_6x6/`. ws1 run dir `~/chatty_stall6/` (code, logs); large data and runs on
`/project/theorie/a/A.Otaifi/chatty_stall6/` (csr, data, runs).
Budget: <= 15 GPU-h + 100 CPU-h. All energies per site; E0 = -0.50380965.

## Method (exact, no sampling in any energy)
- Everything lives in the fully symmetric sector (k=0, A1, flip+; D = 15,804,956 orbit reps). A symmetric state is a
  per-configuration log-amplitude la_r and a sign s_r on the reps; its energy, the Krylov sign step (energy-optimal
  threshold) and the lattice-FN ground state are computed exactly, as in the CPU exact loop
  (`krylov_sign_structure/experiments/closed_fn_krylov_sym6x6.py`).
- GPU implementation (JAX float64): H = diag(sqrt n) C diag(1/sqrt n) with integer codes C (8 c in int16), upper
  triangle only (H symmetric) -> 4.2 GB on the device, fits an RTX 2080 Ti. FN ground state by block-1 LOBPCG with
  the same diagonal preconditioner as the CPU code.
- A network state is the network evaluated on the reps (one evaluation per orbit). Supervised fits use exact data:
  reps drawn from p_r^beta (p = sector weight of the target), mapped to a uniformly random orbit element (random
  space-group element x spin flip, so the ViT is trained to be point-group/flip symmetric), plus K random one-hop
  neighbours with exact target values (canonical-rep lookup on the GPU).

## Log
- 11:20 Setup. CSR build (CPU, 16 cores, 4 min, 1.1 CPU-h): D = 15,804,956, nnz = 1.185e9, E0 from the stored ED
  vector -0.5038096538908782 (table -0.50380965389088), codes 8c exact (max rounding 9e-16), range -72..120.
- 11:24 All full A40s are allocated to multi-day jobs (cip-cl-nv01, kng-cl-nv01/02); the first GPU job (16856428)
  sat in the queue -> cancelled; code rewritten for an 11 GB RTX 2080 Ti (upper-triangular coded H, 4.2 GB).
- 11:27-11:33 **Home quota hit**: the 7.2 GB CSR in `~/chatty_stall6/csr` pushed `/home` over quota (writes failed
  for ~6 min, also for other jobs writing to home). Moved to `/project/theorie/a/A.Otaifi/chatty_stall6/` (306/475 GB);
  home writable again at 11:34. All large files of this task now live on /project.
- 11:37 Unit tests (ws1 login CPU, `test_st6_small.py`): synthetic 400-dim sector vs dense numpy: matvec 4e-15,
  FN ground state 6e-14 (vector 1e-6), Krylov threshold step = brute force over all cuts (same signs). Real 6x6
  table: canonical-rep roundtrip exact on 5000 random orbit images; lookups of configs and one-hop neighbours equal
  to the independent numpy `Psi0` class. `test_st6_fit_cpu.py`: all model variants build and train
  (ViT 154,980 params; ViT d96/h12 392,976 = 2.5x). Adam at lr 1e-4 moves the warm ViT by 1.6 rms in log a in 4
  steps -> warm-start fine-tuning needs lr ~1e-6..1e-5 (scan in the first GPU job).
- 11:45 GPU job 16856662 (2080 Ti): validation vs CPU exact loop + ViT baseline/oracles + lr scan. Queued (all
  2080 Ti / A40 busy).
- 12:00-12:45 GPU jobs 16856662, 16856836, 16856862, 16856977 (2080 Ti) each died after ~25 s with
  RESOURCE_EXHAUSTED in the first FN solve. Diagnosis from the logs: the sector itself works (E0 from one GPU
  matvec -0.5038096538908778 vs -0.5038096538908783; 0.45 s per matvec), but the JAX BFC allocator is capped at 75%
  of the card (8.5 GB) while 5.1 GB are static and the async-queued chunk kernels (+ (D,2) work arrays) need
  > 3.4 GB. Fixes (in order): donated accumulators, one chunk in flight, smaller chunks (2^23 entries), no D-size
  temporaries inside the matvec, XLA_PYTHON_CLIENT_MEM_FRACTION=0.94. Unit test re-passed; resubmitted 16857255.
  (Coordinator note at 12:48: job 16856662 failure acknowledged — same OOM.)
- 12:54 **Validation + baseline done** (job 16857274, 2080 Ti, 3 min; `base_validation.json`):
  - GPU vs CPU exact loop (sym6x6_plain iteration 1, J2=0 amplitude + Marshall): E_FN diff 3.6e-15, Krylov trial
    energy diff 7e-15 (same 15.80M r-groups). FN of the exact guide returns E0 (1 iteration). FN solve 23-34 s,
    Krylov step 7 s, ViT on all 15.8M reps 91 s.
  - ViT point-group/flip asymmetry: mean std of log|psi| over the 16 images 0.006 (amplitude error vs psi0: 0.049),
    so the ViT evaluated on reps is a faithful symmetric version. Binarised phase phi = -0.917.
  - Exact sector numbers (per site, dE = E - E0):

    | guide (amplitude, sign) | <H> | dE <H> | E_FN | dE FN | DMC (M=128) earlier |
    |---|---|---|---|---|---|
    | (\|ViT\|, ViT sign) | -0.5036004 | 2.09e-4 | -0.5036617 | 1.48e-4 | -0.503670(8) |
    | (\|ViT\|, exact sign) | -0.5035753 | 2.34e-4 | -0.5036622 | 1.47e-4 | -0.503682(21) |
    | (\|psi0\|, ViT sign) | -0.5036689 | 1.41e-4 | -0.5037157 | 0.94e-4 | -0.503731(15) |
    | (\|ViT\|, Krylov(ViT sign)) | -0.5036090 | 2.01e-4 | | | |

    ViT sign w_s = 1.30e-4 (as on ED samples); one Krylov step w_s = 3.2e-5. The DMC oracles of the it2 verdict
    are confirmed within ~1 sigma by exact FN. Note: the binarised-sign symmetric ViT has <H> 5e-5 above the
    complex ViT VMC energy -0.503654(21) (phase binarisation); the full-basis complex ViT is the VMC reference.
  - First lr-scan fit OOMed (train step needs ~5 GB next to the 4.2 GB H) -> H is now offloaded to host RAM
    during fits (re-uploaded in ~1 s). Scan resubmitted as 16857288.
- 13:11 **Test 1, supervised L2 + edge fits** (job 16857303, 2000 Adam steps, B=512, K=4, warm ViT; `fit` mode):
  the fit lowers the amplitude error vs psi0 (std dlog 0.049 -> 0.038 at lr 2e-6) but RAISES the energy:
  <H>(fit, exact sign) dE = 2.69e-4 (lr 2e-6), 2.89e-4 (1e-5), 11.5e-4 (5e-5) vs 2.34e-4 for the ViT amplitude.
  Reason: the energy at fixed exact sign is the quadratic form Q(d) = sum_unfrustrated w (d_y-d_x)^2 -
  sum_frustrated w (d_y-d_x)^2 >= 0 of the log-amplitude error d; VMC puts the ViT's errors into soft modes of Q,
  a supervised L2/|w|-edge loss moves them into stiff ones. A pointwise fit is the wrong projection.
- 13:35 Sampled exact-sign VMC (local energies from all 144 neighbours, exact signs, B=256, Adam 3e-6) also gets
  worse (2.34e-4 -> 2.5-2.7e-4 over 1500 steps): Adam noise at B=256 dominates. Cancelled.
- 13:39 **Unprojected exact loop from the ViT** (job 16857743, 7 min; `exact_loop_from_vit.json`): start
  (|ViT|, ViT sign). dE of E_FN per iteration 1.48e-4, 8.6e-5, 6.1e-5, 4.7e-5, 3.7e-5, ..., 6.6e-6 (it 15);
  contraction 0.58, 0.71, 0.76, 0.79, ... 0.88 (slowing, power-law-like). Guide <H> after the step: 1.07e-4 (it 1),
  4.1e-5 (it 4, below RBM+PP 4.5e-5), 6.2e-6 (it 15). w_s 1.9e-5 -> 4.7e-7. The ideal first iteration gains
  6.2e-5/site in E_FN and 1.0e-4 in <H> -- 4x the 1.5e-5 assumed in the it2 verdict.
- 13:50 New fitter `fit_table` (exact sector table): each outer iteration evaluates the network on all reps,
  computes the exact energy and the exact gradient weights c_r = v_r((Hv)_r - E v_r); inner Adam steps use B=4096
  reps drawn ~|c_r| (noise only from which reps are drawn, no local-energy noise, 16x cheaper per sample than
  neighbour-based VMC). Exact energy at every outer iteration -> keep best. Test-1 scan job 16857915.
- 13:53 `fit_table` diverges (dE 2.3e-4 -> 6e-3 in 40 Adam steps): with a stale gradient Adam keeps moving every
  parameter by ~lr in the same direction. Replaced by `fit_exact`: per outer iteration one SR candidate (minSR, N
  reps ~ v^2, exact local energies from the table) and one gradient candidate (exact weights c_r, 16k reps), each
  scored by the EXACT energy; accept the better one only if E decreases (step x1.5), else shrink /3. Every
  accepted step is an exact, noise-free improvement. Jobs: test 1 16857943 (A40, pending), 16858082 (2080 Ti,
  platform allocator); test 2 with this projection on the frozen FN energy 16858018 (A40, pending).
- 14:07 **Test 2 with L2 projection** (job 16858019, 2080 Ti): projected loop it1: E_FN(start) 1.48e-4,
  guide <H> after projection + Krylov 2.06e-4 (exact loop: 1.07e-4); it2: E_FN 1.39e-4, <H> 2.14e-4. The
  projection error vs phi_FN is only 0.012 (std of log a) but the energy gain is gone: **with perfect information
  and an L2 projection the loop stalls at the ViT level**, like the sampled loop.
- 14:44-15:21 **Test 1, exact-verified SR on the full ViT, exact signs** (job 16857943, A40, 37 min;
  `t1_fit_exact_vit.json`): 12 outer iterations, every accepted step an exact decrease. <H>(fit, exact sign)
  dE 2.343e-4 -> 2.225e-4 (-1.2e-5 total, ~1e-6 per step, still slowly decreasing); E_FN(fit, exact) 1.40e-4
  (ViT 1.47e-4); <H>(fit, Krylov sign) 1.93e-4, w_s 2.8e-5. Raw-gradient candidates (RMS step 1e-5..1e-8 per
  parameter) raise E by up to 1.5e-2 -- the energy landscape of the trained ViT is extremely stiff; SR steps of
  0.5-3% of the minSR step are the only ones accepted (N=6144 << P=155k).
- 15:00 Coordinator: add test 0 (symmetry projection of the ViT over D4 x flip). Job 16859153 (2080 Ti): 16 images
  of all reps (88 s each), exact energies of psi_P with own complex phase, binarised own sign, exact sign, one
  Krylov step, FN; and of the amplitude-only projections. Note our earlier asymmetry check measured mean std of
  log|psi| over the 16 images = 0.006 on reps drawn from psi0^2 (max 4.4 on rare configs); the coordinator's ss6
  prep reported rms 0.19-0.26 -- to be reconciled by the exact energies.
- Projected loop with FN-energy projection (A40, pending) reduced to 3 iterations x 4 exact-verified outers
  (the optimizer gains ~1e-6 per outer, so longer runs would not change the picture).
- 15:39 **Test 0 done** (job 16859153, 2080 Ti, 25 min; `sym_test0.json`). Exact sector energies (dE/site):
  | state | dE <H> | dE E_FN |
  |---|---|---|
  | ViT evaluated on reps (no projection), complex phase | 2.093e-4 | |
  | ViT, full basis (VMC -0.503654(21)) | 1.56e-4 | |
  | **psi_P = sum_{D4 x flip} psi_ViT(g x), own complex phase** | **1.319e-4** (E = -0.5036777) | |
  | (\|psi_P\|, own binarised sign) | 1.319e-4 | 1.018e-4 |
  | (\|psi_P\|, exact sign) | 1.561e-4 | 1.051e-4 |
  | (\|psi_P\|, one Krylov step) | 1.266e-4 (w_s 2.4e-5) | 0.943e-4 |
  | amplitude-only projection, arithmetic / geometric mean, ViT-rep sign | 1.44e-4 / 1.41e-4 | |
  The projection is worth 2.4e-5/site vs the full-basis ViT and 7.7e-5 vs the rep-evaluated ViT; after projection
  the phase is binary (sin^2 spread 1.6e-6, no cancellation). RBM+PP: 4.5e-5. So **missing symmetry explains
  ~22% (2.4e-5 of 1.11e-4) of the ViT -> RBM+PP gap**. Important consequence for tests 1-2: the ViT evaluated on
  canonical reps (my earlier baseline, 2.09e-4) is a much worse symmetric state than psi_P; the mean asymmetry 0.006
  of log|psi| on psi0^2 samples hid an energetically relevant asymmetry. Baselines are re-based on psi_P.
- 15:45 Cancelled the pending rep-ViT A40 jobs (16858018, 16858765). New, cheap design on top of the fixed
  psi_P table: amplitude = log|psi_P| (fixed, from the 16-image evaluation) + a trainable correction network evaluated
  on reps (CNN 2.5k / 19k params, RBM alpha=8). Exact-verified SR with N_sr = 8192 (N >= P for the small
  factors). Jobs: test 1/3 fits 16859399; exact loop + FN-energy-projected loop from psi_P 16859400.
- 15:42 **Exact loop from psi_P** (`exact_loop_from_projvit.json`): E_FN dE 1.02e-4, 6.5e-5, 4.9e-5, 3.8e-5, ...,
  8.8e-6 (it 12); contraction 0.64, 0.75, 0.78, ... 0.87. Guide <H> after one step 7.8e-5, after three 4.3e-5
  (below RBM+PP 4.5e-5). Ideal information gives 3.7e-5 (E_FN) / 5.4e-5 (<H>) in the first iteration from psi_P.
- 15:47-16:00 Correction networks evaluated on canonical reps are not smooth symmetric functions (the canonical rep
  picks an arbitrary D4/flip orientation) -> first CNN run gained nothing; fixed by D4 x flip averaging inside the
  factor (invariance verified to 4e-7). But 16x evaluation made every exact table ~75 s on the 2080 Ti and one
  outer iteration ~10 min -> too slow; CNN jobs cancelled (16859399/648, 16859400/652/653).
- 16:05 **Exactly optimisable symmetric family** (`st6_feat.py`): log a = log|psi_P| + sum_k c_k F_k, F_k = Ising
  cluster functions summed over the space group (all 2-body distances 9, 4-site clusters in a 3x3 window 19,
  6-site 16; K = 44). Linear method with all matrices computed exactly over the sector, exact line search; verified
  against dense BFGS on the synthetic sector (agrees to 1e-8) and feature invariance exact. Objectives: <H> at
  fixed sign (capacity) and the frozen FN energy (loop projection). Jobs 16859694 (capacity + Krylov/amplitude
  alternation), 16859695 (projected FN loop, 6 iterations).
- 16:06-16:12 2080 Ti memory: the (K, D) feature matrix (2.8 GB) plus XLA copies of it exhausted the card;
  features are now separate float16 rows (the family is defined by the stored rounded features, used exactly in
  float64; LM still matches dense BFGS to 1e-14 on the synthetic test). An XLA memory analysis showed the
  matvec kernel itself needs < 0.2 GB of temporaries: all earlier "2.3-2.6 GiB" failures were total-memory
  exhaustion, reported at the allocator's region-growth request.
- 16:21 **Cluster family, K = 44 (2-, 4-, 6-body Ising clusters, exactly symmetric), exact optimisation**:
  capacity at the psi_P own sign: dE 1.3194e-4 -> 1.3164e-4 (-3e-7): psi_P is already optimal against all
  short-range symmetric diagonal corrections. **Projected FN loop onto psi_P x exp(clusters)**: captures 0.7%
  (it 1) and 0.1% (it 2) of the exact frozen-FN gain; E_FN 1.02e-4 -> 0.94e-4 (it 2, mostly from the Krylov
  sign step), guide <H> 1.26e-4. Cancelled after 2 iterations (no movement).
  -> The FN amplitude update is not a short-range spin-correlation correction. First-order, phi_FN ~ (1 - tau
  (H_FN - E)) a, i.e. its log-direction is the guide's own FN local energy -- a one-hop function of the amplitude.
- 16:30 Test: one-hop features of the current guide (violating / allowed sums of H a(y)/a(x), their squares,
  product and logs; 7 features) -> capacity at fixed sign and a projected FN loop with these features rebuilt
  each iteration. Job 16859817.
- 16:30-16:41 **One-hop features** (`feat_hop.json`): capacity psi_P own sign 1.319e-4 -> 1.271e-4, Krylov sign
  1.266e-4 -> 1.207e-4 (FN 0.90e-4); projected loop with features rebuilt per iteration captures 7.7%, 2.9%,
  1.1%, 0.4%, 0.2%, 0.1% of the frozen-FN gain, E_FN 1.02e-4 -> 0.90e-4, then stuck. Cluster family alternation
  with Krylov signs (`feat_clusters44.json`): 1.264e-4 -> 1.259e-4 in 3 rounds.
- 16:40 **Accuracy needed** (`interp_fn_update.json`): FN update from psi_P has rms 0.0112 in log a; partial steps
  a exp(t delta) gain linearly (t = 1.25 still better); iid per-orbit log noise of rms sigma costs 0.33 sigma^2
  per site on phi_FN and on psi_P alike (2.6e-6 / 3.5e-5 / 3.1e-4 at 0.003 / 0.01 / 0.03). One ideal iteration
  (4.8e-5) is cancelled by sigma ~ 0.012 = the L2 projection error.
- 16:50 Verdict, decomposition and figure: `README.md`, `stall_6x6.png/.pdf` (`st6_summary.py`).
  Compute total ~4.9 GPU-h (2080 Ti 4.3, A40 0.63), 1.2 CPU-h CPU jobs.
