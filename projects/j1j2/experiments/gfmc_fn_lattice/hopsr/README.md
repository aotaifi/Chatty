# hopsr: sampled hop-aware SR trainer (built 2026-10-26)

Objective: minimise E_1(theta) = E_VMC((1 + alpha (H - E)) v_theta), v = Re[w psi_P] (w fixed), alpha by Ritz in span{v, (H-E)v}.
Motivation (exact 4x4, l8_loop4x4.py stage hopa, /scratch/a/A.Otaifi/l8/loop4x4/hop_*/hopa.json): minimising E_1 closes 79% (RBM-only) /
69% (f-only) of the add-on-to-E0 gap of E_FN(a*h_a), vs 27% / 9.9% for 5 rounds of the storage loop.

## Code
- `hopsr_lib.py`: per-sample vt, h1, h2 (as vmc_moments.py: E_L at x and at every antiparallel neighbour), dvt = Re[u O],
  dh1 = Re[u (O E_L + dE_L)] (jacobian of the fast-update local energy). Gradient in the **zero-variance g-form**
  grad E_1 = 2 <dg~ (H - E_1) g~> / <g~^2> with the Ritz coefficients frozen (exact, since dE_1/dalpha = 0); the moment chain rule
  ('mom') is kept as a variant: in exact mode it equals the g-form to 3e-11, but sampled it is ~15x noisier (4x4, 4096 samples:
  relative jackknife error 10 vs 0.8). Jackknife over chain groups.
- `hopsr_train.py OUT SPEC.json`: SR (S = Re cov(O,O) of psi_P, F = grad/2, diagonal rescaling + shift, |d|_S cap), persistent chains,
  checkpoints/resume (state.npz), maxsec stop, objective E1 or E0 (control).
- `hopsr_check4x4.py REF.npz OUT.json`: deliverable 1 (below). REF = `l8/l8_loop4x4.py OUT gradref` (stage added 2026-10-26; also
  `evalfn` with FLAT=params.npz gives the exact E_1, E_FN(a*h_a) of any 4x4 parameter file).
- `pilot_hopsr.sbatch`, `submit_hopsr.sh` (Paderborn gpu_h100, watcher m13b_8x8/watch_gfmc8.sh, job_finish --wake none), `specs/`.

## Note on the 'alpha frozen' control
For E_1 the Ritz alpha minimises the Rayleigh quotient over the coefficient, so dE_1/dalpha = 0 and the alpha-frozen gradient equals the
true one (exact mode: 3e-11). It is a failing control only for E_FN (l8 hopb). The failing controls used here: 'noDE' (dE_L dropped
from d(Hv), i.e. the hop's own theta-dependence missed) and 'p0' (alpha = 0, the plain E_v gradient).

## Pre-registered checks (written before the sampled runs)
- Exact mode (all 12870 configurations weighted by |psi|^2): sampled-estimator algebra vs l8 exact gradient rel <= 1e-8; noDE, p0 > 1e-2.
- Sampled (>= 65k samples): per-component z over the class: chi2/dof <= 1.5 and |mean z| <= 3/sqrt(n); noDE and p0 chi2/dof > 3.
- 4x4 training (RBM-only class, from the E_v optimum flat_opt.npy, w = w30): exact E_FN(a*h_a)/N of the trained parameters within 20% of
  the gap of the exact optimum: <= -0.5285525325 + 0.2 x (-0.5282946487 - (-0.5285525325)) = -0.5285009557.

## Pre-registered decision rules for the lead's later production (not run here)
- **6x6**: hop-aware E_FN(a*h_a) - E0 must be < 0.7 x the add-on's 5.57e-6/site (results/rbmpp_6x6/runs/table_A.json, lanczos p=1
  dE_FN_site 5.5738e-6), i.e. < 3.90e-6/site -> go to 8x8; otherwise the 4x4 gain does not survive size -> stop.
- **8x8**: E_FN(g) of the hop-aware guide below -0.498946 (+-4.6e-6, M1.3b) by > 3 combined sigma -> GAIN.
