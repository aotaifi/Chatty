"""Cost table for the L = 10, 12 extension of the 8x8 M1 measurement (H100 and A40 hours per item).  python sizing.py
Every input below is a measured number with its source; ASSUMED lines are assumptions the production chains will test.

MEASURED (pilots 2026-10-24/25, guide = short-trained pilot parameters, so only COST numbers carry over, not sigma_walker):
  H100 (Paderborn gpu_h100, express): L10 job 3860368, L12 jobs 3861243 (+ 3862046 chains 8192 probe): SR s/step, eval ms, VMC s/sample, GFMC s per walker-tau.
  A40  (cip-cl-nv01, full A40 48 GB, jax 0.8.2): L10 jobs 17485251 (+17486801 for VMC/GFMC), L12 jobs 17485252 (SR 2048 only, 8192 OOM at chunk 64), 17486881.
  8x8 references (production, Paderborn 3831844 / 3832999-3833007): sigma_walker p0 4.9e-3, p1 2.1e-3 per site; tau_int 0.7 / 1.0-1.3 blocks; VMC E1 pooled error
  1.7e-5 from 163840 samples; hops per walker-tau 0.42 N; evals per hop (p1) 2.27 N; GFMC ms/eval 1.24-1.31 x the timing8 ms/eval (batching overhead, 8x8).
"""
import json, math

N = {8: 64, 10: 100, 12: 144}
# --- measured (H100)
H = {10: dict(sr_ms_chain=3.69, eval_ms=0.61, vmc_s_sample=143.0 / 1024, g0_pilot=0.0262, g1_pilot=4.2, hops_pilot0=24.5),
     12: dict(sr_ms_chain=12.8, eval_ms=1.70, vmc_s_sample=35.7 / 64, g0_pilot=0.117, g1_pilot=20.1, hops_pilot0=32.4)}
# --- measured A40 / H100 time ratios (same code, same chunk class); L12 non-SR ratios filled from job 17486881 if available
# A40 / H100 time per same work: SR 34.93/7.57 (L10, 2048 chains), 189.0/26.25 and 367.7/52.3 (L12); eval ms 4.66/0.61 (L10), 11.9/1.70 (L12);
# VMC 4.78/0.62 (L10), 12.1/1.77 (L12); GFMC per hop p0 10.7/1.07 ms (L10), 26.9/3.6 (L12); p1 per walker-tau 33.3/4.2 s (L10), 134.7/20.1 s (L12)
R_A40 = {10: dict(sr=4.6, eval=7.6, V=7.6, A=10.0, B=7.7), 12: dict(sr=7.1, eval=7.0, V=6.8, A=7.5, B=6.7)}
# --- assumptions
SIG0_8, SIG1_8 = 4.9e-3, 2.1e-3         # 8x8 production sigma_walker per site (p0, p1); ASSUMED to scale as 8/L (constant Var(E_loc)/N of the guide)
TAU0, TAU1 = 0.75, 1.3                   # tau_int in blocks (8x8)
E1_SD8 = 1.7e-5 * math.sqrt(163840)      # per-sample sd of the VMC E1 estimator at 8x8
STEPS = 4500; CHAINS = 8192              # the 8x8 recipe (README compress_6x6 M1.3a: 4500 steps, 8192 chains, lr 0.05 -> 0.005)


def cost(L, tgt_vmc=1e-5, tgt0=5e-6, tgt1=1e-5, M0=1024, M1=64, burn0=100, burn_tau1=15.0, dtau=0.1):
    h = H[L]; n = N[L]; out = {}
    gm0 = 1.24; gm1 = 1.31
    s0 = 0.42 * n * h['eval_ms'] * gm0 * 1e-3                        # s per walker-tau, p0 (1 eval per hop)
    s1 = 0.42 * n * (2.27 * n) * h['eval_ms'] * gm1 * 1e-3           # p1
    out['s_wt_p0'] = s0; out['s_wt_p1'] = s1
    sr = {c: h['sr_ms_chain'] * c * 1e-3 for c in (2048, 4096, 8192)}
    out['sr_s_step'] = sr
    out['train_h'] = {c: STEPS * sr[c] / 3600 for c in sr}
    nsamp = 4e5                                                      # evaluation: 2 seeds x 4e5 samples (rbmpp_eval), eval ms x 1.3 for the sampler
    out['eval_h'] = 2 * nsamp * h['eval_ms'] * 1.3e-3 / 3600
    sd1 = E1_SD8 * 8.0 / L
    nv = (sd1 / tgt_vmc) ** 2
    out['vmc_samples'] = nv; out['V_h'] = nv * h['vmc_s_sample'] / 3600 + 0.15
    s0w = SIG0_8 * 8.0 / L; s1w = SIG1_8 * 8.0 / L
    ec0 = tgt0 * math.sqrt(2); ec1 = tgt1 * math.sqrt(2)             # per chain, two chains
    nwb0 = 2 * TAU0 * s0w ** 2 / ec0 ** 2; nwb1 = 2 * TAU1 * s1w ** 2 / ec1 ** 2
    c0 = (nwb0 * dtau + burn0 * M0 * dtau) * s0 / 3600               # walker-tau x s/wt (burn = burn0 blocks of M0 walkers)
    c1 = (nwb1 * dtau + burn_tau1 * M1) * s1 / 3600
    out['A_h'] = 2 * c0; out['B_h'] = 2 * c1; out['A_blocks'] = nwb0 / M0; out['B_blocks'] = nwb1 / M1
    out['B_sample_h_only'] = 2 * nwb1 * dtau * s1 / 3600; out['B_burn_h'] = 2 * burn_tau1 * M1 * s1 / 3600
    return out


if __name__ == '__main__':
    for L in (10, 12):
        o = cost(L)
        r = R_A40[L]
        print(f'  A40-h: train(8192) {o["train_h"][8192]*r["sr"]:.0f} (4096: {o["train_h"][4096]*r["sr"]:.0f}), eval {o["eval_h"]*r["eval"]:.1f}, V {o["V_h"]*r["V"]:.0f}, A {o["A_h"]*r["A"]:.1f}, B {o["B_h"]*r["B"]:.0f}')
        print(f'L={L}: s/wt p0 {o["s_wt_p0"]:.4f} p1 {o["s_wt_p1"]:.1f}; SR s/step {o["sr_s_step"]}; train h {o["train_h"]}; eval h {o["eval_h"]:.2f}; V h {o["V_h"]:.1f} '
              f'({o["vmc_samples"]:.2e} samples); A h {o["A_h"]:.2f} ({o["A_blocks"]:.0f} blocks M1024); B h {o["B_h"]:.1f} (sample {o["B_sample_h_only"]:.1f} + burn {o["B_burn_h"]:.1f}; {o["B_blocks"]:.0f} blocks M64)')
    print('sensitivity (H100-h): V at 2e-5, p1 at 2e-5, p1 with M=32 / burn 10 tau, sigma_w x1.4 (Var/N x2):')
    for L in (10, 12):
        a = cost(L, tgt_vmc=2e-5); b = cost(L, tgt1=2e-5); c = cost(L, M1=32, burn_tau1=10.0)
        print(f'  L={L}: V {a["V_h"]:.1f}; B(2e-5) {b["B_h"]:.1f}; B(M32,burn10) {c["B_h"]:.1f}; B sample part x2 -> {2*cost(L)["B_sample_h_only"]+cost(L)["B_burn_h"]:.1f}')
