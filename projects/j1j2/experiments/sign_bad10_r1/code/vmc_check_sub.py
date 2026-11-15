"""smoke control: bond-subsampled h2 (vmc10.sub_sample) against the exact two-hop h2 (vmc10.two_hop) on the same walkers.  usage: python vmc_check_sub.py NX NDRAW KB"""
import sys, time, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import vitlib as VL, vmc10 as VM
nx, nd, Kb = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); ev = VL.make_vit('f32h'); S = VL.shipped(); rng = np.random.default_rng(99); X = S[rng.choice(len(S), nx, replace=False)]
ex = [VM.two_hop(ev, x, 1024) for x in X]; h2e = np.array([d['h2'] for d in ex]); h1e = np.array([d['h1'] for d in ex])
t0 = time.time(); draws = []
for _ in range(nd):
    vt, h1, h2, ne = VM.sub_sample(ev, X, rng, Kb, 1024); draws.append(h2); assert np.allclose(h1, h1e, atol=1e-4)
draws = np.array(draws); sec = (time.time() - t0) / (nd * nx)
bias = draws.mean(0) - h2e; se = draws.std(0, ddof=1) / np.sqrt(nd)
print(json.dumps(dict(Kb=Kb, nx=nx, ndraw=nd, sec_per_sample_sub=sec, h2_exact=h2e.tolist(), h2_sub_mean=draws.mean(0).tolist(), bias_over_se=(bias / se).tolist(), sub_sd=draws.std(0, ddof=1).tolist(),
                 sd_h2_across_x=float(h2e.std(ddof=1)), rms_sub_sd_over_sd_across_x=float(np.sqrt((draws.var(0, ddof=1)).mean()) / h2e.std(ddof=1)), h1_exact_equal_sub=True), indent=1))
