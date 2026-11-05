"""decades8_f0.py: 0.999 / 0.9999 quantiles of SIGNED f_J and of |f_J| from decades8 beta0 draws (i.i.d. uniform Sz=0 configs), pooled, per half
(odd / even files) and per file, with a bootstrap standard error.  usage: python3 decades8_f0.py OUT.json files..."""
import sys, json
import numpy as np
out = sys.argv[1]; files = sorted(sys.argv[2:])
fs = [np.load(f)['fx'] for f in files]
def q(f):
    f = f[np.isfinite(f)]; return dict(n=int(f.size), signed_q999=float(np.quantile(f, 0.999)), signed_q9999=float(np.quantile(f, 0.9999)),
                                       abs_q999=float(np.quantile(np.abs(f), 0.999)), abs_q9999=float(np.quantile(np.abs(f), 0.9999)), max=float(f.max()), min=float(f.min()))
allf = np.concatenate(fs); rng = np.random.default_rng(0); bs = [q(rng.choice(allf, allf.size)) for _ in range(400)]
res = dict(files=files, pooled=q(allf), n_nonfinite=int((~np.isfinite(allf)).sum()), half_A=q(np.concatenate(fs[0::2])), half_B=q(np.concatenate(fs[1::2])),
           bootstrap_se={k: float(np.std([b[k] for b in bs])) for k in ('signed_q999', 'signed_q9999', 'abs_q999', 'abs_q9999')}, per_file_signed_q9999=[q(f)['signed_q9999'] for f in fs])
json.dump(res, open(out, 'w'), indent=1)
p = res['pooled']; se = res['bootstrap_se']
print(f"VERDICT f0: signed q.9999 {p['signed_q9999']:.4f} +- {se['signed_q9999']:.4f} (halves {res['half_A']['signed_q9999']:.4f} / {res['half_B']['signed_q9999']:.4f}); "
      f"|f| q.9999 {p['abs_q9999']:.4f} +- {se['abs_q9999']:.4f}; signed q.999 {p['signed_q999']:.4f}; n {p['n']}")
