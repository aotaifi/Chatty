"""Summarise vmc_continue_6x6 runs: dedicated-eval table, block-averaged train energies, linear trend.
usage: python analyze_vmc.py <results_dir containing <tag>/history.jsonl,eval.jsonl> [tags...]"""
import sys, json, os
import numpy as np
root = sys.argv[1]; tags = sys.argv[2:] or ['lr5e-3', 'lr1e-3']
E0 = -0.50380965; EVIT = -0.503654
def rd(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
for tag in tags:
    H = rd(f'{root}/{tag}/history.jsonl'); E = rd(f'{root}/{tag}/eval.jsonl')
    if not H: continue
    # steady-state cumulative training time: drop compile-dominated first step of each invocation
    cum = 0.0; ct = []
    for h in H:
        if not h.get('first_of_invocation'): cum += h['step_dt']
        ct.append(cum)
    print(f'== {tag}: steps={len(H)} train_gpu_h(all)={sum(h["step_dt"] for h in H)/3600:.3f} steady={cum/3600:.3f}')
    print('dedicated evals (E/N +- err, step, train GPU-h, eval samples)')
    for e in E:
        print(f'  step {e["step"]:4d}  GPUh {e["train_gpu_h"]:.2f}  E/N {e["E_site"]:.6f} +- {e["err_site"]:.6f}  n={e["nsamples"]} {"FINAL" if e.get("final") else ""}')
    en = np.array([h['E_site'] for h in H]); st = np.array([h['step'] for h in H]); ct = np.array(ct) / 3600
    print('block-mean of per-step training energies (blocks of 25 steps; naive SE over steps)')
    for a in range(0, len(en), 25):
        b = en[a:a+25]
        if len(b) >= 10: print(f'  steps {st[a]:4d}-{st[a+len(b)-1]:4d} GPUh {ct[a+len(b)-1]:.2f}  <E/N> {b.mean():.6f} +- {b.std(ddof=1)/np.sqrt(len(b)):.6f}')
    if len(en) >= 20:
        A = np.vstack([np.ones_like(ct), ct]).T; c, res, *_ = np.linalg.lstsq(A, en, rcond=None)
        r = en - A @ c; cov = r.var(ddof=2) * np.linalg.inv(A.T @ A)
        print(f'  linear trend in train energies: slope {c[1]:.2e} +- {np.sqrt(cov[1,1]):.1e} per GPU-h; E(0)={c[0]:.6f}')
