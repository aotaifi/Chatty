"""Readout of one shift-scan arm: python3 shiftscan_gate.py RUNDIR  (RUNDIR/train/train.json). Single chain per arm: errors of
block means from the scatter of 10-step sub-block means (one sweep between SR steps, so steps are serially correlated)."""
import json, sys, numpy as np
E0, E0err = -0.4987725, 8.3e-6        # rb8_prod params under 'simple', no training (rb8_pgprobe/a0, single chain, job 4191128)
h = json.load(open(sys.argv[1] + '/train/train.json'))['hist']
E = np.array([x['E'] for x in h]); dn = np.array([x['dnorm'] for x in h]); n = len(E)
def blk(x):
    m = len(x) // 10 * 10; s = x[:m].reshape(-1, 10).mean(1); return x.mean(), s.std(ddof=1) / np.sqrt(len(s))
print(f"arm shift {h[0]['shift']:.0e} lr {h[0]['lr']:g} steps {n} (spec {json.load(open(sys.argv[1]+'/train/spec.json'))['steps']}) t_step median {np.median([x['t_step'] for x in h[3:]]):.2f} s")
for a in range(0, n, 100):
    b = E[a:a + 100]; m, e = blk(b); print(f"  steps {a+1}-{a+len(b)}: {m:.7f} +- {e:.1e}")
m, e = blk(E[-100:]); g = E0 - m; ge = np.hypot(e, E0err)
p, c = np.polyfit(np.arange(n), E, 1, cov=True)
print(f"  last100 {m:.7f} +- {e:.1e}; gain vs simple warm start {g:.2e} +- {ge:.1e} (need >= 2e-5)")
print(f"  slope {p[0]:.2e} +- {np.sqrt(c[0,0]):.1e} /step; |d|_S last 20 steps {dn[-20:].mean():.4f}")
import re, os
vr = open(sys.argv[1] + '/VERDICT_RUN').read() if os.path.exists(sys.argv[1] + '/VERDICT_RUN') else ''
rc = re.findall(r'train_rc=(\d+)', vr); rc = rc[-1] if rc else 'missing'
want = json.load(open(sys.argv[1] + '/train/spec.json'))['steps']
if rc != '0' or n < want or not np.all(np.isfinite(E)):
    print(f"VERDICT: INCOMPLETE (train_rc={rc}, steps {n}/{want}); numbers above are partial, no 2e-5 decision"); sys.exit(0)
bm = [E[a:a + 100].mean() for a in range(0, n, 100)]
rise2 = any(bm[k + 1] > bm[k] and bm[k + 2] > bm[k + 1] for k in range(len(bm) - 2))
cap = bool(np.any(dn >= 0.3 * 0.999))
print(f"  stability: rising over two consecutive 100-step blocks {rise2}; |d|_S hit 0.3 cap {cap} (max {dn.max():.4f})")
if rise2 or cap:
    print(f"VERDICT: UNSTABLE ({'rising blocks' if rise2 else ''}{' ' if rise2 and cap else ''}{'step cap hit' if cap else ''}; gain {g:.2e} +- {ge:.1e})"); sys.exit(0)
print("VERDICT:", "GAIN >= 2e-5" if g >= 2e-5 else "NO GAIN >= 2e-5", f"(gain {g:.2e} +- {ge:.1e}, single chain, not quotable)")
