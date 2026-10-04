"""Score a guide state against the exact 6x6 ground state (results/ed_6x6 samples x ~ |psi0|^2).

For every chain prefix j = 0..K: true wrong-sign probability w_j = min(m, 1-m), m = fraction of samples
with s^(j)(x) != sgn psi0(x) (global sign free), binomial SE, and paired change w_K - w_{K-1}.
Amplitude check: std over x ~ psi0^2 of log a(x) - log|psi0(x)| for a_0 = |ViT| and the guide amplitude
(a smaller value = amplitude closer to exact).
Usage: python ll6_ed.py STATE.json TAG nsamples [chunk]
"""
import sys, json, time
import numpy as np
import ll6_core as C
import jax.numpy as jnp

state, tag, nsamp = sys.argv[1], sys.argv[2], int(sys.argv[3])
chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 512
st = json.load(open(state))
net = C.Net('vit_J2=0.50_N=6x6_k=0.mpack', dtype='float32')
amp_objs = {}
def amp_of(p):
    if p not in amp_objs: amp_objs[p] = C.Amp(net, net.flat0 if p == 'base' else jnp.asarray(np.load(p), net.DT), name=p)
    return amp_objs[p]
guide = C.Guide([amp_of(p) for p in st['chain_params']], st['Ts'], amp_of(st['amp_g']))
K = guide.K
X = []; P = []
for f in ('samples_psi0sq_6x6.npz', 'samples_extra_psi0sq_6x6.npz'):
    d = np.load(f); X.append(d['x'].astype(np.uint64)); P.append(d['psi'])
X = np.concatenate(X)[:nsamp]; P = np.concatenate(P)[:nsamp]
strue = np.sign(P)
assert np.all(C.marshall_vec(X[:1000]) == np.load('samples_psi0sq_6x6.npz')['marshall'][:1000])
t0 = time.time()
S = {j: np.zeros(len(X)) for j in range(K + 1)}
for i in range(0, len(X), chunk):
    Xc = X[i:i + chunk]
    lv, nb = C.build_levels(Xc, K)
    sl, _ = C.chain_signs(lv, nb, guide.amps, guide.Ts, K)
    p0 = np.searchsorted(lv[0], Xc)
    for j in range(K + 1):
        S[j][i:i + chunk] = C.restrict(lv, sl[K - j], K - j, 0)[p0] if K - j > 0 else sl[0][p0]
    if (i // chunk) % 20 == 0:
        C.log('ED', i + len(Xc), '/', len(X), 'evals', net.neval, 'sec', round(time.time() - t0, 1))
out = dict(state=st, tag=tag, n=int(len(X)), K=K, sec=time.time() - t0, evals=net.neval)
for j in range(K + 1):
    m = float(np.mean(S[j] != strue)); g = 1 if m <= 0.5 else -1; w = min(m, 1 - m)
    out[f'w_s{j}'] = dict(w=w, err=float(np.sqrt(w * (1 - w) / len(X))), nwrong=int(round(w * len(X))), global_sign=g)
    C.log('W', j, out[f'w_s{j}'])
for j in range(1, K + 1):
    wj = (S[j] != strue * out[f'w_s{j}']['global_sign']).astype(float)
    wp = (S[j - 1] != strue * out[f'w_s{j-1}']['global_sign']).astype(float)
    dd = wj - wp
    out[f'paired_w{j}_minus_w{j-1}'] = dict(mean=float(dd.mean()), err=float(dd.std(ddof=1) / np.sqrt(len(dd))),
                                           changed=int(np.sum(S[j] != S[j - 1])))
# amplitude closeness to |psi0| (on the first 100k samples)
na = min(len(X), 100000)
lp = np.log(np.abs(P[:na]))
for nm, p in (('a0_base', 'base'), ('a_guide', st['amp_g'])):
    la = amp_of(p)(X[:na]); dl = la - lp
    out[f'std_dlog_{nm}'] = float(np.std(dl))
C.log('AMP', out.get('std_dlog_a0_base'), out.get('std_dlog_a_guide'))
json.dump(out, open(f'ed_{tag}.json', 'w'), indent=1)
C.log('DONE', json.dumps({k: v for k, v in out.items() if k != 'state'}))
