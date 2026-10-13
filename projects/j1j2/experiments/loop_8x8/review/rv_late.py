"""Review T6 (REVIEW_lanczos8x8.md, amendment 2): late-time direct VMC of psi_1.  Chains continue from the final states of a
T5 job (about 35 sweeps after the random start of the referee chains): 40 sweeps of plain Metropolis on |psi_P|^2 (cheap,
one amplitude per step; |psi_1|^2 differs from it only by f^2, with f ~ 1), recording the psi_P local energy every 4
sweeps, then 4 sweeps of delayed-acceptance Metropolis on |psi_1|^2 (rv_da.py), then the exact two-hop E_L1 of every chain
(same lines as l8_direct.py).
  python rv_late.py OUT SPEC.json    SPEC: phi, alpha, Eref, init_samples (samples.npz of a T5 job), psiP_sweeps,
                                           block, da_sweeps, seed, chunk, batch
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, HERE)
import l8_core as C
import rv_da
import jax, jax.numpy as jnp

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True); T0 = time.time()
g = C.Guide('vit_J2=0.50_N=8x8_k=0.mpack', sym=True, batch=SPEC.get('batch', 1024), phi=SPEC['phi'])
al = float(SPEC['alpha']); Er = float(SPEC['Eref']); rg = np.random.default_rng(SPEC['seed'])
z = np.load(SPEC['init_samples']); S = z['X'][z['sweep'] == z['sweep'].max()].copy(); nch = len(S)
run = C.make_sampler(g, 1.0)
X = jnp.asarray(C.bits_to_spins_np(S)); la, _ = jax.jit(g.lz_spins)(g.params, X); key = jax.random.PRNGKey(SPEC['seed'])


def stats(e):
    lo, hi = np.quantile(e, [0.01, 0.99])
    return dict(mean_site=float(e.mean() / C.N), median_site=float(np.median(e) / C.N), wins1_site=float(np.clip(e, lo, hi).mean() / C.N))


trace = []; t = time.time(); done = 0
d0 = C.onehop(g, S); trace.append(dict(psiP_sweeps=0, **stats(d0['ELa']))); C.log(trace[-1])
while done < SPEC['psiP_sweeps']:
    X, la, key, acc = run(g.params, X, la, key, int(SPEC['block'] * C.N)); done += SPEC['block']
    S = C.spins_to_bits_np(np.asarray(X)); d = C.onehop(g, S)
    trace.append(dict(psiP_sweeps=done, accept=int(acc) / (nch * SPEC['block'] * C.N), **stats(d['ELa'])))
    C.log(trace[-1], f'{time.time() - t:.0f}s')


def onehop_f(Sx):
    d = C.onehop(g, Sx)
    return d['la'], np.log(np.abs(1 + al * (d['ELa'] - Er))), d['ELa']


ch_ = rv_da.DAChain(C, S, lambda Sx: g.eval(Sx)[0], onehop_f, rg)
for k in range(SPEC['da_sweeps']):
    ch_.run(C.N); C.log('DA sweep', k + 1, f'{time.time() - t:.0f}s')
X = ch_.S.copy(); CH = np.arange(nch); R = nch
C.dump(os.path.join(OUT, 'trace.json'), dict(trace=trace, accept1=ch_.acc1 / ch_.prop, accept2=ch_.acc2 / max(ch_.acc1, 1)))
EL = np.full(R, np.nan); EL2 = np.full(R, np.nan); t = time.time(); ch = SPEC.get('chunk', 64)
for i in range(0, R, ch):
    Sx = X[i:i + ch]; n = len(Sx)
    NBs, V = C.neighbors(Sx); o, bb = np.nonzero(V); ys = NBs[o, bb]
    Yu = np.unique(np.concatenate([Sx, ys])); d = C.onehop(g, Yu)
    px = np.searchsorted(Yu, Sx); py = np.searchsorted(Yu, ys)
    h = 0.5 * C.JB[bb] * d['s'][px][o] * d['s'][py] * np.exp(np.clip(d['la'][py] - d['la'][px][o], -60, 60))
    EL[i:i + n] = d['ELa'][px]
    EL2[i:i + n] = d['D'][px] * d['ELa'][px] + np.bincount(o, weights=h * d['ELa'][py], minlength=n)
    if (i // ch) % 4 == 0:
        np.savez(os.path.join(OUT, 'late.npz'), X=X, chain=CH, EL=EL, EL2=EL2); C.log(f'twohop {i + n}/{R} {time.time() - t:.0f}s')
np.savez(os.path.join(OUT, 'late.npz'), X=X, chain=CH, EL=EL, EL2=EL2)
EL1 = ((1 - al * Er) * EL + al * EL2) / (1 + al * (EL - Er))
res = dict(spec=SPEC, R=R, E1_site=float(EL1.mean() / C.N), E1_se_site=float(EL1.std(ddof=1) / np.sqrt(R) / C.N),
           **{'E1_' + k: v for k, v in stats(EL1).items()}, trace=trace, sec=time.time() - T0)
C.dump(os.path.join(OUT, 'late.json'), res); C.log('DONE', json.dumps(res, default=float))
