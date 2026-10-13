"""Deliverable C, E3 production (PREREG E3): paired energies of psi_1 and psi_2 from the mixture pi ∝ psi_P^2 (f^2 + g^2).

psi_1 = psi_P f, f = 1 + alpha a1;  psi_2 = psi_P g, g = 1 + b1 a1 + b2 a2  (a_k = ((H - E_ref)^k psi_P)/psi_P, dc_p2.py).
Sampler: delayed-acceptance Metropolis (rv_da.DAChain): stage 1 on psi_P^2 (one amplitude), stage 2 with the ratio of
(f^2 + g^2) (two hops of psi_P at the candidate).  Chains start from psi_1 samples.  On each kept sample: exact three-hop
a1, a2, a3 -> E_L[psi_1], E_L[psi_2] and bounded weights w1 = f^2/(f^2+g^2), w2 = g^2/(f^2+g^2) in [0, 1].
E_k = sum w_k E_Lk / sum w_k, var_k likewise; paired E_2 - E_1; SE by jackknife over chains (dc_p2_analyze.py).
  python dc_p2_prod.py OUT SPEC.json   SPEC: phi, alpha, Eref, b1, b2, init [npz with X], nchains, burn, nsamp, thin, seed, batch
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'loop_8x8')); sys.path.insert(0, os.path.join(HERE, '..', 'loop_8x8', 'review'))
import l8_core as C
import rv_da
import dc_fn as F
import dc_p2 as P

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True); T0 = time.time()
g = C.Guide('vit_J2=0.50_N=8x8_k=0.mpack', sym=True, batch=SPEC.get('batch', 1024), phi=SPEC['phi'])
guide = lambda S: (lambda la, ph: (la, g.sign(ph)))(*g.eval(S))
lat = F.Lattice(8)
al, Er, b1, b2 = (float(SPEC[k]) for k in ('alpha', 'Eref', 'b1', 'b2'))
rg = np.random.default_rng(SPEC['seed']); nch = SPEC['nchains']
X0 = np.unique(np.concatenate([np.load(p)['X'] for p in SPEC['init']]))
S0 = X0[rg.choice(len(X0), nch, replace=False)]


def twohop_a(S):
    """a1, a2 at S (psi_P on the two-hop balls), and log a."""
    S = np.asarray(S, np.uint64); n = len(S)
    NBs, V = C.neighbors(S); o, bb = np.nonzero(V); ys = NBs[o, bb]
    Yu = np.unique(np.concatenate([S, ys])); d = C.onehop(g, Yu)
    px = np.searchsorted(Yu, S); py = np.searchsorted(Yu, ys)
    h = 0.5 * C.JB[bb] * d['s'][px][o] * d['s'][py] * np.exp(np.clip(d['la'][py] - d['la'][px][o], -60, 60))
    a1 = d['ELa'] - Er
    a2 = (d['D'][px] - Er) * a1[px] + np.bincount(o, weights=h * a1[py], minlength=n)
    return d['la'][px], a1[px], a2


def onehop_f(S):
    la, a1, a2 = twohop_a(S)
    f = 1 + al * a1; gg = 1 + b1 * a1 + b2 * a2
    return la, 0.5 * np.log(f * f + gg * gg), a1


ch_ = rv_da.DAChain(C, S0, lambda S: g.eval(S)[0], onehop_f, rg)
nb = int(SPEC['burn']); ns = int(SPEC['nsamp']); th = int(SPEC['thin']); samples = []; stime = []; t = time.time()
for sw in range(1, nb + ns * th + 1):
    ch_.run(C.N)
    if sw > nb and (sw - nb) % th == 0:
        samples.append(ch_.S.copy()); stime.append(sw)
    C.log(f'sweep {sw} acc1 {ch_.acc1 / ch_.prop:.3f} acc2 {ch_.acc2 / max(ch_.acc1, 1):.3f} {time.time() - t:.0f}s')
X = np.stack(samples).reshape(-1); CH = np.tile(np.arange(nch), len(samples)); TT = np.repeat(np.array(stime), nch); R = len(X)
np.savez(os.path.join(OUT, 'samples.npz'), X=X, chain=CH, sweep=TT)
A = np.full((R, 3), np.nan); t = time.time()
for i in range(R):
    d = P.threehop(lat, guide, X[i], Er); A[i] = (d['a1'], d['a2'], d['a3'])
    if i % 16 == 0 or i == R - 1:
        np.savez(os.path.join(OUT, 'p2prod.npz'), X=X, chain=CH, sweep=TT, A=A, alpha=al, Eref=Er, b1=b1, b2=b2)
        C.log(f'threehop {i + 1}/{R} {time.time() - t:.0f}s')
ok = np.isfinite(A[:, 2]); a1, a2, a3 = A[ok].T
f = 1 + al * a1; gg = 1 + b1 * a1 + b2 * a2; m = f * f + gg * gg
EL1 = Er + (a1 + al * a2) / f; EL2 = Er + (a1 + b1 * a2 + b2 * a3) / gg
w1 = f * f / m; w2 = gg * gg / m
E1 = np.sum(w1 * EL1) / w1.sum(); E2 = np.sum(w2 * EL2) / w2.sum()
res = dict(spec=SPEC, R=int(ok.sum()), accept1=ch_.acc1 / ch_.prop, accept2=ch_.acc2 / max(ch_.acc1, 1),
           E1_site=E1 / C.N, E2_site=E2 / C.N, d21_site=(E2 - E1) / C.N,
           var1_site=float(np.sum(w1 * (EL1 - E1) ** 2) / w1.sum() / C.N), var2_site=float(np.sum(w2 * (EL2 - E2) ** 2) / w2.sum() / C.N),
           sec=time.time() - T0)
C.dump(os.path.join(OUT, 'p2prod.json'), res); C.log('DONE', json.dumps(res, default=float))
