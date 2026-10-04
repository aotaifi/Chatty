"""Variational energy of a guide (a_g, s^(K)) paired with the ViT's own energy on the same samples.

x ~ |a_g|^2 (bond-exchange Metropolis chains).  E_guide = mean E_H(x) (local energy of a_g s^(K), full edges).
E_ViT is estimated on the same x by reweighting to |psi_ViT|^2 (w = a_0^2/a_g^2), with the complex ViT local
energy (own phase).  Reported: Delta = <H>_guide - E_ViT (chain jackknife), and <H>_guide = E_ViT_ref + Delta with
E_ViT_ref = -0.5036542608124052(213e-7)/site (bound_check_final.json, 49k-sample VMC).
Usage: python ll6_vmcpair.py STATE.json TAG nchains nrounds [seed]
"""
import sys, json, time
import numpy as np
import ll6_core as C
import jax.numpy as jnp
state, tag, nch, nr = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 77
E_VIT_REF = (-0.5036542608124052, 2.131044877124918e-05)
st = json.load(open(state))
net = C.Net('vit_J2=0.50_N=6x6_k=0.mpack', dtype='float32')
amp_objs = {}
def amp_of(p):
    if p not in amp_objs: amp_objs[p] = C.Amp(net, net.flat0 if p == 'base' else jnp.asarray(np.load(p), net.DT))
    return amp_objs[p]
guide = C.Guide([amp_of(p) for p in st['chain_params']], st['Ts'], amp_of(st['amp_g']))
thg = net.flat0 if st['amp_g'] == 'base' else jnp.asarray(np.load(st['amp_g']), net.DT)
t0 = time.time()
ch = C.Chains(net, nch, seed); ch.advance(thg, 100)
X = np.concatenate([ch.advance(thg, 4) for _ in range(nr)])
d = C.local_chunked(guide, X, 128)
eg = d['eh']
# ViT own complex local energy
zx = net.logpsi_c(net.flat0, X)
NB, V = C.neighbors(X)
own, col = np.nonzero(V)
zy = net.logpsi_c(net.flat0, NB[own, col])
ev = C.diag_vec(V).astype(complex)
np.add.at(ev, own, 0.5 * C.JB[col] * np.exp(zy - zx[own]))
ev = ev.real
w = np.exp(2 * (zx.real - d['lax'])); w /= w.mean()
def jk(f):
    full = f(np.ones(nch, bool)); reps = []
    for c in range(nch):
        m = np.ones(nch, bool); m[c] = False; reps.append(f(m))
    reps = np.asarray(reps); return float(full), float(np.sqrt((nch - 1) / nch * np.sum((reps - reps.mean()) ** 2)))
EG = eg.reshape(-1, nch); EV = ev.reshape(-1, nch); W = w.reshape(-1, nch)
out = dict(state=st, tag=tag, n=len(X), nch=nch, accept=ch.acc / ch.props,
           H_guide=jk(lambda m: EG[:, m].mean() / 36),
           E_vit_rw=jk(lambda m: (W[:, m] * EV[:, m]).sum() / W[:, m].sum() / 36),
           delta=jk(lambda m: EG[:, m].mean() / 36 - (W[:, m] * EV[:, m]).sum() / W[:, m].sum() / 36),
           ess=float(w.sum() ** 2 / np.sum(w * w) / len(w)), evals=net.neval, sec=time.time() - t0)
out['H_guide_via_ref'] = [E_VIT_REF[0] + out['delta'][0], float(np.hypot(E_VIT_REF[1], out['delta'][1]))]
json.dump(out, open(f'vmcpair_{tag}.json', 'w'), indent=1)
C.log('VMCPAIR', json.dumps({k: v for k, v in out.items() if k != 'state'}))
