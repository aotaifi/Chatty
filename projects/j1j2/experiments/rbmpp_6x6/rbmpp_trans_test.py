"""Controls for the 'translations' projection (translations x spin flip, 2N images) and for L = 8.
  python rbmpp_trans_test.py
 1. psi_P(T sigma) = psi_P(sigma) for translations, spin-flip even, for L = 4, 6, 8; NOT invariant under C4 (so the
    option really drops the point group);
 2. fast-update local energy = brute-force (all neighbours re-evaluated) at L = 4, 6 and 8 (float64, rel 1e-10);
 3. 4x4: sum |psi|^2 E_loc / sum |psi|^2 over ALL 12870 configurations = <psi|H|psi>/<psi|psi> (independent sparse H);
 4. 4x4 Metropolis sampler estimate vs that exact value;
 5. a deliberately broken lattice (INV := IMG, wrong inverse images) must fail check 2.
"""
import itertools, sys
import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp
from rbmpp_lib import Lattice, init_params, make_fns, make_sampler, random_configs, to32
jax.config.update("jax_enable_x64", True)
ok = True


def check(name, cond, msg=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name, msg, flush=True); ok &= bool(cond)


def fast_vs_brute(lat, p, S):
    fns = make_fns(lat)
    ef = jax.vmap(fns['eloc'], in_axes=(None, 0))(p, S); eb = jax.vmap(fns['eloc_brute'], in_axes=(None, 0))(p, S)
    rel = float(jnp.max(jnp.abs(ef[0] - eb[0]) / jnp.abs(eb[0]))); dl = float(jnp.max(jnp.abs(ef[1] - eb[1])))
    return rel, dl, ef[0], eb[0]


for L in (4, 6, 8):
    lat = Lattice(L, proj='translations'); N = lat.N
    check(f'L={L} images = 2N', lat.nI == 2 * N, str(lat.nI))
    p = init_params(lat, seed=3, rbm_scale=0.3); fns = make_fns(lat)
    S = random_configs(N, 3, 1)
    lp = lambda s: fns['logpsi'](p, s)
    dev = 0.0
    for t in (1, 5, N - 3):
        T = np.asarray(lat.G[8 * t]); s = np.asarray(S[0])
        dev = max(dev, abs(complex(lp(jnp.asarray(s[T]))) - complex(lp(jnp.asarray(s)))))
    check(f'L={L} translation invariance', dev < 1e-10, f'{dev:.1e}')
    C4 = np.asarray(lat.G[2]); s = np.asarray(S[0])
    dc = abs(complex(lp(jnp.asarray(s[C4]))) - complex(lp(jnp.asarray(s))))
    check(f'L={L} NOT C4 invariant (point group dropped)', dc > 1e-3, f'{dc:.2e}')
    rel, dl, a, b = fast_vs_brute(lat, p, S if L == 8 else random_configs(N, 6, 2))
    check(f'L={L} fast = brute E_loc', rel < 1e-10 and dl < 1e-10, f'max rel dE {rel:.2e} max|dlogpsi| {dl:.2e}')
    if L == 8:
        for e1, e2 in zip(np.asarray(a), np.asarray(b)): print('   8x8 E_loc fast', e1, 'brute', e2)
        S8 = random_configs(N, 2, 77)
        r2 = fast_vs_brute(lat, p, S8)
        check('L=8 fast = brute, 2 more configs', r2[0] < 1e-10, f'{r2[0]:.2e}')
        # broken version must fail
        lb = Lattice(L, proj='translations'); lb.INV = lb.IMG
        rb = fast_vs_brute(lb, p, S)[0]
        check('L=8 broken lattice (INV=IMG) FAILS the check', rb > 1e-6, f'rel {rb:.2e}')

# 3, 4: 4x4 enumeration
L = 4; N = 16
confs = []
for c in itertools.combinations(range(N), N // 2):
    s = -np.ones(N, np.int8); s[list(c)] = 1; confs.append(s)
confs = np.array(confs); idx = {tuple(c): k for k, c in enumerate(confs)}
lat = Lattice(L, proj='translations')
rows, cols, vals = [], [], []
for k, s in enumerate(confs):
    d = 0.0
    for i, j, J in zip(lat.BI, lat.BJ, lat.JB):
        d += 0.25 * J * s[i] * s[j]
        if s[i] != s[j]:
            y = s.copy(); y[i], y[j] = s[j], s[i]; rows.append(idx[tuple(y)]); cols.append(k); vals.append(0.5 * J)
    rows.append(k); cols.append(k); vals.append(d)
H = sp.csr_matrix((vals, (rows, cols)), shape=(len(confs),) * 2)
p = init_params(lat, seed=7, rbm_scale=0.2); fns = make_fns(lat)
l = np.asarray(jax.jit(jax.vmap(fns['logpsi'], in_axes=(None, 0)))(p, jnp.asarray(confs))); psi = np.exp(l - l.real.max())
Eex = (np.vdot(psi, H @ psi) / np.vdot(psi, psi)).real
el, _ = jax.jit(jax.vmap(fns['eloc'], in_axes=(None, 0)))(p, jnp.asarray(confs))
w = np.abs(psi) ** 2; Ew = float(np.sum(w * np.asarray(el)).real / np.sum(w))
check('4x4 sum|psi|^2 E_loc = <H> (translations)', abs(Ew - Eex) < 1e-9 * N, f'{Ew / N:.10f} vs {Eex / N:.10f}')
sweep = make_sampler(lat, make_fns(lat, jnp.float32)); Sm = random_configs(N, 1024, 2); key = jax.random.PRNGKey(1); p32 = to32(p)
Sm, _ = sweep(p32, Sm, key, 200); Es = []
ev = jax.jit(jax.vmap(fns['eloc'], in_axes=(None, 0)))
for t in range(60):
    key, k = jax.random.split(key); Sm, acc = sweep(p32, Sm, k, N); Es.append(np.asarray(ev(p, Sm)[0]).real.mean())
Es = np.array(Es); m = Es.mean(); e = Es.std(ddof=1) / np.sqrt(len(Es))
check('4x4 sampler within 4 sigma of exact <H>', abs(m - Eex) < 4 * e + 1e-9, f'{m / N:.6f} +- {e / N:.1e} vs {Eex / N:.6f}')
print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)
