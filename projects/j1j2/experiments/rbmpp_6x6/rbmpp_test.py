"""Correctness tests of rbmpp_lib (CPU is enough): python rbmpp_test.py [stall_6x6 dir]
 1. space group of the 6x6 lattice = st6_sector.space_group() (same perms, same order);
 2. fast-update local energy = brute-force local energy (4x4 and 6x6, RBM on, projections simple / full / none);
 3. sign convention: symmetric f, no RBM, no projection -> exact singlet (<S^2> = 0) on 4x4 by enumeration;
 4. 4x4 exact enumeration: sum |psi|^2 E_loc / sum |psi|^2 = <psi|H|psi>/<psi|psi> with an independent sparse H;
    ED ground-state energy check;
 5. sampler: Metropolis estimate of <H> on 4x4 agrees with the exact value within errors;
 6. projected state is exactly symmetric (psi_P(g x) = psi_P(x)) for 'full'.
"""
import itertools, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import jax
import jax.numpy as jnp
from rbmpp_lib import Lattice, init_params, make_fns, make_sampler, random_configs, to32
jax.config.update("jax_enable_x64", True)
ok = True


def check(name, cond, msg=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name, msg, flush=True)
    ok &= bool(cond)


# 1
if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])
    import st6_sector as SS
    check('space group 6x6', np.array_equal(Lattice(6).G, SS.space_group()))
    check('bonds 6x6', np.array_equal(Lattice(6).BI, SS.BI.astype(np.int32)) and np.array_equal(Lattice(6).BJ, SS.BJ.astype(np.int32)))

# 2
for L in (4, 6):
    for proj in ('none', 'simple', 'full'):
        lat = Lattice(L, proj=proj)
        p = init_params(lat, seed=3, rbm_scale=0.3)
        fns = make_fns(lat)
        S = random_configs(lat.N, 6, 1)
        ef = jax.vmap(fns['eloc'], in_axes=(None, 0))(p, S)
        eb = jax.vmap(fns['eloc_brute'], in_axes=(None, 0))(p, S)
        d = float(jnp.max(jnp.abs(ef[0] - eb[0]))); dl = float(jnp.max(jnp.abs(ef[1] - eb[1])))
        check(f'fast = brute E_loc L={L} proj={proj}', d < 1e-8 and dl < 1e-10, f'max|dE| {d:.2e} max|dlogpsi| {dl:.2e}')

# 3, 4: 4x4 enumeration
L = 4; N = 16
confs = []
for c in itertools.combinations(range(N), N // 2):
    s = -np.ones(N, np.int8); s[list(c)] = 1; confs.append(s)
confs = np.array(confs); idx = {tuple(c): k for k, c in enumerate(confs)}
latn = Lattice(L, proj='none')


def sparse_H(lat):
    rows, cols, vals = [], [], []
    for k, s in enumerate(confs):
        d = 0.0
        for i, j, J in zip(lat.BI, lat.BJ, lat.JB):
            d += 0.25 * J * s[i] * s[j]
            if s[i] != s[j]:
                y = s.copy(); y[i], y[j] = s[j], s[i]
                rows.append(idx[tuple(y)]); cols.append(k); vals.append(0.5 * J)
        rows.append(k); cols.append(k); vals.append(d)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(confs),) * 2)


H = sparse_H(latn)
E0 = spl.eigsh(H, k=1, which='SA')[0][0]
check('4x4 ED E0/N', abs(E0 / N - (-0.52862)) < 2e-5, f'{E0 / N:.8f}')


def psi_vec(fns, p):
    l = np.asarray(jax.jit(jax.vmap(fns['logpsi'], in_axes=(None, 0)))(p, jnp.asarray(confs)))
    return np.exp(l - np.max(l.real))


pn = init_params(latn, seed=5, rbm=False)
fs = 0.5 * (pn['f'] + pn['f'].T); pn['f'] = fs
psi = psi_vec(make_fns(latn, use_rbm=False), pn)
S2 = 0.75 * N * np.vdot(psi, psi)
for i in range(N):
    for j in range(N):
        if i == j: continue
        perm = confs.copy(); perm[:, [i, j]] = perm[:, [j, i]]
        pidx = np.array([idx[tuple(c)] for c in perm])
        S2 += 0.5 * np.vdot(psi, psi[pidx]) - 0.25 * np.vdot(psi, psi)
S2 = (S2 / np.vdot(psi, psi)).real
check('symmetric-f PP is a singlet', abs(S2) < 1e-9, f'<S^2> = {S2:.2e}')

for proj in ('simple', 'full'):
    lat = Lattice(L, proj=proj)
    p = init_params(lat, seed=7, rbm_scale=0.2)
    fns = make_fns(lat)
    psi = psi_vec(fns, p)
    Eex = (np.vdot(psi, H @ psi) / np.vdot(psi, psi)).real
    el, lp = jax.jit(jax.vmap(fns['eloc'], in_axes=(None, 0)))(p, jnp.asarray(confs))
    w = np.abs(psi) ** 2
    Ew = float(np.sum(w * np.asarray(el)).real / np.sum(w))
    check(f'4x4 sum|psi|^2 E_loc = <H> ({proj})', abs(Ew - Eex) < 1e-9 * N, f'{Ew / N:.10f} vs {Eex / N:.10f}')
    if proj == 'full':
        G = lat.G
        dev = 0.0
        for g in range(0, G.shape[0], 7):
            pidx = np.array([idx[tuple(c)] for c in confs[:, G[g]]])
            dev = max(dev, float(np.max(np.abs(psi[pidx] - psi)) / np.max(np.abs(psi))))
        fl = np.array([idx[tuple(-c)] for c in confs])
        dev = max(dev, float(np.max(np.abs(psi[fl] - psi)) / np.max(np.abs(psi))))
        check('full projection is symmetric', dev < 1e-10, f'{dev:.1e}')
    # 5. sampler
    sweep = make_sampler(lat, make_fns(lat, jnp.float32))
    Sm = random_configs(N, 512, 2); key = jax.random.PRNGKey(1)
    p32 = to32(p)
    Sm, _ = sweep(p32, Sm, key, 200)
    Es = []
    for t in range(40):
        key, k = jax.random.split(key)
        Sm, acc = sweep(p32, Sm, k, N)
        Es.append(np.asarray(jax.vmap(fns['eloc'], in_axes=(None, 0))(p, Sm)[0]).real.mean())
    Es = np.array(Es); m = Es.mean(); e = Es.std() / np.sqrt(len(Es))
    check(f'4x4 Metropolis <H> ({proj})', abs(m - Eex) < 4 * e, f'{m / N:.6f} +- {e / N:.1e} vs {Eex / N:.6f}, acc {float(acc):.2f}')

print('ALL PASS' if ok else 'SOME FAILED')
