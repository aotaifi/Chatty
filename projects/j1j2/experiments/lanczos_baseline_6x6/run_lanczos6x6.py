"""Exact Lanczos vs Krylov-sign baseline on 6x6 J1-J2 (J2/J1 = 0.5), exact symmetric sector (k=0, A1, flip+), one GPU.

States (all energies exact in the sector, per site relative to E0 = -0.50380965):
  P    psi_P  = D4 x flip projected ViT, own (binarised) sign         [sym_tables.npz]
  KP   (|psi_P|, one exact Krylov sign step)
  V    ViT evaluated on the canonical reps, own binarised sign        [vit_reps_tables.npz]  (stall-work baseline)
  KV   (|V|, one exact Krylov sign step)
For each state X: Lanczos p = 1, 2 from X (E, variance, alpha, variance extrapolations), FN energy of X,
Lanczos-then-Krylov (|psi_1| with sign step), FN energy of the p=1 Lanczos state used as a guide.
Usage: python run_lanczos6x6.py OUT
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stall_6x6'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
import jax.numpy as jnp
import st6_sector as SS
from lanczos_lib import lanczos_ritz, extrapolate

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
CSR = os.environ.get('ST6_CSR', '/project/theorie/a/A.Otaifi/chatty_stall6/csr')
TABLE = os.environ.get('ST6_TABLE', '/home/a/A.Otaifi/chatty_stall6/data/psi0_6x6_table.npz')
DATA = os.environ.get('ST6_DATA', '/project/theorie/a/A.Otaifi/chatty_stall6/data')
log = SS.log
N = 36
sec = SS.Sector(CSR, TABLE)
E0 = sec.E0
log(f'E0 = {E0/N:.10f} per site')
RES = {}


def dump():
    json.dump(RES, open(os.path.join(OUT, 'lanczos_baseline_6x6.json'), 'w'), indent=1, default=float)


def site(E):
    return dict(E_site=E / N, dE_site=(E - E0) / N)


def vec_to_la_s(v):
    la = jnp.log(jnp.maximum(jnp.abs(v) / sec.sqrt_n, 1e-300))
    s = jnp.where(v >= 0, 1.0, -1.0).astype(jnp.float32)
    return la, s


def fn_of(v, tag):
    la, s = vec_to_la_s(v)
    _, _, info = sec.fn_solve(la, s)
    log(f'  FN[{tag}] dE {info["dE_FN_site"]:.4e}  ({info["lobpcg_it"]} it, {info["sec"]:.0f}s)')
    return dict(dE_FN_site=info['dE_FN_site'], E_FN_site=info['E_FN_site'], neg_weight=info['neg_weight'])


def krylov_of(v, tag):
    amp = jnp.abs(v) / jnp.linalg.norm(v)
    s = jnp.where(v >= 0, 1.0, -1.0).astype(jnp.float32)
    if float(s[0]) < 0: s = -s
    sn, Eb, info = sec.krylov(amp, s)
    vk = amp * sn
    E = sec.energy(vk)
    log(f'  Krylov[{tag}] dE {(E-E0)/N:.4e}  n_flip {info["n_flip_orbits"]}')
    return vk, dict(**site(E), n_flip_orbits=info['n_flip_orbits'])


def lanczos_of(v, tag):
    nm0 = sec.nmv
    r = lanczos_ritz(lambda x: sec.Hm(x), v, 2)
    out = dict(matvecs=sec.nmv - nm0)
    for d in r:
        out[f'p{d["p"]}'] = dict(**site(d['E']), var=d['var'], var_per_site=d['var'] / N,
                                 **({'alpha': d['alpha']} if 'alpha' in d else {}))
    E = [d['E'] for d in r]; var = [d['var'] for d in r]
    out['extrap_p01'] = site(extrapolate(E[:2], var[:2]))
    out['extrap_p12'] = site(extrapolate(E[1:], var[1:]))
    out['extrap_p012'] = site(extrapolate(E, var))
    log(f'  Lanczos[{tag}] dE p0 {out["p0"]["dE_site"]:.4e} p1 {out["p1"]["dE_site"]:.4e} p2 {out["p2"]["dE_site"]:.4e} | '
        f'var/N {out["p0"]["var_per_site"]:.3e} {out["p1"]["var_per_site"]:.3e} {out["p2"]["var_per_site"]:.3e} | alpha {out["p1"]["alpha"]:.5f} | '
        f'extrap01 {out["extrap_p01"]["dE_site"]:.4e} extrap012 {out["extrap_p012"]["dE_site"]:.4e}')
    return out, r[1]['psi'], r[2]['psi']


# --- starting states
zP = np.load(os.path.join(DATA, 'sym_tables.npz'))
lP = jnp.asarray(zP['lP']); sP = jnp.asarray(zP['sP'].astype(np.float32))
zV = np.load(os.path.join(DATA, 'vit_reps_tables.npz'))
lV = jnp.asarray(zV['la']); sV = jnp.asarray(zV['s'].astype(np.float32))
starts = {'P': sec.vec(lP, sP), 'V': sec.vec(lV, sV)}
if float(sec.v0 @ starts['P']) < 0: pass   # overall sign irrelevant

# sanity: reproduce the stall-work numbers
for k, v in starts.items():
    E = sec.energy(v); log(f'start {k}: dE <H> = {(E-E0)/N:.5e}')
    RES[f'{k}_check'] = site(E)
dump()

for k in ('P', 'V'):
    t0 = time.time()
    log(f'===== state {k}')
    v = starts[k]
    R = dict()
    R['start'] = site(sec.energy(v))
    R['FN'] = fn_of(v, k)
    vk, R['krylov'] = krylov_of(v, k)
    R['krylov']['FN'] = fn_of(vk, 'K' + k)
    # Lanczos from the start state
    R['lanczos'], psi1, psi2 = lanczos_of(v, k)
    R['lanczos']['FN_p1_guide'] = fn_of(psi1, f'{k} p1 guide')
    # Lanczos then Krylov sign step on |psi1|
    _, R['lanczos_then_krylov'] = krylov_of(psi1, f'{k} p1 then K')
    del psi1, psi2
    # Krylov then Lanczos
    R['krylov_then_lanczos'], psi1k, psi2k = lanczos_of(vk, 'K' + k)
    R['krylov_then_lanczos']['FN_p1_guide'] = fn_of(psi1k, f'K{k} p1 guide')
    del psi1k, psi2k, vk
    R['sec'] = time.time() - t0
    RES[k] = R
    dump()
    log(f'===== state {k} done {R["sec"]:.0f}s')
log('DONE total matvecs', sec.nmv)
RES['total_sec'] = None
dump()
