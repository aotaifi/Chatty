"""guides6.py (paired_fn): the exact-6x6 reference guides and pairs of ASTRA_paired.md section 4 "Actual 6x6 reference pairs".

Guide J(B, S) = S B exp(c fJ[B; S]) as q0r.py (fJ = g2lib.fJ_of, verbatim copy of q0r.py's, bad_mode 'zero', with the stored bare energy E'
of the source run); bare guides are c = 0; FNI guides use the frozen sign_fni/la_a.npy (= lB0 + 4.7 fJ[B0; T0]) with the parent sign
T0_a and the hybrid / random_hybrid_size flips (reconstructed exactly as sign_fni/step3_solve.py).  Amplitude: providers.floor_amplitude
(normalise sum n a^2 = 1, floor 1e-15: bond_split.guide, the exact solver's order).
Saved per guide under GUIDE_DIR: NAME.la.npy (float64, log amplitude before normalisation/floor), NAME.sig.npy (int8), NAME.json.
Exact references: F = E_FN/site - E0/site from the saved deterministic solves (paths in each spec); Delta = F_B - F_A (B - A, per site).
"""
import os, sys, json, time, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import providers as P

GUIDE_DIR = '/scratch/a/A.Otaifi/ansatz_runs/paired_fn/guides'
G2CODE = '/scratch/a/A.Otaifi/ansatz_runs/sign_g2/code'
Q0R = '/scratch/a/A.Otaifi/ansatz_runs/sign_q0r'; FNI = '/scratch/a/A.Otaifi/ansatz_runs/sign_fni'

# name: sign, base, E' (bare E_FN[S B] of the source run), c, F (stored E_FN/site - E0/site), Evar (stored E_guide_var, total) or None, source
GUIDES = {
    'S0B0_c50': dict(sign='B2:sP', base='B2:lP', Ep=-18.132765518290974, c=5.0, F=6.402834480863835e-05, Evar=None, src='q0r_a.json rows[0].F_S'),
    'T0B0_c47': dict(sign=f'{Q0R}/T0_a.npy', base='B2:lP', Ep=-18.13271883356696, c=4.7, F=5.875781737844458e-05, Evar=-18.13410241841458,
                     src='q0r_a.json rows[0].F_T; Evar = step3 solves.parent.start_guide.E_guide_var (la_a.npy = this la)'),
    'T1B0_c46': dict(sign=f'{Q0R}/T1_a.npy', base='B2:lP', Ep=-18.13271482099537, c=4.6, F=5.883409235243578e-05, Evar=None, src='q0r_a.json rows[1].F_T'),
    'T1B0_c45': dict(sign=f'{Q0R}/T1_a.npy', base='B2:lP', Ep=-18.13271482099537, c=4.5, F=5.8891017106630114e-05, Evar=None, src="q0r_a.json scans[2].scan['4.5']"),
    'S1B1_c52': dict(sign=f'{Q0R}/T0_b.npy', base=f'{Q0R}/lB1_b.npy', Ep=-18.135499469703262, c=5.2, F=3.132823466452569e-05, Evar=None, src='q0r_b.json rows[1].F_S'),
    'T1B1_c30': dict(sign=f'{Q0R}/T1_b.npy', base=f'{Q0R}/lB1_b.npy', Ep=-18.135494212856003, c=3.0, F=3.3721669676288637e-05, Evar=None, src='q0r_b.json rows[1].F_T'),
    'S0B0_bare': dict(sign='B2:sP', base='B2:lP', c=0.0, F='q0r_a:scans[0].bare', src='q0r_a.json scans[0].bare'),
    'T0B0_bare': dict(sign=f'{Q0R}/T0_a.npy', base='B2:lP', c=0.0, F='q0r_a:scans[1].bare', src='q0r_a.json scans[1].bare'),
    'T1B0_bare': dict(sign=f'{Q0R}/T1_a.npy', base='B2:lP', c=0.0, F='q0r_a:scans[2].bare', src='q0r_a.json scans[2].bare'),
    'fni_parent': dict(sign=f'{Q0R}/T0_a.npy', base=f'{FNI}/la_a.npy', c=None, F='fni:parent', src='step3 solves.parent.start_guide'),
    'fni_hybrid': dict(sign='fni:hybrid', base=f'{FNI}/la_a.npy', c=None, F='fni:hybrid', src='step3 solves.hybrid.start_guide'),
    'fni_rand': dict(sign='fni:random_hybrid_size', base=f'{FNI}/la_a.npy', c=None, F='fni:random_hybrid_size', src='step3 solves.random_hybrid_size.start_guide'),
}
# reference pairs (A, B), ASTRA_paired.md table; Delta = F_B - F_A per site
PAIRS = {
    'first': ('S0B0_c50', 'T0B0_c47'), 'fixed2': ('T0B0_c47', 'T1B0_c46'), 'refr2': ('S1B1_c52', 'T1B1_c30'), 'c45_46': ('T1B0_c45', 'T1B0_c46'),
    'bareS0T0': ('S0B0_bare', 'T0B0_bare'), 'bareT0T1': ('T0B0_bare', 'T1B0_bare'), 'fni_hyb': ('fni_parent', 'fni_hybrid'), 'fni_rand': ('fni_parent', 'fni_rand'),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def resolve_F(spec):
    """(F per site, Evar total or None) from the stored solves."""
    F = spec['F']
    if not isinstance(F, str): return F, spec.get('Evar')
    src, key = F.split(':', 1)
    if src == 'q0r_a':
        d = json.load(open(f'{Q0R}/q0r_a.json')); i = int(key.split('[')[1].split(']')[0]); r = d['scans'][i]['bare']
        return r['dE_site'], r['E_guide_var']
    if src == 'fni':
        d = json.load(open(f'{FNI}/step3_parent_hybrid_A_random_hybrid_size.json'))['solves'][key]['start_guide']
        return d['dE_site'], d['E_guide_var']
    raise ValueError(F)


def exact_delta(pair):
    A, B = PAIRS[pair]; return resolve_F(GUIDES[B])[0] - resolve_F(GUIDES[A])[0]


def build(name, log=print):
    sys.path.insert(0, G2CODE)
    import g2lib as L
    T = P.orbit_tables(); n, t1, t2 = T['n'], T['t1'], T['t2']; g = GUIDES[name]; t0 = time.time(); info = dict(name=name, spec=g)
    S0 = np.load(f'{Q0R}/T0_a.npy').astype(np.float64)
    if g['sign'].startswith('fni:'):
        T1 = np.load(f'{Q0R}/T1_a.npy').astype(np.float64); ix = np.flatnonzero(S0 != T1); h = np.load(f'{FNI}/flip_hybrid.npy')
        f = h if g['sign'] == 'fni:hybrid' else np.random.default_rng(1).choice(ix, len(h), replace=False)
        sig = S0.copy(); sig[f] *= -1; info['n_flips_vs_T0_a'] = int(len(f))
    else:
        sig = L.load_sign(g['sign'])
    if g['c'] is None:
        la = np.load(g['base']).astype(np.float64)
    else:
        lB = L.load_base(g['base'])
        if g['c'] == 0.0:
            la = lB
        else:
            fJ, fi = L.fJ_of(t1, t2, lB, g['Ep'], sig); info['fJ'] = fi; la = lB + g['c'] * fJ
    np.save(f'{GUIDE_DIR}/{name}.la.npy', la); np.save(f'{GUIDE_DIR}/{name}.sig.npy', sig.astype(np.int8))
    info.update(la_sha256=sha(f'{GUIDE_DIR}/{name}.la.npy'), sig_sha256=sha(f'{GUIDE_DIR}/{name}.sig.npy'), build_sec=time.time() - t0)
    json.dump(info, open(f'{GUIDE_DIR}/{name}.json', 'w'), indent=1, default=float)
    log(f'built {name} in {info["build_sec"]:.0f}s')
    return la, sig


def load(name, broken=None, walls_from=None):
    T = P.orbit_tables()
    la = np.load(f'{GUIDE_DIR}/{name}.la.npy'); sig = np.load(f'{GUIDE_DIR}/{name}.sig.npy')
    a = P.floor_amplitude(la, T['n']); del la
    ws = None if walls_from is None else np.load(f'{GUIDE_DIR}/{walls_from}.sig.npy')
    return P.OrbitTableGuide(sig, a, name=name, broken=broken, walls_sig=ws)


def sample_X0(guide, M, rng, law='guide2'):
    """initial orbits: 'guide2' ~ n_r a_r^2 (the guide's |psi|^2), 'uniform' ~ n_r (a random configuration: a far-from-equilibrium start)."""
    n = P.orbit_tables()['n']
    p = n * guide.a ** 2 if law == 'guide2' else n.copy()
    c = np.cumsum(p); c /= c[-1]
    return np.minimum(np.searchsorted(c, rng.random(M), side='right'), len(n) - 1).astype(np.int64)
