"""Test G0 (README "Test G0"): gamma-FN on exact tables (Sorella-Capriotti). Exact 6x6, no training.
  python cx_g0.py OUT SPEC.json   SPEC: {"refresh": run dir, "pp": path, "A": path}
Operator in the amplitude basis u (psi = s u), per sector vector x (weights as sec.vec / sec.fn_solve):
  A_gamma x = Dg_gamma x - 0.5 (h1 - s h2) - gamma B x,   B x = 0.5 (h1 + s h2) - D0 x,   h1,h2 = H x, H (s x)
  Dg_gamma = D0 + (1 + gamma)(Dg - D0).   gamma = 0 is H_FN, gamma = -1 is H.
Controls (run first; the grid runs only if (i), (ii) pass and (iii) fails as it must).
"""
import os, sys, time
import numpy as np
import scipy.linalg
from cx_lib import *


def fn_solve_gamma(sec, D0, la, s, gamma, flip=False, x0=None, tol=1e-10, maxit=600, nsite=N):
    """Ground state of A_gamma[a, s] (LOBPCG loop copied from st6_sector.Sector.fn_solve).
    flip=True is the deliberately broken copy: sign of the gamma B term reversed.
    Returns (E^gamma per site rel E0, u >= 0 normalised, info)."""
    t0 = time.time(); nm0 = sec.nmv
    sg = -1.0 if flip else 1.0
    la = la - jnp.max(la)
    a = jnp.exp(la); a = a / jnp.sqrt(jnp.sum(sec.n * a * a))
    a = jnp.maximum(a, 1e-15)
    w = a * sec.sqrt_n
    del a, la
    h1, h2 = sec.H2(w, s * w)
    Dg = 0.5 * (h1 + s * h2) / w
    del h1, h2
    Dg_g = D0 + (1.0 + gamma) * (Dg - D0)
    del Dg

    def A(x):
        h1, h2 = sec.H2(x, s * x)
        return Dg_g * x - 0.5 * (h1 - s * h2) - sg * gamma * (0.5 * (h1 + s * h2) - D0 * x)

    x = (w if x0 is None else x0); x = x / jnp.linalg.norm(x)
    del w
    Ax = A(x); th = float(x @ Ax)
    den = jnp.maximum(Dg_g - th, 1.0)
    P = None; AP = None
    it = 0; rn = np.inf
    for it in range(1, maxit + 1):
        r = Ax - th * x
        rn = float(jnp.linalg.norm(r))
        if rn <= tol * abs(th): break
        z = r / den
        z = z - (x @ z) * x
        if P is not None: z = z - (P @ z) * P / (P @ P)
        z = z / jnp.linalg.norm(z)
        Az = A(z)
        Sb = [x, z] + ([P] if P is not None else [])
        ASb = [Ax, Az] + ([AP] if P is not None else [])
        m = len(Sb)
        del r
        GA = np.array([[float(Sb[i] @ ASb[j]) for j in range(m)] for i in range(m)])
        GB = np.array([[float(Sb[i] @ Sb[j]) for j in range(m)] for i in range(m)])
        GA = 0.5 * (GA + GA.T)
        try:
            ev, ec = scipy.linalg.eigh(GA, GB)
        except Exception:
            P = None; AP = None; continue
        c = ec[:, 0]
        xn = sum(c[i] * Sb[i] for i in range(m)); Axn = sum(c[i] * ASb[i] for i in range(m))
        del x, Ax, P, AP
        P = sum(c[i] * Sb[i] for i in range(1, m)); AP = sum(c[i] * ASb[i] for i in range(1, m))
        del Sb, ASb, z, Az
        nr = float(jnp.linalg.norm(xn)); x = xn / nr; Ax = Axn / nr
        pn = float(jnp.linalg.norm(P))
        if pn > 0: P = P / pn; AP = AP / pn
        th = float(x @ Ax)
    sgn = 1.0 if float(jnp.sum(x)) >= 0 else -1.0
    x = x * sgn
    negw = float(jnp.sum(jnp.where(x < 0, x * x, 0.0)))      # weight of the negative part of the Perron vector
    u = jnp.abs(x); u = u / jnp.linalg.norm(u)
    E = float(u @ A(u))
    info = dict(gamma=gamma, flip=flip, E_site=(E - sec.E0) / nsite, theta_site=(th - sec.E0) / nsite,
                lobpcg_it=it, res=rn / abs(th), neg_weight=negw, matvecs=sec.nmv - nm0, sec=time.time() - t0)
    return (E - sec.E0) / nsite, u, info


def h_on_phi(sec, u, s, nsite=N):
    """<H> on phi^gamma = s*u (same weighted sector basis as sec.vec), per site rel E0."""
    return (sec.energy(s * u) - sec.E0) / nsite


if __name__ == '__main__':
    OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
    F = os.path.join(OUT, 'cx_g0.json'); T00 = time.time()
    lab = Lab(); sec = lab.sec; D0 = lab.D0

    def norm_la(la):
        a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))
    RF = SPEC['refresh']
    s2 = jnp.asarray(np.load(os.path.join(RF, 's2.npy')).astype(np.float32))
    T0JSON = '/project/theorie/a/A.Otaifi/chatty_compress6/runs/t0_17410220/cx_t0.json'
    EFN_T0 = json.load(open(T0JSON))['bases']['vit']['EFN']
    E_FN_README = 1.0186e-4
    res = dict(spec=SPEC, E_FN_t0json=EFN_T0, controls={}, grid={}); dump(F, res)

    # ---- exact psi0 (Lab nulls sec.la0/s0/v0, so read the table again)
    z = np.load(TABLE); amp = z['amp']; del z
    la0 = norm_la(jnp.log(jnp.maximum(jnp.abs(jnp.asarray(amp)), 1e-300)))
    s0 = jnp.where(jnp.asarray(amp) >= 0, 1.0, -1.0).astype(jnp.float32); del amp
    rng = np.random.default_rng(0)
    x0n = jnp.asarray(np.exp(0.1 * rng.standard_normal(sec.D)))      # perturbed start for (ii), (iii)
    ctl = res['controls']

    # ---- control (i)
    lvit = norm_la(lab.lP0)
    E, u, info = fn_solve_gamma(sec, D0, lvit, s2, 0.0); del u
    sec_E, _, info_ref = sec.fn_solve(lvit, s2)
    ci = dict(E_gamma0=E, E_fn_solve=info_ref['dE_FN_site'], E_t0json=EFN_T0, E_readme=E_FN_README, info=info,
              pass_=bool(abs(E - EFN_T0) < 1e-8 and abs(E - info_ref['dE_FN_site']) < 1e-9 and abs(E - E_FN_README) < 1e-7))
    ctl['i'] = ci; dump(F, res); log('control (i)', ci)

    # ---- controls (ii) and (iii)
    cii = dict(tol=1e-9, runs={}); ciii = dict(tol=1e-9, runs={})
    for g in (0.0, 0.5, 2.0):
        E, u, info = fn_solve_gamma(sec, D0, la0, s0, g, x0=x0n); del u
        cii['runs'][str(g)] = dict(E_site=E, info=info, ok=bool(abs(E) < 1e-9))
    cii['pass_'] = all(r['ok'] for r in cii['runs'].values())
    for g in (0.5, 2.0):
        E, u, info = fn_solve_gamma(sec, D0, la0, s0, g, flip=True, x0=x0n); del u
        ciii['runs'][str(g)] = dict(E_site=E, info=info, fails_as_required=bool(abs(E) >= 1e-9))
    ciii['pass_'] = all(r['fails_as_required'] for r in ciii['runs'].values())      # True = the broken copy failed (-> check OK)
    ctl['ii'] = cii; ctl['iii'] = ciii; dump(F, res); log('control (ii)', cii); log('control (iii)', ciii)
    ctl['all_ok'] = bool(ci['pass_'] and cii['pass_'] and ciii['pass_'])
    dump(F, res)
    if not ctl['all_ok']:
        log('CONTROLS FAILED: grid not run'); res['sec'] = time.time() - T00; dump(F, res); sys.exit(2)

    # ---- grid
    bases = {'vit': lvit}
    for nm, key in (('pp', 'pp'), ('rbmpp', 'A')):
        zz = np.load(SPEC[key]); bases[nm] = norm_la(jnp.asarray(zz['lP'], jnp.float64)); del zz
    GAM = [0.0, 0.25, 0.5, 1.0, 2.0]
    for nm, lb in bases.items():
        rec = dict(gamma=GAM, E_gamma=[], H_phi=[], neg_weight=[], res=[], lobpcg_it=[], sec=[])
        for g in GAM:
            E, u, info = fn_solve_gamma(sec, D0, lb, s2, g)
            Hp = h_on_phi(sec, u, s2); del u
            for k, v in (('E_gamma', E), ('H_phi', Hp), ('neg_weight', info['neg_weight']), ('res', info['res']),
                         ('lobpcg_it', info['lobpcg_it']), ('sec', info['sec'])): rec[k].append(v)
            log(f'[{nm}] gamma {g}: E^gamma {E:.5e}  <H>_phi {Hp:.5e}  negw {info["neg_weight"]:.2e}  res {info["res"]:.1e}  it {info["lobpcg_it"]}')
            res['grid'][nm] = rec; dump(F, res)
        cf = np.polyfit(GAM[:3], rec['E_gamma'][:3], 1)                       # linear fit over gamma in {0, 0.25, 0.5}
        rec['chord_gamma_m1'] = float(np.polyval(cf, -1.0)); rec['chord_slope'] = float(cf[0])
        rec['min_H_phi'] = min(rec['H_phi']); rec['gamma_at_min'] = GAM[int(np.argmin(rec['H_phi']))]
        res['grid'][nm] = rec; dump(F, res)
    res['sec'] = time.time() - T00; dump(F, res); log('DONE', res['sec'])
