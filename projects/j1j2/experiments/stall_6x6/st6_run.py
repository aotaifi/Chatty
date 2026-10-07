"""Stall diagnosis driver (6x6 J1-J2, J2/J1 = 0.5, exact symmetric sector on one A40).

  python st6_run.py base   OUT                    validation of the GPU sector code + ViT baseline table
  python st6_run.py fit    OUT SPEC.json          test 1 / 3: supervised fit(s) to |psi0|, exact scoring
  python st6_run.py loop   OUT SPEC.json          test 2: exact FN/Krylov loop, unprojected and ViT-projected
All energies exact in the sector (k=0, A1, flip+).  Results: OUT/*.json, params OUT/*.npy.
"""
import json, os, sys, time
import numpy as np
import st6_sector as SS
import st6_net as NT
import jax
import jax.numpy as jnp

CSR = os.environ.get('ST6_CSR', '/home/a/A.Otaifi/chatty_stall6/csr')
TABLE = os.environ.get('ST6_TABLE', '/home/a/A.Otaifi/chatty_stall6/data/psi0_6x6_table.npz')
CKPT = os.environ.get('ST6_CKPT', '/home/a/A.Otaifi/chatty_stall6/data/vit_J2=0.50_N=6x6_k=0.mpack')
DATA = os.environ.get('ST6_DATA', '/project/theorie/a/A.Otaifi/chatty_stall6/data')
log = SS.log
mode, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
SPEC = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
T00 = time.time()


def dump(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), 'w'), indent=1, default=float)


def marshall(reps):
    A = 0
    for y in range(6):
        for x in range(6):
            if (x + y) % 2 == 0: A |= 1 << (x + 6 * y)
    pc = jnp.zeros(reps.shape[0], jnp.int32)
    m = reps & jnp.uint64(A)
    for b in range(36):
        pc = pc ^ ((m >> jnp.uint64(b)) & jnp.uint64(1)).astype(jnp.int32)
    return jnp.where(pc == 0, 1.0, -1.0)


sec = SS.Sector(CSR, TABLE)
log(f'GPU mem after sector: {jax.devices()[0].memory_stats().get("bytes_in_use", 0)/1e9:.1f} GB')


def vit_tables(model):
    """|ViT| log-amplitude and binarised ViT sign on all reps (cached)."""
    p = os.path.join(DATA, 'vit_reps_tables.npz')
    if os.path.exists(p):
        z = np.load(p)
        return jnp.asarray(z['la']), jnp.asarray(z['s'].astype(np.float32)), float(z['phi'])
    t = time.time()
    la, im = model.eval_reps(model.flat0, sec.reps, complex_out=True)
    v = sec.vec(la, jnp.ones_like(la))
    phi = 0.5 * float(jnp.angle(jnp.sum(v * v * jnp.exp(2j * im))))
    s = jnp.where(jnp.cos(im - phi) >= 0, 1.0, -1.0)
    if float(s[0]) < 0: s = -s
    np.savez(p, la=np.asarray(la), s=np.asarray(s, np.int8), phi=phi)
    s = s.astype(jnp.float32); del im, v
    log(f'ViT tables on {sec.D} reps in {time.time()-t:.0f}s, phi={phi:.4f}')
    return la, s, phi


def base_tables(name):
    if name == 'projvit':
        z = np.load(os.path.join(DATA, 'sym_tables.npz'))
        return jnp.asarray(z['lP']), jnp.asarray(z['sP'].astype(np.float32))
    raise ValueError(name)


def asym_check(model, flat, n=20000, seed=0):
    """spread of the network over the 16 point-group x flip images of reps drawn from psi0^2."""
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(np.asarray(sec.p0)); idx = np.searchsorted(cdf, rng.random(n) * cdf[-1])
    idx = jnp.asarray(np.minimum(idx, sec.D - 1))
    x = sec.reps[idx]
    las, ims = [], []
    for g in range(8):
        for fl in (False, True):
            y = SS.image(sec.T, x, jnp.full(n, g), jnp.full(n, fl))
            a, b = model.eval_reps(flat, y, batch=4096, complex_out=True)
            las.append(a); ims.append(b)
    la = jnp.stack(las); im = jnp.stack(ims)
    sd = float(jnp.mean(jnp.std(la, axis=0)))
    return dict(asym_std_log_amp=sd, asym_max_log_amp=float(jnp.max(jnp.max(la, 0) - jnp.min(la, 0))), n=n)


def full_eval(tag, la, signs, fn=True, kry_from=None):
    """exact <H> for each sign, E_FN for each sign, optional Krylov step from (la, kry_from)."""
    out = dict(tag=tag)
    for sn, s in signs.items():
        out[f'H[{sn}]'] = sec.score(la, s)
        if fn:
            _, _, info = sec.fn_solve(la, s)
            out[f'FN[{sn}]'] = info
        log(tag, sn, json.dumps({k: out[k] for k in out if sn in k}))
    if kry_from is not None:
        v = sec.vec(la, jnp.ones_like(la))
        s1, E1, kinfo = sec.krylov(v, kry_from)
        out['kry_step'] = kinfo
        out['H[kry]'] = sec.score(la, s1)
        if fn:
            _, _, info = sec.fn_solve(la, s1)
            out['FN[kry]'] = info
        log(tag, 'kry', json.dumps({k: out[k] for k in ('H[kry]', 'kry_step')}))
    return out


# =============================================================================================== fits
def run_fits(runs):
    la_v, s_v, phi = vit_tables(NT.Model({'kind': 'vit'}, CKPT))
    allres = []
    for run in runs:
        tag = run['tag']; t0 = time.time()
        model = NT.Model(run['model'], CKPT, seed=run.get('seed', 0))
        flat = model.flat0
        if 'init_params' in run:
            flat = jnp.asarray(np.load(run['init_params']), jnp.float32)
        log(f'== fit {tag}: model {run["model"]} npar {model.npar} (vit part {model.npar_vit})')

        def ev(fl):
            la = model.eval_reps(fl, sec.reps)
            a = sec.score(la, sec.s0); b = sec.score(la, s_v)
            return dict(dE_exact=a['dE_site'], dE_vitsign=b['dE_site'], std_dlog=a['std_dlog'])
        hist = []
        base = None
        if run.get('base'):
            base, s_own = base_tables(run['base'])
        else:
            s_own = s_v
        if run.get('loss', 'l2') == 'exact':
            s_tab = sec.s0 if run.get('sign', 'exact') == 'exact' else s_own
            other = s_own if run.get('sign', 'exact') == 'exact' else sec.s0

            def ex(fl, la):                      # la already includes the base table
                b = sec.score(la, other)
                d = la - sec.la0; mu = jnp.sum(sec.p0 * d)
                return dict(dE_othersign=b['dE_site'], std_dlog=float(jnp.sqrt(jnp.sum(sec.p0 * (d - mu) ** 2))))
            flat, hist = NT.fit_exact(model, flat, sec, s_tab, mode='var', outer=run.get('outer', 12),
                                      N_sr=run.get('N_sr', 6144), B_g=run.get('B_g', 16384), lam=run.get('lam', 1e-3),
                                      eta_sr=run.get('eta_sr', 0.3), eta_g=run.get('eta_g', 1e-5),
                                      seed=run.get('seed', 0), eval_extra=ex, hist=hist,
                                      mask=model.mask_corr if run.get('train', 'all') == 'corr' else None,
                                      base_tab=base)
        elif run.get('loss', 'l2') == 'table':
            s_tab = sec.s0 if run.get('sign', 'exact') == 'exact' else s_v
            other = s_v if run.get('sign', 'exact') == 'exact' else sec.s0

            def ex(fl, la):
                b = sec.score(la, other)
                d = la - sec.la0; mu = jnp.sum(sec.p0 * d)
                return dict(dE_othersign=b['dE_site'], std_dlog=float(jnp.sqrt(jnp.sum(sec.p0 * (d - mu) ** 2))))
            flat, hist = NT.fit_table(model, flat, sec, s_tab, mode='var', outer=run.get('outer', 15),
                                      inner=run.get('inner', 40), B=run.get('B', 4096), lr=run['lr'],
                                      seed=run.get('seed', 0), warmup=run.get('warmup', 20), eval_extra=ex, hist=hist)
        elif run.get('loss', 'l2') == 'energy':
            s_tab = sec.s0 if run.get('sign', 'exact') == 'exact' else s_v

            def ev2(fl, la):
                a = sec.score(la, sec.s0); b = sec.score(la, s_v)
                o = a if run.get('sign', 'exact') == 'exact' else b
                return dict(obj=o['dE_site'], dE_exact=a['dE_site'], dE_vitsign=b['dE_site'], std_dlog=a['std_dlog'])
            flat, hist = NT.fit_energy(model, flat, sec, s_tab, mode='var', steps=run['steps'], lr=run['lr'],
                                       B=run.get('B', 256), seed=run.get('seed', 0), refresh=run.get('refresh', 500),
                                       warmup=run.get('warmup', 100), eval_fn=ev2, hist=hist)
        else:
            flat, hist = NT.fit(model, flat, sec, sec.la0, beta=run.get('beta', 1.0), steps=run['steps'], lr=run['lr'],
                                B=run.get('B', 512), K=run.get('K', 4), lam_amp=run.get('lam_amp', 1.0),
                                lam_edge=run.get('lam_edge', 1.0), seed=run.get('seed', 0),
                                eval_every=run.get('eval_every', 0), eval_fn=ev, hist=hist,
                                warmup=run.get('warmup', 200), opt=run.get('opt', 'adam'))
        np.save(os.path.join(OUT, f'params_{tag}.npy'), np.asarray(flat))
        la = model.eval_reps(flat, sec.reps)
        if base is not None: la = la + base
        r = dict(tag=tag, run=run, npar=model.npar, hist=hist, fit_sec=time.time() - t0)
        r.update(full_eval(tag, la, {'exact': sec.s0, 'own': s_own}, fn=run.get('fn', True),
                           kry_from=s_own if run.get('kry', True) else None))
        if run.get('asym', True):
            r['asym'] = asym_check(model, flat)
        r['sec'] = time.time() - t0
        allres.append(r)
        dump(f'fit_{tag}.json', r)
        log(f'== done {tag} {r["sec"]:.0f}s', json.dumps({k: r[k] for k in r if k.startswith('H[') or k.startswith('FN[')}))
    return allres


# =============================================================================================== base
if mode == 'base':
    res = dict()
    # (1) exact GS: <H> and FN with the exact guide return E0
    t = time.time()
    E = sec.energy(sec.v0); res['E0_rayleigh_site'] = E / 36; res['E0_table_site'] = sec.E0 / 36
    log('E0 check', E / 36, sec.E0 / 36, f'{time.time()-t:.1f}s (1 matvec + jit)')
    t = time.time(); [sec.H(sec.v0).block_until_ready() for _ in range(5)]; res['sec_per_matvec'] = (time.time() - t) / 5
    log('matvec sec', res['sec_per_matvec'])
    ms = jax.devices()[0].memory_stats(); log(f'mem in use {ms.get("bytes_in_use",0)/1e9:.2f} GB peak {ms.get("peak_bytes_in_use",0)/1e9:.2f} GB limit {ms.get("bytes_limit",0)/1e9:.2f} GB')
    Efn, u, info = sec.fn_solve(sec.la0, sec.s0); res['FN_exact_guide'] = info
    ms = jax.devices()[0].memory_stats(); log(f'mem in use {ms.get("bytes_in_use",0)/1e9:.2f} GB peak {ms.get("peak_bytes_in_use",0)/1e9:.2f} GB')
    log('FN exact guide', info)
    # (2) CPU reference: J2=0 amplitude + Marshall signs, iteration 1 of sym6x6_plain
    z = np.load(os.path.join(DATA, 'init_L6_J20.0.npz'))
    vinit = jnp.abs(jnp.asarray(z['v'])); vinit = vinit / jnp.linalg.norm(vinit)
    sM = marshall(sec.reps); sM = sM * (1.0 if float(sM[0]) > 0 else -1.0)
    la_init = jnp.log(jnp.maximum(vinit / sec.sqrt_n, 1e-300))
    Efn, afn, info = sec.fn_solve(la_init, sM, verbose=True)
    res['cpu_ref_it1_FN'] = dict(gpu=Efn, cpu=-17.45688129087055, diff=Efn + 17.45688129087055, info=info)
    log('CPU ref FN', res['cpu_ref_it1_FN'])
    s1, E1, kinfo = sec.krylov(afn, sM)
    Et = sec.energy(afn * s1)
    res['cpu_ref_it1_trial'] = dict(gpu=Et, cpu=-17.683298665045623, diff=Et + 17.683298665045623, kry=kinfo)
    log('CPU ref Krylov', res['cpu_ref_it1_trial'])
    dump('base_validation.json', res)
    del vinit, sM, la_init, afn, s1, u, z
    # (3) ViT tables and baseline / oracle guides
    model = NT.Model({'kind': 'vit'}, CKPT)
    res['npar_vit'] = model.npar
    t = time.time(); la_v, s_v, phi = vit_tables(model); res['vit_tables_sec'] = time.time() - t
    res['asym'] = asym_check(model, model.flat0)
    log('asym', res['asym'])
    res['vit'] = full_eval('vit', la_v, {'vit': s_v, 'exact': sec.s0}, kry_from=s_v)
    res['psi0_amp'] = full_eval('psi0amp', sec.la0, {'vit': s_v})
    res['sec'] = time.time() - T00
    dump('base_validation.json', res)
    log('DONE base', res['sec'])
    run_fits(SPEC.get('runs', []))
    log('DONE base+fits', time.time() - T00)

# =============================================================================================== sym (test 0)
elif mode == 'sym':
    # Test 0: exact energy of the symmetry-projected ViT psi_P(x) = sum_{g in D4 x flip} psi(g x) (translations are
    # built into the ViT).  psi0 is in the trivial sector (k=0, A1, flip+), so all characters are +1.
    model = NT.Model(SPEC.get('model', {'kind': 'vit'}), CKPT)
    flat = model.flat0 if not SPEC.get('params') else jnp.asarray(np.load(SPEC['params']), jnp.float32)
    la_v, s_v, phi = vit_tables(model)
    res = dict(phi=phi)
    t0 = time.time()
    z0r = z0i = None
    S = None; Sa = None; Sl = None
    for g in range(8):
        for fl in (False, True):
            x = SS.image(sec.T, sec.reps, jnp.full(sec.D, g, jnp.int32), jnp.full(sec.D, fl))
            zr, zi = model.eval_reps(flat, x, complex_out=True)
            del x
            if z0r is None:
                z0r, z0i = zr, zi
                S = jnp.zeros(sec.D, jnp.complex128); Sa = jnp.zeros(sec.D); Sl = jnp.zeros(sec.D)
            S = S + jnp.exp((zr - z0r) + 1j * zi)
            Sa = Sa + jnp.exp(zr - z0r)
            Sl = Sl + (zr - z0r) / 16.0
            del zr, zi
            log(f'sym image g={g} flip={fl} {time.time()-t0:.0f}s')
    res['sec_images'] = time.time() - t0
    # (0) unprojected, rep-evaluated, full complex phase
    def cvec(lmag, ph):
        m = lmag - jnp.max(lmag)
        a = jnp.exp(m) * sec.sqrt_n
        vr = a * jnp.cos(ph); vi = a * jnp.sin(ph)
        nrm = jnp.sqrt(vr @ vr + vi @ vi)
        return vr / nrm, vi / nrm

    def cenergy(vr, vi):
        Y = sec.Hm(jnp.stack([vr, vi], 1))
        return float(vr @ Y[:, 0] + vi @ Y[:, 1])

    vr, vi = cvec(z0r, z0i)
    E = cenergy(vr, vi); res['rep_complex'] = dict(E_site=E / 36, dE_site=(E - sec.E0) / 36)
    log('unprojected rep, complex phase', res['rep_complex'])
    # (a) projected, own complex phase
    lP = z0r + jnp.log(jnp.abs(S)); phP = jnp.angle(S)
    vr, vi = cvec(lP, phP)
    E = cenergy(vr, vi); res['proj_complex'] = dict(E_site=E / 36, dE_site=(E - sec.E0) / 36)
    log('projected, complex phase', res['proj_complex'])
    del vr, vi
    s_P = jnp.where(jnp.cos(phP - phi) >= 0, 1.0, -1.0).astype(jnp.float32)
    if float(s_P[0]) < 0: s_P = -s_P
    res['proj_phase_spread'] = float(jnp.sum(sec.p0 * jnp.sin(phP - phi) ** 2))
    res['proj_absS_over_Sa'] = float(jnp.sum(sec.p0 * jnp.abs(S) / Sa))     # 1 = no phase cancellation
    # (b) |psi_P| with binarised own sign / exact sign / one exact Krylov step, + FN
    res['proj_abs'] = full_eval('projabs', lP, {'own': s_P, 'exact': sec.s0, 'vitrep': s_v}, kry_from=s_P)
    # amplitude-only projections: arithmetic and geometric mean of |psi(gx)|
    la_A = z0r + jnp.log(Sa / 16.0); la_G = z0r + Sl
    res['amp_arith'] = full_eval('amp_arith', la_A, {'vitrep': s_v, 'exact': sec.s0}, fn=False)
    res['amp_geom'] = full_eval('amp_geom', la_G, {'vitrep': s_v, 'exact': sec.s0}, fn=False)
    np.savez(os.path.join(OUT, 'sym_tables.npz'), lP=np.asarray(lP), sP=np.asarray(s_P, np.int8))
    res['sec'] = time.time() - T00
    dump('sym_test0.json', res)
    log('DONE sym', json.dumps({k: v for k, v in res.items() if not isinstance(v, dict) or 'dE_site' in v}))

# =============================================================================================== feat (tests 1-3 on an exact family)
elif mode == 'feat':
    import st6_feat as FT
    base, s_P = base_tables(SPEC.get('base', 'projvit'))
    F, names = FT.build_features(sec, SPEC['features']) if SPEC.get('features') else ([], [])
    fam = FT.Family(sec, base, F)
    K = fam.K
    res = dict(features=SPEC['features'], K=K, names=names)
    c0 = np.zeros(K)
    lm_it = SPEC.get('lm_iters', 8)
    sc0 = sec.score(base, s_P)
    v = sec.vec(base, jnp.ones_like(base))
    s_k, _, _ = sec.krylov(v, s_P); del v
    res['start'] = dict(own=sc0, exact=sec.score(base, sec.s0), kry=sec.score(base, s_k))
    log('start', json.dumps({k: v_['dE_site'] for k, v_ in res['start'].items()}))
    # ---- capacity: best variational energy in the family at fixed signs
    if SPEC.get('capacity', True):
        res['cap'] = {}
        for nm, s in (('own', s_P), ('exact', sec.s0), ('kry', s_k)):
            t0 = time.time()
            c, h = fam.optimize(c0, 'var', s, iters=lm_it)
            la = fam.la(c)
            r = dict(hist_dE=[(e - sec.E0) / 36 for e in h], H=sec.score(la, s), sec=time.time() - t0)
            if SPEC.get('fn', True):
                _, _, info = sec.fn_solve(la, s); r['FN'] = info
            res['cap'][nm] = r
            np.save(os.path.join(OUT, f'c_cap_{nm}.npy'), c)
            log('cap', nm, 'dE', r['hist_dE'][0], '->', r['hist_dE'][-1], 'FN', r.get('FN', {}).get('dE_FN_site'))
            dump('feat.json', res)
    # ---- variational loop on the family: Krylov sign <-> exact amplitude optimisation
    if SPEC.get('alt_rounds', 0):
        c = np.zeros(K); s = s_P; res['alt'] = []
        for rd in range(SPEC['alt_rounds']):
            c, h = fam.optimize(c, 'var', s, iters=lm_it)
            la = fam.la(c); v = sec.vec(la, jnp.ones_like(la))
            s1, _, _ = sec.krylov(v, s); del v
            rec = dict(round=rd, dE_amp_opt=(h[-1] - sec.E0) / 36, after_kry=sec.score(la, s1))
            s = s1; res['alt'].append(rec)
            log('alt', rd, rec['dE_amp_opt'], rec['after_kry']['dE_site'], rec['after_kry']['w_s'])
            dump('feat.json', res)
    # ---- one-hop features: capacity of base * exp(sum c_k hop_k(base)) at fixed signs
    if SPEC.get('hop_capacity', False):
        res['hopcap'] = {}
        for nm, s in (('own', s_P), ('kry', s_k)):
            Fh, nh = FT.hop_features(sec, base, s)
            famh = FT.Family(sec, base, Fh + (F if SPEC.get('hop_with_clusters') else []))
            c, h = famh.optimize(np.zeros(famh.K), 'var', s, iters=lm_it)
            la = famh.la(c)
            r = dict(hist_dE=[(e - sec.E0) / 36 for e in h], H=sec.score(la, s), c=list(map(float, c)), names=nh)
            _, _, info = sec.fn_solve(la, s); r['FN'] = info
            res['hopcap'][nm] = r
            log('hopcap', nm, r['hist_dE'][0], '->', r['hist_dE'][-1], 'FN', info['dE_FN_site'])
            del Fh, famh
            dump('feat.json', res)
    # ---- projected FN loop with one-hop features of the current guide (+ clusters)
    if SPEC.get('hop_loop_iters', 0):
        s = s_P; la = base; res['hoploop'] = []
        for it in range(1, SPEC['hop_loop_iters'] + 1):
            t0 = time.time()
            Efn, u, info = sec.fn_solve(la, s); del u
            H_old = sec.score(la, s)['E']
            Fh, nh = FT.hop_features(sec, la, s)
            famh = FT.Family(sec, la, Fh + (F if SPEC.get('hop_with_clusters') else []))
            c, h = famh.optimize(np.zeros(famh.K), 'fn', s, la_g=la, iters=lm_it)
            la_new = famh.la(c)
            frac = (H_old - h[-1]) / max(H_old - Efn, 1e-300)
            v = sec.vec(la_new, jnp.ones_like(la_new))
            s1, _, _ = sec.krylov(v, s); del v
            sc = sec.score(la_new, s1)
            rec = dict(it=it, dE_FN_prev=info['dE_FN_site'], frozenFN_dE_reached=(h[-1] - sec.E0) / 36,
                       frac_FN_gain_captured=frac, H=sc, H_oldsign=sec.score(la_new, s)['dE_site'], c=list(map(float, c)),
                       sec=time.time() - t0)
            la, s = la_new, s1
            del Fh, famh
            res['hoploop'].append(rec)
            log('HOPLOOP', it, 'E_FN', rec['dE_FN_prev'], 'frozen', rec['frozenFN_dE_reached'], 'frac', round(frac, 3),
                '<H>', sc['dE_site'], 'w_s', sc['w_s'])
            dump('feat.json', res)
        _, _, info = sec.fn_solve(la, s); res['hoploop_final_FN'] = info
        log('HOPLOOP final FN', info['dE_FN_site'])
    # ---- projected exact FN loop onto the family (projection = exact minimisation of the frozen FN energy)
    if SPEC.get('loop_iters', 0):
        c = np.zeros(K); s = s_P; la = base; res['loop'] = []
        for it in range(1, SPEC['loop_iters'] + 1):
            t0 = time.time()
            Efn, u, info = sec.fn_solve(la, s)
            H_old = sec.score(la, s)['E']                    # = frozen FN energy of the guide itself
            c, h = fam.optimize(c, 'fn', s, la_g=la, iters=lm_it)
            la_new = fam.la(c)
            frac = (H_old - h[-1]) / max(H_old - Efn, 1e-300)
            v = sec.vec(la_new, jnp.ones_like(la_new))
            s1, _, _ = sec.krylov(v, s); del v
            sc = sec.score(la_new, s1)
            rec = dict(it=it, dE_FN_prev=info['dE_FN_site'], frozenFN_dE_reached=(h[-1] - sec.E0) / 36,
                       frac_FN_gain_captured=frac, H=sc, sec=time.time() - t0)
            la, s = la_new, s1
            res['loop'].append(rec)
            log('LOOP', it, 'E_FN', rec['dE_FN_prev'], 'frozen', rec['frozenFN_dE_reached'], 'frac', round(frac, 3),
                '<H>', sc['dE_site'], 'w_s', sc['w_s'])
            dump('feat.json', res)
        _, _, info = sec.fn_solve(la, s); res['loop_final_FN'] = info
        log('LOOP final FN', info['dE_FN_site'])
    res['sec'] = time.time() - T00
    dump('feat.json', res)
    log('DONE feat', res['sec'])

# =============================================================================================== interp (accuracy needed)
elif mode == 'interp':
    # From the guide (psi_P amplitude, own sign): exact FN state phi_1, update delta = log phi_1 - log a.
    # (i) partial steps a_t = a exp(t delta);  (ii) iid log-amplitude noise of std sigma on top of phi_1 / on top of a;
    # (iii) delta smoothed by keeping only its projection on the high-weight configurations is not done here.
    base, s_P = base_tables('projvit')
    res = {}
    Efn, u, info = sec.fn_solve(base, s_P); res['FN'] = info
    la1 = jnp.log(jnp.maximum(u / sec.sqrt_n, 1e-300)); del u
    d = la1 - base
    mu = jnp.sum(sec.p0 * d); res['std_delta'] = float(jnp.sqrt(jnp.sum(sec.p0 * (d - mu) ** 2)))
    v = sec.vec(la1, jnp.ones_like(la1)); s1, _, _ = sec.krylov(v, s_P); del v
    res['steps'] = []
    for t in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25):
        la = base + t * d
        r = dict(t=t, H_own=sec.score(la, s_P)['dE_site'], H_kry1=sec.score(la, s1)['dE_site'])
        res['steps'].append(r); log('step', r)
    res['noise'] = []
    key = jax.random.PRNGKey(5)
    for sig in (0.003, 0.01, 0.03):
        key, k = jax.random.split(key)
        xi = jax.random.normal(k, (sec.D,))
        r = dict(sigma=sig, phi1_plus_noise=sec.score(la1 + sig * xi, s1)['dE_site'],
                 psiP_plus_noise=sec.score(base + sig * xi, s_P)['dE_site'])
        del xi
        res['noise'].append(r); log('noise', r)
    res['phi1_kry'] = sec.score(la1, s1)
    dump('interp.json', res)
    log('DONE interp', json.dumps(res))

# =============================================================================================== fit
elif mode == 'fit':
    run_fits(SPEC['runs'])
    log('DONE fit', time.time() - T00)

# =============================================================================================== loop
elif mode == 'loop':
    model = NT.Model(SPEC.get('model', {'kind': 'vit'}), CKPT)
    la_v, s_v, phi = vit_tables(NT.Model({'kind': 'vit'}, CKPT)) if SPEC.get('model', {'kind': 'vit'})['kind'] != 'vit' else vit_tables(model)
    flat = model.flat0
    base = None
    if SPEC.get('base'):
        base, s_own = base_tables(SPEC['base'])
        la_start = base + model.eval_reps(flat, sec.reps); s_start = s_own if SPEC.get('start_sign', 'vit') == 'vit' else sec.s0
    elif SPEC.get('init_params'):
        flat = jnp.asarray(np.load(SPEC['init_params']), jnp.float32)
        la_start = model.eval_reps(flat, sec.reps)
        s_start = s_v if SPEC.get('start_sign', 'vit') == 'vit' else sec.s0
    else:
        la_start = la_v
        s_start = s_v if SPEC.get('start_sign', 'vit') == 'vit' else sec.s0
    nit = SPEC.get('iters', 8)
    hist = []
    # ---- unprojected exact loop from the same start (cheap): a_{k+1} = phi_FN[a_k, s_k], s_{k+1} = Kry(a_{k+1}, s_k)
    if SPEC.get('exact_loop', True):
        la, s = la_start, s_start
        rec = dict(it=0, kind='exact', **{f'H_{k}': v for k, v in sec.score(la, s).items()})
        hist.append(rec); log('EXACT', json.dumps(rec))
        for it in range(1, SPEC.get('exact_iters', 12) + 1):
            Efn, u, info = sec.fn_solve(la, s)
            la = jnp.log(jnp.maximum(u / sec.sqrt_n, 1e-300))
            s1, _, kinfo = sec.krylov(u, s)
            sc = sec.score(la, s1); sc0 = sec.score(la, sec.s0)
            rec = dict(it=it, kind='exact', dE_FN_prev=info['dE_FN_site'], fn=info, kry=kinfo,
                       **{f'H_{k}': v for k, v in sc.items()}, H_exactsign_dE=sc0['dE_site'])
            s = s1
            hist.append(rec); log('EXACT', json.dumps({k: rec[k] for k in ('it', 'dE_FN_prev', 'H_dE_site', 'H_w_s', 'H_std_dlog', 'H_exactsign_dE')}))
            dump('loop_hist.json', hist)
    # ---- projected loop: a_{k+1} = Proj_ViT(phi_FN[a_k, s_k]) (warm-started supervised fit), s exact Krylov
    la, s = la_start, s_start
    fitc = SPEC['fit']
    rec = dict(it=0, kind='proj', **{f'H_{k}': v for k, v in sec.score(la, s).items()})
    hist.append(rec); log('PROJ', json.dumps(rec))
    for it in range(1, nit + 1):
        t0 = time.time()
        Efn, u, info = sec.fn_solve(la, s)
        tlog = jnp.log(jnp.maximum(u / sec.sqrt_n, 1e-300))
        sc_phi = sec.score(tlog, s)
        fh = []
        if fitc.get('loss', 'l2') == 'exact_fn':
            flat, fh = NT.fit_exact(model, flat, sec, s, mode='fn', la_g=la, outer=fitc.get('outer', 6),
                                    N_sr=fitc.get('N_sr', 6144), B_g=fitc.get('B_g', 16384), lam=fitc.get('lam', 1e-3),
                                    eta_sr=fitc.get('eta_sr', 0.3), eta_g=fitc.get('eta_g', 1e-5), seed=1000 * it, hist=fh,
                                    base_tab=base)
        elif fitc.get('loss', 'l2') == 'table_fn':
            flat, fh = NT.fit_table(model, flat, sec, s, mode='fn', la_g=la, outer=fitc.get('outer', 10),
                                    inner=fitc.get('inner', 40), B=fitc.get('B', 4096), lr=fitc['lr'],
                                    seed=1000 * it, warmup=fitc.get('warmup', 10), hist=fh)
        elif fitc.get('loss', 'l2') == 'fn':
            # projection = minimise the frozen FN energy <H_FN[a_k, s_k]> over the network (exact minimiser: phi_FN)
            def ev3(fl, lab, la=la, s=s):
                ef = sec.fn_rayleigh(la, s, lab)
                return dict(obj=(ef - sec.E0) / 36, frozenFN_dE=(ef - sec.E0) / 36)
            flat, fh = NT.fit_energy(model, flat, sec, s, mode='fn', la_g=la, steps=fitc['steps'], lr=fitc['lr'],
                                     B=fitc.get('B', 256), seed=1000 * it, refresh=fitc.get('refresh', 500),
                                     warmup=fitc.get('warmup', 50), eval_fn=ev3, hist=fh)
        else:
            flat, fh = NT.fit(model, flat, sec, tlog if base is None else tlog - base, beta=fitc.get('beta', 1.0),
                              steps=fitc['steps'], lr=fitc['lr'],
                              B=fitc.get('B', 512), K=fitc.get('K', 4), lam_amp=fitc.get('lam_amp', 1.0),
                              lam_edge=fitc.get('lam_edge', 1.0), seed=1000 * it, hist=fh, warmup=fitc.get('warmup', 100),
                              images=base is None, samp_log=None if base is None else tlog)
        np.save(os.path.join(OUT, f'params_it{it}.npy'), np.asarray(flat))
        la_new = model.eval_reps(flat, sec.reps)
        if base is not None: la_new = la_new + base
        d = la_new - tlog; mu = jnp.sum(u * u * d)
        proj_err = float(jnp.sqrt(jnp.sum(u * u * (d - mu) ** 2)))
        vnew = sec.vec(la_new, jnp.ones_like(la_new))
        s1, _, kinfo = sec.krylov(vnew, s)
        sc = sec.score(la_new, s1); sc0 = sec.score(la_new, sec.s0)
        rec = dict(it=it, kind='proj', dE_FN_prev=info['dE_FN_site'], fn=info, kry=kinfo,
                   phi_H_dE_oldsign=sc_phi['dE_site'], proj_std_dlog_vs_phi=proj_err, fit_hist=fh,
                   **{f'H_{k}': v for k, v in sc.items()}, H_exactsign_dE=sc0['dE_site'], sec=time.time() - t0)
        la, s = la_new, s1
        hist.append(rec)
        log('PROJ', json.dumps({k: rec[k] for k in ('it', 'dE_FN_prev', 'H_dE_site', 'H_w_s', 'H_std_dlog', 'H_exactsign_dE', 'proj_std_dlog_vs_phi', 'sec')}))
        dump('loop_hist.json', hist)
    # final FN of the last projected guide
    _, _, info = sec.fn_solve(la, s)
    hist.append(dict(it=nit + 1, kind='proj_final_fn', dE_FN_prev=info['dE_FN_site'], fn=info))
    log('PROJ final FN', info['dE_FN_site'])
    dump('loop_hist.json', hist)
    log('DONE loop', time.time() - T00)
