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
        return jnp.asarray(z['la']), jnp.asarray(z['s']), float(z['phi'])
    t = time.time()
    la, im = model.eval_reps(model.flat0, sec.reps, complex_out=True)
    v = sec.vec(la, jnp.ones_like(la))
    phi = 0.5 * float(jnp.angle(jnp.sum(v * v * jnp.exp(2j * im))))
    s = jnp.where(jnp.cos(im - phi) >= 0, 1.0, -1.0)
    if float(s[0]) < 0: s = -s
    np.savez(p, la=np.asarray(la), s=np.asarray(s, np.int8), phi=phi)
    log(f'ViT tables on {sec.D} reps in {time.time()-t:.0f}s, phi={phi:.4f}')
    return la, s, phi


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
        flat, hist = NT.fit(model, flat, sec, sec.la0, beta=run.get('beta', 1.0), steps=run['steps'], lr=run['lr'],
                            B=run.get('B', 1024), K=run.get('K', 8), lam_amp=run.get('lam_amp', 1.0),
                            lam_edge=run.get('lam_edge', 1.0), seed=run.get('seed', 0),
                            eval_every=run.get('eval_every', 0), eval_fn=ev, hist=hist,
                            warmup=run.get('warmup', 200), opt=run.get('opt', 'adam'))
        np.save(os.path.join(OUT, f'params_{tag}.npy'), np.asarray(flat))
        la = model.eval_reps(flat, sec.reps)
        r = dict(tag=tag, run=run, npar=model.npar, hist=hist, fit_sec=time.time() - t0)
        r.update(full_eval(tag, la, {'exact': sec.s0, 'vit': s_v}, fn=run.get('fn', True),
                           kry_from=s_v if run.get('kry', True) else None))
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

# =============================================================================================== fit
elif mode == 'fit':
    run_fits(SPEC['runs'])
    log('DONE fit', time.time() - T00)

# =============================================================================================== loop
elif mode == 'loop':
    model = NT.Model(SPEC.get('model', {'kind': 'vit'}), CKPT)
    la_v, s_v, phi = vit_tables(NT.Model({'kind': 'vit'}, CKPT)) if SPEC.get('model', {'kind': 'vit'})['kind'] != 'vit' else vit_tables(model)
    flat = model.flat0
    if SPEC.get('init_params'):
        flat = jnp.asarray(np.load(SPEC['init_params']), jnp.float32)
        la_start = model.eval_reps(flat, sec.reps)
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
        flat, fh = NT.fit(model, flat, sec, tlog, beta=fitc.get('beta', 1.0), steps=fitc['steps'], lr=fitc['lr'],
                          B=fitc.get('B', 1024), K=fitc.get('K', 8), lam_amp=fitc.get('lam_amp', 1.0),
                          lam_edge=fitc.get('lam_edge', 1.0), seed=1000 * it, hist=fh, warmup=fitc.get('warmup', 100))
        np.save(os.path.join(OUT, f'params_it{it}.npy'), np.asarray(flat))
        la_new = model.eval_reps(flat, sec.reps)
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
