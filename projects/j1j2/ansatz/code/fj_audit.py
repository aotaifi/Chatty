"""fj_audit.py: audit of the Jacobi-factor f_J defect (no-kept rows set to 0; tiny-amplitude rows unbounded) on every augmented guide the pfaffian angle quotes.
fJ_of is r5_bare.py's fJ_of verbatim (globals s1, t1, t2, D, BS); D0M.efn_vec / BS.build verbatim.  K, Vm built once per sign.
Per guide: (0) control E_FN[lF + c* fJ] vs readout e_aug_min_site (rel 1e-8, else stop); (1) rows SK=0 / den<=0, their fJ value, n, base |A|^2 share;
(2) weight share n|A|^2 e^{2 c fJ}: max orbit, top-10 cumulative, base too; (3) max/q.9999 (n-weighted, unweighted)/min of c fJ, #orbits above q;
(4) E_FN with c fJ capped at the n-weighted q.9999; (5) E_FN with no-kept rows lF + c fJ -> lF - 69.   usage: python3 fj_audit.py OUTDIR"""
import os, sys, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
import delta0 as D0M
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
R = '/scratch/a/A.Otaifi/ansatz_runs'
S_1 = f'{R}/pfaffian_z/s1.npy'; S_G = f'{R}/pfaffian_lswo2/s_G.npy'
# (name, sign file, readout json, key path in json, table path or None = from json)
GUIDES = [
 ('lswo_r1', S_1, f'{R}/pfaffian_lswo/readout/readout.json', None, f'{R}/pfaffian_lswo/run/round_1/la_table.npy'),
 ('r2g_g1', S_G, f'{R}/pfaffian_r2g/readout/g1/readout.json', None, None),
 ('r2g_g2', S_G, f'{R}/pfaffian_r2g/readout/g2/readout.json', None, None),
 ('r3_1200', S_G, f'{R}/pfaffian_r3avg/readout/readout.json', 'r3_1200', None),
 ('r3_theta_bar', S_G, f'{R}/pfaffian_r3avg/readout/readout.json', 'theta_bar', None),
 ('r4_theta_4', S_G, f'{R}/pfaffian_r4/readout/readout.json', 'theta_4', None),
 ('r5_pp2_s1', S_G, f'{R}/pfaffian_r5/readout/readout.json', 'pp2_s1', None),
 ('r5_pp2_s2', S_G, f'{R}/pfaffian_r5/readout/readout.json', 'pp2_s2', None)]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RJ = f'{OUT}/fj_audit.json'; res = json.load(open(RJ)) if os.path.exists(RJ) else {}
def dump(): res['sec'] = time.time() - T0; json.dump(res, open(RJ, 'w'), indent=1, default=float)
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
def fJ_of_raw(lF, EP):                    # same lines as fJ_of, without the assert / zeroing; returns fJ (non-finite kept), SK, den
    fJ = np.empty(D); SKa = np.empty(D); dena = np.empty(D)
    for i0 in range(0, D, 65536):
        i1 = min(D, i0 + 65536); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        ss = s1[i0:i1, None] * s1[tt]; blk = valid & (ss > 0); kpt = valid & (ss < 0)
        ratio = np.exp(np.clip((lF - lF.max())[tt] - (lF - lF.max())[i0:i1, None], -700, 700))
        VA = (blk * BS.HW[None, :] * ratio).sum(1); SK = (kpt * BS.HW[None, :] * ratio).sum(1)
        den = 27.0 - 0.5 * (valid * BS.JW[None, :]).sum(1) + VA - EP
        with np.errstate(divide='ignore', invalid='ignore'): fJ[i0:i1] = np.log(SK) - np.log(den)
        SKa[i0:i1] = SK; dena[i0:i1] = den
    return fJ, SKa, dena
def fJ_of(lF, EP):                                                                           # verbatim r4_bare.py (= r3avg_bare.py = r5_bare.py)
    fJ = np.empty(D); nbad = 0; dmin = np.inf
    for i0 in range(0, D, 65536):
        i1 = min(D, i0 + 65536); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        ss = s1[i0:i1, None] * s1[tt]; blk = valid & (ss > 0); kpt = valid & (ss < 0)
        ratio = np.exp(np.clip((lF - lF.max())[tt] - (lF - lF.max())[i0:i1, None], -700, 700))
        VA = (blk * BS.HW[None, :] * ratio).sum(1); SK = (kpt * BS.HW[None, :] * ratio).sum(1)
        den = 27.0 - 0.5 * (valid * BS.JW[None, :]).sum(1) + VA - EP; dmin = min(dmin, float(den.min())); nbad += int((den <= 0).sum() + (SK <= 0).sum())
        with np.errstate(divide='ignore', invalid='ignore'): fJ[i0:i1] = np.log(SK) - np.log(den)
    bad = np.nonzero(~np.isfinite(fJ))[0]; info = dict(den_min=dmin, n_bad=nbad, bad_rows=[int(i) for i in bad]); assert len(bad) <= 3 and dmin > 0
    fJ[bad] = 0.0; return fJ, info
def wq(x, w, q):                           # weighted quantile (lower): smallest x with cumulative weight >= q
    o = np.argsort(x); cw = np.cumsum(w[o]); return float(x[o][min(np.searchsorted(cw, q * cw[-1]), len(x) - 1)])
def shares(lw, nn):                        # lw = log amplitude (guide a = exp(lw)); w = n a^2 normalised
    l2 = 2 * (lw - lw.max()); w = nn * np.exp(l2); w /= w.sum(); top = np.sort(w)[::-1]
    return dict(max=float(top[0]), top10=float(top[:10].sum()), argmax=int(np.argmax(w)))
for sname, sfile in (('s1', S_1), ('sG', S_G)):
    gs = [g for g in GUIDES if g[1] == sfile]
    s1 = np.load(sfile).astype(np.float64); assert set(np.unique(s1)) <= {-1.0, 1.0}
    res.setdefault('signs', {})[sname] = dict(path=sfile, sha256=sha(sfile)); K, Vm, D0 = BS.build(n, s1); log(f'built K,Vm ({sname})')
    for name, _, rj, key, tpath in gs:
        if name in res.get('guides', {}) and res['guides'][name].get('done'): continue
        d = json.load(open(rj))
        if key is None: rec = d; cst = d['cstar']; eaug = d['e_aug_min_site']; EP = d['E_FN_fit']['E']; e_rep = d['e_fit_site']; tp = tpath or d['fit']; tsha = d['fit_sha256']
        else: rec = d['tables'][key]; cst = rec['cstar']; eaug = rec['e_aug_min_site']; EP = rec['E_FN']['E']; e_rep = rec['e']; tp = rec['path']; tsha = rec['sha256']
        assert sha(sfile) == d['s1_sha256'] if 's1_sha256' in d else True
        assert sha(tp) == tsha, f'{name}: table sha mismatch'
        lF = np.load(tp).astype(np.float64); assert lF.shape == (D,) and np.isfinite(lF).all()
        g = res.setdefault('guides', {}).setdefault(name, {}); g.update(sign=sname, readout=rj, table=tp, table_sha256=tsha, cstar=cst, e_aug_reported=eaug, e_base_reported=e_rep, EP=EP)
        fJ, info = fJ_of(lF, EP); raw, SK, den = fJ_of_raw(lF, EP); fin = np.isfinite(raw); assert (raw[fin] == fJ[fin]).all()
        g['fJ_info'] = info; nok = np.nonzero(SK <= 0)[0]; dbad = np.nonzero(den <= 0)[0]
        # (0) control
        t = time.time(); r, _ = D0M.efn_vec(K, Vm, D0, n, lF + cst * fJ); rel = (r['dE_site'] - eaug) / eaug
        g['control'] = dict(e=r['dE_site'], expect=eaug, rel=rel, pass_=bool(abs(rel) <= 1e-8), iters=r['iters'], res=r['res']); dump()
        log(f'{name}: control e {r["dE_site"]:.9e} expect {eaug:.9e} rel {rel:.2e} {time.time()-t:.0f}s')
        if abs(rel) > 1e-8: res['status'] = f'CONTROL FAIL {name}'; dump(); log('CONTROL FAIL: STOP'); sys.exit(2)
        # (1)
        base_w = n * np.exp(2 * (lF - lF.max())); base_w /= base_w.sum()
        g['rows'] = dict(n_SK_zero=int(len(nok)), rows_SK_zero=[int(i) for i in nok], n_den_le0=int(len(dbad)), rows_den_le0=[int(i) for i in dbad], fJ_set_to=0.0,
                         nokept_n=[float(n[i]) for i in nok], nokept_base_A2_share=[float(base_w[i]) for i in nok], nokept_lF_minus_max=[float(lF[i] - lF.max()) for i in nok],
                         nokept_den=[float(den[i]) for i in nok], nonfinite_rows=[int(i) for i in np.nonzero(~fin)[0]])
        # (2)
        cf = cst * fJ; g['share_aug'] = shares(lF + cf, n); g['share_base'] = shares(lF, n)
        a = shares(lF + cf, n)['argmax']; g['top_orbit'] = dict(row=a, n=float(n[a]), lF_minus_max=float(lF[a] - lF.max()), cfJ=float(cf[a]), base_share=float(base_w[a]))
        # (3)
        qw = wq(cf, n, 0.9999); qu = float(np.quantile(cf, 0.9999))
        g['cfJ'] = dict(max=float(cf.max()), argmax=int(cf.argmax()), min=float(cf.min()), q9999_nweighted=qw, q9999_unweighted=qu, n_orbits_above_q_nweighted=int((cf > qw).sum()),
                        n_orbits_above_q_unweighted=int((cf > qu).sum()), n_config_above_q_nweighted=float(n[cf > qw].sum()))
        # (4)
        t = time.time(); r4, _ = D0M.efn_vec(K, Vm, D0, n, lF + np.minimum(cf, qw)); g['capped'] = dict(q=qw, e=r4['dE_site'], diff_vs_reported=r4['dE_site'] - eaug, iters=r4['iters'], res=r4['res'])
        log(f'{name}: capped e {r4["dE_site"]:.9e} diff {r4["dE_site"]-eaug:.3e} {time.time()-t:.0f}s'); dump()
        # (5)
        if len(nok):
            lm = lF + cf; lm[nok] = lF[nok] - 69.0; r5, _ = D0M.efn_vec(K, Vm, D0, n, lm); g['nokept_floor'] = dict(e=r5['dE_site'], diff_vs_reported=r5['dE_site'] - eaug, iters=r5['iters'], res=r5['res'])
            log(f'{name}: no-kept->-69 e {r5["dE_site"]:.9e} diff {r5["dE_site"]-eaug:.3e}')
        else: g['nokept_floor'] = None
        g['flag_2'] = bool(g['share_aug']['max'] > 1e-3); g['flag_4'] = bool(abs(g['capped']['diff_vs_reported']) > 0.05e-6); g['done'] = True; dump()
        log(f'{name}: share_aug max {g["share_aug"]["max"]:.3e} top10 {g["share_aug"]["top10"]:.3e}; cfJ max {cf.max():.3f} q {qw:.3f} min {cf.min():.3f}; flags {g["flag_2"]} {g["flag_4"]}')
    del K, Vm, D0
res['status'] = 'DONE'; res['done'] = True; dump(); log('DONE')
