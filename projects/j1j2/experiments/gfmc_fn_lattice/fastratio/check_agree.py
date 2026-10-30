"""Fast vs ref bond ratios on fixed configurations (item 1 of the fastratio build; CPU or GPU, f64).
  python check_agree.py OUT.json --params P.npz --cfg cfg.npz --set met --n 10000 [--break sign|noCS] [--shard i/k] [--tol 1e-10]
For every configuration x and every antiparallel bond b: (i) the per-image ratios Psi(z_g^b)/Psi(z_g) (img_quant), (ii) the projected
psi_P(y_b)/psi_P(x) (eloc_full), (iii) the p = 0 guide ratio c(y)/c(x) = Re[w e^{i Im l_x} rP]/Re[w e^{i Im l_x}] (w = -1 as production).
Relative error |fast - ref| / |ref|, sign flips (Re of ratio; for the complex ratios only counted where |Re| > 1e-6 |ratio|; guide ratio: all),
reported for all pairs and for the 1% hardest pairs (smallest |psi_P(x)|, smallest |psi_P(y)|, smallest |c(x)|)."""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from rbmpp_lib import Lattice, make_fns
ap = argparse.ArgumentParser(); ap.add_argument('out')
ap.add_argument('--params', required=True); ap.add_argument('--cfg', required=True); ap.add_argument('--set', default='met')
ap.add_argument('--n', type=int, default=10000); ap.add_argument('--L', type=int, default=8); ap.add_argument('--proj', default='translations')
ap.add_argument('--break', dest='brk', default=None, choices=[None, 'sign', 'noCS']); ap.add_argument('--shard', default='0/1')
ap.add_argument('--chunk', type=int, default=16); ap.add_argument('--tol', type=float, default=1e-10); ap.add_argument('--w', type=float, default=-1.0)
a = ap.parse_args()
si, sk = map(int, a.shard.split('/'))
z = np.load(a.params); lat = Lattice(a.L, 0.5, a.proj)
p = {k: jnp.asarray(z[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}
fr = make_fns(lat, jnp.float64, True, ratio='ref'); ff = make_fns(lat, jnp.float64, True, ratio='fast', _break=a.brk)
BI = np.asarray(lat.BI); BJ = np.asarray(lat.BJ)


def both(fns):
    def one(p, sigma):
        f, b, W = fns['parts'](p)
        Z = sigma.astype(jnp.float64)[lat.IMG] * lat.FL[:, None]
        lg, rat, kap = jax.vmap(lambda zz, ig: fns['img_quant_k'](f, b, W, zz, ig))(Z, lat.INV)
        e, lP, rP, bad = fns['eloc_full_flag'](p, sigma)
        return lg, rat, lP, rP, bad, kap
    return jax.jit(jax.vmap(one, in_axes=(None, 0)))
fR, fF = both(fr), both(ff)
X = np.load(a.cfg)[a.set][:a.n]
idx = np.arange(len(X))[si::sk]
rows = dict(kmap={}, kmax=[], kimg=[], ierr_k=[], cid=[], bid=[], rr=[], rf=[], img_err=[], rp_err=[], rp_flip=[], g_err=[], g_flip=[], lpx=[], lpy=[], cx=[], bad=0, nconf=0, img_flip=[], tot=0)
t0 = time.time()
for s0 in range(0, len(idx), a.chunk):
    ii = idx[s0:s0 + a.chunk]; Xb = jnp.asarray(X[ii].astype(np.float64))
    gR = [np.asarray(v) for v in fR(p, Xb)]; gF = [np.asarray(v) for v in fF(p, Xb)]
    lgR, ratR, lPR, rPR, badR, kapR = gR; lgF, ratF, lPF, rPF, badF, kapF = gF
    rows['bad'] += int(badF.sum()); rows['nconf'] += len(ii); rows['kmax'].append(kapF.max(1)); rows['kmap'].update({int(k): float(v) for k, v in zip(ii, kapF.max(1))}); rows['kimg'].append(kapF.ravel())
    for c, k in enumerate(ii):
        anti = X[k][BI] != X[k][BJ]
        okR = np.isfinite(rPR[c]).all() and np.isfinite(rPF[c]).all()
        if not okR or badF[c]: continue                  # flagged configurations (singular / kappa > kappa_max) take the direct fallback: not part of the fast-vs-ref comparison
        rr = rPR[c][anti]; rf = rPF[c][anti]
        rows['rp_err'].append(np.abs(rf - rr) / np.abs(rr)); rows['cid'].append(np.full(rr.shape, k)); rows['bid'].append(np.nonzero(anti)[0]); rows['rr'].append(rr); rows['rf'].append(rf)
        sig = np.abs(rr.real) > 1e-6 * np.abs(rr)
        rows['rp_flip'].append(((np.sign(rf.real) != np.sign(rr.real)) & sig))
        ph = np.exp(1j * lPR[c].imag); cx = (a.w * ph).real
        gr = (a.w * ph * rr).real / cx; gf = (a.w * ph * rf).real / cx
        rows['g_err'].append(np.abs(gf - gr) / np.abs(gr)); rows['g_flip'].append(np.sign(gf) != np.sign(gr))
        rows['lpx'].append(np.full(rr.shape, lPR[c].real)); rows['lpy'].append(lPR[c].real + np.log(np.abs(rr))); rows['cx'].append(np.full(rr.shape, abs(cx)))
        ir = ratR[c][:, anti]; if_ = ratF[c][:, anti]
        m = np.isfinite(ir).all(0) & np.isfinite(if_).all(0)
        e = np.abs(if_ - ir) / np.abs(ir)
        rows['img_err'].append(e[:, m].max(0) if m.any() else np.zeros(0)); rows['ierr_k'].append(np.stack([np.broadcast_to(kapF[c][:, None], e.shape)[:, m].ravel(), e[:, m].ravel()]))
        rows['img_flip'].append((np.sign(if_.real) != np.sign(ir.real) )[:, m].sum() * np.ones(1))
    if (s0 // a.chunk) % 50 == 0: print(f'{a.set} shard {a.shard} {s0 + len(ii)}/{len(idx)} sec {time.time() - t0:.0f}', flush=True)
cat = lambda k: np.concatenate(rows[k]) if rows[k] else np.zeros(0)
rp_err, rp_flip, g_err, g_flip = cat('rp_err'), cat('rp_flip'), cat('g_err'), cat('g_flip')
lpx, lpy, cx, img_err = cat('lpx'), cat('lpy'), cat('cx'), cat('img_err')
k1 = max(1, int(round(0.01 * len(rp_err))))
masks = dict(all=np.ones(len(rp_err), bool))
for name, key in (('lowest1pct_psiP_x', lpx), ('lowest1pct_psiP_y', lpy), ('lowest1pct_cx', cx)):
    m = np.zeros(len(rp_err), bool); m[np.argsort(key)[:k1]] = True; masks[name] = m
kk = np.concatenate(rows['kimg']); km = np.concatenate(rows['kmax']); ik = np.concatenate(rows['ierr_k'], 1) if rows['ierr_k'] else np.zeros((2, 0))
kb = {}
for lo, hi in ((0, 1e2), (1e2, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 1e8)):
    m = (ik[0] >= lo) & (ik[0] < hi); kb[f'kappa_{lo:.0e}_{hi:.0e}'] = dict(n=int(m.sum()), max_img_rel_err=float(ik[1][m].max()) if m.any() else None)
res = dict(kappa=dict(image_median=float(np.median(kk)), image_max=float(kk.max()), config_max_median=float(np.median(km)), frac_images_gt1e6=float((kk > 1e6).mean()),
                      frac_configs_with_image_gt1e6=float((km > 1e6).mean()), frac_configs_gt1e8=float((km > 1e8).mean())), img_err_by_kappa=kb, args=vars(a), n_configs=rows['nconf'], n_pairs=int(len(rp_err)), n_bad_configs_flagged=rows['bad'],
           img_ratio_max_rel_err=float(img_err.max()) if img_err.size else None, img_ratio_sign_flips=int(np.sum([float(v.sum()) for v in rows['img_flip']])),
           tol=a.tol)
for name, m in masks.items():
    res[name] = dict(n=int(m.sum()), rP_max_rel_err=float(rp_err[m].max()), rP_median=float(np.median(rp_err[m])), rP_sign_flips=int(rp_flip[m].sum()),
                     guide_max_rel_err=float(g_err[m].max()), guide_sign_flips=int(g_flip[m].sum()))
kcfg = np.array([rows['kmap'][int(c)] for c in cat('cid')]) if len(rp_err) else np.zeros(0)
res['rP_err_by_config_max_kappa'] = {}
for lo, hi in ((0, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 1e7), (1e7, 1e8)):
    m = (kcfg >= lo) & (kcfg < hi)
    res['rP_err_by_config_max_kappa'][f'{lo:.0e}_{hi:.0e}'] = dict(n_pairs=int(m.sum()), n_configs=int(len(np.unique(cat('cid')[m]))), rP_max_rel_err=float(rp_err[m].max()) if m.any() else None,
                                                                    guide_sign_flips=int(g_flip[m].sum()), rP_sign_flips=int(rp_flip[m].sum()))
w = np.argsort(rp_err)[::-1][:30]
res['worst_rP_pairs'] = [dict(config=int(cat('cid')[i]), bond=int(cat('bid')[i]), rel_err=float(rp_err[i]), config_max_image_kappa=rows['kmap'][int(cat('cid')[i])], ref=[float(cat('rr')[i].real), float(cat('rr')[i].imag)], fast=[float(cat('rf')[i].real), float(cat('rf')[i].imag)], log_abs_psiP_x=float(lpx[i]), log_abs_psiP_y=float(lpy[i])) for i in w]
res['PASS'] = bool(all(res[n]['rP_max_rel_err'] <= a.tol and res[n]['rP_sign_flips'] == 0 and res[n]['guide_sign_flips'] == 0 for n in masks) and (res['img_ratio_max_rel_err'] or 0) <= a.tol)
res['sec'] = time.time() - t0
json.dump(res, open(a.out, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'args'}, indent=1))
