"""Stage 4: referee on the stored two-hop data (ref.npz; no ViT evaluations).
For b = a exp(f) with the guide sign s (fixed): exact local energy E_L^b(x) = D(x) + sum_{valid y} J/2 s_x s_y a_y/a_x
exp(f_y - f_x) over ALL valid neighbours; samples x ~ a^2, weights exp(2 f(x)) (self-normalised).  Paired difference
Delta = <H>_b - <H>_a with SE from per-chain means of the influence d_i = w_i (E_L^b_i - E_b) - (E_L^a_i - E_a).
Also: the unprojected semi-implicit target b_T = a exp(T) (diagnostic: gain contained in the target), one Lanczos step on
the guide (moments <E_L>, <E_L^2>, <E_L (H^2 psi)/psi>), absolute energies.
  python l8_eval.py OUT SPEC.json   SPEC: ref (path or list of paths), runs: [{dir, tags, steps: ['best','final',10000]}]
"""
import sys, os, json, time
import numpy as np
import l8_core as C
import jax, jax.numpy as jnp
from l8_net import Corr, feats

OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'eval.json'); N = C.N; T00 = time.time()
refs = SPEC['ref'] if isinstance(SPEC['ref'], list) else [SPEC['ref']]
parts = [dict(np.load(p)) for p in refs]
# concatenate several referee files (independent seeds): offset owners and chains
Z = {}; offx = 0; offc = 0
for z in parts:
    z = dict(z); z['own'] = z['own'] + offx; z['chain'] = z['chain'] + offc
    offx += len(z['X']); offc += int(z['chain'].max()) + 1
    for k, v in z.items():
        if np.ndim(v) == 0: Z[k] = v; continue
        Z.setdefault(k, []).append(v)
Z = {k: (np.concatenate(v) if isinstance(v, list) else v) for k, v in Z.items()}
R = len(Z['X']); own = Z['own']; ch = Z['chain']
JBe = C.JB[Z['bond'].astype(np.int64)]
sxy = Z['x_s'][own] * Z['y_s']
rxy = np.exp(np.clip(Z['y_la'] - Z['x_la'][own], -60, 60))
hxy = 0.5 * JBe * sxy * rxy                                         # H_xy psi_y / psi_x (guide sign)
D = Z['x_D']; ELa = Z['x_ELa']
chk = D + np.bincount(own, weights=hxy, minlength=R)
res = dict(spec=SPEC, R=R, edges=len(own), nchains=int(len(np.unique(ch))),
           check_ELa_maxdiff=float(np.max(np.abs(chk - ELa))))
C.log('R', R, 'edges', len(own), 'E_L check', res['check_ELa_maxdiff'])


def chain_mean_se(d):
    ids, inv = np.unique(ch, return_inverse=True)
    m = np.bincount(inv, weights=d) / np.bincount(inv)
    return float(d.mean()), float(m.std(ddof=1) / np.sqrt(len(ids)))


Ea, Ea_se = chain_mean_se(ELa)
res['guide'] = dict(E_site=Ea / N, se=Ea_se / N, sd_EL_site=float(ELa.std() / N))
ec, ec_se = chain_mean_se(Z['x_ELc_re'])
res['vit_complex'] = dict(E_site=ec / N, se=ec_se / N)
C.log('guide <H>', res['guide'], 'ViT complex', res['vit_complex'])


def score(fx, fy, tag):
    """paired referee for b = a exp(f): fx on samples, fy on edges."""
    ELb = D + np.bincount(own, weights=hxy * np.exp(np.clip(fy - fx[own], -60, 60)), minlength=R)
    lw = 2 * fx; w = np.exp(lw - lw.max()); w = w / w.mean()
    Eb = float(np.sum(w * ELb) / np.sum(w))
    d = w * (ELb - Eb) - (ELa - Ea)
    m, se = chain_mean_se(d)
    Delta = Eb - Ea
    _, se_b = chain_mean_se(w * (ELb - Eb))
    out = dict(tag=tag, Delta_site=Delta / N, Delta_se_site=se / N, z=Delta / se if se > 0 else float('nan'),
               E_site=Eb / N, E_se_site=se_b / N, ess_frac=float(w.sum() ** 2 / (w * w).sum() / R),
               f_sd=float(fx.std()), df_edge_rms=float(np.sqrt(np.mean((fy - fx[own]) ** 2))))
    C.log('SCORE', json.dumps(out))
    return out, d


# ---------------------------------------------------------------- unprojected semi-implicit target (diagnostic)
tx = Z['x_T']; ty = Z['y_T']
res['target_SI'], _ = score(tx - tx.mean(), ty - tx.mean(), 'SI target exp(T), unprojected')
for lam_ in (0.25, 0.5):
    res[f'target_SI_x{lam_}'], _ = score(lam_ * (tx - tx.mean()), lam_ * (ty - tx.mean()), f'exp({lam_} T)')

# ---------------------------------------------------------------- one Lanczos step on the guide (binarised sign)
ELy = Z['y_ELa']
EL2 = D * ELa + np.bincount(own, weights=hxy * ELy, minlength=R)      # (H^2 psi)(x) / psi(x)


def lanczos(idx):
    h1 = ELa[idx].mean(); h2 = (ELa[idx] ** 2).mean(); h3 = (ELa[idx] * EL2[idx]).mean()
    # E(al) = (h1 + 2 al h2 + al^2 h3) / (1 + 2 al h1 + al^2 h2); stationary points: quadratic in al
    # dE/dal = 0  <=>  (h2 + al h3)(1 + 2 al h1 + al^2 h2) - (h1 + 2 al h2 + al^2 h3)(h1 + al h2) = 0
    #         <=>  al^2 (h3 h1 - h2^2) + al (h3 - h1 h2) + (h2 - h1^2) = 0
    A = h3 * h1 - h2 * h2; Bq = h3 - h1 * h2; Cq = h2 - h1 * h1
    roots = np.roots([A, Bq, Cq]); roots = roots[np.isreal(roots)].real
    E = lambda al: (h1 + 2 * al * h2 + al * al * h3) / (1 + 2 * al * h1 + al * al * h2)
    al = min(roots, key=E) if len(roots) else 0.0
    return E(al), al, h2 - h1 * h1


El, al, var = lanczos(np.arange(R))
ids = np.unique(ch); jk = []
for blk in np.array_split(ids, 32):
    jk.append(lanczos(np.nonzero(~np.isin(ch, blk))[0])[0])
jk = np.array(jk); se_l = float(np.sqrt((len(jk) - 1) * np.mean((jk - jk.mean()) ** 2)))
# paired: Lanczos minus guide, jackknife
jk_g = np.array([ELa[~np.isin(ch, blk)].mean() for blk in np.array_split(ids, 32)])
dj = jk - jk_g; se_dl = float(np.sqrt((len(dj) - 1) * np.mean((dj - dj.mean()) ** 2)))
res['lanczos1'] = dict(E_site=El / N, se=se_l / N, alpha=float(al), Delta_site=(El - Ea) / N, Delta_se_site=se_dl / N,
                       var_site2=var / N ** 2, cost_note='free given the two-hop referee data (same 2-hop evaluations)')
C.log('LANCZOS', res['lanczos1'])
C.dump(F, res)

# ---------------------------------------------------------------- arms
SPX = jnp.asarray(C.bits_to_spins_np(Z['X'])); Ys = Z['Y']
res['arms'] = []; infl = {}
for run in SPEC.get('runs', []):
    tr = json.load(open(os.path.join(run['dir'], 'train.json')))
    for arm in tr['arms']:
        if run.get('tags') and arm['tag'] not in run['tags']: continue
        P_ = np.load(os.path.join(run['dir'], f'params_{arm["tag"]}.npz'))
        stats = [tuple(s) for s in P_['stats']]
        model = Corr(32, 4, seed=0)
        fsym = jax.jit(model.f)
        FXr = jnp.asarray(feats(Z['x_la'], Z['x_V'], Z['x_W'], stats))
        steps = []
        for s in run.get('steps', ['best', 'final']):
            st = int(P_['best']) if s == 'best' else (arm['final_step'] if s == 'final' else int(s))
            if f's{st}' in P_ and st not in [x[1] for x in steps]: steps.append((s, st))
        for lab, st in steps:
            flat = jnp.asarray(P_[f's{st}'])
            t0 = time.time()
            fx = np.concatenate([np.asarray(fsym(flat, SPX[i:i + 4096], FXr[i:i + 4096])) for i in range(0, R, 4096)]).astype(float)
            fy = np.empty(len(own)); bs = 8192
            FYa = feats(Z['y_la'], Z['y_V'], Z['y_W'], stats)
            for i in range(0, len(own), bs):
                sp = jnp.asarray(C.bits_to_spins_np(Ys[i:i + bs])); ft = jnp.asarray(FYa[i:i + bs])
                if len(sp) < bs:
                    pad = bs - len(sp); sp = jnp.concatenate([sp, jnp.repeat(sp[:1], pad, 0)]); ft = jnp.concatenate([ft, jnp.repeat(ft[:1], pad, 0)])
                fy[i:i + bs] = np.asarray(fsym(flat, sp, ft))[:min(bs, len(own) - i)]
            out, d = score(fx, fy, f'{arm["tag"]}@{lab}({st})')
            out.update(arm=arm['tag'], kind=arm['kind'], opt=arm['opt'], step=st, label=lab, run=run['dir'], sec=time.time() - t0)
            res['arms'].append(out); infl[(arm['tag'], lab)] = (out['Delta_site'], d)
            C.dump(F, res)
# paired FN vs control differences at the pre-registered checkpoint ('best')
pairs = []
fn_tags = [k for k in infl if k[1] == 'best' and any(a['arm'] == k[0] and a['kind'] == 'fn' for a in res['arms'])]
vm_tags = [k for k in infl if k[1] == 'best' and any(a['arm'] == k[0] and a['kind'] == 'vmc' for a in res['arms'])]
for kf in fn_tags:
    for kv in vm_tags:
        _, se = chain_mean_se(infl[kf][1] - infl[kv][1])
        pairs.append(dict(fn=kf[0], vmc=kv[0], diff_site=infl[kf][0] - infl[kv][0], se_site=se / N))
res['pairs'] = pairs; res['sec'] = time.time() - T00
C.dump(F, res); C.log('DONE', res['sec'])
