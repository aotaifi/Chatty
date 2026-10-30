"""compare acc_fast.py outputs with the stored f64 reference: acc_fast_compare.py ACC_F64_DIR OUTDIR > txt (writes OUTDIR/acc_fast_summary.json)"""
import sys, os, json
import numpy as np
A, O = sys.argv[1], sys.argv[2]
def gratio(d, Ex, lx, Ey, ly, wi_loc, ux_loc):
    al, E, w = float(d['alpha']), float(d['E']), complex(d['w'])
    fx = 1 + al * (Ex - E); fy = 1 + al * (Ey - E); ph = np.exp(1j * lx.imag)
    cx = (w * ph * fx).real; cy = (w * np.exp(ly - lx.real[ux_loc]) * fy).real
    return cx, cy, cy / cx[ux_loc]
out = {}
for name in ('met', 'uni'):
    f = {m: np.load(f'{O}/{name}_{m}.npz') for m in ('fast', 'ref') if os.path.exists(f'{O}/{name}_{m}.npz')}
    if 'fast' not in f: continue
    d = dict(np.load(f'{A}/{name}.npz')); wi = d['wi']; fa = f['fast']; pi = fa['pairs']; ux = fa['ux']
    loc = np.searchsorted(ux, wi[pi])
    # stored (GPU f64 ref path) on the selected pairs
    al, E, w = float(d['alpha']), float(d['E']), complex(d['w'])
    fx0 = 1 + al * (d['Ex'] - E); fy0 = 1 + al * (d['Ey'] - E); ph0 = np.exp(1j * d['lx'].imag)
    cx0 = (w * ph0 * fx0).real; cy0 = (w * np.exp(d['ly'] - d['lx'].real[wi]) * fy0).real; r0 = (cy0 / cx0[wi])
    npair = len(wi); k = max(1, int(round(0.01 * npair)))
    masks = {}
    for nm, key in (('lowest1pct_cx', np.abs(cx0)[wi]), ('lowest1pct_psiP_x', np.exp(d['lx'].real)[wi]), ('lowest1pct_cy', np.abs(cy0))):
        m = np.zeros(npair, bool); m[np.argsort(key)[:k]] = True; masks[nm] = m[pi]
    masks['random_sample'] = np.ones(len(pi), bool)       # selection + random (union): reported as "evaluated set"
    def stats(r1, rr0, mk):
        e = np.abs(r1 - rr0) / np.abs(rr0)
        return dict(n=int(mk.sum()), max_rel_err=float(np.nanmax(e[mk])), median=float(np.nanmedian(e[mk])), n_nonfinite=int((~np.isfinite(e[mk])).sum()), sign_flips=int((np.sign(r1) != np.sign(rr0))[mk].sum()))
    res = dict(pairs_total=int(npair), pairs_evaluated=int(len(pi)), fallback_fast=int(fa['nfallback']))
    r_fast = gratio(d, fa['Ex'], fa['lx'], fa['Ey'], fa['ly'], None, loc)[2]
    res['fast_vs_storedf64'] = {nm: stats(r_fast, r0[pi], mk) for nm, mk in masks.items()}
    if 'ref' in f:
        rf = f['ref']; r_ref = gratio(d, rf['Ex'], rf['lx'], rf['Ey'], rf['ly'], None, loc)[2]
        res['refCPU_vs_storedf64'] = {nm: stats(r_ref, r0[pi], mk) for nm, mk in masks.items()}
        res['fast_vs_refCPU'] = {nm: stats(r_fast, r_ref, mk) for nm, mk in masks.items()}
        res['ref_flip_cases_in_evaluated'] = int((np.sign(r_ref) != np.sign(r0[pi])).sum())
    out[name] = res
    print(name, json.dumps(res, indent=1))
json.dump(out, open(f'{O}/acc_fast_summary.json', 'w'), indent=1)
