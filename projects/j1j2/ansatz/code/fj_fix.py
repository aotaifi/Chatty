"""fj_fix.py: fixed Jacobi factor (PLAN_pfaffian.md, "FIX").  Raw f_J = the arithmetic of r5_bare.fJ_of verbatim (loop over 65536-row blocks), without the zeroing of non-finite rows.
fJ_of_fixed(lF, EP, sign, mode, c, f0=None) -> (add, info): the guide is lF + add, add = c f_J (the c-multiplied term).
  mode 'old': non-finite -> 0, add = c f_J (= fJ_of, used for the unit check);  'a1': no-kept (non-finite) rows -> add = -69 - 0 (guide lF - 69), others c f_J;
  'a2': non-finite rows -> f_J = f_min (unweighted 1e-4 quantile of the finite f_J);  'b1': a2 and add = c sign(f_J) min(|f_J|, f0) (= c_eff f_J, c_eff = c min(1, f0/|f_J|));
  'b2': a2 and add = c f0 tanh(f_J/f0).  f0 = n-weighted 0.9999 quantile of |f_J| over the finite rows unless given (np.inf -> no cap).
weight_check(lF_aug, n) -> (max single-orbit share, top-10 share) of n exp(2 lF_aug); raises if max >= 1e-3.
init(n) must be called once (orbit multiplicities); BS tables are read here."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS
_t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); _t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r'); D = len(_t1)
def raw_fJ(lF, EP, sign):
    s1 = np.asarray(sign, dtype=np.float64); fJ = np.empty(D); den_min = np.inf
    for i0 in range(0, D, 65536):
        i1 = min(D, i0 + 65536); t = np.concatenate([_t1[i0:i1], _t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        ss = s1[i0:i1, None] * s1[tt]; blk = valid & (ss > 0); kpt = valid & (ss < 0)
        ratio = np.exp(np.clip((lF - lF.max())[tt] - (lF - lF.max())[i0:i1, None], -700, 700))
        VA = (blk * BS.HW[None, :] * ratio).sum(1); SK = (kpt * BS.HW[None, :] * ratio).sum(1)
        den = 27.0 - 0.5 * (valid * BS.JW[None, :]).sum(1) + VA - EP; den_min = min(den_min, float(den.min()))
        with np.errstate(divide='ignore', invalid='ignore'): fJ[i0:i1] = np.log(SK) - np.log(den)
    return fJ, den_min
def wquantile(x, w, q):
    o = np.argsort(x); cw = np.cumsum(w[o]); return float(x[o][min(np.searchsorted(cw, q * cw[-1]), len(x) - 1)])
def apply_mode(raw, n, mode, c, f0=None):
    bad = ~np.isfinite(raw); fin = raw[~bad]; info = dict(mode=mode, c=c, n_nonfinite=int(bad.sum()))
    if mode == 'old': f = np.where(bad, 0.0, raw); info['f0'] = None; return c * f, info
    if mode == 'a1': add = c * np.where(bad, 0.0, raw); add[bad] = -69.0; info['f0'] = None; return add, info
    fmin = float(np.quantile(fin, 1e-4)); f = np.where(bad, fmin, raw); info['f_min'] = fmin
    if mode == 'a2': info['f0'] = None; return c * f, info
    if f0 is None: f0 = wquantile(np.abs(fin), n[~bad], 0.9999)
    info['f0'] = float(f0)
    if mode == 'b1': return c * np.sign(f) * np.minimum(np.abs(f), f0), info
    if mode == 'b2': return (c * f if np.isinf(f0) else c * f0 * np.tanh(f / f0)), info
    raise ValueError(mode)
def fJ_of_fixed(lF, EP, sign, mode, c, n, f0=None):
    raw, dmin = raw_fJ(lF, EP, sign); assert dmin > 0; add, info = apply_mode(raw, n, mode, c, f0); info['den_min'] = dmin; return add, info
def weight_check(lF_aug, n):
    l2 = 2 * (lF_aug - lF_aug.max()); w = n * np.exp(l2); w /= w.sum(); top = np.sort(w)[::-1]; mx = float(top[0]); t10 = float(top[:10].sum())
    if mx >= 1e-3: raise ValueError(f'single-orbit share {mx:.3e} >= 1e-3 (row {int(np.argmax(w))}); top10 {t10:.3e}')
    return mx, t10
