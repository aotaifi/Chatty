"""Test O2a (README "De-oracling the G1 guide" O2 + Amendment O2-1 item 1): the FIT-FREE reference chain.
  la_0 = VMC-trained PP table; sign s_2 (oracle, refresh_17409419). Round k = 0..4:
    (a) g_k = |exp(la_k)| |1 + alpha_V (E_L - E)|, E_L of (|exp(la_k)|, s_2), alpha_V = O1 rule (bounded argmin of exact <H> of the one-hop state);
    (b) E_FN(g_k) and phi_k = the FN solution (u / sqrt_n) of sec.fn_solve(g_k, s_2);
    (c) la_{k+1} = log phi_k, normalised (floor 1e-15 of the normalised amplitude, as fn_solve floors its guide).
  A sixth evaluation (alpha_V, E_FN of g_5) closes c_4. c_k = (E_FN(g_{k+1}) - 9.15e-7) / (E_FN(g_k) - 9.15e-7); STOP at once if any c_k > 1 (RISE).
  Per round: expelled set {phi_k/g_k < 0.1}: count fraction, |psi0|^2 weight, split by s_2 == s0 / != s0.
  Control: zero-hop E_FN of the PP table must reproduce 5.3677e-5 (T0).      python cx_o2a.py OUT SPEC.json
  SPEC: {"refresh", "pp", "code_sha256"}   Exact 6x6, no training.
"""
import os, sys, time, hashlib
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_o2a.json'); T00 = time.time()
CEIL = 9.15e-7; NROUND = 5; TAU = 0.1; FLOOR = 1e-15
res = dict(spec=SPEC, done=False, rounds=[])
def progress(msg):
    log(msg); open(os.path.join(OUT, 'progress.txt'), 'a').write(f'PROGRESS {time.strftime("%H:%M:%S")} {msg}\n')
HERE_ = os.path.dirname(os.path.abspath(__file__)); CODE_FILES = ('cx_o2a.py', 'cx_lib.py')
SHA_RUN = {fn: (hashlib.sha256(open(os.path.join(HERE_, fn), 'rb').read()).hexdigest() if os.path.exists(os.path.join(HERE_, fn)) else None) for fn in CODE_FILES}
SHA_EXP = SPEC.get('code_sha256'); res['code_sha256'] = dict(expected=SHA_EXP, running=SHA_RUN, dir=HERE_)
bad = [fn for fn in CODE_FILES if (SHA_EXP or {}).get(fn) != SHA_RUN[fn]]
res['code_sha256']['match'] = not bad
if bad:
    res['code_sha256']['mismatch'] = bad; dump(F, res); progress(f'CODE DRIFT: sha256 mismatch for {bad}; exit 3'); sys.exit(3)
dump(F, res)
lab = Lab(); sec = lab.sec
def norm_la(la):
    a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))
def fn(la, s):
    _, u, info = sec.fn_solve(la, s); return info['dE_FN_site'], u, info
RF = SPEC['refresh']
s2 = jnp.asarray(np.load(os.path.join(RF, 's2.npy')).astype(np.float32))
# exact psi0 (as cx_g0.py; Lab nulls sec.la0/s0/v0, so read the table again)
z = np.load(TABLE); amp = z['amp']; del z
la0 = norm_la(jnp.log(jnp.maximum(jnp.abs(jnp.asarray(amp)), 1e-300)))
s0 = jnp.where(jnp.asarray(amp) >= 0, 1.0, -1.0).astype(jnp.float32); del amp
W0 = sec.n * jnp.exp(2 * la0); W0 = W0 / jnp.sum(W0)          # |psi0|^2 weight per rep (sums to 1)
RIGHT = (s2 == s0); NTOT = float(jnp.sum(sec.n)); NREP = float(sec.D)
progress(f's2==s0 weight of |psi0|^2: {float(jnp.sum(jnp.where(RIGHT, W0, 0.0))):.6f}')
z = np.load(SPEC['pp']); lpp = norm_la(jnp.asarray(z['lP'], jnp.float64)); del z
# ---------------------------------------------------------------- control: zero-hop E_FN of the PP table = 5.3677e-5 (T0)
e0, _, _ = fn(lpp, s2); rel = abs(e0 - 5.3677e-5) / 5.3677e-5
res['control'] = dict(EFN_zero_PP=e0, ref=5.3677e-5, rel=rel, ok=bool(rel < 1e-3)); dump(F, res)
progress(f'CONTROL PP zero-hop E_FN {e0:.4e} vs 5.3677e-5 (rel {rel:.2e}); ok={res["control"]["ok"]}')
if not res['control']['ok']:
    res['sec'] = time.time() - T00; dump(F, res); progress('CONTROL FAILED: stop'); sys.exit(2)
def hop(la, tag, need_phi):
    """(a)+(b): alpha_V, g, E_FN(g) [, phi, expelled-set diagnostics]."""
    t0 = time.time()
    lb = norm_la(la)
    _, V, W = lab.VW(lb, s2); v = sec.vec(lb, s2); Eb = float(v @ sec.H(v)); del v
    ELb = lab.D0 + V - W; del V, W
    ok_fin = bool(jnp.all(jnp.isfinite(ELb)))
    sdE = float(jnp.sqrt(jnp.sum(sec.vec(lb, s2) ** 2 * (ELb - Eb) ** 2)))
    def la_c(a): return lb + jnp.log(jnp.maximum(jnp.abs(1.0 + a * (ELb - Eb)), 1e-30))
    lo, hi = -3.0 / sdE, 3.0 / sdE
    r_ = minimize_scalar(lambda a: float(lab.H_site(la_c(a), s2)), bounds=(lo, hi), method='bounded', options=dict(maxiter=100, xatol=1e-4 / sdE))
    aV = float(r_.x); HV = float(r_.fun)
    g = norm_la(la_c(aV)); del ELb
    e, u, info = fn(g, s2)
    rec = dict(tag=tag, alpha_V=aV, H_V=HV, EFN=e, sdE=sdE, E_base=Eb, bound=hi, at_bound=bool(abs(abs(aV) - hi) < 1e-3 * hi),
               nfev=r_.nfev, EL_finite=ok_fin, lobpcg_it=info['lobpcg_it'], fn_res=info['res'], neg_weight=info['neg_weight'])
    progress(f'[{tag}] alpha_V {aV:.5f} (bound +-{hi:.3f}, nfev {r_.nfev}) <H>-E0 {HV:.4e} E_FN(g) {e:.4e} (lobpcg {info["lobpcg_it"]}, res {info["res"]:.1e}, EL finite {ok_fin})')
    lphi_raw = jnp.log(jnp.maximum(u, 1e-300) / sec.sqrt_n); lphi = norm_la(lphi_raw)
    if need_phi:
        X = jnp.exp(lphi - g) < TAU
        wx = jnp.where(X, W0, 0.0)
        rec.update(expelled=dict(tau=TAU, count_frac_rep=float(jnp.sum(X)) / NREP, count_frac_config=float(jnp.sum(sec.n * X)) / NTOT,
                                 w_psi0=float(jnp.sum(wx)), w_psi0_s2_eq_s0=float(jnp.sum(jnp.where(RIGHT, wx, 0.0))),
                                 w_psi0_s2_ne_s0=float(jnp.sum(jnp.where(RIGHT, 0.0, wx)))))
        ex = rec['expelled']
        progress(f'[{tag}] expelled phi/g<{TAU}: count frac (config) {ex["count_frac_config"]:.4e} (rep {ex["count_frac_rep"]:.4e}); |psi0|^2 weight {ex["w_psi0"]:.4e} '
                 f'(s2==s0 {ex["w_psi0_s2_eq_s0"]:.4e}, s2!=s0 {ex["w_psi0_s2_ne_s0"]:.4e})')
        # la_{k+1} = log phi_k normalised, floored at FLOOR of the normalised amplitude (as the solver floors its guide)
        amax = float(jnp.max(jnp.exp(lphi))); nfl = int(jnp.sum(jnp.exp(lphi) < FLOOR))
        rec['n_floored'] = nfl
        lnext = norm_la(jnp.log(jnp.maximum(jnp.exp(lphi), FLOOR)))
        rec['sec'] = time.time() - t0
        return rec, lnext
    rec['sec'] = time.time() - t0
    return rec, lphi
la = lpp; stop = None
for k in range(NROUND + 1):
    rec, lnext = hop(la, f'k{k}', need_phi=(k < NROUND))
    rec['k'] = k
    if res['rounds']:
        prev = res['rounds'][-1]
        prev['c'] = (rec['EFN'] - CEIL) / (prev['EFN'] - CEIL)
        progress(f'c_{k-1} = (E_FN(g_{k}) - {CEIL:.3g}) / (E_FN(g_{k-1}) - {CEIL:.3g}) = {prev["c"]:.4f}')
        if prev['c'] > 1.0:
            stop = f'RISE at c_{k-1} = {prev["c"]:.4f}'
    res['rounds'].append(rec); dump(F, res)
    if stop:
        progress(f'VERDICT {stop}: stop'); break
    la = lnext
    if k == NROUND - 1: np.save(os.path.join(OUT, 'la_5.npy'), np.asarray(la, np.float32))
res['stop'] = stop; res['sec'] = time.time() - T00; res['done'] = True; dump(F, res); progress(f'DONE {res["sec"]:.0f}')
