"""Test S1 (README "Test S1: the ceiling of the Pfaffian's own sign"): cx_o2a.py's FIT-FREE hop+FN chain with the sign as an input.
  la_0 = VMC-trained PP table (|exp(lP)|); sign = 's2' (O2a's oracle sign, refresh s2.npy) or 'own' (sP of the same sym_tables_PP table, the
  code path of cx_b0.py's control: z['sP'].astype(float32), raw, no global-sign flip; E_FN does not depend on a global sign).
  Round k: g_k = |exp(la_k)||1 + alpha_V (E_L - E)| (alpha_V = O1 rule), E_FN(g_k), phi_k = FN solution, la_{k+1} = log phi_k normalised (floor 1e-15).
  Up to NROUND maps (default 15, i.e. evaluations k = 0..15); STOP when |E_FN(g_{k+1})/E_FN(g_k) - 1| < TOL (2%) (the ceiling is then reached).
  Logged per round: E_FN(g_k), alpha_V, c_k = (E_{k+1} - 9.15e-7)/(E_k - 9.15e-7) (O2a definition), ratio E_{k+1}/E_k, and the Aitken estimate
  E_inf = E_c - (E_c - E_b)^2 / ((E_c - E_b) - (E_b - E_a)) from the last three rounds a, b, c.
  Disagreement weights of the sign (|psi0|^2 weights, raw and modulo the global sign) and the expelled-set split (sign right/wrong modulo global).
  SPEC: {"refresh", "pp", "sign": "s2"|"own", "nround" [15], "tol" [0.02], "flip_frac" [0: deliberately flip the sign on this random fraction of
         orbit representatives, seed flip_seed [0]; a deliberately BROKEN-sign control only], "zero_ref" [optional zero-hop E_FN control, rel 1e-3],
         "ref_k" [optional list of reference E_FN(g_k), k = 0.., rel 1e-6 control; exit 2 on failure], "code_sha256": {cx_s1.py, cx_lib.py}}
  python cx_s1.py OUT SPEC.json                                                            Exact 6x6, no training.
"""
import os, sys, time, hashlib
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_s1.json'); T00 = time.time()
CEIL = 9.15e-7; NROUND = int(SPEC.get('nround', 15)); TOL = float(SPEC.get('tol', 0.02)); TAU = 0.1; FLOOR = 1e-15
SIGN = SPEC['sign']; assert SIGN in ('s2', 'own')
res = dict(spec=SPEC, done=False, rounds=[])
def progress(msg):
    log(msg); open(os.path.join(OUT, 'progress.txt'), 'a').write(f'PROGRESS {time.strftime("%H:%M:%S")} {msg}\n')
HERE_ = os.path.dirname(os.path.abspath(__file__)); CODE_FILES = ('cx_s1.py', 'cx_lib.py')
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
z = np.load(TABLE); amp = z['amp']; del z
la0 = norm_la(jnp.log(jnp.maximum(jnp.abs(jnp.asarray(amp)), 1e-300)))
s0 = jnp.where(jnp.asarray(amp) >= 0, 1.0, -1.0).astype(jnp.float32); del amp
W0 = sec.n * jnp.exp(2 * la0); W0 = W0 / jnp.sum(W0)          # |psi0|^2 weight per rep (sums to 1)
NTOT = float(jnp.sum(sec.n)); NREP = float(sec.D)
z = np.load(SPEC['pp']); lpp = norm_la(jnp.asarray(z['lP'], jnp.float64))
sP = jnp.asarray(z['sP'].astype(np.float32)); del z          # as cx_b0.py control
s = s2 if SIGN == 's2' else sP
FLIP = float(SPEC.get('flip_frac', 0.0))
if FLIP > 0:                                                   # deliberately broken sign (control 3 only)
    rng = np.random.default_rng(int(SPEC.get('flip_seed', 0))); m = rng.random(int(sec.D)) < FLIP
    s = jnp.where(jnp.asarray(m), -s, s); res['flip'] = dict(frac=FLIP, n_flipped=int(m.sum()), seed=int(SPEC.get('flip_seed', 0)))
def dis(sg):
    w0 = float(jnp.sum(jnp.where(sg != s0, W0, 0.0))); w2 = float(jnp.sum(jnp.where(sg != s2, W0, 0.0)))
    return dict(w_ne_s0=w0, w_ne_s0_modsign=min(w0, 1 - w0), w_ne_s2=w2, w_ne_s2_modsign=min(w2, 1 - w2))
res['sign_name'] = SIGN; res['disagree'] = dis(s)
progress(f'sign={SIGN} flip={FLIP}  |psi0|^2 disagreement weights (raw / mod global): {res["disagree"]}')
# "right" = agrees with the exact sign s0 modulo the global sign (a global sign is a convention)
RIGHT = (s == s0)
if float(jnp.sum(jnp.where(RIGHT, W0, 0.0))) < 0.5: RIGHT = jnp.logical_not(RIGHT)
res['sign_right_weight'] = float(jnp.sum(jnp.where(RIGHT, W0, 0.0))); dump(F, res)
# ---------------------------------------------------------------- control: zero-hop E_FN of the PP table
e0, _, _ = fn(lpp, s); res['EFN_zero_PP'] = e0
if SPEC.get('zero_ref'):
    rel = abs(e0 - SPEC['zero_ref']) / SPEC['zero_ref']; res['control_zero'] = dict(ref=SPEC['zero_ref'], rel=rel, ok=bool(rel < 1e-3))
    progress(f'CONTROL zero-hop E_FN {e0:.5e} vs {SPEC["zero_ref"]:.5e} (rel {rel:.2e}); ok={res["control_zero"]["ok"]}')
    if not res['control_zero']['ok']:
        res['sec'] = time.time() - T00; res['stop'] = 'CONTROL FAILED'; dump(F, res); progress('CONTROL FAILED: stop'); sys.exit(2)
else:
    progress(f'zero-hop E_FN {e0:.5e} (no reference given)')
dump(F, res)
def hop(la, tag, need_phi):
    """(a)+(b): alpha_V, g, E_FN(g) [, phi, expelled-set diagnostics]  (cx_o2a.hop with the sign s)."""
    t0 = time.time()
    lb = norm_la(la)
    _, V, W = lab.VW(lb, s); v = sec.vec(lb, s); Eb = float(v @ sec.H(v)); del v
    ELb = lab.D0 + V - W; del V, W
    ok_fin = bool(jnp.all(jnp.isfinite(ELb)))
    sdE = float(jnp.sqrt(jnp.sum(sec.vec(lb, s) ** 2 * (ELb - Eb) ** 2)))
    def la_c(a): return lb + jnp.log(jnp.maximum(jnp.abs(1.0 + a * (ELb - Eb)), 1e-30))
    lo, hi = -3.0 / sdE, 3.0 / sdE
    r_ = minimize_scalar(lambda a: float(lab.H_site(la_c(a), s)), bounds=(lo, hi), method='bounded', options=dict(maxiter=100, xatol=1e-4 / sdE))
    aV = float(r_.x); HV = float(r_.fun)
    g = norm_la(la_c(aV)); del ELb
    e, u, info = fn(g, s)
    rec = dict(tag=tag, alpha_V=aV, H_V=HV, EFN=e, sdE=sdE, E_base=Eb, bound=hi, at_bound=bool(abs(abs(aV) - hi) < 1e-3 * hi),
               nfev=r_.nfev, EL_finite=ok_fin, lobpcg_it=info['lobpcg_it'], fn_res=info['res'], neg_weight=info['neg_weight'])
    progress(f'[{tag}] alpha_V {aV:.5f} (bound +-{hi:.3f}, nfev {r_.nfev}) <H>-E0 {HV:.4e} E_FN(g) {e:.5e} (lobpcg {info["lobpcg_it"]}, res {info["res"]:.1e}, EL finite {ok_fin})')
    lphi = norm_la(jnp.log(jnp.maximum(u, 1e-300) / sec.sqrt_n))
    if need_phi:
        X = jnp.exp(lphi - g) < TAU
        wx = jnp.where(X, W0, 0.0)
        rec.update(expelled=dict(tau=TAU, count_frac_rep=float(jnp.sum(X)) / NREP, count_frac_config=float(jnp.sum(sec.n * X)) / NTOT,
                                 w_psi0=float(jnp.sum(wx)), w_psi0_sign_right=float(jnp.sum(jnp.where(RIGHT, wx, 0.0))),
                                 w_psi0_sign_wrong=float(jnp.sum(jnp.where(RIGHT, 0.0, wx)))))
        ex = rec['expelled']
        progress(f'[{tag}] expelled phi/g<{TAU}: count frac (config) {ex["count_frac_config"]:.4e}; |psi0|^2 weight {ex["w_psi0"]:.4e} '
                 f'(sign right {ex["w_psi0_sign_right"]:.4e}, wrong {ex["w_psi0_sign_wrong"]:.4e})')
        rec['n_floored'] = int(jnp.sum(jnp.exp(lphi) < FLOOR))
        lnext = norm_la(jnp.log(jnp.maximum(jnp.exp(lphi), FLOOR)))
        rec['sec'] = time.time() - t0
        return rec, lnext
    rec['sec'] = time.time() - t0
    return rec, lphi
def aitken(E):
    if len(E) < 3: return None
    a, b, c = E[-3:]; d = (c - b) - (b - a)
    return None if d == 0 else c - (c - b) ** 2 / d
la = lpp; stop = None; Es = []
for k in range(NROUND + 1):
    rec, lnext = hop(la, f'k{k}', need_phi=(k < NROUND))
    rec['k'] = k; Es.append(rec['EFN']); rec['aitken'] = aitken(Es)
    if res['rounds']:
        prev = res['rounds'][-1]
        prev['c'] = (rec['EFN'] - CEIL) / (prev['EFN'] - CEIL); prev['ratio'] = rec['EFN'] / prev['EFN']
        progress(f'round {k-1}->{k}: E_FN ratio {prev["ratio"]:.4f}, c_{k-1} (on E-9.15e-7) {prev["c"]:.4f}')
        if abs(prev['ratio'] - 1) < TOL: stop = f'CEILING: |E_{k}/E_{k-1} - 1| = {abs(prev["ratio"]-1):.4f} < {TOL}'
        elif prev['ratio'] > 1: progress(f'NOTE: E_FN rose at round {k-1}->{k} (ratio {prev["ratio"]:.4f}); continuing')
    res['rounds'].append(rec); res['aitken'] = rec['aitken']; res['E_last'] = Es[-1]; dump(F, res)
    if SPEC.get('ref_k') and k < len(SPEC['ref_k']):                    # early control: exit 2 at once, not after 15 rounds
        rk = abs(rec['EFN'] - SPEC['ref_k'][k]) / SPEC['ref_k'][k]
        progress(f'CONTROL round {k}: E_FN {rec["EFN"]:.8e} vs {SPEC["ref_k"][k]:.8e} (rel {rk:.2e})')
        if rk >= 1e-6: res['stop'] = 'CONTROL FAILED'; res['sec'] = time.time() - T00; dump(F, res); progress('CONTROL FAILED: exit 2'); sys.exit(2)
    if rec['aitken'] is not None: progress(f'[k{k}] Aitken estimate from the last three rounds: {rec["aitken"]:.5e}')
    if stop: progress(f'VERDICT {stop}: stop'); break
    la = lnext
ref = SPEC.get('ref_k')
if ref:
    rels = [abs(res['rounds'][i]['EFN'] - r) / r for i, r in enumerate(ref) if i < len(res['rounds'])]
    res['control_ref_k'] = dict(ref=ref, rel=rels, ok=bool(len(rels) == len(ref) and max(rels) < 1e-6)); progress(f'CONTROL ref_k: rel {rels}; ok={res["control_ref_k"]["ok"]}')
res['stop'] = stop; res['sec'] = time.time() - T00; res['done'] = True; dump(F, res); progress(f'DONE {res["sec"]:.0f}')
if ref and not res['control_ref_k']['ok']: sys.exit(2)
