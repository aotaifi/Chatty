"""Test O2b (README "De-oracling the G1 guide" O2, Amendment O2-1 item 2): the FITTED loop. O2a (cx_o2a.py, 2cf5250) with a fit
in the middle of each round.  Arm PP + 3x3 backflow, start = the VMC PP params with zero backflow (table = sym_tables_PP, no
oracle).  Sign s_2 (oracle, refresh_17409419).  Round k = 0 .. NROUND-1:
  (a) g_k = |a_k| |1 + alpha_V (E_L - E)|, E_L of (|a_k|, s_2), alpha_V = O1 rule (bounded argmin of the exact <H> of the hop state);
  (b) phi_k = FN solution of sec.fn_solve(g_k, s_2); E_FN(g_k); expelled set {phi_k/g_k < 0.1} split by s_2 == s0 (as O2a);
      E_FN(hop(phi_k)) = the O2a quantity (the exact chain's next value from the same phi_k), computed here;
  (c) fit a_{k+1} to log phi_k (normalised, floored at 1e-15 as O2a): cx_t1s.py as a subprocess, arm ppbf, warm from a_k's theta
      (round 0: the VMC PP params with zero backflow), decade-stratified on the decades of phi_k, B = B_acc = 8208, tau
      0.05 -> 0.01, fit_sec per round; full table of a_{k+1}; its readout gives the per-decade MEAN and rms fit error against
      phi_k, E_FN(a_{k+1}) (zero hop) and the splice (fitted >= -9, phi_k below, zero hop).
  Then g_{k+1} = hop(a_{k+1}) opens round k+1 and closes round k:
      c_k = (E_FN(g_{k+1}) - 9.15e-7) / (E_FN(g_k) - 9.15e-7);
      fit tax_k = E_FN(hop(a_{k+1})) - E_FN(hop(phi_k)); exact gain_k = E_FN(g_k) - E_FN(hop(phi_k));
      viability_k (Amendment O2-1 item 5, recorded only): fit tax_k < 0.5 exact gain_k.
  Stop rules (pre-registered): RISE (c_k > 1), STALL (two consecutive c_k > 0.9 while E_FN(g_{k+1}) > 1.5e-5).  After the last
  round, one more hop of a_NROUND closes c_{NROUND-1}.
Controls: zero-hop E_FN of the PP table = 5.3677e-5 (T0); E_FN(g_0) and E_FN(hop(phi_0)) = O2a rounds 0 and 1
  (2.369190e-5, 8.106143e-6; o2a_17430739).  Rel > 1e-3 on any of them: stop before the first fit (exit 2).
Checkpoints: every round's fit lives in OUT/round_k (cx_t1s.json, la_table.npy, theta.npy, target_la.npy); spec 'resume_from'
  (a previous O2b run dir) reuses its finished rounds' fits instead of refitting (the hops are recomputed, they are exact).
  python cx_o2b.py OUT SPEC.json
SPEC: refresh, pp_params, pp_table, code_sha256, nround (5), fit_sec (2400), [resume_from], [dry, fit_overrides]
"""
import os, sys, time, hashlib, subprocess
import numpy as np
from scipy.optimize import minimize_scalar
from cx_lib import *
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
F = os.path.join(OUT, 'cx_o2b.json'); T00 = time.time()
CEIL = 9.15e-7; TAU_X = 0.1; FLOOR = 1e-15
NROUND = int(SPEC.get('nround', 5)); FIT_SEC = float(SPEC.get('fit_sec', 2400)); DRY = bool(SPEC.get('dry', False))
FIT_SEC0 = float(SPEC.get('fit_sec_round0', FIT_SEC))   # Amendment O2b-1: round 0 = the one-time initial training (10800 s)
FN_MAXIT = 2 if DRY else 400; A_MAXIT = 3 if DRY else 100
O2A = dict(EFN_g0=2.369190165069707e-05, EFN_g1=8.106142726936759e-06, EFN_zero_PP=5.3677e-05)   # o2a_17430739, T0
res = dict(spec=SPEC, done=False, rounds=[], design=dict(nround=NROUND, fit_sec=FIT_SEC, fit_sec_round0=FIT_SEC0, ceil=CEIL, tau_expel=TAU_X, floor=FLOOR))


def progress(msg):
    log(msg); open(os.path.join(OUT, 'progress.txt'), 'a').write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}\n')


HERE_ = os.path.dirname(os.path.abspath(__file__))
CODE_FILES = ('cx_o2b.py', 'cx_t1s.py', 'cx_lib.py', 'rbmpp_lib.py')
SHA_RUN = {fn: (hashlib.sha256(open(os.path.join(HERE_, fn), 'rb').read()).hexdigest()
                if os.path.exists(os.path.join(HERE_, fn)) else None) for fn in CODE_FILES}
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
    _, u, info = sec.fn_solve(la, s, maxit=FN_MAXIT); return info['dE_FN_site'], u, info


RF = SPEC['refresh']
s2 = jnp.asarray(np.load(os.path.join(RF, 's2.npy')).astype(np.float32))
z = np.load(TABLE); amp = z['amp']; del z
la0 = norm_la(jnp.log(jnp.maximum(jnp.abs(jnp.asarray(amp)), 1e-300)))
s0 = jnp.where(jnp.asarray(amp) >= 0, 1.0, -1.0).astype(jnp.float32); del amp
W0 = sec.n * jnp.exp(2 * la0); W0 = W0 / jnp.sum(W0); del la0
RIGHT = (s2 == s0); NTOT = float(jnp.sum(sec.n)); NREP = float(sec.D); del s0
z = np.load(SPEC['pp_table']); lpp = norm_la(jnp.asarray(z['lP'], jnp.float64)); del z


def hop(la, tag, need_phi):
    """(a)+(b) of cx_o2a.hop: alpha_V, g, E_FN(g) [, log phi normalised + floored, expelled-set diagnostics]."""
    t0 = time.time()
    lb = norm_la(la)
    _, V, W = lab.VW(lb, s2); v = sec.vec(lb, s2); Eb = float(v @ sec.H(v)); del v
    ELb = lab.D0 + V - W; del V, W
    ok_fin = bool(jnp.all(jnp.isfinite(ELb)))
    sdE = float(jnp.sqrt(jnp.sum(sec.vec(lb, s2) ** 2 * (ELb - Eb) ** 2)))

    def la_c(a): return lb + jnp.log(jnp.maximum(jnp.abs(1.0 + a * (ELb - Eb)), 1e-30))
    lo, hi = -3.0 / sdE, 3.0 / sdE
    r_ = minimize_scalar(lambda a: float(lab.H_site(la_c(a), s2)), bounds=(lo, hi), method='bounded',
                         options=dict(maxiter=A_MAXIT, xatol=1e-4 / sdE))
    aV = float(r_.x); HV = float(r_.fun)
    g = norm_la(la_c(aV)); del ELb
    e, u, info = fn(g, s2)
    rec = dict(tag=tag, alpha_V=aV, H_V=HV, EFN=e, sdE=sdE, E_base=Eb, bound=hi, at_bound=bool(abs(abs(aV) - hi) < 1e-3 * hi),
               nfev=r_.nfev, EL_finite=ok_fin, lobpcg_it=info['lobpcg_it'], fn_res=info['res'], neg_weight=info['neg_weight'])
    progress(f'[{tag}] alpha_V {aV:.5f} (nfev {r_.nfev}) <H>-E0 {HV:.4e} E_FN {e:.4e} (lobpcg {info["lobpcg_it"]}, res {info["res"]:.1e})')
    lnext = None
    if need_phi:
        lphi = norm_la(jnp.log(jnp.maximum(u, 1e-300) / sec.sqrt_n))
        X = jnp.exp(lphi - g) < TAU_X; wx = jnp.where(X, W0, 0.0)
        rec['expelled'] = dict(tau=TAU_X, count_frac_rep=float(jnp.sum(X)) / NREP, count_frac_config=float(jnp.sum(sec.n * X)) / NTOT,
                               w_psi0=float(jnp.sum(wx)), w_psi0_s2_eq_s0=float(jnp.sum(jnp.where(RIGHT, wx, 0.0))),
                               w_psi0_s2_ne_s0=float(jnp.sum(jnp.where(RIGHT, 0.0, wx))))
        rec['n_floored'] = int(jnp.sum(jnp.exp(lphi) < FLOOR))
        lnext = norm_la(jnp.log(jnp.maximum(jnp.exp(lphi), FLOOR))); del lphi, X, wx
        ex = rec['expelled']
        progress(f'[{tag}] expelled: config frac {ex["count_frac_config"]:.3e}, |psi0|^2 w {ex["w_psi0"]:.3e} '
                 f'(s2==s0 {ex["w_psi0_s2_eq_s0"]:.3e}, s2!=s0 {ex["w_psi0_s2_ne_s0"]:.3e})')
    del u, g
    rec['sec'] = time.time() - t0
    return rec, lnext


# ---------------------------------------------------------------- controls
e0, _, _ = fn(lpp, s2); rel0 = abs(e0 - O2A['EFN_zero_PP']) / O2A['EFN_zero_PP']
res['control'] = dict(EFN_zero_PP=e0, ref_zero_PP=O2A['EFN_zero_PP'], rel_zero_PP=rel0); dump(F, res)
progress(f'CONTROL zero-hop PP E_FN {e0:.6e} vs {O2A["EFN_zero_PP"]:.4e} (rel {rel0:.1e})')


def ctl_fail(msg):
    res['control']['ok'] = False; res['stop'] = 'CONTROL FAILED: ' + msg; res['sec'] = time.time() - T00; res['done'] = True
    dump(F, res); progress(res['stop']); sys.exit(2)


if not DRY and rel0 > 1e-3: ctl_fail('zero-hop PP')


def fit_round(k, target_path, warm_theta):
    """(c): cx_t1s.py subprocess (or a finished round of resume_from)."""
    rp = SPEC.get('resume_from')
    if rp and os.path.exists(os.path.join(rp, f'round_{k}', 'cx_t1s.json')):
        rd = os.path.join(rp, f'round_{k}'); j = json.load(open(os.path.join(rd, 'cx_t1s.json')))
        if j.get('done'):
            progress(f'[round {k}] fit reused from {rd}'); return rd, j, 0, True
    rd = os.path.join(OUT, f'round_{k}'); os.makedirs(rd, exist_ok=True)
    sp = dict(arm='ppbf', refresh=RF, pp_params=SPEC['pp_params'], pp_table=SPEC['pp_table'], target_la=target_path,
              seed=1 + k, B=8208, K=2, B_acc=8208, B_ho=4096, lam_b=1.0, lam0=0.01, lam_cap=1.0, tau=0.05, tau_min=0.01,
              anneal_steps=20, z_acc=2.0, ho_every=10, min_steps=50, min_window=50, flat_tol=0.02, min_move=1e-3, splice_dec=-9,
              fit_sec=FIT_SEC0 if k == 0 else FIT_SEC, skip_control2=True,
              code_sha256={f: SHA_EXP[f] for f in ('cx_t1s.py', 'cx_lib.py', 'rbmpp_lib.py')})
    if warm_theta: sp['warm_theta'] = warm_theta
    sp.update(SPEC.get('fit_overrides', {}))
    spf = os.path.join(rd, 'spec_fit.json'); json.dump(sp, open(spf, 'w'), indent=1)
    sec.offload()                                    # free the device copy of H while the fit job holds the GPU
    t0 = time.time()
    with open(os.path.join(rd, 'fit.log'), 'w') as lf:
        rc = subprocess.run([sys.executable, os.path.join(HERE_, 'cx_t1s.py'), rd, spf], cwd=HERE_, stdout=lf, stderr=subprocess.STDOUT).returncode
    sec.reload()
    j = json.load(open(os.path.join(rd, 'cx_t1s.json'))) if os.path.exists(os.path.join(rd, 'cx_t1s.json')) else {}
    progress(f'[round {k}] fit subprocess exit {rc} in {time.time() - t0:.0f}s')
    return rd, j, rc, False


def fit_record(rd, j, rc, reused):
    out = dict(dir=rd, exit=rc, reused=reused, done=bool(j.get('done')))
    if not out['done']: return out
    rb = j['readout']['rms_by_decade']
    out.update(converged=j['converged'], E_FN_zero_hop=j.get('E_FN'), E_FN_splice=j.get('E_FN_splice'),
               tau_hist=j['fit'].get('tau_hist'), steps=j['converged'].get('steps'), fit_sec=j['converged'].get('fit_sec'),
               table_sec=j.get('table_sec'),
               err_by_decade={d: dict(mean=rb[d]['mean_count'], rms=rb[d]['rms_count'], rms_T0=rb[d]['rms_T0'],
                                      rms_heldout=rb[d]['rms_count_heldout']) for d in rb},
               offset_count=j['readout']['offset_count'], bond_error_total=j['readout']['bond_error']['total'])
    return out


# ---------------------------------------------------------------- the loop
la = lpp; theta_prev = None; stop = None; k = 0
for k in range(NROUND + 1):
    last = k == NROUND
    recA, lnext = hop(la, f'g{k}', need_phi=not last)
    rec = dict(k=k, hop_a=recA, EFN=recA['EFN'])
    if k == 0:
        r_ = abs(recA['EFN'] - O2A['EFN_g0']) / O2A['EFN_g0']; res['control'].update(EFN_g0=recA['EFN'], rel_g0=r_)
        progress(f'CONTROL E_FN(g_0) {recA["EFN"]:.6e} vs O2a {O2A["EFN_g0"]:.6e} (rel {r_:.1e})')
        if not DRY and r_ > 1e-3: ctl_fail('E_FN(g_0) vs O2a')
    if res['rounds']:                                 # close round k-1
        p = res['rounds'][-1]
        p['c'] = (recA['EFN'] - CEIL) / (p['EFN'] - CEIL)
        p['fit_tax'] = recA['EFN'] - p['EFN_hop_phi']; p['exact_gain'] = p['EFN'] - p['EFN_hop_phi']
        p['viability_tax_lt_half_gain'] = bool(p['fit_tax'] < 0.5 * p['exact_gain'])
        progress(f'round {k - 1}: c = {p["c"]:.4f}  fit tax {p["fit_tax"]:.3e}  exact gain {p["exact_gain"]:.3e}')
        if p['c'] > 1.0:
            stop = f'RISE: c_{k - 1} = {p["c"]:.4f} > 1'
        elif len(res['rounds']) >= 2 and res['rounds'][-2].get('c', 0) > 0.9 and p['c'] > 0.9 and recA['EFN'] > 1.5e-5:
            stop = f'STALL: c_{k - 2} = {res["rounds"][-2]["c"]:.4f}, c_{k - 1} = {p["c"]:.4f} > 0.9 with E_FN {recA["EFN"]:.3e} > 1.5e-5'
    res['rounds'].append(rec); dump(F, res)
    if stop or last:
        if stop: progress(f'VERDICT {stop}: stop')
        break
    recB, _ = hop(lnext, f'hop_phi{k}', need_phi=False)
    rec['hop_phi'] = recB; rec['EFN_hop_phi'] = recB['EFN']
    if k == 0:
        r_ = abs(recB['EFN'] - O2A['EFN_g1']) / O2A['EFN_g1']; res['control'].update(EFN_hop_phi0=recB['EFN'], rel_hop_phi0=r_)
        progress(f'CONTROL E_FN(hop(phi_0)) {recB["EFN"]:.6e} vs O2a round 1 {O2A["EFN_g1"]:.6e} (rel {r_:.1e})')
        if not DRY and r_ > 1e-3: ctl_fail('E_FN(hop(phi_0)) vs O2a')
        res['control']['ok'] = True
    dump(F, res)
    tp = os.path.join(OUT, f'round_{k}'); os.makedirs(tp, exist_ok=True); tp = os.path.join(tp, 'target_la.npy')
    np.save(tp, np.asarray(lnext, np.float64)); del lnext
    rd, j, rc, reused = fit_round(k, tp, theta_prev)
    rec['fit'] = fit_record(rd, j, rc, reused); dump(F, res)
    if not rec['fit']['done']:
        stop = f'FIT FAILED in round {k} (exit {rc}; see {rd}/fit.log)'; progress(stop); break
    f_ = rec['fit']
    progress(f'[round {k}] fit: converged {f_["converged"]["converged"]} ({f_["steps"]} steps), rms(-11) {f_["err_by_decade"]["-11"]["rms"]:.4f} '
             f'mean(-11) {f_["err_by_decade"]["-11"]["mean"]:+.4f}  E_FN(a_k+1) {f_["E_FN_zero_hop"]:.4e}  splice {f_["E_FN_splice"]:.4e}')
    la = jnp.asarray(np.load(os.path.join(rd, 'la_table.npy')), jnp.float64); theta_prev = os.path.join(rd, 'theta.npy')
res['stop'] = stop; res['sec'] = time.time() - T00; res['done'] = True; dump(F, res)
progress(f'DONE {res["sec"]:.0f}s  stop={stop}')
if stop and stop.startswith('FIT FAILED'): sys.exit(4)
