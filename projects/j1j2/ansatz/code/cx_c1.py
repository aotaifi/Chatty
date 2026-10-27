"""Test C driver = cx_f1.py (sha 587cef6f...) with: files cx_c1.py/cx_c1fit.py; options --complex --warm_theta P --im_sigma S --seed N (also as spec keys);
  tax_0 = E_FN(copy, zero hop) - E_FN(phi_1, zero hop) with E_FN(phi_1, zero hop) = 9.85e-6 (PLAN_pfaffian.md Test C; REF key EFN_phi1_zero_hop) reported next to F1's one-hop tax.
Test F1 (README "Test F1": is the refit tax capacity or under-training?).  ONE round-1 fit of the O2b loop, as cx_o2b.py does it:
  target  = log phi_1 = O2b round_1/target_la.npy;  warm start = a_1 = O2b round_0/theta.npy (3x3 backflow; embedded into 5x5 for window 5);
  fit     = cx_f1fit.py (a copy of cx_t1s.py + options bf_window / freeze_outer / max_steps), spec = O2b round_1/spec_fit.json with
            fit_sec and the options replaced (B 8208, K 2, tau 0.05 -> 0.01, seed 2, ... unchanged);
  tax     = E_FN(hop(a_2)) - E_FN(hop(phi_1)), both through the SAME hop() as cx_o2b.py (alpha_V = O1 rule, g = a|1 + alpha_V (E_L - E)|,
            one fn_solve); reference E_FN(hop(phi_1)) = 4.951226e-6 (o2b_17431886 round 1 'EFN_hop_phi'), O2b round-1 tax 5.93021e-6 (80 steps).
  Readings: tax, steps, s/step, held-out loss curve (res fit.curve in round_1/cx_t1s.json), per-decade mean/rms error (readout of the fit),
            bond error, E_FN of a_2 zero hop and splice.
Controls (stop before the fit, exit 2): E_FN(hop(phi_1)) vs 4.951226e-6 (rel 1e-3); optional check_o2b_table: E_FN(hop(O2b round_1 la_table)) vs
  1.0881439e-5 (= O2b E_FN(g_2)).  Reported after the fit: tax vs expect_tax (rel 5%) when the spec gives it.
rate_only: no Lab, no hops: one fit subprocess that stops after max_steps and reports s/step.
  python cx_f1.py OUT SPEC.json
SPEC: refresh, pp_params, pp_table, code_sha256 {cx_f1.py, cx_f1fit.py, cx_lib.py, rbmpp_lib.py}, target_la, warm_theta, window (3|5), freeze_outer,
  fit_sec, [max_steps], [rate_only], [o2b_table], [expect_tax], [fit_overrides], [break_embed]
"""
import os, sys, time, hashlib, subprocess, json
import numpy as np
OUT = sys.argv[1]; SPEC = json.load(open(sys.argv[2])); os.makedirs(OUT, exist_ok=True)
_a = sys.argv[3:]; _i = 0                       # C1 options: --complex --warm_theta P --im_sigma S --seed N (override the spec)
while _i < len(_a):
    if _a[_i] == '--complex': SPEC['complex'] = True; _i += 1
    elif _a[_i] == '--warm_theta': SPEC['warm_theta'] = _a[_i + 1]; _i += 2
    elif _a[_i] == '--im_sigma': SPEC['im_sigma'] = float(_a[_i + 1]); _i += 2
    elif _a[_i] == '--seed': SPEC.setdefault('fit_overrides', {})['seed'] = int(_a[_i + 1]); SPEC['fit_overrides']['im_seed'] = int(_a[_i + 1]); _i += 2
    else: raise SystemExit(f'unknown option {_a[_i]}')
HERE_ = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(OUT, 'cx_c1.json'); T00 = time.time()
RATE = bool(SPEC.get('rate_only', False))
REF = dict(EFN_hop_phi1=4.951226482141635e-06, EFN_g2_o2b=1.0881438700119065e-05, tax_o2b_round1=5.93021221797743e-06, EFN_phi1_zero_hop=9.85e-06)   # o2b_17431886 cx_o2b.json
CEIL = 9.15e-7; FLOOR = 1e-15
res = dict(spec=SPEC, done=False, ref=REF)


def progress(msg):
    print(time.strftime('[%H:%M:%S]'), msg, flush=True)
    open(os.path.join(OUT, 'progress.txt'), 'a').write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}\n')


CODE_FILES = ('cx_c1.py', 'cx_c1fit.py', 'cx_lib.py', 'rbmpp_lib.py')
SHA_RUN = {fn: hashlib.sha256(open(os.path.join(HERE_, fn), 'rb').read()).hexdigest() for fn in CODE_FILES}
SHA_EXP = SPEC.get('code_sha256') or {}
bad = [fn for fn in CODE_FILES if SHA_EXP.get(fn) != SHA_RUN[fn]]
res['code_sha256'] = dict(expected=SHA_EXP, running=SHA_RUN, match=not bad, dir=HERE_)
json.dump(res, open(F, 'w'), indent=1, default=float)
if bad:
    progress(f'CODE DRIFT: sha256 mismatch for {bad}; exit 3'); sys.exit(3)

if not RATE:
    from cx_lib import *
    lab = Lab(); sec = lab.sec
    FN_MAXIT = 400; A_MAXIT = 100
    from scipy.optimize import minimize_scalar

    def norm_la(la):
        a = jnp.exp(la - jnp.max(la)); return la - jnp.max(la) - 0.5 * jnp.log(jnp.sum(sec.n * a * a))

    s2 = jnp.asarray(np.load(os.path.join(SPEC['refresh'], 's2.npy')).astype(np.float32))

    def hop(la, tag):
        """cx_o2b.hop (a)+(b) without the expelled-set diagnostics: alpha_V (O1 rule), g, E_FN(g)."""
        t0 = time.time(); lb = norm_la(la)
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
        _, u, info = sec.fn_solve(g, s2, maxit=FN_MAXIT); e = info['dE_FN_site']; del u, g
        rec = dict(tag=tag, alpha_V=aV, H_V=HV, EFN=e, sdE=sdE, E_base=Eb, bound=hi, at_bound=bool(abs(abs(aV) - hi) < 1e-3 * hi), nfev=r_.nfev,
                   EL_finite=ok_fin, lobpcg_it=info['lobpcg_it'], fn_res=info['res'], neg_weight=info['neg_weight'], sec=time.time() - t0)
        progress(f'[{tag}] alpha_V {aV:.5f} (nfev {r_.nfev}) <H>-E0 {HV:.4e} E_FN {e:.6e} (lobpcg {info["lobpcg_it"]}, res {info["res"]:.1e}) {rec["sec"]:.0f}s')
        return rec

    def ctl_fail(msg):
        res['stop'] = 'CONTROL FAILED: ' + msg; res['sec'] = time.time() - T00; res['done'] = True
        json.dump(res, open(F, 'w'), indent=1, default=float); progress(res['stop']); sys.exit(2)

    res['control'] = {}
    la_t = jnp.asarray(np.load(SPEC['target_la']), jnp.float64)
    recR = hop(la_t, 'hop_phi1'); del la_t
    r_ = abs(recR['EFN'] - REF['EFN_hop_phi1']) / REF['EFN_hop_phi1']
    res['hop_phi1'] = recR; res['control']['rel_hop_phi1'] = r_
    progress(f'CONTROL E_FN(hop(phi_1)) {recR["EFN"]:.6e} vs O2b {REF["EFN_hop_phi1"]:.6e} (rel {r_:.1e})')
    json.dump(res, open(F, 'w'), indent=1, default=float)
    if r_ > 1e-3: ctl_fail('E_FN(hop(phi_1)) vs O2b')
    if SPEC.get('o2b_table'):
        rec0 = hop(jnp.asarray(np.load(SPEC['o2b_table']), jnp.float64), 'hop_o2b_a2')
        r_ = abs(rec0['EFN'] - REF['EFN_g2_o2b']) / REF['EFN_g2_o2b']
        res['hop_o2b_a2'] = rec0; res['control']['rel_hop_o2b_a2'] = r_
        progress(f'CONTROL E_FN(hop(O2b a_2)) {rec0["EFN"]:.6e} vs O2b E_FN(g_2) {REF["EFN_g2_o2b"]:.6e} (rel {r_:.1e})')
        json.dump(res, open(F, 'w'), indent=1, default=float)
        if r_ > 1e-3: ctl_fail('E_FN(hop(O2b a_2)) vs O2b')

# ---------------------------------------------------------------- the fit (subprocess, as cx_o2b.fit_round)
rd = os.path.join(OUT, 'round_1'); os.makedirs(rd, exist_ok=True)
sp = json.load(open(SPEC['o2b_fit_spec']))                      # O2b round_1/spec_fit.json: every fit knob as O2b ran it
for k in ('refresh', 'pp_params', 'pp_table', 'target_la', 'warm_theta'): sp[k] = SPEC[k]
sp.update(fit_sec=float(SPEC['fit_sec']), bf_window=int(SPEC['window']), freeze_outer=bool(SPEC.get('freeze_outer', False)), skip_control2=True,
          code_sha256={f: SHA_EXP[f] for f in ('cx_c1fit.py', 'cx_lib.py', 'rbmpp_lib.py')})
for k in ('max_steps', 'rate_only', 'break_embed', 'complex', 'im_sigma'):
    if k in SPEC: sp[k] = SPEC[k]
sp.update(SPEC.get('fit_overrides', {}))
spf = os.path.join(rd, 'spec_fit.json'); json.dump(sp, open(spf, 'w'), indent=1)
res['fit_spec'] = sp
if not RATE: sec.offload()
t0 = time.time()
with open(os.path.join(rd, 'fit.log'), 'w') as lf:
    rc = subprocess.run([sys.executable, os.path.join(HERE_, 'cx_c1fit.py'), rd, spf], cwd=HERE_, stdout=lf, stderr=subprocess.STDOUT).returncode
res['fit_exit'] = rc; res['fit_wall_sec'] = time.time() - t0
progress(f'fit subprocess exit {rc} in {time.time() - t0:.0f}s')
jf = os.path.join(rd, 'cx_t1s.json')
j = json.load(open(jf)) if os.path.exists(jf) else {}
if not j.get('done') or rc != 0:
    res['stop'] = f'FIT FAILED (exit {rc}; see {rd}/fit.log)'; res['done'] = True; json.dump(res, open(F, 'w'), indent=1, default=float); progress(res['stop']); sys.exit(4)
hist = j['fit']['hist']; secs = [h['sec'] for h in hist]
res['fit'] = dict(steps=j['converged']['steps'], fit_sec=j['converged']['fit_sec'], converged=j['converged']['converged'], reason=j['converged']['reason'],
                  median_sec_per_step=float(np.median(secs)), mean_sec_per_step=float(np.mean(secs)), tau_hist=j['fit'].get('tau_hist'),
                  acceptance_all=j['converged'].get('acceptance_all'), L_ho_curve=[(q['step'], q['L_ho']) for q in j['fit']['curve']],
                  L_ho_step0=j['fit']['curve'][0]['L_ho'], npar=j.get('npar'), n_active=j.get('n_active'))
if RATE:
    res['rate'] = j['rate']; res['done'] = True; res['sec'] = time.time() - T00; json.dump(res, open(F, 'w'), indent=1, default=float)
    progress(f'DONE rate_only {j["rate"]}'); sys.exit(0)
rb = j['readout']['rms_by_decade']
res['fit']['err_by_decade'] = {d: dict(mean=rb[d]['mean_count'], rms=rb[d]['rms_count'], rms_heldout=rb[d]['rms_count_heldout']) for d in rb}
res['fit'].update(offset_count=j['readout']['offset_count'], bond_error_total=j['readout']['bond_error']['total'],
                  bond_error_warm_total=j['readout']['bond_error_warm']['total'], E_FN_zero_hop=j.get('E_FN'), E_FN_splice=j.get('E_FN_splice'), table_sec=j.get('table_sec'))
json.dump(res, open(F, 'w'), indent=1, default=float)
sec.reload()
la2 = jnp.asarray(np.load(os.path.join(rd, 'la_table.npy')), jnp.float64)
recA = hop(la2, 'hop_a2'); res['hop_a2'] = recA
res['tax'] = recA['EFN'] - recR['EFN']; res['tax_vs_ref_hop_phi1'] = recA['EFN'] - REF['EFN_hop_phi1']
res['tax_0'] = res['fit']['E_FN_zero_hop'] - REF['EFN_phi1_zero_hop']; res['tax_0_note'] = 'E_FN(copy, zero hop) - 9.85e-6'
res['tax_o2b_round1'] = REF['tax_o2b_round1']; res['tax_rel_to_o2b'] = res['tax'] / REF['tax_o2b_round1'] - 1
if SPEC.get('expect_tax'):
    res['control']['expect_tax'] = SPEC['expect_tax']; res['control']['tax_rel_dev'] = abs(res['tax'] / SPEC['expect_tax'] - 1)
    res['control']['tax_within_5pct'] = bool(res['control']['tax_rel_dev'] <= 0.05)
res['sec'] = time.time() - T00; res['done'] = True; json.dump(res, open(F, 'w'), indent=1, default=float)
progress(f'FIT TAX {res["tax"]:.4e} = E_FN(hop(a_2)) {recA["EFN"]:.6e} - E_FN(hop(phi_1)) {recR["EFN"]:.6e}; O2b round 1 {REF["tax_o2b_round1"]:.4e} '
         f'(rel {res["tax_rel_to_o2b"]:+.3f}); steps {res["fit"]["steps"]}, {res["fit"]["median_sec_per_step"]:.2f} s/step')
progress(f'tax_0 {res["tax_0"]:.4e} (E_FN zero hop {res["fit"]["E_FN_zero_hop"]:.4e}); one-hop tax {res["tax"]:.4e}')
progress(f'DONE {res["sec"]:.0f}s')
