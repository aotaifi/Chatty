"""m5p2x_gate.py (= m5p2_gate.py for EXTENSION SEGMENT 1: fit steps 600, tail average theta 1200..2100; thresholds unchanged): mechanical reading of the M5' readout against PLAN_pfaffian.md "M5'" (thresholds only; the lead decides).
  python3 m5p_gate.py rel   READOUT_JSON    relax phase: e_rel <= 1.25e-5 -> PASS (go to the round) / FAIL (relax 1000 more steps, at most twice, then stop and report)
  python3 m5p_gate.py accept READOUT_JSON   after a PASS: writes pfaffian_m5p/relax_accepted.json and copies the phase's theta_rel.npy to pfaffian_m5p/theta_rel.npy (sha-checked)
  python3 m5p_gate.py final [READOUT_JSON]  both registered readouts, each as registered, no combination:
     (1) r_5(M5') = (e_rel - e_end)/(e_rel - 4.367093e-6): structure matters iff 0.4929 - r > 0.15
     (2) aug_min(M5', fixed f_J b1) - 3.2584e-6 > 0.3e-6: structure matters
  Also prints the checks: control, relative weight check, fit steps."""
import sys, json, os, shutil, hashlib
Q = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_m5p2'; sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
mode = sys.argv[1]; RJ = sys.argv[2] if len(sys.argv) > 2 else f'{Q}/readout/readout.json'; d = json.load(open(RJ)); assert d.get('done'), 'readout not done'
assert d['control']['pass_'], 'control failed'
if mode in ('rel', 'accept'):
    x = d['relax']; ok = x['pass_']
    print(f"{x['phase']} [{x['cluster']}] steps {x['fit_steps']} e_rel {x['e_rel']:.6e} (limit 1.25e-5; theta_4 {x['control_e_theta_4']:.6e}) control rel {d['control']['rel']:.1e}")
    print(f"VERDICT RELAX: {'PASS' if ok else 'FAIL'}")
    if mode == 'accept':
        assert ok, 'not accepted: e_rel above the limit'
        assert sha(x['theta_rel']) == x['theta_rel_sha256'] and x['theta_equals_fit_final']
        shutil.copy(x['theta_rel'], f'{Q}/theta_rel.npy'); assert sha(f'{Q}/theta_rel.npy') == x['theta_rel_sha256']
        json.dump(dict(phase=x['phase'], e_rel=x['e_rel'], table=x['table'], table_sha256=x['table_sha256'], theta_rel=f'{Q}/theta_rel.npy', theta_rel_sha256=x['theta_rel_sha256'], readout=RJ), open(f'{Q}/relax_accepted.json', 'w'), indent=1)
        print(f'accepted: {Q}/relax_accepted.json, {Q}/theta_rel.npy sha {x["theta_rel_sha256"][:12]}')
    sys.exit(0 if ok else 1)
V = d['verdict_numbers']; r = V['r_5_M5p']; ea = V['e_aug_b1']; a = d['aug_b1']
print(f"control rel {d['control']['rel']:.1e}; ext1 fit steps {d['theta']['fit_steps']}{'' if d['theta']['fit_steps'] == 600 else ' (NOT 600)'}; theta_5 cluster vs local max|diff| {d['theta']['max_abs_diff_cluster_vs_local']:.1e}")
print(f"e_rel {V['e_rel']:.6e}  e_end {V['e_end']:.6e}  r_5(M5') {r:.4f} (full-digit floor {V['r_5_M5p_fulldigits']:.4f})  aug_min {ea:.6e} at c* {V['cstar_b1']} (grid {sorted(float(k) for k in a['scan'])}; relative weight check ok for all c: {a['rel_weight_ok_all']})")
v1 = 0.4929 - r > 0.15; v2 = ea - 3.2584e-6 > 0.3e-6
print(f"READOUT (1): 0.4929 - r = {0.4929 - r:+.4f} (> 0.15: {v1}) -> {'STRUCTURE MATTERS' if v1 else 'PARAMETER COUNT (structure does not matter)'}")
print(f"READOUT (2): aug_min - 3.2584e-6 = {(ea - 3.2584e-6) * 1e6:+.4f}e-6 (> 0.3e-6: {v2}) -> {'STRUCTURE MATTERS' if v2 else 'PARAMETER COUNT (structure does not matter)'}")
print("NOTE readout (2): comparator 3.2584e-6 is the old-f_J pp2_s1 value; b1 vs old f_J differ by <= 1e-8 (f_J audit, lead 2026-11-04), immaterial against the 0.3e-6 margin")
print(f"VERDICT M5' ext1 (tail average 1200..2100): (1) {'structure' if v1 else 'count'}; (2) {'structure' if v2 else 'count'}; {'agree' if v1 == v2 else 'DISAGREE: report both as registered, claim nothing'}")
