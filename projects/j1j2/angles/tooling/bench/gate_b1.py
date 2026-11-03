"""gate for B1 jobs: python3 -I gate_b1.py RUN_DIR STAGE  (A = correct+drift json in OUT, T = time_cpu.json, G = time_a40.json).  Prints one verdict line."""
import json, sys, os
OUT = '/scratch/a/A.Otaifi/tooling_bench/b1'
stage = sys.argv[2] if len(sys.argv) > 2 else 'A'
try:
    if stage == 'A':
        c = json.load(open(OUT + '/correct.json')); d = json.load(open(OUT + '/drift.json'))
        ok = all(t['R']['sign_mismatches'] == 0 and t['Q']['sign_mismatches'] == 0 for t in c['tests'].values()) and c['tests']['n50_random_10x10_f']['Q_broken_idx_swapped']['max_rel_elementwise'] > 1e-3
        u = d['uniform']; k = lambda s: u['first_exceed_' + s]
        print('GATE A: %s correct.json tests=%d R,Q zero sign mismatches and broken control fails=%s; drift uniform first>1e-8 L_elem median=%s min=%s H_elem median=%s min=%s (steps)' % (
            'OK' if ok else 'CHECK', len(c['tests']), ok, k('L_elementwise')['median'], k('L_elementwise')['min'], k('H_elementwise')['median'], k('H_elementwise')['min']))
    else:
        fn = {'T': 'time_cpu.json', 'G': 'time_a40.json'}[stage]
        t = json.load(open(OUT + '/' + fn)); r = t['results']
        bad = [k for k, v in r.items() if 'error' in v or not v.get('finite', False)]
        print('GATE %s: %s %s entries=%d failed_or_nonfinite=%d %s devices=%s' % (stage, 'OK' if not bad else 'CHECK', fn, len(r), len(bad), bad[:6], t['devices']))
except Exception as e:
    print('GATE %s: FAILED to read outputs: %r' % (stage, e))
