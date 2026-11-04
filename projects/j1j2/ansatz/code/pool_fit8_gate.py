"""pool_fit8_gate.py RUNDIR WHAT: prints ONE verdict line from the json files of a pool_fit8 job (pool_fit8_job.sbatch).  Exit 0 always (the line carries the verdict).
WHAT t8abcd: check_a.json .. check_d.json of pool_fit8_test.py: each needs PASS and BROKEN_FAILS.
WHAT t8e / t8e2: timing jsons exist (no gate).
WHAT c6a / c6b / c6c: the existing 6x6 pool_checks.py a / b / c(pert) json: 'pass' true AND the numbers equal the recorded baseline
  (pfaffian_v/cpu/checka_job/check_a/check_a.json, check_b/check_b.json, check_c_pert/check_c.json; pool_fit_sha256 differs by construction)."""
import sys, json, os
import numpy as np
D, W = sys.argv[1], sys.argv[2]
V = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_v/cpu'


def load(p):
    try: return json.load(open(p))
    except Exception: return None


def maxdiff(a, b):
    """max |a - b| over matching numeric leaves of two json trees (keys of a)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return max([maxdiff(a[k], b[k]) for k in a if k in b and k not in ('sec', 'sec_per_step', 'pool_fit_sha256')] or [0.0])
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b): return max([maxdiff(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool): return abs(a - b) / max(abs(b), 1e-300) if abs(b) > 1e-30 else abs(a - b)
    return 0.0


if W == 't8abcd':
    parts = []; ok = True
    for k in 'abcd':
        j = load(f'{D}/check_{k}.json')
        if j is None: parts.append(f'{k}: MISSING'); ok = False; continue
        good = bool(j.get('PASS')) and bool(j.get('BROKEN_FAILS')); ok &= good; parts.append(f'{k}: PASS={j.get("PASS")} BROKEN_FAILS={j.get("BROKEN_FAILS")}')
    print(f'VERDICT pf8 {"ALL PASS" if ok else "FAIL"}; ' + '; '.join(parts))
elif W == 'recheck':
    import hashlib
    parts = []; ok = True
    for lab, f in (('a', f'{D}/check_a.json'), ('d', f'{D}/check_d.json'), ('a_dedup', f'{D}/dedup/check_a.json'), ('d_dedup', f'{D}/dedup/check_d.json')):
        j = load(f)
        if j is None: parts.append(f'{lab}: MISSING'); ok = False; continue
        good = bool(j.get('PASS')) and bool(j.get('BROKEN_FAILS')); ok &= good; parts.append(f'{lab}: PASS={j.get("PASS")} BROKEN_FAILS={j.get("BROKEN_FAILS")} pp2_lemma_sha256={str(j.get("pp2_lemma_sha256"))[:12]}')
    print(f'VERDICT pf8 recheck {"ALL PASS" if ok else "FAIL"}; ' + '; '.join(parts))
elif W in ('t8e', 't8e2'):
    j = load(f'{D}/check_{"e" if W == "t8e" else "e2"}.json'); print(f'VERDICT pf8 {W}: ' + ('timing json present' if j else 'MISSING'))
elif W in ('c6a', 'c6b', 'c6c'):
    new = {'c6a': f'{D}/check_a/check_a.json', 'c6b': f'{D}/check_b/check_b.json', 'c6c': f'{D}/check_c_pert/check_c.json'}[W]
    old = {'c6a': f'{V}/checka_job/check_a/check_a.json', 'c6b': f'{V}/check_b/check_b.json', 'c6c': f'{V}/check_c_pert/check_c.json'}[W]
    n, o = load(new), load(old)
    if n is None or o is None: print(f'VERDICT pf8 {W}: MISSING (new {n is not None}, baseline {o is not None})')
    else:
        md = maxdiff(n, o); print(f'VERDICT pf8 {W}: pass new={n.get("pass")} baseline={o.get("pass")}; max relative deviation of every numeric field from the baseline run = {md:.3e}; pool_fit sha new {str(n.get("pool_fit_sha256"))[:12]} old {str(o.get("pool_fit_sha256"))[:12]}')
