"""vmc_gate.py : the pass gate of the VMC constants stage.  passed iff the paired exact-vs-subsampled h2 check exists with |z| <= 3 AND alpha_err/|alpha| <= 0.15.
usage as a script: python vmc_gate.py CONSTS.json [--inflate-alpha-err F] [--set-z Z] [--exact]  -> prints the verdict, exit 0 = passed, 3 = failed (the --inflate/--set-z options are the deliberately broken checks)."""
import sys, json
ZMAX = 3.0; REL_MAX = 0.15
def gate(d, need_paired=True):
    why = []; pz = d.get('paired_exact_h2')
    if need_paired:
        if not pz: why.append('no paired exact-vs-subsampled h2 check')
        elif not (abs(pz['z']) <= ZMAX): why.append(f"paired h2 |z| = {abs(pz['z']):.2f} > {ZMAX}")
    rel = abs(d['alpha_err'] / d['alpha']) if d['alpha'] else float('inf')
    if not (rel <= REL_MAX): why.append(f'alpha_err/|alpha| = {rel:.3f} > {REL_MAX}')
    return (not why), why, rel
if __name__ == '__main__':
    d = json.load(open(sys.argv[1])); a = sys.argv[2:]
    if '--inflate-alpha-err' in a: d['alpha_err'] *= float(a[a.index('--inflate-alpha-err') + 1])
    if '--set-z' in a: d.setdefault('paired_exact_h2', {})['z'] = float(a[a.index('--set-z') + 1])
    ok, why, rel = gate(d, need_paired=('--exact' not in a)); print('VMC_GATE', 'PASS' if ok else 'FAIL', f'alpha_err/|alpha|={rel:.4f}', '; '.join(why)); sys.exit(0 if ok else 3)
