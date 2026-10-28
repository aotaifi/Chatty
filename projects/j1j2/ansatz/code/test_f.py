"""test_f.py: TEST F (PLAN.md, pre-registered by the lead 2026-10-28). True E_FN cost of the deep region: splice e = 0 into the complex
copies c1_cplx_s{1,2,3} on the configurations the fit never draws and re-solve E_FN exactly with sign s2.

Solver: bond_split.build + bond_split.efn (the code that wrote pfaffian_d/split/split_s*.json E_FN_copy / E_FN_target), imported, unchanged.
Bands / normalisation: as test_e.py (phi = exp(target_la) normalised sum_reps n phi^2 = 1; per-config phi^2).
e = ln|A| - ln phi - c, c = offset_c.npy.  Splice on mask M: ln|A| := ln phi + c on M (e = 0 there), untouched elsewhere.
  control  : unmodified copy; E must equal split_s{k}.json E_FN_copy.E to 1e-9 relative, else STOP.
  arm A    : M = phi^2 < 1e-13 (test_e band 'lo', d <= -14)
  arm B    : M = phi^2 < 1e-11 (bands -12, -13, lo)
  broken   : (seed 1) random orbits from bands -5..-9 with the same phi^2 weight (sum n phi^2) as arm A.  The same COUNT is impossible
             (arm A has ~1.07e7 orbits, bands -5..-9 together ~1.2e5), so the draw matches the weight only; count reported.
tax = E_FN(copy variant) - E_FN(target), per site.  removed = 1 - tax_arm / tax_0.

usage: python3 test_f.py OUTDIR
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS

PC = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs'
TARGET = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_d/o2b_round_1/target_la.npy'
SPLIT = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_d/split'
SEEDS = (1, 2, 3)
T0 = time.time()


def log(m):
    print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)


def main():
    OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
    n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s = np.load(BS.S2).astype(np.float64); D = n.size
    assert n.sum() == 9075135300 and set(np.unique(s)) <= {-1.0, 1.0}
    la_t = np.load(TARGET).astype(np.float64)
    lp = la_t - la_t.max(); lp = lp - 0.5 * np.log((n * np.exp(2 * lp)).sum())       # test_e / bond_split normalisation
    l2 = 2 * lp; ph2 = n * np.exp(l2)
    dec = np.floor(l2 / np.log(10.0)).astype(np.int32)
    mA = l2 < np.log(1e-13); mB = l2 < np.log(1e-11)
    assert np.array_equal(mA, dec <= -14) and np.array_equal(mB, dec <= -12)
    WA = float(ph2[mA].sum())
    res = dict(target=TARGET, D=int(D), bs_sha='see REPORT', maskA=dict(orbits=int(mA.sum()), configs=float(n[mA].sum()), phi2=WA),
               maskB=dict(orbits=int(mB.sum()), configs=float(n[mB].sum()), phi2=float(ph2[mB].sum())), solves={})
    # broken mask: random orbits of bands -5..-9, weight-matched to arm A
    rng = np.random.default_rng(20261028)
    cand = np.nonzero((dec <= -5) & (dec >= -9))[0]; perm = rng.permutation(cand); cw = np.cumsum(ph2[perm])
    kcut = int(np.searchsorted(cw, WA))                      # prefix strictly below WA, then the one candidate that best fills the gap
    gap = WA - (cw[kcut - 1] if kcut > 0 else 0.0); rest = perm[kcut:]
    pick = rest[np.argmin(np.abs(ph2[rest] - gap))]
    mX = np.zeros(D, bool); mX[perm[:kcut]] = True; mX[pick] = True
    res['maskX'] = dict(orbits=int(mX.sum()), configs=float(n[mX].sum()), phi2=float(ph2[mX].sum()), candidates=int(cand.size),
                        cand_phi2=float(ph2[cand].sum()), by_dec={int(d): int((mX & (dec == d)).sum()) for d in range(-9, -4)},
                        rng_seed=20261028)
    log(f"masks: A {res['maskA']}  B {res['maskB']}  X {res['maskX']}")

    def dump():
        res['sec'] = time.time() - T0
        json.dump(res, open(os.path.join(OUT, 'test_f.json'), 'w'), indent=1, default=float)

    log('building K, Vm (bond_split.build)')
    tb = time.time(); K, Vm, D0 = BS.build(n, s); res['build_sec'] = time.time() - tb
    log(f'built: K nnz {K.nnz}, Vm nnz {Vm.nnz} ({res["build_sec"]:.0f}s)')

    def solve(tag, la):
        t = time.time(); r = BS.efn(K, Vm, D0, n, la); r['sec'] = time.time() - t
        res['solves'][tag] = r; dump()
        log(f'{tag}: E {r["E"]!r} dE_site {r["dE_site"]:.6e} res {r["res"]:.1e} it {r["iters"]} {r["sec"]:.0f}s')
        return r

    rt = solve('target', la_t)
    ref_t = json.load(open(f'{SPLIT}/split_s1.json'))['E_FN_target']['E']
    res['target_check'] = dict(ref=ref_t, rel=(rt['E'] - ref_t) / abs(ref_t))
    log(f"target vs split: rel {res['target_check']['rel']:.2e}")
    summ = {}
    for k in SEEDS:
        rd = f'{PC}/c1_cplx_s{k}/round_1'
        la = np.load(f'{rd}/la_table.npy').astype(np.float64); c = float(np.load(f'{rd}/offset_c.npy'))
        ref = json.load(open(f'{SPLIT}/split_s{k}.json'))
        r0 = solve(f's{k}_control', la)
        rel = (r0['E'] - ref['E_FN_copy']['E']) / abs(ref['E_FN_copy']['E'])
        rel_dE = (r0['dE_site'] - ref['E_FN_copy']['dE_site']) / abs(ref['E_FN_copy']['dE_site'])
        res[f's{k}_control_check'] = dict(ref=ref['E_FN_copy']['E'], rel_E=rel, rel_dE_site=rel_dE, passed=bool(abs(rel) <= 1e-9))
        log(f's{k} CONTROL rel(E) {rel:.2e} rel(dE_site) {rel_dE:.2e}')
        if abs(rel) > 1e-9:
            log(f'CONTROL FAIL s{k}: STOP'); res['status'] = f'CONTROL FAIL s{k}'; dump(); sys.exit(2)
        out = dict(offset_c=c, tax0=r0['dE_site'] - rt['dE_site'])
        arms = [('A', mA), ('B', mB)] + ([('X', mX)] if k == 1 else [])
        for nm, m in arms:
            lam = la.copy(); lam[m] = lp[m] + c
            e = lam - lp - c
            assert np.abs(e[m]).max() < 1e-12 and np.array_equal(lam[~m], la[~m]); res[f's{k}_{nm}_maxabs_e_on_mask'] = float(np.abs(e[m]).max())
            r = solve(f's{k}_{nm}', lam)
            out[f'tax_{nm}'] = r['dE_site'] - rt['dE_site']
            out[f'removed_{nm}'] = 1 - out[f'tax_{nm}'] / out['tax0']
        summ[f's{k}'] = out; res['summary'] = summ; dump()
        log(f's{k}: ' + ' '.join(f'{a}={b:.4e}' for a, b in out.items()))
    for key in ('tax0', 'tax_A', 'tax_B', 'removed_A', 'removed_B'):
        v = [summ[f's{k}'][key] for k in SEEDS]
        summ.setdefault('mean', {})[key] = float(np.mean(v)); summ.setdefault('sd', {})[key] = float(np.std(v, ddof=1))
    mA_ = summ['mean']['removed_A']; mB_ = summ['mean']['removed_B']
    res['verdict'] = dict(kill=bool(mA_ < 0.05), A_in_pred=bool(0.10 <= mA_ <= 0.25), B_in_pred=bool(0.35 <= mB_ <= 0.60))
    res['status'] = 'DONE'; dump()
    log(f'VERDICT {res["verdict"]} mean removed A {mA_:.4f} B {mB_:.4f}')
    log('DONE')


if __name__ == '__main__':
    main()
