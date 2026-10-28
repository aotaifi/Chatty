"""test_e.py: TEST E (PLAN.md, pre-registered 2026-10-28). Where does the complex copies' remaining pair cost sit, per phi^2 decade
band of the fit (cx_c1fit.py), against the weight the fit's bond term puts there?  Exact 6x6 symmetric sector, numpy, CPU.

Bands: d = floor(log10 phi^2), phi normalised sum_reps n phi^2 = 1 (cx_c1fit.norm_la + Lab.decade_of; identical to bond_split's
normalisation).  d = -5..-13 are the fit's decades; 'hi' = d >= -4 and 'lo' = d <= -14 are outside (never drawn as x by the fit).
Pairs: blocked bonds of s2 (valid bonds with s2(x) s2(y) > 0; bond_split.cost_pass with sa = sphi = s2), |H_xy| = J/2 (bond_split.HW).
Cost:  c_xy = 2|H_xy| phi_x phi_y (cosh(e_x - e_y) - 1), e = ln|A| - ln phi - c, c = offset_c.npy (cancels in c_xy).  Identical to
       bond_split's |H| phi_y^2 (r - r*)^2 / r (checked against its cost_pass on a row slice and its split_*.json totals).
       Assignments: 'small' = band of the smaller-phi end; 'half' = half to the band of each end.
Fit weight of the bond term (cx_c1fit docstring l.21-25): ordered (x -> y), x a TRAINING orbit (not Lab.HO) in d = -5..-13:
       (1/9) (n_x / N_d) w_xy / Wbar_d,  w_xy = |H_xy| phi_y / phi_x,  N_d = sum n over training orbits in d,  Wbar_d = the fit's
       recorded estimate (cx_t1s.json 'Wbar').  Per band of x this is 1/9 by construction (exactly 1/9 * Wbar_exact/Wbar_fit);
       it is also reassigned per pair by the 'small' / 'half' rules for a like-for-like comparison.
Broken control: e shuffled across orbit reps within each band (seed 1); the cost table must change.

usage: python3 test_e.py OUTDIR [NPROC]
"""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bond_split as BS                                  # TAB, HW, S2, CSRD and cost_pass (cross-check) reused

PC = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs'
TARGET = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_d/o2b_round_1/target_la.npy'
SEEDS = (1, 2, 3)
DECS = list(range(-5, -14, -1))
LAB = ['hi'] + [str(d) for d in DECS] + ['lo']           # band index 0 = d >= -4, 1..9 = -5..-13, 10 = d <= -14
NB = len(LAB)
T0 = time.time()
G = {}


def log(m):
    print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)


def band_of(dec):
    return np.where(dec >= -4, 0, np.where(dec <= -14, NB - 1, -4 - dec)).astype(np.int8)    # -5 -> 1, ..., -13 -> 9


def work(rows):
    i0, i1 = rows
    n, lp, band, tr, Es, Wfit, Nd = G['n'], G['lp'], G['band'], G['tr'], G['E'], G['Wfit'], G['Nd']
    s = G['s']
    t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
    t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
    viol = valid & (s[i0:i1, None] * s[tt] > 0)
    rr, kk = np.nonzero(viol); x = rr + i0; y = tt[rr, kk]; H = BS.HW[kk]; del t, valid, tt, viol
    lpx, lpy = lp[x], lp[y]; bx, by = band[x], band[y]
    bs = np.where(lpx <= lpy, bx, by)                                       # band of the smaller-phi end
    pre = n[x] * H * np.exp(lpx + lpy)                                      # 0.5 (both orderings) * 2|H| phi_x phi_y * n_x
    out = dict(npairs=x.size)
    # fit weight of the ordered entry (x -> y), x = row end
    bxi = bx.astype(np.int64)
    inb = tr[x] & (bxi >= 1) & (bxi <= 9)
    k = np.clip(bxi - 1, 0, 8)
    w = H * np.exp(np.clip(lpy - lpx, -60, 60))                             # cx_c1fit bonds_of clip
    fw = np.where(inb, n[x] / Nd[k] * w / Wfit[k] / 9.0, 0.0)               # Wfit per seed: use seed-independent shape below
    out['fit_by_x'] = np.bincount(bx, weights=fw, minlength=NB)
    # the per-pair fit weight = sum of its two ordered entries; each ordered entry here is assigned like the cost
    out['fit_small'] = np.bincount(bs, weights=fw, minlength=NB)
    out['fit_half'] = 0.5 * (np.bincount(bx, weights=fw, minlength=NB) + np.bincount(by, weights=fw, minlength=NB))
    # exact Wbar_d: n-weighted mean of W_x over training orbits of d (W_x = sum over blocked bonds of w)
    out['Wsum'] = np.bincount(bx, weights=np.where(tr[x], n[x] * w, 0.0), minlength=NB)
    for tag, e in Es.items():
        de = e[x] - e[y]
        c = pre * 2.0 * np.sinh(0.5 * de) ** 2                              # cosh(de) - 1 = 2 sinh^2(de/2)
        out[f'{tag}_tot'] = float(c.sum())
        out[f'{tag}_small'] = np.bincount(bs, weights=c, minlength=NB)
        out[f'{tag}_half'] = 0.5 * (np.bincount(bx, weights=c, minlength=NB) + np.bincount(by, weights=c, minlength=NB))
    return out


def main():
    OUT = sys.argv[1]; NP = int(sys.argv[2]) if len(sys.argv) > 2 else 8; os.makedirs(OUT, exist_ok=True)
    n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s = np.load(BS.S2).astype(np.float64); D = n.size
    assert n.sum() == 9075135300 and set(np.unique(s)) <= {-1.0, 1.0}
    la_t = np.load(TARGET).astype(np.float64)
    lp = la_t - la_t.max(); lp = lp - 0.5 * np.log((n * np.exp(2 * lp)).sum())          # norm_la: sum n phi^2 = 1
    l2 = 2 * lp                                                                            # Lab.decade_of: lp = 2 lan - lz
    dec = np.floor(l2 / np.log(10.0)).astype(np.int32); band = band_of(dec)
    tr = ~(((((np.arange(D, dtype=np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)) >> np.uint64(7)) % np.uint64(5)) == 0)
    ph2 = n * np.exp(l2)
    res = dict(target=TARGET, D=int(D), dec_min=int(dec.min()), dec_max=int(dec.max()), labels=LAB, heldout_frac=float(1 - tr.mean()))
    res['band'] = {LAB[b]: dict(orbits=int((band == b).sum()), configs=float(n[band == b].sum()), phi2=float(ph2[band == b].sum()),
                                train_orbits=int(((band == b) & tr).sum())) for b in range(NB)}
    res['outside_detail'] = {str(d): dict(orbits=int((dec == d).sum()), configs=float(n[dec == d].sum()), phi2=float(ph2[dec == d].sum()))
                             for d in np.unique(dec) if d >= -4 or d <= -14}
    Nd = np.array([n[(band == b) & tr].sum() for b in range(1, 10)])
    # the fit's own decade counts (cx_t1s.json decade_orbits) must agree with ours
    E, offs, Wfits = {}, {}, {}
    for k in SEEDS:
        rd = f'{PC}/c1_cplx_s{k}/round_1'
        r = json.load(open(f'{rd}/cx_t1s.json'))
        for d in DECS:
            assert r['decade_orbits'][str(d)]['train'] == int(((dec == d) & tr).sum()), ('decade_orbits mismatch', k, d)
            assert r['decade_orbits'][str(d)]['heldout'] == int(((dec == d) & ~tr).sum()), ('decade_orbits mismatch HO', k, d)
        c = float(np.load(f'{rd}/offset_c.npy')); offs[k] = c; Wfits[k] = np.array(r['Wbar'])
        E[f's{k}'] = np.load(f'{rd}/la_table.npy').astype(np.float64) - lp - c
    log('decade_orbits of all three cx_t1s.json match this binning (train and held-out)')
    rng = np.random.default_rng(12345); esh = E['s1'].copy()
    for b in range(NB):
        idx = np.nonzero(band == b)[0]; esh[idx] = esh[rng.permutation(idx)]
    E['s1shuf'] = esh
    Wfit = np.mean([Wfits[k] for k in SEEDS], 0)          # fit weights: the mean of the three recorded Wbar (they agree to ~2%)
    res['Wbar_fit'] = {k: Wfits[k].tolist() for k in SEEDS}; res['offset_c'] = offs
    G.update(n=n, lp=lp, band=band, tr=tr, E=E, Wfit=Wfit, Nd=Nd, s=s)
    R = 65536; chunks = [(i, min(D, i + R)) for i in range(0, D, R)]
    log(f'pass over {len(chunks)} chunks, {NP} processes')
    acc = None
    with Pool(NP) as p:
        for j, o in enumerate(p.imap_unordered(work, chunks, chunksize=2)):
            acc = o if acc is None else {kk: acc[kk] + o[kk] for kk in acc}
            if j % 40 == 0: log(f'chunk {j + 1}/{len(chunks)}')
    log('pass done')
    # cross-check against bond_split.cost_pass on a row slice and against its split_*.json totals
    sl = (0, 400000)
    cp = BS.cost_pass(n, s, la_t, np.load(f'{PC}/c1_cplx_s1/round_1/la_table.npy'), rows=sl)
    mine = work(sl)['s1_tot']
    res['xcheck_slice'] = dict(rows=sl, bond_split=cp['total'], test_e=mine, rel=(mine - cp['total']) / cp['total'])
    res['xcheck_full'] = {}
    for k in SEEDS:
        ref = json.load(open(f'/scratch/a/A.Otaifi/ansatz_runs/pfaffian_d/split/split_s{k}.json'))['cost_total']
        res['xcheck_full'][f's{k}'] = dict(bond_split=ref, test_e=acc[f's{k}_tot'], rel=(acc[f's{k}_tot'] - ref) / ref)
    res['npairs_ordered'] = int(acc['npairs'])
    Wex = acc['Wsum'][1:10] / Nd
    res['Wbar_exact'] = Wex.tolist()
    res['fit_share_by_x'] = (acc['fit_by_x'] / acc['fit_by_x'].sum()).tolist()
    res['fit_share_small'] = (acc['fit_small'] / acc['fit_small'].sum()).tolist()
    res['fit_share_half'] = (acc['fit_half'] / acc['fit_half'].sum()).tolist()
    for tag in [f's{k}' for k in SEEDS] + ['s1shuf']:
        res[tag] = dict(total=acc[f'{tag}_tot'], total_site=acc[f'{tag}_tot'] / 36,
                        small=(acc[f'{tag}_small'] / acc[f'{tag}_tot']).tolist(), half=(acc[f'{tag}_half'] / acc[f'{tag}_tot']).tolist())
    res['mean'] = dict(small=np.mean([res[f's{k}']['small'] for k in SEEDS], 0).tolist(),
                       half=np.mean([res[f's{k}']['half'] for k in SEEDS], 0).tolist(),
                       small_sd=np.std([res[f's{k}']['small'] for k in SEEDS], 0, ddof=1).tolist(),
                       half_sd=np.std([res[f's{k}']['half'] for k in SEEDS], 0, ddof=1).tolist())
    # gate (pre-registered): bands holding >= 10% of the cost; fit share = 1/9 per band (bond term, by x); outside bands fit 0
    fit = np.array([0.0] + [1 / 9] * 9 + [0.0])
    gate = {}
    for asg in ('small', 'half'):
        for tag in [f's{k}' for k in SEEDS] + ['mean']:
            cs = np.array(res[tag][asg]); big = np.nonzero(cs >= 0.10)[0]
            rat = {LAB[b]: (float(cs[b] / fit[b]) if fit[b] > 0 else float('inf')) for b in big}
            sup = any(v >= 2 for v in rat.values()); kill = all(1 / 1.5 <= v <= 1.5 for v in rat.values())
            gate[f'{asg}_{tag}'] = dict(bands=rat, verdict='SUPPORTS' if sup else ('KILLS' if kill else 'NEITHER'))
    res['gate'] = gate
    # e_x^2 vs bond term: NOT recorded separately; per-decade held-out rms_e and wb-weighted bond_rms are (final curve point)
    est = {}
    for k in SEEDS:
        cv = json.load(open(f'{PC}/c1_cplx_s{k}/round_1/cx_t1s.json'))['fit']['curve'][-1]
        e2 = np.mean([cv['by_decade'][str(d)]['rms_e'] ** 2 for d in DECS]); b2 = np.mean([cv['by_decade'][str(d)]['bond_rms'] ** 2 for d in DECS])
        est[f's{k}'] = dict(step=cv['step'], L_ho=cv['L_ho'], mean_rms_e2=e2, mean_bond_rms2=b2, e_share_est=e2 / (e2 + b2), sum_est=e2 + b2)
    res['e2_term_estimate'] = est
    res['sec'] = time.time() - T0
    json.dump(res, open(os.path.join(OUT, 'test_e.json'), 'w'), indent=1, default=float)
    log(f'DONE {res["sec"]:.0f}s')


if __name__ == '__main__':
    main()
