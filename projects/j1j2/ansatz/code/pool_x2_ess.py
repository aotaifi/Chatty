"""pool_x2_ess.py: stage-1 bond / configuration ESS per decade -7..-9 on nested subsets of the single-rung beta 0.25 draws (PLAN_pfaffian.md "Pool expansion"; rule in pool_x2/PREREG_stage1.md).
usage: python pool_x2_ess.py RUND OUT.json [--nt 40] [--boot 200]      RUND/prod/out_T.npz (draws), RUND/prod/fj_T.npz (f_J rows).
Uses gates_cE/code/ess_lib.decade_ess (pool_fit.pool_weights) for the point values (single rung + one dummy beta-1 entry that only fixes the median, outside decades -7..-9)
and an equivalent fast single-rung routine for the bootstrap (asserted equal on the full set)."""
import sys, os, json, argparse, glob
import numpy as np
sys.path.insert(0, '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8build/gates_cE/code')
import ess_lib as E
ap = argparse.ArgumentParser(); ap.add_argument('rund'); ap.add_argument('out'); ap.add_argument('--nt', type=int, default=40); ap.add_argument('--boot', type=int, default=200)
ap.add_argument('--sub', default='prod'); ap.add_argument('--fjsub', default=None); ap.add_argument('--ms', default='200000,500000,1000000,2000000'); a = ap.parse_args()
C = 4.0; DECS = [-7, -8, -9]; BETA = 0.25; LN10 = np.log(10.0); HEAVY = -0.5; TARGET = 5.4e4
REG = {-7: 6.2e3, -8: 8.1e3, -9: 1.0e4}                    # registered pool (4.5e5 per rung) bond ESS, PLAN gate (c)
D = os.path.join(a.rund, a.sub); DF = os.path.join(a.rund, a.fjsub or a.sub)
# ---- load: per task all rows (words not needed): lnB, chain, row; f_J rows
T = []; 
for t in range(1, a.nt + 1):
    z = np.load(f'{D}/out_{t}.npz'); f = np.load(f'{DF}/fj_{t}.npz'); assert int(z['final']) == 1 and int(f['final']) == 1 and len(f['idx']) == int(f['n_sel'])
    n = int(z['n_done']); row = np.arange(n); T.append(dict(lnB=z['lnB'], chain=(int(z['task']) * int(z['E']) + row % int(z['E'])), row=row, half=str(z['half']), fidx=f['idx'], fx=f['fx'], WB=f['WB'], nblk=f['nblk'], fy=f['fy'], med=float(f['med']), E=int(z['E']), thin=int(z['thin'])))
med = T[0]['med']; assert all(abs(t['med'] - med) < 1e-12 for t in T)
def build(m_per_task):
    """concatenate rows < m_per_task of every task, only draws in decades -7..-9; rho from the f_J rows."""
    L, CH, RHO, FX, HF, DEC = [], [], [], [], [], []
    for t in T:
        k = t['fidx'] < m_per_task; ix = t['fidx'][k]; lnB = t['lnB'][ix]; fx = t['fx'][k]; fy = t['fy'][k]
        fxv = np.where(np.isfinite(fx), fx, 0.0); fyv = np.where(np.isfinite(fy), fy, 0.0); ok = (t['nblk'][k] > 0)[:, None] & (t['fy'][k].shape[1] > 0)
        rho = np.where(ok & (t['nblk'][k][:, None] > 0), t['WB'][k][:, None] * np.exp(C * (fyv - fxv[:, None])), 0.0)
        L.append(lnB); CH.append(t['chain'][ix]); RHO.append(rho); FX.append(fx); HF.append(np.full(len(ix), t['half'] == 'A')); DEC.append(np.floor((2 * lnB - med) / LN10).astype(int))
    return dict(lnB=np.concatenate(L), chain=np.concatenate(CH), rho=np.concatenate(RHO), fx=np.concatenate(FX), halfA=np.concatenate(HF), dec=np.concatenate(DEC))
def fast(lnB, rho, dec, sel=None):
    """single rung: w_in = exp(-2 beta lnB) normalised per decade; returns {d: (config ESS, bond ESS)}."""
    out = {}; r = rho.mean(1)
    for d in DECS:
        m = dec == d if sel is None else (dec == d) & sel
        if m.sum() == 0: out[d] = (0.0, 0.0); continue
        lw = -2 * BETA * lnB[m]; w = np.exp(lw - lw.max()); w /= w.sum(); wr = w * r[m]
        out[d] = (1.0 / np.sum(w * w), np.sum(wr) ** 2 / np.sum(wr * wr) if np.sum(wr * wr) > 0 else 0.0)
    return out
res = dict(nt=a.nt, med=med, thin=sorted(set(t['thin'] for t in T)), E=T[0]['E'], target=TARGET, registered_bond_ess=REG, subsets=[])
ms = [int(float(x)) for x in a.ms.split(',')]; BS = {}
for m in ms:
    mp = m // a.nt; B = build(mp); BS[m] = B; n = len(B['lnB'])
    # library point values (dummy beta-1 entry fixes the median at med)
    lnB = np.concatenate([[med / 2], B['lnB']]); ch = np.concatenate([[0], np.ones(n, int)]); rho = np.concatenate([np.zeros((1, B['rho'].shape[1])), B['rho']])
    lib, _ = E.decade_ess(lnB, ch, [1.0, BETA], DECS, rho); fa = fast(B['lnB'], B['rho'], B['dec'])
    row = dict(m=m, per_task=mp, n_selected=int(n), decades={})
    for d in DECS:
        assert abs(lib[d]['x_ess'] - fa[d][0]) < 1e-6 * max(1, fa[d][0]) and abs(lib[d]['bond_ess'] - fa[d][1]) < 1e-6 * max(1, fa[d][1]), (d, lib[d], fa[d])
        md = B['dec'] == d; hv = int(np.sum(md & np.isfinite(B['fx']) & (B['fx'] < HEAVY)))
        row['decades'][d] = dict(n=int(md.sum()), config_ess=lib[d]['x_ess'], bond_ess=lib[d]['bond_ess'], ratio=lib[d]['bond_ess'] / lib[d]['x_ess'] if lib[d]['x_ess'] > 0 else None, heavy=hv,
                                 heavy_A=int(np.sum(md & B['halfA'] & np.isfinite(B['fx']) & (B['fx'] < HEAVY))), heavy_B=int(np.sum(md & ~B['halfA'] & np.isfinite(B['fx']) & (B['fx'] < HEAVY))),
                                 top1=float(lib[d]['top1']), top10=float(lib[d]['top10']), n_nonfinite_fx=int(np.sum(md & ~np.isfinite(B['fx']))))
    res['subsets'].append(row); print('m %8d  selected %7d' % (m, n), {d: (round(row['decades'][d]['config_ess']), round(row['decades'][d]['bond_ess']), round(row['decades'][d]['ratio'], 4), row['decades'][d]['heavy']) for d in DECS}, flush=True)
# ---- cluster bootstrap over chains (same resample for all m)
rng = np.random.default_rng(12345); chains = np.unique(np.concatenate([B['chain'] for B in BS.values()])); nch = len(chains); idxs = {m: {} for m in ms}
for m in ms:
    B = BS[m]; o = np.argsort(B['chain'], kind='stable'); cs = B['chain'][o]; st = np.searchsorted(cs, chains); en = np.searchsorted(cs, chains, side='right'); idxs[m] = (o, st, en)
BOOT = {m: {d: [] for d in DECS} for m in ms}; HB = {m: {d: [] for d in DECS} for m in ms}
for b in range(a.boot):
    pick = rng.integers(0, nch, nch)
    for m in ms:
        B = BS[m]; o, st, en = idxs[m]; sel = np.concatenate([o[st[c]:en[c]] for c in pick]); fa = fast(B['lnB'][sel], B['rho'][sel], B['dec'][sel])
        for d in DECS:
            BOOT[m][d].append(fa[d][1] / fa[d][0] if fa[d][0] > 0 else np.nan)
            HB[m][d].append(float(np.sum((B['dec'][sel] == d) & np.isfinite(B['fx'][sel]) & (B['fx'][sel] < HEAVY))))
for row in res['subsets']:
    for d in DECS:
        x = np.array(BOOT[row['m']][d]); row['decades'][d]['ratio_boot_sd'] = float(np.nanstd(x)); row['decades'][d]['ratio_boot_16_84'] = [float(np.nanpercentile(x, 16)), float(np.nanpercentile(x, 84))]
# ---- verdict rule (PREREG_stage1.md), decade -7
m_lo, m_hi = ms[1], ms[-1]; x_lo = np.array(BOOT[m_lo][-7]); x_hi = np.array(BOOT[m_hi][-7]); dif = x_hi - x_lo
r_lo = [r for r in res['subsets'] if r['m'] == m_lo][0]['decades'][-7]['ratio']; r_hi = [r for r in res['subsets'] if r['m'] == m_hi][0]['decades'][-7]['ratio']
sd = float(np.nanstd(dif)); z = float((r_hi - r_lo) / sd) if sd > 0 else float('nan')
bes = [r['decades'][-7]['bond_ess'] for r in res['subsets'] if r['m'] >= m_lo]; hv = [r['decades'][-7]['heavy'] for r in res['subsets'] if r['m'] >= m_lo]
slope = float(np.polyfit(np.log(hv), np.log(bes), 1)[0]) if len(hv) >= 2 and min(hv) > 0 else float('nan')
# bootstrap slope
sl = []
for b in range(a.boot):
    be = [BOOT[m][-7][b] * 1.0 for m in ms if m >= m_lo]; 
    # bond ESS = ratio x config ESS (config ESS of the point value; its own bootstrap scatter is small relative)
    ce = [[r for r in res['subsets'] if r['m'] == m][0]['decades'][-7]['config_ess'] for m in ms if m >= m_lo]; h = [HB[m][-7][b] for m in ms if m >= m_lo]
    if min(h) > 0 and np.all(np.isfinite(be)): sl.append(np.polyfit(np.log(h), np.log(np.array(be) * np.array(ce)), 1)[0])
res['verdict'] = dict(rule='STOP iff r_-7(2e6) < r_-7(5e5) by more than 2 sigma of the bootstrap difference', r_5e5=r_lo, r_2e6=r_hi, diff=r_hi - r_lo, sd_diff=sd, z=z, loglog_slope_bondESS_vs_heavy=slope,
                      slope_boot_sd=float(np.std(sl)) if sl else None, decision='STOP' if (np.isfinite(z) and z < -2.0) else 'CONTINUE', ratio_by_m={r['m']: r['decades'][-7]['ratio'] for r in res['subsets']})
# ---- projection (linear in m under CONTINUE)
last = [r for r in res['subsets'] if r['m'] == ms[-1]][0]; proj = {}
for d in DECS:
    be = last['decades'][d]['bond_ess']; mstar = ms[-1] * TARGET / be; net = max(TARGET - REG[d], 0.0); mnet = ms[-1] * net / be
    proj[d] = dict(bond_ess_at_mmax=be, m_star_new_only=mstar, m_star_net_of_registered=mnet, bond_ess_per_draw=be / ms[-1])
res['projection'] = proj; res['projection_max_new_only'] = max(v['m_star_new_only'] for v in proj.values()); res['projection_max_net'] = max(v['m_star_net_of_registered'] for v in proj.values())
json.dump(res, open(a.out, 'w'), indent=1, default=float)
print('VERDICT', json.dumps(res['verdict'], default=float)); print('PROJECTION', json.dumps(proj, default=float))
print('STAGE1 %s: -7 ratio bond/config at m=5e5 %.4f -> 2e6 %.4f (z %.2f), loglog slope bondESS vs heavy %.2f; draws for bond ESS >= 5.4e4: new-only %.2e, net of registered %.2e' % (
    res['verdict']['decision'], r_lo, r_hi, z, slope, res['projection_max_new_only'], res['projection_max_net']))
