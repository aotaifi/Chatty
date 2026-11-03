"""test_w_pf.py: Test W of PLAN_pfaffian.md ("where does the second PP store the round correction?"), CPU only.
(The older ansatz/code/test_w.py is a different test, PLAN.md "TEST W"; not touched.)
usage: python3 test_w_pf.py OUTDIR [--quick]     (python = runs/a40slice/venv_jax_0.8.2/bin/python; jax on CPU)
Part 1 (numpy, all orbits, all blocked bonds; blocked = s_G(x) s_G(y) > 0, H_xy = J/2 > 0):
  B4 = pfaffian_r4 theta_4 table, T5 = pfaffian_r5/target_T5_sG.npy, arms = pp2_s1 and ctl_s2 tab_t5 tables.
  la4 normalised sum n B4^2 = 1 (per configuration).  Delta*(x) = ln T5 - ln B4, Delta(x) = ln B5 - ln B4, each minus its n|B4|^2-weighted mean.
  decade d = floor(log10 B4(x)^2) per configuration; a bond is binned by the decade of its smaller-|B4| side x.
  site:  kappa_site_d = sum n Delta Delta* / sum n Delta*^2  (n-weighted, orbit reps in decade d)
  bond:  kappa_bond_d = sum w dD dD* / sum w dD*^2, dD = Delta_x - Delta_y, w = 0.5 n_x |H_xy| B4(x) B4(y) over directed (rep, bond) entries
         (each unordered pair seen from both ends, factor 0.5; as bond_split.cost_pass); share_d = sum_d w dD*^2 / sum w dD*^2.
  controls: 'self' (Delta = Delta*) must give kappa = 1; 'perm' (Delta = random permutation of Delta*, seed 1) ~ 0.
Part 2 (jax CPU): theta_5 of pp2_s1 (pfaffian_r5/pp2_s1/tab/theta_5.npy), P_1 and lambda P_2 separately for a stratified sample
  (<= 20000 orbits per decade, rng seed 12345), model code = cx_r1fit.py A_one (arm pp2) / rbmpp_lib.pp2_combine, copied verbatim.
  check |P_1 + lam P_2| reproduces la_table of pp2_s1 up to a global offset; rho = |lam P_2 / P_1| per decade (median, 10/90%), Spearman(rho, Delta)."""
import os, sys, json, time, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = sys.argv[1]; QUICK = '--quick' in sys.argv; os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(m): print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)
import bond_split as BS
R5 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5'; R4 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4'
S1P = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy'
TABS = dict(B4=(f'{R4}/pdb_tabs/tab_t4/round_1/la_table.npy', '8e358ecb'), T5=(f'{R5}/target_T5_sG.npy', '7e946616'),
            pp2=(f'{R5}/pp2_s1/tab/tab_t5/round_1/la_table.npy', 'f5d9d1aa'), ctl=(f'{R5}/ctl_s2/tab/tab_t5/round_1/la_table.npy', None))
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RJ = f'{OUT}/w_result.json'; res = dict(host=os.uname().nodename, quick=QUICK)
def dump(): json.dump(res, open(RJ + '.tmp', 'w'), indent=1, default=float); os.replace(RJ + '.tmp', RJ)
rd = json.load(open(f'{R5}/readout/readout.json')); TABS['ctl'] = (TABS['ctl'][0], rd['tables']['ctl_s2']['sha256'][:8])
res['shas'] = {}
for k, (p, sh) in TABS.items():
    h = sha(p); assert h.startswith(sh), (k, h, sh); res['shas'][k] = h
res['s_G_sha256'] = sha(S1P); res['code_sha256'] = {f: sha(f'{HERE}/{f}') for f in ('test_w_pf.py', 'bond_split.py', 'cx_r1fit.py', 'rbmpp_lib.py')}
log(f'shas ok {res["shas"]}'); dump()
n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); D = len(n); assert n.sum() == 9075135300
s1 = np.load(S1P).astype(np.float64); assert set(np.unique(s1)) <= {-1.0, 1.0}
def load(p): a = np.load(p).astype(np.float64); assert a.shape == (D,) and np.isfinite(a).all(); return a
la4 = load(TABS['B4'][0]); la4 = la4 - la4.max(); la4 -= 0.5 * np.log((n * np.exp(2 * la4)).sum())
wB = n * np.exp(2 * la4); assert abs(wB.sum() - 1) < 1e-12
dec = np.floor(2 * la4 / np.log(10.0)).astype(np.int64); DOFF = 40; ND = 50       # dec index = dec + DOFF
assert dec.min() + DOFF >= 0 and dec.max() + DOFF < ND, (dec.min(), dec.max())
def delta_of(la): d = la - la4; return d - (wB * d).sum()
Dst = delta_of(load(TABS['T5'][0]))
Darm = dict(pp2=delta_of(load(TABS['pp2'][0])), ctl=delta_of(load(TABS['ctl'][0])))
Darm['self'] = Dst.copy(); Darm['perm'] = Dst[np.random.default_rng(1).permutation(D)]
ARMS = list(Darm)
RMAX = 2_000_000 if QUICK else D
# ---------------------------------------------------------------------------- part 1
dI = dec + DOFF
cnt_orb = np.bincount(dI, minlength=ND); cnt_cfg = np.bincount(dI, weights=n, minlength=ND)
site_xs = {a: np.bincount(dI[:RMAX], weights=(n * Darm[a] * Dst)[:RMAX], minlength=ND) for a in ARMS}
site_ss = np.bincount(dI[:RMAX], weights=(n * Dst * Dst)[:RMAX], minlength=ND)
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
b_xs = {a: np.zeros(ND) for a in ARMS}; b_ss = np.zeros(ND); b_n = np.zeros(ND); R = 65536
B4 = np.exp(la4)
for i0 in range(0, RMAX, R):
    i1 = min(RMAX, i0 + R); t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
    blk = valid & (s1[i0:i1, None] * s1[tt] > 0)
    rr, kk = np.nonzero(blk); x = rr + i0; y = tt[rr, kk]
    w = 0.5 * n[x] * BS.HW[kk] * B4[x] * B4[y]
    d_ = np.where(la4[x] <= la4[y], dI[x], dI[y])
    dst = Dst[x] - Dst[y]; b_ss += np.bincount(d_, weights=w * dst * dst, minlength=ND); b_n += np.bincount(d_, minlength=ND)
    for a in ARMS: b_xs[a] += np.bincount(d_, weights=w * (Darm[a][x] - Darm[a][y]) * dst, minlength=ND)
    if (i0 // R) % 40 == 0: log(f'bonds {i1}/{RMAX}')
shares = b_ss / b_ss.sum()
rows = []
for j in range(ND):
    if cnt_orb[j] == 0 or site_ss[j] <= 0 or b_ss[j] <= 0: continue
    r = dict(decade=j - DOFF, n_orbits=int(cnt_orb[j]), n_configs=float(cnt_cfg[j]), n_bond_entries=int(b_n[j]), share=float(shares[j]))
    for a in ARMS:
        r[f'kappa_site_{a}'] = float(site_xs[a][j] / site_ss[j]) if site_ss[j] > 0 else None
        r[f'kappa_bond_{a}'] = float(b_xs[a][j] / b_ss[j]) if b_ss[j] > 0 else None
    rows.append(r)
res['part1'] = dict(rows=rows, rmax=RMAX, bond_weight_total=float(b_ss.sum()))
lo = lambda r: r['decade'] < -9
def ksum(sel): return float(sum(r['share'] * (r['kappa_bond_pp2'] - r['kappa_bond_ctl']) for r in rows if sel(r) and r['kappa_bond_pp2'] is not None))
res['kill'] = dict(sum_lt_m9=ksum(lo), sum_ge_m9=ksum(lambda r: not lo(r)), share_lt_m9=float(sum(r['share'] for r in rows if lo(r))))
res['kill']['KILL_near_node_reshaping'] = bool(res['kill']['sum_ge_m9'] > res['kill']['sum_lt_m9'])
res['controls'] = dict(self_kappa_bond_minmax=[min(r['kappa_bond_self'] for r in rows if r['kappa_bond_self'] is not None), max(r['kappa_bond_self'] for r in rows if r['kappa_bond_self'] is not None)],
                       self_kappa_site_minmax=[min(r['kappa_site_self'] for r in rows), max(r['kappa_site_self'] for r in rows)],
                       perm_kappa_bond_absmax=max(abs(r['kappa_bond_perm']) for r in rows if r['kappa_bond_perm'] is not None),
                       perm_kappa_site_absmax=max(abs(r['kappa_site_perm']) for r in rows))
dump()
print(f'{"dec":>4} {"orbits":>9} {"configs":>11} {"share":>8} | site: pp2 ctl | bond: pp2 ctl diff | self perm(bond)')
for r in rows:
    print(f'{r["decade"]:>4} {r["n_orbits"]:>9} {r["n_configs"]:>11.3e} {r["share"]:8.4f} | {r["kappa_site_pp2"]:7.3f} {r["kappa_site_ctl"]:7.3f} | '
          f'{r["kappa_bond_pp2"]:7.3f} {r["kappa_bond_ctl"]:7.3f} {r["kappa_bond_pp2"] - r["kappa_bond_ctl"]:7.3f} | {r["kappa_bond_self"]:.4f} {r["kappa_bond_perm"]:.3f}', flush=True)
log(f'KILL {res["kill"]}  controls {res["controls"]}')
# ---------------------------------------------------------------------------- part 2
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from rbmpp_lib import Lattice, pp2_combine
N = 36; NH = 18; NA = 4; NP1 = 2 * N * N + NA * (N + 1)
lat = Lattice(6, 0.5, 'full'); IMG = lat.IMG; FLP = lat.FL.astype(jnp.float64)
th = np.load(f'{R5}/pp2_s1/tab/theta_5.npy'); res['theta_5_sha256'] = sha(f'{R5}/pp2_s1/tab/theta_5.npy'); assert res['theta_5_sha256'].startswith('4558d78f') and th.size == 5334
th = jnp.asarray(th)
f = (th[:N * N] + 1j * th[N * N:2 * N * N]).reshape(N, N); wb = th[2 * N * N:NP1].reshape(NA, N + 1)
f2 = (th[NP1:NP1 + N * N] + 1j * th[NP1 + N * N:NP1 + 2 * N * N]).reshape(N, N); lam = th[NP1 + 2 * N * N] + 1j * th[NP1 + 2 * N * N + 1]
res['lambda'] = [float(lam.real), float(lam.imag)]; res['rbm_block_maxabs'] = float(jnp.abs(wb).max())
def log_one(f, zz):                                              # verbatim cx_r1fit.py log_one, COMPLEX, no BF
    order = jnp.argsort(-zz, stable=True); Uo = order[:NH]; Do = order[NH:]
    Fm = f[Uo][:, Do]; sg, ld = jnp.linalg.slogdet(Fm)
    up = (zz > 0).astype(jnp.int32); dn = 1 - up; inv = jnp.sum(up * (jnp.cumsum(dn) - dn))
    return ld, (1.0 - 2.0 * (inv % 2).astype(jnp.float64)) * sg
def lncosh(x): ax = jnp.abs(x); return ax + jnp.log1p(jnp.exp(-2.0 * ax)) - jnp.log(2.0)
def parts(sigma):                                                # cx_r1fit.py A_one (PP2) split: returns m + R, S1, lam S2
    Z = FLP[:, None] * sigma.astype(jnp.float64)[IMG]
    ld, sgn = jax.vmap(lambda zz: log_one(f, zz))(Z); ld2, sgn2 = jax.vmap(lambda zz: log_one(f2, zz))(Z)
    m, _ = pp2_combine(ld, sgn, ld2, sgn2, lam)
    S1 = jnp.sum(sgn * jnp.exp(ld - m)); S2 = lam * jnp.sum(sgn2 * jnp.exp(ld2 - m))
    Rx = jnp.sum(lncosh(Z @ wb[:, :N].T + wb[:, N][None, :]))
    return m + Rx, S1, S2
parts_j = jax.jit(jax.vmap(parts))
zz_ = np.load('/project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz')['reps']; assert zz_.shape == (D,)
rng = np.random.default_rng(12345); MS = 300 if QUICK else 20000
dB = Darm['pp2']; sample = {}; CH = 256; tt0 = time.time(); p2 = []
AR = np.arange(N, dtype=np.uint64)
for d in sorted(set(dec.tolist())):
    ids = np.nonzero(dec == d)[0]
    if ids.size > MS: ids = np.sort(rng.choice(ids, MS, replace=False))
    outm, o1, o2 = [], [], []
    for c0 in range(0, ids.size, CH):
        ii = ids[c0:c0 + CH]; reps = zz_[ii].astype(np.uint64); m_ = len(ii)
        sig = (((reps[:, None] >> AR[None, :]) & np.uint64(1)).astype(np.float64) * 2 - 1)
        if m_ < CH: sig = np.concatenate([sig, np.repeat(sig[:1], CH - m_, 0)])
        a, b, c = parts_j(jnp.asarray(sig)); outm.append(np.asarray(a)[:m_]); o1.append(np.asarray(b)[:m_]); o2.append(np.asarray(c)[:m_])
    mR = np.concatenate(outm); S1 = np.concatenate(o1); S2 = np.concatenate(o2)
    lB = mR + np.log(np.abs(S1 + S2)); rho = np.abs(S2) / np.abs(S1)
    sample[d] = dict(ids=ids, lB=lB, rho=rho, S1=S1, S2=S2)
    log(f'decade {d}: {ids.size} orbits of {int((dec == d).sum())}  {time.time() - tt0:.0f}s')
# check vs table
allid = np.concatenate([sample[d]['ids'] for d in sample]); allB = np.concatenate([sample[d]['lB'] for d in sample])
tab_pp2 = load(TABS['pp2'][0]); dv = allB - tab_pp2[allid]; off = float(np.median(dv)); dv -= off
res['part2_check'] = dict(n=int(allid.size), offset=off, max_abs_dev_log=float(np.abs(dv).max()), max_rel_dev_amp=float(np.abs(np.expm1(dv)).max()), pass_1e_8=bool(np.abs(np.expm1(dv)).max() < 1e-8))
from scipy.stats import spearmanr
p2rows = []
for d in sorted(sample):
    s = sample[d]
    if len(s['ids']) < 5: continue
    q = np.percentile(s['rho'], [10, 50, 90]); De = dB[s['ids']]
    sr = spearmanr(s['rho'], De) if len(De) > 2 else None
    srS = spearmanr(np.abs(s['S2']), De) if len(De) > 2 else None
    p2rows.append(dict(decade=int(d), n=int(len(De)), rho_p10=float(q[0]), rho_median=float(q[1]), rho_p90=float(q[2]), spearman_rho_Delta=float(sr.statistic), spearman_p=float(sr.pvalue),
                       spearman_absP2_Delta=float(srS.statistic), spearman_rho_DeltaStar=float(spearmanr(s['rho'], Dst[s['ids']]).statistic)))
res['part2'] = p2rows; dump()
print('check', res['part2_check']); print(f'{"dec":>4} {"n":>6} {"rho10":>9} {"rho50":>9} {"rho90":>9} {"Spearman(rho,Delta)":>20}')
for r in p2rows: print(f'{r["decade"]:>4} {r["n"]:>6} {r["rho_p10"]:9.3e} {r["rho_median"]:9.3e} {r["rho_p90"]:9.3e} {r["spearman_rho_Delta"]:20.3f}')
res['done'] = True; res['sec'] = time.time() - T0; dump(); log('DONE')
