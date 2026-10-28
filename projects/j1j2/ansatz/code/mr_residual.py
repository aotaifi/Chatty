"""mr_residual.py: pre-registered test M-R (PLAN_pfaffian.md, "Proposal M-R" + "M-R additions"), CPU only.
usage: python3 mr_residual.py OUTDIR [ALPHA ...]        (default alphas 4 16)
Stages (each cached in OUTDIR, restartable):
  1  phi' = FN(A_s1): delta0.efn_vec twice (same operator/start/tol as delta0.py); controls vs delta0.json; saves lphi.npy
  2  R* = ln phi' - ln A_s1 ; one pass over all blocked pairs: row weights, Wtot, C(R=0), D(R*,R*)
  3  per alpha: symmetric RBM residual R_theta (576 images), Levenberg-Marquardt on sqrt(w)(e_x-e_y), pairs sampled prop. to w
  4  exact readouts: R_theta on all reps, C(e), D(.,.), E_FN[A_s1 e^R]; controls R=0, R=R*, R=-R*
Blocked pair = (rep x, bond) with s2_x s2_y > 0 (cost_pass 'viol').  Weight of an entry (rep r, bond k, neighbour rep y):
  w = n_r * 0.5 * HW_k * phi'_r phi'_y  (phi' per-config amplitude, sum n phi'^2 = 1)  -- each unordered pair seen from both ends, as cost_pass.
C(e) = sum w * 2 (cosh(e_x - e_y) - 1);   D(a,b) = sum w (a_x - a_y)(b_x - b_y).
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import bond_split as BS
import delta0 as D0M
from rbmpp_lib import Lattice

OUT = sys.argv[1]; ALPHAS = [int(a) for a in sys.argv[2:]] or [4, 16]
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(m):
    print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)

A0 = D0M.A0
REPS = '/project/theorie/a/A.Otaifi/Chatty/runs/ll6/data/psi0_6x6_table.npz'
E0 = BS.E0; NS = BS.NS; HW = BS.HW
DEL = json.load(open('/scratch/a/A.Otaifi/ansatz_runs/pfaffian_g/delta0/delta0.json'))
REF_COPY = DEL['copy_s1']['first']['dE_site']; REF_PHI = DEL['copy_s1']['second']['dE_site']
RS = 65536
B_TRAIN = 8192; B_HO = 16384; MAXSTEP = 2000; HO_EVERY = 10; FLAT_WIN = 50; FLAT_TOL = 0.02; MINSTEP = 100

n = np.load(f'{BS.CSRD}/norb.npy').astype(np.float64); s = np.load(BS.S2).astype(np.float64)
D = len(n); assert n.sum() == 9075135300 and set(np.unique(s)) <= {-1.0, 1.0}
t1 = np.load(f'{BS.TAB}/nb_J1.npy', mmap_mode='r'); t2 = np.load(f'{BS.TAB}/nb_J2.npy', mmap_mode='r')
lA = np.load(A0).astype(np.float64)
reps = np.load(REPS)['reps']; assert reps.shape == (D,) and np.all(reps[1:] > reps[:-1])
res_all = dict(A0=A0, E0=E0, ref_copy_dE_site=REF_COPY, ref_phi_dE_site=REF_PHI, host=os.uname().nodename)
def dump():
    res_all['sec'] = time.time() - T0; json.dump(res_all, open(f'{OUT}/mr_result.json', 'w'), indent=1, default=float)

# ---------------------------------------------------------------- stage 1: phi'
K = Vm = D0 = None
def get_ops():
    global K, Vm, D0
    if K is None:
        K, Vm, D0 = BS.build(n, s); log(f'built K nnz {K.nnz}')
    return K, Vm, D0

def efn_la(la, tag):
    K_, Vm_, D0_ = get_ops()
    t = time.time(); r, v = D0M.efn_vec(K_, Vm_, D0_, n, la); r['sec'] = time.time() - t
    log(f'{tag}: dE_site {r["dE_site"]:.12e} res {r["res"]:.1e} it {r["iters"]} {r["sec"]:.0f}s'); return r, v

LP = f'{OUT}/lphi.npy'
if os.path.exists(LP) and os.path.exists(f'{OUT}/stage1.json'):
    lphi = np.load(LP); res_all['stage1'] = json.load(open(f'{OUT}/stage1.json')); log('lphi loaded')
else:
    r1, v = efn_la(lA, 'E_FN[A_s1]')
    chi = np.abs(v) / np.sqrt(n); lphi = np.log(np.maximum(chi, 1e-300))
    r2, _ = efn_la(lphi, "E_FN[phi']")
    st = dict(E_FN_A=r1, E_FN_phi=r2, rel_ctl_A=(r1['dE_site'] - REF_COPY) / REF_COPY, rel_ctl_phi=(r2['dE_site'] - REF_PHI) / REF_PHI,
              gain=r1['dE_site'] - r2['dE_site'])
    st['controls_pass'] = bool(abs(st['rel_ctl_A']) < 1e-6 and abs(st['rel_ctl_phi']) < 1e-6)
    log(f'stage1 {st}'); np.save(LP, lphi); json.dump(st, open(f'{OUT}/stage1.json', 'w'), indent=1); res_all['stage1'] = st
    if not st['controls_pass']: dump(); log('CONTROL FAIL: STOP'); sys.exit(2)
dump()
gain = res_all['stage1']['gain']; E_PHI = res_all['stage1']['E_FN_phi']['dE_site']

# ---------------------------------------------------------------- stage 2: R*, weights
lp = lphi - lphi.max(); lp = lp - 0.5 * np.log((n * np.exp(2 * lp)).sum())
Rstar = lphi - lA
log(f'R*: min {Rstar.min():.3f} max {Rstar.max():.3f}')

def pass_all(elist, want_Wr=False):
    """over all blocked pairs: for each e in elist: C(e), D(e,R*), D(e,e); also D(R*,R*), Wtot, (row weights Wr)."""
    ne = len(elist); C = np.zeros(ne); DeR = np.zeros(ne); Dee = np.zeros(ne); DRR = 0.0; Wtot = 0.0; npair = 0
    Wr = np.zeros(D) if want_Wr else None
    for i0 in range(0, D, RS):
        i1 = min(D, i0 + RS)
        t = np.concatenate([t1[i0:i1], t2[i0:i1]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
        viol = valid & (s[i0:i1, None] * s[tt] > 0); rr, kk = np.nonzero(viol); x = rr + i0; y = tt[rr, kk]
        w = n[x] * 0.5 * HW[kk] * np.exp(lp[x] + lp[y])
        dR = Rstar[x] - Rstar[y]
        Wtot += w.sum(); DRR += (w * dR * dR).sum(); npair += w.size
        if want_Wr: Wr[i0:i1] = np.bincount(rr, weights=w, minlength=i1 - i0)
        for j, e in enumerate(elist):
            d = e[x] - e[y]; C[j] += (w * 2 * (np.cosh(np.clip(d, -50, 50)) - 1)).sum()
            DeR[j] += (w * d * dR).sum(); Dee[j] += (w * d * d).sum()
        if (i0 // RS) % 40 == 0: log(f'pass {i1}/{D}')
    return dict(C=C, DeR=DeR, Dee=Dee, DRR=DRR, Wtot=Wtot, npair=npair, Wr=Wr)

S2F = f'{OUT}/stage2.npz'
if os.path.exists(S2F):
    z = np.load(S2F); Wr = z['Wr']; Wtot = float(z['Wtot']); DRR = float(z['DRR']); C0 = float(z['C0']); NPAIR = int(z['npair'])
else:
    r = pass_all([-Rstar], want_Wr=True)
    Wr, Wtot, DRR, C0, NPAIR = r['Wr'], r['Wtot'], r['DRR'], r['C'][0], r['npair']
    np.savez(S2F, Wr=Wr, Wtot=Wtot, DRR=DRR, C0=C0, npair=NPAIR)
res_all['stage2'] = dict(Wtot=Wtot, D_RR=DRR, C_R0=C0, n_blocked_entries=NPAIR, rms_Rstar_bond=float(np.sqrt(DRR / Wtot)))
log(f'stage2 {res_all["stage2"]}'); dump()
cumWr = np.cumsum(Wr); cumWr /= cumWr[-1]

# ---------------------------------------------------------------- pair sampler (prop. to w)
def draw_pairs(rng, M):
    r = np.minimum(np.searchsorted(cumWr, rng.random(M)), D - 1)
    order = np.argsort(r); r = r[order]
    t = np.concatenate([t1[r], t2[r]], 1); valid = t >= 0; tt = np.where(valid, t, 0)
    viol = valid & (s[r, None] * s[tt] > 0)
    w = np.where(viol, n[r, None] * 0.5 * HW[None, :] * np.exp(lp[r, None] + lp[tt]), 0.0)
    cw = np.cumsum(w, 1); u = rng.random(M) * cw[:, -1]
    k = np.minimum((cw < u[:, None]).sum(1), 143)
    return r, tt[np.arange(M), k]

def bits_of(idx):
    return (((reps[idx][:, None] >> np.arange(36, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.float64) * 2 - 1)

# ---------------------------------------------------------------- RBM residual
lat = Lattice(6, 0.5, 'full')
IMG = lat.IMG; INV = lat.INV; FLP = lat.FL.astype(jnp.float64)

def make_model(alpha):
    P = alpha * 37
    def R_one(theta, sig):
        b = theta[:alpha]; W = theta[alpha:].reshape(alpha, 36)
        Z = FLP[:, None] * sig[IMG]
        a = Z @ W.T + b[None, :]
        return jnp.sum(jnp.abs(a) + jnp.log1p(jnp.exp(-2 * jnp.abs(a))) - jnp.log(2.0))
    vg = jax.vmap(jax.value_and_grad(R_one), in_axes=(None, 0))
    @jax.jit
    def val_grad(theta, S):                      # S (M,36) -> R (M,), G (M,P), chunks of 256
        Sc = S.reshape(-1, 256, 36)
        out = jax.lax.map(lambda sc: vg(theta, sc), Sc)
        return out[0].reshape(-1), out[1].reshape(-1, P)
    @jax.jit
    def val(theta, S):
        return jax.lax.map(lambda sc: jax.vmap(R_one, in_axes=(None, 0))(theta, sc), S.reshape(-1, 256, 36)).reshape(-1)
    @jax.jit
    def table_chunk(theta, S):                   # GEMM form with permuted weights (checked against R_one)
        b = theta[:alpha]; W = theta[alpha:].reshape(alpha, 36)
        Wg = (W[:, INV] * FLP[None, :, None])                       # (alpha, 576, 36)
        Wm = jnp.transpose(Wg, (2, 1, 0)).reshape(36, 576 * alpha)  # column = g*alpha + f
        a = (S @ Wm).reshape(S.shape[0], 576, alpha) + b[None, None, :]
        return jnp.sum(jnp.abs(a) + jnp.log1p(jnp.exp(-2 * jnp.abs(a))) - jnp.log(2.0), axis=(1, 2))
    return P, R_one, val_grad, val, table_chunk

def full_table(table_chunk, theta, ch=4096):
    out = np.empty(D)
    for i0 in range(0, D, ch):
        i1 = min(D, i0 + ch); S = bits_of(np.arange(i0, i1))
        if S.shape[0] < ch: S = np.concatenate([S, np.zeros((ch - S.shape[0], 36))])
        out[i0:i1] = np.asarray(table_chunk(theta, jnp.asarray(S)))[:i1 - i0]
        if (i0 // ch) % 800 == 0: log(f'table {i1}/{D}')
    return out

def fit(alpha):
    P, R_one, val_grad, val, table_chunk = make_model(alpha)
    rng_th = np.random.default_rng(100 + alpha); theta = np.concatenate([np.zeros(alpha), 0.03 * rng_th.standard_normal(alpha * 36)])
    # model checks: invariance and GEMM-vs-direct
    sg = bits_of(np.arange(0, D, D // 256)[:256]); thr = jnp.asarray(0.3 * rng_th.standard_normal(P))
    Rd = np.asarray(val(thr, jnp.asarray(sg))); Rg = np.asarray(table_chunk(thr, jnp.asarray(sg)))
    G = np.asarray(lat.G); perm = G[35]
    Rp = np.asarray(val(thr, jnp.asarray(sg[:, perm]))); Rf = np.asarray(val(thr, jnp.asarray(-sg)))
    chk = dict(gemm_vs_direct=float(np.abs(Rd - Rg).max()), group_invariance=float(np.abs(Rd - Rp).max()), flip_invariance=float(np.abs(Rd - Rf).max()))
    log(f'alpha {alpha} model checks {chk}'); assert max(chk.values()) < 1e-9, chk
    rho = np.random.default_rng(777); xh, yh = draw_pairs(rho, B_HO)
    Sh = jnp.asarray(np.concatenate([bits_of(xh), bits_of(yh)])); dsh = jnp.asarray(Rstar[xh] - Rstar[yh])
    C0h = float(np.mean(2 * (np.cosh(np.clip(np.asarray(dsh), -50, 50)) - 1)))      # R=0 on the held-out batch
    def Lbatch(th, S, ds):
        R = val(th, S); m = ds.shape[0]; d = R[:m] - R[m:] - ds
        return float(jnp.mean(2 * (jnp.cosh(jnp.clip(d, -50, 50)) - 1)))
    rng = np.random.default_rng(5000 + alpha); lam = 1.0; hist = []; reason = 'maxstep'
    th = jnp.asarray(theta); t0 = time.time()
    for step in range(1, MAXSTEP + 1):
        x, y = draw_pairs(rng, B_TRAIN); S = jnp.asarray(np.concatenate([bits_of(x), bits_of(y)])); ds = jnp.asarray(Rstar[x] - Rstar[y])
        Rv, Gm = val_grad(th, S); m = B_TRAIN
        d = Rv[:m] - Rv[m:] - ds; J = Gm[:m] - Gm[m:]
        L_old = float(jnp.mean(2 * (jnp.cosh(jnp.clip(d, -50, 50)) - 1)))
        g = J.T @ d; H = J.T @ J; dg = jnp.diag(H) + 1e-12 * float(jnp.trace(H)) / P
        acc = False; Ln = L_old
        for _ in range(8):
            dl = -jnp.linalg.solve(H + lam * jnp.diag(dg), g)
            thn = th + dl; Ln = Lbatch(thn, S, ds)
            if Ln < L_old: th, acc, lam = thn, True, max(lam * 0.5, 1e-6); break
            lam *= 4.0
        if step % HO_EVERY == 0 or step == 1:
            Lh = Lbatch(th, Sh, dsh); hist.append((step, Lh / C0h))
            log(f'a{alpha} step {step} batch L {L_old:.4e}->{Ln if acc else L_old:.4e} lam {lam:.1e} acc {acc} ho eps2 {Lh / C0h:.5f} ({time.time() - t0:.0f}s)')
            hh = {k: v for k, v in hist}
            if step >= MINSTEP and step - FLAT_WIN in hh and abs(hh[step] - hh[step - FLAT_WIN]) / hh[step] < FLAT_TOL:
                reason = f'flat (<{FLAT_TOL:.0%} over {FLAT_WIN} steps) at step {step}'; break
    return np.asarray(th), dict(steps=step, stop=reason, ho_eps2_hist=hist, checks=chk, n_params=P, C0_ho_batch=C0h, fit_sec=time.time() - t0), table_chunk

# ---------------------------------------------------------------- stage 4
def energy_of_R(Rtab, tag):
    r, _ = efn_la(lA + Rtab, tag); return r

CTL = f'{OUT}/controls.json'
ctl = json.load(open(CTL)) if os.path.exists(CTL) else {}
for key, Rt_, tag in (('R0', np.zeros(D), 'control R=0 (E_FN[A_s1])'), ('Rstar', Rstar, 'control R=R*'), ('Rneg', -Rstar, 'control R=-R*')):
    if key not in ctl:
        ctl[key] = energy_of_R(Rt_, tag); json.dump(ctl, open(CTL, 'w'), indent=1)
for k in ('R0', 'Rstar', 'Rneg'):
    ctl[k]['tax_site'] = ctl[k]['dE_site'] - E_PHI; ctl[k]['tax_over_gain'] = ctl[k]['tax_site'] / gain
res_all['controls'] = ctl; log('controls ' + json.dumps({k: (v['dE_site'], v['tax_over_gain']) for k, v in ctl.items()})); dump()

for alpha in ALPHAS:
    TF = f'{OUT}/theta_a{alpha}.npy'; RF = f'{OUT}/R_a{alpha}.npy'; JF = f'{OUT}/fit_a{alpha}.json'
    P, _, _, _, table_chunk = make_model(alpha)
    if os.path.exists(TF) and os.path.exists(JF):
        th = np.load(TF); info = json.load(open(JF)); log(f'alpha {alpha}: theta loaded')
    else:
        th, info, table_chunk = fit(alpha); np.save(TF, th); json.dump(info, open(JF, 'w'), indent=1, default=float)
    if os.path.exists(RF): Rt = np.load(RF)
    else:
        Rt = full_table(table_chunk, jnp.asarray(th)); np.save(RF, Rt)
    e = Rt - Rstar
    p = pass_all([e]); C = p['C'][0]; DeR = p['DeR'][0]; Dee = p['Dee'][0]
    eps_par = -DeR / DRR; perp = (Dee + 2 * eps_par * DeR + eps_par ** 2 * DRR) / DRR
    en = energy_of_R(Rt, f'alpha {alpha} E_FN[A_s1 e^R]')
    tax = en['dE_site'] - E_PHI
    out = dict(alpha=alpha, n_params=P, steps=info['steps'], stop=info['stop'], ho_eps2_last=info['ho_eps2_hist'][-3:], E_FN=en,
               tax_site=tax, tax_over_gain=tax / gain, eps_C2=C / C0, C=C, C_R0=C0, eps_par=eps_par, eps_par2=eps_par ** 2, perp_share=perp,
               ratio_taxgain_over_epsC2=(tax / gain) / (C / C0),
               verdict_pred=bool(tax <= 0.3 * gain and 0.5 <= (tax / gain) / (C / C0) <= 2.0), verdict_kill=bool(tax >= 0.7 * gain or not (1 / 3 <= (tax / gain) / (C / C0) <= 3.0)))
    res_all[f'alpha{alpha}'] = out; log(f'RESULT alpha {alpha}: {json.dumps(out, default=float)}'); dump()
res_all['gain_site'] = gain; res_all['status'] = 'DONE'; dump(); log('DONE')
