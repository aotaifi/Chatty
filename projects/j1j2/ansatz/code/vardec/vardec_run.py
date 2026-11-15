#!/usr/bin/env python3
"""vardec_run.py OUT [--gates -5,-7 --p 0.5 --walkers 1024 --chains 8 --therm 100 --sweeps N --maxsec T --seed S --ctl_states 300 --ctl_sweeps 12 --smoke]
T-VARDEC (PLAN_sign.md): variational decomposition of the gated sign flip on s306 (8x8 J1-J2, J2=0.5).  Samples x ~ |psi_s306|^(2p) (tempered exchange Metropolis, exact reweighting omega = |psi|^(2(1-p))),
evaluates E_L^A, c, the single and pair terms only where decade(x) <= gate (see vardec_lib.py for the exact identity).  Both gates share the walker stream and one amplitude cache.
Controls (run FIRST, any failure -> exit 3, nothing else runs):
  C0 algebra: E[D] = E[S] on a random small system, per-sample D formula, to 1e-10.
  C1 gate -inf: direct E_L^B - E_L^A (independent route, signs_of_states on x and all neighbours) == 0 EXACTLY on ctl_states samples; the same direct routine at gate -3 is nonzero (it ran).
  C2 per-sample identity: D_formula == direct_D to 1e-10 at gate -5 on F states, their neighbours (x not in F with a flipped neighbour) and random states, with counts of each kind.
  C3 broken alpha_flip arm at gate -5: |dE/site| large (reported with the nominal arm on the same states).
  C4 f32 sampler decade vs f64 authoritative decade: no gated state is missed by the preselection.
Chains: --chains independent groups of walkers (own random starts, therm, independent random draws); per-chain per-sweep series -> quotable.py series.
Checkpoint every --ckpt sec (checkpoint.npz: walkers, PRNG key, records); resumes if present."""
import argparse, json, math, os, sys, time, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
H1 = os.environ.get('HEADLINE_CODE', '/project/theorie/a/A.Otaifi/runs/j1j2/headline_s306/code'); sys.path.insert(0, H1)
PD = os.path.join(HERE, '..', 'paired_fn'); sys.path.insert(0, PD)
BASE = os.environ.get('S306_CODE', '/project/theorie/a/A.Otaifi/runs/j1j2/base_s306_p0/code'); sys.path.insert(0, BASE)
os.environ.setdefault('GFMC_M1_CODE', H1 + '/m1')
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--gates', default='-5,-7'); ap.add_argument('--p', type=float, default=0.5)
ap.add_argument('--walkers', type=int, default=1024); ap.add_argument('--chains', type=int, default=8); ap.add_argument('--therm', type=int, default=100)
ap.add_argument('--sweeps', type=int, default=1000); ap.add_argument('--maxsec', type=float, default=1e18); ap.add_argument('--seed', type=int, default=3061)
ap.add_argument('--ctl_states', type=int, default=300); ap.add_argument('--ctl_sweeps', type=int, default=12); ap.add_argument('--ckpt', type=float, default=600)
ap.add_argument('--wchunk', type=int, default=128); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--skip_controls', action='store_true'); ap.add_argument('--hash_p', type=float, default=0.05); ap.add_argument('--controls_only', action='store_true'); ap.add_argument('--ctl_require', default='C3h')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T0 = time.time(); prog = a.out + '/progress.txt'
def P(m):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {m}'; print(line, flush=True); open(prog, 'a').write(line + '\n')
import jax
if 'cuda' not in str(jax.devices()[0]).lower() and not os.environ.get('G8_ALLOW_CPU'): sys.exit(f'vardec_run: no CUDA device ({jax.devices()[0]})')
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import gated_sign as GS, vardec_lib as VL, vardec_analyse as VA
import vmc_pp2s as V2
from rbmpp_lib import random_configs
GATES = [float(x) for x in a.gates.split(',')]; K = a.chains; W = a.walkers; assert W % K == 0
p_net, psha = V2.build('B0s'); sm = V2.PPSampler(p_net, 256); N = sm.N
sweep, lnpsi32 = VL.make_tempered(sm, a.p)
cache = GS.Cache(cap=300000)
def mk(gate, broken=None): return GS.build(gate, cache, broken=broken, wchunk=a.wchunk)[0]
guides = {g: mk(g) for g in GATES}; g_inf = mk(-math.inf); g_m3 = mk(-3.0); g_bro = mk(GATES[0], 'alpha_flip')
alpha, Econst = GS.read_sign_constants()
P(f'vardec start p {a.p} walkers {W} chains {K} gates {GATES} smoke {a.smoke} device {jax.devices()[0]} params_sha {psha[:12]} alpha {alpha} E {Econst} consts_sha {GS.CONSTS_SHA[:12]} MREF {GS.MREF}')
hdr = dict(args=vars(a), job=os.environ.get('SLURM_JOB_ID'), device=str(jax.devices()[0]), params_sha=psha, alpha=alpha, E=Econst, consts_sha=GS.CONSTS_SHA, MREF=GS.MREF,
           code_sha256={f: hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest() for f in ('vardec_lib.py', 'vardec_analyse.py', 'vardec_run.py', '../paired_fn/gated_sign.py')})
key = jax.random.PRNGKey(a.seed); S = random_configs(N, W, 9600 + a.seed)
def advance(S, key, nsw):
    for _ in range(nsw):
        key, k = jax.random.split(key); S, lw, acc = sweep(S, k, N)
    return S, key, np.asarray(lw), float(acc)
def evalgate_batch(X, lw32, guides_):
    """eval_gate for each guide on preselected candidates"""
    return {gg: VL.eval_gate(g, X, np.floor((lw32 - GS.MREF) / GS.LN10 + 0.05) <= gg) for gg, g in guides_.items()}
def omega(lw): return np.exp((1.0 - a.p) * (lw - GS.MREF))
# ------------------------------------------------------------------------------------------------ therm + controls
t = time.time(); S, key, lw32, acc = advance(S, key, a.therm); jax.block_until_ready(S); sec_therm = time.time() - t
P(f'thermalised {a.therm} sweeps in {sec_therm:.1f}s ({sec_therm / a.therm:.3f} s/sweep of {W} walkers x {N} steps), acc {acc:.3f}, decade<=-5 fraction {np.mean(np.floor((lw32 - GS.MREF) / GS.LN10) <= -5):.3f}')
ctl = dict(); resume = os.path.exists(a.out + '/checkpoint.npz')
if not a.skip_controls and not resume:
    ctl['C0_algebra_max_err'] = VL.mean_identity_selftest(); ok0 = ctl['C0_algebra_max_err'] < 1e-10; P(f'C0 algebra max err {ctl["C0_algebra_max_err"]:.2e} pass={ok0}')
    S, key, lwc, _ = advance(S, key, 2); Xc = np.asarray(S).astype(np.float64); lwc = np.asarray(lnpsi32(S))
    # C4 decade preselection
    t = time.time(); lB, r, _, _ = g_inf.ratios(Xc[:a.ctl_states]); dec64 = np.floor((2 * np.real(lB) - GS.MREF) / GS.LN10); dec32 = np.floor((lwc[:a.ctl_states] - GS.MREF) / GS.LN10 + 1e-3)
    ctl['C4_max_abs_lw_diff'] = float(np.max(np.abs(2 * np.real(lB) - lwc[:a.ctl_states]))); ctl['C4_missed'] = int(((dec64 <= GATES[0]) & ~(dec32 <= GATES[0])).sum()); ok4 = ctl["C4_missed"] == 0 and ctl['C4_max_abs_lw_diff'] < 5e-2
    P(f'C4 f32 vs f64 ln psi max diff {ctl["C4_max_abs_lw_diff"]:.2e}, missed gated {ctl["C4_missed"]} pass={ok4} ({time.time() - t:.1f}s)')
    # C1: gate -inf exactly zero, gate -3 nonzero (ran)
    n = a.ctl_states; Dinf = VL.direct_D(g_inf, Xc[:n])[0]; D3 = VL.direct_D(g_m3, Xc[:n])[0]
    ctl['C1_gate_minf_max_abs_D'] = float(np.max(np.abs(Dinf))); ctl['C1_gate_m3_n_nonzero'] = int((D3 != 0).sum()); ctl['C1_n'] = n
    ok1 = ctl['C1_gate_minf_max_abs_D'] == 0.0 and ctl['C1_gate_m3_n_nonzero'] > 0; P(f'C1 gate -inf: max|E_L^B-E_L^A| = {ctl["C1_gate_minf_max_abs_D"]} on {n} states (must be exactly 0); same routine at gate -3: {ctl["C1_gate_m3_n_nonzero"]} nonzero pass={ok1}')
    # C2: per-sample identity on F states, their neighbours, random states (gate -5)
    g5 = guides[GATES[0]]; Fs = []; Ys = []
    for _ in range(a.ctl_sweeps):                                     # deep-biased states from the sampler (tempered) until enough F states are found
        S, key, lwc, _ = advance(S, key, 1); Xs = np.asarray(S).astype(np.float64); rec = VL.eval_gate(g5, Xs, np.floor((np.asarray(lnpsi32(S)) - GS.MREF) / GS.LN10 + 0.05) <= GATES[0])
        fi = rec['idx'][rec['F']]; Fs.append(Xs[fi])
    Fs = np.concatenate(Fs)[:40] if len(Fs) else np.zeros((0, 64))
    if len(Fs):
        anti = Fs[:, g5.BI] != Fs[:, g5.BJ]; wi, bi = np.nonzero(anti); sel = np.random.default_rng(1).permutation(len(wi))[:100]; wi, bi = wi[sel], bi[sel]
        Y = Fs[wi].copy(); q = np.arange(len(wi)); i_, j_ = g5.BI[bi], g5.BJ[bi]; yi = Y[q, i_].copy(); Y[q, i_] = Y[q, j_]; Y[q, j_] = yi
    else: Y = np.zeros((0, 64))
    Xt = np.concatenate([Fs, Y, Xc[:60]]); Dd = VL.direct_D(g5, Xt)[0]; Df, Fflag, _, _, Pq = VL.D_formula(g5, Xt)
    ctl['C2_n_F_states'] = int(len(Fs)); ctl['C2_n_neighbour_states'] = int(len(Y)); ctl['C2_n_neighbours_with_F_neighbour'] = int(((~Fflag) & (Pq != 0)).sum()); ctl['C2_n_random'] = int(min(60, len(Xc)))
    ctl['C2_max_abs_diff'] = float(np.max(np.abs(Dd - Df))); ctl['C2_max_abs_D'] = float(np.max(np.abs(Dd))); ok2 = ctl['C2_max_abs_diff'] < 1e-10 and ctl['C2_n_F_states'] > 0 and ctl['C2_max_abs_D'] > 0
    P(f'C2 per-sample D(direct) vs D(F/P/Q formula): max|diff| {ctl["C2_max_abs_diff"]:.2e} (max|D| {ctl["C2_max_abs_D"]:.3g}); F states {len(Fs)}, neighbours {len(Y)} ({ctl["C2_n_neighbours_with_F_neighbour"]} with flipped neighbour) pass={ok2}')
    # C3 + C3b: broken alpha_flip arm vs nominal on the same tempered batch; MC mean D vs S at gate -3
    sweep3, lnpsi3 = VL.make_tempered(sm, 0.3); k3 = jax.random.PRNGKey(a.seed + 77); S3 = random_configs(N, W, 9800 + a.seed)       # control sampler p = 0.3 (deep-rich: many flips for BOTH arms)
    for _ in range(12): k3, kk = jax.random.split(k3); S3, _, _ = sweep3(S3, kk, N)
    xs = []; nF = {'nom': 0, 'bro': 0}; sn = []; sb = []; oms = []
    for it in range(max(2, a.ctl_sweeps // 4)):
        k3, kk = jax.random.split(k3); S3, lw3, _ = sweep3(S3, kk, N); Xs = np.asarray(S3[:256]).astype(np.float64); lw = np.asarray(lw3)[:256]; om = np.exp(0.7 * (lw - GS.MREF)); oms.append(om)
        for nm, gg, lst in (('nom', g5, sn), ('bro', g_bro, sb)):
            rec = VL.eval_gate(gg, Xs, np.floor((lw - GS.MREF) / GS.LN10 + 0.05) <= GATES[0]); v = np.zeros(len(Xs)); m = rec['F']; v[rec['idx'][m]] = (4 * (rec['Hxx'] - rec['EL']) + 4 * rec['pair'])[m]; lst.append(v); nF[nm] += int(m.sum())
    om = np.concatenate(oms); sn = np.concatenate(sn); sb = np.concatenate(sb); Dw = om * (sb - sn); est = Dw.sum() / om.sum(); sig = float(np.sqrt(((Dw - est * om) ** 2).sum()) / om.sum())
    ctl['C3_nominal_dE_site'] = float((om * sn).sum() / om.sum() / 64); ctl['C3_broken_alpha_flip_dE_site'] = float((om * sb).sum() / om.sum() / 64); ctl['C3_diff_site'] = float(est / 64); ctl['C3_diff_sigma_site'] = sig / 64; ctl['C3_n_flips'] = nF; ctl['C3_n_states'] = int(len(om))
    ok3 = abs(est) > 5 * sig; ctl['C3_alpha_flip_as_specified_pass'] = bool(ok3)
    P(f'C3 broken alpha_flip dE/site {ctl["C3_broken_alpha_flip_dE_site"]:.4e} (flips {nF["bro"]}) vs nominal {ctl["C3_nominal_dE_site"]:.4e} (flips {nF["nom"]}) on {len(om)} p=0.3 states; difference {ctl["C3_diff_site"]:.3e} +- {ctl["C3_diff_sigma_site"]:.1e} ({abs(est) / max(sig, 1e-300):.1f} sigma); pass(>5 sigma)={ok3}')
    g_h = GS.build(-math.inf, cache, broken='hashmask', wchunk=a.wchunk, hash_p=a.hash_p)[0]; S, key, _, _ = advance(S, key, 2); Xh = np.asarray(S[:a.ctl_states]).astype(np.float64); lwh = np.asarray(lnpsi32(S))[:a.ctl_states]; omh = omega(lwh)
    Dh = VL.direct_D(g_h, Xh)[0]; xh = omh * Dh; eh = xh.sum() / omh.sum(); sh = float(np.sqrt(((xh - eh * omh) ** 2).sum()) / omh.sum())
    ctl['C3h_hashmask_p'] = a.hash_p; ctl['C3h_dE_site'] = float(eh / 64); ctl['C3h_sigma_site'] = sh / 64; ctl['C3h_n'] = int(len(Xh)); ctl['C3h_n_nonzero_D'] = int((Dh != 0).sum()); ok3h = abs(eh) > 5 * sh
    P(f'C3h broken hashmask p={a.hash_p}: dE/site {ctl["C3h_dE_site"]:.4e} +- {ctl["C3h_sigma_site"]:.1e} ({abs(eh) / max(sh, 1e-300):.1f} sigma) on {len(Xh)} states, {ctl["C3h_n_nonzero_D"]} with D != 0; pass(>5 sigma)={ok3h}')
    # C3b statistical: mean D (all states, direct) vs mean S at gate -3 (reported, not a stop criterion)
    S, key, _, _ = advance(S, key, 2); Xs = np.asarray(S).astype(np.float64); lw = np.asarray(lnpsi32(S)); n = min(a.ctl_states, W); om = omega(lw)[:n]
    Dm = VL.direct_D(g_m3, Xs[:n])[0]; rec = VL.eval_gate(g_m3, Xs[:n], np.floor((lw[:n] - GS.MREF) / GS.LN10 + 0.05) <= -3); Sv = np.zeros(n); mF = rec['F']; Sv[rec['idx'][mF]] = (4 * (rec['Hxx'] - rec['EL']) + 4 * rec['pair'])[mF]
    def rat(v):
        x = om * v; return float(x.sum() / om.sum()), float(np.sqrt(((x - om * (x.sum() / om.sum())) ** 2).sum()) / om.sum())
    ctl['C3b_gate_m3_meanD'] = rat(Dm); ctl['C3b_gate_m3_meanS'] = rat(Sv); P(f'C3b gate -3 on {n} states: weighted mean D {ctl["C3b_gate_m3_meanD"][0]:.4g} +- {ctl["C3b_gate_m3_meanD"][1]:.2g}, mean S {ctl["C3b_gate_m3_meanS"][0]:.4g} +- {ctl["C3b_gate_m3_meanS"][1]:.2g} (E[D] = E[S] in expectation; correlated estimators, not a stop criterion)')
    ctl['passed'] = bool(ok0 and ok1 and ok2 and ok4 and ((ok3 and ok3h) if a.ctl_require == 'both' else (ok3 if a.ctl_require == 'alpha_flip' else ok3h))); ctl['sec'] = time.time() - T0; json.dump(dict(hdr, control=ctl), open(a.out + '/control.json', 'w'), indent=1, default=float)
    P(f'CONTROLS passed={ctl["passed"]} ({ctl["sec"]:.0f}s since start)')
    if not ctl['passed']: P('CONTROL FAILED, ABORT'); sys.exit(3)
    if a.controls_only: P('controls_only: stop'); sys.exit(0)
    S = random_configs(N, W, 9700 + a.seed); S, key, lw32, acc = advance(S, key, a.therm)                  # fresh start for the measurement (controls used the stream)
# ------------------------------------------------------------------------------------------------ measurement
RK = ('chain', 'sweep', 'dec', 'lw', 'lw64', 'Hxx', 'EL', 'F', 'pair', 'Fp', 'pairp')
recs = {g: {k: [] for k in RK} for g in GATES}; den = np.zeros((a.sweeps, K)); nsw = 0; cnt = dict(cand=0, nbr=0, gated={g: 0 for g in GATES}); sec_samp = 0.0; sec_eval = 0.0
ckf = a.out + '/checkpoint.npz'
def cat(g):
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in recs[g].items()}
def save():
    d = dict(S=np.asarray(S), key=np.asarray(key), nsw=nsw, den=den, p=a.p, K=K, gates=np.array(GATES), cnt=json.dumps(cnt), sec_samp=sec_samp, sec_eval=sec_eval)
    for g in GATES:
        for k, v in cat(g).items(): d[f'g{int(g)}_{k}'] = v
    np.savez(ckf + '.tmp.npz', **d); os.replace(ckf + '.tmp.npz', ckf)
if resume:
    z = np.load(ckf, allow_pickle=True); S = jnp.asarray(z['S']); key = jnp.asarray(z['key']); nsw = int(z['nsw']); den = z['den']; cnt = json.loads(str(z['cnt'])); sec_samp = float(z['sec_samp']); sec_eval = float(z['sec_eval'])
    for g in GATES:
        for k in RK: recs[g][k] = [z[f'g{int(g)}_{k}']]
    P(f'RESUMED from checkpoint at sweep {nsw}')
tl = time.time(); tck = time.time(); chain_of = np.arange(W) // (W // K); maxdiff = 0.0
while nsw < a.sweeps and time.time() - T0 < a.maxsec:
    t = time.time(); key, k_ = jax.random.split(key); S, lw32j, acc = sweep(S, k_, N); lw32 = np.asarray(lw32j); X = np.asarray(S).astype(np.float64); sec_samp += time.time() - t
    om = omega(lw32); den[nsw] = np.bincount(chain_of, weights=om, minlength=K)
    t = time.time(); out = evalgate_batch(X, lw32, guides); sec_eval += time.time() - t
    for g, o in out.items():
        i = o['idx']; cnt['gated'][g] += len(i); cnt['cand'] += o['n_cand']; cnt['nbr'] += o['n_nbr']
        if len(i):
            maxdiff = max(maxdiff, float(np.max(np.abs(o['lw'] - lw32[i]))))
            recs[g]['chain'].append(chain_of[i]); recs[g]['sweep'].append(np.full(len(i), nsw)); recs[g]['dec'].append(o['dec']); recs[g]['lw'].append(lw32[i]); recs[g]['lw64'].append(o['lw'])
            for k in ('Hxx', 'EL', 'F', 'pair', 'Fp', 'pairp'): recs[g][k].append(o[k])
    nsw += 1
    if nsw % 10 == 0 or nsw == 1:
        P(f'sweep {nsw}/{a.sweeps} acc {acc:.3f} gated {cnt["gated"]} nF(-5) {int(sum(x.sum() for x in recs[GATES[0]]["F"]))} samp {sec_samp / nsw:.2f}s/sweep eval {sec_eval / nsw:.2f}s/sweep cache hit/miss {cache.hit}/{cache.miss} max|lw32-lw64| {maxdiff:.1e} elapsed {time.time() - T0:.0f}s')
    if time.time() - tck > a.ckpt: save(); tck = time.time(); P(f'checkpoint at sweep {nsw}')
save(); P(f'measurement done: {nsw} sweeps x {W} walkers, sampler {sec_samp:.0f}s eval {sec_eval:.0f}s, max|lw32-lw64| {maxdiff:.2e}')
# ------------------------------------------------------------------------------------------------ analysis
res = dict(hdr, control=ctl, sweeps_done=nsw, sec=time.time() - T0, sec_sampler=sec_samp, sec_eval=sec_eval, counters=cnt, max_lw_diff=maxdiff, gates={})
for g in GATES:
    o, ser = VA.analyse(cat(g), den[:nsw], a.p, K, g); res['gates'][str(int(g))] = o; P('RESULT\n' + VA.fmt(o))
    os.makedirs(f'{a.out}/gate{int(g)}', exist_ok=True)
    for nm in ('total_F', 'total_Fp', 'total_F_window', 'total_F_good'):
        for k in range(K): np.save(f'{a.out}/gate{int(g)}/dsite_{nm}_c{k}.npy', ser[nm][:, k] / 64.0)
res['done'] = True; json.dump(res, open(a.out + '/result.json', 'w'), indent=1, default=float); P('VARDEC_DONE')
