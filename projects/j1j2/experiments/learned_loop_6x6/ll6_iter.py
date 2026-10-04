"""One iteration of the LEARNED FN/Krylov loop on 6x6 J1-J2 (J2/J1 = 0.5).

Input guide (a_k, s_k) = state JSON {chain_params, Ts, amp_g}; 'base' = the 6x6 ViT checkpoint.
  s_k = s^(K) (Krylov chain from Marshall, see ll6_core), a_k = |ViT(theta_k)|.
Stage A  learn a_{k+1} = |ViT(theta)| = a_k exp(r_theta), r_theta = log|psi_theta| - log|psi_theta_k|
         (fine-tune of the real log-amplitude of the ViT, warm start r = 0) by SR on the FROZEN FN energy
         L(theta) = <a_theta s_k| H_FN[a_k, s_k] |a_theta s_k> / <a_theta|a_theta>.
         Every SR step draws FRESH samples from |a_theta|^2 (persistent bond-exchange Metropolis chains),
         independent train and validation chains.  Direction: (S + shift) d = g on the train half,
         g = 2<(E_L - L)(O - <O>)>, solved exactly in the n x n sample space.  Step length: trust region
         RMS_train(delta log a) = trust.  Accept only if the VALIDATION estimate (same val samples,
         reweighted by exp(2 delta r): paired) decreases; backtracking factors 1,1/2,1/4,1/8; stop at the
         first rejection or after --sr-steps.
Stage B  independent evaluation sets: A ~ |a_k|^2, B ~ |a_{k+1}|^2 (fresh chains):
         L before/after (unpaired and paired), <H>_{a_{k+1} s_k}.
Stage C  Krylov sign step on (a_{k+1}, s_k): r = (H a_{k+1} s_k)/(a_{k+1} s_k), s_{k+1} = s_k sgn(T - r),
         T label-free = argmin of the sampled variational energy of a_{k+1} s_{k+1} on an independent
         training sample of |a_{k+1}|^2 ("no flip" is a candidate).  Held-out check on set B:
         <H>_{a_{k+1} s_{k+1}} vs <H>_{a_{k+1} s_k} (paired).
Writes  <out>_params.npy (theta_{k+1}), <out>_state.json (new guide), <out>_state_amponly.json
        (a_{k+1} with the old signs), <out>.json (all numbers).
"""
import argparse, json, time, os
import numpy as np
import ll6_core as C
import jax, jax.numpy as jnp

ap = argparse.ArgumentParser()
ap.add_argument('--state', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--dtype', default='float32')
ap.add_argument('--ntr', type=int, default=2048)
ap.add_argument('--nva', type=int, default=2048)
ap.add_argument('--sr-steps', type=int, default=15)
ap.add_argument('--shift', type=float, default=0.05)
ap.add_argument('--trust', type=float, default=0.01)
ap.add_argument('--between', type=int, default=4)
ap.add_argument('--burn', type=int, default=100)
ap.add_argument('--ess-min', type=float, default=0.5)
ap.add_argument('--neval-chains', type=int, default=2048)
ap.add_argument('--neval-rounds', type=int, default=2)
ap.add_argument('--nthr-chains', type=int, default=512)
ap.add_argument('--nthr-rounds', type=int, default=2)
ap.add_argument('--chunk-local', type=int, default=256)
ap.add_argument('--chunk-thr', type=int, default=16)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--skip-sign', action='store_true')
args = ap.parse_args()
T0 = time.time()
res = dict(args=vars(args))
st = json.load(open(args.state)); res['state_in'] = st
net = C.Net('vit_J2=0.50_N=6x6_k=0.mpack', dtype=args.dtype)
C.log('backend', jax.default_backend(), 'npar', net.npar)


def load_flat(p):
    return net.flat0 if p == 'base' else jnp.asarray(np.load(p), net.DT)


amp_objs = {}
def amp_of(p):
    if p not in amp_objs: amp_objs[p] = C.Amp(net, load_flat(p), name=p)
    return amp_objs[p]


guide = C.Guide([amp_of(p) for p in st['chain_params']], st['Ts'], amp_of(st['amp_g']))
theta0 = load_flat(st['amp_g']); theta = theta0
K = guide.K
C.log('guide K', K, 'Ts', guide.Ts)


def resid(th, d, X):
    return net.logabs(th, X) - d['lax'], net.logabs(th, d['child']) - d['lac']


def chain_mean_se(v, nch):
    v = np.asarray(v, float); cm = v.reshape(-1, nch).mean(0)
    return float(v.mean()), float(cm.std(ddof=1) / np.sqrt(nch))


def wmean_se(v, w, nch):
    """self-normalised weighted mean; SE by chain blocks (ratio estimator, delta method)."""
    v = np.asarray(v, float); w = np.asarray(w, float); m = float(np.sum(w * v) / np.sum(w))
    num = (w * (v - m)).reshape(-1, nch).sum(0); den = w.reshape(-1, nch).sum(0)
    se = float(np.sqrt(np.sum(num ** 2)) / den.sum())
    return m, se


# ------------------------------------------------------------------ Stage A: frozen-H_FN SR
ch_tr = C.Chains(net, args.ntr, 1000 * args.seed + 1)
ch_va = C.Chains(net, args.nva, 1000 * args.seed + 2)
ch_tr.advance(theta, args.burn); ch_va.advance(theta, args.burn)
rows = []; trust = args.trust; stop = 'max_steps'; nacc = 0; nsamp = 0; L_tr0 = L_va0 = None
for step in range(1, args.sr_steps + 1):
    t0 = time.time(); n0 = net.neval
    Xtr = ch_tr.advance(theta, args.between); Xva = ch_va.advance(theta, args.between)
    nsamp += len(Xtr) + len(Xva)
    dtr = C.local_chunked(guide, Xtr, args.chunk_local); dva = C.local_chunked(guide, Xva, args.chunk_local)
    t1 = time.time()
    rx, rc = resid(theta, dtr, Xtr); EL, EH = C.trial_elocs(dtr, rx, rc)
    Ltr = float(EL.mean())
    rxv, rcv = resid(theta, dva, Xva); ELv, _ = C.trial_elocs(dva, rxv, rcv); Lva = float(ELv.mean())
    if L_tr0 is None: L_tr0, L_va0 = Ltr, Lva
    n = len(Xtr)
    O = net.jac(theta, Xtr)
    Ob = (O - O.mean(0, keepdims=True)) / float(np.sqrt(n))   # python float keeps float32
    del O
    eps = jnp.asarray((EL - Ltr) / np.sqrt(n), Ob.dtype)
    Km = np.asarray(Ob @ Ob.T, np.float64)
    alpha = np.linalg.solve(Km + args.shift * np.eye(n), 2.0 * np.asarray(eps, np.float64))
    d = Ob.T @ jnp.asarray(alpha, Ob.dtype)
    rms = float(jnp.linalg.norm(Ob @ d))
    gnorm = float(jnp.linalg.norm(2.0 * (Ob.T @ eps)))
    del Ob
    eta = trust / max(rms, 1e-300)
    t2 = time.time()
    acc = None; cands = []
    for fac in (1.0, 0.5, 0.25, 0.125):
        th = theta - eta * fac * d
        rx1, rc1 = resid(th, dva, Xva); EL1, _ = C.trial_elocs(dva, rx1, rc1)
        lw = 2.0 * (rx1 - rxv); w = np.exp(lw - lw.max()); w /= w.sum()
        L1 = float(np.sum(w * EL1)); ess = float(1.0 / np.sum(w * w) / len(w))
        dl = rx1 - rxv; drms = float(np.sqrt(np.mean((dl - dl.mean()) ** 2)))
        cands.append(dict(fac=fac, Lva_new=L1, dLva=L1 - Lva, ess=ess, drms_va=drms))
        if L1 < Lva and ess >= args.ess_min:
            acc = (th, fac); break
    row = dict(step=step, L_tr=Ltr, L_va=Lva, EH_tr=float(EH.mean()), gnorm=gnorm, rms_per_eta=rms, eta=eta,
               trust=trust, cands=cands, t_local=t1 - t0, t_sr=t2 - t1, t_ls=time.time() - t2,
               evals=net.neval - n0, edges_tr=int(len(dtr['child'])),
               frac_allowed=float(dtr['allowed'].mean()))
    if acc is None:
        row['accepted'] = False; rows.append(row); stop = 'val_reject'
        C.log('SR', json.dumps(row)); break
    theta, fac = acc; nacc += 1
    if fac < 1.0: trust *= fac
    ch_tr.reset_la(theta); ch_va.reset_la(theta)
    row.update(accepted=True, factor=fac); rows.append(row)
    C.log('SR', json.dumps(row))
res['sr'] = dict(rows=rows, n_accepted=nacc, stop=stop, samples_used=nsamp, L_tr_start=L_tr0, L_va_start=L_va0,
                 accept_rate_tr=ch_tr.acc / max(ch_tr.props, 1), sec=time.time() - T0)
np.save(args.out + '_params.npy', np.asarray(theta))
theta1 = theta
dlog_total = None

# ------------------------------------------------------------------ Stage B: independent evaluation
t0 = time.time()
nch = args.neval_chains
def eval_draw(th, seed):
    ch = C.Chains(net, nch, seed); ch.advance(th, args.burn)
    return np.concatenate([ch.advance(th, args.between) for _ in range(args.neval_rounds)])

XA = eval_draw(theta0, 1000 * args.seed + 11)
XB = eval_draw(theta1, 1000 * args.seed + 12)
dA = C.local_chunked(guide, XA, args.chunk_local); dB = C.local_chunked(guide, XB, args.chunk_local)
ev = {}
# set A ~ a_k^2: L(theta_k) = <H>_{a_k s_k};  L(theta_{k+1}) paired by reweighting
rxA, rcA = resid(theta1, dA, XA); EL1A, EH1A = C.trial_elocs(dA, rxA, rcA)
wA = np.exp(2 * (rxA - rxA.max()))
ev['A_L0'] = chain_mean_se(dA['eh'], nch)
ev['A_L1_rw'] = wmean_se(EL1A, wA, nch)
ev['A_ess'] = float(wA.sum() ** 2 / np.sum(wA * wA) / len(wA))
ev['A_dlog_rms'] = float(np.sqrt(np.average((rxA - np.average(rxA, weights=None)) ** 2)))
# set B ~ a_{k+1}^2
rxB, rcB = resid(theta1, dB, XB); EL1B, EH1B = C.trial_elocs(dB, rxB, rcB)
wB0 = np.exp(-2 * (rxB - rxB.min()))     # weights to go back to a_k^2
ev['B_L1'] = chain_mean_se(EL1B, nch)
ev['B_H_new_amp_old_sign'] = chain_mean_se(EH1B, nch)
ev['B_L0_rw'] = wmean_se(dB['eh'], wB0, nch)
ev['B_ess_back'] = float(wB0.sum() ** 2 / np.sum(wB0 * wB0) / len(wB0))
ev['B_FNpenalty_L1_minus_H1'] = chain_mean_se(EL1B - EH1B, nch)
# paired difference L1 - L0 on set B with jackknife over chains
def jk_diff(v1, v0, w0, nch):
    v1 = v1.reshape(-1, nch); v0 = v0.reshape(-1, nch); w0 = w0.reshape(-1, nch)
    full = v1.mean() - np.sum(w0 * v0) / np.sum(w0); reps = []
    for c in range(nch):
        m = np.ones(nch, bool); m[c] = False
        reps.append(v1[:, m].mean() - np.sum(w0[:, m] * v0[:, m]) / np.sum(w0[:, m]))
    reps = np.asarray(reps)
    return float(full), float(np.sqrt((nch - 1) / nch * np.sum((reps - reps.mean()) ** 2)))
ev['B_dL_paired_jk'] = jk_diff(EL1B, dB['eh'], wB0, nch)
ev['A_dL_paired_jk'] = (lambda r: (-r[0], r[1]))(jk_diff(dA['eh'], EL1A, wA, nch))
ev['sec'] = time.time() - t0; ev['nA'] = len(XA); ev['nB'] = len(XB)
for k_, v_ in list(ev.items()):
    if isinstance(v_, tuple): ev[k_] = list(v_)
res['eval'] = ev
C.log('EVAL', json.dumps(ev))
res['per_site'] = {k_: (v_[0] / 36 if isinstance(v_, list) else v_ / 36) for k_, v_ in ev.items()
                   if k_.startswith(('A_L', 'B_L', 'B_H', 'A_dL', 'B_dL'))}
st_amp = dict(chain_params=st['chain_params'], Ts=st['Ts'], amp_g=os.path.abspath(args.out + '_params.npy'))
json.dump(st_amp, open(args.out + '_state_amponly.json', 'w'), indent=1)
json.dump(res, open(args.out + '.json', 'w'), indent=1)

# ------------------------------------------------------------------ Stage C: Krylov sign step
if not args.skip_sign:
    t0 = time.time(); n0 = net.neval
    A1 = C.Amp(net, theta1, name='new')
    chT = C.Chains(net, args.nthr_chains, 1000 * args.seed + 21); chT.advance(theta1, args.burn)
    XT = np.concatenate([chT.advance(theta1, args.between) for _ in range(args.nthr_rounds)])
    parts = [C.threshold_edges(guide, A1, XT[i:i + args.chunk_thr]) for i in range(0, len(XT), args.chunk_thr)]
    E = C.merge_edges(parts); del parts
    base, cand, curve = C.energy_curve(E)
    ib = int(np.argmin(curve)); noflip = ib == len(cand) - 1
    Tn = 1e9 if noflip else float(cand[ib])
    thr = dict(T=Tn, noflip=bool(noflip), gain_train=float(curve[ib] - base), gain_train_site=float((curve[ib] - base) / 36),
               flip_frac_train=float(np.mean(E['rx_node'] > Tn)), ntrain=len(XT),
               r_quantiles=np.quantile(E['rx_node'], [0, .01, .5, .99, .999, 1]).tolist(), sec_train=time.time() - t0,
               evals_train=net.neval - n0)
    ch_ = np.arange(len(XT)) % args.nthr_chains
    for nm, m in (('even', ch_ % 2 == 0), ('odd', ch_ % 2 == 1)):
        b2, c2, cu2 = C.energy_curve(E, m); i2 = int(np.argmin(cu2))
        thr[f'T_half_{nm}'] = float(c2[i2]); thr[f'gain_half_{nm}'] = float(cu2[i2] - b2)
    C.log('THRESHOLD', json.dumps(thr))
    np.savez_compressed(args.out + '_thrcurve.npz', cand=cand, curve=curve, base=base)
    del E
    # held-out: set B (~a_{k+1}^2): <H> of a_{k+1} s_k (no flip) and a_{k+1} s_{k+1}
    t1 = time.time(); n0 = net.neval
    parts = [C.threshold_edges(guide, A1, XB[i:i + args.chunk_thr]) for i in range(0, len(XB), args.chunk_thr)]
    EB = C.merge_edges(parts); del parts
    e_old = C.eloc_at_T(EB, 1e300); e_new = C.eloc_at_T(EB, Tn)
    thr['check_noflip_vs_EH1B_maxdiff'] = float(np.max(np.abs(e_old - EH1B)))
    thr['B_H_new_amp_old_sign'] = chain_mean_se(e_old, nch)
    thr['B_H_new_guide'] = chain_mean_se(e_new, nch)
    thr['B_dH_sign_step'] = chain_mean_se(e_new - e_old, nch)
    thr['B_flip_frac'] = float(np.mean(EB['rx_node'] > Tn))
    bB, cB, cuB = C.energy_curve(EB); iB = int(np.argmin(cuB))
    thr['B_T_eval_optimal_biased'] = float(cB[iB]); thr['B_gain_eval_optimal_biased'] = float(cuB[iB] - bB)
    thr['sec_eval'] = time.time() - t1; thr['evals_eval'] = net.neval - n0
    for k_, v_ in list(thr.items()):
        if isinstance(v_, tuple): thr[k_] = list(v_)
    res['sign_step'] = thr
    res['per_site']['B_H_new_guide'] = thr['B_H_new_guide'][0] / 36
    res['per_site']['B_dH_sign_step'] = thr['B_dH_sign_step'][0] / 36
    C.log('SIGNSTEP', json.dumps(thr))
    st_new = dict(chain_params=st['chain_params'] + [os.path.abspath(args.out + '_params.npy')],
                  Ts=st['Ts'] + [Tn], amp_g=os.path.abspath(args.out + '_params.npy'))
    json.dump(st_new, open(args.out + '_state.json', 'w'), indent=1)
res['total_evals'] = net.neval; res['wall_sec'] = time.time() - T0
json.dump(res, open(args.out + '.json', 'w'), indent=1)
C.log('DONE', json.dumps(res['per_site']))
