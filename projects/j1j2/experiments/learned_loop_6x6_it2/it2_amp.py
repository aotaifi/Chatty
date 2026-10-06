"""Amplitude step of the learned loop with a stored guide sign [+ one exact hop] (it2_core spec).

Learn a_new = |ViT(theta)| = a_g exp(r_theta) (warm start theta = theta_g) by SR on the FROZEN FN energy
  L(theta) = <a_theta s_g| H_FN[a_g, s_g] |a_theta s_g> / <a_theta|a_theta>     (as ll6_iter Stage A).
Differences to ll6_iter (iteration 1):
  * signs are stored (+ at most one hop) -> local data need level 1 (+1), never the recursive level 3, so the
    step can use more FRESH samples (default 16384 train + 16384 validation per SR step, new chain states every step);
  * acceptance gate on the independent validation chains (paired, reweighted by exp(2 delta r) on the SAME
    validation states; logged with its chain-block SE); on rejection the trust radius is halved and the run
    continues; stop after --max-reject consecutive rejections or --sr-steps;
  * the solve (S + shift) alpha = 2 eps is done on the GPU in float64; every accepted theta is saved.
Final, independent check (fresh chains): set A ~ a_g^2, set B ~ a_new^2, paired dL both ways (chain jackknife),
<H>_{a_new s_g}.  Writes params_<out>.npy, spec_<out>.json (= input spec with amp_g -> new), amp_<out>.json.
Usage: python it2_amp.py --spec G.json --out TAG [--ntr 16384 --nva 16384 --sr-steps 30 ...]
"""
import argparse, json, time, os, shutil
import numpy as np
import jax, jax.numpy as jnp
import ll6_core as C
import it2_core as I

ap = argparse.ArgumentParser()
ap.add_argument('--spec', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--ntr', type=int, default=16384)
ap.add_argument('--nva', type=int, default=16384)
ap.add_argument('--sr-steps', type=int, default=30)
ap.add_argument('--max-reject', type=int, default=3)
ap.add_argument('--shift', type=float, default=10.0)
ap.add_argument('--trust', type=float, default=0.02)
ap.add_argument('--trust-min', type=float, default=0.0025)
ap.add_argument('--between', type=int, default=4)
ap.add_argument('--burn', type=int, default=100)
ap.add_argument('--neval-chains', type=int, default=2048)
ap.add_argument('--neval-rounds', type=int, default=16)
ap.add_argument('--chunk-local', type=int, default=1024)
ap.add_argument('--jac-chunk', type=int, default=256)
ap.add_argument('--seed', type=int, default=1)
args = ap.parse_args()
clk = I.GpuClock(); res = dict(args=vars(args))
def save(): res['gpu_h'] = clk.hours(); json.dump(res, open(f'amp_{args.out}.json', 'w'), indent=1)

net = C.Net(I.CKPT, dtype='float32', batch=16384, jac_chunk=args.jac_chunk)
spec, bd = I.load_spec(args.spec); res['spec_in'] = spec
G = I.HGuide(spec, net, base_dir=bd)
for p in spec['nets']:
    q = I.resolve(p, bd)
    if not os.path.exists(os.path.basename(q)): shutil.copy(q, os.path.basename(q))
theta0 = I.load_flat(net, I.resolve(spec['amp_g'], bd)); theta = theta0
I.log('backend', jax.default_backend(), 'npar', net.npar, 'spec', spec)


def resid(th, d, X):
    return net.logabs(th, X) - d['lax'], net.logabs(th, d['child']) - d['lac']


def wdiff_se(EL1, w, EL0, nch):
    """paired reweighted difference sum(w EL1)/sum(w) - mean(EL0), chain-block SE (delta method)."""
    m1 = float(np.sum(w * EL1) / np.sum(w)); m0 = float(EL0.mean())
    z = (w / w.mean()) * (EL1 - m1) - (EL0 - m0)
    cm = z.reshape(-1, nch).mean(0)
    return m1 - m0, float(cm.std(ddof=1) / np.sqrt(nch))


ch_tr = C.Chains(net, args.ntr, 1000 * args.seed + 1)
ch_va = C.Chains(net, args.nva, 1000 * args.seed + 2)
ch_tr.advance(theta, args.burn); ch_va.advance(theta, args.burn)
clk.tick('burn')
rows = []; trust = args.trust; stop = 'max_steps'; nacc = 0; nrej = 0; cum = 0.
for step in range(1, args.sr_steps + 1):
    t0 = time.time(); n0 = net.neval
    Xtr = ch_tr.advance(theta, args.between); Xva = ch_va.advance(theta, args.between)
    dtr = G.local_chunked(Xtr, args.chunk_local); dva = G.local_chunked(Xva, args.chunk_local)
    t1 = time.time()
    rx, rc = resid(theta, dtr, Xtr); EL, EH = C.trial_elocs(dtr, rx, rc); Ltr = float(EL.mean())
    rxv, rcv = resid(theta, dva, Xva); ELv, _ = C.trial_elocs(dva, rxv, rcv); Lva = float(ELv.mean())
    n = len(Xtr)
    O = net.jac(theta, Xtr)
    mu = O.mean(0, keepdims=True)
    Ob = (O - mu) * np.float32(1.0 / np.sqrt(n)); del O, mu
    eps = jnp.asarray((EL - Ltr) / np.sqrt(n), Ob.dtype)
    Km = (Ob @ Ob.T).astype(jnp.float64)
    alpha = jnp.linalg.solve(Km + args.shift * jnp.eye(n, dtype=jnp.float64), 2.0 * eps.astype(jnp.float64))
    d = Ob.T @ alpha.astype(Ob.dtype)
    rms = float(jnp.linalg.norm(Ob @ d)); gnorm = float(jnp.linalg.norm(2.0 * (Ob.T @ eps)))
    eigs_top = [float(x) for x in jnp.linalg.eigvalsh(Km)[-3:]] if step == 1 else None
    del Ob, Km
    eta = trust / max(rms, 1e-300)
    t2 = time.time()
    acc = None; cands = []
    for fac in (1.0, 0.5):
        th = theta - eta * fac * d
        rx1, rc1 = resid(th, dva, Xva); EL1, _ = C.trial_elocs(dva, rx1, rc1)
        lw = 2.0 * (rx1 - rxv); w = np.exp(lw - lw.max())
        dL, se = wdiff_se(EL1, w, ELv, args.nva)
        ess = float(w.sum() ** 2 / np.sum(w * w) / len(w))
        dl = rx1 - rxv; drms = float(np.sqrt(np.mean((dl - dl.mean()) ** 2)))
        cands.append(dict(fac=fac, dLva=dL, dLva_se=se, z=dL / max(se, 1e-300), ess=ess, drms_va=drms))
        if dL < 0 and ess >= 0.5:
            acc = (th, fac); break
    row = dict(step=step, L_tr=Ltr, L_va=Lva, EH_tr=float(EH.mean()), gnorm=gnorm, rms_per_eta=rms, eta=eta,
               trust=trust, cands=cands, t_local=t1 - t0, t_sr=t2 - t1, t_ls=time.time() - t2,
               evals=net.neval - n0, frac_allowed=float(dtr['allowed'].mean()), Km_top_eigs=eigs_top)
    if acc is None:
        row['accepted'] = False; rows.append(row); nrej += 1; trust *= 0.5
        I.log('SR', json.dumps(row))
        if nrej >= args.max_reject or trust < args.trust_min:
            stop = 'rejects'; break
        continue
    theta, fac = acc; nacc += 1; nrej = 0; cum += cands[-1]['dLva']
    if fac < 1.0: trust *= fac
    ch_tr.reset_la(theta); ch_va.reset_la(theta)
    np.save(f'params_{args.out}_s{step}.npy', np.asarray(theta))
    row.update(accepted=True, factor=fac, cum_dLva_site=cum / 36); rows.append(row)
    I.log('SR', json.dumps(row))
    res['sr'] = dict(rows=rows, n_accepted=nacc); save()
res['sr'] = dict(rows=rows, n_accepted=nacc, stop=stop, cum_dLva_site=cum / 36,
                 accept_rate_tr=ch_tr.acc / max(ch_tr.props, 1))
np.save(f'params_{args.out}.npy', np.asarray(theta))
theta1 = theta
clk.tick('sr'); save()

# ---------------- independent evaluation
nch = args.neval_chains
def draw(th, seed):
    ch = C.Chains(net, nch, seed); ch.advance(th, args.burn)
    return np.concatenate([ch.advance(th, args.between) for _ in range(args.neval_rounds)])
XA = draw(theta0, 1000 * args.seed + 11); XB = draw(theta1, 1000 * args.seed + 12)
dA = G.local_chunked(XA, args.chunk_local); dB = G.local_chunked(XB, args.chunk_local)
ev = {}
rxA, rcA = resid(theta1, dA, XA); EL1A, EH1A = C.trial_elocs(dA, rxA, rcA)
wA = np.exp(2 * (rxA - rxA.max()))
ev['A_L0_site'] = I.chain_mean_se(dA['eh'] / 36, nch)
ev['A_ess'] = float(wA.sum() ** 2 / np.sum(wA * wA) / len(wA))
ev['A_dlog_rms'] = float(np.std(rxA))
rxB, rcB = resid(theta1, dB, XB); EL1B, EH1B = C.trial_elocs(dB, rxB, rcB)
wB0 = np.exp(-2 * (rxB - rxB.min()))
ev['B_L1_site'] = I.chain_mean_se(EL1B / 36, nch)
ev['B_H_newamp_site'] = I.chain_mean_se(EH1B / 36, nch)
ev['B_ess_back'] = float(wB0.sum() ** 2 / np.sum(wB0 * wB0) / len(wB0))
ev['B_dL_paired_site'] = [x / 36 for x in I.jk_ratio_diff(EL1B, dB['eh'], wB0, nch)]
ev['A_dL_paired_site'] = (lambda r: [-r[0] / 36, r[1] / 36])(I.jk_ratio_diff(dA['eh'], EL1A, wA, nch))
ev['B_FNpenalty_L1_minus_H1_site'] = I.chain_mean_se((EL1B - EH1B) / 36, nch)
ev['nA'] = len(XA); ev['nB'] = len(XB)
res['eval'] = ev; I.log('EVAL', json.dumps(ev))
spec_new = dict(spec); spec_new['nets'] = [os.path.basename(I.resolve(p, bd)) for p in spec['nets']]
spec_new['amp_g'] = os.path.abspath(f'params_{args.out}.npy'); spec_new['name'] = f'{spec.get("name", "G")}_amp'
spec_new['amp_from'] = os.path.abspath(args.spec)
I.spec_dump(spec_new, f'spec_{args.out}.json')
clk.tick('eval'); res['vit_evals'] = int(net.neval); save()
I.log('DONE', json.dumps(res['gpu_h']))
