#!/usr/bin/env python3
"""T3 amplitude stage (6x6, J2/J1 = 0.5): fixed-sign second-order VMC on the ViT, the 4x4 T2 recipe.

Guide sign s = stored composite sign net [+ one exact hop] (it2_core spec, FIXED during this stage); amplitude
b_theta = |ViT(theta)| fine-tuned directly from spec['amp_g'] (warm start).  Objective: fixed-sign VMC energy
E_H[b] = <b s|H|b s>/<b|b>.
Per step (RGN with a trust region verified on fresh samples, 4x4 vit_rgn_4x4.py 'rgntr' + adapt):
  * N train + N verification states from |b|^{2 beta} (tempered bond-exchange Metropolis, new chains each step);
    estimator weights w ~ b^{2-2beta} (self-normalised);
  * g = 2 E_w[(E_loc - E) (O - <O>)];  M = 2(Hbar - E S) (RGN) = E_w[2 (E_loc - E) dO dO^T] + E_w[sum_y c_xy (O_x - O_y)(..)^T],
    c_xy = -(s_x H_xy s_y) b_y/b_x (negative on sign-violating edges); A_+ = kept-edge part (PSD damping);
  * curvature on a random subset of --ncurv training states with ALL their H-neighbours (jvp/vjp, never stored);
  * subspace solve: Krylov basis {S^+ g, (S^+ M)^j S^+ g} (S^+ = minSR pseudo-inverse on the training samples),
    k = --krylov vectors; projected (M + lam A_+) c = -g; lam: Levenberg-Marquardt, accepted iff the paired decrease
    measured on the INDEPENDENT verification states >= 0.25 x predicted (lam/3 if > 0.75, else lam*4, <= 6 tries);
  * adapt: lam reset to lam0 at stage start and capped at 30 lam0 each step; N doubled (<= Nmax) after a fully failed step.
Certificate (fresh chains from b^2, beta = 1): paired <H>_G - E_ViT (ViT local energy reweighted, chain jackknife) at
  start (optional) and end.  Writes params_<out>.npy, spec_<out>.json (amp_g -> new), rgn_<out>.json.
Usage: python t3_rgn.py --spec G.json --out TAG [--steps 3 --N 8192 --beta 0.5 ...]
"""
import argparse, json, os, shutil, time
import numpy as np
import jax, jax.numpy as jnp
import scipy.linalg as sl
import ll6_core as C
import it2_core as I

ap = argparse.ArgumentParser()
ap.add_argument('--spec', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=3)
ap.add_argument('--N', type=int, default=8192)
ap.add_argument('--Nmax', type=int, default=16384)
ap.add_argument('--beta', type=float, default=0.5)
ap.add_argument('--lam0', type=float, default=0.03)
ap.add_argument('--krylov', type=int, default=8)
ap.add_argument('--ncurv', type=int, default=2048)
ap.add_argument('--eps-k', type=float, default=1.0, help='minSR shift for the Krylov basis, relative to mean eig of K')
ap.add_argument('--nch', type=int, default=1024)
ap.add_argument('--between', type=int, default=4)
ap.add_argument('--burn', type=int, default=100)
ap.add_argument('--cert-start', type=int, default=0)
ap.add_argument('--cert-end', type=int, default=32768)
ap.add_argument('--chunk-local', type=int, default=1024)
ap.add_argument('--jac-chunk', type=int, default=256)
ap.add_argument('--jvp-chunk', type=int, default=8192)
ap.add_argument('--seed', type=int, default=11)
ap.add_argument('--diag', type=int, default=0)
ap.add_argument('--mode', choices=['tr', 'ls'], default='ls')
ap.add_argument('--ls-rms', default='')
ap.add_argument('--ls-rms2', default='0.003,0.006,0.012,0.024')
ap.add_argument('--z-acc', type=float, default=2.0)
ap.add_argument('--wcap', type=float, default=0.999, help='winsorisation quantile for curvature weights (1 = off)')
ap.add_argument('--mdiag', type=float, default=1.0, help='weight of the 2(E_loc-E) dO dO term in M (1 = RGN)')
args = ap.parse_args()
clk = I.GpuClock(); res = dict(args=vars(args))
def save(): res['gpu_h'] = clk.hours(); json.dump(res, open(f'rgn_{args.out}.json', 'w'), indent=1)

net = C.Net(I.CKPT, dtype='float32', batch=16384, jac_chunk=args.jac_chunk)
spec, bd = I.load_spec(args.spec); res['spec_in'] = spec
for p in spec['nets']:
    q = I.resolve(p, bd)
    if not os.path.exists(os.path.basename(q)): shutil.copy(q, os.path.basename(q))
G = I.HGuide(spec, net, base_dir=bd)
theta = I.load_flat(net, I.resolve(spec['amp_g'], bd)); P = net.npar
I.log('backend', jax.default_backend(), 'P', P, 'spec', spec)

f_bits = net._f_bits
_jvp = jax.jit(lambda fl, v, S: jax.jvp(lambda f: f_bits(f, S), (fl,), (v,))[1])
_vjp = jax.jit(lambda fl, S, ct: jax.vjp(lambda f: f_bits(f, S), fl)[1](ct)[0])


def padded_chunks(S, c):
    S = np.asarray(S, np.uint64)
    for i in range(0, len(S), c):
        s = S[i:i + c]; m = len(s)
        if m < c: s = np.concatenate([s, np.repeat(s[:1], c - m)])
        yield i, m, jnp.asarray(s)


def jvp_states(fl, v, S):
    out = np.empty(len(S))
    for i, m, s in padded_chunks(S, args.jvp_chunk): out[i:i + m] = np.asarray(_jvp(fl, v, s))[:m]
    return out


def vjp_states(fl, S, ct):
    acc = jnp.zeros(P, net.DT)
    for i, m, s in padded_chunks(S, args.jvp_chunk):
        c = np.zeros(args.jvp_chunk, np.float32); c[:m] = ct[i:i + m]
        acc = acc + _vjp(fl, s, jnp.asarray(c, net.DT))
    return acc


def set_amp(th):
    G.amp_g = I.FnCache(lambda S, fl=th: net.logabs(fl, S), name='cur')


def draw(th, n, beta, seed):
    nr = max(1, int(np.ceil(n / args.nch)))
    X, acc = I.sample(net, th, beta, args.nch, nr, args.between, args.burn, seed)
    return X.reshape(-1), acc          # row-major (rounds, chains)


def certificate(th, n, seed, tag):
    """paired <H>_{b s} - E_ViT on x ~ b^2 (beta=1); ViT complex local energy reweighted by |psi_ViT|^2/b^2."""
    set_amp(th)
    X, acc = draw(th, n, 1.0, seed)
    d = G.local_chunked(X, args.chunk_local)
    zx = net.logpsi_c(net.flat0, X)
    NB, V = C.neighbors(X); own, col = np.nonzero(V)
    zy = net.logpsi_c(net.flat0, NB[own, col])
    ev = C.diag_vec(V).astype(complex); np.add.at(ev, own, 0.5 * C.JB[col] * np.exp(zy - zx[own])); ev = ev.real
    w = np.exp(2 * (zx.real - d['lax'])); w /= w.mean()
    nch = args.nch
    delta = [x / 36 for x in I.jk_ratio_diff(d['eh'], ev, w, nch)]
    out = dict(n=int(len(X)), accept=acc, H_site=I.chain_mean_se(d['eh'] / 36, nch), delta_vs_vit_site=delta,
               H_via_vit_ref=[I.E_VIT_SITE[0] + delta[0], float(np.hypot(I.E_VIT_SITE[1], delta[1]))],
               ess_vit_weights=float(w.sum() ** 2 / np.sum(w * w) / len(w)),
               per_sample_sd_site=float(np.std(d['eh'] - w * ev) / 36))
    I.log('CERT', tag, json.dumps(out)); return out


if args.cert_start:
    res['cert_start'] = certificate(theta, args.cert_start, 1000 * args.seed + 7, 'start'); clk.tick('cert_start'); save()

lam = args.lam0; Ncur = args.N; rows = []
for step in range(1, args.steps + 1):
    t0 = time.time(); lam = min(lam, 30 * args.lam0); set_amp(theta)
    X, acc_tr = draw(theta, Ncur, args.beta, 1000 * args.seed + 10 * step + 1)
    Xv, acc_va = draw(theta, Ncur, args.beta, 1000 * args.seed + 10 * step + 2)
    d = G.local_chunked(X, args.chunk_local); dv = G.local_chunked(Xv, args.chunk_local)
    clk.tick('local'); t1 = time.time()
    n = len(X)
    w = np.exp((2 - 2 * args.beta) * (d['lax'] - d['lax'].max())); w /= w.sum()
    eh = d['eh']; Eb = float(w @ eh)
    # Jacobian at the training samples, built in GPU chunks and kept on the HOST (fits 11 GB GPUs), centred
    O = np.empty((n, P), np.float32)
    for i, m, s in padded_chunks(X, args.jac_chunk):
        O[i:i + m] = np.asarray(net._grad_bits(theta, s))[:m]
    sqw = np.sqrt(w)
    Obar = (w.astype(np.float32) @ O).astype(np.float32)
    O -= Obar[None, :]
    e = 2.0 * sqw * (eh - Eb)
    g = (O.T @ (sqw * e).astype(np.float32)).astype(np.float64)        # 2 E_w[(E_loc - E)(O - Obar)]
    K = (O @ O.T).astype(np.float64) * np.outer(sqw, sqw); K = 0.5 * (K + K.T)
    epsK = args.eps_k * float(np.trace(K)) / n
    kev, kU = np.linalg.eigh(K)
    kinv2 = 1.0 / (np.maximum(kev, 0.0) + epsK) ** 2
    def Splus(z):             # minSR pseudo-inverse on the sample tangent space: O^T sqw (K+eps)^-2 sqw O z
        a_ = kU @ (kinv2 * (kU.T @ (sqw * (O @ z.astype(np.float32)))))
        return (O.T @ (sqw * a_).astype(np.float32)).astype(np.float64)
    clk.tick('jacobian')
    # curvature subset and its edges
    rgc = np.random.default_rng(1000 * args.seed + step)
    sub = np.sort(rgc.choice(n, min(args.ncurv, n), replace=False)) if args.ncurv > 0 else np.arange(n)
    insub = np.zeros(n, bool); insub[sub] = True
    em = insub[d['own']]
    e_own = d['own'][em]; e_child = d['child'][em]; e_rate = d['rate'][em]; e_al = d['allowed'][em]
    uc_states, uc_inv = np.unique(e_child, return_inverse=True)
    wsub = w[sub] / w[sub].sum(); Esub = float(wsub @ eh[sub])
    wfull_sub = np.zeros(n); wfull_sub[sub] = wsub
    cH = np.where(e_al, e_rate, -e_rate) * wfull_sub[e_own]    # w x c_xy for the RGN edge term
    cA = np.where(e_al, e_rate, 0.) * wfull_sub[e_own]
    adiag = args.mdiag * 2.0 * wfull_sub * (eh - Esub)         # diagonal (E_loc - E) term on the subset
    if args.wcap < 1:          # winsorise heavy-tailed curvature weights (curvature is only a preconditioner)
        def cap(a_):
            nz = np.abs(a_[a_ != 0]); q = np.quantile(nz, args.wcap) if len(nz) else 0.
            return np.clip(a_, -q, q)
        cH = cap(cH); cA = cap(cA); adiag = cap(adiag)
    Obar_sub = (wsub.astype(np.float32) @ O[sub]).astype(np.float64) + Obar

    def fvals(v):
        ux = (O @ v.astype(np.float32)).astype(np.float64) + float(Obar @ v)   # all training samples
        uy = jvp_states(theta, jnp.asarray(v, net.DT), uc_states)[uc_inv]    # per edge child
        return ux, uy

    def Mvec(v, ux, uy):
        uxc = ux - float(wsub @ ux[sub])
        du = ux[e_own] - uy
        cx = adiag * uxc
        cx_e = np.bincount(e_own, weights=cH * du, minlength=n)
        tot = cx + cx_e
        r = (O.T @ tot.astype(np.float32)).astype(np.float64) + Obar * float(tot.sum()) - Obar_sub * float(cx.sum())
        r = r - np.asarray(vjp_states(theta, uc_states, np.bincount(uc_inv, weights=cH * du, minlength=len(uc_states))), np.float64)
        return r

    V = []; FV = []
    v = Splus(g)
    for j in range(args.krylov):
        for q in V: v = v - (q @ v) * q
        nv = float(np.linalg.norm(v))
        if not np.isfinite(nv) or nv < 1e-12: break
        v = v / nv; V.append(v); ux, uy = fvals(v); FV.append((ux, uy))
        if j < args.krylov - 1: v = Splus(Mvec(v, ux, uy))
    k = len(V)
    gV = np.array([float(q @ g) for q in V])
    Vj = [jnp.asarray(q, net.DT) for q in V]
    UX = np.stack([f[0] for f in FV]); UY = np.stack([f[1] for f in FV])
    UXc = UX - (UX[:, sub] @ wsub)[:, None]
    DU = UX[:, e_own] - UY
    MV = (UXc * adiag[None, :]) @ UXc.T + (DU * cH[None, :]) @ DU.T
    AV = (DU * cA[None, :]) @ DU.T
    UXa = UX - (UX @ w)[:, None]
    SV = (UXa * w[None, :]) @ UXa.T
    if args.diag:                       # curvature stability: halves of the curvature subset, diagonal vs edge parts
        half = np.zeros(n, bool); half[sub[::2]] = True
        dg = {}
        for name, hm in (('h0', half), ('h1', insub & ~half)):
            emh = hm[e_own]; wh = np.where(hm, w, 0.); wh /= wh.sum(); Eh = float(wh @ eh)
            UXh = UX - (UX @ wh)[:, None]
            Dd = (UXh * (2 * wh * (eh - Eh))[None, :]) @ UXh.T
            cHh = np.where(emh, cH, 0.) * 2; cAh = np.where(emh, cA, 0.) * 2
            Ed = (DU * cHh[None, :]) @ DU.T; Ah = (DU * cAh[None, :]) @ DU.T
            ev = lambda A_: [float(x) for x in np.linalg.eigvalsh(0.5 * (A_ + A_.T))]
            dg[name] = dict(diag_term=ev(Dd), edge_term=ev(Ed), A_plus=ev(Ah), M=ev(Dd + Ed))
        dg['S'] = [float(x) for x in np.linalg.eigvalsh(SV)]
        dg['gV'] = gV.tolist(); dg['ess_w'] = float(1.0 / np.sum(w * w) / n)
        I.log('DIAG', json.dumps(dg)); res.setdefault('diag', []).append(dg)
    clk.tick('curvature'); t2 = time.time()
    # verification baseline (paired, fixed verification states, fixed sign)
    wv0 = np.exp((2 - 2 * args.beta) * (dv['lax'] - dv['lax'].max()))
    Ecur = float(wv0 @ dv['eh'] / wv0.sum())

    lw0 = (2 - 2 * args.beta) * dv['lax']; wv0n = np.exp(lw0 - lw0.max()); E0v = dv['eh']
    chain_v = np.arange(len(Xv)) % args.nch

    def measure(th):
        rx = net.logabs(th, Xv) - dv['lax']; rc = net.logabs(th, dv['child']) - dv['lac']
        _, EH = C.trial_elocs(dv, rx, rc)
        lw = 2 * (dv['lax'] + rx) - 2 * args.beta * dv['lax']; wv = np.exp(lw - lw.max())
        full = float(wv @ EH / wv.sum() - wv0n @ E0v / wv0n.sum())
        sw1 = np.bincount(chain_v, weights=wv, minlength=args.nch); se1 = np.bincount(chain_v, weights=wv * EH, minlength=args.nch)
        sw0 = np.bincount(chain_v, weights=wv0n, minlength=args.nch); se0 = np.bincount(chain_v, weights=wv0n * E0v, minlength=args.nch)
        reps = (se1.sum() - se1) / (sw1.sum() - sw1) - (se0.sum() - se0) / (sw0.sum() - sw0)
        se = float(np.sqrt((args.nch - 1) / args.nch * np.sum((reps - reps.mean()) ** 2)))
        return Ecur + full, float(np.sqrt(np.mean((rx - rx.mean()) ** 2))), se

    info = dict(accepted=False, tries=0)
    trA = max(float(np.trace(AV)), 1e-300); trS = max(float(np.trace(SV)), 1e-300); trM = abs(float(np.trace(MV)))
    Dm = 0.5 * (AV / trA + SV / trS) + 1e-10 * np.eye(k)          # PSD damping: kept-edge Laplacian + S (trace 1)
    Lc = np.linalg.cholesky(Dm); Li = np.linalg.inv(Lc)
    mu_min = float(np.linalg.eigvalsh(Li @ (0.5 * (MV + MV.T)) @ Li.T).min())
    lam_pd = max(0.0, -mu_min) / max(trM, 1e-300)                    # smallest lam with M + lam trM Dm PSD
    info['lam_pd'] = lam_pd
    if args.mode == 'tr':
        for tr in range(8):
            lam = max(lam, 1.5 * lam_pd + 1e-9)
            Mx = MV + lam * trM * Dm
            c = -np.linalg.solve(Mx, gV)
            pred = float(gV @ c + 0.5 * c @ MV @ c)
            dth = sum(float(ci) * q for ci, q in zip(c, Vj))
            cand = theta + dth
            Enew, drms, se = measure(cand); meas = Enew - Ecur
            info.update(tries=tr + 1, pred_site=pred / 36, meas_site=meas / 36, meas_se_site=se / 36, drms_va=drms, lam=lam)
            if pred < 0 and meas < 0 and meas / pred > 0.25:
                theta = cand; info['accepted'] = True
                if meas / pred > 0.75: lam /= 3
                break
            lam *= 4
    else:   # 'ls': candidates = RGN subspace solutions (several lam) + SR direction at several step sizes;
            # all measured on the independent verification states; accept the best if its decrease is > z_acc SE
        cands = []
        for f in (1.5, 6., 24., 96.):
            lm = max(f * lam_pd, 1e-6)
            c = -np.linalg.solve(MV + lm * trM * Dm, gV)
            cands.append(('rgn', lm, sum(float(ci) * q for ci, q in zip(c, Vj))))
        vsr = -Splus(g); usr = (O @ vsr.astype(np.float32)).astype(np.float64); rms_sr = float(np.sqrt(w @ (usr - w @ usr) ** 2))
        for tgt in [t for t in args.ls_rms.split(',') if t]:
            a_ = float(tgt) / max(rms_sr, 1e-300); cands.append(('sr', float(tgt), jnp.asarray(a_ * vsr, net.DT)))
        # better-generalising directions: plain gradient and strongly shifted SR (fewer effective directions)
        extra = [('gd', -g)]
        for sh in (1.0, 10.0):
            kin = 1.0 / (np.maximum(kev, 0.0) + sh * float(np.trace(K)) / n) ** 2
            a2 = kU @ (kin * (kU.T @ (sqw * (O @ g.astype(np.float32)))))
            extra.append((f'sr_eps{sh:g}', -(O.T @ (sqw * a2).astype(np.float32)).astype(np.float64)))
        for kind, vdir in extra:
            u_ = (O @ vdir.astype(np.float32)).astype(np.float64); r_ = float(np.sqrt(w @ (u_ - w @ u_) ** 2))
            for tgt in args.ls_rms2.split(','):
                cands.append((kind, float(tgt), jnp.asarray(float(tgt) / max(r_, 1e-300) * vdir, net.DT)))
        meas_all = []
        for kind, par, dth in cands:
            Enew, drms, se = measure(theta + dth); meas_all.append(dict(kind=kind, par=par, meas_site=(Enew - Ecur) / 36,
                                                                      se_site=se / 36, drms_va=drms))
        ok = [i for i, m in enumerate(meas_all) if m['meas_site'] < -args.z_acc * m['se_site']]
        j = min(ok, key=lambda i: meas_all[i]['meas_site']) if ok else int(np.argmin([m['meas_site'] for m in meas_all]))
        info.update(cands=meas_all, best=j, tries=len(cands))
        if ok:
            theta = theta + cands[j][2]; info['accepted'] = True
    if not info['accepted']: Ncur = min(2 * Ncur, args.Nmax)
    clk.tick('trust_region')
    row = dict(step=step, n=n, N_next=Ncur, k=k, E_tr_site=Eb / 36, E_va_site=Ecur / 36, gnorm=float(np.linalg.norm(g)),
               eigM=[float(x) for x in np.linalg.eigvalsh(MV)[:2]], acc_tr=acc_tr, n_curv_edges=int(len(e_own)),
               n_curv_states=int(len(uc_states)), t_local=t1 - t0, t_curv=t2 - t1, t_tr=time.time() - t2, **info)
    rows.append(row); res['steps'] = rows; I.log('STEP', json.dumps(row))
    np.save(f'params_{args.out}.npy', np.asarray(theta)); save()
    del O, K

if args.cert_end:
    res['cert_end'] = certificate(theta, args.cert_end, 1000 * args.seed + 9, 'end'); clk.tick('cert_end')
spec_new = dict(spec); spec_new['nets'] = [os.path.basename(I.resolve(p, bd)) for p in spec['nets']]
if spec.get('hop') and spec['hop']['amp'] != 'base':
    spec_new['hop'] = dict(spec['hop']); spec_new['hop']['amp'] = os.path.abspath(I.resolve(spec['hop']['amp'], bd))
spec_new['amp_g'] = os.path.abspath(f'params_{args.out}.npy'); spec_new['name'] = f'{spec.get("name", "G")}_rgn'
spec_new['amp_from'] = os.path.abspath(args.spec)
I.spec_dump(spec_new, f'spec_{args.out}.json')
res['vit_evals'] = int(net.neval); save(); I.log('DONE', json.dumps(res['gpu_h']))
