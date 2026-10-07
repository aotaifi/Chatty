#!/usr/bin/env python3
"""AMP6 (6x6, J2/J1 = 0.5): fixed-sign amplitude steps on the ViT at LARGE N with a ZERO-HOP sign.

Sign (fixed during the stage, one evaluation per state):
  --spec spec.json  : stored composite sign net(s), no hop (it2_core spec); or
  --spec vit        : the ViT's own binarised phase (control D).
Amplitude b_theta = |ViT(theta)|, warm start from --init (default: spec amp_g or the ViT).  Objective E_H[b] = <b s|H|b s>/<b|b>.
Per step (all matrix-free in parameter space; no Jacobian is stored, so N = 1e5-5e5 is possible):
  * N training + Nv verification states from |b|^{2 beta} (tempered bond-exchange Metropolis, fresh chains);
  * g = 2 E_w[(E_loc - E)(O - <O>)] by one vjp; SR directions (S + shift) d = g by CG with S v = J^T W J v (jvp+vjp),
    shift = eps * tr(S)/N for eps in --sr-eps;
  * candidates: -g and -d at several train RMS(delta log b); each is measured on the INDEPENDENT verification states
    (paired, self-normalised); accept the most negative one with decrease > --z-acc chain-jackknife SE;
  * generalisation diagnostic: RMS(delta log b) on training vs verification states for every candidate.
Certificates: paired <H>_G - E_ViT on fresh x ~ b^2 at start/end (ViT complex local energy reweighted).
Usage: python amp6_large.py --spec S.json|vit --out TAG [--init params.npy] [--N 100000 --steps 3 ...]
"""
import argparse, json, os, shutil, time
import numpy as np
import jax, jax.numpy as jnp
import scipy.sparse.linalg as sla
import ll6_core as C
import it2_core as I

ap = argparse.ArgumentParser()
ap.add_argument('--spec', required=True)
ap.add_argument('--init', default=None)
ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=3)
ap.add_argument('--N', type=int, default=100000)
ap.add_argument('--Nv', type=int, default=None)
ap.add_argument('--beta', type=float, default=0.5)
ap.add_argument('--nch', type=int, default=4096)
ap.add_argument('--between', type=int, default=4)
ap.add_argument('--burn', type=int, default=100)
ap.add_argument('--sr-eps', default='0.01,1')
ap.add_argument('--cg-iters', type=int, default=40)
ap.add_argument('--ls-rms', default='0.002,0.005,0.012')
ap.add_argument('--z-acc', type=float, default=2.0)
ap.add_argument('--cert-start', type=int, default=0)
ap.add_argument('--cert-end', type=int, default=0)
ap.add_argument('--chunk-local', type=int, default=4096)
ap.add_argument('--vchunk', type=int, default=8192)
ap.add_argument('--seed', type=int, default=51)
args = ap.parse_args()
Nv = args.Nv or args.N
clk = I.GpuClock(); res = dict(args=vars(args))
def save(): res['gpu_h'] = clk.hours(); json.dump(res, open(f'amp6_{args.out}.json', 'w'), indent=1)

net = C.Net(I.CKPT, dtype='float32', batch=16384)
P = net.npar
pool = np.load('energy_krylov_vs_vit_6x6_indep.npz')['states'].astype(np.uint64).reshape(-1)
if args.spec == 'vit':
    phi = I.vit_phase(net, pool)
    sgn = I.FnCache(lambda S: np.where(np.cos(net.logpsi_c(net.flat0, S).imag - phi) >= 0, 1., -1.), name='vit_s')
    init = args.init or 'base'; spec = dict(name='vit_sign', phi=phi)
else:
    spec, bd = I.load_spec(args.spec); assert not spec.get('hop'), 'zero-hop sign required'
    for p in spec['nets']:
        q = I.resolve(p, bd)
        if not os.path.exists(os.path.basename(q)): shutil.copy(q, os.path.basename(q))
    st = I.StoredSign([I.resolve(p, bd) for p in spec['nets']]); sgn = st
    init = args.init or (I.resolve(spec['amp_g'], bd) if spec['amp_g'] != 'base' else 'base')
res['spec'] = spec; res['init'] = init
theta = I.load_flat(net, init)
I.log('backend', jax.default_backend(), 'P', P, 'spec', spec, 'init', init)

f_bits = net._f_bits
_jvp = jax.jit(lambda fl, v, S: jax.jvp(lambda f: f_bits(f, S), (fl,), (v,))[1])
_vjp = jax.jit(lambda fl, S, ct: jax.vjp(lambda f: f_bits(f, S), fl)[1](ct)[0])


def chunks(S, c):
    S = np.asarray(S, np.uint64)
    for i in range(0, len(S), c):
        s = S[i:i + c]; m = len(s)
        if m < c: s = np.concatenate([s, np.repeat(s[:1], c - m)])
        yield i, m, jnp.asarray(s)


def jvp_states(fl, v, S):
    out = np.empty(len(S)); vj = jnp.asarray(v, net.DT)
    for i, m, s in chunks(S, args.vchunk): out[i:i + m] = np.asarray(_jvp(fl, vj, s))[:m]
    return out


def vjp_states(fl, S, ct):
    acc = jnp.zeros(P, net.DT)
    for i, m, s in chunks(S, args.vchunk):
        c = np.zeros(args.vchunk, np.float32); c[:m] = ct[i:i + m]
        acc = acc + _vjp(fl, s, jnp.asarray(c, net.DT))
    return np.asarray(acc, np.float64)


def guide(th):
    la = I.FnCache(lambda S, fl=th: net.logabs(fl, S), name='cur')
    return I.FuncGuide(la, sgn)


def draw(th, n, beta, seed):
    X, acc = I.sample(net, th, beta, args.nch, max(1, int(np.ceil(n / args.nch))), args.between, args.burn, seed)
    return X.reshape(-1), acc


def local(G, X):
    parts = []
    for i in range(0, len(X), args.chunk_local):
        d = G.local(X[i:i + args.chunk_local]); d['own'] = d['own'] + i; parts.append(d)
    out = {k: np.concatenate([p[k] for p in parts]) for k in C.LOCAL_KEYS}; out['n'] = len(X)
    return out


def certificate(th, n, seed, tag):
    G = guide(th); X, acc = draw(th, n, 1.0, seed); d = local(G, X)
    zx = net.logpsi_c(net.flat0, X); NB, V = C.neighbors(X); own, col = np.nonzero(V)
    zy = net.logpsi_c(net.flat0, NB[own, col])
    ev = C.diag_vec(V).astype(complex); np.add.at(ev, own, 0.5 * C.JB[col] * np.exp(zy - zx[own])); ev = ev.real
    w = np.exp(2 * (zx.real - d['lax'])); w /= w.mean()
    delta = [x / 36 for x in I.jk_ratio_diff(d['eh'], ev, w, args.nch)]
    out = dict(n=int(len(X)), accept=acc, H_site=I.chain_mean_se(d['eh'] / 36, args.nch), delta_vs_vit_site=delta,
               H_via_vit_ref=[I.E_VIT_SITE[0] + delta[0], float(np.hypot(I.E_VIT_SITE[1], delta[1]))],
               ess_vit_weights=float(w.sum() ** 2 / np.sum(w * w) / len(w)), per_sample_sd_site=float(np.std(d['eh'] - w * ev) / 36))
    I.log('CERT', tag, json.dumps(out)); return out


if args.cert_start:
    res['cert_start'] = certificate(theta, args.cert_start, 1000 * args.seed + 7, 'start'); clk.tick('cert_start'); save()

rows = []
for step in range(1, args.steps + 1):
    t0 = time.time(); G = guide(theta)
    X, acc_tr = draw(theta, args.N, args.beta, 1000 * args.seed + 10 * step + 1)
    Xv, acc_va = draw(theta, Nv, args.beta, 1000 * args.seed + 10 * step + 2)
    clk.tick('sample'); t1 = time.time()
    d = local(G, X); dv = local(G, Xv)
    clk.tick('local'); t2 = time.time()
    n = len(X); w = np.exp((2 - 2 * args.beta) * (d['lax'] - d['lax'].max())); w /= w.sum()
    eh = d['eh']; Eb = float(w @ eh)
    g = vjp_states(theta, X, 2 * w * (eh - Eb))
    def Smv(v):
        u = jvp_states(theta, v, X); uc = u - w @ u
        return vjp_states(theta, X, w * uc)
    rg = np.random.default_rng(step); z = rg.choice([-1.0, 1.0], size=P)
    trS = float(z @ Smv(z))                          # one-probe Hutchinson estimate of tr(S)
    dirs = [('gd', -g)]
    for eps in [float(x) for x in args.sr_eps.split(',') if x]:
        sh = eps * trS / n
        A = sla.LinearOperator((P, P), matvec=lambda v, sh=sh: Smv(v) + sh * v, dtype=np.float64)
        dsr, info = sla.cg(A, g, rtol=1e-4, atol=0.0, maxiter=args.cg_iters)
        dirs.append((f'sr_eps{eps:g}', -dsr))
    clk.tick('directions'); t3 = time.time()
    # verification baseline
    lw0 = (2 - 2 * args.beta) * dv['lax']; wv0 = np.exp(lw0 - lw0.max()); E0v = dv['eh']
    chain_v = np.arange(len(Xv)) % args.nch
    Ecur = float(wv0 @ E0v / wv0.sum())

    def measure(th):
        rx = net.logabs(th, Xv) - dv['lax']; rc = net.logabs(th, dv['child']) - dv['lac']
        _, EH = C.trial_elocs(dv, rx, rc)
        lw = 2 * (dv['lax'] + rx) - 2 * args.beta * dv['lax']; wv = np.exp(lw - lw.max())
        full = float(wv @ EH / wv.sum() - Ecur)
        sw1 = np.bincount(chain_v, weights=wv, minlength=args.nch); se1 = np.bincount(chain_v, weights=wv * EH, minlength=args.nch)
        sw0 = np.bincount(chain_v, weights=wv0, minlength=args.nch); se0 = np.bincount(chain_v, weights=wv0 * E0v, minlength=args.nch)
        reps = (se1.sum() - se1) / (sw1.sum() - sw1) - (se0.sum() - se0) / (sw0.sum() - sw0)
        se = float(np.sqrt((args.nch - 1) / args.nch * np.sum((reps - reps.mean()) ** 2)))
        return full, se, float(np.std(rx))

    cands = []
    for kind, D in dirs:
        u = jvp_states(theta, D, X); r_w = float(np.sqrt(w @ (u - w @ u) ** 2)); r_u = float(np.std(u))
        for tgt in [float(x) for x in args.ls_rms.split(',')]:
            a = tgt / max(r_w, 1e-300)
            m, se, drms_va = measure(theta + jnp.asarray(a * D, net.DT))
            cands.append(dict(kind=kind, rms_target=tgt, alpha=a, meas_site=m / 36, se_site=se / 36,
                              drms_tr=a * r_u, drms_va=drms_va, gen_ratio=drms_va / max(a * r_u, 1e-300)))
            I.log('CAND', json.dumps(cands[-1]))
    ok = [i for i, c in enumerate(cands) if c['meas_site'] < -args.z_acc * c['se_site']]
    j = min(ok, key=lambda i: cands[i]['meas_site']) if ok else int(np.argmin([c['meas_site'] for c in cands]))
    accepted = bool(ok)
    if accepted:
        c = cands[j]; D = dict(dirs)[c['kind']]; theta = theta + jnp.asarray(c['alpha'] * D, net.DT)
    clk.tick('verify')
    row = dict(step=step, n=n, nv=int(len(Xv)), E_tr_site=Eb / 36, E_va_site=Ecur / 36, gnorm=float(np.linalg.norm(g)), trS=trS,
               acc_tr=acc_tr, accepted=accepted, best=j, cands=cands, t_sample=t1 - t0, t_local=t2 - t1, t_dirs=t3 - t2,
               t_verify=time.time() - t3, ess_w=float(1 / np.sum(w * w) / n))
    rows.append(row); res['steps'] = rows; I.log('STEP', json.dumps({k: v for k, v in row.items() if k != 'cands'}))
    np.save(f'params_{args.out}.npy', np.asarray(theta)); save()

if args.cert_end:
    res['cert_end'] = certificate(theta, args.cert_end, 1000 * args.seed + 9, 'end'); clk.tick('cert_end')
res['vit_evals'] = int(net.neval); save(); I.log('DONE', json.dumps(res['gpu_h']))
