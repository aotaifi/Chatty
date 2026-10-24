"""VMC estimator of the p = 0 and p = 1 (one signed Lanczos hop, Sorella cond-mat/0009149 style) energies and variances of the
real guide v = Re[w psi_P] built from a complex RBM+PP psi_P (any L, any projection; 8x8 'translations').

  python vmc_moments.py OUT --params P.npz [--L 8 --proj translations --chains 1024 --sweeps 40 --seeds 1,2 --chunk 256]

Definitions (H real, w a unit complex number): v(x) = Re[w psi_P(x)], (Hv)(x) = Re[w psi_P(x) E_L(x)], E_L = (H psi_P)/psi_P.
Sampling x ~ |psi_P|^2 (Metropolis exchange moves, float32 acceptance as in training); with u(x) = w psi_P(x)/|psi_P(x)|:
  v/|psi|         = Re u                                   =: vt
  (Hv)/|psi|(x)   = Re[u E_L(x)]                            =: h1
  (H^2 v)/|psi|(x)= diag(x) h1 + sum_{b antiparallel} (J_b/2) Re[u psi_P(y_b)/psi_P(x) E_L(y_b)]   =: h2
(E_L at x and at every antiparallel neighbour y_b: 1 + n_anti fast-update evaluations per sample).
Moments of H' = H - E_ref:  <v|v> = <vt^2>, <v|H'|v> = <vt h1'>, <v|H'^2|v> = <h1'^2> (= <vt h2'>, both accumulated: a check),
<v|H'^3|v> = <h1' h2'>, <v|H'^4|v> = <h2'^2>   (sums over samples, ratio to <vt^2>); then
  E = E_ref + m1, mu_k central moments, var = mu2, p = 1 Ritz in span{v, (H-E)v}:  alpha = (mu3 - sqrt(mu3^2 + 4 mu2^3)) / (2 mu2^2),
  E_1 = E + alpha mu2 (= (E_1 - E)/var identity), p = 1 state psi_1 = v + alpha (H - E) v:
  n = 1 + alpha^2 mu2, E_1 = E + (2 alpha mu2 + alpha^2 mu3)/n, var_1 = (mu2 + 2 alpha mu3 + alpha^2 mu4)/n - ((E_1 - E))^2.
Phase w: lowest eigenvector of the 2x2 problem in span{Re psi, Im psi} (moments <e^{2 i arg psi}>, <e^{2 i arg psi} E_L>, <Re E_L>),
v = c0 Re psi + c1 Im psi = Re[(c0 - i c1) psi].
Errors: jackknife over groups of chains (chains are independent starts) of the whole nonlinear function.
Writes OUT/vmc_s<seed>.json, OUT/E0_s<seed>.npy and OUT/E1_s<seed>.npy (per-sweep estimates over all chains, time order; input of
tools/quotable.py series), OUT/stage_A.json (w, E_ref), OUT/vmc_sums_s<seed>.npz.
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'rbmpp_6x6'))
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from rbmpp_lib import Lattice, make_fns, make_sampler, random_configs, to32


def lanczos_from_moments(m, E_ref):
    """m = (m1, m2, m3, m4) raw moments of H' = H - E_ref in the normalised v. Returns dict of E, var, alpha, E1, var1."""
    m1, m2, m3, m4 = m
    mu2 = m2 - m1 ** 2
    mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    alpha = (mu3 - np.sqrt(mu3 ** 2 + 4 * mu2 ** 3)) / (2 * mu2 ** 2)
    n = 1 + alpha ** 2 * mu2
    dE1 = (2 * alpha * mu2 + alpha ** 2 * mu3) / n
    var1 = (mu2 + 2 * alpha * mu3 + alpha ** 2 * mu4) / n - dE1 ** 2
    return dict(E=E_ref + m1, var=mu2, mu3=mu3, mu4=mu4, alpha=alpha, E1=E_ref + m1 + dE1, var1=var1,
                E1_ritz=E_ref + m1 + alpha * mu2)


def from_sums(s, E_ref):
    """s = (s0, s1, s2, s3, s4) sums (or means) of vt^2, vt h1', h1'^2, h1' h2', h2'^2."""
    s0 = s[0]
    return lanczos_from_moments((s[1] / s0, s[2] / s0, s[3] / s0, s[4] / s0), E_ref)


def phase_from_moments(b1, BH, Ec):
    """w = c0 - i c1 (|w| = 1) from <e^{2i phi}> = b1, <e^{2i phi} E_L> = BH, <Re E_L> = Ec (all with |psi|^2 weights)."""
    B = np.array([[0.5 * (1 + b1.real), 0.5 * b1.imag], [0.5 * b1.imag, 0.5 * (1 - b1.real)]])
    A = np.array([[0.5 * (Ec + BH.real), 0.5 * BH.imag], [0.5 * BH.imag, 0.5 * (Ec - BH.real)]])
    ev, U = np.linalg.eigh(B)
    keep = ev > 1e-10 * ev.max()
    T = U[:, keep] / np.sqrt(ev[keep])
    w_, V = np.linalg.eigh(T.T @ A @ T)
    c = T @ V[:, 0]
    w = c[0] - 1j * c[1]
    return w / abs(w), float(w_[0]), float(B[1, 1] / (B[0, 0] + B[1, 1]))


class Sampler:
    def __init__(self, params, lat, use_rbm, chunk=256):
        self.lat = lat; self.N = lat.N; self.chunk = chunk
        self.BI = np.asarray(lat.BI); self.BJ = np.asarray(lat.BJ); self.JB = np.asarray(lat.JB, float)
        self.p = {k: jnp.asarray(params[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}
        f64 = make_fns(lat, jnp.float64, use_rbm); f32 = make_fns(lat, jnp.float32, use_rbm)
        self.sweep = make_sampler(lat, f32); self.p32 = to32(self.p)
        self.ff = jax.jit(jax.vmap(f64['eloc_full'], in_axes=(None, 0)))
        self.fe = jax.jit(jax.vmap(f64['eloc'], in_axes=(None, 0)))
        self.n_eval = 0; self.t_eval = 0.0

    def start(self, C, seed, therm):
        self.key = jax.random.PRNGKey(7000 + seed)
        S = random_configs(self.N, C, 9500 + seed)
        for _ in range(therm):
            self.key, k = jax.random.split(self.key); S, acc = self.sweep(self.p32, S, k, self.N)
        self.S = S; self.acc = float(acc) if therm else float('nan')

    def step(self):
        self.key, k = jax.random.split(self.key)
        self.S, acc = self.sweep(self.p32, self.S, k, self.N); self.acc = float(acc)
        return np.asarray(self.S).astype(np.float64)

    def _chunk(self, fn, Z, nout):
        K = Z.shape[0]; c = self.chunk; outs = None
        for i in range(0, K, c):
            n = min(c, K - i); blk = Z[i:i + n]
            if n < c: blk = np.concatenate([blk, np.repeat(blk[:1], c - n, 0)])
            o = [np.asarray(a)[:n] for a in fn(self.p, jnp.asarray(blk))]
            if outs is None: outs = [np.empty((K,) + a.shape[1:], a.dtype) for a in o]
            for k_ in range(nout): outs[k_][i:i + n] = o[k_]
        return outs

    def at_x(self, X):
        t0 = time.time(); r = self._chunk(self.ff, X, 3); self.n_eval += X.shape[0]; self.t_eval += time.time() - t0; return r

    def at_y(self, Y):
        t0 = time.time(); r = self._chunk(self.fe, Y, 2); self.n_eval += Y.shape[0]; self.t_eval += time.time() - t0; return r

    def diag(self, X):
        return 0.25 * np.sum(self.JB[None] * X[:, self.BI] * X[:, self.BJ], 1)

    def neighbours(self, X):
        anti = X[:, self.BI] != X[:, self.BJ]
        wi, bi = np.nonzero(anti)
        Y = X[wi].copy(); r = np.arange(len(wi))
        Y[r, self.BI[bi]] = X[wi, self.BJ[bi]]; Y[r, self.BJ[bi]] = X[wi, self.BI[bi]]
        return wi, bi, Y


def stage_phase(sm, C, sweeps, seed, therm, log):
    sm.start(C, seed, therm)
    b1 = 0j; BH = 0j; Ec = 0.0; n = 0
    for it in range(sweeps):
        X = sm.step(); E, l, rP = sm.at_x(X)
        e2 = np.exp(2j * l.imag)
        b1 += e2.sum(); BH += (e2 * E).sum(); Ec += E.real.sum(); n += len(E)
    b1 /= n; BH /= n; Ec /= n
    w, lam, imw = phase_from_moments(b1, BH, Ec)
    log(f'stage A (phase): <e^2i phi> {b1:.6f}, <Re E_L>/N {Ec / sm.N:.6f}, w = {w:.6f} (|arg| {np.angle(w):.4f}), '
        f'lowest 2x2 eigenvalue/N {lam / sm.N:.6f}, weight of Im psi in the basis {imw:.3e}')
    return w, Ec, dict(b1=[b1.real, b1.imag], BH=[BH.real, BH.imag], Ec=Ec, w=[w.real, w.imag], lam=lam, im_weight=imw)


def config_quantities(sm, X, w):
    """per configuration x (rows of X, +-1 float64): vt = v/|psi|, h1 = (Hv)/|psi|, h2 = (H^2 v)/|psi| for v = Re[w psi_P];
    also the max relative deviation between psi_P(y)/psi_P(x) from the separate evaluation at y and from the fast update at x."""
    E, l, rP = sm.at_x(X)
    wi, bi, Y = sm.neighbours(X)
    Ey, ly = sm.at_y(Y)
    u = w * np.exp(1j * l.imag)
    vt = u.real; h1 = (u * E).real
    ratio = np.exp(ly - l[wi])
    dev = float(np.max(np.abs(ratio - rP[wi, bi]) / np.abs(rP[wi, bi])))
    Hvy = (u[wi] * ratio * Ey).real
    h2 = sm.diag(X) * h1 + np.bincount(wi, weights=0.5 * sm.JB[bi] * Hvy, minlength=X.shape[0])
    return vt, h1, h2, dev


def stage_moments(sm, w, E_ref, C, sweeps, seed, therm, G, log, tag='', maxsec=1e18):
    sm.start(C, seed, therm)
    grp = np.arange(C) % G
    sums = np.zeros((sweeps, G, 6))            # s0, s1, s2, s3, s4, s2b(= <vt h2'>)
    dev = 0.0; t0 = time.time(); sm.n_eval = 0; sm.t_eval = 0.0
    done = sweeps
    for it in range(sweeps):
        if time.time() - t0 > maxsec:
            done = it; log(f'PROGRESS{tag} MAXSEC stop after {done} of {sweeps} sweeps'); break
        X = sm.step()
        vt, h1, h2, d_ = config_quantities(sm, X, w)
        dev = max(dev, d_)
        h1s = h1 - E_ref * vt
        h2s = h2 - 2 * E_ref * h1 + E_ref ** 2 * vt
        q = np.stack([vt * vt, vt * h1s, h1s * h1s, h1s * h2s, h2s * h2s, vt * h2s], 1)
        for g in range(G): sums[it, g] = q[grp == g].sum(0)
        if (it + 1) % max(1, sweeps // 10) == 0:
            tot = sums[:it + 1].sum((0, 1)); r = from_sums(tot[:5] / tot[0], E_ref)
            log(f'PROGRESS{tag} sweep {it + 1}/{sweeps} E/N {r["E"] / sm.N:.6f} E1/N {r["E1"] / sm.N:.6f} acc {sm.acc:.2f} '
                f'{time.time() - t0:.0f}s evals {sm.n_eval} ({sm.t_eval / max(sm.n_eval, 1) * 1e3:.3f} ms/eval)')
    return sums[:done], dict(ratio_vs_fastupdate_maxrel=dev, sec=time.time() - t0, n_eval=sm.n_eval, sec_eval=sm.t_eval, samples=done * C, sweeps_done=done)


def analyse(sums, E_ref, N, C, sweeps):
    """full estimate and jackknife (over chain groups) errors; all energies per site."""
    G = sums.shape[1]; scale_E = 1.0 / N
    tot = sums.sum((0, 1)); n = C * sweeps
    full = from_sums(tot[:5] / n, E_ref)
    keys = ['E', 'var', 'alpha', 'E1', 'var1', 'mu3', 'mu4']
    jk = {k: [] for k in keys}
    for g in range(G):
        t = tot - sums[:, g].sum(0)
        r = from_sums(t[:5] / (n * (G - 1) / G), E_ref)
        for k in keys: jk[k].append(r[k])
    out = {}
    for k in keys:
        a = np.array(jk[k]); err = float(np.sqrt((G - 1) / G * np.sum((a - a.mean()) ** 2)))
        bias_corrected = G * full[k] - (G - 1) * a.mean()
        out[k] = dict(value=full[k], err=err, jk_bias=float(bias_corrected - full[k]))
    scale = {'E': 1 / N, 'E1': 1 / N, 'var': 1 / N, 'var1': 1 / N}
    out['per_site'] = {k: dict(value=out[k]['value'] * scale[k], err=out[k]['err'] * scale[k]) for k in scale}
    rr = np.array([(tot[2] - sums[:, g, 2].sum()) / (tot[5] - sums[:, g, 5].sum()) for g in range(G)])
    nsw = sums.shape[0]
    out['convergence'] = {}
    for frac in (0.25, 0.5, 1.0):
        k = max(int(round(nsw * frac)), 1); t = sums[:k].sum((0, 1)); r = from_sums(t[:5] / t[0], E_ref)
        out['convergence'][str(frac)] = dict(sweeps=k, E=r['E'] * scale_E, E1=r['E1'] * scale_E, var=r['var'] * scale_E, mu3=r['mu3'], mu4=r['mu4'], alpha=r['alpha'])
    out['check_h2'] = dict(s2_over_s2b=float(tot[2] / tot[5]),                     # <h1'^2> / <vt h2'> must be 1 (Hermiticity)
                           err=float(np.sqrt((G - 1) / G * np.sum((rr - rr.mean()) ** 2))))
    return out


def series(sums, E_ref):
    tot = sums.sum(1)                          # (sweeps, 6)
    E0 = np.array([E_ref + t[1] / t[0] for t in tot])
    E1 = np.array([from_sums(t[:5], E_ref)['E1'] for t in tot])
    return E0, E1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--params', required=True); ap.add_argument('--L', type=int, default=8)
    ap.add_argument('--proj', default='translations'); ap.add_argument('--chains', type=int, default=1024)
    ap.add_argument('--sweeps', type=int, default=40); ap.add_argument('--seeds', default='1,2'); ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--therm', type=int, default=100); ap.add_argument('--groups', type=int, default=16)
    ap.add_argument('--maxsec_seed', type=float, default=1e18); ap.add_argument('--stageA_sweeps', type=int, default=20); ap.add_argument('--stageA_chains', type=int, default=512)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prog = os.path.join(a.out, 'progress.txt')

    def log(msg):
        line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {msg}'; print(line, flush=True); open(prog, 'a').write(line + '\n')
    z = np.load(a.params); use_rbm = bool(z['rbm']) if 'rbm' in z.files else True
    lat = Lattice(a.L, 0.5, a.proj); N = lat.N
    log(f'PROGRESS vmc_moments L {a.L} proj {a.proj} images {lat.nI} params {a.params} chains {a.chains} sweeps {a.sweeps}')
    sm = Sampler({k: z[k] for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}, lat, use_rbm, a.chunk)
    w, Ec, infoA = stage_phase(sm, a.stageA_chains, a.stageA_sweeps, 0, a.therm, log)
    # short pass with the phase to centre the moments (E_ref = <v|H|v> estimate)
    s0, _ = stage_moments(sm, w, Ec, a.stageA_chains, max(a.stageA_sweeps // 2, 4), 0, a.therm, 4, log, tag=' B0')
    E_ref = Ec + s0[:, :, 1].sum() / s0[:, :, 0].sum()
    infoA.update(E_ref=E_ref, E_ref_site=E_ref / N)
    json.dump(infoA, open(os.path.join(a.out, 'stage_A.json'), 'w'), indent=1)
    log(f'PROGRESS stage A/B0 done: w {w:.6f} E_ref/N {E_ref / N:.6f}')
    all_sums = []
    for seed in [int(s) for s in a.seeds.split(',')]:
        sums, info = stage_moments(sm, w, E_ref, a.chains, a.sweeps, seed, a.therm, a.groups, log, tag=f' s{seed}', maxsec=a.maxsec_seed)
        res = analyse(sums, E_ref, N, a.chains, info['sweeps_done'])
        res.update(info=info, seed=seed, w=[w.real, w.imag], E_ref=E_ref, L=a.L, proj=a.proj, params=a.params)
        all_sums.append(sums)
        E0s, E1s = series(sums, E_ref)
        np.save(os.path.join(a.out, f'E0_s{seed}.npy'), E0s / N); np.save(os.path.join(a.out, f'E1_s{seed}.npy'), E1s / N)
        np.savez(os.path.join(a.out, f'vmc_sums_s{seed}.npz'), sums=sums, E_ref=E_ref)
        json.dump(res, open(os.path.join(a.out, f'vmc_s{seed}.json'), 'w'), indent=1, default=float)
        ps = res['per_site']
        log(f'PROGRESS RESULT seed {seed}: p=0 E/N {ps["E"]["value"]:.6f} +- {ps["E"]["err"]:.1e} var/N {ps["var"]["value"]:.4e} +- {ps["var"]["err"]:.1e}; '
            f'alpha {res["alpha"]["value"]:.5f} +- {res["alpha"]["err"]:.1e}; p=1 E1/N {ps["E1"]["value"]:.6f} +- {ps["E1"]["err"]:.1e} var1/N {ps["var1"]["value"]:.4e}; '
            f'check <h1^2>/<vt h2> {res["check_h2"]["s2_over_s2b"]:.6f}; fast-update ratio dev {info["ratio_vs_fastupdate_maxrel"]:.1e}; '
            f'{info["sec"]:.0f}s for {info["samples"]} samples')
    # pooled estimate over the independent seeds (same w and E_ref) and the constants file for the walkers
    n_by = [x.shape[0] for x in all_sums]
    pooled = np.concatenate(all_sums, 0)
    resP = analyse(pooled, E_ref, N, a.chains, pooled.shape[0])
    resP.update(w=[w.real, w.imag], E_ref=E_ref, L=a.L, proj=a.proj, params=a.params, sweeps_by_seed=n_by)
    json.dump(resP, open(os.path.join(a.out, 'vmc_pooled.json'), 'w'), indent=1, default=float)
    json.dump(dict(alpha=resP['alpha']['value'], E=resP['E']['value'], w=[w.real, w.imag], source='vmc_pooled.json', params=a.params),
              open(os.path.join(a.out, 'consts.json'), 'w'), indent=1)
    pp = resP['per_site']
    log(f'PROGRESS RESULT pooled: p=0 E/N {pp["E"]["value"]:.6f} +- {pp["E"]["err"]:.1e} var/N {pp["var"]["value"]:.4e}; alpha {resP["alpha"]["value"]:.5f}; '
        f'p=1 E1/N {pp["E1"]["value"]:.6f} +- {pp["E1"]["err"]:.1e} var1/N {pp["var1"]["value"]:.4e}')
    log('VMC_DONE')
