"""Fixed-node GFMC at general L (8x8: 'translations' x spin-flip RBM+PP guide) with the p = 0 guide Re[w psi_P] or the p = 1 guide
(one signed Lanczos hop, alpha from the Ritz identity with the moments from VMC, vmc_moments.py).  Absolute energies per site.

  python gfmc_run.py OUT --params P.npz --consts vmc_s1.json --p 1 --L 8 --proj translations --M 512 --dtau 0.2 --blocks 200 --burn 50 --seed 1
OUT/progress.txt (PROGRESS lines, last `GFMC_DONE ...`), OUT/ckpt.npz (walkers + block series, atomic, every --ckpt_every blocks;
--resume continues bit-identically), OUT/blocks.npz, OUT/result.json.
--consts: json of vmc_moments.py (keys alpha.value, E, w, E_ref ...) or a small json {"alpha":.., "E":.., "w":[re,im]} (E total).
The gfmc6_run.py path (6x6, full projection, table_A constants) is left unchanged; this script reproduces it with
--L 6 --proj full --consts <json from table_A> (see verify_general_6x6.py).
"""
import argparse, json, os, sys, time, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'rbmpp_6x6'))
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from rbmpp_lib import Lattice, make_fns, make_sampler, random_configs, to32
from gfmc_fn_lib import RBMPPGuideG, GFMC, blocked_error, iat


def load_consts(path, p):
    c = json.load(open(path))
    if 'alpha' in c and isinstance(c['alpha'], dict):                # vmc_moments.py output
        alpha1 = c['alpha']['value']; E = c['E']['value']
    else:
        alpha1 = c['alpha']; E = c['E']
    w = complex(*c['w']) if 'w' in c else 1.0 + 0j
    return dict(alpha=(alpha1 if p == 1 else 0.0), E=E, w=w, alpha1=alpha1)


def init_walkers(lat, z, M, seed, use_rbm, nsweep=2000):
    p = {k: jnp.asarray(z[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}
    sweep = make_sampler(lat, make_fns(lat, jnp.float32, use_rbm))
    S = random_configs(lat.N, M, seed); key = jax.random.PRNGKey(seed); p32 = to32(p)
    for i in range(nsweep // 100):
        key, k = jax.random.split(key); S, acc = sweep(p32, S, k, 100)
    return np.asarray(S).astype(np.int8), float(acc)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--params', required=True); ap.add_argument('--consts', required=True)
    ap.add_argument('--p', type=int, default=1, choices=[0, 1]); ap.add_argument('--L', type=int, default=8); ap.add_argument('--proj', default='translations')
    ap.add_argument('--M', type=int, default=512); ap.add_argument('--dtau', type=float, default=0.2)
    ap.add_argument('--blocks', type=int, default=200); ap.add_argument('--burn', type=int, default=50)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--ckpt_every', type=int, default=10); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--maxsec', type=float, default=1e18, help='stop cleanly after this many seconds (checkpoint written)')
    ap.add_argument('--ref', type=float, default=-0.498886, help='reference energy per site printed in the progress lines (published RBM+PP)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prog = os.path.join(a.out, 'progress.txt')

    def P(msg):
        line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}'
        print(line, flush=True); open(prog, 'a').write(line + '\n')

    z = np.load(a.params); use_rbm = bool(z['rbm']) if 'rbm' in z.files else True
    lat = Lattice(a.L, 0.5, a.proj); N = lat.N
    c = load_consts(a.consts, a.p)
    g = RBMPPGuideG({k: z[k] for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}, lat, c['alpha'], c['E'], w=c['w'], chunk=a.chunk, use_rbm=use_rbm)
    P(f'guide L {a.L} proj {a.proj} images {lat.nI} p {a.p} alpha {c["alpha"]:.8f} E/N {c["E"] / N:.10f} w {c["w"]:.6f} params {a.params} consts {a.consts} '
      f'M {a.M} dtau {a.dtau} seed {a.seed}')
    ck = os.path.join(a.out, 'ckpt.npz')
    rec = []; b0 = 0
    if a.resume and os.path.exists(ck):
        zz = np.load(ck, allow_pickle=True)
        X0 = zz['X']; rec = list(zz['rec'].tolist()); b0 = int(zz['nb'])
        gf = GFMC(g, X0, a.dtau, a.seed)
        gf.rng.bit_generator.state = pickle.loads(zz['rng'].tobytes())
        P(f'RESUMED at block {b0} from {ck}')
    else:
        t0 = time.time(); X0, acc = init_walkers(lat, z, a.M, a.seed, use_rbm)
        P(f'init walkers from |psi_P|^2 Metropolis, acc {acc:.3f}, {time.time() - t0:.0f}s')
        gf = GFMC(g, X0, a.dtau, a.seed)
        P(f'initial rows {a.M} walkers {g.n_eval} amplitude evaluations {g.t_eval:.0f}s')
    t_start = time.time(); tot = a.burn + a.blocks

    def save(nb):
        tmp = ck + '.tmp.npz'
        np.savez(tmp, X=gf.X, rec=np.array(rec, dtype=object), nb=nb, rng=np.frombuffer(pickle.dumps(gf.rng.bit_generator.state), np.uint8))
        os.replace(tmp, ck)
    stopped = False
    for b in range(b0, tot):
        tb = time.time(); n0 = g.n_eval
        r = gf.block(); r['sec'] = time.time() - tb; r['evals'] = g.n_eval - n0; r['block'] = b
        rec.append(r)
        P(f'block {b + 1}/{tot} E_mix/N {r["emix"] / N:.7f} E_grow/N {r["egrow"] / N:.7f} d_ref {r["emix"] / N - a.ref:+.3e} ess {r["ess"]:.0f} '
          f'elvar {r["elvar"]:.3f} hops {r["hops"]} iters {r["iters"]} evals {r["evals"]} sec {r["sec"]:.1f} maxIm {g.max_im:.1e}')
        if (b + 1) % a.ckpt_every == 0: save(b + 1)
        if time.time() - t_start > a.maxsec:
            save(b + 1); stopped = True; P(f'MAXSEC reached at block {b + 1}'); break
    nb = len(rec); save(nb)
    em = np.array([r['emix'] for r in rec[a.burn:]]); eg = np.array([r['egrow'] for r in rec[a.burn:]])
    np.savez(os.path.join(a.out, 'blocks.npz'), emix=np.array([r['emix'] for r in rec]), egrow=np.array([r['egrow'] for r in rec]),
             ess=np.array([r['ess'] for r in rec]), elvar=np.array([r['elvar'] for r in rec]), hops=np.array([r['hops'] for r in rec]),
             sec=np.array([r['sec'] for r in rec]), evals=np.array([r['evals'] for r in rec]), burn=a.burn, dtau=a.dtau, M=a.M, N=N, p=a.p,
             alpha=c['alpha'], E_guide=c['E'])
    out = dict(args=vars(a), consts=dict(alpha=c['alpha'], E=c['E'], w=[c['w'].real, c['w'].imag]), nblocks=nb, completed=not stopped, N=N)
    if len(em) >= 32:
        for name, x in (('mix', em), ('grow', eg)):
            m, errs, tau = blocked_error(x)
            out[name] = dict(E_site=m / N, err_site=max(errs[-4:]) / N, err_level0_site=errs[0] / N, tau_int_blocks=iat(x), tau_blocking=tau)
    out['sec_per_block_median'] = float(np.median([r['sec'] for r in rec])); out['evals_per_block_mean'] = float(np.mean([r['evals'] for r in rec]))
    out['sec_per_walker_tau'] = out['sec_per_block_median'] / (a.M * a.dtau)
    json.dump(out, open(os.path.join(a.out, 'result.json'), 'w'), indent=1, default=float)
    P(f'GFMC_DONE blocks {nb} completed {not stopped} ' + (f'E_mix/N {out["mix"]["E_site"]:.7f} +- {out["mix"]["err_site"]:.1e}' if 'mix' in out else ''))
