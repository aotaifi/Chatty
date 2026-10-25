"""6x6 J1-J2 (J2/J1 = 0.5) fixed-node GFMC with the RBM+PP + one signed Lanczos hop guide (M1.1).

  python gfmc6_run.py OUT --M 2000 --dtau 0.1 --blocks 400 --burn 50 --seed 1 [--resume]
Writes OUT/progress.txt (PROGRESS lines), OUT/ckpt.npz (walkers + block series; every --ckpt_every blocks, atomic),
OUT/blocks.npz (block series) and OUT/result.json at the end; the last progress line is `GFMC_DONE ...`.
Energies are per site relative to E0 = -0.503809653890878 (exact, table_A.json: E_site - dE_site).
alpha = (E_1 - E)/var from table_A.json lanczos p=0/1 (Ritz identity; verified exactly by verify_alpha_6x6.py).
"""
import argparse, json, os, sys, time, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'rbmpp_6x6'))
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from rbmpp_lib import Lattice, make_fns, make_sampler, random_configs, to32
from gfmc_fn_lib import RBMPPGuide, GFMC, blocked_error, iat

PARAMS = os.environ.get('GFMC_PARAMS', '/project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpT4_sr2_17164867/params_final.npz')
TABLE_A = os.environ.get('GFMC_TABLE_A', os.path.join(HERE, '..', '..', 'results', 'rbmpp_6x6', 'runs', 'table_A.json'))
N = 36


def guide_constants(tabA):
    t = json.load(open(tabA))
    E_site = t['guide']['E_site']; dE0 = t['guide']['dE_site']
    E0_site = E_site - dE0
    l0, l1 = t['lanczos'][0], t['lanczos'][1]
    alpha = (l1['dE_site'] - l0['dE_site']) / l0['var_site']
    return dict(E_site=E_site, E0_site=E0_site, alpha=alpha, EFN_target_site=l1['dE_FN_site'])


def make_guide(chunk, params=PARAMS, tabA=TABLE_A):
    c = guide_constants(tabA)
    z = np.load(params)
    lat = Lattice(6, 0.5, 'full')
    g = RBMPPGuide({k: z[k] for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}, lat, c['alpha'], c['E_site'] * N, chunk=chunk,
                   use_rbm=bool(z['rbm']) if 'rbm' in z.files else True)
    return g, lat, c


def init_walkers(lat, params, M, seed, nsweep=2000):
    """Metropolis (exchange moves) sample of |psi_P|^2, float32 acceptance, as the VMC trainer does."""
    z = np.load(params)
    p = {k: jnp.asarray(z[k]) for k in ('f', 'bre', 'bim', 'Wre', 'Wim')}
    fns32 = make_fns(lat, jnp.float32, bool(z['rbm']) if 'rbm' in z.files else True)
    sweep = make_sampler(lat, fns32)
    S = random_configs(36, M, seed)
    key = jax.random.PRNGKey(seed)
    p32 = to32(p)
    for i in range(nsweep // 100):
        key, k = jax.random.split(key)
        S, acc = sweep(p32, S, k, 100)
    return np.asarray(S).astype(np.int8), float(acc)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--M', type=int, default=1000); ap.add_argument('--dtau', type=float, default=0.1)
    ap.add_argument('--blocks', type=int, default=200); ap.add_argument('--burn', type=int, default=50)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--chunk', type=int, default=512)
    ap.add_argument('--ckpt_every', type=int, default=10); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--maxsec', type=float, default=1e18, help='stop cleanly after this many seconds (checkpoint written)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prog = os.path.join(a.out, 'progress.txt')

    def P(msg):
        line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}'
        print(line, flush=True); open(prog, 'a').write(line + '\n')

    g, lat, c = make_guide(a.chunk)
    P(f'guide alpha {c["alpha"]:.8f} E_site {c["E_site"]:.12f} E0_site {c["E0_site"]:.12f} target dE_FN {c["EFN_target_site"]:.6e}')
    ck = os.path.join(a.out, 'ckpt.npz')
    rec = []; b0 = 0
    if a.resume and os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        X0 = z['X']; rec = list(z['rec'].tolist()); b0 = int(z['nb'])
        gf = GFMC(g, X0, a.dtau, a.seed)
        gf.rng.bit_generator.state = pickle.loads(z['rng'].tobytes())
        P(f'RESUMED at block {b0} from {ck}')
    else:
        t0 = time.time(); X0, acc = init_walkers(lat, PARAMS, a.M, a.seed)
        P(f'init walkers from |psi_P|^2 Metropolis, acc {acc:.3f}, {time.time()-t0:.0f}s')
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
        E0 = c['E0_site'] * N
        P(f'block {b + 1}/{tot} dE_mix {(r["emix"] - E0) / N:.4e} dE_grow {(r["egrow"] - E0) / N:.4e} ess {r["ess"]:.0f} '
          f'hops {r["hops"]} iters {r["iters"]} evals {r["evals"]} sec {r["sec"]:.1f} maxIm {g.max_im:.1e}')
        if (b + 1) % a.ckpt_every == 0: save(b + 1)
        if time.time() - t_start > a.maxsec:
            save(b + 1); stopped = True; P(f'MAXSEC reached at block {b + 1}'); break
    nb = len(rec); save(nb)
    em = np.array([r['emix'] for r in rec[a.burn:]]); eg = np.array([r['egrow'] for r in rec[a.burn:]])
    E0 = c['E0_site'] * N
    np.savez(os.path.join(a.out, 'blocks.npz'), emix=np.array([r['emix'] for r in rec]), egrow=np.array([r['egrow'] for r in rec]),
             ess=np.array([r['ess'] for r in rec]), elvar=np.array([r['elvar'] for r in rec]), hops=np.array([r['hops'] for r in rec]),
             sec=np.array([r['sec'] for r in rec]), evals=np.array([r['evals'] for r in rec]), burn=a.burn, dtau=a.dtau, M=a.M, E0=E0)
    out = dict(args=vars(a), consts=c, nblocks=nb, completed=not stopped)
    if len(em) >= 32:
        for name, x in (('mix', em), ('grow', eg)):
            m, errs, tau = blocked_error(x)
            out[name] = dict(dE_site=(m - E0) / N, err_site=max(errs[-4:]) / N, err_level0_site=errs[0] / N, tau_int_blocks=iat(x),
                             tau_blocking=tau)
    out['sec_per_block_median'] = float(np.median([r['sec'] for r in rec])); out['evals_per_block_mean'] = float(np.mean([r['evals'] for r in rec]))
    json.dump(out, open(os.path.join(a.out, 'result.json'), 'w'), indent=1, default=float)
    P(f'GFMC_DONE blocks {nb} completed {not stopped} ' + (f'dE_mix {out["mix"]["dE_site"]:.4e} +- {out["mix"]["err_site"]:.1e}' if 'mix' in out else ''))
