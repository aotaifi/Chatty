#!/usr/bin/env python3
"""Stage C (CPU, lattice_symmetries env): exact scoring of trained sign nets, incl. one/two exact Krylov hops.

For every run in <runs>/N<N>/*/ with logits.npy (from sl_train.py) and no evals/N<N>/<name>.json yet:
  pred = sgn(logit) on every orbit;  stored sign  s_net = s_{k-1} * pred  (target c_k)  or  Marshall * pred  (gs).
  w_lab   = sum p0 [pred != label]                 (label error; label = c_k or sigma0)
  w_net   = w_s(s_net) vs ED sign                  (= sign error of the stored sign)
  hop     : s_hop = s_net * sgn[T - r(a, s_net)], energy-optimal T (exact amplitude, as in the exact chain)
  w_hop   = w_s(s_hop);  ref = s_{k+1} (exact chain one step further) for c_k, exact sign for gs
  d_hop   = sum p0 [s_hop != ref]; repaired / residual / new error weight (net label-error set vs hop error set)
  hop2    : a second exact hop (w_hop2, d_hop2 vs s_{k+2})
  eps     : relative energy errors of (a, s_net), (a, s_hop)
  resolved: weight by decade of per-configuration |psi0|^2; margins delta = r_{k-1} - T_{k-1} (and r_0 - T_0);
            logit distributions of all vs wrong states; seen (training) vs unseen orbits.
"""
import argparse, json, os, sys, time, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sl_common as SC
import closed_fn_krylov_sym6x6 as C
import torus_cluster as tc

ap = argparse.ArgumentParser()
ap.add_argument('--N', type=int, required=True)
ap.add_argument('--data', default='/home/a/A.Otaifi/chatty_signlearn/data')
ap.add_argument('--runs', default='/home/a/A.Otaifi/chatty_signlearn/runs')
ap.add_argument('--evals', default='/home/a/A.Otaifi/chatty_signlearn/evals')
ap.add_argument('--cache-dir', default='/home/a/A.Otaifi/ChattyRun/j1j2/sign_design_tests/cache')
ap.add_argument('--force', action='store_true')
ap.add_argument('--pattern', default='*')
ap.add_argument('--loop', type=int, default=0, help='poll for new runs for this many seconds')
args = ap.parse_args()
log = lambda *a: print(*a, flush=True)
N = args.N; dd = SC.data_dir(args.data, N)
meta = json.load(open(f'{dd}/meta.json'))
cfg = SC.CLUSTERS[N]
cluster = None; L = cfg['L']
if cfg['torus'] is not None:
    t = cfg['torus']; cluster = tc.Cluster((t[0], t[1]), (t[2], t[3])); ctag = 'T' + '_'.join(map(str, t)).replace('-', 'm')
else:
    ctag = f'L{L}'
S = C.Sym(L, 0.5, log=log, cache=os.path.join(args.cache_dir, f'orb_{ctag}.npz'), cluster=cluster)
assert np.array_equal(S.states, np.load(f'{dd}/states.npy'))
p0 = np.load(f'{dd}/p0.npy'); a = np.sqrt(p0); sg0 = np.load(f'{dd}/sg0.npy'); M = np.load(f'{dd}/marshall.npy')
sig0 = np.load(f'{dd}/sigma0.npy'); n_orb = np.load(f'{dd}/n_orb.npy').astype(np.float64)
E0 = meta['E0']; kmax = len(meta['steps']) - 1
sk = {k: np.load(f'{dd}/s_{k}.npy') for k in range(kmax + 1)}
px = p0 / n_orb
dec = np.clip(np.floor(np.log10(np.maximum(px, 1e-300))), -40, 0).astype(int) + 40     # 0..40
DEC = np.arange(41) - 40
LOGD = np.linspace(-6, 3, 28)                 # log10 |delta| bin edges
LOGIT = np.linspace(-20, 20, 81)
tr_cache = {}


def energy(s):
    psi = a * s; return float(psi @ S.H(psi))


def align(s, ref):
    return s if np.sum(p0 * s * ref) >= 0 else -s


def wdiff(s, ref):
    return float(p0[align(s, ref) != ref].sum())


def dhist(mask, delta):
    ld = np.clip(np.log10(np.maximum(np.abs(delta), 1e-12)), LOGD[0], LOGD[-1] - 1e-9)
    out = {}
    for sgn, m2 in (('pos', delta >= 0), ('neg', delta < 0)):
        mm = mask & m2
        out[sgn] = np.histogram(ld[mm], bins=LOGD, weights=p0[mm])[0].tolist()
    return out


def wquant(x, w, qs=(0.1, 0.5, 0.9)):
    if w.sum() <= 0: return [None] * len(qs)
    o = np.argsort(x); c = np.cumsum(w[o]) / w.sum()
    return [float(x[o][min(len(x) - 1, np.searchsorted(c, q))]) for q in qs]


def score(rd):
    run = json.load(open(f'{rd}/run.json')); A = run['args']
    lg = np.load(f'{rd}/logits.npy').astype(np.float64)
    pred = np.where(lg >= 0, 1, -1).astype(np.int8)
    tgt = A['target']
    if tgt == 'gs':
        k = None; y = sig0; s_net = (M * pred).astype(np.int8); ref = sg0; ref2 = sg0
        delta = np.load(f'{dd}/r_0.npy') - meta['steps'][1]['T_prev']
    elif tgt.startswith('m'):
        k = int(tgt[1:]); y = (sk[k] * M).astype(np.int8)
        if np.sum(p0 * y) < 0: y = -y
        s_net = (M * pred).astype(np.int8)
        ref = sk[k + 1] if k + 1 <= kmax else None; ref2 = sk[k + 2] if k + 2 <= kmax else None
        delta = np.load(f'{dd}/r_{k-1}.npy') - meta['steps'][k]['T_prev']
    else:
        k = int(tgt[1:]); y = np.load(f'{dd}/c_{k}.npy')
        s_net = (sk[k - 1] * pred).astype(np.int8)
        ref = sk[k + 1] if k + 1 <= kmax else None; ref2 = sk[k + 2] if k + 2 <= kmax else None
        delta = np.load(f'{dd}/r_{k-1}.npy') - meta['steps'][k]['T_prev']
    wrong = pred != y
    key = (A['beta'], A['seed'])
    if key not in tr_cache:
        tr_cache.clear(); tr_cache[key] = dict(np.load(f"{dd}/train_b{A['beta']}_s{A['seed']}.npz"))
    idx = tr_cache[key][f"{'nbr' if A['nbr'] else 'smp'}_{A['n']}"]
    seen = np.zeros(S.D, bool); seen[idx] = True
    t0 = time.time()
    s_hop, _, _, _ = S.krylov(a, s_net)
    s_hop2, _, _, _ = S.krylov(a, s_hop)
    t_hop = time.time() - t0
    E_net, E_hop, E_hop2 = energy(s_net), energy(s_hop), energy(s_hop2)
    res = dict(name=run['name'], args=A, nparams=run['nparams'], n_train=run['n_train'], nsteps=run['nsteps'],
               t_train=run['t_train'], t_eval=run['t_eval'], device=run['device'], final_bce=run['hist'][-1]['bce'],
               N=N, D=int(S.D), target=tgt, k=k,
               w_triv=float(p0[y < 0].sum()), w_lab=float(p0[wrong].sum()),
               w_net=wdiff(s_net, sg0), w_hop=wdiff(s_hop, sg0), w_hop2=wdiff(s_hop2, sg0),
               eps_net=(E_net - E0) / abs(E0), eps_hop=(E_hop - E0) / abs(E0), eps_hop2=(E_hop2 - E0) / abs(E0),
               p0_seen=float(p0[seen].sum()), frac_orbits_seen=float(seen.mean()),
               w_lab_seen=float(p0[wrong & seen].sum()), w_lab_unseen=float(p0[wrong & ~seen].sum()),
               err_train_unweighted=float(wrong[seen].mean()), frac_orbits_wrong=float(wrong.mean()),
               n_wrong_orbits=int(wrong.sum()), t_hop=t_hop)
    if k is not None:
        res['w_ref_k'] = meta['steps'][k]['w_s']
        res['w_ref_k1'] = meta['steps'][k + 1]['w_s'] if k + 1 <= kmax else None
        res['w_ref_k2'] = meta['steps'][k + 2]['w_s'] if k + 2 <= kmax else None
    if ref is not None:
        he = align(s_hop, ref) != ref
        res.update(d_hop=float(p0[he].sum()), hop_repaired=float(p0[wrong & ~he].sum()),
                   hop_residual=float(p0[wrong & he].sum()), hop_new=float(p0[~wrong & he].sum()))
    else:
        he = None
    if ref2 is not None:
        res['d_hop2'] = wdiff(s_hop2, ref2)
    # weight-resolved by decade of per-configuration |psi0|^2
    ed_net = align(s_net, sg0) != sg0; ed_hop = align(s_hop, sg0) != sg0
    res['decades'] = dict(dec=DEC.tolist(), tot=np.bincount(dec, p0, 41).tolist(), n=np.bincount(dec, None, 41).tolist(),
                          lab_wrong=np.bincount(dec, p0 * wrong, 41).tolist(), n_lab_wrong=np.bincount(dec, wrong, 41).tolist(),
                          ed_net=np.bincount(dec, p0 * ed_net, 41).tolist(), ed_hop=np.bincount(dec, p0 * ed_hop, 41).tolist(),
                          ref_hop=None if he is None else np.bincount(dec, p0 * he, 41).tolist(),
                          unseen=np.bincount(dec, p0 * ~seen, 41).tolist(), lab_wrong_unseen=np.bincount(dec, p0 * (wrong & ~seen), 41).tolist())
    # margins
    ad = np.abs(delta)
    res['margin'] = dict(edges_log10=LOGD.tolist(), all=dhist(np.ones(S.D, bool), delta), wrong=dhist(wrong, delta),
                         q_all=wquant(ad, p0), q_wrong=wquant(ad[wrong], p0[wrong]),
                         frac_wrong_below={str(x): float(p0[wrong & (ad < x)].sum() / max(p0[wrong].sum(), 1e-300)) for x in (0.01, 0.1, 1.0, 3.0)},
                         frac_all_below={str(x): float(p0[ad < x].sum()) for x in (0.01, 0.1, 1.0, 3.0)},
                         frac_flip_below={str(x): float(p0[(y < 0) & (ad < x)].sum() / max(p0[y < 0].sum(), 1e-300)) for x in (0.01, 0.1, 1.0, 3.0)})
    alg = np.abs(lg)
    res['logit'] = dict(edges=LOGIT.tolist(), all=np.histogram(np.clip(lg, -19.99, 19.99), LOGIT, weights=p0)[0].tolist(),
                        wrong=np.histogram(np.clip(lg[wrong], -19.99, 19.99), LOGIT, weights=p0[wrong])[0].tolist(),
                        q_abs_all=wquant(alg, p0), q_abs_wrong=wquant(alg[wrong], p0[wrong]),
                        frac_wrong_abs_below1=float(p0[wrong & (alg < 1)].sum() / max(p0[wrong].sum(), 1e-300)),
                        frac_all_abs_below1=float(p0[alg < 1].sum()),
                        corr_logit_delta=float(np.corrcoef(lg[:2_000_000], -np.arcsinh(delta[:2_000_000]))[0, 1]))
    return res


os.makedirs(f'{args.evals}/N{N}', exist_ok=True)
t_end = time.time() + args.loop
while True:
    todo = []
    for rd in sorted(glob.glob(f'{args.runs}/N{N}/{args.pattern}')):
        nm = os.path.basename(rd); ef = f'{args.evals}/N{N}/{nm}.json'
        if os.path.exists(f'{rd}/run.json') and os.path.exists(f'{rd}/logits.npy') and (args.force or not os.path.exists(ef)):
            todo.append((rd, ef))
    for rd, ef in todo:
        t0 = time.time()
        try:
            res = score(rd)
        except Exception:
            import traceback; traceback.print_exc(); log('FAILED', rd); continue
        res['t_score'] = time.time() - t0
        json.dump(res, open(ef, 'w'), indent=1)
        log('EVAL', json.dumps({k: res.get(k) for k in ('name', 'w_triv', 'w_lab', 'w_net', 'w_hop', 'd_hop', 'hop_residual', 'p0_seen', 'w_lab_unseen', 'eps_net', 'eps_hop', 't_score')}))
    if time.time() > t_end: break
    if not todo: time.sleep(60)
log('DONE')
