#!/usr/bin/env python3
"""Exact scaling test: per-site gain of ONE Lanczos step vs ONE Krylov sign step, J1-J2 (J2/J1=0.5) periodic tori,
N = 16..36, fully symmetric sector (k=0, A1, flip +).  Everything exact (full sector vectors, no sampling).

Reuses (imported unmodified): krylov_sign_structure/experiments/closed_fn_krylov_sym6x6.py (Sym, krylov, init_amplitude),
torus_cluster.py, and experiments/lanczos_baseline_6x6/lanczos_lib.py (lanczos_ritz: alpha from the 2x2 problem in span{psi,H psi}).

Guides (all in the sector basis, vector v>=0 amplitude per orbit rep, sign s=+-1):
  A   exact |psi0| + Marshall
  B   |GS(J2=0)| + Marshall
  Cx  |psi0| exp(eps) + exact sign sgn(psi0)    eps iid N(0, sigma^2) per orbit rep (renormalised)
  Cm  |psi0| exp(eps) + Marshall                same eps
Operations on each guide psi = s v:
  K   one Krylov sign step (exact energy-optimal threshold, S.krylov)           [1 H-application for r + 1 for energy pass]
  L1  one Lanczos step (1+alpha H)psi, optimal alpha                            [2 matvecs]
  L2  two Lanczos steps = lowest Ritz vector in span{psi, H psi, H^2 psi}       [3 matvecs]
  L1L1 two sequential single-alpha steps
  KL1 Krylov then L1 ;  KL2: Krylov then Ritz(p=2)
Scores per site: dE=(E-E0)/N, variance/N, wrong-sign weight w_s=(1-|sum p0 sgn(psi) sgn(psi0)|)/2.
"""
import argparse, json, os, sys, time, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.environ.get("FN_CODE", "/home/a/A.Otaifi/chatty_fn6x6/code"),
          os.environ.get("LANCZOS_LIB", os.path.join(HERE, "..", "lanczos_baseline_6x6"))):
    sys.path.insert(0, p)
import closed_fn_krylov_sym6x6 as C
import ed_common as ec
from lanczos_lib import lanczos_ritz

ap = argparse.ArgumentParser()
ap.add_argument('--L', type=int, default=4)
ap.add_argument('--torus', default=None)
ap.add_argument('--tag', required=True)
ap.add_argument('--sigma', type=float, default=0.05)
ap.add_argument('--seeds', type=int, default=4)
ap.add_argument('--gs-vec', default=None); ap.add_argument('--gs-states', default=None)
ap.add_argument('--cache-dir', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--guides', default='A,B,Cx,Cm', help='subset of A,B,Cx,Cm,A2,C2,J')
ap.add_argument('--kappa', type=float, default=0.0, help='guide J: |psi0| exp(kappa (n_antiNN - <n_antiNN>)), exact signs')
ap.add_argument('--calibrate-kappa', default=None)
ap.add_argument('--calibrate', default=None, help='comma list of sigmas: print mean dE/site of Cx guides and exit')
args = ap.parse_args()
log = lambda *a: print(*a, flush=True)
cluster = None; ctag = f"L{args.L}"
if args.torus:
    import torus_cluster
    tv = [int(z) for z in args.torus.split(',')]
    cluster = torus_cluster.Cluster((tv[0], tv[1]), (tv[2], tv[3]))
    ctag = f"T{'_'.join(map(str, tv))}".replace('-', 'm')
os.makedirs(args.cache_dir, exist_ok=True)
t00 = time.time()
S = C.Sym(args.L, 0.5, log=log, cache=os.path.join(args.cache_dir, f"orb_{ctag}.npz"), cluster=cluster)
N = S.N
E0, v0 = C.target_gs(S, types.SimpleNamespace(gs_vec=args.gs_vec, gs_states=args.gs_states, ed_tol=1e-12), log)
a0 = np.abs(v0); sg0 = np.where(v0 >= 0, 1, -1).astype(np.int8); p0 = v0 * v0
marsh = C.canonical(S.marshall.copy())


def meas(psi):
    psi = psi / np.linalg.norm(psi)
    Hp = S.H(psi)
    E = float(psi @ Hp)
    var = float(Hp @ Hp) - E * E
    O = float(np.sum(p0 * np.where(psi >= 0, 1, -1) * sg0))
    return dict(dE=(E - E0) / N, var=var / N, w_s=max(0.0, (1 - abs(O)) / 2))


def noisy(seed, sigma):
    rng = np.random.default_rng(1000 + seed)
    v = a0 * np.exp(sigma * rng.standard_normal(S.D))
    return v / np.linalg.norm(v)


if args.calibrate:
    for sg in [float(z) for z in args.calibrate.split(',')]:
        d = [meas(noisy(sd, sg) * sg0)['dE'] for sd in range(args.seeds)]
        log('CAL', ctag, sg, np.mean(d), np.std(d))
    sys.exit(0)

G = args.guides.split(',')
# ---- J2=0 amplitude (guide B)
EB = float('nan')
if 'B' in G:
    vB, EB = C.init_amplitude(S, 0.0, 1e-12, log=log, cache=os.path.join(args.cache_dir, f"j20_{ctag}.npz"))
    vB = vB / np.linalg.norm(vB)
    log(f"[B] J2=0 E/N={EB/N:.8f}  (guide B uses |GS(J2=0)|)")

# ---- cost: mean number of antiparallel bonds (= distinct off-diagonal neighbours) weighted by p0
nn, nnn = (ec.bonds(args.L) if cluster is None else (cluster.nn, cluster.nnn))
st = S.states
def anti(bonds):
    c = np.zeros(S.D, dtype=np.int16)
    for (i, j) in bonds:
        c += (((st >> np.uint64(i)) ^ (st >> np.uint64(j))) & np.uint64(1)).astype(np.int16)
    return c
n1 = anti(nn); n2 = anti(nnn)
cost = dict(nn_hops=float(np.sum(p0 * n1)), nnn_hops=float(np.sum(p0 * n2)), nbonds=len(nn) + len(nnn))
cost['hops_per_config'] = cost['nn_hops'] + cost['nnn_hops']
log('[cost]', cost)
gJ = n1.astype(np.float64) - float(np.sum(p0 * n1))
del n1, n2


def run_guide(name, v, s):
    t0 = time.time()
    psi = v * s
    out = dict(start=meas(psi))
    # Krylov
    sK, emin, G, r = S.krylov(v, s)
    psiK = v * sK
    out['K'] = meas(psiK)
    out['K']['emin_check'] = (emin - E0) / N
    # Lanczos
    rz = lanczos_ritz(S.H, psi, 2)
    out['L1'] = meas(rz[1]['psi']); out['L1']['alpha'] = rz[1]['alpha']
    out['L2'] = meas(rz[2]['psi'])
    rz11 = lanczos_ritz(S.H, rz[1]['psi'], 1)
    out['L1L1'] = meas(rz11[1]['psi']); out['L1L1']['alpha'] = rz11[1]['alpha']
    rzk = lanczos_ritz(S.H, psiK, 2)
    out['KL1'] = meas(rzk[1]['psi']); out['KL1']['alpha'] = rzk[1]['alpha']
    out['KL2'] = meas(rzk[2]['psi'])
    log('GUIDE', name, json.dumps({k: {kk: float(f"{vv:.4e}") for kk, vv in d.items()} for k, d in out.items()}),
        f'{time.time()-t0:.1f}s')
    return out


if args.calibrate_kappa:
    for kp in [float(z) for z in args.calibrate_kappa.split(',')]:
        vj = a0 * np.exp(kp * gJ); vj /= np.linalg.norm(vj)
        log('CALK', ctag, kp, meas(vj * sg0)['dE'])
    sys.exit(0)
res = {g: [] for g in G}
if 'J' in G:
    vj = a0 * np.exp(args.kappa * gJ); vj /= np.linalg.norm(vj)
    res['J'].append(run_guide('J', vj, sg0))
if 'A' in G: res['A'].append(run_guide('A', a0, marsh))
if 'B' in G: res['B'].append(run_guide('B', vB, marsh))
if 'A2' in G or 'C2' in G:
    # exact amplitude, signs after two label-free Krylov steps from Marshall (the depth-vs-N ladder, d=2)
    s1, _, _, _ = S.krylov(a0, marsh); s2, _, _, _ = S.krylov(a0, s1)
    log('[A2] signs after 2 Krylov steps: w_s =', meas(a0 * s2)['w_s'])
if 'A2' in G: res['A2'].append(run_guide('A2', a0, s2))
for sd in range(args.seeds):
    vc = noisy(sd, args.sigma)
    if 'Cx' in G: res['Cx'].append(run_guide(f'Cx_s{sd}', vc, sg0))
    if 'Cm' in G: res['Cm'].append(run_guide(f'Cm_s{sd}', vc, marsh))
    if 'C2' in G: res['C2'].append(run_guide(f'C2_s{sd}', vc, s2))
out = dict(cluster=ctag, N=N, D=S.D, E0=E0, E0_per_site=E0 / N, sigma=args.sigma, seeds=args.seeds, cost=cost,
           J20_E_per_site=EB / N, matvecs=S.nmv, wall_s=time.time() - t00, guides=res,
           note='exact sector (k=0,A1,flip+); see scaling.py docstring')
with open(args.out, 'w') as f:
    json.dump(out, f, indent=1)
log('WROTE', args.out, f'{time.time()-t00:.0f}s matvecs={S.nmv}')
