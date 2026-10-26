"""L8 refit diagnosis (after F 3923271 ended NO_RISE): step size or data starvation?  Everything at v_0, from existing dumps, no GFMC.
  python l8_diag.py OUT --params0 P --consts C --dumps D1,D2 [--L 8] [--first_block B] [--parts a,b] [--groups 20]
(a) per-chain gradients g_c = E_{w~,c}[D] - E_{v^2}[D] (the VMC term from DISJOINT halves of the VMC chains for c = 1, 2, so shared VMC noise
    cannot fake a positive cosine), natural gradients d_c = (S + shift)^-1 g_c (S = (S_v + S_w)/2 as in l8_lib.sr_step, diagonally rescaled,
    shift relative), S-metric cosine cos = d1.S.d2 / sqrt(d1.S.d1 d2.S.d2), predicted held-out changes g_2.d_1 and g_1.d_2 (d at lr 1; the
    first-order change of (1/2) log R_held), for shift 0.1, 1, 10.  Jackknife: delete-one over G block groups (contiguous blocks; group k of
    both chains and VMC chain group k removed together); S is held fixed.  Reference: the same cosine between the two halves of chain 1 (blocks
    first half / second half: correlated across the cut, expected larger).
(b) short refits: steps x lr x shift, both assignments (train c1 / held c2 and train c2 / held c1), held-out R per step with error.
Writes OUT/diag.json and PROGRESS lines in OUT/progress.txt; last line DIAG_DONE.
"""
import argparse, json, os, sys, time
import numpy as np
import scipy.linalg as sl
HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'rbmpp_6x6')):
    sys.path.insert(0, d)
from rbmpp_lib import Lattice
from gfmc_run import load_consts
import l8_lib as LL

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--params0', required=True); ap.add_argument('--consts', required=True); ap.add_argument('--dumps', required=True)
ap.add_argument('--L', type=int, default=8); ap.add_argument('--proj', default='translations'); ap.add_argument('--first_block', type=int, default=None)
ap.add_argument('--parts', default='a,b'); ap.add_argument('--groups', type=int, default=20); ap.add_argument('--chains', type=int, default=1024)
ap.add_argument('--vmc_sweeps', type=int, default=16); ap.add_argument('--shifts', default='0.1,1,10'); ap.add_argument('--chunk', type=int, default=256)
ap.add_argument('--lrs', default='0.01,0.03,0.1'); ap.add_argument('--bshifts', default='0.1,1.0'); ap.add_argument('--steps', type=int, default=5)
ap.add_argument('--maxn', type=float, default=0.1, help='S-norm cap in (b); 0.1 so that lr (not the cap) sets the step (production: 0.02)')
ap.add_argument('--sweeps_per_step', type=int, default=8); ap.add_argument('--seed', type=int, default=1)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
prog = os.path.join(a.out, 'progress.txt')


def P(msg):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} PROGRESS {msg}'; print(line, flush=True); open(prog, 'a').write(line + '\n')


T0 = time.time()
z = np.load(a.params0); lat = Lattice(a.L, 0.5, a.proj)
c = load_consts(a.consts, 1); alpha, E, w = c['alpha'], c['E'], c['w']
model = LL.Model(lat, z, True, a.chunk); flat0 = np.asarray(model.flat0)
dirs = a.dumps.split(',')
D = LL.load_dumps(dirs, 0.0, a.first_block)
Gv = LL.guide_at(model, flat0, D['X'], alpha, E, w)
P(f'l8_diag L {a.L} walkers {len(D["p"])} chains {len(dirs)} guide values done {time.time() - T0:.0f}s')
out = dict(args=vars(a))
G = a.groups

if 'a' in a.parts:
    t = time.time()
    # walker per-group sums A (P-vectors) and B (scalars) per chain; groups = contiguous blocks
    wsum = {}
    for ch in range(len(dirs)):
        m = D['chain'] == ch
        blocks = D['block'][m]; ub = np.unique(blocks); gi = np.searchsorted(ub, blocks) * G // len(ub)
        nb = len(ub); pw = D['p'][m] / nb; cw = pw * (Gv['vt0'][m] / Gv['cg'][m]); Xc = D['X'][m]
        A = np.zeros((G, model.P)); B = np.zeros(G)
        for k in range(G):
            mk = gi == k; A[k] = model.wgrad(flat0, Xc[mk], cw[mk], w); B[k] = cw[mk].sum()
        wsum[ch] = (A, B, Xc, cw)
    P(f'(a) walker group gradients {time.time() - t:.0f}s')
    # VMC samples: chain index -> half (parity) and group (index mod G)
    vm = LL.VMC(model, a.chains, a.seed, therm=50); Ss = []
    for k in range(a.vmc_sweeps): Ss.append(vm.sample(flat0))
    S = np.concatenate(Ss); chain_idx = np.tile(np.arange(a.chains), a.vmc_sweeps)
    Dv = model.jac(flat0, S, w); l = model.logpsi(flat0, S); vt = (w * np.exp(1j * l.imag)).real; wt = vt ** 2
    half = chain_idx % 2; grp = (chain_idx // 2) % G
    VA = np.zeros((2, G, model.P)); VB = np.zeros((2, G))
    for h in (0, 1):
        for k in range(G):
            mk = (half == h) & (grp == k); VA[h, k] = wt[mk] @ Dv[mk]; VB[h, k] = wt[mk].sum()
    wn = wt / wt.sum(); Db = wn @ Dv; Sv = ((Dv - Db) * wn[:, None]).T @ (Dv - Db)
    rng = np.random.default_rng(7)
    Xall = np.concatenate([wsum[ch][2] for ch in wsum]); call = np.abs(np.concatenate([wsum[ch][3] / wsum[ch][3].sum() for ch in wsum]))
    idx = rng.choice(len(call), a.chains, p=call / call.sum()); Dw = model.jac(flat0, Xall[idx], w); Dwm = Dw - Dw.mean(0)
    Sm = 0.5 * (Sv + Dwm.T @ Dwm / len(idx))
    dg = np.diag(Sm).copy(); dg = np.maximum(dg, 1e-2 * np.median(dg[dg > 0])); sq = np.sqrt(dg); Sr = Sm / np.outer(sq, sq)
    P(f'(a) VMC {len(S)} samples, metric built {time.time() - t:.0f}s')

    def grad(ch, h, keep_w, keep_v):
        A, B = wsum[ch][0], wsum[ch][1]
        return A[keep_w].sum(0) / B[keep_w].sum() - VA[h][keep_v].sum(0) / VB[h][keep_v].sum()

    allk = np.ones(G, bool)
    res = {}
    for shift in [float(x) for x in a.shifts.split(',')]:
        Lc = sl.cho_factor(Sr + shift * np.eye(model.P))
        nat = lambda g: sl.cho_solve(Lc, g / sq) / sq
        Sn = lambda x, y: float(x @ Sm @ y)

        def stats(g1, g2):
            d1, d2 = nat(g1), nat(g2)
            return np.array([Sn(d1, d2) / np.sqrt(Sn(d1, d1) * Sn(d2, d2)), g2 @ d1, g1 @ d2, g2 @ d1 / np.sqrt(Sn(d1, d1)), g1 @ d2 / np.sqrt(Sn(d2, d2))])

        # cross-chain (chains 0 and 1, VMC halves 0 and 1)
        full = stats(grad(0, 0, allk, allk), grad(1, 1, allk, allk)); jk = []
        for k in range(G):
            kk = allk.copy(); kk[k] = False; jk.append(stats(grad(0, 0, kk, kk), grad(1, 1, kk, kk)))
        jk = np.array(jk); err = np.sqrt((G - 1) / G * np.sum((jk - jk.mean(0)) ** 2, 0))
        # same chain halves (chain 0: groups < G/2 vs >= G/2)
        lo = np.arange(G) < G // 2
        fullh = stats(grad(0, 0, lo, allk), grad(0, 1, ~lo, allk)); jh = []
        for k in range(G // 2):
            kw = allk.copy(); kw[k] = False; kw[k + G // 2] = False; kv = allk.copy(); kv[k] = False; kv[k + G // 2] = False
            jh.append(stats(grad(0, 0, lo & kw, kv), grad(0, 1, ~lo & kw, kv)))
        jh = np.array(jh); n2 = G // 2; errh = np.sqrt((n2 - 1) / n2 * np.sum((jh - jh.mean(0)) ** 2, 0))
        names = ['cos', 'g2.d1', 'g1.d2', 'g2.d1_per_Snorm', 'g1.d2_per_Snorm']
        res[str(shift)] = dict(cross={n: [float(full[i]), float(err[i])] for i, n in enumerate(names)},
                               halves_chain1={n: [float(fullh[i]), float(errh[i])] for i, n in enumerate(names)})
        P(f'(a) shift {shift}: cross-chain cos {full[0]:+.4f} +- {err[0]:.4f} ({full[0] / err[0]:+.1f} sig); g2.d1 {full[1]:+.3e} +- {err[1]:.1e}; '
          f'g1.d2 {full[2]:+.3e} +- {err[2]:.1e}; same-chain halves cos {fullh[0]:+.4f} +- {errh[0]:.4f}')
    out['a'] = res
    json.dump(out, open(os.path.join(a.out, 'diag.json'), 'w'), indent=1, default=float)

if 'b' in a.parts:
    out['b'] = []
    for hc in (1, 0):                                    # held-out chain index: 1 = train c1/held c2, 0 = train c2/held c1
        Dm = dict(D); Dm['held'] = D['chain'] == hc
        tr = LL.WalkerSet(Dm, Gv, ~Dm['held'], w, 2); ho = LL.WalkerSet(Dm, Gv, Dm['held'], w, 2)
        for lr in [float(x) for x in a.lrs.split(',')]:
            for shift in [float(x) for x in a.bshifts.split(',')]:
                t = time.time(); vm = LL.VMC(model, a.chains, a.seed + 10, therm=50); rng = np.random.default_rng(1000)
                flat = flat0.copy(); curve = []
                for it in range(a.steps + 1):
                    S = np.concatenate([vm.sample(flat) for _ in range(a.sweeps_per_step)]); ni, ni_e, l_s, vt_s = LL.ninv(model, flat, flat0, S, w)
                    if it == 0: r_tr, r_ho = tr.r0, ho.r0
                    else: r_tr = tr.ratios(model.logpsi(flat, tr.X)); r_ho = ho.ratios(model.logpsi(flat, ho.X))
                    qt, qte = tr.Q(r_tr); qh, qhe = ho.Q(r_ho)
                    Rh = qh * qh * ni; Rhe = Rh * np.sqrt((2 * qhe / qh) ** 2 + (ni_e / ni) ** 2); Rt = qt * qt * ni
                    rec = dict(it=it, R_held=Rh, R_held_err=Rhe, R_train=Rt)
                    if it < a.steps:
                        dl, info = LL.sr_step(model, flat, S, vt_s, tr, r_tr, w, lr, shift, a.maxn, rng); flat = flat + dl; rec.update(info)
                    curve.append(rec)
                bi = 1 + int(np.argmax([r['R_held'] for r in curve[1:]]))
                zb = (curve[bi]['R_held'] - 1) / curve[bi]['R_held_err']
                out['b'].append(dict(train=f'c{2 - hc}', held=f'c{hc + 1}', lr=lr, shift=shift, curve=curve, best_step=bi, best_R=curve[bi]['R_held'],
                                     best_err=curve[bi]['R_held_err'], best_z=zb))
                P(f'(b) train c{2 - hc}/held c{hc + 1} lr {lr} shift {shift}: R_held ' + ' '.join(f'{r["R_held"]:.5f}({r["R_held_err"]:.1e})' for r in curve) +
                  f' | R_train ' + ' '.join(f'{r["R_train"]:.5f}' for r in curve) + f' | |d|_S ' + ' '.join(f'{r.get("dnorm", 0):.3f}' for r in curve[:-1]) +
                  f' | best step {bi} z {zb:+.1f} ({time.time() - t:.0f}s)')
                json.dump(out, open(os.path.join(a.out, 'diag.json'), 'w'), indent=1, default=float)
P(f'DIAG_DONE {time.time() - T0:.0f}s')
