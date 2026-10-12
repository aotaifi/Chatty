"""Thresholds, energies and disagreement (CPU) from s8_post.py ref outputs.
  python s8_energy.py tm  qA.npz OUT.json                     -> T_M (energy-optimal Marshall step on RA) + half splits
  python s8_energy.py eval qA.npz OUT.json T1 T_M qB1.npz qB2.npz ...
      T2 for each net base (nV, nM) energy-optimal on RA; energies of all sign variants on RB (paired, per-chain SE);
      the same on RA (secondary, thresholds selected there).
Variants: V (ViT sign), K1V = V sgn(T1 - r_V), nV, nV+hop; M, K1M = M sgn(T_M - r_M), nM, nM+hop.
"""
import sys, json
import numpy as np
import s8_q as Q

N = Q.N


def load(paths):
    qs = []
    for k, p in enumerate(paths):
        q = dict(np.load(p)); q['chain'] = q['chain'] + 100000 * k
        for z in ('net_sec', 'net_states'): q.pop(z, None)
        qs.append(q)
    return Q.concat_q(qs)


def half_T(q, b, nsplit=8, seed=0):
    rg = np.random.default_rng(seed); uc = np.unique(q['chain']); out = []
    for _ in range(nsplit):
        sel = rg.choice(uc, len(uc) // 2, replace=False); m = np.isin(q['chain'], sel)
        out.append((Q.opt_T(q, b, m)[0], Q.opt_T(q, b, ~m)[0]))
    return out


def variants(q, Ts):
    E = {'V': Q.eloc(q, 'V'), 'K1V': Q.eloc(q, 'V', Ts['T1']), 'M': Q.eloc(q, 'M'), 'K1M': Q.eloc(q, 'M', Ts['TM'])}
    sx = {'V': q['s_V'], 'K1V': q['s_V'] * np.where(q['r_V'] <= Ts['T1'], 1, -1),
          'M': q['s_M'], 'K1M': q['s_M'] * np.where(q['r_M'] <= Ts['TM'], 1, -1)}
    for b in ('nV', 'nM'):
        if f'r_{b}' in q:
            E[b] = Q.eloc(q, b); E[b + '+hop'] = Q.eloc(q, b, Ts['T2' + b])
            sx[b] = q[f's_{b}']; sx[b + '+hop'] = q[f's_{b}'] * np.where(q[f'r_{b}'] <= Ts['T2' + b], 1, -1)
    return E, sx


def table(q, Ts):
    E, sx = variants(q, Ts); ch = q['chain']; out = dict(n=int(len(ch)), nchains=int(len(np.unique(ch))))
    out['E_site'] = {k: Q.chain_mean_se(v / N, ch) for k, v in E.items()}
    out['vs_V'] = {k: Q.chain_mean_se((v - E['V']) / N, ch) for k, v in E.items() if k != 'V'}
    out['vs_K1V'] = {k: Q.chain_mean_se((E[k] - E['K1V']) / N, ch) for k in ('nV', 'nV+hop') if k in E}
    out['vs_K1M'] = {k: Q.chain_mean_se((E[k] - E['K1M']) / N, ch) for k in ('nM', 'nM+hop') if k in E}
    dis = lambda a, b: float(np.mean(a != b))
    out['disagree_x'] = {f'{a}|{b}': dis(sx[a], sx[b]) for a, b in
                         (('nV', 'K1V'), ('nV+hop', 'K1V'), ('K1V', 'V'), ('nV', 'V'), ('nM', 'K1M'), ('nM+hop', 'K1M'),
                          ('K1M', 'V'), ('nM+hop', 'V'), ('M', 'V')) if a in sx and b in sx}
    return out


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'tm':
        qa = load([sys.argv[2]]); T, g = Q.opt_T(qa, 'M')
        res = dict(T_M=T, gain_in_sample=g, E_M=Q.chain_mean_se(Q.eloc(qa, 'M') / N, qa['chain']),
                   E_V=Q.chain_mean_se(Q.eloc(qa, 'V') / N, qa['chain']), half=half_T(qa, 'M'),
                   E_K1M_insample=Q.chain_mean_se(Q.eloc(qa, 'M', T) / N, qa['chain']))
        Q.dump(sys.argv[3], res); print(json.dumps(res, indent=1, default=float))
    else:
        qa = load([sys.argv[2]]); out = sys.argv[3]; T1 = float(sys.argv[4]); TM = float(sys.argv[5])
        qb = load(sys.argv[6:])
        Ts = dict(T1=T1, TM=TM)
        res = dict(T=Ts, T_half={})
        for b in ('nV', 'nM'):
            if f'r_{b}' in qa:
                Ts['T2' + b], g = Q.opt_T(qa, b); res['T_half'][b] = half_T(qa, b)
        res['RB'] = table(qb, Ts); res['RA_secondary'] = table(qa, Ts)
        # K1V threshold re-chosen on RA (diagnostic for T1 transfer)
        res['T1_on_RA'] = Q.opt_T(qa, 'V')
        Q.dump(out, res); print(json.dumps(res, indent=1, default=float))
