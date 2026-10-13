"""Check (a8), analysis of a8_run.py output: cross-fitted (2-fold by chain, 16 splits) gains of one sign-only step,
global G vs local C_w, bases V (ViT sign) and M (Marshall); chain bootstrap of the whole procedure; leave-one-batch-out;
diagnostics.  Criteria: results/local_krylov/PREREG.md.
  python a8_analyze.py a8_delta.npz a8_run_meta.json OUT.json
"""
import sys, json
import numpy as np

N = 64; NSPLIT = 16; NBOOT = 200
D = dict(np.load(sys.argv[1])); META = json.load(open(sys.argv[2])); OUT = sys.argv[3]
NNB = {2: (4, 2), 3: (12, 8), 4: (24, 18)}
uc, inv = np.unique(D['gchain'], return_inverse=True); nch = len(uc)
cn = np.bincount(inv).astype(float)
run_of_chain = np.zeros(nch, int); run_of_chain[inv] = D['run']


def chain_sums(d):
    out = np.zeros((nch, d.shape[1])); np.add.at(out, inv, d); return out


def chain_mean_se(v):
    m = np.bincount(inv, weights=v) / cn
    return float(v.mean()), float(m.std(ddof=1) / np.sqrt(nch))


RULESETS = {'G': ['G'], 'C2': ['C2'], 'C3': ['C3'], 'C4': ['C4'], 'Csel': ['C3', 'C4']}


def analyse(b):
    d = {r: D[f'd_{b}_{r}'] for r in ('G', 'C2', 'C3', 'C4')}
    CS = {r: chain_sums(v) for r, v in d.items()}

    def select(wch, rs):
        best = None
        for r in rs:
            e = (wch @ CS[r]) / (wch @ cn)
            j = int(np.argmin(e))
            if best is None or e[j] < best[2]: best = (r, j, e[j])
        return best

    res = {}
    rng = np.random.default_rng(0)
    splits = [rng.random(nch) < 0.5 for _ in range(NSPLIT)]
    oof = {}
    for name, rs in RULESETS.items():
        acc = np.zeros(len(inv)); sel = []
        for A in splits:
            for fold, other in ((A, ~A), (~A, A)):
                r, j, _ = select(other.astype(float), rs)
                m = fold[inv]; acc[m] += d[r][m, j]; sel.append((r, j))
        oof[name] = acc / NSPLIT
        g, se = chain_mean_se(oof[name] / N)
        r_all, j_all, e_all = select(np.ones(nch), rs)
        grid = D[f'grid_{r_all}_{b}']
        thr = [float(D[f'grid_{r}_{b}'][j]) for r, j in sel]
        res[name] = dict(xfit_gain=g, se_chain=se, insample_gain=float(e_all / N), insample_rule=r_all,
                         insample_col=j_all, insample_threshold=float(grid[j_all]),
                         xfit_thresholds_min_max=[min(thr), max(thr)],
                         xfit_rules=sorted(set(r for r, _ in sel)) if name == 'Csel' else None,
                         w_choice_counts={r: sum(1 for rr, _ in sel if rr == r) for r in rs})
    # paired C - G
    for name in ('C2', 'C3', 'C4', 'Csel'):
        g, se = chain_mean_se((oof[name] - oof['G']) / N); res[name]['paired_minus_G'] = g; res[name]['paired_se_chain'] = se
    # chain bootstrap of the whole cross-fit procedure
    rb = np.random.default_rng(1); boot = {k: [] for k in RULESETS}; bootp = {k: [] for k in ('C2', 'C3', 'C4', 'Csel')}
    for _ in range(NBOOT):
        wc = np.bincount(rb.integers(0, nch, nch), minlength=nch).astype(float)
        tot = {k: 0.0 for k in RULESETS}
        for _s in range(NSPLIT):
            A = rb.random(nch) < 0.5
            for fold, other in ((A, ~A), (~A, A)):
                for name, rs in RULESETS.items():
                    r, j, _ = select(wc * other, rs)
                    tot[name] += (wc * fold) @ CS[r][:, j]
        den = NSPLIT * (wc @ cn) * N
        for k in RULESETS: boot[k].append(tot[k] / den)
        for k in bootp: bootp[k].append((tot[k] - tot['G']) / den)
    for k in RULESETS:
        res[k]['se_boot'] = float(np.std(boot[k], ddof=1)); res[k]['se'] = max(res[k]['se_boot'], res[k]['se_chain'])
    for k in bootp:
        res[k]['paired_se_boot'] = float(np.std(bootp[k], ddof=1))
        res[k]['paired_se'] = max(res[k]['paired_se_boot'], res[k]['paired_se_chain'])
    # leave-one-batch-out
    for name, rs in RULESETS.items():
        acc = np.zeros(len(inv)); lob = []
        for k in range(3):
            r, j, _ = select((run_of_chain != k).astype(float), rs)
            m = D['run'] == k; acc[m] = d[r][m, j]; lob.append(dict(rule=r, threshold=float(D[f'grid_{r}_{b}'][j]),
                                                                     gain=float(acc[m].mean() / N)))
        g, se = chain_mean_se(acc / N); res[name]['lobo_gain'] = g; res[name]['lobo_se'] = se; res[name]['lobo'] = lob
    # diagnostics at the pooled (in-sample) selection
    E = chain_mean_se(D[f'rx_{b}'] / N); res['E_site'] = E
    eb = META[f'ebar_{b}']
    for name in ('G', 'C2', 'C3', 'C4'):
        j = res[name]['insample_col']; thr = D[f'grid_{name}_{b}'][j]; R = res[name]
        if name == 'G':
            fx = float(np.mean(D[f'rx_{b}'] > thr)); R['threshold_minus_E'] = float(thr - E[0] * N)
        else:
            w = int(name[1]); nc = D[f'ncomp_{b}_{name}'][:, j]; fx = float(np.mean(nc % 2 == 1))
            R['threshold_minus_ref'] = float(thr - (NNB[w][0] * eb[0] + NNB[w][1] * eb[1]))
            R['ncomp_hist_samples'] = np.bincount(nc).tolist()
        R['flip_frac_x'] = fx
        R['flip_frac_neighbours'] = float(D[f'nflipnb_{b}_{name}'][:, j].sum() / D['nnb'].sum())
    q3 = D[f'ncomp_{b}_C3'][:, res['C3']['insample_col']] % 2; q4 = D[f'ncomp_{b}_C4'][:, res['C4']['insample_col']] % 2
    either = (q3 + q4) > 0
    res['agree_C3_C4_samples'] = dict(all=float(np.mean(q3 == q4)), among_flipped=float(np.mean(q3[either] == q4[either])) if either.any() else None,
                                       n_flipped_either=int(either.sum()))
    # locality premise
    rx = D[f'rx_{b}']; sh = D[f'share4_{b}']
    loc = {}
    for p in (0.01, 0.001):
        m = rx >= np.quantile(rx, 1 - p)
        loc[str(p)] = dict(n=int(m.sum()), median_share=float(np.median(sh[m])), frac_share_gt_half=float(np.mean(sh[m] > 0.5)),
                           q25_q75=[float(np.quantile(sh[m], 0.25)), float(np.quantile(sh[m], 0.75))],
                           median_total_excess=float(np.median(D[f'tot_{b}'][m])))
    loc['uniform_reference'] = 42 / 256
    res['locality'] = loc
    return res


def verdict(R):
    out = {}
    for name in ('Csel', 'C3', 'C4'):
        m = R['M'][name]; gG = R['M']['G']['xfit_gain']
        pM = dict(ratio_ok=m['xfit_gain'] <= 1.5 * min(gG, 0.0), paired_ok=m['paired_minus_G'] < -2 * m['paired_se'],
                  own_ok=m['xfit_gain'] < -2 * m['se'])
        pM['pass'] = all(pM.values())
        v = R['V'][name]; pV = dict(own_ok=v['xfit_gain'] < -2 * v['se']); pV['pass'] = pV['own_ok']
        out[name] = dict(Marshall=pM, ViT=pV)
    return out


if __name__ == '__main__':
    R = {b: analyse(b) for b in ('V', 'M')}
    R['verdict'] = verdict(R)
    R['n_samples'] = int(len(inv)); R['n_chains'] = int(nch)
    json.dump(R, open(OUT, 'w'), indent=1, default=float)
    for b in ('V', 'M'):
        print('base', b, 'E/site', R[b]['E_site'])
        for k in ('G', 'C2', 'C3', 'C4', 'Csel'):
            r = R[b][k]
            print(f"  {k:5s} xfit {r['xfit_gain']:+.3e} (chain {r['se_chain']:.2e}, boot {r['se_boot']:.2e})  lobo {r['lobo_gain']:+.3e}({r['lobo_se']:.1e})"
                  f"  in-sample {r['insample_gain']:+.3e}  thr {r['insample_threshold']:.3f}"
                  + (f"  C-G {r['paired_minus_G']:+.3e} ({r['paired_se']:.2e})" if k != 'G' else ''))
        print('  locality', R[b]['locality'])
    print(json.dumps(R['verdict'], indent=1))
