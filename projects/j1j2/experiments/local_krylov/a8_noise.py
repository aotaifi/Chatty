"""Check (a8) diagnostic (post hoc, labelled as such): noise scales of the two rules at the a^2 samples.
sigma_tot = sd of E_L(x); sigma_W = sd of one window's excess dE_W(x) (w = 2, 3, 4; pooled over the 64 translates);
mean correlation of disjoint windows; distribution of max_W dE_W on 'clean' samples (total excess below its median),
i.e. the look-elsewhere floor the local rule must clear.  Bases V and M.
  python a8_noise.py OUT.json RUN_DIR [RUN_DIR ...]
"""
import sys, os, glob, json
from multiprocessing import Pool
import numpy as np
import a8_lib as A


def one(path):
    C = A.chunk_states(path); out = {}
    for b in ('V', 'M'):
        e = A.bond_energies(C, b)[C['px']]
        dg = np.where(C['V1'][C['px']], -0.25, 0.25) * A.JB[None, :]      # diagonal part of e_b
        out[f'r_{b}'] = e.sum(1); out[f'rd_{b}'] = dg.sum(1)
        for w in (2, 3, 4):
            out[f'S{w}_{b}'] = e @ A.WB[w].T; out[f'Sd{w}_{b}'] = dg @ A.WB[w].T
    return out


if __name__ == '__main__':
    OUT = sys.argv[1]; files = sum([sorted(glob.glob(os.path.join(r, 'chunk_*.npz'))) for r in sys.argv[2:]], [])
    with Pool(int(os.environ.get('NPROC', 6))) as P: parts = P.map(one, files)
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    R = {}
    for b in ('V', 'M'):
        r = D[f'r_{b}']; rd = D[f'rd_{b}']; top = r >= np.quantile(r, 0.99)
        R[b] = dict(sigma_tot=float(r.std()), mean_tot=float(r.mean()), q99_minus_mean=float(np.quantile(r, 0.99) - r.mean()),
                    sigma_tot_diag=float(rd.std()), sigma_tot_offdiag=float((r - rd).std()),
                    top1pct_excess_diag=float(rd[top].mean() - rd.mean()), top1pct_excess_offdiag=float((r - rd)[top].mean() - (r - rd).mean()))
        clean = r <= np.median(r)
        for w in (2, 3, 4):
            S = D[f'S{w}_{b}']; dS = S - S.mean()          # every translate has the same mean by symmetry
            c = np.corrcoef(dS[:, [0, 4 + 8 * 4]].T)[0, 1]  # windows at (0,0) and (4,4): disjoint for w <= 4
            mx = dS.max(1)
            R[b][f'w{w}'] = dict(sigma_W=float(dS.std()), corr_disjoint=float(c),
                                 max_clean_median=float(np.median(mx[clean])), max_clean_q99=float(np.quantile(mx[clean], 0.99)),
                                 sigma_sum_of_windows_over_tot=float(dS.sum(1).std() / r.std()))
            Sd = D[f'Sd{w}_{b}']; So = S - Sd
            R[b][f'w{w}'].update(sigma_W_diag=float(Sd.std()), sigma_W_offdiag=float(So.std()),
                                 corr_diag_offdiag=float(np.corrcoef(Sd[:, 0], So[:, 0])[0, 1]))
    json.dump(R, open(OUT, 'w'), indent=1)
    print(json.dumps(R, indent=1))
