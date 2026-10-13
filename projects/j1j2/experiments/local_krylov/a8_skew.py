"""F1 secondary (post hoc, labelled): per-site energy cumulants kappa2 = <(H-E)^2>/N and kappa3 = <(H-E)^3>/N of the 8x8
guides (ViT sign and Marshall, amplitude |psi_P|) from the signs_8x8 two-hop chunks, to place them on the Lanczos
dilution rule N_1/2(L) = c kappa3^2 / kappa2^3 (a0f).  <(H-E)^3> = E_{a^2}[(E_L - E) (H^2 psi/psi - 2 E E_L + E^2)],
with (H^2 psi)(x)/psi(x) = D(x) E_L(x) + sum_y c_xy E_L(y) exact from the two-hop data.  SE from per-chain means
(the third moment is heavy-tailed; also reported with the top 0.1% of |E_L - E| removed as a sensitivity check).
  python a8_skew.py OUT.json RUN_DIR [RUN_DIR ...]
"""
import sys, os, glob, json
from multiprocessing import Pool
import numpy as np
import a8_lib as A


def one(path):
    C = A.chunk_states(path); out = dict(chain=C['chain'])
    starts = np.searchsorted(C['own'], np.arange(len(C['S'])))
    for b in ('V', 'M'):
        r = A.bond_energies(C, b).sum(1); c = A.edge_coeffs(C, b)
        Dx = Q_diag(C)
        out[f'EL_{b}'] = r[C['px']]
        out[f'H2_{b}'] = Dx * r[C['px']] + np.add.reduceat(c * r[C['py']], starts)
    return out


def Q_diag(C):
    V = C['V1'][C['px']]
    return A.Q.diag_vec(V)


if __name__ == '__main__':
    OUT = sys.argv[1]; files = sum([sorted(glob.glob(os.path.join(r, 'chunk_*.npz'))) for r in sys.argv[2:]], [])
    with Pool(int(os.environ.get('NPROC', 4))) as P: parts = P.map(one, files)
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    # chain ids: chunks are in run order; make global chain id from file index ranges
    gid = np.concatenate([p['chain'] + 100000 * (0 if 's8RA' in f else 1 if 's8RB1' in f else 2) for p, f in zip(parts, files)])
    uc, inv = np.unique(gid, return_inverse=True); cn = np.bincount(inv)
    N = 64; R = {}
    for b in ('V', 'M'):
        EL = D[f'EL_{b}']; H2 = D[f'H2_{b}']; E = EL.mean()
        m2s = (EL - E) ** 2; m3s = (EL - E) * (H2 - 2 * E * EL + E ** 2)
        def cm(v):
            m = np.bincount(inv, weights=v) / cn
            return float(v.mean()), float(m.std(ddof=1) / np.sqrt(len(uc)))
        k2, k2e = cm(m2s); k3, k3e = cm(m3s)
        keep = np.abs(EL - E) <= np.quantile(np.abs(EL - E), 0.999)
        k3t = float(m3s[keep].mean()); k2t = float(m2s[keep].mean())
        m3cl = float(((EL - E) ** 3).mean())
        R[b] = dict(E_site=float(E / N), m2=k2, m2_se=k2e, m3=k3, m3_se=k3e, kappa2=k2 / N, kappa3=k3 / N,
                    m3_trim999=k3t, m2_trim999=k2t, m3_classical_EL=m3cl,
                    skew_total=k3 / k2 ** 1.5, ratio_k3sq_over_k2cube=(k3 / N) ** 2 / (k2 / N) ** 3)
    json.dump(R, open(OUT, 'w'), indent=1)
    print(json.dumps(R, indent=1))
