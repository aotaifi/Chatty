"""pool8_wb.py: partner-weight statistics at 8x8 WITHOUT labels.  W^B_x = sum over blocked bonds b of |H_b| |B(y_b)/B(x)|  (|H_b| = J_b / 2), blocked = antiparallel bond whose Marshall-stub
sign s(x) = (-1)^{#up on sublattice A (x + y even)} satisfies s(x) s(y) > 0 (= the J2 bonds; J1 swaps flip s).  This is pool_fit.build_pool's W^B_x with the 6x6 s_G replaced by the Marshall stub;
rho = W^B_x exp[(lnT_y - lnB_y) - (lnT_x - lnB_x)] needs the target T and is NOT computed here.
  python pool8_wb.py extract OUT.npz N_PER_RUNG FILE.npz ...   (one_swap_ratios on the stored thinned configs of pool8 runs; PP2 at PROD8)
  python pool8_wb.py stats OUT.json MED2 BETAS EXTRACT.npz ...  (per decade: pool_fit.pool_weights' w_in, Wbar_d, max W/Wbar_d, E_w[(W/Wbar)^2], ESS fraction 1/E_w[(W/Wbar)^2])"""
import sys, os, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pool8 as P8


def marshall_blocked(lat, cfg):
    L = lat.L; xs = np.arange(lat.N) % L; ys = np.arange(lat.N) // L; A = ((xs + ys) % 2 == 0)
    sx = (-1.0) ** ((cfg[:, A] > 0).sum(1))
    # y = swap on bond b: sublattice-A up count changes by (sigma_j - sigma_i)/2 * (A_i - A_j) ... compute directly
    sy = np.empty((len(cfg), lat.NB))
    for b in range(lat.NB):
        i, j = lat.BI[b], lat.BJ[b]; d = (cfg[:, j] - cfg[:, i]) / 2.0 * (float(A[i]) - float(A[j]))    # up-count on A changes by d (only if antiparallel)
        sy[:, b] = (-1.0) ** (((cfg[:, A] > 0).sum(1) + d).astype(np.int64))
    anti = (cfg[:, lat.BI] * cfg[:, lat.BJ]) < 0
    return anti & (sx[:, None] * sy > 0)


def extract(out, nper, files):
    ev = P8.load_eval(8, 'pp2'); rng = np.random.default_rng(5); lat = ev.lat
    C, LB, RUNG = [], [], []
    for f in files:
        z = np.load(f); cfg = z['cfg']; lb = z['lnb_thin']; T, R, E, N = cfg.shape; betas = z['betas']
        for q in range(R):
            c = cfg[:, q].reshape(-1, N); l = lb[:, q].reshape(-1); k = rng.permutation(len(c))[:nper]; C.append(c[k]); LB.append(l[k]); RUNG.append(np.full(len(k), betas[q]))
    C = np.concatenate(C).astype(np.float64); LB = np.concatenate(LB); RUNG = np.concatenate(RUNG); W = np.zeros(len(C)); lbn = np.zeros(len(C)); nblk = np.zeros(len(C), int)
    for i in range(0, len(C), 32):
        lb_, r = ev.ratios(C[i:i + 32]); blk = marshall_blocked(lat, C[i:i + 32]); a = np.abs(r)
        W[i:i + 32] = (np.where(blk, lat.HW[None, :] * a, 0.0)).sum(1); nblk[i:i + 32] = blk.sum(1); lbn[i:i + 32] = np.real(lb_)
    np.savez(out, W=W, lnB=lbn, lnB_stored=LB, beta=RUNG, nblk=nblk, cfg=C.astype(np.int8)); print('extracted', len(C), 'max |lnB - stored|', np.abs(lbn - LB).max(), 'mean nblk', nblk.mean())


def stats(out, med2, files):
    import pool_fit as PF
    Z = [np.load(f) for f in files]; W = np.concatenate([z['W'] for z in Z]); lB = np.concatenate([z['lnB'] for z in Z]); be = np.concatenate([z['beta'] for z in Z])
    betas = sorted(set(be.tolist()), reverse=True); chain = np.array([betas.index(b) for b in be], np.int8); decs = list(range(2, -7, -1))
    # pool_weights takes the beta = 1 median from the draws; here the measured one is injected by shifting lnB so that the median matches
    q1 = betas.index(1.0); shift = med2 - 2.0 * np.median(lB[chain == q1]) * 1.0; assert abs(shift) < 0.5, shift
    Pw = PF.pool_weights(lB, chain, betas, decs, W[:, None], np.zeros(len(W), np.int64), ho_fn=lambda o: np.zeros(len(o), bool))
    res = dict(n=int(len(W)), betas=betas, med2lnB_sampled=float(Pw['med2lnB']), decades={})
    for g, d in enumerate(decs):
        m = Pw['dec'] == d; w = Pw['w_in'][m]; Wb = Pw['Wbar'][g]
        if not m.any() or Wb <= 0: continue
        r = W[m] / Wb; e2 = float(np.sum(w * r * r)); res['decades'][str(d)] = dict(n=int(m.sum()), Wbar=float(Wb), max_over_Wbar=float(r.max()), mean_unweighted_over_Wbar=float(r.mean()), Ew_r2=e2,
                                                                                    ess_fraction=float(1.0 / e2), p99_over_Wbar=float(np.percentile(r, 99)), ess_pool_cell=float(1.0 / np.sum(w * w)))
    json.dump(res, open(out, 'w'), indent=1); [print(d, {k: round(v, 3) if isinstance(v, float) else v for k, v in x.items()}) for d, x in res['decades'].items()]


if __name__ == '__main__':
    if sys.argv[1] == 'extract': extract(sys.argv[2], int(sys.argv[3]), sys.argv[4:])
    else: stats(sys.argv[2], float(sys.argv[3]), sys.argv[5:])
