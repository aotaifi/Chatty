"""analysis.py (paired_fn): blocking, long-run variances and covariances, kappa_work (ASTRA_paired.md eqs 11-13), quotability gate.

S_g   = long-run variance of arm g's block series (sum of all lag covariances) = n * (plateau binning error)^2
S_D   = same for the aligned difference series Delta_i = E_B,i - E_A,i
S_AB  = (S_A + S_B - S_D) / 2  (eq 11, includes all lags),  rho = S_AB / sqrt(S_A S_B)  (long-run correlation)
C_pair = c_pair S_D,  C_ind = (c_A + c_B)(S_A^sa + S_B^sa) with standalone-measured S^sa and costs c (sec per block),
C_ind_opt = (sqrt(c_A S_A^sa) + sqrt(c_B S_B^sa))^2,  kappa_work = C_pair / C_ind.
"""
import subprocess
import numpy as np

QUOTABLE = '/project/theorie/a/A.Otaifi/tools/quotable.py'


def binning(x, min_bins=32):
    x = np.asarray(x, float); errs = []; lv = x.copy()
    while len(lv) >= min_bins:
        errs.append(float(lv.std(ddof=1) / np.sqrt(len(lv)))); k = len(lv) // 2; lv = 0.5 * (lv[:2 * k:2] + lv[1:2 * k:2])
    return errs


def mean_err(x, min_bins=32):
    """mean and binning error: the maximum over the last three levels with >= min_bins bins (conservative plateau)."""
    x = np.asarray(x, float); e = binning(x, min_bins)
    if not e: return float(x.mean()), float('nan'), e
    return float(x.mean()), float(max(e[-3:])), e


def long_run_var(x, min_bins=32):
    m, e, _ = mean_err(x, min_bins); return len(x) * e * e


def pair_stats(eA, eB, cost_pair=None, saA=None, saB=None, cost_A=None, cost_B=None, NS=1):
    """eA, eB: aligned measured block series of one ensemble (total energy).  saA/saB: standalone series (for S^sa)."""
    eA = np.asarray(eA, float); eB = np.asarray(eB, float); d = eB - eA
    SA = long_run_var(eA); SB = long_run_var(eB); SD = long_run_var(d)
    SAB = 0.5 * (SA + SB - SD); rho = SAB / np.sqrt(SA * SB)
    out = dict(n=len(d), delta_site=float(d.mean() / NS), sigma_site=float(np.sqrt(SD / len(d)) / NS), S_A=SA, S_B=SB, S_D=SD, S_AB=SAB, rho=float(rho),
               pearson_same_block=float(np.corrcoef(eA, eB)[0, 1]))
    if cost_pair is not None and saA is not None and saB is not None:
        SAs = long_run_var(saA); SBs = long_run_var(saB)
        Cp = cost_pair * SD; Ci = (cost_A + cost_B) * (SAs + SBs); Cio = (np.sqrt(cost_A * SAs) + np.sqrt(cost_B * SBs)) ** 2
        out.update(S_A_sa=SAs, S_B_sa=SBs, c_pair=cost_pair, c_A=cost_A, c_B=cost_B, C_pair=Cp, C_ind=Ci, C_ind_opt=Cio,
                   kappa_work=float(Cp / Ci), kappa_work_opt=float(Cp / Cio), r_cost=float(cost_pair / (cost_A + cost_B)))
    return out


def fifths(x):
    x = np.asarray(x, float); n = len(x) // 5
    return [float(x[i * n:(i + 1) * n].mean()) for i in range(5)]


def quotable(files, name, plot=None):
    cmd = ['python3', QUOTABLE, 'series', *files, '--name', name] + (['--plot', plot] if plot else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    lines = (p.stdout + p.stderr).strip().splitlines()
    return dict(exit=p.returncode, verdict=lines[0] if lines else '', text='\n'.join(lines))


def combine(means, errs):
    """inverse-variance weighted mean of independent ensembles and the larger of its error and the ensemble scatter."""
    m = np.asarray(means, float); e = np.asarray(errs, float); w = 1 / e ** 2
    mu = float((w * m).sum() / w.sum()); se = float(1 / np.sqrt(w.sum()))
    scat = float(m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 1 else float('nan')
    chi2 = float((((m - mu) / e) ** 2).sum())
    return dict(mean=mu, err_w=se, err_scatter=scat, err=max(se, scat if np.isfinite(scat) else 0), chi2=chi2, dof=len(m) - 1)
