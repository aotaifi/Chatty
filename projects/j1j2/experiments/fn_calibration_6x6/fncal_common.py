"""Constants and the window estimator shared by fncal_dmc.py (GPU) and fncal_analyze.py (numpy only)."""
import numpy as np

N = 36
EXACT = {   # exact sector FN energy per site (results/stall_6x6/base_validation.json, st6_run.py base)
    'vitvit': -0.503661717200079, 'vitex': -0.5036622422881156, 'exvit': -0.5037157035266815}
E0_SITE = -0.5038096538908782
GUIDES = {'vitvit': 'ViT amp + ViT sign', 'vitex': 'ViT amp + exact sign', 'exvit': 'exact amp + ViT sign'}


def window_estimates(Eref, tau, burn, beta_target):
    """per population: mean of E_{t+1} over steps with beta_after >= burn and beta_before < beta_target (ll6_fn rule).
    Eref, tau (P, S+1) with tau_t = 0 for inactive steps; Eref[t+1] is the post-step population energy."""
    P = Eref.shape[0]
    b_after = np.cumsum(tau, 1); b_before = b_after - tau
    act = (tau > 0) & (b_before < beta_target)
    use = act & (b_after >= burn)
    Epost = np.concatenate([Eref[:, 1:], Eref[:, -1:]], 1)
    n = use.sum(1)
    return np.where(n > 0, (Epost * use).sum(1) / np.maximum(n, 1), np.nan), n


