# F2 control 2 (exec'd inside cx_f2fit.py when spec lv_check is true; no fit).  Brute-force twin of L_v written independently of the code path:
# plain numpy/python bit logic, the bond list from st6_sector, the 288 space-group images + spin flip for the orbit representative, the
# violating flag s_2(x) s_2(y) > 0 from s2.npy, w_xy = n_x (JB/2) phi_x phi_y, d = (A_y - A_x) - (log phi_y - log phi_x).  Only the network values A
# come from the code's fwd (the quantity being tested is the bond selection / flag / weights / sampling, not the network).
tchk = time.time()
perms = SS.space_group(); NN_, NNN_ = SS.NN, SS.NNN
bl = list(NN_) + list(NNN_); Jb = [1.0] * len(NN_) + [SS.J2] * len(NNN_)
reps_h = np.asarray(sec.reps); s2_h = np.asarray(s2); ls2_h = np.asarray(ls2); n_h_ = np.asarray(sec.n)
POW = (np.int64(1) << np.arange(36, dtype=np.int64)); FULL_ = (1 << 36) - 1
AR288 = np.arange(perms.shape[0])[:, None]


def canon_bf(c):
    """index of the orbit representative of the 36-bit config c (python int): min over the 288 images and their spin flips."""
    b = np.array([(c >> i) & 1 for i in range(36)], np.int64)
    img = np.zeros((perms.shape[0], 36), np.int64); img[AR288, perms] = b[None, :]
    v = img @ POW; m = int(min(v.min(), (FULL_ - v).min()))
    i = int(np.searchsorted(reps_h, np.uint64(m)))
    assert reps_h[i] == np.uint64(m), 'brute-force orbit representative is not in the table'
    return i


def bonds_bf(ix, flip=False):
    """all violating (flip: mislabelled = kept) bonds of the orbit rep ix, by plain python: list of (J, iy, y_config)."""
    x = int(reps_h[ix]); out = []
    for (i, j), J in zip(bl, Jb):
        if ((x >> i) & 1) == ((x >> j) & 1): continue               # parallel: H_xy = 0
        y = x ^ (1 << i) ^ (1 << j); iy = canon_bf(y)
        prod = float(s2_h[ix]) * float(s2_h[iy])
        if (prod < 0) if flip else (prod > 0): out.append((J, iy, y))
    return out


lvc = dict(note='brute force vs code; rel tolerance 1e-6')
# ---- A: the training sampler's own bonds (a fresh sampler), 4000 bonds
VSA = VSampler(Wx, ~ho_h, 4242); vbA = VSA.draw(4000)
ix_ = np.asarray(vbA['ix']); iy_ = np.asarray(vbA['iy']); x_ = np.asarray(vbA['x']); y_ = np.asarray(vbA['y'])
bad_flag = 0; bad_iy = 0; bad_x = 0; bad_bond = 0
for k in range(ix_.size):
    xk = int(x_[k]); yk = int(y_[k]); dif = xk ^ yk
    if xk != int(reps_h[ix_[k]]): bad_x += 1
    sites = [i for i in range(36) if (dif >> i) & 1]
    if len(sites) != 2 or tuple(sites) not in set(tuple(sorted(b)) for b in bl) or ((xk >> sites[0]) & 1) == ((xk >> sites[1]) & 1): bad_bond += 1
    j = canon_bf(yk)
    if j != int(iy_[k]): bad_iy += 1
    if not (float(s2_h[ix_[k]]) * float(s2_h[j]) > 0): bad_flag += 1
uq = np.unique(np.concatenate([ix_, iy_])); An = np.asarray(fwd(theta, sec.reps[jnp.asarray(uq)])); amap = dict(zip(uq.tolist(), An.tolist()))
dA = np.array([(amap[int(b)] - amap[int(a)]) - (ls2_h[b] - ls2_h[a]) for a, b in zip(ix_, iy_)])
Lv_bf = float(np.mean(dA ** 2)); Lv_code, _ = lv_value(theta, vbA)
lvc['A_sampled'] = dict(n_bonds=int(ix_.size), mismatch_x=bad_x, not_a_bond=bad_bond, mismatch_canon_y=bad_iy, not_violating=bad_flag, L_v_bruteforce=Lv_bf, L_v_code=Lv_code,
                        rel=abs(Lv_code / Lv_bf - 1))
progress(f'[lv_check A] 4000 sampled bonds: not violating {bad_flag}, canon mismatches {bad_iy}, not a bond {bad_bond}; L_v brute {Lv_bf:.9e} code {Lv_code:.9e} rel {abs(Lv_code / Lv_bf - 1):.1e}')
# ---- B: the README formula by full enumeration over LV_CHECK_REPS random orbits, true and mislabelled flag
rr_ = np.random.default_rng(777); ridx = np.sort(rr_.choice(np.nonzero(Wx > 0)[0], size=LV_CHECK_REPS, replace=False))
res_B = {}
for flip in (False, True):
    bl_all = {int(i): bonds_bf(int(i), flip) for i in ridx}
    uq = np.unique(np.concatenate([ridx] + [np.array([b[1] for b in v], np.int64) for v in bl_all.values() if v])); An = np.asarray(fwd(theta, sec.reps[jnp.asarray(uq)])); amap = dict(zip(uq.tolist(), An.tolist()))
    num = den = 0.0; nbd = 0
    for i, v in bl_all.items():
        for J, j, _ in v:
            w = n_h_[i] * 0.5 * J * math.exp(ls2_h[i] + ls2_h[j]); d = (amap[j] - amap[i]) - (ls2_h[j] - ls2_h[i])
            num += w * d * d; den += w; nbd += 1
    bf = num / den; code, nb_code = lv_exact(theta, ridx, flip)
    res_B['mislabelled' if flip else 'true'] = dict(n_bonds_bruteforce=nbd, n_bonds_code=nb_code, L_v_bruteforce=bf, L_v_code=code, rel=abs(code / bf - 1))
    progress(f'[lv_check B {"MISLABELLED" if flip else "true"}] {LV_CHECK_REPS} orbits, {nbd} bonds (code {nb_code}): L_v brute {bf:.9e} code {code:.9e} rel {abs(code / bf - 1):.1e}')
res_B['mislabelled_vs_true_rel_diff'] = abs(res_B['mislabelled']['L_v_code'] / res_B['true']['L_v_code'] - 1)
lvc['B_enumeration'] = res_B
# ---- C: the sampler draws x with the right law: fractions per decade of x vs W_x sums (200k bonds)
vbC = VSampler(Wx, ~ho_h, 4343).draw(200000); dC = dec_h[np.asarray(vbC['ix'])]
Wd = {d: float(Wx[(~ho_h) & (dec_h == d)].sum()) for d in DECS}; tot = sum(Wd.values())
frac = {d: float(np.mean(dC == d)) for d in DECS}
lvc['C_decade_fractions'] = {str(d): dict(sampled=frac[d], exact=Wd[d] / tot, z=(frac[d] - Wd[d] / tot) / math.sqrt(max(Wd[d] / tot * (1 - Wd[d] / tot), 1e-30) / 200000)) for d in DECS}
progress('[lv_check C] decade fractions sampled/exact: ' + ' '.join(f'{d}:{frac[d]:.4f}/{Wd[d] / tot:.4f}' for d in DECS))
# ---- D (F2-1): the fixed-set assembly.  D1 deduplicated configs (orbit reps of y) vs raw y; D2 chunked normal equations vs the dense rows of the first F2 run; D3 swapped bond ends must be caught
vbD = VSampler(Wx, ~ho_h, 4545).draw(2048); FXD = make_fixed(vbD); MD = FXD['M']
S_u = fwd(theta, FXD['cfg']); Lv_fx = lv_fixed(S_u, FXD); Lv_raw, _ = lv_value(theta, vbD)
Xd = jnp.concatenate([vbD['x'], vbD['y']]); Sd = fwd(theta, Xd); Gd = jac(theta, Xd); sqd = math.sqrt(LAMV / MD)
Jd = ((Gd[MD:] - Gd[:MD]) * sqd).astype(f64); rd_ = sqd * v_resid(Sd, vbD).astype(f64)
Sm_d = Jd.T @ Jd; g_d = Jd.T @ rd_; r2_d = float(jnp.sum(rd_ * rd_)); del Gd, Jd
Sm_f, g_f, r2_f = gn_fixed(theta, FXD, S_u, LAMV); Sm_s, g_s, r2_s = gn_fixed(theta, FXD, S_u, LAMV, swap=True)
rel = lambda a, b: float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
lvc['D_fixed_set'] = dict(n_bonds=MD, n_unique_configs=FXD['nU'], rel_L_v_fixed_vs_raw=abs(Lv_fx / Lv_raw - 1), rel_Sm=rel(Sm_f, Sm_d), rel_g=rel(g_f, g_d), rel_r2=float(abs(r2_f / r2_d - 1)),
                          broken_swap_rel_g=rel(g_s, g_d), broken_swap_rel_Sm=rel(Sm_s, Sm_d))
progress('[lv_check D] fixed set: ' + json.dumps(lvc['D_fixed_set']))
lvc['sec'] = time.time() - tchk
res['lv_check'] = lvc
res['lv_check_pass'] = bool(lvc['A_sampled']['not_violating'] == 0 and lvc['A_sampled']['mismatch_canon_y'] == 0 and lvc['A_sampled']['not_a_bond'] == 0 and lvc['A_sampled']['rel'] < 1e-6
                            and res_B['true']['rel'] < 1e-6 and res_B['true']['n_bonds_bruteforce'] == res_B['true']['n_bonds_code'] and res_B['mislabelled']['rel'] < 1e-6
                            and res_B['mislabelled_vs_true_rel_diff'] > 1e-3 and max(abs(v['z']) for v in lvc['C_decade_fractions'].values()) < 5
                            and lvc['D_fixed_set']['rel_L_v_fixed_vs_raw'] < 1e-8 and lvc['D_fixed_set']['rel_Sm'] < 1e-8 and lvc['D_fixed_set']['rel_g'] < 1e-8 and lvc['D_fixed_set']['broken_swap_rel_g'] > 0.5)
res['sec'] = time.time() - T00; res['done'] = True; dump(F, res); progress(f'[lv_check] pass {res["lv_check_pass"]}  {lvc["sec"]:.0f}s')
