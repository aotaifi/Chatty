"""Table for results/itfit_6x6/README.md from the run JSONs (results/itfit_6x6/runs/*.json).
  python itfit_summary.py RESULTS_DIR
"""
import json, os, sys, glob

R = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(R, 'runs', '*.json'))):
    r = json.load(open(f))
    for a in r.get('arms', []):
        comp = a['spec'].get('composite')
        for lp in a['loops']:
            if 'end' not in lp: continue
            e = lp['end']
            st = lp.get('steps', [])
            vr = [s_['val_end'] / s_['val0'] for s_ in st] if st else []
            rows.append(dict(arm=a['tag'], it=lp['it'], frac=e['frac'], hopFk=e['frac_hop_Fk'], comp=e['frac_composite'],
                             H_kry=e['H_kry_dE_site'], H_comp=e['H_composite_dE_site'],
                             EFN_next=lp.get('next_E_FN_dE_site'), EFN_comp=e.get('E_FN_composite_dE_site'),
                             start_EFN=lp['start']['E_FN_dE_site'], G=lp['start']['G_site'], lam_max=lp['start'].get('lambda_max'),
                             evals=lp.get('evals_loop'), val_ratio=vr, n_it=[s_['n_it'] for s_ in st],
                             kept=lp.get('kept_per_step'), incr=lp.get('energy_increase_events'),
                             vmc_steps=lp.get('vmc_steps'), sec=lp.get('train_sec'), comp_arm=bool(comp), file=os.path.basename(f)))
hdr = ('| arm | it | E_FN start | frac (net) | + F_k hop | composite guide | <H> net, Krylov | <H> composite | '
       'E_FN next guide | fit val-loss ratio per step | GN its | evals | train s |')
print(hdr); print('|' + '---|' * 13)
for q in rows:
    efn = q['EFN_next']
    print(f"| {q['arm']} | {q['it']} | {q['start_EFN']:.3e} | {q['frac']:.4f} | {q['hopFk']:.3f} | {q['comp']:.3f} | "
          f"{q['H_kry']:.3e} | {q['H_comp']:.3e} | {efn:.3e} | {', '.join(f'{v:.4f}' for v in q['val_ratio'])} | "
          f"{q['n_it'] or q['vmc_steps']} | {q['evals']:.2e} | {q['sec']:.0f} |")
json.dump(rows, open(os.path.join(R, 'summary_rows.json'), 'w'), indent=1)
