import json,numpy as np
S=json.load(open('bound_check_summary.json')); fn=S['FN']; vm=S['variational_sym']
N=36
ex=json.load(open('../fn_guides_comparison/fn_guides_comparison.json'))['6x6']['FN']['krylov']
reps={}
for g,e in fn.items():
    for s,v in zip(e['seeds'],e['E_site_reps']): reps[(g,s)]=v
reps[('T_FN_M128_b1.2_tau.025',9502)]=ex['rep_values'][1]/N   # existing run, identical code (seed 9501 reproduced bit-for-bit)
sel=lambda key:[v for (g,s),v in reps.items() if key in g]
f=lambda a:(float(np.mean(a)),float(np.std(a,ddof=1)/np.sqrt(len(a))),len(a))
res=dict(E_FN_all_T_FN_M128=f(sel('T_FN')),E_FN_beta1p2_tau025=f(sel('T_FN_M128_b1.2_tau.025')),E_FN_beta2p4=f(sel('b2.4')),
         E_FN_tau_half=f(sel('tau.0125')),E_FN_T_old=f(sel('T_old')))
H=vm['T_FN']; Hv=H['E_ViT_sampled_site']+H['dE_site']; Hse=float(np.hypot(H['E_ViT_SE_site'],H['dSE_site']))
res['H_guide_T_FN_site']=(Hv,Hse); res['dE_T_FN']=(H['dE_site'],H['dSE_site']); res['E_ViT_sampled']=(H['E_ViT_sampled_site'],H['E_ViT_SE_site'])
Ho=vm['T_old']; res['H_guide_T_old_site']=(Ho['E_ViT_sampled_site']+Ho['dE_site'],Hse); res['dE_T_old']=(Ho['dE_site'],Ho['dSE_site'])
a=res['E_FN_all_T_FN_M128']; d=a[0]-Hv; dse=float(np.hypot(a[1],Hse)); res['E_FN_minus_H_guide']=(d,dse,d/dse)
res['T_scan_dE_site']={k:(v['dE_site'],v['dSE_site']) for k,v in vm.items()}
res['all_reps']={f"{g}|{s}":v for (g,s),v in reps.items()}
json.dump(res,open('bound_check_final.json','w'),indent=1)
print(json.dumps({k:v for k,v in res.items() if k!='all_reps'},indent=1))
