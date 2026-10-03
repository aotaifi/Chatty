# Recompute the validated iter2 SR direction and save several trust-region candidates.
exec(open("fnmle8_sr_iter2_train.py").read().replace(
'rows=[]; candidates=[]\nfor eta in [1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,2e-2,3e-2]:',
'''# Save physically gated candidate checkpoints before the likelihood-only chooser.
for eta_save in [0.0003,0.001,0.003]:
 p_save=unravel(flatc-eta_save*sol)
 vv=dict(vc); vv["params"]=p_save
 tag={0.0003:"0003",0.001:"001",0.003:"003"}[eta_save]
 open(f"vit8_fnmle_sr_iter2_eta{tag}.mpack","wb").write(flax.serialization.to_bytes(vv))
 print("ITER2_SAVED_CANDIDATE",eta_save,f"vit8_fnmle_sr_iter2_eta{tag}.mpack",flush=True)
rows=[]; candidates=[]
for eta in [1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,2e-2,3e-2]:'''))
