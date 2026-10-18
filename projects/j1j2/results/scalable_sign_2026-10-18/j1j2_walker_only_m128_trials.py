#!/usr/bin/env python3
"""Post-hoc local sign candidate test for the 128-walker NCE amplitude guide."""
import json,torch,numpy as np
from pathlib import Path
from j1j2_walker_only_amplitude20 import T,M,CountedGuide,LocalModelSampler,eval_trials
from j1j2_walker_only_eval20 import orbit_true_gain
P=Path('/mnt/data');ck=torch.load(P/'j1j2_walker_only_amplitude20_m128_model.pt',map_location='cpu',weights_only=True)
net=T.Net();net.load_state_dict(ck['model']);net.eval()
truth=np.load(P/'j1j2_walker_only_fn_truth_audit20.npz')['fn'];trials=eval_trials(net)
for r in trials:
 s=LocalModelSampler(CountedGuide())
 for key in ('fitted','jastrow'):
  selected=r['arms'][key]['selected_keys'];r['arms'][key]['true_fn_amplitude_gain']=float(orbit_true_gain(selected,s,truth))
print('TRUE',[(r['seed'],{m:r['arms'][m]['true_fn_amplitude_gain'] for m in ('fitted','jastrow')}) for r in trials],flush=True)
res=dict(trials=trials,summary={m:dict(n_positive=int(sum(r['arms'][m]['true_fn_amplitude_gain']>0 for r in trials)),mean_true_gain=float(np.mean([r['arms'][m]['true_fn_amplitude_gain'] for r in trials])),mean_ed_weight_change=float(np.mean([r['arms'][m]['ED_weight_improvement'] for r in trials]))) for m in ('fitted','jastrow')},scope='128-walker guide trained without FN labels; post-hoc exact FN evaluation; sampled candidate pool differs from 3500-walker guide, so not a paired cross-population ranking test')
(P/'j1j2_walker_only_m128_trials.json').write_text(json.dumps(res,indent=2))