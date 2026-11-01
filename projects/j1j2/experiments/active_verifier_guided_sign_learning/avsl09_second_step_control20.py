#!/usr/bin/env python3
"""Independently attribute AVSL09B second-step FN gain to guide refresh vs signs."""
import argparse,json
from pathlib import Path
import numpy as np
from avsl09_k1_fn_loop20 import fn_exact_reference

def main(args):
 p=json.load(open(args.two))
 ns=fn_exact_reference(args.data)
 for block in p['runs']:
  for name,z in block['arms'].items():
   current=np.load(z['newdata_path'])
   same_E,same_phi,info=ns['fn_step'](current['amp'],
                  current['guide'].astype(float),'AVSL09 no-sign-change guide reproject',tol=2e-9)
   old_E=float(z['first_fn']['energy'])
   new_E=float(z['second_fn']['energy'])
   z['second_step_control']=dict(
      FN_energy_if_no_second_sign_change=float(same_E),
      guide_only_energy_gain=float(old_E-same_E),
      second_K1_sign_incremental_gain=float(same_E-new_E),
      second_composite_total_gain=float(old_E-new_E),
      FN_residual=float(info['resid']),
      separates_guide_from_sign=True)
   print('CONTROL',block['first_block'],name,
         'second K1 flips',z['second_k1']['n_selected_orbits'],
         'guide',round(old_E-same_E,9),'K1',round(same_E-new_E,9),flush=True)
 Path(args.out).write_text(json.dumps(p,indent=2))
 print('SAVED',args.out)
if __name__=='__main__':
 a=argparse.ArgumentParser()
 a.add_argument('--data',required=True);a.add_argument('--two',required=True);a.add_argument('--out',required=True)
 main(a.parse_args())
