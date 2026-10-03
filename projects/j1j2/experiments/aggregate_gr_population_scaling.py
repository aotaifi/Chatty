import numpy as np, json
rows=[]
for M in (32,64,128):
    z=np.load(f'/Users/aliotaifi/j1j2_vit_bench/gr_6x6_population_scaling_M{M}.npz')
    sig=z['sigma_pair']; rms=float(np.sqrt(np.mean(sig[1:]**2)))
    er=z['rows'][:,1]
    rows.append(dict(M=M,gcombined=z['gcombined'].tolist(),
                     sigma_pair=sig.tolist(),rms_bins123=rms,
                     rms_sqrtM=rms*np.sqrt(M),
                     Emeans=er.tolist(),Ediff=float(er[0]-er[1]),
                     mixed_counts=z['mixed_counts'].astype(int).tolist()))
print(json.dumps(rows,indent=2))
np.savez_compressed('/Users/aliotaifi/j1j2_vit_bench/gr_6x6_population_scaling_summary.npz',
    M=np.array([r['M'] for r in rows]),rms=np.array([r['rms_bins123'] for r in rows]),
    rms_sqrtM=np.array([r['rms_sqrtM'] for r in rows]),
    Ediff=np.array([r['Ediff'] for r in rows]))
