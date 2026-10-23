# gtax_full (Paderborn H100; remote /scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6/runs/gtax_full; local /project/theorie/a/A.Otaifi/chatty_compress6/runs/gtax_full)
Question (README compress_6x6 "M1 + gate", G-tax): tax_2 = E_FN(K_2(a_1)) - E_FN(K_2(phi_0)) after a fixed 400-step RBM+PP refit; held-out loss must fall. Expected ~25 min against 60 min limit.
Files: cx_gtax.json {control1, control2, heldout, a1{im_weight}, tax, stop}, progress.txt. No MC chains (quotable.py n/a).
Gate: python3 -c "import json;d=json.load(open('/project/theorie/a/A.Otaifi/chatty_compress6/runs/gtax_full/cx_gtax.json'));print(d.get('stop'),d.get('heldout'),d.get('tax'))"
Next: report tax_2, components, held-out curve, im_weight, s/step.
