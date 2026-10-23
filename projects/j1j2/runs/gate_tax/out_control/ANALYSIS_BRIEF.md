# gtax_control (Paderborn H100; remote /scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6/runs/gtax_control; local /project/theorie/a/A.Otaifi/chatty_compress6/runs/gtax_control)
Question (README compress_6x6 "M1 + gate", G-tax, control stage): do controls 1 (0-step fit returns A, 4096 orbits to 1e-12; broken version fails) and 2 (E_FN(K_1(A)) = 5.5738e-6; broken hop fails) pass, and what is s/step of a 10-step RBM+PP fit?
Files: cx_gtax.json {control1, control2, g0, phi0, heldout{s_per_step}, stop}, progress.txt. No MC chains (quotable.py n/a).
Gate: python3 -c "import json;d=json.load(open('/project/theorie/a/A.Otaifi/chatty_compress6/runs/gtax_control/cx_gtax.json'));print(d.get('stop'),d.get('control1',{}).get('ok'),d.get('control2',{}).get('ok'),d.get('heldout'))"
Next: both controls ok and done -> submit STAGE=full (400 steps). A control failure (stop key, exit 2) -> report, no full run.
