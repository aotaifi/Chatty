# Brief for the Pfaffian angle: 3-round loop on the retrained 8x8 base (PI GO 2026-11-11; lead)

Base: rb8_retrain s306 params_final (sha 45fa54e6; 'simple' projection, 256 images; quotable VMC -0.498838(2), replica s307 -0.498841(2));
/scratch/a/A.Otaifi/rbmpp8/rb8_retrain/s306/train/params_final.npz (Paderborn: runs/rb8_retrain/s306/train/). Sign: the base's own, fixed.

Design (register before any run):
- R1..R3 at the fixed own sign, branch-2 pipeline: new pool drawn from the NEW base, labels c=4 f_J, two-PP refit with R1's fitter and
  warm-start rule.
- Bare c=0 FN grade after EACH round, paired against the base through the same code.
- EARLY KILL: stop if round k's bare gain < 0.5 x round k-1's gain (rounds do not accumulate).
- Final headline (base + p=1 vs B_3 + p=1, paired, same code) only if the rounds accumulate. Lead prior ~35% for a > 3 sigma final win.

First check: does the pipeline (pool8 draws, labeller, pool_fit two-PP model, gfmc grade, P1Guide) support the 'simple' 256-image
projection? If not, list the changes and their cost. Re-size the pool by bond ESS under the new base.

Budget: ~10 H100-h to the round-3 bare grade, ~20 for the headline; ceiling 40 H100-h for this test. December express + cont; theorie A40
where sooner. CPU parts (pool draws, labels) may start now; GPU steps report to the lead before submission.
