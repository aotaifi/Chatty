# T-BAD10 round-1 SMOKE (written at submission 2026-11-15): tiny-size run of the five stages with their controls. Not production data; no gate (no averages quoted).
Read smoke/all.log. Pass conditions: S1 'RESUME_PROOF PASS' and the BROKEN verify returns 3; S2a bias_over_se within ~3 and sd ratio reported, S2b writes consts json + sha;
S3 BROKEN sha rc != 0, labels controls verdict PASS (C1 max dE_L small, C1_broken detected, C2 fractions > 0.5%, C3 mismatches 0), R1_LABELS_DONE; S4 R1_FIT_DONE + fit_*.json with held-out disagreement and E_L error;
S5 FN with the smoke net rc 0, BROKEN net sha rc != 0, analyze prints the READ line (NOT QUOTABLE expected at 3 blocks). Any FAIL -> stop, report to lead.
