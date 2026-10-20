# AVSL-09: correlated block proposals vs frustration-aware exploration (2026-10-20)

**Two-tier plan because Mac/workstation unavailable.**

## Tier 1 completed locally: exact 4×4 test; not a production FN-M128 result

We tested BCCC-inspired blind spin-conserving 2×2 four-spin plaquette flips against simple legal Heisenberg configuration-space hops. At J1=1, J2/J1=.5 on the 16-site torus (12,870 Sz=0 configurations, 441 symmetry orbits), computed exact frozen-guide FN amplitudes in Marshall and post-K1 sign chambers. Independently drew stationary roots from the *exact* importance-transformed distribution g·phi_FN; this was NOT a finite-population M128 projector. Proposed configurations saw no ground-state labels or oracle gains. Full-orbit fixed-amplitude energy gains were evaluated only after proposal.

The J2-preferred 2-hop policy (random J2 legal exchanges with unrestricted fallback only when blocked) found 1.154 positive single-orbit flips per 64 distinct orbit-verification calls over 500 independent root sets vs unrestricted random2 0.614 and blind 2×2 plaquette 0.116. At 128 checks: 2.580 vs 1.435 vs 0.380 (200 root sets). At J2=.6: 3.575 vs 2.640 (200); at .4 gains were very sparse but 0.070 vs 0.020. These are proposal hit counts on ONE 4×4 physical FN chamber, NOT physical scaling uncertainty.

**Corrected control:** Two perfect Néel states have no legal J2 hops and carry ~9.73% of the post-K1 exact mixed root weight. The first policy's unrestricted fallback is thus significant. A *strict J2-only no-backtracking* implementation was independently tested: 1.163 profitable targets per 64 distinct checks vs 0.560 for unrestricted two-hop *also without backtracking* (300 root sets). This strict control is the clean physical test and is implemented for 20-site geometry in `avsl09_j2_twohop_proposer.py`; blocked roots NEVER silently take a J1 hop.

### FN energy rather than only sign hits

200 further 4×4 candidate-set trials per method, 64 distinct oracle scores, up to two best positive symmetry orbits jointly flipped, followed by an **independent exact FN eigensolve using the same frozen guide across all arms**:

| Proposer | Mean positive orbits | Number of trials lowering FN energy | Median ΔE_FN gain | Mean ΔE_FN gain |
|---|---:|---:|---:|---:|
| Strict J2-J2 no-backtrack | 1.235 | **144/200 (72%)** | 1.94×10⁻⁵ | 8.82×10⁻⁵ |
| Unrestricted random2 no-backtrack | 0.660 | 101/200 (50.5%) | 4.52×10⁻⁶ | **1.095×10⁻⁴** |

The discovered-positive-count advantage is clear conditional on this toy. **Mean reoptimized FN gain is NOT significantly better**: paired difference (strict J2 minus random) −2.13×10⁻⁵ with descriptive 95% interval [−7.19×10⁻⁵, +2.92×10⁻⁵]. Rare large random-discovered corrections can dominate mean energy. *No claim of solving the FN–Krylov scaling problem.*

Crucial limitation: exact 4×4 projected amplitudes were used by the **verification oracle** to score and rank candidate orbits; this is not the actual scalable Krylov verifier. The method gives a good simple cheap proposal baseline, NOT a ready-to-run learned sign iteration. The "BCCC-inspired" generic 4-site block idea failed in this test; the advantage comes instead from J2-specific frustration and connectivity.

### Tier 2 queued for Mac return — no claim it ran

Use the existing 20-site actual M128 mixed-FN code and independent archived seed sets (`avsl08_mixed_fn20.py`), untouched prior baselines, and `avsl09_j2_twohop_proposer.py`.

**Step A: candidate discovery cost.** On the *same* genuine FN walkers, compare unrestricted random2, strict J2-J2, strict J2-J2 no-backtracking, blind plaquette and the archived learned AVSL policy. Allocate 120 distinct **actual Krylov sign verifications** per seed; match and report full independent guide-amplitude calls, neighbor enumerations, and model inference, not only terminal orbit scores. Record blocked Néel roots and pool autocorrelation. No free exact FN amplitude for candidate selection.

**Step B: amplitude accessibility and joint FN energy.** After proposal, evaluate whether Krylov's pointwise sign test can truly be implemented on rare out-of-support targets with current learned/callable FN amplitudes. Cost all candidate-neighbor amplitudes; evaluate error against the exact 20-site FN oracle **only after selection**. Apply consistent globally callable sign corrections (not a free global 2^N sign table), then independently FN-reproject and measure ΔE_FN. Repeat on a changed FN iteration and genuinely heldout amplitudes. Include any failed joint corrections and negative ΔE.

**Step C: 6×6 only after Step B.** If full-cost FN improvement fails despite positive hits, prioritize off-support amplitude uncertainty / robust verification, not new RL or larger block ansätze. No experiments were executed on the inaccessible workstation.

Five local physics and energy identity unit tests passed. Local exact code, JSON logs and this report are available separately from the ChatGPT conversation (cannot directly push Python-generated binary files through GitHub connector here).

**Project decision:** Make a cheap frustration-aware proposer a serious baseline. Do NOT spend GPU training budget on BCCC-style block proposals without evidence they beat it on independent FN energy and honest compute cost.
