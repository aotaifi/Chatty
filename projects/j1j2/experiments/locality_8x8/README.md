# Pair-interaction (locality) measurement at 8x8 (pre-registered 2026-10-13, lead approved; before any analysis)
**Question.** Is the outer hop local at 8x8? The light-cone fold-back (compress_6x6 D2) needs psi_P's log-amplitude
to be pair-additive for bond swaps beyond a short radius:
- I(b, b') = la(x^{bb'}) + la(x) - la(x^b) - la(x^{b'}) must be small at larger bond distance;
- the sign product s(x^{bb'}) s(x) s(x^b) s(x^{b'}) must be +1 there.
At 6x6 (FB-1) the exact ground state and psi_P both have rms I = 1.69 / 1.02 / 0.80 / 0.61 / 0.43 / 0.72 at distance
1 / 2 / 2.83 / 3 / 3.16 / 3.61, with sign non-additivity 19% -> 2-6%.

**Data (no new ViT evaluations).** Existing two-hop referee chunks of experiments/signs_8x8:
- run `s8RB1_17160055`: beta = 1, i.e. x ~ |psi_P|^2; 24 chunks x 128 = 3072 samples;
- the D4 x flip projected 8x8 ViT (log amplitude and binarised own sign) on every configuration within two exchange
  moves of each sample.
- For every sample: all pairs of valid, site-disjoint bonds (b, b'). x^b, x^{b'} and x^{bb'} all lie in the stored
  two-hop set.

**Measured** by bond distance (minimum periodic site distance, as at 6x6), up to the 8x8 maximum:
- count, rms I, mean |I|, sign non-additivity;
- weighted rms I with weights r_b r_b' = a(x^b) a(x^{b'}) / a(x)^2;
- mean contribution error r_b' |sigma e^I - 1|, i.e. the error the additive approximation makes in the b'-term of
  E_L(x^b).
- Compared with the 6x6 numbers at equal distance (`results/compress_6x6/runs/cx_pairint.json`).

**Threshold X (derived from 6x6 D2).**
1. At 6x6 the rho = 2 light-cone level is exact for bonds within distance 2 and additive beyond (rms I 0.4-0.9
   there). Its target error is 0.027, against 0.017 with no neighbour information and a viability bar of
   0.25 x 0.017 = 0.0043.
2. The error is linear in the additive-approximation error (sigma e^I - 1 ~ I for small I), so the far-pair rms I
   must drop by about 0.027 / 0.0043 = 6.3: from ~0.8 to <= 0.13.
3. 8x8 has about twice as many far pairs per bond; with random-sign accumulation the error grows by sqrt(2). Hence
   **X = 0.10**.
4. A sign defect contributes |sigma e^I - 1| ~ 2, i.e. a rate p adds about 2 sqrt(p) to the rms. So p <= 0.25%.

**Readings (pre-registered):**
- **Outer hop LOCAL at 8x8:** rms I <= 0.10 AND sign non-additivity <= 0.3% at every distance >= 3.
- **NOT LOCAL:** rms I >= 0.30 OR sign non-additivity >= 2% at some distance >= 3.
- Otherwise MARGINAL: a quantitative extrapolation, no verdict.

Code `s8_pairint.py` (CPU, numpy). Budget: CPU only, < 0.2 h on the `small` partition.

## Amendment L-1 (2026-10-13 17:00, after the pair statistics, before this addition; descriptive, no verdict change)
- **Pair statistics:** NOT LOCAL by the pre-registered rule. At distance >= 3: rms I = 0.25-0.72 and sign
  non-additivity 0.6-4.8%.
- **Addition.** The error this causes in the quantity the light cone needs, E_L of psi_P at the neighbour x^b, from
  the same data:
  - exact for b' within rho of b, additive beyond, for rho = 0, 1, 2, 3, 4, 5;
  - reported as rms over (x, valid b), unweighted and edge-weighted (J_b/2) a(x^b)/a(x), in units of the sample sd
    of E_L(x).

## Locality vs J2 (pre-registered 2026-10-13, PI request; before any run; code `ed_pairint.py`; CPU <= 1 h)
**Question.** Is the large double-swap interaction I(b, b') a property of the near-critical point J2 = 0.5? Does
locality return away from it?
- A new 6x6 sector Hamiltonian at another J2 would need a CSR rebuild, so this uses a smaller cluster with full
  exact diagonalisation: the 4x6 torus (24 sites, Sz = 0, 2.7e6 states, scipy eigsh), at J2 = 0, 0.3, 0.4, 0.5,
  0.6.
- For each ground state: 2000 samples of |psi0|^2, all valid site-disjoint bond pairs, rms I and sign
  non-additivity by bond distance (up to the cluster maximum of about 3).
- Calibration: 4x6 at J2 = 0.5 against the 6x6 exact numbers at equal distance.
- Expectation written down now: at J2 = 0 the Marshall sign makes the sign exactly additive (non-additivity 0).
  Amplitude interactions may still be sizeable (long-range Neel order).

**Readings (pre-registered):**
- **"Locality returns at J2 <= J2*":** for all J2 <= J2*, rms I at distance >= 2.8 is <= 0.10 and sign
  non-additivity <= 0.3%. Same bar X as at 8x8.
- **"Locality is J2-dependent, but not to the bar":** the J2 = 0 values at distance >= 2 are >= 3x below the
  J2 = 0.5 values, without reaching 0.10.
- **"Not J2-driven":** within a factor 1.5 at every J2.
- Caveat stated in advance: on 4x6 a distance of ~3 is half the cluster, so this compares J2 at equal distance; it
  is not an asymptotic decay measurement.
