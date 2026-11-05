# CNN-MPS two-swap ratio cost, 8x8 and 14x14 (2026-11-05, reading and arithmetic only, nothing run)
Code: /scratch/a/A.Otaifi/laqx_review/repo_dir/laqx, HEAD c6a3ee5 (git log). File F = laqx/networks/cnn_mps.py, NB = laqx/networks/network_block.py.
Record: arXiv 2603.14425 v1 text, 10x10 best energy -0.4976939(2) for (h,D,layers)=(32,20,20); 6x6 uses the same h,layers. The TeX in src/ gives only 6x6 (D=20).

## Architecture as implemented
- Embed: input[(site state)] + pe[site], h=32 (F:120); polarized -> 2 physical states (F:31, NB:439). 6x6 script: docs/examples/spin/6x6_J1J2_cnnmps.sh (hidden 32, layers 20, MLP_hidden 64, MLP_layers 1, mpsdim 20, D4).
- CNN: 20 residual blocks h = h + SiLU(conv3x3(h)), then LayerNorm per site (F:123-139). Kernel 3x3 = --cutoff default 3 (main.py:57; NB:88), h->h channels, PBC wrap (NB:80-86).
- Site MLP: 1 linear 32->64 (F:57-67,143-144), then output linear 64 -> 2 phys x D^2 x 2 heads = 1600 (F:76,152). All 2 phys slices are computed, one is selected afterwards (F:166).
- MPS: D=20, heads=2 (main.py:56, summed), PBC trace (F:170, NB:496-509), linear scan, one D^3 x heads product per site. Tensors A_i depend on the CNN feature at site i.
- Path: row-major raster i = x*L2+y (F:121 reshape; TeX l.285), NOT a boustrophedon snake.
- Symmetry 'D4': rot180 x rot90 x flip = 8 images, each a full forward, vmapped (NB:271-319 with F:182-210). Marshall sign via --marshall (F:92).

## Light cone and path distance
- One changed site: Chebyshev radius 20 (20 layers x 1; LayerNorm, MLP, output are per site). Pair swap: box (2l+1)x(2l+2) after l layers.
- With PBC it covers the whole lattice from layer 4 (8x8) and layer 7 (14x14). So every MPS tensor A_i changes.
- Path distance between lattice neighbours: horizontal 1; vertical L (8, 14); mean over bonds (L+1)/2 = 4.5, 7.5. Wrap: horizontal L-1 (7, 13); vertical wrap (L-1)L = 56, 182 linear, L (8, 14) via the ring. Moot, see next section.

## FLOPs (1 MAC = 2 FLOP), per site per image
- CNN 20 x 9x32x32 = 184,320 MAC; MLP 32x64 = 2,048; output 64x1600 = 102,400; MPS 2 x 20^3 = 16,000. Total 304,768 MAC.
- (b) one image: 8x8 N=64 -> 19.505 MMAC = 39.0 MFLOP; 14x14 N=196 -> 59.735 MMAC = 119.5 MFLOP. With 8 images: 312 MFLOP and 956 MFLOP.
- (a) cached, one swap: CNN only on the box, summed over layers: 8x8 sum_l A_l = 12+30+56+17x64 = 1186 of 1280 site-layers; 14x14 12+30+56+90+132+182+14x196 = 3246 of 3920.
  MLP + output must run at all N sites (features changed everywhere): 104,448 MAC/site. MPS must be fully redone (environments invalid): 16,000 MAC/site.
  8x8: 10.93 + 6.68 + 1.02 = 18.64 MMAC = 37.3 MFLOP/image (x8 = 298 MFLOP). 14x14: 29.92 + 20.47 + 3.14 = 53.52 MMAC = 107.0 MFLOP/image (x8 = 856 MFLOP).
- (a) two far-apart swaps (box doubled, capped at N): 8x8 sum_l = 24+60+18x64 = 1236 -> 19.10 MMAC; 14x14 sum_l = 24+60+112+180+16x196 = 3512 -> 55.97 MMAC.
- Ratio (a)/(b): 8x8 0.956 (one swap) to 0.979 (two swaps); 14x14 0.896 (one) to 0.937 (two). The same with 8 images: the images are different configurations, each needs its own cache.
- Even an ideal cache (selected-q-only output, which would halve the output term) saves the same fraction from (b) once the same change is made to (b).

## Versus a plain step
- Reading: a plain walker step = one proposal forward = (b); mcmc.py:101-117 and the mh_update call f(params,x2,cache) with a full forward (cache ignored, F:105). So (a)/plain = 0.96-0.98 at 8x8 and 0.90-0.94 at 14x14: about 1x, not 5-11x.
- Local energy: the number of connected configs is ~2N-4N (J1+J2 bonds; about half antiparallel, inference) so a full E_loc costs ~2N plain steps: ~130-260 at 8x8, ~390-780 at 14x14. This is the "generic ~2N x" class; caching does not move it.

## Does the code cache anything
- apply_cnn_mps ignores its cache (F:105 `del cache`; F:183 `del cache`; make_cnn_mps returns None for the cache init, F:212). The returned dict has only 'sign','logdet' (F:172, F:210): no activations, no MPS environments.
- Only sign/logdet of the current x is kept: ham_utils.py:50-54 (init forward of x), mcmc.py:104.
- Each off-diagonal element runs a full forward: ham_utils.py:42 (_t_fast). The fast_update path (ham_utils.py:75-79) reads slice_cache['neighbor'], which cnn_mps never returns, so --fast_update would break (not tested).
- Other nets have light-cone code (NB:93-117 nearby_convolution*), not wired into cnn_mps.

## Kill line (a > 50x plain step at 14x14): NOT triggered. (a) = 0.90-0.94 x plain, because caching saves almost nothing, not because it is cheap.
Not checked: record hyperparameters of 10x10 beyond (32,20,20) and kernel 3x3 (taken from the repo default and the paper's supplement example); whether 14x14 would use other hyperparameters; wall time (GPU, tf32), which is set by memory traffic not FLOPs.

## Lead reading (2026-11-05)
The kill line was meant for the JACOBI step, not one ratio: a Jacobi step needs ~d^2 = (2N)^2 ratios per walker move. With caching saving
only 2-10% per ratio (20-layer light cone covers the lattice from layer 4/7; every MPS tensor changes; cache unused in code, cnn_mps.py:105),
each ratio ~ one full forward, so one Jacobi step ~ 2N x a plain local-energy step (~400x at 14x14): KILL FIRES for CNN-MPS (record depth 20).
A shallow CNN (repo default --layers 4, main.py:51) has light-cone radius 4, still ~81 of 196 sites at 14x14 plus a raster-path MPS that changes
across ~9 rows: no constant-factor update either. CNN-MPS stays usable as a node sign (T-HYB class), not as the walled amplitude.
