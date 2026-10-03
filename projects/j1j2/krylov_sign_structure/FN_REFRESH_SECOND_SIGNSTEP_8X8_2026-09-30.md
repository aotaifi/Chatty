# 8x8 FN-refresh -> projected second sign-step verdict — 2026-09-30

Job 3492241, H100. Completed in 1m34s.

## Setup
Starting threshold:
T0 = -28.37107876288694

First FN four-bin amplitude correction:
g0 = [0.14059277629401276, 0.012627841129205876, -0.08930914480654009, -0.06367384713127924]

Refreshed projected threshold:
T1 = -28.002706773427757

## Decisive node diagnostic
Using the hidden ViT phase only as an after-the-fact diagnostic, and physical |a1|^2 importance weights:

O0_phys = 0.9982528688929572
O1_phys = 0.9981550698887751

wrong0_phys = 0.0008735655535214047
wrong1_phys = 0.0009224650556123614

changed_phys = 5.2305655290754705e-05
hit_phys = 1.7030765998990257e-06
harm_phys = 5.060257869085568e-05
precision_phys = 0.032560085337465475

Only 3/1024 sampled states changed sign:
- 1 flip corrected the hidden ViT sign
- 2 flips damaged previously correct signs

Thus the second projected sign-step after the current coarse FN amplitude refresh is slightly worse than the first-step node on this 8x8 diagnostic.

## Interpretation
This does NOT falsify the general amplitude-refresh -> sign-step loop, because the refreshed amplitude is represented only by a four-bin static-r0 density-ratio correction. It DOES falsify the claim that the current coarse 8x8 FN handoff is already accurate enough that another projected sign step improves the node.

Current evidence:
- 4x4 exact refreshed/projected iteration can converge to exact signs.
- 6x6 coarse FN refresh reached a practical fixed point with zero sign changes.
- 8x8 coarse FN refresh changes only ~5.2e-5 physical sign mass, but those changes are mostly harmful relative to ViT (1 helpful, 2 harmful).

So the current scaling bottleneck is amplitude-handoff fidelity / resolution, not raw Krylov depth.
