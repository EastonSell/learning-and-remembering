# Same-state replay diagnostic: measured results and limits

The retained finite bundle contains 16 fixed states in four seed clusters. Its original saved records report one execution, exit 0 and empty stderr, with 19 original artifacts totaling 1,060,107 bytes including COMPLETE. This public release preserves every branch result, summary, completion marker and saved numerical analysis. It supplies labeled provenance projections rather than the original address-bearing freeze and execution records. No new scientific run or inference was performed to prepare this release.

## Main finding: update composition reverses the local selector contrast

For MIR minus high-loss old-class macro cross-entropy, lower is better:
- Deployed JOINT update: −0.09549 nats/example; descriptive four-seed bootstrap 95% interval [−0.21708, −0.00955].
- Same-state/same-selection SEQUENTIAL update: +0.16334; interval [+0.10774, +0.23660].
- JOINT minus SEQUENTIAL contrast: −0.25883; interval [−0.45555, −0.12788].

This shows that the older audit composition cannot substitute for the deployed update in this local diagnostic. It does not identify which component of the update change causes the reversal. This new sequential comparator is MIR versus high-loss at the same selected sets, not a numerical rerun of the historical selected-versus-shuffled estimand. The original historical adverse result is preserved.

JOINT MIR minus the mean of four within-class uniform selections is −0.04724 [−0.09163, −0.00339]. The frozen 0.005-nat yardstick is an engineering interpretive resolution, not a powered effect or retention threshold. The high-loss contrast meets its candidate-local-signal rule; the uniform interval does not establish an advantage beyond that full 0.005 margin. Neither is fresh confirmation.

## Essential adverse context and plasticity

- JOINT MIR still INCREASES old-panel loss from the original state by +0.71271 nats on average [ +0.46861, +0.94405 ]. Relative selector benefit is not absolute protection or absence of forgetting.
- Against fresh-only SGD, JOINT MIR reduces old-panel loss by −0.63616 [−0.83626,−0.50523] but INCREASES new-class panel loss by +0.37056 [+0.20716,+0.56280]. Replay has a measured old/new one-step trade-off here.
- Relative to high-loss, JOINT MIR new-class loss difference is −0.07164 [−0.21222,+0.01624]; relative to uniform it is −0.01909 [−0.04536,+0.00717]. These intervals cross zero, with worse outcomes in six and seven states respectively. No general plasticity advantage follows.
- Old-panel MIR beats high-loss in 10/16 states and LOSES in 6; it beats uniform in 12/16 and loses in 4. The mean is not an all-state guarantee.
- All four seed means favor MIR over high-loss, but seed 2002 carries much of the magnitude. Omitting it changes the mean from −0.09549 to −0.03261. Every leave-one-seed-out high-loss mean remains negative, but four historical clusters remain a small, conditional sample.

## Seed and state heterogeneity

| Seed | JOINT old MIR−high-loss | JOINT old MIR−uniform | JOINT new MIR−high-loss |
|---|---:|---:|---:|
| 2001 | -0.078739 | -0.062414 | +0.026576 |
| 2002 | -0.284114 | -0.119746 | -0.284931 |
| 2003 | -0.003103 | -0.007267 | +0.005910 |
| 2004 | -0.015991 | +0.000481 | -0.034108 |

Per-state signs, with every adverse row retained:

| Seed | Boundary | JOINT old MIR−HL | JOINT old MIR−uniform | JOINT new MIR−HL |
|---|---:|---:|---:|---:|
| 2001 | 0 | -0.078013 | -0.006086 | -0.024107 |
| 2001 | 1 | +0.015631 | -0.001891 | +0.000520 |
| 2001 | 2 | -0.097015 | -0.150296 | +0.048226 |
| 2001 | 3 | -0.155561 | -0.091384 | +0.081664 |
| 2002 | 0 | -0.023253 | -0.000673 | -0.002764 |
| 2002 | 1 | -0.565776 | -0.053696 | -0.327339 |
| 2002 | 2 | -0.403149 | -0.232684 | -0.775821 |
| 2002 | 3 | -0.144279 | -0.191933 | -0.033802 |
| 2003 | 0 | +0.000037 | +0.000045 | -0.000983 |
| 2003 | 1 | -0.027449 | +0.002950 | -0.021133 |
| 2003 | 2 | +0.000321 | -0.030747 | +0.010671 |
| 2003 | 3 | +0.014681 | -0.001316 | +0.035086 |
| 2004 | 0 | +0.000385 | -0.001507 | -0.000813 |
| 2004 | 1 | +0.027353 | -0.000405 | -0.133999 |
| 2004 | 2 | -0.068010 | +0.000623 | -0.043310 |
| 2004 | 3 | -0.023691 | +0.003213 | +0.041690 |

## Ties do not explain this measured reversal

For each selector, all 80 logged class cutoffs had a single exact cutoff item and no additional item within 1e-7. There were zero exact cutoff-straddling ties, zero randomized-tie selection changes and zero tie-induced contrast shift across 16 states. This does not prove historical summary-level tie effects were absent; it rules out that explanation in this fixed local audit.

## What remains unidentified

JOINT uses fresh/replay mean-loss weights 16/26 and 10/26, both gradients at the original state. SEQUENTIAL uses full-rate fresh then full-rate replay evaluated at the changed state. Thus relative weights, effective step magnitude and replay-gradient evaluation location change together. The observed composition interaction is real for these source-defined branches, but does not isolate any one factor.

The study is post-hoc on historical uniform-origin checkpoints at class transitions, one known dataset, previously used validation data, one rate/model and four seed/stream clusters. Sixteen states and four uniform draws are NOT independent scientific samples. Bootstrap resamples four seed means; exact empirical four-seed bootstrap and leave-one-out sensitivity are in saved-analysis.json. No general dataset/domain uncertainty, unseen confirmation, biological roles, final accuracy or future retention was measured.

## Reproduction and evidence

Run python -B recount_saved.py from this directory, or pass its full location from any working directory. It reads saved JSON only and emits the exact bytes of saved-analysis.json. It imports neither model source nor datasets/checkpoints. NumPy 2.3.5 matches the recorded environment. Optional --output requires a new destination and never overwrites.

Run python -B verify_release.py --recount to verify the public allowlist, retained original result hashes, explicit projections and exact recount. These checks establish the retained-evidence boundary described in README.md; they do not claim to verify undistributed original records or reproduce scientific model outputs. Full-precision signed values remain in saved-analysis.json and all sixteen state files.
