> PUBLIC DERIVATIVE of the complete original saved-analysis report. Numeric results, adverse findings, all comparison families and controls are preserved. Changes are limited to review-status text, relative publication links, removal of private clearance wording, and the saved-only verification section. Original and public hashes are recorded in ../transformations.json.

# Historical replay update decomposition: saved results

One scientific invocation exited 0 with empty stderr. Nineteen artifacts, sixteen fixed states and four historical seed clusters verify, including exact complete-inclusive accounting of 3,175,082 bytes. All 350 scalar summaries were independently recounted from saved raw branch losses and controls exactly. No model, checkpoint arrays, image arrays, official-test inputs, training, tuning or scientific rerun were used in this analysis. The original saved bundle subsequently passed independent saved-result review; this public packaging needs its own release review.

## Adverse context and incomplete replication first

- Actual normalized JOINT MIR INCREASES absolute old-panel loss in all 16 states and all four seed means: +0.905519 nats/example, descriptive interval[+0.526345,+1.480879]. Relative selector advantage is not prevention of forgetting.
- Its old MIR-minus-highloss mean is -0.025069, but the interval[-0.051397,-0.004300] does NOT clear the frozen -0.005 engineering margin. The original accepted audit did clear that margin; this stronger result does not reproduce here.
- The all-four-seed endpoint sign pattern also does NOT reproduce: seed 2006 has positive JOINT contrast+0.002542 and positive sequential contrast+0.098271. Only three of four seed means reverse. JOINT loses to highloss in 7/16 states.
- Against unit-rate fresh-only, JOINT MIR improves old loss by-0.853972 but worsens new loss by+0.335974 [+0.162430,+0.488773], with plasticity cost in 12/16 states. Against highloss its new-panel interval crosses zero and its new loss is worse in 11/16 states. Do not hide these signs behind the favorable mean.
- These are additional mechanically selected historical states on the same old data and validation, not fresh independent confirmation. Four seeds, not sixteen states or repeated replay draws, are the uncertainty units. Do not pool this with the earlier audit to call it doubled independent evidence.

## Main component finding

At the aggregate old-panel MIR-minus-highloss contrast, neither changing weighting alone nor moving replay at deployed weights produces the adverse sequential sign: D00=-0.025069; D01=-0.070031; D10=-0.131991; D11=+0.340123. The specified combined intervention changes sign through a large positive interaction, not an additive sum of adverse individual effects:

W=-0.106923; L=-0.044963; I=+0.517077; total=+0.365192 =W+L+I.

Here W changes combined relative/total weights with replay still at the original state; L moves replay at deployed weights. I=(location effect at unit weights)-(location effect at deployed weights). Equivalently, I is the difference of the two weighting simple effects. The replay-gradient displacement is weight-matched, not numerically identical across weighting levels. This is NOT a pure replay-ratio, total-scale, step-count, long-term retention or biological explanation. State signs and four-cluster uncertainty remain essential.

## Old-panel contrasts and all component effects

### mir_minus_highloss

All values are nats/example. Negative D favors MIR; signs for W/L/I describe component direction, not selector success. Intervals are descriptive seed-cluster95%, not confirmatory tests.

| Contrast | Equal-seed mean | Descriptive interval | State negative/positive | Seed negative/positive |
|---|---:|---|---:|---:|
| cell00 | -0.025069 | [-0.051397, -0.004300] | 9/7 | 3/1 |
| cell01 | -0.070031 | [-0.126709, -0.013354] | 11/5 | 4/0 |
| cell10 | -0.131991 | [-0.229444, -0.031527] | 9/7 | 3/1 |
| cell11 | +0.340123 | [+0.077903, +0.828661] | 4/12 | 0/4 |
| weight_at_original | -0.106923 | [-0.190448, -0.023397] | 10/6 | 3/1 |
| location_at_deployed | -0.044963 | [-0.087263, -0.006472] | 9/7 | 3/1 |
| interaction | +0.517077 | [+0.164894, +1.034945] | 3/13 | 0/4 |
| total_sequential_minus_joint | +0.365192 | [+0.110142, +0.845876] | 4/12 | 0/4 |
| weight_at_updated | +0.410155 | [+0.151297, +0.852347] | 3/13 | 0/4 |
| location_at_unit | +0.472115 | [+0.131617, +1.015729] | 3/13 | 0/4 |

### mir_minus_uniform

All values are nats/example. Negative D favors MIR; signs for W/L/I describe component direction, not selector success. Intervals are descriptive seed-cluster95%, not confirmatory tests.

| Contrast | Equal-seed mean | Descriptive interval | State negative/positive | Seed negative/positive |
|---|---:|---|---:|---:|
| cell00 | -0.061029 | [-0.087268, -0.034789] | 14/2 | 4/0 |
| cell01 | -0.045999 | [-0.101146, +0.013013] | 12/4 | 3/1 |
| cell10 | -0.062817 | [-0.187196, +0.107394] | 10/6 | 3/1 |
| cell11 | +0.507174 | [+0.152119, +0.986153] | 0/16 | 0/4 |
| weight_at_original | -0.001788 | [-0.109360, +0.170663] | 9/7 | 3/1 |
| location_at_deployed | +0.015029 | [-0.032929, +0.059314] | 8/8 | 1/3 |
| interaction | +0.554961 | [+0.190530, +1.116960] | 1/15 | 0/4 |
| total_sequential_minus_joint | +0.568202 | [+0.219164, +1.040780] | 0/16 | 0/4 |
| weight_at_updated | +0.553173 | [+0.169706, +1.024838] | 1/15 | 0/4 |
| location_at_unit | +0.569991 | [+0.241562, +1.113851] | 0/16 | 0/4 |

## New-panel contrasts and all component effects

### mir_minus_highloss

All values are nats/example. Negative D favors MIR; signs for W/L/I describe component direction, not selector success. Intervals are descriptive seed-cluster95%, not confirmatory tests.

| Contrast | Equal-seed mean | Descriptive interval | State negative/positive | Seed negative/positive |
|---|---:|---|---:|---:|
| cell00 | -0.028776 | [-0.058482, +0.003624] | 5/11 | 3/1 |
| cell01 | -0.058689 | [-0.186993, +0.008726] | 8/8 | 2/2 |
| cell10 | -0.147789 | [-0.276096, -0.012842] | 7/9 | 3/1 |
| cell11 | +0.456718 | [+0.116858, +1.025437] | 3/13 | 0/4 |
| weight_at_original | -0.119013 | [-0.214156, -0.019924] | 10/6 | 3/1 |
| location_at_deployed | -0.029913 | [-0.152064, +0.050339] | 7/9 | 2/2 |
| interaction | +0.634420 | [+0.236463, +1.259009] | 1/15 | 0/4 |
| total_sequential_minus_joint | +0.485494 | [+0.148182, +1.073373] | 3/13 | 0/4 |
| weight_at_updated | +0.515407 | [+0.219337, +1.036759] | 2/14 | 0/4 |
| location_at_unit | +0.604507 | [+0.191659, +1.268547] | 2/14 | 0/4 |

### mir_minus_uniform

All values are nats/example. Negative D favors MIR; signs for W/L/I describe component direction, not selector success. Intervals are descriptive seed-cluster95%, not confirmatory tests.

| Contrast | Equal-seed mean | Descriptive interval | State negative/positive | Seed negative/positive |
|---|---:|---|---:|---:|
| cell00 | -0.042298 | [-0.075181, -0.016670] | 13/3 | 4/0 |
| cell01 | -0.046847 | [-0.126311, +0.032617] | 8/8 | 3/1 |
| cell10 | -0.072788 | [-0.160942, +0.018105] | 11/5 | 3/1 |
| cell11 | +0.437996 | [+0.104915, +0.971885] | 3/13 | 0/4 |
| weight_at_original | -0.030490 | [-0.093016, +0.039007] | 10/6 | 3/1 |
| location_at_deployed | -0.004549 | [-0.068033, +0.058935] | 7/9 | 2/2 |
| interaction | +0.515333 | [+0.137276, +1.041259] | 2/14 | 0/4 |
| total_sequential_minus_joint | +0.480294 | [+0.129312, +1.023906] | 3/13 | 0/4 |
| weight_at_updated | +0.484843 | [+0.169312, +0.976602] | 1/15 | 0/4 |
| location_at_unit | +0.510784 | [+0.144959, +1.078430] | 4/12 | 0/4 |

Random-exact-tie MIR-minus-highloss has EXACTLY the same old/new cells, W, L, I, total and conditional simple effects as source-ID MIR-minus-highloss. Random-tie contrast shift and all its decompositions are exactly zero on both panels. These two prespecified comparison families are not omitted or treated as independent replications; their full rows are in ../analysis/aggregate-components.csv and ../analysis/saved-analysis.public.json.

## Per-seed primary old-panel decomposition

| Seed | D00 | D01 | D10 | D11 | W | L | I | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2005 | -0.014192 | -0.003851 | -0.207049 | +1.072124 | -0.192857 | +0.010341 | +1.268832 | +1.086316 |
| 2006 | +0.002542 | -0.022858 | +0.026980 | +0.098271 | +0.024438 | -0.025399 | +0.096690 | +0.095729 |
| 2007 | -0.063799 | -0.120708 | -0.251839 | +0.060756 | -0.188040 | -0.056909 | +0.369505 | +0.124556 |
| 2008 | -0.024825 | -0.132709 | -0.096057 | +0.129342 | -0.071232 | -0.107884 | +0.333283 | +0.154167 |

Old MIR-minus-highloss endpoint sign categories: {"seed": {"both_negative": 0, "both_positive": 1, "joint_negative_sequential_positive": 3, "joint_positive_sequential_negative": 0, "location_alone_at_deployed_flips_joint_sign": 1, "weight_alone_flips_joint_sign": 0}, "state": {"both_negative": 3, "both_positive": 6, "joint_negative_sequential_positive": 6, "joint_positive_sequential_negative": 1, "location_alone_at_deployed_flips_joint_sign": 8, "weight_alone_flips_joint_sign": 2}}. These counts do not turn state rows into independent samples.

Seed 2005 carries the largest interaction and sequential effect. Leaving it out reduces mean old MIR-highloss interaction from +0.517077 to+0.266493 but does not make it negative. All four interaction seed means are positive. All leave-one-seed-out JOINT means remain negative and sequential means positive; these are unplanned descriptive influence checks, not opportunities to discard seeds or strengthen a confirmation claim.

Full signed state and seed results for BOTH panels and ALL four comparison families are exported in ../analysis/signed-state-components.csv and ../analysis/signed-seed-components.csv. Every one of the 350 prespecified outcomes/control/decomposition summaries, not a selected subset, is in ../analysis/all-350-summary-metrics.csv and ../analysis/saved-analysis.public.json. Original raw state artifacts remain immutable.

## Absolute old/new controls

The following are MIR losses relative to each frozen control; positive means worse. Both fresh scales are shown. The lower-rate fresh control is not interchangeable with the original full-rate fresh virtual step.

| Cell | Panel | Versus original | Versus deployed-weight fresh | Versus unit-weight fresh |
|---|---|---:|---:|---:|
| 00 | old | +0.905519 | -0.116621 | -0.853972 |
| 00 | new | -8.836032 | -0.057774 | +0.335974 |
| 01 | old | +0.362925 | -0.659215 | -1.396566 |
| 01 | new | -9.407497 | -0.629239 | -0.235491 |
| 10 | old | +1.629763 | +0.607623 | -0.129728 |
| 10 | new | -9.236317 | -0.458059 | -0.064311 |
| 11 | old | +1.202602 | +0.180462 | -0.556889 |
| 11 | new | -9.412248 | -0.633990 | -0.240242 |

Cell00 improves new loss relative to the lower-rate fresh control on average (-0.057774), while it sacrifices new learning relative to the unit-rate fresh control (+0.335974). This matched-scale distinction must not be relabelled as a universal plasticity benefit. Every cell retains absolute old/new values and all selections, not merely MIR.

## Numerical recovery, ties and evidence

All 128 selected-set endpoint checks pass: fresh-unit/direct-fresh and cell11/direct-sequential are bitwise equal; actual00 remains the actual concatenated source step. The separate-gradient mathematical00 maximum parameter difference is 2.38418579102e-07, maximum fixed-tolerance fraction 0.0514788271188, and maximum old/new panel-loss difference6.17742538456e-07. None is silently treated as exact numerical identity or a fifth scientific cell.

Tie log totals: {"cutoff_records_including_random_tie_sensitivities": 320, "extra_exact_cutoff_items": 0, "extra_near_cutoff_items": 0, "highloss_selection_changed_states": 0, "mir_selection_changed_states": 0, "straddling_cutoffs": 0}. Tie sensitivity does not explain this decomposition in these fixed states. No conclusion about absent ties elsewhere follows.

The unchanged resource contract passed. Recorded samples before completion writing are CPU 3.344052105 seconds, wall 3.365502604 seconds and RSS high-water 96,952,320 bytes. These are not final end-to-end process timings. Exact artifact bytes do include COMPLETE and were reverified after observed external exit 0. Before-launch invocation/freeze/preflight/UTC receipt, original stdout/stderr/exit, completion/identity envelope receipt and original review are retained. No automatic retry or cap expansion occurred.

## What this improves and what remains unknown

This resolves the earlier compound-update ambiguity more specifically: in these historical states the adverse sequential MIR contrast is associated with positive weighting-by-replay-location nonadditivity, while either isolated change at the specified reference level is favorable on average against highloss. It does not establish a uniquely dominant universal cause, a better deployment policy, a retained trajectory advantage, generality across eta/model/data, or any brain mapping. The weight factor still bundles ratio with total scale. A separate ratio-versus-scale factor and a separately frozen continuation/interference study would be needed to address those questions. No such further scientific work is authorized or executed by this report.

Saved-only verification, no model/data/checkpoint access, from this study directory:

    env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B verify_saved.py

The verifier checks release inventory, hashes and all saved arithmetic, including the fixed descriptive bootstrap, without importing the reference model. It is not a scientific rerun. The original historical checkpoints, image inputs and complete private execution provenance are absent. Acceptance of the original saved bundle is not independent confirmation of its scientific claim, and this packaging does not upgrade it to full-execution reproduction.
