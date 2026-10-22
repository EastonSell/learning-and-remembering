# Matched-capacity role-mask control

Fresh additive experiment, 2026-10-22. Starting source: public commit
`b2bdba110d8f21e27f5c2b4701c2f8d1864a1514`. The original pilot and its
implementation are preserved. This is new work, not a reconstruction of any
unpublished experiment.

## Question and fixed design

Does the pilot depend on correct supplied input-role boundaries, rather than
merely having two branches and a context gate? We train the *same* functional
model after exchanging 0, 1, 2 or 3 coordinates between its two six-input masks.
There are always 132 trainable parameters, two width-8 branches, the same gate,
the same patch granularity, and exactly two validation mapping candidates.

The swaps are nested within each seed. Independent permutations of the six
coordinates on each side use RNG seed `training_seed + 888`. Nothing is selected
on losses, labels, validation performance or test results. Input coordinate order
within each branch is otherwise preserved, and initial parameters are identical
across conditions. Dataset seeds, examples, optimizer, minibatch order and 120
epoch budget are the pilot defaults. All conditions receive all 12 evidence
coordinates across their two branches; the context coordinate is unchanged.

The four mask conditions are not new kinds of biological models. Their display
labels are metadata, and every one uses the same `Model('functional', seed)`
code. To separately check name/order dependence, each trained model is copied,
its branch parameters and masks exchanged, and its gate weights/bias negated.
This algebraic reparameterization must preserve predictions and corresponding
patches after the patch index is exchanged. It is an implementation sanity check,
not evidence for or against a neuroscience hypothesis.

## Measured results

Mean ± sample SD in percentage points across seeds 11, 23, 37, 53, 71 on the
50%-category-agreement test split (2,048 examples per seed):

| Coordinates exchanged per branch | Parameters | Prediction accuracy | Memory IIA | Changed-case IIA |
|---|---:|---:|---:|---:|
| 0, aligned | 132 | 97.66 ± 0.23 | 97.45 ± 0.44 | 97.54 ± 0.80 |
| 1 | 132 | 87.80 ± 9.22 | 85.50 ± 14.47 | 80.54 ± 24.71 |
| 2 | 132 | 78.42 ± 5.67 | 70.42 ± 10.24 | 58.08 ± 23.25 |
| 3 | 132 | 79.79 ± 4.16 | 72.98 ± 8.04 | 72.42 ± 2.32 |

All five seeds have lower accuracy and IIA in every mixed-mask condition than
in the aligned condition. The response is **not monotonic**: three swaps recover
some performance relative to two, including a large recovery for seed 53.
Coordinate identity matters, not just the count. Raw masks, per-seed metrics,
paired differences and all three test distributions are retained in
[results/role_controls](../results/role_controls).

The aligned condition exactly matches every shared metric from the published
functional pilot. Maximum prediction/patch logit discrepancy after branch
renaming is 7.11e-15 over all conditions and splits (floating-point roundoff).

## Interpretation and limits

At fixed parameter count, training budget, patch size and mapping search budget,
the helpful inductive bias relies materially on giving each branch a correct
role boundary. Naming a branch does not change its computation. This strengthens
the existing narrow interpretation: engineered task alignment, not demonstrated
biological specificity or autonomously discovered causal structure.

This control **does not remove circularity**: task labels, supplied aligned masks
and intervention targets still share designed-in roles. Mixed-mask models are
also less accurate, so IIA losses are not isolated from predictive competence.
Equal parameter count is not equal function class, optimization difficulty or
accuracy. Only one nested mask draw is used per training seed; mask variability
and optimization variability are not independently estimated. Some coordinates
are much more informative than others, one is nuisance, and splitting nonlinear
interactions can change the representable function. The five paired differences
are descriptive, not a population estimate or corrected significance test.
No real images, learned retrieval, external datasets or biological measurements
are involved. A future stronger control would independently cross mask draws and
training seeds, then compare models at predeclared validation-accuracy bands.

## Reproduce and audit

No installation or downloads are needed with the repository's NumPy dependency
available. From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src python -m functional_circuits.role_controls \
  --output results/role-controls-local
PYTHONPATH=src python scripts/verify_run.py results/role_controls
cmp results/role_controls/metrics.json results/role-controls-local/metrics.json
cmp results/role_controls/summary.json results/role-controls-local/summary.json
cmp results/role_controls/learning_curves.json results/role-controls-local/learning_curves.json
```

The full five-seed control was rerun independently and all three JSON artifacts
matched byte for byte in the measured environment. The manifest records data and
source hashes, dependency versions and configuration. No binary checkpoints are
published for this control; training is regenerated from seeds. Output directories
must be empty or absent to prevent overwriting existing evidence. All test splits
must contain both contexts; invalid small-sample configurations are rejected
before training or creating output files. The original
pilot was also rerun before implementation: all 90 metric rows and 30 regenerated
checkpoint hashes matched its published manifest exactly.
