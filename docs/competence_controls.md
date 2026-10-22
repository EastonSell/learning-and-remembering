# Validation-matched competence study

Fresh extension of public commit `04151cfb4948bc3d489882d2e60c4a3bfe06e235`,
2026-10-22. Previous source and experiment results are preserved.

## Question and prerecording

Do intervention-accuracy differences persist when prediction accuracy is
approximately comparable, without choosing models using test outcomes?

The [protocol](protocols/competence_controls_v1.json) was written before the
experiment was executed. Its SHA-256 is
`f94fc0c6d8f05e89d31834abc67ba3466950d05d13e1a5cf3cd6aa139330c914`.
This is a local prerecord, not an independent registry or externally timestamped
preregistration. Results did not trigger changes to the selection rules.

Five fresh seeds (101, 127, 149, 173, 197) use separate generated training,
selection, mapping and test splits. Every model has 132 parameters. Each aligned
or mixed-mask model is trained from the same initialization to each epoch count
in the fixed grid 1, 2, 4, 8, 16, 32, 64, 120. Candidate training restarts from
initialization, so the optimizer state is not accidentally reset mid-trajectory.

For each aligned-versus-mixed comparison, prediction-only selection requires:

- At least 70% overall validation accuracy for both candidates.
- At most 2 percentage points difference in overall validation accuracy.
- At most 4 percentage points difference in either context's validation accuracy.
- Among eligible pairs, maximize the smaller overall accuracy, then minimize
  the overall gap, then prefer the smaller aligned and mixed epoch counts.

Every candidate score and selected epoch pair is saved before any test examples
are generated. The selections file is hashed and checked unchanged after test
evaluation. Model weights are kept in memory, not persisted as checkpoints.
An independent mapping split then chooses between the same two branch patches.
Held-out evaluation uses 8,192 examples per seed. Unmatched comparisons remain
explicitly unmatched, and **all validation-selected pairs are reported** even
when test accuracy matching fails. No test-based model selection or deletion is
performed. Selection/mapping data have 2,048/1,024 examples respectively.

## Results: matching is incomplete

All gaps below are aligned minus mixed, in percentage points. Means use every
validation-selected pair, not just pairs with transferred test matching.

| Swaps per branch | Validation matched / attempted | Test matching transferred / selected | Mean test accuracy gap | Mean test memory-IIA gap |
|---|---:|---:|---:|---:|
| 1 | 4 / 5 | 3 / 4 | -0.07 | +3.44 |
| 2 | 3 / 5 | 1 / 3 | +0.02 | +7.65 |
| 3 | 3 / 5 | 2 / 3 | +1.00 | +8.78 |

Only 10 of 15 attempted pairs match on validation. Only 6 of those 10 retain
the overall and per-context tolerances on test. Mean accuracy gaps can conceal
opposing per-seed and per-context mismatches; they do not establish equivalence.
The selected aligned models are often deliberately undertrained (2 epochs),
whereas their mixed comparators can receive up to 120 epochs. This is a
matched-competence experiment, not a matched-training-duration comparison.

Raw paired IIA gaps, ordered by matched seed:

- 1 swap: +10.89, -0.73, +1.60, +1.99 points.
- 2 swaps: +8.12, +14.01, +0.83 points.
- 3 swaps: +15.41, +5.70, +5.25 points.

These are descriptive results with few, dependent comparisons and no formal
significance test. In particular, the one-swap advantage is not universal.
See [all candidate scores](../results/competence_controls/candidates.json),
[locked selections](../results/competence_controls/selections.json),
[every attempted pair](../results/competence_controls/metrics.json), and
[summary](../results/competence_controls/summary.json).

## Secondary diagnostic exposes a structural identity

The protocol also reports IIA on the shared test subset where *both* models are
correct on the factual input and on an explicit replacement of the retrieved
input block. Coverage ranges from 3,624 to 7,133 of 8,192 examples; changed-case
coverage is reported separately. Empty subsets produce null, never fake zeroes.

The aligned model has exactly 100% IIA on every such subset. This is **not new
independent evidence for causal interpretability**: with its memory branch
selected, replacing that branch's activation equals replacing the memory input
block algebraically. The subset already requires the latter prediction to be
correct. A regression test verifies this identity; it need not hold for mixed
input masks. The diagnostic therefore makes an existing architectural advantage
explicit rather than solving architecture-label circularity. Conditioning on
joint correctness can also introduce selection bias and is not a causal
adjustment for predictive competence.

## What this establishes, and what it does not

The validation-only procedure can find approximately prediction-matched models
in some cases, and activation-intervention differences remain in the reported
pairs. But incomplete validation matching, imperfect transfer, different error
structures, undertraining, small sample of seeds, and engineered input/patch
alignment prevent a clean claim that IIA gains are independent of competence.
The honest result is **partial matching plus evidence of structural dependence**,
not a general interpretability advantage or biological specificity.

A stronger future study would prespecify denser checkpoint grids and independent
replication seeds, retain adequate per-context validation precision, and test
interventions not algebraically privileged by the supplied masks. It should
report matching failures rather than relax tolerances after observing outcomes.
No learned retrieval, real images or biological measurements were used here.

## Reproduce and verify

With the existing NumPy dependency, no installs or downloads are needed:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src python -m functional_circuits.competence_controls \
  --output results/competence-controls-local
PYTHONPATH=src python scripts/verify_run.py results/competence_controls
for file in candidates selections metrics summary; do
  cmp "results/competence_controls/$file.json" \
      "results/competence-controls-local/$file.json"
done
```

The full study was repeated and the four JSON artifacts matched byte for byte.
Manifests record the protocol, selection lock, source and data hashes. Existing
nonempty output directories are protected. Tiny custom protocols are useful for
software tests, not substitutes for the published study. Invalid splits without
both contexts raise an error; an interrupted run can leave partial files, which
must be preserved or moved before retrying into a fresh output directory.
