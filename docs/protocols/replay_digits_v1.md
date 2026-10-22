# Replay on writer-held-out digits: pre-run protocol v1

Status: **design checkpoint, unrun; implementation and scientific review required
before execution**. Date: 2026-10-22. No dataset examples or experimental outcomes have been inspected for
this experiment; only public metadata was reviewed. This is a prospective protocol, not a registered study.
Related evidence: [neural motifs memo](../neural_motifs.md).

## Hypothesis and scope

A fixed-capacity, history-wide rehearsal buffer reduces forgetting relative to
an equally sized recent-history buffer at identical network capacity and update
count. Primary contrast: reservoir replay minus global FIFO replay on final
10-class macro accuracy. Current-only repetition is a compute-matched secondary
control. This tests historical coverage as an engineering mechanism inspired by
complementary learning/replay accounts, not a unique biological mechanism.

Unlike previous studies here, targets are external digit annotations. No feature
role masks, task-ID input, separate task heads, oracle retrieval, activation
mapping search or IIA target is used. The class-blocked stream is still a
researcher-imposed distribution shift, not a natural biological experience.

## Data contract and leakage checks

- Source: the download linked by the [official UCI dataset page](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), DOI
  [10.24432/C50P49](https://doi.org/10.24432/C50P49), CC BY 4.0. Record the resolved
  HTTPS URL, access date, license notice, byte count and SHA-256 before parsing.
  Do not redistribute dataset bytes in this repository; cite creators E.
  Alpaydin and C. Kaynak. No third-party loader/package installation is needed.
- Parse only `optdigits.tra` and `optdigits.tes`: require 3,823/1,797 rows,
  64 integer features in [0,16], labels 0–9, no missing/nonfinite values. Reject
  unexpected archive paths or size over 10 MB compressed / 50 MB expanded.
- Keep the official 1,797-example test split out of fitting and model evaluation
  until configuration is frozen. Preflight format/hash/duplicate auditing is
  permitted, but cannot be used for tuning or model selection. The provider reports different writers in train and test; writer IDs
  are not assumed available per row. This does not imply writer-disjoint
  validation within the training file.
- On the training file, use NumPy `default_rng(24601)` to permute row indices
  separately for labels 0 through 9, in that order. Reserve the first
  `floor(0.20 * class_count)` of each permutation for validation; the rest train.
  Save IDs and hashes. Divide features by 16; no learned normalization.
- Audit exact feature duplicates within and across splits before optimization.
  Stop and document any cross-split duplicate or conflicting duplicate label;
  amend the data policy prospectively rather than silently removing test rows.
  Repeated same-label rows within training are retained and reported.
- Validation and test IDs must never enter a training/replay buffer. Store IDs
  alongside entries and test this invariant. Do not substitute the 1,797-row
  scikit-learn convenience dataset for the official training file.

## Model, arms and equal-compute contract

All arms share a 64→32 tanh→10 softmax MLP: 2,410 trainable parameters. Use
float64 weights, zero biases, Glorot-uniform weights, cross-entropy, SGD without
momentum/weight decay, and no dropout. Fixed common learning rate is selected
below. Identical initial weights and stream order are paired within each seed.

1. **Reservoir:** 100-example buffer, updated with standard sequential reservoir
   sampling. At the nth observed example, append if n≤100; otherwise draw an
   integer uniformly from [0,n), replacing that slot only if it is below 100.
2. **FIFO:** 100-example global chronological ring buffer, replacing the oldest
   entry. No per-class quotas and no task labels/boundary resets.
3. **Current-only repeat:** no persistent historical examples; the additional
   samples come from the current minibatch. This controls extra example
   presentations/gradient work, but intentionally does not match memory usage.

Fresh minibatches contain 16 examples (last minibatch may be smaller). Perform
exactly four SGD updates per fresh minibatch in every arm. Each update combines
m fresh examples with m replay examples sampled uniformly **with replacement**
from the pre-batch buffer. With an empty buffer, sample from the current batch
instead. Current-only always uses that fallback. Resample replay each update;
all losses are averaged over 2m examples. Only after all four updates insert the
fresh examples into the buffer, in stream order. Every arm therefore processes
the same number of fresh and additional training examples, with identical
network operations and SGD-step count. Report realized counts, wall time and
buffer/optimizer bytes; do not call memory costs matched for current-only.

## Stream, tuning and freeze

- Each run permutes the ten labels into five consecutive two-class blocks.
  Traverse each training example exactly once as a fresh observation, shuffling
  row IDs within each block. Replay/repetition creates additional presentations
  explicitly counted above. Block labels/boundaries are for stream construction
  and scoring only, never model input or buffer allocation.
- Development seeds: 3 and 7. Candidate common learning rates: 0.03, 0.1, 0.3.
  Run all three arms under every candidate with the same budget. Select one
  common learning rate maximizing mean final validation macro accuracy across
  both seeds and all arms; tie within 1e-12 favors the smaller rate. This shared
  compromise is not a separately optimized best-case result for each method.
- Evaluation seeds fixed in advance: 211, 223, 227, 229, 233, 239, 241, 251,
  257, 263. Each controls a new class/block order, example order, initialization
  and replay realization, paired across methods. Use independent RNG streams
  created by `SeedSequence([seed, stream_id])`: 0 label order, 1 within-block
  order, 2 initialization, 3 buffer admission, 4 replay sampling. No extra seed
  selection after seeing results.
- Before loading test examples for model evaluation, save chosen learning rate,
  all development scores, model/code/dependency versions, protocol SHA-256,
  split IDs/hashes and evaluation seed list. Then run all 30 evaluation models
  from scratch on the fixed training subset. No validation/test early stopping.
  Test inference cannot change weights, buffers, RNG state or selections.

## Outcomes and uncertainty

Primary: final macro-average recall across ten classes on the official test set,
paired reservoir minus FIFO. Report each seed, mean difference in percentage
points and a descriptive 95% percentile bootstrap interval from 10,000 paired
seed resamples with RNG seed 9801. The seed, not the image, is the resampling
unit; uncertainty is conditional on this one dataset/split and algorithm setup.

Secondary, all reported regardless of sign:

- Micro accuracy; per-class recall; reservoir minus current-only final macro
  accuracy; complete test accuracy matrices after each block (snapshots are
  evaluated only after the configuration lock).
- Forgetting for each of the first four blocks: maximum accuracy on that block
  from its learning boundary through the penultimate boundary, minus final
  accuracy; average those four differences. Do not clip negatives. Also report
  each block's immediate post-learning accuracy so poor acquisition is not
  mistaken for low forgetting. All predictions always compete over 10 labels.
- Memory class coverage, entry ages and replay counts over time; wall time,
  peak RSS and exact training-example/SGD-step counts. Buffer contents are
  observed training examples, not learned episodic representations.

A retained-history benefit is supported *for this setup* only if primary mean
improvement is at least 2 points and the descriptive interval excludes zero.
Otherwise call it small, inconclusive or negative as appropriate, not proof of
no effect. Secondary comparisons are exploratory, without multiplicity-adjusted
claims. Report if acquisition collapses (mean immediate block accuracy below
50% for any arm); do not add epochs or change the rate after test inspection.
Do not select “successful” seeds, classes or checkpoints for the headline.

## Verification and resource gates

Before data use: numerical gradient checks for every MLP parameter; softmax
stability; reservoir inclusion-frequency sanity test with a predeclared loose
bound; exact FIFO order; empty-buffer and partial-minibatch handling; same
initialization/step/presentation counts; replay IDs exclude holdouts; hand-worked
forgetting/macro metrics; deterministic two-run comparison; output overwrite
protection. Review the implementation against this protocol before running.

NumPy is already available; no GPU or model downloads are required. Proposed
caps, not measured forecasts: two CPU threads, 1 GiB peak RSS, 250 MB artifacts,
15 minutes for development and 30 minutes for evaluation. Check resource use
on development only. If infeasible or data checks fail, stop before test,
report the blocker and version a revised protocol. A test-stage resource
failure is reported as incomplete, not a reason to change study parameters.

## Interpretation gate

Positive replay results would establish retention from data reuse, a known ML
technique, with historical coverage isolated more carefully than current-only
baselines. They cannot attribute a gain uniquely to biological inspiration or
validate hippocampal/neocortical anatomy. Null results would reject the chosen
computational implementation/scale, not complementary learning theory. Advancing
to neural recordings or more realistic memory requires a separate design.
