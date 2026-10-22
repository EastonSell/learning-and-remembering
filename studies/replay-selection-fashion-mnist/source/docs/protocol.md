# Prospective protocol: matched-coverage replay selection

Status: **design only; implementation and execution not authorized by this document**. Freeze a version and its hash before development. Any amendment after validation must be dated; never call a post-test amendment prospective.

## 1. Data and split gate

1. Verify original Fashion-MNIST source, pinned revision, MIT notice, HTTPS assets, dimensions and hashes. Do not fetch pretrained models or install a framework.
2. Confirm that no prior project run has scored this official test for method selection. If it has, stop the confirmatory path and choose a new evaluation resource under a revised protocol. The previously inspected Optdigits official test is explicitly ineligible for selecting this design.
3. Using NumPy PCG64 seed 73102, independently permute official training indices within each label, labels in ascending order. Allocate the first 1,000/class to training and next 500/class to validation: 10,000 training, 5,000 validation. Ignore the remainder. Save actual index lists, library version and hashes; an algorithm name/seed alone is insufficient provenance.
4. Preprocess each 28×28 uint8 image into a 14×14 float32 image by averaging each non-overlapping 2×2 block, then divide by 255. No fitted normalization. Train/validation duplicates must be audited before training. If any are found, stop and amend the split using feature-grouped allocation before development; do not silently train on leaked validation images. Before unlocking official-test scoring, audit exact raw and transformed cross-split duplicates; report counts. Exclude test items whose raw OR transformed features exactly match training/validation by a predeclared feature-hash rule, with no label-dependent filtering. Report both deduplicated primary and full-official descriptive metrics. If retained counts fall below 500/class, stop the confirmatory path rather than changing the rule after scoring. No claim of writer/person-disjointness.
5. Test files may be integrity-checked without model predictions. Neither scores nor per-item errors enter tuning. Log first test scoring timestamp and require the complete lock first.

## 2. Common learner and stream

- NumPy MLP: 196 inputs → 32 ReLU units → 10 softmax outputs, biases included, **6,634 trainable scalars**, float32. Stable cross-entropy; ordinary SGD, no momentum/weight decay, no early stopping, no augmentation.
- He-normal weights (standard deviation sqrt(2/fan-in)), zero biases. All ten labels compete throughout. No task heads or class masks at inference.
- Five class-pair blocks. Each run uses its seeded uniform permutation of 0–9, consecutive pairs defining blocks; shuffle examples within each block. Seeded stream, initialization, reservoir admission, candidate draws and score-shuffle randomness use separate named RNG streams. Persist realized stream/order hashes. Pair all arms within seed.
- Fresh minibatch 16, with exactly 2,000 examples/block, hence 125 batches/block. Four optimizer updates per fresh batch, each using those 16 plus ten additional presentations. This is a repeated-presentation streaming experiment, **not single-pass SGD**. Total: 2,500 committed updates, 40,000 fresh-example presentations, 25,000 additional presentations/arm. Count unique observations separately: 10,000.
- A shared-policy uniform reservoir holds at most 200 transformed examples. Insert the 16 newly observed examples only AFTER their four updates, with a standard sequential reservoir admission counter. Replay never changes admissions. Paired admission randomness yields identical resident example IDs across arms, regardless of their parameter states.
- If the buffer is empty (first fresh batch), fill ten extra slots by a deterministic cycle over the current batch; classify those 40 first-batch extras as current repetition, not historical replay. With the stated schedule, historical presentations then total 24,960. No subsequent historical access outside the reservoir. Data loaded by the evaluator must be inaccessible to the learner.

## 3. Selector intervention and simpler controls

Before EACH committed update, form the same candidate-ID set across paired arms:

1. Draw up to 50 distinct buffer IDs using a class-balanced round-robin allocation: cycle through resident labels in increasing order, rotate the starting label by global update index modulo resident-label count, and allocate one candidate per visit until 50 or all resident IDs are allocated. Skip exhausted classes. Draw each class's allocated IDs uniformly without replacement.
2. Allocate ten replay slots by the same rotating label cycle, subject to candidate counts, skipping exhausted classes. Thus replay label counts, candidate IDs and selection opportunity are matched across arms, and no replacement is needed once the reservoir is nonempty. Quotas use observed training labels, a privileged engineering convenience shared by ALL arms, not a biological claim.
3. Each arm computes its own virtual parameters theta_v = theta − eta × gradient(mean fresh-batch loss). Do not commit this update. For each candidate i compute current loss L_i and loss at theta_v; delta_i = L_i(theta_v) − L_i(theta).
4. Select the quota within each class, break score ties by increasing stored source index:
   - **MIR:** largest delta_i (first MIR score, not historical-best-loss variant).
   - **High-loss:** largest current L_i. Simple weakness/hard-example control.
   - **Uniform:** uniformly random candidates within each class's quota.
   - **Shuffled MIR:** independently permute delta values among candidate IDs WITHIN class, then select largest assigned score. Preserves score distribution and quotas while destroying score-to-item alignment.
Keep signed delta values: no clipping at zero. If all values are zero, use the same source-index tie rule; if all are negative, choose the least-negative values. Record zero/negative fractions and tie frequency. These cases still consume the same replay quota and compute. Nonfinite losses invalidate the run rather than triggering a favorable fallback.

5. Commit ONE SGD update using mean cross-entropy on the concatenated 16 fresh and ten selected observations. Discard virtual parameters and scores. No gradients through selection, persistent priorities, loss caches, extra learner or stored logits.

All four diagnostic arms execute the SAME virtual gradient and two candidate forward passes, even when a score is unused. This deliberately equalizes selection workload; it is not an efficient implementation claim. Scores differ because learners diverge, which is intended. Compare native efficient cost separately.

## 4. Useful-compute control and cost ledger

Add a fifth arm, **uniform-extra-replay**, with the same reservoir and learner. It omits all candidate scoring and virtual gradients. After its usual fresh-plus-replay update, spend the unused operation budget on additional ten-example uniform class-balanced replay-only SGD updates. Never advance fresh data or reservoir admissions during these extras. Use existing reservoir labels/IDs; no 50-candidate constraint is needed for this stronger practical control. Before any test run, freeze an integer operation-counting convention for every matrix multiply, activation, softmax, gradient and update; compute saved budget from the exact diagnostic-arm graph and charge every extra operation. Carry residual budget to subsequent updates; never exceed cumulative diagnostic budget. First-batch extras use only available current observations. Record residual unspent budget and exact extra updates. This is compute-capped, not update- or presentation-matched. It tests useful allocation of a budget, not the isolated selector intervention.

Publish separately for every arm:

| Resource | Required accounting |
|---|---|
| Trainable state | 6,634 float32 weights/biases = 26,536 raw bytes. No extra trainable selector. |
| Persistent history | 200 × (196 float32 features + uint8 label + uint64 source index) = **158,600 raw bytes**, plus measured arrays/objects, reservoir counter and RNG states. This exceeds parameter bytes; do not hide it behind parameter matching. |
| Other persistent learner state | No momentum/cache by design. Count any implementation-added arrays, class-index lists and RNG state. Audit logs are not usable memory and must remain write-only to the learner. |
| Transient state | Virtual parameter copy at least 26,536 bytes, gradients, candidate activations, minibatches and sorting buffers. Report maximum allocated bytes and process RSS, not just model size. |
| Computation | Fresh and replay presentations; committed and virtual gradients separately; candidate forward evaluations; operation count; elapsed time; training/scoring/I/O times separately. Candidate scoring sees stored data and counts as replay-related computation even without a gradient. |
| Native efficiency | Also report theoretical/practically measured cost if uniform/high-loss omit unnecessary diagnostic computations. Never describe padded equal-cost arms as their fastest implementations. |

Use <=2 CPU threads, <=1 GiB peak RSS, <=250 MiB output artifacts, <=15 minutes for feasibility and <=2 CPU-hours forecast total. These are gates, not measured runtimes. If forecast exceeds limits, stop before final runs and revise the protocol using development-only evidence. No silent GPU, paid compute, or budget expansion.

## 5. Staged reproducibility and selection

**A. Software checks, no official test predictions.** Validate finite-difference gradients, virtual-update nonmutation, scoring sign, class quotas, reservoir stream identity, first-batch handling, byte ledger, operation cap, and determinism. Add fixtures where current loss and future loss increase rank different items. Test that an empty/incomplete result bundle fails verification. Do not run research training as part of this proposal.

**B. Development.** Seeds 1001 and 1002 only; eta grid {0.03, 0.10, 0.30}; all five arms, identical search budget. Select one common eta for the four diagnostic arms by mean final validation macro accuracy pooled equally over those arms/seeds; ties choose smaller eta. Separately select the fifth arm's eta from the same three trials/two seeds. Report all candidate outcomes. This common-rate diagnostic comparison is not a claim of independently optimized best performance. Poor acquisition is a reason to revise before testing, never to silently remove bad seeds.

**C. Lock.** Hash protocol, source, tests, data/index manifests, selected rates, architecture, stopping criteria, seed list and cost convention. No official-test results yet. Record any development amendments and resource extrapolation. Freeze evaluation seed list 2001–2020; start all models fresh. Save every block-boundary checkpoint before scoring.

**D. Evaluation.** Twenty paired seeds across five arms, no outcome-based restarts. Score locked checkpoints on the deduplicated official test after all configurations are fixed. Report all seeds, exceptions and full 5×5 block accuracy matrices. A failed run makes the bundle incomplete; fix reproducible software bugs under a dated amendment and rerun the complete affected paired comparison, never select favorable surviving seeds. Numerical reruns establish reproducibility, not additional samples.

## 6. Outcomes, intervention audit and decisions

Primary endpoint: final ten-class macro accuracy, MIR minus high-loss. Report mean paired difference, a 95% percentile paired-seed bootstrap interval (10,000 resamples, PCG64 seed 73103), and all individual differences. Prespecified support threshold: mean >=1 percentage point AND interval lower bound >0. This is one primary comparison. Comparisons against uniform, shuffled MIR and compute-capped uniform are secondary; report intervals as descriptive, not additional independent confirmatory wins.

Report immediate acquisition, mean forgetting (first four blocks' best boundary accuracy minus final accuracy, no clipping), per-class recall, memory class coverage and replay-selection frequency. If MIR's mean immediate acquisition is >2 points below high-loss, flag a plasticity trade-off and do not present retention alone as general improvement.

At every 25th update AFTER the first block, perform a **separate, read-only mechanism audit**: freeze current weights and fresh batch, evaluate loss change under the virtual fresh-only update on ALL currently stored examples, and report candidate score rank versus this exhaustive virtual harm, selected versus eligible mean harm, and overlap of MIR/high-loss selections. Exhaustive harm is the algorithm's own criterion, so it is an implementation/coverage check, not independent mechanistic confirmation. For a less circular outcome, clone weights into audit-only branches, compare the resulting ten-example selected-replay update versus within-class shuffled replay after the SAME fresh update, and measure old-class loss on 100 deterministic validation images (ten per class) never supplied to training. These sequential fresh-then-replay audit branches differ from the joint minibatch learner, so their effects are a diagnostic, not an exact decomposition of its updates. Report branch differences descriptively and separately from deployment metrics; this audit uses extra compute and external validation access, is not charged as learner memory, and cannot feed back into training/tuning after the lock. All arms receive identical audit opportunities and auditing costs are disclosed separately.

Intervals quantify seed/order randomness on one fixed dataset, not uncertainty over human subjects or visual domains. Twenty seeds are a bounded design choice, not a proven power calculation. No neural mechanism or novel algorithm claim follows. A robust positive outcome justifies considering a second dataset or a separately designed neural-behavioral test; a null/negative outcome is publishable and should prevent premature scaling.
