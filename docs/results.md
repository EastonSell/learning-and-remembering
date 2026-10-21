# Pilot results and interpretation

## Execution and reproducibility

The planned default configuration was executed unchanged: five seeds, 2,048 training examples, 512 mapping-validation examples, 2,048 examples per held-out distribution, 120 epochs, batch 128, Adam learning rate 0.01. Six architectures were trained from scratch, yielding 30 fits and 90 evaluated model/seed/distribution rows. Training and testing use distinct seeds; the intervention mapping is chosen on validation only.

Executions were repeated after metadata-only changes: recording the command and summary hashes, clarifying metric labels, and allowing intentionally omitted checkpoint files during audit; no data, training, architecture, or hyperparameter was changed. Numerical results and checkpoints matched exactly. A separate seed-11 repeat was also run. The final five-seed run used approximately 15 seconds, with one configured BLAS thread. See [verification](../results/verification.json), [manifest](../results/pilot/manifest.json), and [all tables](../results/pilot/summary.md).

## Main outcomes

Functional roles reached 97.66% accuracy on the balanced-agreement split and 97.45% memory IIA. The approximately parameter-matched monolithic model reached 91.37% and 79.60%. Paired seed differences were +6.29 percentage points in accuracy (descriptive 95% t interval ±1.24 pp) and +17.85 pp IIA (±3.00 pp). These intervals summarize five seeds; they are not multiple-comparison-corrected confirmatory inference.

The dense modular control reached 95.70% accuracy and 95.81% IIA despite receiving no role-specific input masks. Functional-minus-dense differences were +1.95 pp accuracy (±0.79 pp) and +1.64 pp IIA (±0.66 pp). Its 228 parameters exceed the functional model's 132. This control makes a specifically biological interpretation less compelling: ordinary gated modular computation already does well.

The learned functional gate aligned with the designated memory role on all evaluation examples across the five seeds. Its irrelevant-stream replacement consistency was 98.52%, compared with 89.22% for the monolithic baseline. This learned soft gate still allows some cross-stream influence; the oracle hard gate had 100% consistency by construction.

## Ablations and informative failures

- Random partitions preserve parameter count but break the supplied semantic alignment: 77.41% accuracy on the balanced-agreement split and 68.17% IIA, with substantial seed variability. Different partitions retain different amounts of task information. The outcome supports the usefulness of the correct connectivity prior within this task.
- No gate reaches 74.84% accuracy on the balanced-agreement split and 50.18% conflict accuracy. Without context, even a perfect representation has expected accuracy ceilings of 90%, 75%, and 50% on ID, balanced and conflict distributions respectively. This is an information-removal sanity check, not evidence that generic networks are inherently weak.
- Oracle routing improves accuracy on the balanced-agreement split to 98.28%, versus 97.66% for the learned functional model. Control learning still has a measurable cost.
- Monolithic changed-case IIA varies strongly by seed: 69.1% ±21.9% SD. The corresponding functional figure is 97.5% ±0.8% SD. The small sample and greedy mapping search make this an unstable quantitative comparison, not a definitive bound on what monolithic representations contain.
- Zeroing the selected memory representation reduces functional memory-context accuracy by 46.85 pp on average, while sensory-context accuracy changes by -0.41 pp in the drop convention (a small improvement). Off-manifold lesion effects require care. Activation interchange and changed-case outcomes are more informative than lesion magnitude alone.

## Why the apparent advantage is not an independent discovery

The ground-truth roles used by the intervention evaluator are also encoded in the functional input masks. Correct architecture and correct explanation therefore share a designed-in causal prior. The pilot asks whether this explicit prior yields usable, testable interfaces under a finite training budget. It does not establish that naming modules after brain functions causes general interpretability.

Predictive performance differs too. Low IIA may partly reflect a poor underlying classifier rather than an inaccessible causal representation. The monolithic mapping search evaluates 45 candidates greedily, selecting an average 3.8 of 9 hidden units; modular mappings evaluate two expert-logit choices and patch one scalar. Search budget and intervention bandwidth are not equal. Dense subspace/rotation mappings could materially change the comparison.

Memory banks are generated and exactly queried before neural inference. No model learns retrieval, persistence, interference resistance, replay, or consolidation. Distinct mechanisms and biological evidence are described in [research.md](research.md); they should not be inferred from these scores.

## Decision

The implementation and diagnostics provide a useful research foundation, and the measured result is consistent with a benefit from an explicitly correct functional decomposition. Broad claims remain open. The next decisive experiments are matched-accuracy, matched-bandwidth comparisons and the [real-image / learned-memory protocol](next_experiment.md). A small additional scale-up is justified as a test, not as presumed confirmation.
