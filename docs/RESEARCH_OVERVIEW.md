# Learning and Remembering
## Replay coverage, replay selection and supplied functional structure in small neural networks

Evidence snapshot: 22 October 2026. This overview synthesizes the linked completed studies and proposes follow-ups. Original protocols, reports and amendments retain their own dates and status. For the subsequently added JOINT diagnostic, portable reproduction and 2×2 decomposition, see the [current evidence map](EVIDENCE_MAP.md); those conditional one-step findings do not change the locked primary selection result.

## Abstract

This project investigates two ways to constrain a small neural learner: retaining selected past observations, and separating computational roles through architecture. A writer-held-out digit study found substantially better retention with history-wide reservoir memory than equally sized FIFO memory under an imposed class-blocked stream. A subsequent Fashion-MNIST experiment held memory contents and class coverage fixed while comparing replay selectors. Its primary MIR-versus-high-loss result was inconclusive, and a prespecified protection audit was adverse. Synthetic activation-interchange studies found effects of supplied role boundaries, with incomplete competence matching and an algebraically privileged diagnostic. Together, the studies separate memory coverage, selection cost and architectural alignment as distinct explanations. Generalization to natural streams, learned role discovery and biological mechanisms remain open.

## Question and motivation

A learner can fail after a distribution shift because useful observations have disappeared, because available observations are rehearsed poorly, or because new updates interfere with old computations. These explanations require different controls. Similarly, a model can appear interpretable because its architecture was built to match the experiment's intervention target.

The program therefore asks:

1. Does historical coverage improve retention at fixed buffer capacity and training budget?
2. Given matched coverage, does predicted interference select more useful replay than current loss or uniform sampling?
3. Do supplied functional boundaries support intervention behavior beyond differences in predictive competence and the mechanics of the intervention itself?

## Related primary work

- [Aljundi et al., Online Continual Learning with Maximally Interfered Retrieval](https://arxiv.org/abs/1908.04742v3) defines the virtual-update, loss-increase selector used in the Fashion-MNIST study. MIR is an existing method; this project tests it under additional coverage and compute controls.
- [Schapiro et al., Human hippocampal replay during rest prioritizes weakly learned information and predicts memory performance](https://pubmed.ncbi.nlm.nih.gov/30254219/) motivates asking how replay is prioritized. Supervised classification loss serves as a computational analogy whose biological correspondence remains untested. The [mechanism evidence map](../research/replay-mechanism-map.md) distinguishes observations, computational accounts and proposed analogies.
- [Geiger et al., Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997v2) motivates testing a representation through interchange interventions. The synthetic pilot uses an original engineered task with supplied connectivity, rather than discovering an abstraction from unconstrained behavior.

These sources frame the questions; the experiments do not reproduce all of their methods or results. The broader [research notes](research.md) and [claim matrix](claim_matrix.md) preserve the original literature scope.

## Hypotheses and design logic

**Historical coverage.** The completed digit protocol predicted a reservoir–FIFO advantage in final macro accuracy, with a fixed support rule and immediate-acquisition checks. Reservoir and FIFO both stored 100 examples. Current-only repetition matched extra presentations but intentionally had no historical memory.

**Selection beyond coverage.** The completed Fashion-MNIST protocol predicted a practical MIR advantage over high-loss selection and protection in a separate old-class-loss audit. The primary support rule required a mean gain of at least one percentage point and a paired 95% interval entirely above zero. Uniform and within-class shuffled-score controls tested whether scoring and score-to-item assignment added value. A useful-compute uniform arm spent saved selector work on extra replay.

**Role structure beyond competence.** The synthetic extensions tested mixed role masks at equal parameter count and selected approximately accuracy-matched pairs using validation data only. Matching failures were retained. The intervention diagnostic also exposed a structural identity that restricts its interpretation.

## Methods and controls

The [digit protocol](protocols/replay_digits_v1.md) uses a 2,410-parameter MLP, an official UCI train/test division, a fixed training/validation partition and ten paired evaluation seeds after development selection. Class boundaries construct the stream but are not model inputs. Acquisition and per-class retention distinguish forgetting from failure to learn.

The [Fashion-MNIST protocol](../studies/replay-selection-fashion-mnist/source/docs/protocol.md) uses a 6,634-parameter MLP, a 200-item reservoir, matched candidates and per-class replay quotas across four diagnostic arms, plus the separately selected useful-compute control. Twenty paired evaluation seeds produce 100 runs. Logical numerical work, presentation counts, runtime and memory are recorded separately; equal logical operation counts do not imply equal wall time.

The [functional pilot](results.md) compares six architectures. [Role controls](role_controls.md) equalize parameter counts, and [competence controls](competence_controls.md) lock validation-selected checkpoints before held-out evaluation. The task supplies the roles that the interventions later test.

## Measured findings

- **Digit retention:** reservoir 88.44%, FIFO 21.16%, current-only 19.70% final macro accuracy. Reservoir minus FIFO was 67.28 percentage points, with a descriptive paired-seed interval of 65.27–69.04 points. This interval is conditional on the dataset and design. [Full report](replay_results.md).
- **Fashion-MNIST primary contrast:** MIR minus high-loss was +0.6265 points, interval −0.3595 to +1.5330. Neither support condition was met. This remains inconclusive rather than an equivalence result. Uniform and useful-compute uniform had higher descriptive means; the latter changed presentations and learning rate. The separate MIR old-loss audit was adverse, with an update sequence differing from actual joint training. [All outcomes and ledgers](../studies/replay-selection-fashion-mnist/results/evaluation-report-public.md).
- **Supplied roles:** the pilot's functional model was more accurate as well as more intervention-consistent than several controls. In the later competence study, 10/15 comparisons matched on validation and only 6/10 retained test tolerances. Exactly 100% intervention accuracy on a jointly correct subset followed from the aligned architecture's identity. [Competence report](competence_controls.md).

## Reproducibility and limits

Start with [the learner guide](START_HERE.md), then the [root run commands](../README.md#reproduce). Each study separates saved-evidence verification from new training. [Checkpoint availability](checkpoints.md), the [verifier correction](audit_coverage_fix.md) and study-specific export notes document what can be reconstructed.

Seed repetitions cover initialization and stream order within fixed tasks; domain and participant variability are outside that uncertainty estimate. Exact repeats establish software reproducibility, not additional statistical evidence. Engineered streams, privileged class quotas, supplied masks and limited competence matching constrain interpretation. External dataset licenses and creator attribution remain applicable.

## Falsifiable next steps

1. Freeze a recurrent-class stream before outcomes. Compare reservoir and FIFO with equal historical capacity and updates. A disappearing coverage advantage would narrow the class-block interpretation.
2. Test whether a selection audit predicts protection under the exact joint update used by the learner, with validation kept outside state updates. Failure of that correspondence would count against the proposed protection mechanism.
3. Evaluate role interventions that are not algebraically equivalent to replacing the supplied input block, using independent replication seeds and fixed matching tolerances. Report inability to match rather than loosening the criterion.

The [delayed-memory proposal](next_experiment.md) and next experiment in the [mechanism map](../research/replay-mechanism-map.md) are prospective. No new corpus, learner or harness described as future work is a completed experiment.
