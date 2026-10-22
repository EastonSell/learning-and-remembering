# Which memories deserve replay?

Prospective research proposal, 2026-10-22. **No experiment implemented or executed.**

## Decision

Test whether **predicted interference**, rather than simple weakness or class coverage, is a useful signal for allocating limited replay. Keep the stored examples identical across policies. Compare a small, established maximally-interfered-retrieval (MIR) rule with high-loss, uniform and shuffled-score controls under matched class quotas. This is an original controlled study design, **not a new replay algorithm**.

The completed Optdigits experiment establishes that historical coverage matters in its deliberately class-blocked setting: reservoir replay reached 88.44% final macro accuracy, versus 21.16% for FIFO and 19.70% for current-only. It does not show that replay should be biologically prioritized. Its official test has already been inspected and must not select this experiment's methods. Also, its update counts were 772 **or 776**, seed dependent, matched within each seed; 772 is not a universal run count.

## Biological motif and falsifiable claim

Schapiro et al. (2018) measured human hippocampal item reactivation with fMRI after learning. Weaker prior memories were replayed more; later replay predicted subsequent memory. This motivates studying **which** memories are rehearsed, but does not identify the selection algorithm. Mattar and Daw (2018) offer a normative reward-based theory in which replay utility reflects gain and future need. Neither source establishes that supervised loss change is the brain's utility signal.

Operational hypothesis: at fixed stored history, replay labels, optimizer updates and selection workload, selecting items whose loss would increase after the incoming update gives at least a 1-percentage-point final macro-accuracy improvement over selecting the currently highest-loss items. A within-class score-permutation control tests whether item-specific score alignment matters beyond class exposure.

MIR is the explicit ML precedent, not a newly discovered biological mechanism. Current loss is a cheaper weakness proxy, not a faithful implementation of human memory strength. Their disagreement creates an informative test: an already weak item need not be the one most threatened by the next update.

## What would change our conclusion?

- MIR beats high-loss and shuffled-score controls: anticipated interference has incremental algorithmic value in this task, beyond those proxies and matched class exposure.
- High-loss matches or beats MIR: the more elaborate anticipatory mechanism is unnecessary here. Prioritize the cheaper explanation.
- Uniform matches both: history and class coverage explain the benefit; detailed ranking is unsupported here.
- MIR beats controls only with equal update counts, but loses to uniform given equal useful compute: ranking can help per update without being compute-efficient.
- Scores do not rank counterfactual harm better than shuffled scores on an independent audit subset: the intended selection mechanism failed, even if a final-accuracy difference remains.

A confidence interval including zero is inconclusive, not proof of equivalence. If its upper end is below the prespecified 1-point practical threshold, that threshold is unsupported for this setup. No outcome refutes complementary learning systems generally.

## Scope and deliverables

Use a small class-incremental Fashion-MNIST subset with a fresh, locked official-test evaluation. The same tiny MLP and uniform reservoir are used everywhere. There are no named brain-region layers, task-ID inputs, auxiliary neural labels, learned retrieval networks or extra trainable parameters. See [protocol.md](protocol.md) for staging, leakage safeguards, exact selector rules and cost accounting; [sources.md](sources.md) records four newly inspected primary papers and evidence limits.

The study can causally identify consequences of changing the artificial selector through controlled intervention. It cannot establish human hippocampal implementation, sleep consolidation, a layer-to-region mapping, or general mechanistic interpretability. Counterfactual loss auditing explains this selector, not the semantic responsibilities of all hidden units.

Evidence about human mechanisms would require a separate preregistered task with independent neural/behavioral measurements, predictions distinguishing candidate models on held-out participants, nuisance controls for difficulty/reward/frequency, and ideally an appropriately approved selective perturbation. Agreement between a classifier and a published group-level trend is only a weak analogy, not neural validation.
