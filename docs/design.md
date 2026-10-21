# Functional Circuits: design written before implementation

## Question and narrow hypothesis
Can explicitly separating **current evidence**, **retrieved evidence**, and **context-dependent control** make a small neural classifier easier to explain with causal interventions, while retaining task performance? These functional roles are loosely motivated by neuroscience. This is an engineering experiment, not a model of brain anatomy, a claim of biological fidelity, or a claim of novelty.

The first study deliberately gives the architecture the correct factorization. It tests the value and limits of that prior, not whether modules autonomously discover brain-like organization. Success does not establish that biological inspiration beats ordinary modular design.

## Task and causal graph
Generate two independent six-dimensional evidence streams conditional on balanced latent binary categories. One stream is current sensory evidence. The other is retrieved from an episode-local key-value array using an exact, shared preprocessing lookup. A binary context selects which category is the label. Retrieval is identical for every model; memory is not learned and no claim about hippocampal learning is possible.

Each category is the sign of a fixed nonlinear function of its six-dimensional stream. The two categories agree with probability 0.8 during training and ID evaluation. A balanced-agreement distribution (0.5) and a conflicting-evidence distribution (0.0) test reliance on the irrelevant stream. Independent random seeds produce train, validation, and each test split. No held-out labels guide training or mapping selection.

## Models and controls
- Functional: two learned tanh MLP experts, each restricted to its role-specific stream, combined by a learned sigmoid gate of context. No auxiliary role labels or gate supervision.
- Monolithic: one tanh MLP seeing both streams and context, approximately parameter-matched.
- Random partition: same architecture, parameter count, and cue gate as Functional, but semantic stream coordinates are mixed across experts. Tests the importance of the supplied functional decomposition.
- Dense modular: both experts see all evidence. Larger capacity is reported, not hidden. Tests whether a gate alone suffices.
- No gate: same role-specific experts averaged equally, deliberately removing context selection. This is an information-removal ablation, not a fair predictive baseline.
- Oracle gate: correct context routing is supplied. This is a mechanistic positive control, not a learned competitor.

All trainable models use the same cross-entropy objective, optimizer, minibatch order, steps, and data for each seed. No pretrained model or external dataset is used.

## Measurement
Primary outcomes: held-out accuracy and memory interchange intervention accuracy (IIA) on independent balanced data. IIA patches a donor representation into a base forward pass and compares the output to a known high-level counterfactual: donor memory category when context selects memory, otherwise base sensory category. Also report IIA restricted to cases where replacing memory actually changes the high-level answer, preventing high scores through unchanged examples.

For modular models, choose which of the two expert logits represents memory using validation-only IIA. For the monolithic model, use a validation-only greedy hidden-unit subset search, retaining its best prefix. Report mapping search counts and patch size. Different architectural granularity is a confound: this is a transparent small benchmark, not a calibrated universal interpretability ranking.

Secondary diagnostics: lesion-by-context accuracy drops, irrelevant-stream input replacement consistency, learned routing alignment, train loss, parameter count, wall time, and paired seed differences. Zero-lesion effects are not by themselves causal proof because off-manifold interventions can be misleading. Counterfactual interventions are more diagnostic but still conditional on the supplied causal model.

## Resource and integrity limits
CPU NumPy only; one BLAS thread; five seeds; no installs, downloads, paid resources, external writes, publication, or private data. Initial environment: Python 3.12.14, NumPy 2.3.5; PyTorch and pytest absent. About 9.7 GiB RAM and 28 GiB disk free at inspection. Use standard-library unittest. Original code and synthetic outputs are MIT licensed. Cited publications are linked, not redistributed. A manifest records versions, hyperparameters, seed schedules, file hashes, elapsed time and exact command.

## Decision rule and limits
Report all models and seeds, including negative findings. A useful pilot shows interpretable intervention behavior with competitive accuracy, rather than interpreting module names or a wiring diagram as evidence. Five-seed intervals are descriptive, not confirmatory statistical evidence. This toy problem bakes in functional roles, uses easy retrieval, and is not language modeling, neuroscience, AGI, or proof of broad robustness.
