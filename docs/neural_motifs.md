# Neural motifs: evidence, analogies, and the next falsification test

Research checkpoint, 2026-10-22. **No new experiment has been run.** This targeted
review is not systematic. Source scope and access limits are recorded in the
[machine-readable bibliography](bibliography/neural_motifs.json).

## Decision

Prioritize **bounded episodic replay supporting a slower parametric learner**.
Test retention on real handwritten-digit observations, not whether a branch
patch reproduces a counterfactual defined by that same branch. This is a small
engineering test motivated by neuroscience, not a model of an anatomical region
or a new replay algorithm. Its operational details are in the
[unrun protocol](protocols/replay_digits_v1.md).

## Evidence-to-computation matrix

Each prediction below is our proposed computational test, not a result reported
by the biological paper. No motif establishes a general ML advantage by itself.

| Motif and primary evidence | Candidate module, explicitly an analogy | Falsifiable prediction and control | Evidence boundary / priority |
|---|---|---|---|
| Fast acquisition plus slower integration: [McClelland, McNaughton & O'Reilly, 1995](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf) give a computational account of interleaved learning and consolidation. Some simulations treat hippocampal storage as an assumed interface, not a mechanistic network. | A bounded store of past examples feeding an ordinary parametric learner. | At equal model size and update count, broader historical replay improves held-out retention over recent-only replay. Failure against that control rejects the proposed benefit at this scale. | Theory/simulation, not a new biological experiment; exact stored vectors and backpropagation are engineering choices. Selected. |
| Replay-associated consolidation: [Girardeau et al., 2009](https://pubmed.ncbi.nlm.nih.gov/19749750/) selectively suppressed post-training hippocampal sharp-wave ripples and impaired rat spatial-memory performance. | Interleave past examples during later learning; ablate old-example access without changing gradient budget. | Replacing historical replay with current-example repetition worsens retention. Compare FIFO too, to distinguish historical coverage from mere extra rehearsal. | Causal perturbation supports a role for ripples in that setting, not a proof of uniform reservoir sampling or supervised label replay. Selected motivation, not anatomical validation. |
| Nonlinear mixed selectivity: [Rigotti et al., 2013](https://pubmed.ncbi.nlm.nih.gov/23685452/) analyze monkey PFC recordings, population decoding and representational dimensionality; reduced dimensionality accompanies errors. | Shared nonlinear representations or communicating modules, rather than mandatory disjoint role masks. | A shared representation should handle held-out factor combinations better than equally sized disjoint branches in an appropriately controlled task; a matched generic baseline may erase any benefit. | Recording/decoding and computational analysis, not a selective causal lesion of mixed-selectivity cells. Strong counterevidence to simplistic one-role/one-module stories; defer new task. |
| Context-dependent dendritic feedback: [Fişek et al., 2023](https://www.nature.com/articles/s41586-023-06007-6) combine mouse LM–V1 connectivity mapping, calcium imaging and optogenetics; feedback effects depend on retinotopic relationships and involve local dendritic events. | A context-sensitive nonlinear feedback path into a feature layer. | With equal capacity and compute, genuine feedback should improve an occlusion/context task beyond feedforward, shuffled-context and recurrence-only controls. | Direct circuit evidence, but neither a scalar context gate nor an arbitrary two-branch MLP recreates this circuitry. More difficult to isolate cheaply; defer. |
| Abstract rule representation: [Wallis, Anderson & Miller, 2001](https://pubmed.ncbi.nlm.nih.gov/11418860/) record rule-related monkey PFC responses while same/different rules generalize to new pictures. | Learned context state modulating a shared processor. | Context modulation should transfer to novel stimulus instances and survive removal of superficial cue correlations; compare a generic recurrent network. | Recordings support rule representation, not hardwired expert boundaries or better ML interpretability. Explicit task cues can manufacture the desired separation; defer. |

## What machine learning has already demonstrated

[Chaudhry et al., 2019, v4](https://arxiv.org/html/1902.10486v4) study small
rehearsal memories, including reservoir and class-balanced strategies, in
single-pass supervised continual learning. Their experiments demonstrate useful
replay baselines; memory policy and task protocol matter. Their task descriptors
and task-specific labels must not be silently equated with task-ID-free
class-incremental testing. Our global FIFO control is also **not** their
class-balanced ring-buffer policy.

[Rolnick et al., NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html)
demonstrate CLEAR in continual reinforcement learning, combining replay with
off-policy learning and behavioral cloning. That establishes an ML precedent,
not evidence that plain supervised replay reproduces hippocampal computation.
The proposed study is an inexpensive controlled extension, not a replication of
CLEAR and not a novelty claim.

## Public data options checked

| Resource | Verified accessibility / scale | Fit and gate |
|---|---|---|
| [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) | Official metadata lists 5,620 samples, 64 integer features, 30 training writers and 13 different test writers, and CC BY 4.0. Class counts total 3,823 train / 1,797 test. | Selected: small enough for NumPy. Preserve official writer-disjoint test split; record file hashes before use. Do not pool it with scikit-learn's copy of the test subset. |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) | Official repository lists 60,000/10,000 28×28 images, 10 labels and MIT terms; compressed files total roughly 30 MB. | Useful second visual domain after a resource/replication gate. Existing delayed-memory protocol remains unrun; it does not solve circularity just by replacing synthetic features with pixels. |
| [Allen Visual Coding–Neuropixels](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html), [DANDI 000021](https://dandiarchive.org/dandiset/000021/draft) | Real mouse electrophysiology; the checked DANDI listing is hundreds of GB, tagged CC BY 4.0 and also points to Allen terms. | Future neural-response validation, not the current CPU study. Select a small immutable session/subset, resolve applicable terms and access costs first. Do not infer a blanket redistribution right from “open.” |

The [scikit-learn loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
contains 1,797 examples copied from the UCI test subset. Randomly resplitting
that convenience copy would answer a different question from the official
writer-disjoint evaluation. No benchmark files were downloaded for this memo.

## Claim discipline

The next result can support or fail an algorithmic retention hypothesis. A
positive result does not validate hippocampal anatomy, sleep replay dynamics,
consciousness, biological learning rules, or a discovered neural functional map.
A null result does not refute complementary learning systems. Direct biological
validation would require fitting and predicting independently held-out neural or
behavioral measurements, with a separate protocol and appropriate controls.
