# Research grounding and claim boundaries

Sources checked on 2026-10-21. This is a targeted foundation, not a systematic or exhaustive literature review. Paper claims below are limited to inspected author/publisher abstracts or accessible source excerpts. No paper text, figures, datasets, or third-party code are redistributed.

## Empirical biological evidence

1. **Wallis, Anderson and Miller (2001), “Single neurons in prefrontal cortex encode abstract rules.”** Monkey recordings identified rule-related prefrontal responses in a task whose rules generalized to new images. This motivates treating task context as an explicit experimental variable, not assuming that a two-parameter gate is a prefrontal cortex. The study does not establish that a rigid modular artificial network is more interpretable.
   - Publisher and abstract: https://www.nature.com/articles/35082081
   - DOI: https://doi.org/10.1038/35082081
   - Inspected: publisher abstract and bibliographic metadata.

2. **Rigotti et al. (2013), “The importance of mixed selectivity in complex cognitive tasks.”** Analysis of monkey prefrontal activity and a computational account emphasize distributed, nonlinear mixed selectivity. This is a counterweight to simplistic one-function/one-module biological stories. The present restrictive masks are an engineering prior and may discard exactly the mixed representations that help biological computation.
   - Publisher: https://www.nature.com/articles/nature12160
   - Author manuscript: https://pmc.ncbi.nlm.nih.gov/articles/PMC4412347/
   - DOI: https://doi.org/10.1038/nature12160
   - Inspected: publisher abstract and author-manuscript introduction excerpt.

## Computational neuroscience and theoretical synthesis

3. **Miller and Cohen (2001), “An Integrative Theory of Prefrontal Cortex Function.”** This theoretical review proposes that maintained goal representations bias other processing. The engineering analogy is a context-dependent gate. Our gate has no recurrent maintenance, attention circuit, anatomical correspondence, spiking dynamics, or neurophysiological validation.
   - Publisher: https://www.annualreviews.org/content/journals/10.1146/annurev.neuro.24.1.167
   - DOI: https://doi.org/10.1146/annurev.neuro.24.1.167
   - Inspected: publisher abstract; explicitly a review/theory, not a new biological experiment.

4. **McClelland, McNaughton and O'Reilly (1995), “Why There Are Complementary Learning Systems in the Hippocampus and Neocortex: Insights From the Successes and Failures of Connectionist Models of Learning and Memory.”** Their connectionist account distinguishes rapid acquisition of episodes from slower integration across experience. It motivates keeping a retrieved-evidence interface distinct from current evidence. This pilot implements neither complementary learning rates nor consolidation: its lookup is exact shared preprocessing.
   - Author-hosted paper: https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf
   - DOI: https://doi.org/10.1037/0033-295X.102.3.419
   - Inspected: author-hosted first-page abstract and source metadata.

5. **Daw, Niv and Dayan (2005), “Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control.”** This computational account proposes arbitration between controllers using uncertainty. It supports studying control over multiple computational pathways as a question. Our gate receives a context cue, not uncertainty, and is not a reproduction or test of that account.
   - Primary article abstract indexed at PubMed: https://pubmed.ncbi.nlm.nih.gov/16286932/
   - DOI: https://doi.org/10.1038/nn1560
   - Inspected: indexed original abstract.

## Direct machine-learning precedents

6. **Goyal et al., “Recurrent Independent Mechanisms” (arXiv 2019, revised 2020).** Modular recurrent mechanisms and sparse communication are established research directions; modularity is not novel here. Our models are feed-forward, with two expert heads and a small gate, not a RIMs reproduction.
   - Exact inspected version: https://arxiv.org/abs/1909.10893v6
   - Inspected: arXiv abstract and revision history.

7. **Geiger, Lu, Icard and Potts (2021), “Causal Abstractions of Neural Networks.”** This motivates evaluating a hypothesized representation with interchange interventions rather than merely reading a module name or training a decoder. We implement a small original task and validation-only mapping search; this is not their benchmark or implementation.
   - Exact inspected version: https://arxiv.org/abs/2106.02997v2
   - Inspected: arXiv abstract and bibliographic metadata.

8. **Geiger et al. (2022 revision), “Inducing Causal Structure for Interpretable Neural Networks.”** Interchange intervention training is an existing approach to inducing known causal structure. Our pilot does not use intervention training or auxiliary causal labels, but its supplied connectivity is also a strong prior. A future controlled comparison should include IIT instead of claiming a new interpretability paradigm.
   - Exact inspected version: https://arxiv.org/abs/2112.00826v2
   - Inspected: arXiv abstract and revision history.

## What this repository can establish

It can establish how six specified small architectures behave on one generated causal task, under reproducible training and evaluation. It cannot establish anatomical fidelity, neural explanation, learned episodic memory, a newly discovered learning principle, superiority on real-world tasks, or whether biology is the source of an engineering benefit. The architectural restriction and intervention target deliberately share the same factorization. That circularity is disclosed rather than treated as a scientific discovery.
