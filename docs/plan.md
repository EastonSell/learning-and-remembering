# Implementation and evaluation plan

Written before implementation on 2026-10-21.

1. Verify primary sources and classify empirical biology, computational theory, and machine-learning methods separately. Record exact URLs and only claims supported by the inspected text.
2. Implement deterministic episode generation and shared exact retrieval; test labels, retrieval, split independence, agreement distributions, and counterfactual targets.
3. Implement small NumPy MLPs and functional/generic modular variants with analytic gradients. Verify gradients by finite differences, including learned gate gradients.
4. Implement deterministic Adam training with shared schedules, no test tuning, stable binary cross-entropy, and finite-value checks.
5. Implement validation-only representation mapping, activation patching, lesions, and irrelevant-stream substitutions. Unit test intervention semantics and self-patching invariance.
6. Run a smoke test, then freeze the five-seed configuration and execute all six architectures. Preserve raw measurements, summaries, model checkpoints, split hashes, and run manifest.
7. Re-run an independent seed/configuration to verify determinism. Run all tests and check artifact completeness. Report the full results, uncertainty, caveats and failures without selecting a best seed.
8. Deliver README, research background, data/model provenance, MIT license, reproducibility instructions, results, and a staged roadmap. Do not publish remotely.

## Planned default experiment
Train 2,048 examples; validation 512; each test set 2,048; batch 128; 120 epochs; Adam learning rate 0.01; tanh hidden width 8 per modular expert or 9 in the monolithic model; seeds 11, 23, 37, 53, 71. CPU single-threaded. These are modest engineering choices, not tuned optimal settings. Any deviations must be documented in the results.

## Next-scale stages, not executed here
A. Replicate with 20+ seeds, held-out task functions, correlation sweeps and parameter/compute-matched generic mixture-of-experts controls. Predefine an accuracy-equivalence band before interpreting IIA comparisons.
B. Add learned recurrent episodic storage, distractor delay, cue noise, compositional task switches, and distributed function assignments. Match retrieval access and recurrent compute across models.
C. Compare causal abstraction mappings at equal search budgets and intervention bandwidth; include learned rotations, distributed subspaces, rescue interventions, and held-out intervention types.
D. Only after strong toy evidence, evaluate small pretrained sequence models with separately verified licenses, explicit resource budgets, and model/data cards. No model download is part of this pilot.
