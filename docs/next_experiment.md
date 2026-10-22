# Next experiment: image evidence and learned delayed memory

Status: protocol only, not executed. This is the next falsification opportunity, not a promise that the pilot result will transfer.

## Data and rights

Use Fashion-MNIST's real product-image inputs: official train/test split, 28×28 grayscale, 10 categories. Sources: [dataset paper, exact v2](https://arxiv.org/abs/1708.07747v2), [official repository](https://github.com/zalandoresearch/fashion-mnist), [MIT license, Zalando SE 2017](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE). The official repository lists approximately 30.4 MB of compressed files. This repository has not downloaded or redistributed them.

**Status clarification (2026-10-22):** The no-download/no-redistribution sentence above records the historical status when this proposal was written, not the repository's current data holdings. The separate, completed [Fashion-MNIST replay-selection study and public export](../studies/replay-selection-fashion-mnist/README.md#get-and-verify-the-evidence) includes official test image/label assets with their original MIT notice; raw training downloads are omitted from that export. This delayed-memory proposal remains protocol only and has not been executed.

Before download, resolve and record an immutable repository commit and exact HTTPS asset URLs. Retain the upstream copyright/license notice, verify official published MD5 values for consistency and record new SHA-256 hashes. MD5 is used only to compare the upstream manifest, not as a security guarantee. Stop if content, license, or file sizes differ materially. Original URLs using HTTP must be replaced with verified HTTPS assets in the official repository, never bypassed security warnings.

## Task and split protocol

1. Reserve 10,000 of the official 60,000 training images as validation using a published stratified split seed. Keep official 10,000 test images untouched until configuration freeze.
2. Each episode shows a memory image, 0–8 distractor images, a current image, and a cue selecting the remembered versus current category. All underlying images in a train/validation/test episode must come from that split. Store image IDs, episode seeds and hashes to prevent leakage.
3. Phase A supplies a shared recalled image to all architectures to isolate perceptual/role specialization. Phase B removes exact retrieval and requires every architecture to learn delayed storage under matched memory dimensions.
4. Include independent class pairs, shifted pair correlations, unseen pair combinations, cue noise, and longer distractor delays. Episode composition is synthetic even though pixels come from real images. This remains an engineered benchmark.

## Models and equal-information controls

- Functional: shared learned image encoder, role-specific current/recalled pathways, learned cue router. For Phase B, replace recalled input with a learned recurrent state.
- Generic modular: same encoder, expert count, total parameters, state size and communication budget; no designated expert role.
- Monolithic recurrent: same encoder and parameter budget, matched recurrent state; give it the same input order and cue access.
- Permuted roles and no-routing ablations; oracle routing and perfect recall as explicitly privileged positive controls.
- Causal-training baseline using original implementation of interchange-intervention training, only after validating gradients and matching intervention supervision across a designated comparison.

Keep model parameters within ±2% and track measured forward/backward operations and wall time. Because a larger or better-optimized baseline may explain the toy gap, also run an accuracy-matched baseline family. Tune learning rates/widths on validation under equal trial budgets, then freeze selections.

## Preregistered outcomes and analysis gates

- Twenty initialization/episode seeds after a two-seed feasibility run; never select seeds by outcome.
- Primary: held-out changed-case IIA, conditioned on both base and source factual predictions being correct, alongside unconditional IIA and coverage of that conditioning.
- Predictive competence: an accuracy-equivalence band of ±1 percentage point established before looking at test outcomes. If that band is not met, report descriptive joint accuracy/IIA trade-offs and decline a standalone interpretability superiority claim.
- Counterfactual coverage: balance intervention effects, include irrelevant-variable invariance, learned rotated-subspace mappings, equal-dimensional patches, and equal mapping-search compute. Evaluate rescue and unseen intervention types.
- Report all runs, failures, costs, bootstrap intervals clustered by seed, and sensitivity to intervention granularity. Treat biological motivation as a design rationale, not an experimental factor.

## Resource gates

These are conservative proposed caps, not benchmarked costs.

- **Before download:** ≥1 GiB free disk and ≥2 GiB free RAM; ≤50 MB compressed input budget; no paid network/API. Reconfirm official source and license. No pretrained weights are needed.
- **Before implementation:** inspect existing frameworks. Current pilot environment lacks PyTorch. Either extend the NumPy baseline or separately authorize/install a reputable pinned CPU framework if needed; do not silently install it as part of a test command.
- **Feasibility run:** ≤100,000 trainable parameters; two seeds; ≤2 CPU threads; ≤1 GiB measured peak RSS; ≤15 minutes total wall time; ≤250 MB saved artifacts. Stop and profile if any cap is crossed rather than silently using a GPU or cloud service.
- **Full run:** start only if timing extrapolation fits ≤2 CPU-hours and existing hardware. Record actual versus forecast costs. Cap files at 1 GiB. No paid GPU, external logging account, or publication is part of execution.
- **Scaling gate:** only expand to pretrained sequence models once matched-accuracy, equal-intervention-bandwidth comparisons hold on multiple task families. For each candidate, independently verify exact model/data licenses, immutable revisions, download size, RAM/VRAM estimate and execution budget before fetching any weights.

## Advancement or stopping criteria

Advance if evidence survives equal-information, matched-accuracy and matched-intervention comparisons across task shifts. If generic modular models match the functional model, report that functional decomposition is a usable interface rather than a unique biological advantage. If benefits vanish on learned memory or real pixels, retain and publish the negative result and reassess the role prior before scaling.
