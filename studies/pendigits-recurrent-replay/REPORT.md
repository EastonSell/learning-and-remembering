# Actual trained-model experiment: PenDigits recurrent replay

Completed 24 October 2026. This is a new empirical pilot, not a replay of historical results, a theorem, or a diagram.

## Result

The proposed MIR replay selector did not earn support over uniform replay.

| Selector | Final held-out macro accuracy | Current-pair acquisition, averaged across phase boundaries |
| --- | ---: | ---: |
| Uniform replay, primary baseline | 79.94% | 94.06% |
| MIR, proposed change | 78.48% | 86.18% |
| High-loss replay, descriptive secondary control | 85.65% | 93.56% |

MIR minus uniform: −1.46 percentage points, paired-seed bootstrap 95% interval [−3.16, +0.33] points. The predeclared support condition required at least +1 point, an interval lower bound above zero, and no more than a 2-point mean acquisition loss. MIR met none of these requirements and its acquisition loss was 7.88 points. The interval crossing zero means final-accuracy benefit was inconclusive, not proof of equivalence or a population-wide harm effect. The acquisition guardrail failed descriptively. High-loss performed best here, but was a secondary comparison and does not establish universal superiority.

## What actually ran

- 30 fresh trained neural networks: 10 paired seeds × 3 selectors.
- Public UCI PenDigits trajectory data, independent of earlier Optdigits/Fashion-MNIST and reader-study data. 7,494 official training records; 5,995 used for training and 1,499 held aside as development, unused for tuning. 3,498 official independent-writer test records. No exact feature overlap among train/development/test.
- 16→64→10 ReLU MLP, 1,738 learned parameters, mean cross entropy, SGD learning rate 0.05.
- 10 class-pair phases with recurring classes and disjoint examples on their second visit; no task identifiers given to the model.
- Identical 256-example reservoir, fresh data order, initial weights, candidate 128 panels, 760 actual SGD updates and 760 virtual updates per model. 22,800 actual SGD updates total.
- Each model processed 23,980 fresh-example presentations, 24,192 replay presentations and 192,000 pre/post candidate scores. All 30 cases had exactly identical accounting.
- Official test first opened after independent protocol/code review and a source/protocol hash freeze. No tuning, configuration changes, seed replacement, favorable checkpoint choice or further training after outcomes.
- All 300 phase checkpoint/prediction pairs retained. Verification reconstructed predictions and probabilities exactly from every checkpoint, recomputed primary interval, checked all 30 cases and matching compute counters, and checked frozen source/protocol hashes.
- Execution completed in 17.46 seconds including data transfer and recorded scoring/output work; peak process RSS 140,091,392 bytes (133.6 MiB). This is a measured runtime on the available cloud CPU, not a promised scaling benchmark. Verification ran separately and is not included in that runtime.

Independent post-run audit also re-downloaded the official dataset, matched archive/train/test hashes, checked every saved test feature/label against the official member, reconstructed all split/count/boundary/primary statistics and reproduced all 300 checkpoint outputs. Detailed per-step selection traces were not logged; compute counts were reconstructed from the frozen procedure and saved accounting. Resource measurements are reported by the execution process, not independently instrumented hardware measurements.

## Limits and next decision

This is a small, deliberately constrained trajectory-classification pilot, not an LLM, natural language task, reader study or deployment benchmark. The class recurrence is imposed rather than natural. One dataset/split and ten training seeds do not represent ten independent datasets or populations; bootstrap uncertainty describes algorithmic variation only. The official provider describes the test writers as separate; individual writer IDs were not reconstructed. Development is row-stratified, not writer-held-out.

Uniform and high-loss deliberately performed unused MIR-style scoring to match computational operations. That isolates selector effects, but does not prove anything about best use of a fixed wall-clock budget or practical efficiency. A useful-compute uniform baseline remains untested.

Do not claim MIR improved learning or retention from this result. Do not promote the best secondary method into a post-hoc confirmed winner. Keep this negative finding and decide whether a genuinely new, preregistered hypothesis warrants another experiment. The data do not support biological, reader-comprehension, generalization-to-LLMs, state-of-the-art or broadly optimal-memory claims.

## Sources and reproduction

Dataset: Alpaydin and Alimoglu, Pen-Based Recognition of Handwritten Digits, UCI Machine Learning Repository, CC BY 4.0: https://doi.org/10.24432/C5MG6K

The accompanying protocol, source, independent pre-run review, frozen hashes, complete seed/phase metrics and verification report specify the run. The existing results directory is intentionally never overwritten. For verification in the retained full bundle: OPENBLAS_NUM_THREADS=1 python verify_results.py. Retraining would require a separately named output directory and explicit replication status because these test outcomes are now known.
