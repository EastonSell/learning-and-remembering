# Matched-coverage replay selection: locked Fashion-MNIST evaluation

## Main finding

**The prespecified primary result is inconclusive.** MIR minus high-loss final macro accuracy is +0.6265 percentage points (pp), paired-seed 95% percentile bootstrap interval [-0.3595, +1.5330] pp. The fixed support criterion was mean ≥1 pp and lower interval bound >0; neither requirement is met. The interval includes both no benefit and the prespecified practical benefit, so this is not an equivalence result or proof of zero effect.

Uniform replay and useful-compute uniform achieve higher mean final accuracy than MIR. The prespecified audit also does not support the proposed protection interpretation: selecting high virtual-harm items increases, rather than decreases, old-class validation loss relative to within-class shuffled selection in the separate sequential audit branches. These adverse findings are retained, not used to tune, restart or omit seeds.

## Design and evidence

Twenty paired seeds (2001–2020), five arms, 100 freshly initialized runs, 500 block-boundary checkpoints and 500 retained prediction arrays. Common learning rate 0.3 for the four diagnostic arms; useful-compute uniform 0.1, selected from all 30 development cases using the frozen rule. No grid expansion despite the common rate landing at its upper boundary. All training/checkpoints finished before official-test loading. All100 cases are complete and pass the strict bundle verifier; no research failures, exclusions, retries or source changes occurred.

Lock created: 2026-10-22T09:11:05.562440+00:00. First test-scoring marker: 2026-10-22T09:26:23.783502+00:00. All10,000 official test items survived the prospective exact-duplicate rule; full and deduplicated metrics coincide for every case. Fixed train/validation sizes are 10,000/5,000. No person/writer-disjointness claim follows.

Source freeze SHA256: `737efd1614adca5b923262e11990e054b26325ef30fad7c5fabf1534c9ef14b3`.
Evaluation lock SHA256: `56211a0a4c53de83e54dd77b424dbfe0099d44f28686a528c4a413b0c9c623ba`.
Evaluation bundle manifest SHA256: `064c2b19322cdb1f6d9fc298f22305a640626ef8816b174941b3b78ad00c3873`.

## Outcomes

Means across 20 seeds; accuracy/acquisition are percent, forgetting is percentage points. Forgetting uses the frozen first-four-block best post-acquisition/pre-final boundary minus final boundary, without clipping.

| Arm | Final macro accuracy | Immediate acquisition | Forgetting |
|---|---:|---:|---:|
| mir | 70.8425 | 92.7000 | 27.5612 |
| highloss | 70.2160 | 94.1840 | 29.9600 |
| uniform | 72.1355 | 94.1175 | 27.6394 |
| shuffledscore | 71.4035 | 94.3415 | 28.6913 |
| uniform-extra-replay | 75.3325 | 91.3260 | 19.9969 |

MIR acquisition is 1.484 pp below high-loss. This does not cross the prespecified >2 pp plasticity-warning threshold, but the measured acquisition reduction remains visible. Useful-compute uniform also has lower immediate acquisition than MIR while improving final accuracy; retention should not be described as universal improvement.

| MIR minus comparator | Mean pp | 95% paired bootstrap pp | Status |
|---|---:|---|---|
| highloss | +0.6265 | [-0.3595, +1.5330] | Primary: inconclusive |
| uniform | -1.2930 | [-2.4040, -0.2674] | Secondary: descriptive only |
| shuffledscore | -0.5610 | [-1.9635, +0.9840] | Secondary: descriptive only |
| uniform-extra-replay | -4.4900 | [-5.6015, -3.4205] | Secondary: descriptive only |

Bootstrap: 10,000 paired-seed resamples, PCG64 seed 73103. Secondary intervals are descriptive and not multiplicity-adjusted confirmatory findings. The20 seed/order pairs are not20 independent visual domains or human subjects.

Every primary paired difference in seed order 2001–2020 (pp): +2.65, +2.66, -2.05, -1.68, +1.76, +0.03, -0.24, +3.83, -0.11, +0.25, -5.05, +2.70, -0.45, +2.72, +0.25, +1.80, +2.48, +3.10, -1.03, -1.09.

### Per-class final recall (%)

| Arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| mir | 69.28 | 89.05 | 48.11 | 70.19 | 57.47 | 77.31 | 33.05 | 85.56 | 87.72 | 90.69 |
| highloss | 73.13 | 89.04 | 47.70 | 70.00 | 53.21 | 75.29 | 31.20 | 84.43 | 88.15 | 90.02 |
| uniform | 69.00 | 87.37 | 53.48 | 72.63 | 58.84 | 77.57 | 38.62 | 85.18 | 87.67 | 90.99 |
| shuffledscore | 67.98 | 87.52 | 51.62 | 69.69 | 55.24 | 78.19 | 39.06 | 85.57 | 87.21 | 91.95 |
| uniform-extra-replay | 74.44 | 88.42 | 59.22 | 74.12 | 66.56 | 78.72 | 44.95 | 86.22 | 88.81 | 91.86 |

All per-seed recalls, acquisition, forgetting and replay-frequency summaries are in `evaluation-cases-v2.csv`; all 100 full 5×5 matrices and raw class recalls remain in `evaluation-v2/cases.jsonl`.

## Prespecified mechanism and coverage audits

Exactly 80 audit opportunities/run, 8,000 total; all are retained. The following aggregates average within each seed then across seeds. Audit opportunities are not treated as independent inferential samples. Lower selected-minus-shuffled old-class cross-entropy is better.

| Arm | Selected virtual harm | Eligible virtual harm | MIR/high-loss overlap | Audit old-loss contrast |
|---|---:|---:|---:|---:|
| mir | 0.547620 | 0.303035 | 0.370625 | +0.078770 |
| highloss | 0.274000 | 0.262119 | 0.406875 | +0.010742 |
| uniform | 0.226490 | 0.276021 | 0.449437 | +0.004495 |
| shuffledscore | 0.225528 | 0.272624 | 0.455000 | +0.002988 |
| uniform-extra-replay | 0.104651 | 0.130378 | 0.445625 | -0.002594 |

MIR selects greater virtual harm by construction. Candidate rank versus exhaustive virtual harm is effectively 1 in all arms; this is a circular implementation/coverage check, not mechanistic confirmation. MIR’s +0.078770 old-loss contrast is adverse. The audit uses sequential fresh-only then replay-only branches, whereas training uses a joint fresh-plus-replay update, so that contrast is not an exact causal decomposition of deployment updates. Audit validation images never feed back to learner state, selection or post-lock tuning.

| Arm | Negative score % | Zero score % | Updates with score ties % | Unique examples replayed | Historical presentations | Max single-example replays |
|---|---:|---:|---:|---:|---:|---:|
| mir | 39.57 | 3.33 | 16.30 | 795.15 | 24960 | 292.85 |
| highloss | 38.14 | 5.17 | 25.74 | 591.50 | 24960 | 430.80 |
| uniform | 38.75 | 2.94 | 15.15 | 912.45 | 24960 | 151.30 |
| shuffledscore | 38.34 | 2.99 | 15.00 | 912.60 | 24960 | 147.90 |
| uniform-extra-replay | N/A | N/A | N/A | 972.85 | 183370 | 1069.50 |

Unique counts and single-example maxima above are seed means; historical frequencies exclude current-only repetitions and include useful-control extra replay. No negative interference score clipping occurred. Reservoir/candidate/quota/stream hashes match across all five arms within each seed. Mean final pre-update reservoir counts for labels 0–9 are 20.95, 20.90, 20.70, 20.60, 18.75, 19.15, 19.35, 20.55, 19.30, 19.75, identically across arms. Full per-update coverage and per-source replay counts remain in every case’s saved arrays. The final trace row precedes final admissions, as disclosed in diagnostics.

## Computation and memory

The 196→32ReLU→10softmax NumPy model has 6,634 trainable float32 scalars (26,536 bytes). Reservoir features/labels/source IDs consume 158,600 raw bytes: parameters plus reservoir 185,136 bytes. The history buffer is larger than the model. Additional mutable nontrainable state includes PCG64 states, counters/hashes and useful-control budget residual; these are not zero-cost memory. There are no learned priorities, cached losses/logits or optimizer momentum. Virtual copies add 26,536 transient bytes before gradients/activations and language/runtime overhead. Per-case RNG serialization, Python counter size, traced allocation and sampled RSS are retained in raw ledgers.

Checkpoint files, traces, frequency/coverage summaries and audits are write-only or post-training evidence for the learner, not an extra readable history store. The audit callback receives copied state and returns nothing. API separation is not an OS security sandbox.

Per diagnostic run: 2,500 base updates, 40,000 fresh presentations, 40 current-repeat presentations, 24,960 historical presentations, 10,000 unique observations, 124,648 candidate draws, 249,296 candidate forward examples and 2,500 virtual gradients. All four diagnostic arms consume 6,200,562,008 declared logical numerical operations. Useful uniform consumes 6,200,428,622 (133,386 unspent), with 15,853 extra ten-item replay updates: 120 extra current and 158,410 extra historical presentations. Its total historical count is 183,370. It is compute-capped, not update/presentation-matched, and has a separately selected rate.

| Arm | Training CPU s/run incl. audit | Audit CPU s/run | Scoring CPU s/run | Learner I/O wall s/run | Native theoretical operations/run |
|---|---:|---:|---:|---:|---:|
| mir | 6.936 | 0.497 | 0.345 | 0.520 | 6200562008 |
| highloss | 6.829 | 0.492 | 0.357 | 0.504 | 3440143680 |
| uniform | 7.123 | 0.496 | 0.343 | 0.544 | 1784195000 |
| shuffledscore | 7.103 | 0.499 | 0.350 | 0.531 | 6200562008 |
| uniform-extra-replay | 16.438 | 0.495 | 0.356 | 0.797 | 6200428622 |

Native cost is a theoretical unpadded numerical-graph count, not a measured optimized implementation. Padded uniform/high-loss deliberately execute unused selector work. Useful uniform takes more measured CPU despite the logical-kernel cap, since many small updates, RNG, sorting, copying and I/O are outside that counting convention. No hardware FLOP, wall-time efficiency or optimized speedup equality is claimed.

Evaluation charge: 945.178 CPU s and 963.521 wall s. Development+evaluation charge: 1232.275 CPU s (20.54 min). Stage-wide evaluation sampled RSS peak: 84,770,816 bytes (80.84 MiB). Tracked retained-data/artifact peak: 201,638,231 bytes (192.30 MiB), below the revised 218.41 MiB forecast and 250 MiB cap. Original 2,705.79-second CPU forecast and 1 GiB RSS cap were not increased. Research execution used single-thread BLAS with an actual ≤2-thread gate. These are cooperative sampled thresholds, not hard OS-enforced continuous peaks. Analysis/review sidecars are separately retained outside immutable run bundles; their small sizes do not threaten the remaining 57.70 MiB margin.

## Interpretation and limitations

This bounded experiment does not establish a practical MIR advantage over the simpler high-loss selector, and descriptively favors uniform replay, especially spending saved numerical work on extra replay. It therefore does not justify scaling a MIR-protection or brain-layer claim on this evidence. This is not a novel algorithm: MIR is an existing replay-selection method. No correspondence between artificial layer names and brain regions/responsibilities was tested.

Results concern one fixed downsampled Fashion-MNIST split, tiny MLP, limited reservoir, privileged class-balanced quotas, fixed common diagnostic learning rate and repeated-presentation stream. Common-rate results are not independently optimized best-arm performance. Twenty seeds are bounded design coverage, not a proven power analysis. Numerical reproducibility is not additional statistical evidence. Exact-duplicate audits do not establish identity-disjoint data.

Fashion-MNIST selection was prospective within this research program; prior inspected Optdigits results were ineligible for this study’s test selection. This local provenance is not a claim that the public benchmark was globally unseen, absent from pretraining or unused in unrelated projects. The frozen protocol, all 30 development outcomes, both source-freeze reviews, fixed lock and all 100 evaluation outcomes are preserved. No further training or tuning is part of this report.

## Retained evidence

- `evaluation-summary-v2.json`: frozen summary, all 20 paired differences per comparison, bootstrap intervals and plasticity flag.
- `evaluation-diagnostics-v2.json`: all 100 per-case audit/frequency/coverage summaries, resource ledgers and arm means.
- `evaluation-cases-v2.csv`: reusable 100-case analysis table.
- `evaluation-v2/`: immutable complete raw bundle, manifest, all checkpoints, predictions, audit lines and traces.
- `development-v2/`, `development-report.md`, `evaluation-lock-v2.json`, authorization receipts and `budget.jsonl`: selection, lock, review and resource provenance.
- Analysis scripts are retained alongside outputs; they do not modify frozen training source.

Independent final review is requested separately; this report does not claim an independent pass before that review is returned.
