# Frozen v2 development results

Status: all 30 predeclared cases completed once; independent bundle verifier accepts all 30 unique cases, 150 checkpoints and 363 inventoried artifacts. No failures, retries, dropped cases, source edits, locks or official-test predictions. The source freeze still verifies.

Bundle manifest SHA256: `3d385b851af36cd2b9965b0cdd5571498b9adcc878de68303da251304c591432`.

## Predetermined learning-rate selection
Common diagnostic learning rate: **0.3**; useful-compute uniform learning rate: **0.1**.

Mean final validation macro recall across the two development seeds:

| Learning rate | MIR | High loss | Uniform | Shuffled score | Useful uniform | Diagnostic pooled |
|---|---|---|---|---|---|---|
| 0.03 | 68.87% | 65.70% | 67.35% | 67.96% | 75.58% | 67.47% |
| 0.1 | 72.51% | 69.87% | 70.89% | 71.57% | 75.89% | 71.21% |
| 0.3 | 72.32% | 69.27% | 71.02% | 73.67% | 74.03% | 71.57% |

The fixed common-rate selection pools all four diagnostic arms, not the per-arm optimum. The winning common rate is at the predefined grid boundary; the grid is not expanded. At that selected rate, shuffled-score exceeds MIR on these development seeds. These are hyperparameter-selection data, not confirmatory evidence for MIR, mechanistic novelty, or biological claims. The useful-compute control has its separately selected rate and more replay updates, so it is not an update/presentation-matched comparison.

## Actual resource use
- Charged development CPU: 287.097 s; wall time: 291.117 s.
- Training-call CPU (including audits): 281.905 s; validation scoring CPU: 2.074 s. Remaining charge includes setup, preparation loading, resource checks and bundle finalization.
- Stage-wide sampled peak RSS: 60,719,104 bytes (57.91 MiB). This is sampled, not continuous/hard peak measurement.
- Tracked retained data/artifact peak: 77,309,571 bytes (73.73 MiB). The journal includes loader/scoring/finalization samples; this is not merely learner RSS.

## Revised prospective estimate for all development plus evaluation
- CPU: 2214.9 s (36.9 min), versus original 2705.8 s. Formula: actual development + twice (100/30) times (actual development + validation-scoring CPU), allowing doubled test-set scoring size.
- Retained data/output: 229,023,499 bytes (218.41 MiB), versus original 166,411,215 bytes (158.70 MiB). Output estimate increases because real case audit/trace files are larger than the synthetic extrapolation. Formula: current tracked peak + 20%-padded per-arm maximum observed case bytes for 100 cases +10 MiB for predictions/provenance/manifest/trained-case records.
- RSS planning estimate remains 240,881,664 bytes (229.72 MiB), above the observed development sample peak. Evaluation additionally loads the official test data only after all locked training.
- Revised output margin to 250 MiB cap: 31.59 MiB. No budget increase; original forecast is preserved unchanged. Estimates require pre-evaluation review, and actual cooperative checks remain active.

## Scientific and state boundary
The model has 6,634 float32 parameters (26,536 bytes); reservoir raw arrays occupy 158,600 bytes, giving 185,136 persistent parameter-plus-reservoir array bytes. RNG state/counters and transient numerical allocations are separately recorded in every ledger. Checkpoints, selection traces and mechanism audits are write-only for the learner, not extra adaptive learner memory.

All diagnostic cases use 6,200,562,008 logical numerical operations; useful-control cases use 6,200,428,622 with 133,386 residual operations and 15,853 extra replay updates. The convention excludes RNG, sorting, copying and I/O and is not a machine-instruction or wall-time equality claim.

Fashion-MNIST was prospectively selected within this research program. Previously inspected Optdigits results are not eligible for selection here. This local boundary does not mean the public benchmark is globally unseen or absent from model pretraining. No brain-region responsibilities or neuroscience discoveries follow from these artificial-network layers.

