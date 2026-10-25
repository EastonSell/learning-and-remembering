# Useful-compute PenDigits control: MIR hypothesis unsupported

## Result

The one admitted run completed all **36 planned models, 12 paired seeds and 360 boundary checkpoints**, with no failed, dropped or replaced cases and no retries. The primary MIR improvement hypothesis was not supported. Giving uniform replay useful additional updates with a prospectively fixed matrix-work allowance produced much higher accuracy than MIR, but also longer measured training time. The two uniform baselines show an accuracy/runtime tradeoff, not a universal winner.

| Arm | Mean final macro recall | Mean training seconds per seed |
|---|---:|---:|
| Useful uniform, equal base updates | 80.2649% | 0.095805 |
| MIR | 77.7495% | 0.243045 |
| Useful uniform, credited extra replay | 89.2730% | 0.374888 |

These short in-process timings exclude boundary evaluation and checkpoint serialization. They are not end-to-end deployment latency, repeated hardware benchmarks, or evidence of equal wallclock budgets.

## Prespecified primary comparison

MIR minus credited uniform:

- Final macro recall: **−11.5236 percentage points**, paired-seed 95% percentile-bootstrap interval **[−12.9609, −10.0623]**.
- Mean current-pair acquisition: **−4.3929 points**, interval **[−4.9348, −3.8109]**.
- Mean training-time difference: **−0.131843 seconds per seed**, interval **[−0.144540, −0.122250]**.
- Geometric-mean MIR/credited-uniform training-time ratio: **0.649263**, interval **[0.630384, 0.664895]**.

The required accuracy gain of at least +1 point, strictly positive lower confidence bound, and acquisition difference of at least −2 points were not satisfied. Accuracy support is **false**. The joint within-environment accuracy/runtime dominance gate is also **false**. MIR's shorter measured time against credited uniform does not rescue its adverse accuracy and acquisition results.

## Descriptive equal-update comparison

MIR minus useful equal-update uniform:

- Final macro recall: **−2.5155 points**, interval **[−4.3060, −0.6973]**.
- Mean current-pair acquisition: **−6.7969 points**, interval **[−7.2098, −6.3958]**.
- Mean training-time difference: **+0.147240 seconds per seed**, interval **[+0.144413, +0.150425]**.
- Geometric-mean MIR/uniform training-time ratio: **2.540705**, interval **[2.465375, 2.601852]**.

Thus equal-update uniform was descriptively more accurate and faster than MIR in this run. This was the prespecified descriptive comparison, not a second confirmatory hypothesis or a general efficiency claim. There is no accuracy/time scalar ranking.

## What the control changes

The earlier frozen PenDigits study deliberately made every arm compute a discarded virtual update and both candidate scores. That isolates selection under matched computations, but uniform's work is knowingly wasted. This new implementation removes that unused work from both uniform arms and samples directly from the reservoir.

The credited arm spends a predetermined integer allowance on replay-only SGD, never on extra fresh exposure. A forward scoring row is charged 1,664 matrix multiply-accumulates; an SGD training row is charged 3,968. The accounting excludes softmax/ReLU/log, reductions, sorting, RNG, allocation, Python overhead and parameter-update elementwise work. **This is a matrix-MAC proxy, not full compute, energy or wallclock equality.**

Each seed recorded the following identical schedule totals:

| Accounting quantity | Equal-update uniform | MIR | Credited uniform |
|---|---:|---:|---:|
| Base SGD steps | 760 | 760 | 760 |
| Extra replay-only SGD steps | 0 | 0 | 3,261 |
| Virtual SGD steps | 0 | 756 | 0 |
| Actual fresh presentations | 23,980 | 23,980 | 23,980 |
| Base replay presentations | 24,192 | 24,192 | 24,192 |
| Extra replay presentations | 0 | 0 | 104,352 |
| Candidate loss-scoring rows | 0 | 192,000 | 0 |
| Matrix-work units, 128 MACs each | 1,493,332 | 4,728,744 | 4,728,244 |

The credited arm's 500-unit residual is below the 992-unit cost of one further complete replay batch. MIR work minus credited-uniform work equals this residual exactly for every seed. Both uniform arms performed zero virtual steps and zero candidate scoring. Their base replay traces matched, and all three reservoir traces matched within each seed. No arm's schedule was changed to match observed time.

## Design and boundaries

The [public prose-redacted protocol](PROTOCOL.md), with its original predata digest preserved in SOURCE_PRE_DATA_MANIFEST.json, uses the original pinned PenDigits train/test files, fixed stratified development reservation, 16→64→10 ReLU MLP, SGD .05, memory256, fresh/replay32, MIR candidate128, recurrent class pairs and twelve new seeds51001–51012. All six arm execution orders are used twice. Data and analysis choices were frozen and independently reviewed before this extension was fitted.

The dataset and its historical results had already been observed. This is a **prospective new-seed extension on a reused dataset**, not fresh external replication or an unseen-test benchmark. Algorithmic seeds do not supply independent writer/population replication. No claim about language models, human/biological memory, SOTA, broad continual-learning performance or deployment efficiency follows. Timing uncertainty is across these seed executions in one process/environment, not across replicated machines or load conditions.

All paired intervals use 10,000 frozen paired-seed resamples, RNG20261027 and linear percentile quantiles. No outliers, adverse seeds or completed cases were excluded. The support gate was not changed after outcomes. The original PenDigits study and historical files were not altered.

## Resources and checks

- External process supervisor: **15.519273 seconds**, return code0, no termination; frozen ceiling600 seconds.
- Peak independently sampled RSS: **133,455,872 bytes**, below the frozen1GiB ceiling.
- Runner's pre-finalization walltime measurement:14.470762 seconds. The supervisor additionally includes process startup/imports and exit, so the two are not interchangeable.
- Python3.12.14, NumPy2.3.5, sklearn1.8.0, one numerical thread. No installs, accelerators or paid resources.
- Before execution:30 synthetic checks passed normally and under Python optimization; integrated repository suite133/133 passed. Synthetic checks are engineering evidence only.
- [Independent outcome audit](review/independent-postrun.json): recomputed all360 checkpoint predictions, probabilities and boundary metrics directly from saved weights and retained test inputs without importing the training engine; checked all36 cases, paired main accuracy/runtime intervals and matrix-work accounting. It did not replay training or independently remeasure the elapsed time.

The private full evidence backup retains all checkpoint weights/probabilities/predictions, raw pinned source members, split/phase IDs, original provenance, source, admission/reviews, supervisor logs and complete outputs. The public compact package retains all case/boundary numeric records and paired signed outcomes but does not contain checkpoint binaries or raw test inputs. Do not claim the compact package alone supports fresh checkpoint inference. See [publication and reproduction levels](PUBLICATION.md) and [transparent metadata derivatives](PUBLIC_PROVENANCE_DERIVATIONS.json).
