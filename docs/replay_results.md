# Historical replay on writer-held-out digits

Completed 2026-10-22 under the unchanged [prospective protocol](protocols/replay_digits_v1.md).
This is a controlled test of a known rehearsal mechanism, not a new learning
algorithm or a biological validation.

## Result

Ten prespecified seeds, paired across three identical 2,410-parameter MLPs:

| Memory policy | Final macro accuracy | Final micro accuracy | Mean forgetting | Immediate acquisition |
|---|---:|---:|---:|---:|
| Uniform reservoir, 100 examples | 88.44% | 88.49% | 11.66 points | 97.78% |
| Global FIFO, 100 examples | 21.16% | 21.20% | 97.16 points | 98.90% |
| Current-only repetition, no history | 19.70% | 19.74% | 99.03 points | 98.93% |

The primary reservoir-minus-FIFO difference is **67.28 percentage points**,
with a descriptive 95% interval **[65.27, 69.04] points**. Every paired seed
favored reservoir. The prespecified criterion (mean ≥2 points and interval
excluding zero) was met for this setup. Reservoir minus current-only is an
exploratory 68.74-point mean difference, not an additional confirmatory test.

The interval is the 2.5th/97.5th percentile of **10,000 paired-seed bootstrap
resamples**, NumPy RNG seed **9801**. Each resample samples ten paired seed
differences with replacement. It describes variation across these algorithmic
seeds on **one fixed dataset and split**, not independent populations, writers,
or biological subjects. Stored JSON differences are accuracy fractions;
this report multiplies them by 100 to give percentage points.

Forgetting averages the first four blocks' best accuracy from initial learning
through the penultimate boundary minus final accuracy, without clipping
negative values. Immediate acquisition averages each block's accuracy directly
after learning it. All predictions compete over all ten labels. No arm crossed
the prespecified acquisition-collapse threshold. Thus the poor final FIFO and
current-only scores accompany forgetting, not inability to learn the current
block. Full boundary matrices and per-class recalls are retained.

## Selection and comparison integrity

The official source and split passed the [data integrity gate](replay_implementation.md):
3,062 training, 761 validation and 1,797 official test examples, with no exact
feature duplicates. The provider reports different training/test writers; row
writer IDs were not independently reconstructed. Test bytes were used only for
integrity auditing until the configuration was locked.

All 18 development comparisons were completed before test scoring. Mean final
validation macro accuracy across both development seeds and all three arms:

| Common learning rate | Validation mean |
|---|---:|
| 0.03 | 40.61% |
| 0.10 | 43.26% |
| **0.30, selected** | **45.04%** |

The complete [selection](../results/replay_development/selection.json) was saved,
hashed and checked before evaluation. Its SHA-256 is
`3d2e8b04fd4058cdbaae26506246399398f7a72723de70df1b24140a7227b82f`.
The [evaluation lock](../results/replay_evaluation/evaluation_lock.json) pins
that selection, runtime sources, versions and data identities. Weights were
not frozen from development: all 30 evaluation models started fresh, with no
test-based rate, seed, checkpoint or epoch selection.

Reservoir and FIFO have the same 100-example budget, network and update count.
Current-only intentionally has no historical buffer. Every model processed
3,062 fresh observations, 12,248 fresh-example presentations and 12,248 extra
presentations. Depending on block sizes, seeds used 772 or 776 SGD updates;
**all three arms within a seed had exactly the same count and initialization**.
There are no role masks, oracle retrieval, task-ID inputs, separate task heads
or architecture-defined intervention targets.

## Evidence and reproduction

- [Development manifest](../results/replay_development/manifest.json) and all candidate scores.
- [Evaluation metrics](../results/replay_evaluation/metrics.json), [summary](../results/replay_evaluation/summary.json) and [manifest](../results/replay_evaluation/manifest.json).
- [Exact-repeat verification](../results/replay_repeat_verification.json): all 30
  configurations were rerun with the same locked selection. Every numerical
  metric, replay diagnostic and summary matched exactly; only timing/RSS and
  lock creation time were excluded. The repetition is a software check, **not
  another 30 independent research samples**. Redundant repeat output is kept
  locally rather than duplicated in the public bundle; the command and hashes
  are recorded for regeneration.
- [Execution commands and resource limits](replay_execution.md) describe the
  pre-run driver checkpoint. Its historical “unrun” status is superseded by
  this report; the source and protocol were not changed after outcomes.

Both stages completed normally, with no incomplete marker. Model work took
1.18 seconds for development training plus validation, and 1.89 seconds for
evaluation training; these exclude some I/O/scoring overhead and are not a
performance benchmark. Evaluation process peak RSS was 35,332,096 bytes.
The launcher limited numerical libraries to one thread. The 53-test suite and
all run/data manifests pass. Dataset bytes are not redistributed; source credit:
E. Alpaydin and C. Kaynak (1998), [UCI Optdigits, DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49), CC BY 4.0.

## What this does and does not establish

Broad historical coverage preserves digit-class performance better than an
exclusively recent buffer under the imposed class-blocked stream. This is a
non-role-circular behavioral result: targets are external digit annotations,
not counterfactual labels defined by an architectural branch.

The large effect is unsurprising for this strongly nonstationary stream: FIFO
can retain only recent classes while reservoir retains earlier ones. The study
is not a state-of-the-art comparison, a naturalistic task, evidence for optimal
memory policy, or proof that neuroscience uniquely improves ML. The common
three-rate grid is not a separately optimized best case for each method.
The small compressed-image dataset, one split and ten seeds limit transfer.
No neural recordings, biological learning rule, hippocampal anatomy, sleep
replay dynamics or interpretability advantage was validated. Those require
separate tests rather than reinterpretation of this positive algorithmic result.

## Audit-tool correction

A [post-run verifier fix](audit_coverage_fix.md) prevents empty or incomplete
comparisons from passing. It changes no frozen replay source or result. The
repaired tool independently confirms all 30 repeated numerical rows; historical
audit-tool bytes are archived rather than silently rewriting old manifests.
