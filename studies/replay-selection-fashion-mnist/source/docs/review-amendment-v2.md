# Prospective source-review amendment v2

2026-10-22, before any Fashion-MNIST research training or test prediction.
Independent review of v1 identified provenance and lifecycle-resource gate gaps.
No scientific outcome, learner, selector, split, seed, threshold or operation-budget
rule changed. The v1 source snapshot and freeze remain preserved.

Prior freeze SHA256:
7f8ab5ff6f4f3575a66db0224d0c5c98b552a41b3a29d0d9ad55a9fdfb99ae57

## Exact changes

1. Both lock creation and verification now require a completed DEVELOPMENT-stage
   bundle, and require its retained freeze and data-manifest hashes to equal the
   requested lock inputs. A valid software-stage fixture cannot become development
   evidence merely because it contains the right case identities or metrics.
2. Evaluation binds its actual freeze, prepared-data manifest and feasibility
   forecast hashes to the lock BEFORE loading development data or training.
   Forecast path/hash are retained in the lock and independently revalidated by
   the stage coordinator. Summary generation requires an evaluation-stage bundle.
3. Cumulative CPU timing starts at entry to the stage, including source checks and
   data setup. Loader RSS, scoring, post-case writes and finalization are checked,
   not just learner updates. The budget journal is closed and charged on failures.
4. Resource paths include a prepared directory outside the raw-data root. A union
   of resolved file paths prevents double-counting nested roots.
5. A resource overrun after completion sealing leaves FAILED evidence. COMPLETE
   may also remain as historical evidence, but the independent verifier rejects
   any bundle containing FAILED or INTERRUPTED; it cannot be accepted as complete.
   The final budget-journal write is followed by another resource check.

Checks remain cooperative, not OS-enforced hard ceilings. They can detect an
allocation or write after it occurred. Failure is retained, never silently erased.

## Regression evidence

Twelve new review regressions cover wrong stage/provenance, early evaluation
binding, external/nested resource paths, each resource dimension, scoring-only
RSS overrun, loader-only CPU overrun and finalization-only disk overrun. The initial eleven fail
against frozen v1 (seven failures and four absent-helper errors); a twelfth tests
retention of observed resource peaks even on failure. All pass v2.
Only synthetic arrays/software-stage fixtures are used. No official-test loading
or predictions are part of these regression tests.

Stage resource samples through budget closure are retained in the journal, including
loader/scoring checks. Per-case learner RSS remains a training-only measurement;
no sampled value is presented as an exact continuous whole-study peak.
