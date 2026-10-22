# Replay execution driver: source checkpoint, unrun

2026-10-22. The [prospective protocol](protocols/replay_digits_v1.md) is unchanged.
**No UCI model fitting or scoring has been performed.** The driver and its
synthetic/mocked tests are ready for review before the study starts. Historical
[data/core checkpoint](replay_implementation.md) results remain preserved.

## Independent stages

1. `develop` runs all 18 prespecified validation comparisons: three common
   learning rates × two development seeds × three arms. It selects one rate by
   the fixed six-score mean rule. Official test bytes may be integrity-audited,
   but no test prediction or model score occurs in this stage.
2. A completed development directory contains `selection.json` and a hash
   manifest, with no `incomplete.json`. The selection records every candidate
   score, fixed configuration, versions and data/source identities. Its SHA-256
   must be supplied explicitly to evaluation. The lock does **not** persist
   trained weights; evaluation starts fresh.
3. `evaluate` rejects modified/incomplete selection records and data/source/
   dependency mismatches before model access to test examples. It writes
   `evaluation_lock.json` before the first test prediction, then executes all
   30 prespecified evaluation runs. No rate, seed, epoch or model choice is
   made from test results. It records all five boundary snapshots' scores,
   final metrics, exposure/memory diagnostics and the fixed paired comparison.

The official archive SHA-256, published audit SHA-256 and frozen protocol SHA-256
are pinned in source. The fixed split IDs are reconstructed and checked against
the audited IDs, not accepted from a caller's configuration. Different data or
protocols require a separately reviewed version, not a hidden CLI override.

## Resource and failure behavior

Use the fresh-process launcher so numerical-library environment limits are set
**before** importing NumPy. It uses one thread, within the protocol's two-thread
ceiling. Resource enforcement currently requires Linux.

- Hard process-local wall timers: 900 seconds development, 1,800 evaluation.
- A conservative 1 GiB **virtual address-space** limit, stricter than the
  protocol's 1 GiB peak-RSS ceiling; peak RSS is also checked after runs.
- At most 250 MB of artifacts per stage, checked before each JSON write.
- Existing output directories are refused, even empty ones.
- Resource/configuration failures attempt to write `incomplete.json` and raise
  an error. Never treat an incomplete directory as a completed study, even if
  some earlier files exist. Do not change settings after a test-stage failure.

These are process-local limits, not system configuration changes. A stricter
memory cap may stop an otherwise feasible implementation; that is reported as a
resource gate failure, not scientific evidence for or against replay.

## Commands after source review

Keep the downloaded official archive outside the repository. No data or model
weights are downloaded by this driver, and no package installation is needed.

```sh
PYTHONPATH=src python scripts/run_replay_digits.py develop \
  --archive /path/to/optdigits.zip --output results/local-replay-development
```

Record the SHA-256 printed on successful development, review the selection and
its completion manifest, then explicitly supply that exact digest:

```sh
PYTHONPATH=src python scripts/run_replay_digits.py evaluate \
  --archive /path/to/optdigits.zip \
  --selection results/local-replay-development/selection.json \
  --selection-sha256 THE_RECORDED_DEVELOPMENT_DIGEST \
  --output results/local-replay-evaluation
PYTHONPATH=src python scripts/verify_run.py results/local-replay-development
PYTHONPATH=src python scripts/verify_run.py results/local-replay-evaluation
```

The hash verifier checks content, not completion status; also check that neither
stage has an `incomplete.json`. Wall times and process RSS are diagnostic and
will vary across machines. Bootstrap intervals concern these ten algorithmic
seeds conditional on this dataset, not a population of writers or brain systems.

## Tests and remaining claim boundary

42 tests pass: the prior 34 plus eight orchestration checks. New tests use a
synthetic fixture and mocked training/scoring, enforce the 18/30 comparison
counts, prove no development test predictions, check the pre-prediction lock,
reject changed rates/configuration/versions/data and incomplete records, and
exercise thread/artifact/time-limit failures. The real alarm test verifies
restoration of process-local memory limits and signal handlers.

Passing these checks establishes software safeguards, not replay efficacy.
Actual development and evaluation remain unrun; the next milestone must report
all outcomes, including acquisition failure or an inconclusive primary contrast.
