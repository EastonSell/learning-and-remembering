# Post-run audit-tool correction

A release audit found that the original `scripts/verify_run.py --compare`
checked only rows and checkpoint files present in the candidate. Consequently,
an empty candidate metric list with no checkpoints could incorrectly pass.
A truncated candidate could also be presented as a complete comparison.
This was a genuine verifier bug, not evidence that the saved complete runs
were empty. Their expected row counts were checked separately before release.

The corrected tool requires a nonempty metric list, unique keys, exact expected
key coverage, equal numerical field coverage/values, and every expected candidate
checkpoint with no unexpected files. Expected coverage comes from the reference
metrics and manifest, not the candidate. Timings and peak RSS alone are excluded
from metric equality. Nonfinite JSON constants are rejected. Hash-only auditing
without `--compare` verifies recorded hashes, not experimental completeness.

Partial comparisons now require `--compare-seeds`. A candidate must contain
exactly the requested existing reference seeds, not a silently accepted subset.
`--allow-missing-checkpoints` permits missing **reference** binaries only; it
never permits missing candidate checkpoints. When reference binaries were
omitted from this text-only publication, regenerated candidate checkpoint bytes
must match their recorded hashes. When both binaries exist, array names, dtypes
and values are compared exactly.

## Historical provenance

This is a post-run tooling correction, not a change to the experiment or its
selection. Historical pilot, role-control and competence manifests recorded the
old verifier as source. Its exact bytes are preserved in
[`archive/verify_run_pre_coverage_fix.py.txt`](archive/verify_run_pre_coverage_fix.py.txt),
SHA256 `b9be1d1fb77d531eeeabd97ff58caf7f8801be50a72589a21b3bcfa30ee1c34b`.
Only that exact historical hash for that exact audit-tool path resolves to the
snapshot, with an explicit message. No training source hash is substituted or
skipped. Old manifests and results are unchanged. The frozen replay source lock
does not include this verifier; its model, data, driver, protocol, selection and
all outcomes remain unchanged.

## Verification after the correction

The full suite passes **53 tests**, including 11 verifier regressions covering
empty/truncated/duplicate rows, missing fields and checkpoints, extra checkpoints,
explicit partial scope, nonfinite constants, and the narrow historical snapshot
exception. Eight failure cases were reproduced against the pre-fix verifier.

All 30 replay evaluation configurations still match the separately repeated run
exactly, excluding the documented timing/RSS fields. The legacy seed-11 pilot was
regenerated solely to verify 18 rows and all six recorded checkpoint hashes; this
is not additional independent research evidence. See the compact
[verification record](../results/verifier_coverage_verification.json).
The committed `results/repeat_seed11` folder intentionally has no binary
checkpoints and therefore cannot itself satisfy a checkpoint comparison.
Regenerate a complete candidate instead:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python -m functional_circuits.experiment --seeds 11 \
  --output results/local-verifier-seed11
OPENBLAS_NUM_THREADS=1 python scripts/verify_run.py results/pilot \
  --compare results/local-verifier-seed11 --compare-seeds 11 \
  --allow-missing-checkpoints
```

This compares only seed 11 by explicit request. A full pilot comparison must
regenerate all five reference seeds and omit `--compare-seeds`.
