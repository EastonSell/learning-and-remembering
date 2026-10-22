# Replay implementation and data-integrity checkpoint

2026-10-22. **No model has been trained or scored on UCI Optdigits.** The
[frozen protocol](protocols/replay_digits_v1.md) is unchanged. This checkpoint
implements offline data gates and numerical training primitives, not the final
staged experiment driver. Synthetic unit fixtures are not research results.

## Verified official data

The archive linked from the [UCI dataset page](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
was retrieved over HTTPS. Its 591,292 bytes hash to
`0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0`.
The [download record](../results/replay_data_validation/download_provenance.json)
records the exact resolved URL and retrieval time; the
[integrity report](../results/replay_data_validation/report.json) records member
hashes, class counts, original-file split IDs and the protocol/source versions.

- Official training/test file sizes and class counts match the protocol.
- Fixed stratified split: 3,062 training, 761 validation, 1,797 official test rows.
- No exact feature duplicates, cross-split duplicates or conflicting labels were
  found. No row was removed or reassigned; no protocol amendment was needed.
- Only format/hash/split/duplicate checks were performed. Test labels/features
  were not used for optimization, scoring or selection.
- Raw data are not committed. Credit: E. Alpaydin and C. Kaynak (1998), DOI
  [10.24432/C50P49](https://doi.org/10.24432/C50P49), CC BY 4.0.

## Implemented components and checks

`functional_circuits.replay_data` validates bounded ZIP members without archive
extraction, checks strict integer ranges and official counts, makes the fixed
split, audits duplicates, and writes an exclusive-create JSON report. It is
**offline**: recording the official URL does not authenticate an arbitrary
caller-supplied archive. Compare its SHA-256 against the download record.

`functional_circuits.replay_core` provides the 2,410-parameter MLP, stable
cross-entropy and gradients, FIFO/reservoir memories, fixed single-pass training
schedule, five snapshots, metrics, shared validation-rate selection and paired
seed bootstrap. Training receives only training arrays, never holdouts. Buffer
entries index those arrays; storage diagnostics distinguish logical example
payload from Python/container overhead and the resident dataset. Actual process
memory still needs measurement by the future driver.

The 34-test suite passes, including the prior 22 tests and 12 new checks:
all-parameter finite differences; large-logit stability; reservoir inclusion
frequencies (fixed loose tolerance); FIFO order; empty buffers and partial
batches; equal compute and paired initialization; replay membership; deterministic
repeat; hand-worked metrics/selection; strict parsing; safe archive paths/counts;
disjoint stratified IDs; duplicate gates; and report overwrite protection.

## Reproduce the offline gate

Download the ZIP through the official dataset page, keeping its license and
provenance record. Then, with the existing NumPy dependency:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m functional_circuits.replay_data \
  --archive /path/to/optdigits.zip --output /tmp/optdigits-audit.json
PYTHONPATH=src python scripts/verify_run.py results/replay_data_validation
```

A policy-blocking duplicate audit produces a report and exit code 2; malformed
archives raise an error before a success report exists. Existing outputs are
never overwritten. An exact rerun in the recorded environment reproduced the
saved report, including split IDs and hashes.

## Remaining execution gate

Review these primitives, then add a separately tested development/freeze/test
driver. It must enforce all declared seeds/rates, immutable selection records,
source/data/protocol hash checks, test evaluation only after selection, and
resource limits. No claim of replay efficacy, matched predictive competence or
biological fidelity follows from passing software and data-integrity tests.
