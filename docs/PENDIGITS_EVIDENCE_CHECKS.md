# Fail-closed checks for the published PenDigits evidence

This is an additive software-verification improvement, not a new experiment.
The original PenDigits study directory, frozen source/protocol, catalog,
publication manifest, data, reports and receipts are unchanged.

## Run without training or downloads

From the repository root, with the recorded NumPy 2.3.5 already available:

```sh
OPENBLAS_NUM_THREADS=1 python -B scripts/check_pendigits_evidence.py
OPENBLAS_NUM_THREADS=1 python -O -B scripts/check_pendigits_evidence.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -B -m unittest discover -s tests -v
```

An alternate published study copy can be checked with `--study /path/to/study`.
The checker only reads that directory. Successful checks print JSON with
`status: PASS` and exit 0; an evidence failure prints `status: FAIL` with the
failing field/path and exits 1. It never rewrites a verification receipt.

The frozen [`verify_saved.py`](../studies/pendigits-recurrent-replay/verify_saved.py)
uses Python assertions. Python's `-O` flag or `PYTHONOPTIMIZE` disables those
checks: a temporary copy with a hash-corrupted report can incorrectly print
PASS. Keep that historical file unchanged and use the new checker for checks
that must remain active under optimization. This finding does not change any
published metric or demonstrate that the historical audit was run with `-O`.

## What is checked

- Exact 14-file compact catalog coverage and SHA-256 agreement; missing files,
  omitted entries and substituted symlinks fail. Hash inputs use original
  bytes, preserving the published CSV's CRLF line endings.
- Source/protocol/review hash relationships, completion-marker counts and
  source identity, and absence of an `INCOMPLETE` marker.
- Exactly one case for each of ten seeds and three arms, all ten ordered
  recurrent class pairs per case, and the complete recorded operation counts.
- All 300 boundaries' recall ranges, macro means, current-pair acquisition and
  old-seen-class means. First-phase old-class accuracy must be null.
- All final and mean-acquisition scalars, exact CSV case coverage and agreement
  with JSON, all arm means, and ten paired MIR-minus-uniform differences.
- The specified 10,000-resample, ten-paired-seed percentile bootstrap with RNG
  seed 20261025, and all three prospective support gates separately.
- Strict finite JSON numbers, unique JSON object keys, finite/nonnegative
  cross entropy, bounded micro accuracy and recorded resource-cap consistency.

Recomputed floating values are compared at absolute tolerance `1e-12` and zero
relative tolerance. Counts, identities, schedule and booleans are exact. Gate
thresholds use the original direct comparisons, without an added tolerance.
The tested runtime is Python 3.12.14 and NumPy 2.3.5; the output reports NumPy's
actual version. The bootstrap is not guaranteed bit-identical across versions.

The regression suite corrupts temporary copies, including refreshing individual
hashes to test semantic validation rather than just checksum rejection. It also
checks unchanged evidence bytes after both ordinary and optimized runs. Hand-
specified synthetic arithmetic fixtures ensure a favorable primary result alone
cannot erase an acquisition failure or an interval touching zero.

## What PASS does not establish

PASS is a software/evidence-consistency result, not support for MIR. The original
negative pilot remains unchanged: MIR minus uniform is −1.4643 percentage points,
paired-seed interval [−3.1601, +0.3314], acquisition difference −7.8846 points,
and all three support components fail. Crossing zero proves neither equivalence
nor confirmed inferiority. High-loss remains a descriptive secondary control.

This does not load original checkpoints, predictions, labels or test inputs,
recompute micro accuracy or cross entropy from predictions, reconstruct live
selection traces, attest historical execution/access timing, independently
measure resource use, or establish writer/population replication. Completion
checkpoint counts are checked as a recorded claim, not as publicly available
checkpoint files. It adds no new dataset or experiment and imports no model code.

The catalog is a consistency reference, not an externally pinned signature.
Changed but coherent evidence can pass after its catalog hashes are refreshed,
without changing this checker. In particular, a modified report with a refreshed
catalog hash can pass: report prose is hash-checked, not semantically audited.
This is consistency checking, not exact-release authentication. Use a trusted
repository revision for provenance. Original
checkpoint availability and replication boundaries remain documented in the
[study guide](../studies/pendigits-recurrent-replay/README.md).
