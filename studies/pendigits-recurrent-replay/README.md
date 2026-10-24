# PenDigits recurrent replay: an actual negative empirical pilot

Completed 24 October 2026. Thirty newly trained MLPs, ten paired seeds, one previously unused trajectory dataset. This is an empirical pilot, not the earlier toy animation or a reanalysis of old checkpoints.

**MIR did not meet the prospective support criterion.** Final held-out macro accuracy was 79.94% for uniform, 78.48% for MIR, and 85.65% for the descriptive high-loss control. MIR minus uniform was −1.46 percentage points, paired-seed 95% bootstrap interval [−3.16, +0.33]. Current-pair acquisition was 7.88 points lower with MIR. The final-accuracy interval crosses zero: do not claim equivalence, confirmed inferiority, or a general harm effect. The acquisition guardrail failed. High-loss is not a post-hoc confirmatory winner.

Read [the full report and limitations](REPORT.md), [prospective protocol](PROTOCOL.md), [pre-outcome review](independent-review.json), [freeze](freeze.json), [complete seed/phase metrics](results/metrics.json), [seed CSV](results/seed-results.csv), and [independent post-run audit](results/independent-audit.json).

## What is bundled

This Git study contains the original frozen experiment and protocol, pre-run review and freeze, execution log, all 30 seed/arm and 300 boundary scalar results, bootstrap summary, completion marker, original verification code/receipts, and a new portable saved-text verifier. Frozen files have not been rewritten to change their historical status. In particular, the protocol says “not yet executed” because that is the document reviewed before execution.

The 300 binary checkpoint/prediction pairs, split indices, saved test inputs and raw training member are **not bundled in Git or the compact report ZIP**. A separate full evidence ZIP is preserved for the owner in Library; Library retention is not a public download service. Its availability and checksum are described in ARCHIVE.md. A public checkout can recompute the reported primary comparison and acquisition summary from saved scalar metrics and can run a labeled replication using the frozen source plus the official data download below. A public reader without the full archive cannot perform inference against the original saved weights. Do not confuse regeneration with access to those originals.

## A. Verify published evidence without training

With NumPy 2.3.5 available, from this directory:

```sh
OPENBLAS_NUM_THREADS=1 python -B verify_saved.py
```

This verifies the original compact evidence catalog, all 30 case identities, boundary recall arithmetic, matching accounting and the 10,000-resample paired-seed bootstrap. It neither downloads data nor trains or runs inference. It is not independent evidence that the original execution occurred, nor a cryptographically trusted timestamp service.

## B. Verify original predictions from the full archive

If the separate full archive is available, unpack it into a new directory. In its `run` directory, with the recorded dependencies:

```sh
OPENBLAS_NUM_THREADS=1 python -B verify_results.py
```

This recomputes predictions and probabilities from all 300 original checkpoints and checks exact equality, then rewrites its verification receipt. Use a disposable extracted copy if preserving archive bytes matters. The historical `independent_outcome_audit.py` also asserts original filesystem timestamp ordering: that check is **not portable after copying or extracting**, so do not run it on a checkout and interpret a timestamp failure as a scientific discrepancy. The saved audit receipt records what was checked in the original environment. The receipt's independent official-data re-download check was performed separately after the script, not by that script itself.

## C. Intentionally retrain a labeled replication

No training was rerun for publication. These instructions are for an intentional new replication, not saved-evidence verification. Published test outcomes are now known; copying the old freeze does not create a new preregistration. Report any subsequent run as a replication, with its actual time, environment and changes.

Use Linux, Python 3.12 and the recorded NumPy/scikit-learn versions in an isolated environment. NumPy/BLAS/platform changes can change floating-point results. Choose a new empty workspace, preserving this checkout. Copy `experiment.py`, `PROTOCOL.md` and `freeze.json` unchanged into its `run` subdirectory. Save the following as `download_data.py` in the new workspace and run it there:

```python
import hashlib, io, json, pathlib, urllib.request, zipfile
root = pathlib.Path(__file__).resolve().parent
freeze = json.loads((root / 'run/freeze.json').read_text())
archive = urllib.request.urlopen(freeze['source_url'], timeout=30).read()
assert hashlib.sha256(archive).hexdigest() == freeze['archive_sha256']
raw = zipfile.ZipFile(io.BytesIO(archive)).read('pendigits.tra')
assert hashlib.sha256(raw).hexdigest() == freeze['training_sha256']
with (root / 'pendigits.tra').open('xb') as f:
    f.write(raw)
```

Then from that new workspace:

```sh
python download_data.py
OPENBLAS_NUM_THREADS=1 python run/experiment.py --selftest
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 600s python run/experiment.py --run
```

The frozen source reads `pendigits.tra` one directory above itself and downloads/hash-checks the official archive for the test member. Network access to UCI is required. It refuses to overwrite an existing `run/results` directory. The 600-second external timeout supplements its cooperative time/RSS guards. Successful completion requires `results/COMPLETE.json` and no `INCOMPLETE` marker. A failure is not a completed study; do not remove the failure marker and pretend otherwise. The original 17.46-second runtime and 133.6 MiB peak RSS are observations on one cloud CPU, not guarantees for a replication. Copy `verify_results.py` to the new `run` directory for inference verification after completion.

## Provenance, licenses and interpretation

- Data: Alpaydin and Alimoglu, *Pen-Based Recognition of Handwritten Digits*, UCI Machine Learning Repository, [DOI 10.24432/C5MG6K](https://doi.org/10.24432/C5MG6K), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Official archive URL and raw archive/training hashes are in `freeze.json`; test-member hash is in `results/summary.json`. Scaled coordinates and saved test labels inherit dataset attribution obligations.
- Original study code, documentation and trained weights follow the repository's [MIT license](../../LICENSE). NumPy and scikit-learn have their own licenses. This does not relicense the UCI data or dependencies.
- Source/protocol hashes were reviewed and frozen before test-member access according to the contemporaneous records and inspected control flow. Filesystem ordering is supporting evidence, not tamper-proof independent attestation of all historical access. Review and audit were separate automated checks, not external peer review.
- One dataset, one row-stratified train/development split, ten algorithmic seeds and imposed recurrence are not population or writer replication. The official test is provider-described independent-writer; writer IDs were not reconstructed. Development was unused for tuning.
- Uniform and high-loss deliberately compute unused MIR-style scores. Matching those operations isolates the selector but does not establish useful-compute efficiency or fixed-wall-clock superiority. No practical efficiency or LLM/neuroscience claim follows.
- Per-step selection traces were not persisted. Independent stream reconstruction and frozen shared-loop inspection verify counts, not a complete live runtime trace. Resource usage was producer-measured.

This additive study does not replace or revise the earlier Optdigits, Fashion-MNIST or historical one-step findings.
