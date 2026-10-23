# Portable JOINT replay reproduction

All sixteen state JSON files in the reported portable rerun are byte-identical to the original diagnostic evidence. Independent saved-evidence checks also find every scientific summary field exactly equal; only the explicitly excluded measured-resource sample differs. No numerical tolerance was used.

The saved numerical and integrity checks pass independently. Foreground execution with scientific exit 0 and comparison exit 0 is reported from the author's retained original tool receipt. The saved-only reviewer was not a separate live witness, and no independent OS/process attestation is supplied. The private observer/tool receipts are not distributed here.

This is an additive reproduction package for the [original JOINT diagnostic](https://github.com/EastonSell/learning-and-remembering/tree/92d7d01817c0ca4853767027dede83c53fb069ba/studies/replay-joint-update-diagnostic). It does not alter that study, its original publication, or its adverse findings. Reproducing the same historical computation is not fresh confirmation, a new seed cohort, restored test blindness, checkpoint retraining, future-retention evidence or biological validation.

## Verify the public saved evidence

With Python 3.12 available on Linux, from this package directory:

    python -B verify_release.py
    python -B verify_release.py --output /absolute/NEW-verification-report.json
    python -B -m unittest discover -s tests -v

No NumPy, raw dataset, installation, download, scientific execution or numerical-array/image decoding is needed. The verifier hashes the included binary input bytes without deserializing their numeric payloads. It checks the exact public inventory, the unchanged executed source bundle, the complete nineteen-artifact numerical result, all 82 source/input-freeze bindings, sixteen byte-identical state files, the exact scientific-summary projection, and the retained comparator result. It works from a different working directory when invoked by its full path. Optional --output is exclusive and refuses an existing destination; keep generated files outside this closed package.

The three wrapper tests exercise saved-only verification, inventory corruption and no-overwrite behavior. They are software checks, not additional scientific samples. The verifier explicitly reports execution_independently_attested=false and does not use the comparator's caller-supplied zero as proof that a process ran.

## What was reproduced

The target was the same one-step diagnostic at uniform-origin historical checkpoints: seeds 2001–2004, blocks 0–3, eta 0.3, sixteen dependent states and four reused seed/stream clusters. Selection, fresh batch, replay quota, evaluator panels, kernel and numerical functions remain fixed.

- Sixteen complete state JSON files: exact original bytes.
- Scientific summary: exact equality after removing only resources_sampled_before_summary_write.
- Canonical scientific-summary SHA256: fa9f4312192fc5289610d2d8c6d033878e74b4d462a0f8badd122b9388f1fbf6.
- New run-input freeze: fd604745e8986617d22c2e736fc44519000a07a753b51c3477538e3b09d2ed16.
- New COMPLETE: 7c66754505b6efa6440ad5dfb4199a251836457f3223a01d7bb782820bddfc66.
- Nineteen scientific artifacts, 1,060,646 bytes including COMPLETE; no FAILURE.
- The new source/input freeze is not the original address-bearing execution freeze.

The original adverse outcomes remain essential: JOINT MIR increases old-panel macro loss by +0.712708648 nats versus the original checkpoints and costs +0.370564498 new-panel loss versus fresh-only SGD. Relative selector benefit is not absolute forgetting protection or a general plasticity advantage. [Full signed results](executed-bundle/RESULTS.md), [saved analysis](executed-bundle/saved-analysis.json) and [method/limitations](executed-bundle/SCIENTIFIC-PROTOCOL.md) remain intact.

The scientific-summary projection uses UTF-8 JSON with sorted keys, comma/colon separators, allow_nan=False and a final LF. The exact [saved comparison](comparison.json), [independent numerical comparison](provenance/independent-scientific-comparison.json), [scientific execution projection](provenance/reproduction.scientific-projection.json) and [saved-only review projection](provenance/postrun-review.scientific-projection.json) document the evidence and its limits.

## Resources and observation boundary

Recorded before completion-marker write: CPU 1.588399489 seconds, wall 1.607101257002796 seconds, RSS high-water 95,006,720 bytes, existing output 1,057,586 bytes. These are pre-completion samples, not final process-end measurements. The summary's earlier resource sample also differs from the original run, as expected. No scientific summary field was excluded other than that resource-sample field.

The unchanged launcher uses CPU soft/hard limits 180/181 seconds, 768 MiB address space, 32 MiB per-file size, a 300-second signal timer, one thread, sampled RSS/aggregate-output gates, and an external 330-second TERM/+5-second KILL backstop. Sampled RSS/aggregate-output and signal timers are not hard aggregate-RSS or exact-wall guarantees. External execution receipts are outside the scientific-directory byte ledger and are deliberately omitted from this public derivative.

The new scientific directory is 539 bytes larger than the original 1,060,107-byte directory. That difference is exactly the new role-relative freeze's 10,259 bytes minus the original freeze's 9,720 bytes, not unexplained scientific output.

Independent review checked saved artifacts, source/input identities, receipt consistency and exact numerical JSON agreement. It did not rerun a model, recompute losses/gradients, witness the original live command, instrument runtime file I/O or attest the historical origin of a private observation file. The reviewed source/recipe is TRAIN-only; that is a source conclusion, not a syscall-trace claim. Original private receipts are represented only by clearly labeled public scientific projections and opaque source commitments.

## Portable source and inputs

[executed-bundle/](executed-bundle/README.md) is the exact eighty-file v2 bundle used by the reported rerun. Its PACKAGE-FILES.json SHA256 remains 3cb7c99ff63c95d0a24cf8c77fa8fabbb3bed96d699ac26c0176bbba37aace0f. Its pre-execution README and VALIDATION are preserved source artifacts describing the validation state when that bundle was frozen, not this outer release's latest result.

Keeping this immutable inner bundle is intentional: changing its documentation would change source/input bindings and invalidate the accepted new-run freeze. This outer README is the current evidence guide. The saved run freeze's role names resolve relative to executed-bundle, not to this outer directory.

All non-TRAIN computational inputs are included: exact split/duplicate-hash metadata, streams/class orders, sixteen checkpoints, deployed kernel and scientific source. The historical public input/license chain and transparent evaluation-lock projection remain in [the input catalog](executed-bundle/input-catalog.json) and [source-equivalence record](executed-bundle/provenance/source-equivalence.json). No historical private lock prose, machine path, agent instruction or clearance record is required.

A researcher who deliberately wants a NEW scientific run can follow the unchanged [TRAIN-only acquisition](executed-bundle/ACQUIRE-TRAINING-DATA.md) and [exclusive launch recipe](executed-bundle/README.md), working from executed-bundle with a new external workspace. That is separate computation, not what verify_release.py does. The recorded environment was Linux x86_64, Python 3.12.14, NumPy 2.3.5, single-thread BLAS. Scripts install nothing. No TEST asset is needed. Public results do not become unseen again.

## Release integrity and provenance

PUBLIC-FILES.json is this new outer release's exact allowlist, excluding itself and interpreter caches. [transformations.json](transformations.json) maps every exact copy and every new projection/document to its source identity and transformation. The executed v2 bundle and complete numerical run remain exact bytes; no old freeze, completion marker or private observer record is relabeled as a new original.

This package contains no private launch script, prelaunch machine/runtime paths, original tool response, chunk identifier, observer-routing data, external receipt bundle or private review text. Review projections identify their omitted originals by digest and state their selection rules; those commitments alone are not independently inspectable proof of omitted text.

Project source and original work retain the [MIT notice](LICENSE); [historical project](executed-bundle/LICENSE-historical-study) and [Fashion-MNIST](executed-bundle/inputs/data/LICENSE-Fashion-MNIST) notices are preserved. Raw TRAIN downloads are omitted. No new license grant or unrelated rights clearance is asserted.

