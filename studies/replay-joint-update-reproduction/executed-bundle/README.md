# Portable JOINT diagnostic reproduction candidate v2

This is a new portable, scientific-only rerun bundle for the [public JOINT diagnostic](https://github.com/EastonSell/learning-and-remembering/tree/92d7d01817c0ca4853767027dede83c53fb069ba/studies/replay-joint-update-diagnostic). It packages the historical fixed inputs and preserves the numerical diagnostic, but it is not the original execution directory, original freeze, a new scientific run, or an independently validated numerical replication.

## What is reproducible

Given the two official Fashion-MNIST TRAIN downloads, this bundle supplies all other computational inputs for the same one-step diagnostic: exact training/validation IDs and duplicate-hash arrays, four saved uniform-origin streams and class orders, sixteen fixed checkpoint files, the deployed learner kernel, selection/update/evaluation functions, bootstrap procedure, and bounded launcher.

It does not retrain the historical checkpoints, rerun the earlier 100-case experiment, or restore test blindness. Its target is exactly the JOINT-versus-sequential diagnostic at seeds 2001–2004 and blocks 0–3, eta 0.3, not a fresh confirmation. No official TEST image/label asset is included, acquired, or needed.

The adverse original findings remain unchanged: JOINT MIR worsened old-panel loss by +0.712708648 nats versus the original checkpoints and cost +0.370564498 new-panel loss versus fresh-only. See [results](RESULTS.md), [scientific method](SCIENTIFIC-PROTOCOL.md), and [saved analysis](saved-analysis.json).

## Validation actually performed

Only source/AST identity checks, whole-file hashing, standard-library integrity checks, relocation/preflight checks and tiny synthetic software tests were performed while preparing this candidate. No checkpoint parameter array or image was decoded, no model loss was scored, no research model was run, and no new experimental result was generated. The executable rerun command below was NOT run during packaging. See [validation scope](VALIDATION.md).

The mathematical functions and numerical constants match the accepted diagnostic. This is source evidence, not evidence that a newly launched numerical run completed or matched. Cross-platform BLAS/NumPy differences remain possible. A future run must retain its actual exit code, terminal evidence and comparison result.

## Prepare inputs

Use Python 3.12.14 and NumPy 2.3.5 on Linux, matching the recorded original environment as closely as practical. Only NumPy is non-standard; [requirements](requirements.txt) records the version. Scripts do not install packages. Select your own isolated environment before running.

[Acquisition instructions](ACQUIRE-TRAINING-DATA.md) provide official commit-pinned HTTPS download URLs and SHA256/MD5/size checks. Keep the two compressed training assets in a separate directory, not inside this bundle. Do not run a historical all-data preparation command: exact consumed prepared inputs are already included, and TEST assets are unnecessary.

From this bundle directory, replace /absolute/TRAIN and /absolute/NEW-WORK with your own paths. NEW-WORK must NOT exist; the full workflow below creates it atomically and stops on any setup collision. All generated files must remain outside this bundle, whose allowlist is intentionally closed.

    python -B verify_inputs.py
    python -B verify_inputs.py --training-assets /absolute/TRAIN

The above commands are standard-library-only and hash compressed bytes without decompression.

For hash-only preparation, use a separate absent HASH-ONLY-WORK directory. This block performs no model execution:

    (
    set -u
    set -C
    work=/absolute/HASH-ONLY-WORK
    train=/absolute/TRAIN
    mkdir -- "$work" || exit 1
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B run.py --freeze-inputs --training-assets "$train" --freeze "$work/reproduction-freeze.json" || exit 1
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B run.py --preflight --training-assets "$train" --freeze "$work/reproduction-freeze.json" || exit 1
    )

These modes import NumPy but never load any saved array or decode an image. The generated freeze is a NEW role-relative source/input freeze, not a recreation of the historical execution freeze. Relocating this bundle or TRAIN does not change the frozen role identities.

## Execute a new scientific reproduction, only when you intend to

This next command performs new scientific computation. It has not been executed for this candidate.

The complete workflow uses a different absent NEW-WORK directory, creates a fresh freeze, and reserves all three external receipts exclusively BEFORE invoking the scientific command. Run the entire subshell, not isolated lines. Any setup failure stops before execution. Never remove a collision automatically; choose a new workspace or inspect the existing evidence yourself.

    (
    set -u
    set -C
    work=/absolute/NEW-WORK
    train=/absolute/TRAIN
    mkdir -- "$work" || exit 1
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B run.py --freeze-inputs --training-assets "$train" --freeze "$work/reproduction-freeze.json" || exit 1
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B run.py --preflight --training-assets "$train" --freeze "$work/reproduction-freeze.json" || exit 1
    # BEGIN EXCLUSIVE RECEIPTS
    exec 3> "$work/stdout.txt" || exit 1
    exec 4> "$work/stderr.txt" || exit 1
    exec 5> "$work/exit-code.txt" || exit 1
    code=0
    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=5s 330s python -B run.py --execute-reproduction --training-assets "$train" --freeze "$work/reproduction-freeze.json" --output "$work/diagnostic" >&3 2>&4 || code=$?
    printf '%s\n' "$code" >&5 || exit 1
    exec 3>&- 4>&- 5>&-
    # END EXCLUSIVE RECEIPTS
    if [ "$code" -ne 0 ]; then
        printf 'Scientific command failed with exit %s; preserve all receipts and partial artifacts.\n' "$code" >&2
        exit "$code"
    fi
    python -B compare_reproduction.py --run "$work/diagnostic" --exit-code "$code" --output "$work/comparison.json"
    )

Do not invent a zero exit after a failed or interrupted run. No automatic scientific retry occurs. Missing COMPLETE, any FAILURE, a nonzero observed exit, or mismatched output identities invalidates that run. The launcher installs the original CPU 180s soft/181s hard, 768 MiB address-space, 32 MiB per-file, 300s signal timer and sampled aggregate-output/RSS gates before NumPy/input verification. The external timeout adds a 330s TERM and 5s KILL backstop. These are not hard aggregate-RSS or exact-wall guarantees.

The comparison requires exactly nineteen regular run files and the expected completion status, validates COMPLETE byte counts and bound files, and parses the exact new freeze schema/configuration/caps plus all 82 role bindings against this current package/manifest and the two TRAIN catalog hashes. It then compares all sixteen state JSON payloads exactly and the saved summary exactly except its measured resource sample. It reports receipt integrity, source/input-claim validation, and the caller-supplied exit separately. The comparator does not observe execution or independently prove that computation occurred; preserve the actual command receipts for that evidence. No numeric tolerance is silently accepted. Timing, new source/input-freeze bytes and completion-marker bytes are expected to differ. An exact scientific mismatch must be investigated and reported; a source-equivalent package is not a numerical pass.

## Synthetic software tests

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest discover -s tests -v

Tests use tiny generated numeric fixtures, uninterpreted hash bytes, and clearly labeled synthetic provenance fixtures containing copies of saved JSON. Copying saved science into a synthetic receipt fixture is not new scientific execution or evidence that a model ran. The two launcher-admission probes substitute a fake runner and deliberately stop before real science. No tests read parameter arrays or images from the retained scientific inputs. Software test counts are not scientific sample sizes.

## Provenance, licenses and transformations

[The input catalog](input-catalog.json) binds every included scientific input to the [historical public study](https://github.com/EastonSell/learning-and-remembering/tree/37663be762de74aa1ba6047e01f853a1ad8b8f7d/studies/replay-selection-fashion-mnist) allowlist. All sixteen checkpoint files are exact copies of already-public retained artifacts. They have not been opened as NumPy archives. Both original project MIT notices and the official Fashion-MNIST MIT notice are retained.

The redistribution basis is the historical study's published package-level MIT license and its published retained-artifact inventory, together with the upstream Fashion-MNIST MIT notice. No new license grant or third-party rights clearance is asserted. Raw TRAIN data is not redistributed here. The provenance catalog records the concrete license and identity sources, not inferred permission from mere public accessibility.

[The evaluation-lock projection](provenance/evaluation-lock.scientific-projection.json) removes four address fields and the internal provenance narrative, retaining every other historical field unchanged. Historical no-prior-test-selection is a recorded claim, not a new attestation. Its bytes and digest differ from the original lock. No private review document, historical clearance, agent instruction or machine path is needed by this launcher.

The original whole-experiment inventories and historical runner snapshot are provenance-only for this diagnostic; direct exact subset bindings replace their operational integrity role. The original kernel remains byte-identical. [Source equivalence](provenance/source-equivalence.json) records the exact function-level checks and every allowed adaptation. [The original input-freeze projection](provenance/historical-source-input-freeze.projection.json) retains all 55 historical digest commitments, including omitted provenance-only entries; commitments alone are not proof of undistributed text.

The expected/ directory contains only the seventeen saved scientific JSON files needed for comparison, not an unchanged complete original run bundle. No historical COMPLETE marker is reissued there. PACKAGE-FILES.json is a new exact candidate inventory, excluding itself and interpreter caches. It is not an execution-era freeze.

A separate factorial decomposition bundle is not included. This candidate does not expand the original study or conflate its seeds/results with a later study.

