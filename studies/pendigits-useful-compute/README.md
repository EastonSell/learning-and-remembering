# PenDigits useful-compute control

This is a public prose-redaction derivative prepared after the completed new-seed extension. Unrelated private workflow text was removed. Experimental implementation, numeric configuration and results are unchanged. The exact original predata README remains in the verified private evidence archive. SOURCE_PRE_DATA_MANIFEST.json preserves its original digest; PRE_DATA_MANIFEST.json binds this postrun public derivative, not the original preregistration. See PUBLICATION.md for current status.

The runnable derivative compares MIR with (1) uniform replay that does no discarded selector work and (2) useful uniform plus extra replay funded by a fixed matrix-MAC ledger. It records accuracy and actual runtime separately. Proxy work matching is not wallclock, energy, or full-algorithm compute matching. Read PROTOCOL.md for every frozen choice and claim boundary.

## Synthetic verification only

From this directory:

    OPENBLAS_NUM_THREADS=1 python -B run.py --selftest
    OPENBLAS_NUM_THREADS=1 python -O -B run.py --selftest

These tests include a synthetic 36-case / 360-boundary run and failure/corruption controls. All synthetic outputs live in temporary directories and are labeled synthetic. No data files or network access are required. Passing these tests does not establish efficacy.

## Gate for a later admitted run

A separate reviewer must review the exact PRE_DATA_MANIFEST.json hash, code, protocol, tests and provenance caveat. The reviewer supplies a review JSON with status APPROVED, manifest_sha256 and reviewer. A separately authorized coordinator supplies an admission JSON with status APPROVED_FOR_EXECUTION, the same manifest_sha256, review_path and review_sha256. The runner binds those files to the exact frozen implementation before it opens either data member. These integrity gates do not themselves grant user authorization; do not self-approve a run.

Only after admission, provide the exact pinned raw pendigits.tra and pendigits.tes members via an authorized source. The public repository intentionally does not contain those raw inputs. The CLI does not download, reconstruct, substitute or silently search for them. Do not claim this package alone can execute the real experiment without supplying those inputs. The complete CLI form, for a later admitted run, is:

    python -B run.py --execute-admitted --admission /path/to/admission.json --training /path/to/pendigits.tra --test /path/to/pendigits.tes --output /path/to/new-output

Use an external 600-second supervisor as well as the cooperative cap. Record the supervisor's actual command and termination outcome. A timeout, invalid source, missing case, bad accounting or nonfinite outcome is INCOMPLETE, not a negative completed result or a usable support decision. Preserve the output directory and all attempts. Never resume by overwriting or silently retrying it.

No additional empirical execution was performed to prepare this public redaction derivative. The completed study and its original admission bind the source freeze, not this new public manifest. A future rerun requires a separate review and admission binding the current derivative hash.
