# Useful-compute PenDigits extension

**Completed and independently audited; primary MIR hypothesis unsupported.** Read the [result](REPORT.md), [all36 case rows](results/all-36-cases.csv), [all boundary metrics and accounting](results/cases.json), [primary paired outcomes](results/paired-mir-minus-uniform_credit.csv), and [descriptive paired outcomes](results/paired-mir-minus-uniform_base.csv).

The implementation, [numeric protocol](protocol.json), original synthetic tests and all numeric outcomes are byte-identical to the executed source. The public [README.md](README.md) and [PROTOCOL.md](PROTOCOL.md) are explicitly postrun prose-redaction derivatives: unrelated private workflow text was removed and provenance labels clarified, with no experimental rule changed. The exact original prose remains in the verified private archive. [SOURCE_PRE_DATA_MANIFEST.json](SOURCE_PRE_DATA_MANIFEST.json) is the byte-identical original predata manifest; [PRE_DATA_MANIFEST.json](PRE_DATA_MANIFEST.json) instead binds this public derivative and is not the original preregistration. Original result/admission hashes continue to refer to the source freeze. Added derivative-integrity tests ensure the old admission is rejected for the new public manifest and any future run requires a new matching review/admission. These changes do not rerun or amend the experiment.

## What is available

- Public: exact executed implementation/numeric protocol/original tests, explicitly prose-redacted protocol/README derivatives; all36 case records with360 boundary metrics, full sample/work counters, timings and signed paired results; source input hashes; resource/supervision summaries; independent predata and postrun audit summaries; a complete public file-hash manifest.
- Retained in a durable private evidence archive: all360 weight/prediction/probability checkpoints, split indices, phase IDs, retained test inputs, raw UCI train/test members, exact unredacted provenance/admission/review records and process-supervision evidence. It is not anonymously downloadable from this repository.
- Public environment/supervisor/reviewer metadata are explicitly labeled derivatives. Local paths and internal role labels were generalized; unrelated private workflow statements were omitted from the public protocol/README and review summary. Provenance descriptions now distinguish the original source freeze from this postrun derivative. Numeric outcomes, numeric protocol and executed implementation were not redacted or rewritten. Original and derivative digests are recorded in PUBLIC_PROVENANCE_DERIVATIONS.json. Public completion metadata is named completion-public.json, not presented as the original byte-identical COMPLETE.json.

Full archive SHA256:62a93f6e69ad955344e04c3e6dc2c260e3faa62ce0a6233f46246180c42ef44d; size104,045,357 bytes. Archive inventory SHA256:159ee322a75138dbfbf53e9d62c0d2c6a7bc101a457035b4a891112373bd8cab. These hashes identify the retained original evidence, not public availability.

## Reproduction levels

1. Saved arithmetic: inspect cases.json and summary.json; recompute paired statistics with engine.analyze and accounting with engine.validate_accounting. This requires no fitting, raw data or checkpoint binaries, but it is consistency checking, not independent inference or release authentication.
2. Software checks: run `python -B run.py --selftest` and `python -O -B run.py --selftest` from this directory. All generated test data is synthetic, with temporary outputs explicitly labeled synthetic. Passing tests is not an efficacy result.
3. Checkpoint inference: requires the retained full binary archive; recompute from the saved weights and exact test inputs. The independent audit performed this level for every checkpoint.
4. Fresh training: requires pinned raw source members, the frozen runtime, a new explicitly authorized output directory and independently reviewed admission. The public package does not automatically download inputs or admit another run. A newly rerun experiment must not be silently substituted for this one.

Data: [UCI Pen-Based Recognition of Handwritten Digits](https://doi.org/10.24432/C5MG6K), CC BY4.0, Alpaydin and Alimoglu. Original public archive SHA256:1e02bea023613c2b11c9492f6f34caf975420455934f3527d270cee9a1f03b64. Training and test member hashes are pinned in protocol.json. Raw source bytes were not reconstructed or rewritten.

This additive extension preserves the original study. It is a new-seed comparison on already studied data, with a fixed matrix-MAC proxy and separately measured one-environment runtime, not a general efficiency winner.
