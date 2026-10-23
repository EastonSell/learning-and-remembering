# JOINT replay composition: a post-hoc same-state diagnostic

A one-step diagnostic on sixteen fixed transition states from four historical Fashion-MNIST seed/stream clusters. Changing update composition reverses the average MIR-versus-high-loss old-panel contrast at these states. This is not an improvement in a trained trajectory, independent confirmation, persistent retention evidence, or biological validation.

## Result, including the adverse outcomes

Old-panel macro cross-entropy differences are in nats/example; negative MIR-minus-control favors MIR.

- JOINT MIR minus high-loss: -0.095486853; descriptive four-seed bootstrap 95% interval [-0.217083497, -0.009546816].
- Same-state, same-selection fresh-then-replay MIR minus high-loss: +0.163339534 [+0.107742256, +0.236596653].
- JOINT minus sequential contrast: -0.258826387 [-0.455551976, -0.127882223]. All four seed means change sign, but the reversal is not universal per state.
- JOINT MIR minus four-draw uniform: -0.047236700 [-0.091626549, -0.003393057]. The interval does not resolve an advantage beyond the frozen 0.005-nat engineering margin.

Essential adverse context: JOINT MIR still worsens old-panel loss relative to the original checkpoint by +0.71271 nats/example (full precision +0.712708648). It also costs +0.37056 new-panel loss versus fresh-only (full precision +0.370564498), with worse new-panel outcomes in 14/16 states. Relative selector benefit is not absolute protection against forgetting. MIR loses to high-loss on old-panel loss in 6/16 states. Seed 2002 is influential. New-panel differences versus high-loss and uniform remain unresolved by their descriptive intervals.

Read [the full results and all adverse state rows](RESULTS.md), [scientific-method projection](SCIENTIFIC-PROTOCOL.md), and [full-precision saved analysis](saved-analysis.json). All sixteen raw state files preserve signed losses, scores, baseline changes, selected source IDs, fixed panel IDs and parameter-state digests exactly.

## Reproduce the saved-output arithmetic

Recorded runtime: Python 3.12.14, NumPy 2.3.5, Linux, single-thread BLAS. NumPy is the only non-standard recount dependency; requirements.txt records its version. No package installation is performed by the scripts.

From this study directory:

    python -B verify_release.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B verify_release.py --recount
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B recount_saved.py

The first command is standard-library-only. The recount reads packaged JSON and reproduces saved-analysis.json byte-for-byte. It works from any working directory when invoked by its path. Optional --output NEW_FILE writes once and refuses an existing file. It loads no dataset, checkpoint, model source, images or predictions, and performs no scientific rerun. Bootstrap calculations only recount already-saved scalar values.

Pure synthetic software tests, not scientific evidence:

    cd source
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest discover -s tests -v

The 17 synthetic tests use tiny generated arrays and temporary files only. Test count is not independent scientific sample size.

## Exact originals versus public derivatives

- Exact original bytes: sixteen state JSONs, summary.json, COMPLETE.json, saved-analysis.json, resource-ledger.json, mechanics/kernel/admission source and included synthetic tests.
- Portable derivatives: recount_saved.py changes only address/binding lookup; its arithmetic and numeric AST constants match the original. source/jointaudit/runner.py changes only three external-input path assignments and its module docstring. Its scientific function ASTs and numerical constants match the executed source. It is reference source, not an independently executed or newly frozen original.
- Explicit projections: provenance/source-input-freeze.projection.json replaces original absolute addresses with role-relative labels while preserving all 55 hash bindings and scientific metadata. Some omitted procedural metadata is represented only by opaque labels and original digests. provenance/execution-summary.projection.json selects saved execution fields without distributing original address-bearing records. Neither projection is an executed original or proof of undistributed content.
- Editorial derivatives: this README, RESULTS.md and SCIENTIFIC-PROTOCOL.md disclose their scope. The results retain all scientific findings and adverse signed values; the protocol is a scientific-method projection, not the original freeze.

transformations.json maps original and packaged SHA256/git blob identities and records AST-equivalence rules. PUBLIC-FILES.json is the exact public allowlist, excluding itself and interpreter-generated __pycache__ files. The verifier rejects unlisted files, symlinks, hash mismatches and an unexpected state inventory. No archives or chunk reconstructor are necessary for this small package.

Original COMPLETE.json remains unchanged. Its 1,060,107-byte total and run-input-freeze.json hash describe the ORIGINAL 19-artifact run directory, not this public subset. The undistributed original run-input freeze had SHA256 7e38b60132d88979e6c7c49850cc537221527fa7e4a3074196944017676dc16b. The supplied projection has different bytes, a different name and a different digest. The verifier checks 17 retained files named by COMPLETE, links its omitted freeze commitment to the labeled projection, and checks the exact completion-marker digest. It does not pretend that the public directory is the unchanged original 19-artifact bundle.

The recount's original freeze digest and observed exit 0 are explicitly recorded commitments read from those projections; they cannot be independently recomputed from undistributed originals. All row, summary and COMPLETE hashes are recomputed from packaged bytes. Reproducing saved arithmetic is not an independent numerical model-output validation.

## Data, source provenance and full-rerun boundary

The earlier [historical replay-selection study](https://github.com/EastonSell/learning-and-remembering/tree/37663be762de74aa1ba6047e01f853a1ad8b8f7d/studies/replay-selection-fashion-mnist) is pinned to repository commit 37663be762de74aa1ba6047e01f853a1ad8b8f7d. It remains unchanged, including its inconclusive primary accuracy result and adverse historical selected-versus-shuffled audit. The new sequential comparator here is MIR versus high-loss on the same selections, not a rerun of that historical estimand.

provenance/external-inputs.json identifies all 38 original external input bindings. Thirty-five match the historical public allowlist exactly, including split/duplicate metadata, four uniform-origin streams/class-pair records, sixteen checkpoints, inventories and historical kernel/runner. Some historical artifacts require that study's own evidence reconstruction. Two training assets are official pinned Fashion-MNIST downloads, not redistributed here. The remaining exact original evaluation lock is not public; the historical evaluation-lock-public.json is a labeled editorial projection and does not have the original lock hash. These are source/byte-identity checks, not a new inference pass.

Full scientific rerun is NOT packaged or validated. It would require official TRAIN images/labels, exact prepared split and duplicate metadata, historical checkpoints/streams, the exact original lock or a deliberately adapted new lock, and a new complete portable execution/freeze procedure. The reference runner points to unprovided role-relative external-inputs locations. The original scientific launcher and whole-original freeze workflow are not distributed. Supplying these assets or substituting the public lock does not make the old freeze valid. No new freeze or scientific execution was made here. The recount does not need any of them.

The data catalog documents the official pinned training download URLs and SHA256/MD5 checksums. It does not download them. The dataset is Fashion-MNIST at zalandoresearch/fashion-mnist revision b2617bb6d3ffa2e429640350f613e3291e10b141. Source code is MIT under LICENSE; official data has its upstream MIT notice, linked in the catalog and retained by the historical study.

## Inferential limits and the next question

Four seed clusters, reused historical validation/data, uniform-origin boundary-state conditioning, one model/rate/dataset, fixed overlapping panel examples and post-hoc design sharply limit inference. Sixteen states and repeated uniform draws are not independent samples. Bootstrap intervals are descriptive, not powered confirmation. The JOINT effect clears the -0.005 engineering yardstick against high-loss, but not the stricter -0.010 upper-bound sensitivity yardstick. Composition changes weighting, parameter location and step count simultaneously; no individual cause is isolated.

A useful next scientific question is a predeclared decomposition of weighting, gradient location and step count, followed by short-horizon continuation with independent retention panels. No such work is included or performed by this release.
