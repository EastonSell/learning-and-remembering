# Fashion-MNIST replay selection: an inconclusive primary result

A completed, bounded replay-selection study: 20 paired evaluation seeds, five arms, 100 runs, 500 checkpoint/prediction pairs, and 8,000 mechanism audits. This is an additive study; earlier neurofunctional-lab research is unchanged.

**MIR did not meet the prespecified support criterion.** MIR minus high-loss final macro accuracy was +0.6265 percentage points, with a paired 95% bootstrap interval [-0.359525, +1.5330375] pp. The prespecified threshold was a mean ≥1 pp and a lower interval bound >0. Neither condition was met. This is inconclusive, not equivalence or proof of zero effect.

Uniform replay was descriptively better (72.1355% versus MIR's 70.8425%); useful-compute uniform reached 75.3325%. MIR's audit selected-minus-shuffled old-class loss contrast was +0.078770, adverse to the protection interpretation. Secondary intervals are descriptive, not multiplicity-adjusted confirmatory claims. MIR is an established algorithm; this study does not validate brain-region layers or human neural mechanisms.

Read [the readable final report](results/evaluation-report-public.md) ([exact original](results/evaluation-report-v2.md)), [case table](results/evaluation-cases-v2.csv), [summary](results/evaluation-summary-v2.json), [public lock](results/evaluation-lock-public.json), and [historical protocol](source/docs/protocol.md). Protocol documents predate execution; their “not run” statements describe that historical stage, not current status. The exact report also predates final verification and refers to privately retained complete bundles; the public export qualifications below supersede its distribution descriptions, not its scientific findings.

## Get and verify the evidence

From this study directory, first reconstruct the archived evidence (the helper is standard-library-only and rejects hash mismatches and unsafe archive members):

    python unpack_evidence.py

Recorded environment: Linux, Python 3.12.14, NumPy 2.3.5, single-thread BLAS. NumPy is the only non-standard runtime dependency. The experiment's resource accounting uses Unix resource and Linux /proc facilities. requirements.txt pins the recorded NumPy; install it in your own environment if needed. No installation is performed by these scripts.

Verify every allowed file, every retained scientific artifact against its original inventory, source Python hashes, case/ledger/checkpoint schemas, rate selection, prediction-derived metrics, bootstrap comparisons, replay traces and audits:

    python verify_public.py

Additionally reproduce all 500 saved prediction arrays exactly from their retained checkpoints (inference only, no training):

    python verify_public.py --reproduce-predictions

Run the full synthetic software suite independently of official data:

    cd source
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v

The archive includes all 30 development and 100 evaluation case artifacts, checkpoints, predictions, audits and traces, the synthetic feasibility artifacts, and complete split/duplicate-integrity metadata. Official test image/label assets are included with their original MIT notice. Raw training downloads are deliberately omitted, not required to verify retained predictions. Their pinned URLs, SHA256 and published MD5 checksums are in data/prepared-v1/manifest.json; dataset revision is b2617bb6d3ffa2e429640350f613e3291e10b141 of zalandoresearch/fashion-mnist. All dataset verification metadata and required result artifacts are retained.

## What this export proves, and does not

This is an explicitly redacted historical export, NOT an unchanged original complete run bundle. Internal clearance/review correspondence, historical workflow instructions, and redundant provenance snapshots are not public. Original manifest, inventory and completion-marker bytes remain unmodified; bundle-exports.json lists every omitted provenance path and its original SHA256. Those commitments are not independently inspectable proof of the omitted text. The original verify_bundle function must fail on this redacted subset and should not be used to claim full historical verification. verify_public.py separately checks public evidence without fabricating missing records or changing the original completion claim.

PUBLIC-FILES.json is the exact public-file allowlist and hash inventory (excluding itself, declared root transport files, and generated .pyc interpreter caches). PUBLICATION-NOTES.json catalogs editorial changes and exclusions. Original source freeze 737efd1614adca5b923262e11990e054b26325ef30fad7c5fabf1534c9ef14b3 is retained as a historical hash catalog. All Python source/tests remain exact; public documentation has declared editorial changes, so a whole-tree comparison against that original freeze is not claimed. The original final report SHA256 is c9e3130b139d461af5f1193fe88e7fe410567ffaaff8e66d5a3e8f17bb2114f2.

Scientific validation of the private original bundles covered 100 evaluation cases, 500 predictions/checkpoints, all 8,000 audits and exact checkpoint prediction reproduction. The public verifier independently repeats the retained-evidence checks described above; it does not reconstruct private review correspondence. Original charged development+evaluation CPU was 1,232.275 seconds. Retention including analysis/review sidecars was 194.63 MiB under the 250 MiB cap. Resource ceilings were cooperative sampled gates, not OS-enforced hard bounds. Publication copies/transport are not experimental resource measurements.

## Running a new experiment

Historical test results are now public: a new run on this same test set is a reproducibility replication, not a fresh confirmatory discovery. Do not tune on these results and call the test unseen. Use a genuinely independent review and truthful project-local test-selection basis. No historical private approval is supplied or needed for your own new run.

Create a new source freeze and receipt only after independent review of this public source/protocol. Supply your actual review file and reviewer, not a fabricated fixture:

    python prepare_new_run.py --workspace /absolute/new-work --review-report /absolute/review.md --reviewer 'Reviewer name' --accept-reviewed-protocol --attest-no-prior-test-selection --no-prior-test-selection-basis 'Describe the truthful selection boundary for this replication'

The explicit --attest-no-prior-test-selection flag asserts that the new design was not selected using test predictions. Do not use it after test-driven method selection or tuning; a nonempty explanation alone is not a valid attestation. Publishing this replication does not restore test blindness. If that boundary cannot truthfully be asserted, the unchanged protocol cannot be used to run a new confirmatory experiment.

The helper records the new public-tree freeze, reviewer report digest, and your explicit execution decision using the unchanged coordinator's schema. It does not assert that historical test results are unseen, perform review, download data or run an experiment. For a test-only fixture, clearly label it synthetic and never use it as research approval.

Keep new data outside source/ and use new, absent outputs. From source/:

    python -m replaystudy.data --data-root /absolute/new-data
    python run.py verify-freeze --freeze /absolute/new-work/source-freeze.json
    python run.py feasibility --freeze /absolute/new-work/source-freeze.json --data-root /absolute/new-data --output /absolute/new-work/feasibility
    python run.py development --freeze /absolute/new-work/source-freeze.json --authorization /absolute/new-work/clearance.json --prepared /absolute/new-data/prepared-v1 --data-root /absolute/new-data --forecast /absolute/new-work/feasibility/forecast.json --budget-journal /absolute/new-work/budget.jsonl --output /absolute/new-work/development
    python run.py lock --freeze /absolute/new-work/source-freeze.json --authorization /absolute/new-work/clearance.json --prepared /absolute/new-data/prepared-v1 --forecast /absolute/new-work/feasibility/forecast.json --development /absolute/new-work/development --output /absolute/new-work/lock.json
    python run.py evaluation --freeze /absolute/new-work/source-freeze.json --authorization /absolute/new-work/clearance.json --prepared /absolute/new-data/prepared-v1 --data-root /absolute/new-data --forecast /absolute/new-work/feasibility/forecast.json --budget-journal /absolute/new-work/budget.jsonl --lock /absolute/new-work/lock.json --output /absolute/new-work/evaluation
    python run.py summarize --bundle /absolute/new-work/evaluation --output /absolute/new-work/summary.json

Each new stage remains subject to the unchanged validation, provenance, lifecycle and resource gates. Full original-bundle verification is available for these NEW complete local runs. This publication did not rerun research or tune any method.

## License

Project source: MIT (LICENSE). Fashion-MNIST data: MIT, copyright 2017 Zalando SE (data/LICENSE-Fashion-MNIST). Scientific references and interpretation limits: source/docs/sources.md. Preserve both notices when redistributing.
