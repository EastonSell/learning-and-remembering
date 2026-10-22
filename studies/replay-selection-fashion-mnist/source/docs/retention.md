# Retention and independent verification

`replaystudy.evidence` uses only the Python standard library and NumPy. It never
imports the learner, orchestration, data loader or scoring functions. Verification
is read-only; it cannot train, predict, infer missing rows or repair evidence.

## Public API

- `expected_cases("development")`: exactly 30 configurations: seeds 1001/1002,
  rates 0.03/0.1/0.3, and each of the five arms.
- `expected_cases("evaluation", diagnostic_eta=..., extra_eta=...)`: exactly 100
  configurations: seeds 2001–2020 and five arms. Both rates must come from the grid;
  the four diagnostic arms share the diagnostic rate.
- `make_case_id(seed, arm, eta)`: canonical identity, for example
  `seed-1001__mir__eta-0.1`. Arm spellings are `mir`, `highloss`, `uniform`,
  `shuffledscore`, `uniform-extra-replay`.
- `Bundle(path, stage, expected_cases, provenance, *, required_fields=None)`:
  exclusive writer. Positional or keyword arguments are accepted. `path` must not
  exist and its parent must exist. `Bundle.path` is a `pathlib.Path`.
- `bundle.append_case(record)`: append the whole raw record, flush and `fsync`, then
  validate it. Failed statuses, prediction checks and malformed numeric results
  are retained before an exception is raised.
- `bundle.complete()`: verify exact coverage and all evidence, synchronize artifacts,
  write their inventory and completion digest, and independently verify again.
- `verify_bundle(path, require_complete=True)`: returns a report containing stage,
  case count, canonical case IDs, completion state, manifest SHA256 and artifact
  count, or raises `VerificationError` (a subclass of `EvidenceError`/`ValueError`).
- `sha256_file(path)`: SHA256 of the file bytes.

Use the writer as a context manager around the entire run, including scoring.
Normal context exit without `complete()` is a failure, not implicit success.
`close()` alone leaves the output incomplete. A failed bundle cannot be resumed or
appended to; any justified rerun needs a new directory and its dated amendment.

## Raw case schema

Each successful case has these keys:

- `seed`, `arm`, `eta`, `case_id`: exactly the declared configuration, not just a
  loosely matching ID string.
- `status`: the literal string `"ok"`.
- `checkpoints`: exactly five objects with `block` integer 0–4, a bundle-relative
  `path`, and the file's `sha256`. Paths must be distinct across the entire bundle.
  The runner uses `<case_id>/training/block-<block>.npz`. The conventional
  `checkpoints/` directory is also supported. Additional unlisted conventional
  block checkpoint files invalidate coverage; side evidence such as selection
  traces remains permitted and is hashed in the artifact inventory.
- `metrics`: numeric results with the stage-specific fields below.

Mandatory shapes, expressed as lists in the manifest:

| Stage | Required fields | Shape |
|---|---|---|
| software | `metrics.validation_recall` | `[10]` |
| development | `metrics.validation_matrix` | `[5, 5]` |
| development | `metrics.validation_recall` | `[10]` |
| evaluation | `metrics.test_matrix`, `metrics.full_test_matrix` | `[5, 5]` each |
| evaluation | `metrics.test_recall`, `metrics.full_test_recall` | `[10]` each |

Arrays must contain finite numbers, not strings or booleans. Accuracy and recall
are fractions in [0, 1]. `required_fields={"metrics.custom": [5]}` can add numeric
shape requirements but cannot weaken mandatory stage fields. Empty lists as shape
specifications mean scalar numeric values. Other metadata, ledgers, prediction
arrays and diagnostics can be retained in the record.

Optional `prediction_checks` maps software-invariant names to booleans; any value
other than `true` fails the case after writing it. Do not use this field for the
scientific hypothesis. A null/negative scientific result is a valid result and may
be retained under `scientific_prediction` or another ordinary diagnostic field
without failing the bundle. Nonfinite raw values are preserved at their original
locations as explicit `{"__nonfinite__":"nan"}` / `inf` / `-inf` objects, so failed
records remain strict JSON rather than nonstandard JSON NaN. An unserializable
record produces a retained `serialization_failed` record with its representation.

## Research-only structural checks

Development/evaluation completion additionally requires the actual learner resource
ledger. The verifier checks 2,500 base updates, 40,000 fresh presentations, 40
current-repeat extras, 24,960 historical presentations, 10,000 unique observations,
25,000 selected base items, reservoir counter 10,000, 6,634 parameters, 26,536 raw
parameter bytes and 158,600 raw reservoir bytes. The prescribed early buffer sizes
imply 124,648 candidate draws; diagnostic arms require 2,500 virtual gradients,
249,296 candidate forward examples and 124,648 scored candidates. The extra-replay
arm requires zero diagnostic virtual gradients/candidate scoring.

Operation allowance, usage, residual and extra-update counts must be nonnegative
integers. Usage cannot exceed allowance; residual must equal allowance minus usage.
The frozen v1 arithmetic graph is independently bound to 6,200,562,008 allowed
operations per case. Diagnostic usage is the full allowance with zero residual and
zero extra updates. Uniform-extra-replay usage is 6,200,428,622, with 15,853 extra
updates and 133,386 residual operations. Internally consistent fabricated budgets
are therefore rejected. These constants follow candidate counts `[16]*8 + [32]*4
+ [48]*4 + [50]*2484`, base-update cost 713,678 and replay-only update cost 278,574.
The complete numeric ledger is checked for nonfinite failure values. These are
end-of-run consistency checks, not a reconstruction of every operation from raw
instructions; the core's prefix-budget tests provide a separate check.

For every seed, candidate, reservoir, quota and stream SHA256 fields and operation
allowance must agree across all arms and all development rates. The stream file
`<case_id>/stream-indices.npy` is rehashed and must contain exactly 10,000 distinct
integer source IDs in the official-training range 0–59,999. Candidate/reservoir/quota
digests are alignment checks on retained commitments, not independently recomputed
hashes: the compact selection trace does not retain every candidate/resident ID.

The mechanism audit file `<case_id>/mechanism-audit.jsonl` must contain exactly 80
ordered opportunities, at updates 525, 550, ..., 2,500. Every opportunity must
retain finite selected/eligible mean harm, selected-minus-shuffled old-loss change,
MIR/high-loss overlap in [0,1], and a positive integer operation cost. Rank
correlation must be finite in [-1,1] or explicitly null for ties. Update-only audit
rows cannot masquerade as completed measurements; nonfinite values are rejected. Each research checkpoint must actually be a bounded NPZ with exactly four
finite float32 arrays: p0 `(196,32)`, p1 `(32,)`, p2 `(32,10)`, p3 `(10,)`. Merely
hashing arbitrary bytes or arrays with the wrong shape/dtype cannot satisfy this
check. No test data are opened to perform these inspections. Software-stage
fixtures intentionally retain generic checkpoint-byte support and do not claim
research completion when using shortened synthetic streams.

## Provenance and completion

`provenance` maps category names to files, directories or lists of them. Required
categories are `source`, `tests`, `protocol`, `data`, `indices`, plus `lock` for
evaluation. Source/test/protocol directories are copied recursively into
`provenance/<category>/<input-number>/...`, preserving relative paths. Cache and
VCS directories (`__pycache__`, `.pytest_cache`, `.git`) are explicitly excluded.
Empty categories, symlinks and an output directory nested inside a provenance
source are rejected. Data/index manifests should retain actual index lists,
content hashes, exclusions, data versions and duplicate audits, not only seeds.
Snapshots do not depend on original host paths remaining available.

For evaluation, independent verification also checks the retained
`evaluation-lock-v1` JSON: completion and no-prior-test-selection attestations,
20 frozen seeds, exact five arms, 196–32–10 architecture, both selected rates,
case identities derived from those rates, and its data-manifest SHA256 against the
retained data provenance. An attestation records the stated gate; it is not proof
of otherwise unobserved historical actions.

The lifecycle is deliberately conservative:

1. Reserve a fresh directory with exclusive `mkdir`, write `RUNNING`, then create
   an empty `cases.jsonl` and frozen provenance/manifest before the first case.
2. Append and synchronize each case before validation or the next case.
3. On an exception preserve `RUNNING` and write `FAILED`; on `KeyboardInterrupt` or
   `SystemExit`, write `INTERRUPTED`. Never delete checkpoints or earlier cases.
4. Completion requires all and only the expected successful case identities,
   stage schema, exactly five checkpoints each, and intact provenance. Hash all
   retained artifacts into `artifacts.json`; bind that file's SHA256 in `COMPLETE`.
   Remove `RUNNING` only after the completion marker is synchronized.
5. Final verification rejects conflicting lifecycle markers, missing/empty or
   truncated JSONL, duplicate JSON keys, missing/duplicate/unexpected rows,
   changed configurations, nonfinite/wrong-shape metrics, bad checkpoint hashes,
   unsafe paths/symlinks, and missing/modified/unlisted artifacts. Marker and
   inventory files are treated specially to avoid self-hash cycles.

`require_complete=False` is a pre-finalization gate, not permission to accept an
incomplete experiment: exact successful case coverage is still mandatory.
Hash integrity detects accidental alteration, not coordinated rewriting of every
hash and marker by an adversary; authenticate the frozen manifest separately when
that threat matters. File and directory `fsync` requests durability on the local
POSIX filesystem but do not prove power-loss guarantees for every storage stack.

## Synthetic failure audit

Run `python -m unittest discover -s tests -p test_evidence.py -v`. Tests use isolated
temporary directories and synthetic bytes/metrics only, with no network or real
data scoring. They inject failures/interruptions before the first case and after a
complete case, retain wrong prediction checks and falsified scientific outcomes,
reject output reuse, exercise the complete 30/100 configuration grids, independently
corrupt records/lock fields, tamper with checkpoint/provenance bytes, and check
append-time `fsync`. A separate directory must be used for every reproduction;
compare deterministic checkpoint/prediction/order artifacts byte-for-byte while
reporting measured timing fields separately. Numerical reproductions are not new
independent samples.
