# Prospective implementation resolutions, 2026-10-22

These clarifications were written before any Fashion-MNIST training or model
predictions. The original proposal remains preserved as protocol.md. They resolve
implementation details, not outcome-driven changes. Source freeze identifies the
binding combined protocol.

## Unambiguous rules

- Global update indices are zero-based for candidate/quota rotation and one-based
  in audit reports. First-batch candidates are the sixteen current observations;
  all four diagnostic arms still execute a fresh virtual update and both candidate
  forward passes. Their ten committed additional examples are deterministic current
  repetition, overriding ranking/quota selection only in that first batch. The
  useful-compute arm's corresponding reference allowance uses sixteen candidates.
- The useful-compute arm retains the same base 50-candidate sampler and base class
  quotas as the other arms, but no scores. Extra replay-only steps select directly
  from the reservoir with the same rotating class cycle and independent RNG. All
  extra steps during the first batch use the deterministic current cycle.
- PCG64 streams use SeedSequence([run seed, named integer]): init=1, stream=2,
  reservoir=3, candidate=4, uniform=5, shuffle=6, extra=7, audit=8. A single stream
  RNG permutes labels then each class-pair block. Source indices break score ties.
  Candidate/reservoir/quota hashes establish paired realized schedules.
- Audit opportunities are updates 525, 550, ..., 2500 (80/arm). Old classes are
  blocks completed before the current block. The audit uses first ten validation
  indices in each class's saved ordering, exactly 100 images. Within-class shuffled
  replay means permuting candidate interference scores then selecting the same
  quotas, NOT merely reordering an unchanged selected set. The branch contrast is
  selected-minus-shuffled old-class cross-entropy after a fresh-only then replay-only
  sequence. Lower is better. Candidate rank versus exhaustive harm is explicitly
  tautological for identical candidates, an implementation/coverage diagnostic.
- All evaluation training/checkpoints finish before the first official prediction.
  Test predictions are generated once per checkpoint; the retained-subset metrics
  reuse those predictions. No class masks or task-specific heads are used.
- Immediate acquisition is mean diagonal block accuracy. Forgetting averages the
  first four blocks' best post-acquisition, pre-final boundary accuracy minus final
  boundary accuracy, without clipping. For block k (zero-based), take max of rows
  k..3 in its column, then subtract row 4. Excluding pre-acquisition and final rows
  resolves the original ambiguity and allows negative forgetting for improvement. Per-class final recall and
  full/deduplicated 5x5 matrices are retained.
- Evaluation rate selection, seed list and interpretation thresholds are fixed in
  protocol.md. Exact double-precision mean ties choose smaller eta. No additional
  seeds, outcome restarts or exploratory methods are supported by these commands.

## Provenance scope

The study provenance attestation states that Fashion-MNIST test predictions were not used to select this
study within this research program. A read-only audit of eight reachable Git commits,
current source/results/locks under the prior neurofunctional-lab repository and its
publication receipts found only Optdigits or synthetic executed experiments.
README.md:7 and docs/next_experiment.md:3,7 describe Fashion-MNIST as unrun/not
previously downloaded. All nine publication receipt hashes matched referenced files.
This is project-local evidence, not a claim that the public benchmark is globally
unseen, absent from pretraining or unused in unrelated projects. The official test
is still locked until reviewed development and a complete evaluation lock.

## State and access boundaries

The learner has 6,634 trainable float32 scalars (26,536 bytes), a capacity-200
reservoir with features/uint8 labels/uint64 indices (158,600 bytes), reservoir
admission counter and separate RNG states. There are no priorities, historical
loss caches, logits, momentum, neural labels or learned selector parameters.
Counters and RNG states are mutable nontrainable state. The cost residual is also
mutable nontrainable state in the useful-compute arm and must not be confused with
memory-free computation. Resident arrays plus model total 185,136 raw bytes;
array headers, Python objects/RNG states and transient allocations are additional.

Selection traces are append-only from the learner. Frequency/coverage arrays are
created by the coordinator AFTER training, not maintained as readable learner
memory. The trace does not flow back into training. The learner receives no
validation/test arrays. The coordinator supplies one fresh batch at a time within
train_stream; full input arrays are host data delivery, not replay-accessible
history. This is auditable API/control-flow separation in trusted Python, not an
OS-enforced information-flow sandbox. The audit callback gets copies and returns
nothing. It has validation access but cannot mutate learner parameters. Test arrays
are loaded only by the locked coordinator after all evaluation training is done.

Mutable metadata (counters, hashes and RNG state) is counted/disclosed separately
from write-only evidence. Native allocator/BLAS bookkeeping is not fully explained
by raw ndarray bytes. Tracemalloc peak is the measured Python/NumPy allocator
high-water indicator, sampled RSS is observed process memory, neither a claim of
an exact OS-wide allocation inventory nor a hard memory sandbox.

## Failure and resource rules

Every new bundle requires an absent output directory; successful and failed
artifacts are retained. A failed run makes the entire research bundle incomplete.
An interrupted budget reservation requires reconciliation before a new run, not
silent budget resetting. CPU is charged across the common append-only journal;
outputs and source freezes are never overwritten. No removal of seeds, case rows,
poor eta candidates or disappointing outcomes is permitted. Changed code requires a
new freeze/review and dated amendment; rerun the complete affected paired comparison.

One synthetic feasibility pass (no research dataset predictions) forecasts all 30
development and 100 evaluation runs. It must finish within 15 minutes, forecast
<=7,200 CPU seconds, <=250 MiB including data and artifacts, and <=1 GiB RSS. The
launcher sets BLAS/OMP threads to one and execution verifies <=2 actual threads.
Training samples RSS every update and checks remaining CPU budget. These cooperative
checks can detect an overrun after a transient allocation; they are not hard OS caps.
Audit/scoring time is separately disclosed. The exact logical compute budget below
is not a claim of equal measured wall time or hardware instruction counts.
