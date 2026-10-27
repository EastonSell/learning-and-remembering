# Learning and Remembering

## Abstract

Small controlled neural-network studies examine what helps a learner acquire new classes without losing earlier ones. The replay studies distinguish **which examples stay in memory**, **which stored examples are rehearsed**, and **how much training work each method receives**. Historical coverage helped in a class-blocked digit stream, but selecting examples by predicted interference (MIR) did not demonstrate an advantage in the published recurrent PenDigits comparisons. In the latest 12-paired-seed, 36-model control, MIR reached 77.7495% final macro recall versus 89.2730% for uniform replay with extra updates credited by a fixed matrix-work allowance. This is a negative result within the tested setup, not a general efficiency ranking. A separate synthetic role-boundary branch remains confounded by predictive competence and engineered alignment. [Current replay report](studies/pendigits-useful-compute/REPORT.md) · [Coverage evidence](docs/replay_results.md) · [Role controls](docs/competence_controls.md).

## Research question

**In the tested recurrent digit-classification streams, does selecting replay examples by predicted interference improve retention over uniform replay after separating memory coverage, example selection and allowed training work?**

The broader motivation is learning without catastrophic forgetting. The finite question here concerns final macro recall and current-class acquisition in small classifiers, not human memory or language-model deployment. MIR (maximally interfered retrieval) selects examples whose loss is predicted to increase after a virtual update; it is not simply replaying the examples with the highest current loss. [Concept guide](docs/START_HERE.md).

## Methods and evidence scope

- **Coverage:** the earlier Optdigits study compares equal-sized reservoir and FIFO memories with a current-only control across ten paired seeds. This tests historical coverage under an imposed class-blocked stream, not MIR efficacy. [Design and results](docs/replay_results.md).
- **Selection under recurrence:** the original PenDigits pilot uses ten paired seeds and three arms (30 models), with the same reservoir and deliberately matched selector computations. Its class pairs recur in an imposed schedule. [Protocol](studies/pendigits-recurrent-replay/PROTOCOL.md).
- **Useful training work:** the prospective PenDigits extension uses 12 new paired seeds, three arms and 360 boundary checkpoints. It compares MIR with uniform replay without discarded selector work, and with uniform replay receiving credited extra replay-only updates. The fixed matrix-MAC accounting and measured training time are separate quantities. This is a new-seed extension on a reused dataset, not unseen-data replication. [Protocol and accounting](studies/pendigits-useful-compute/REPORT.md#what-the-control-changes).

## Answer and current results

**No MIR advantage was demonstrated in the published PenDigits comparisons.** The original recurrent pilot did not support MIR, and the prospective useful-compute extension favored credited uniform on the primary accuracy comparison. The latter completed all **36 models across 12 paired seeds**, with these mean final macro recalls:

| Replay arm | Final macro recall |
|---|---:|
| MIR | 77.7495% |
| Useful uniform, equal base updates | 80.2649% |
| Useful uniform, credited extra replay | 89.2730% |

The primary MIR-minus-credited-uniform difference was **−11.5236 percentage points**, with a paired-seed 95% percentile-bootstrap interval of **[−12.9609, −10.0623]**. The prespecified accuracy-support and accuracy/runtime-dominance gates both failed. Credited uniform was more accurate but slower; equal-update uniform was descriptively both more accurate and faster than MIR. These are different comparisons, not a universal winner. [Complete signed results](studies/pendigits-useful-compute/REPORT.md) · [Publication and reproduction boundaries](studies/pendigits-useful-compute/PUBLICATION.md).

**The allowance is a fixed matrix-MAC proxy, not full-compute, energy or wallclock matching.** The short in-process training timings exclude boundary evaluation and checkpoint serialization; they do not establish hardware or end-to-end deployment efficiency. Seed intervals describe this one dataset/setup, not independent writer or population replication. [Accounting exclusions and timing limits](studies/pendigits-useful-compute/REPORT.md).

### Original recurrent PenDigits pilot

The earlier **30-model, ten-paired-seed** pilot found final macro accuracy of 78.48% for MIR versus 79.94% for uniform replay: **−1.4643 percentage points**, paired-seed 95% bootstrap interval **[−3.1601, +0.3314]**. Current-pair acquisition was 7.88 points lower with MIR. The interval crossing zero is not proof of equivalence or confirmed inferiority. High-loss reached 85.65% descriptively, not as a confirmed winner. [Full report and limitations](studies/pendigits-recurrent-replay/REPORT.md), [frozen protocol](studies/pendigits-recurrent-replay/PROTOCOL.md), and [reproduction levels](studies/pendigits-recurrent-replay/README.md#what-is-bundled) preserve the original evidence; the extension does not replace it.

## Earlier findings and separate role study

**Keeping historical coverage helped in the digit stream.** Across ten paired seeds, final macro accuracy was 88.44% for reservoir replay, 21.16% for equally sized FIFO memory and 19.70% for current-only repetition. Immediate acquisition was strong in every arm. The [digit report](docs/replay_results.md) links the selection lock, all development/evaluation runs and reproduction instructions.

**More elaborate selection did not clear the Fashion-MNIST study's support criterion.** Across 20 paired seeds and 100 runs, maximally interfered retrieval (MIR) exceeded high-loss replay by 0.6265 percentage points; the paired 95% bootstrap interval was −0.3595 to +1.5330 points. The primary result is inconclusive. Uniform replay had a higher descriptive mean, and the prespecified protection audit was adverse to MIR. [All results and compute controls](studies/replay-selection-fashion-mnist/results/evaluation-report-public.md) are retained.

**Engineered role boundaries affected both prediction and interventions.** The [six-architecture pilot](docs/results.md), [equal-parameter role controls](docs/role_controls.md) and [validation-matched competence study](docs/competence_controls.md) trace this dependence. Matching was incomplete, and one intervention diagnostic follows algebraically from the supplied masks. A competence-independent interpretability advantage remains unestablished.

These are task-specific computational findings. The [biological evidence map](research/replay-mechanism-map.md) explains the motivation and where the analogy ends.

## Reproduce

### 1. Check saved results without training or inference

With Python and NumPy already available, check the original PenDigits pilot from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python -B scripts/check_pendigits_evidence.py
OPENBLAS_NUM_THREADS=1 python -O -B scripts/check_pendigits_evidence.py
```

The [fail-closed checker](docs/PENDIGITS_EVIDENCE_CHECKS.md) verifies the 30 cases, 300 boundaries, hashes, completion consistency, saved arithmetic, CSV agreement and paired bootstrap. It remains active under optimization and does not import model code. PASS means saved-evidence consistency, not MIR efficacy or independent checkpoint verification.

For the **36-model useful-compute extension**, recompute saved statistics and validate its work ledger from the study directory:

```sh
cd studies/pendigits-useful-compute
OPENBLAS_NUM_THREADS=1 python -B - <<'PY'
import json
from pathlib import Path
import engine

p = json.loads(Path('protocol.json').read_text())
rows = json.loads(Path('results/cases.json').read_text())
engine.validate_accounting(rows, p['seeds'])
actual = engine.analyze(rows, p['seeds'], p['bootstrap_seed'], p['bootstrap_replicates'])
expected = json.loads(Path('results/summary.json').read_text())
expected.pop('case_loop_seconds')  # Recorded elapsed time is not recomputed.
if actual != expected:
    raise SystemExit('FAIL: recomputed saved statistics differ')
print('PASS: saved accounting and paired statistics agree; no fitting or inference')
PY
```

This exact comparison uses the recorded Python 3.12.14 / NumPy 2.3.5 environment; bootstrap values are not guaranteed bit-identical across versions. It checks consistency, not release authentication or historical execution. Neither command remeasures runtime. Original source freezes, study files and historical manifests should remain unchanged; they do not form a current whole-repository manifest. [Evidence guide](docs/EVIDENCE_MAP.md#6-which-document-is-current).

### 2. Test software using synthetic fixtures

From `studies/pendigits-useful-compute`, with NumPy 2.3.5 and scikit-learn 1.8.0 already available:

```sh
OPENBLAS_NUM_THREADS=1 python -B run.py --selftest
OPENBLAS_NUM_THREADS=1 python -O -B run.py --selftest
```

These tests include small synthetic training runs in temporary directories, failure controls and admission-integrity checks. They require no real data or downloads and are **software checks, not new empirical evidence**.

### 3. Distinguish checkpoint inference from fresh training

The public PenDigits packages contain numeric results but not the original raw inputs and checkpoint binaries needed for independent checkpoint inference. The [current publication guide](studies/pendigits-useful-compute/PUBLICATION.md#reproduction-levels) distinguishes saved arithmetic, synthetic tests, full-checkpoint inference and fresh training. A fresh useful-compute run requires pinned inputs and a separately reviewed admission; it is not the default verification route. Other studies have their own [reproduction levels](docs/EVIDENCE_MAP.md#5-choose-a-reproduction-level-before-running-a-command).

To intentionally retrain only the **separate synthetic role pilot**, use the following from the repository root with Python and NumPy available; the full repository test suite also needs scikit-learn. This block starts with training and is not a saved-only verification route:

```sh
bash scripts/run_pilot.sh --output results/local-reproduction
PYTHONPATH=src python scripts/verify_run.py results/pilot \
  --allow-missing-checkpoints --compare results/local-reproduction
PYTHONPATH=src python -m unittest discover -s tests -v
```

The recorded runtime was Python 3.12.14 with NumPy 2.3.5; `requirements.txt` pins NumPy. These commands perform no downloads. Use a fresh output directory. The text-only pilot publication omits binary checkpoints; [availability and regeneration](docs/checkpoints.md) explain that boundary. The [verifier correction](docs/audit_coverage_fix.md) documents its exact-coverage checks. Each real-data study has separate setup and evidence-verification instructions.

## Limitations and open questions

PenDigits already tests **imposed class recurrence**; recurrence itself is not an untested condition here. The remaining transfer question is whether these findings hold under **naturalistic recurrence**, other datasets, architectures and genuinely measured compute budgets. The results do not identify a biological mechanism or a generally optimal replay policy. Can a local selection audit predict later protection under the learner's actual joint update? Do role effects survive interventions that lack a designed-in alignment? The [delayed-memory proposal](docs/next_experiment.md) and [replay mechanism map](research/replay-mechanism-map.md) describe prospective work, not completed evidence.

[Design](docs/design.md), [claim matrix](docs/claim_matrix.md), [source](src/functional_circuits), [tests](tests) and [provenance](docs/provenance.md) provide the audit trail. Original code, generated data, trained weights and original documentation are [MIT licensed](LICENSE). External datasets, dependencies and papers retain their own licenses and attribution requirements.

## Post-hoc JOINT replay diagnostic

[Same-state update-composition diagnostic and complete signed outputs](studies/replay-joint-update-diagnostic/README.md) find that the local MIR-versus-high-loss contrast changes sign between deployed JOINT and sequential updates. Important adverse context: MIR still raises old-panel loss by +0.71271 nats from the original state and raises new-panel loss by +0.37056 versus fresh-only SGD. These are conditional one-step findings from four reused historical seed clusters, not fresh confirmation, future-retention evidence or a general plasticity advantage.

[Portable rerun inputs and independently checked saved numerical reproduction](studies/replay-joint-update-reproduction/README.md) are available as an additive package. This reproduces the same historical diagnostic, not fresh confirmation, and leaves its adverse findings unchanged.

## Historical 2×2 replay update decomposition

[Complete signed component diagnostic and saved-evidence verification](studies/replay-update-decomposition/README.md) separate combined weighting from replay-gradient location on four additional historical seed clusters. JOINT MIR still increases old loss by +0.905519 nats; its −0.025069 MIR-minus-high-loss contrast does not clear the frozen −0.005 interval margin. The +0.517077 interaction is a local arithmetic finding, not robust retention efficacy or a biological mapping. Both fresh-control scales and all adverse new-class results are retained.

## Learn the ideas and explore the evidence

- [Start with a tiny working example](docs/START_HERE.md): forgetting, replay buffers and current error versus predicted interference.
- [Research overview](docs/RESEARCH_OVERVIEW.md): hypotheses, controls, related work and open questions.
- [Evidence map](docs/EVIDENCE_MAP.md): connect historical replay and role findings to later JOINT/decomposition diagnostics, with saved-check versus inference/training routes.
- [Fashion-MNIST report](studies/replay-selection-fashion-mnist/results/evaluation-report-public.md) and [protocol/evidence guide](studies/replay-selection-fashion-mnist/README.md).

## Watch the narrated explanation

[![Narrated concept illustration](docs/animations/narrated/poster.png)](docs/animations/narrated/concept.mp4)

[53.7-second narrated MP4](docs/animations/narrated/concept.mp4) · [Complete transcript](docs/animations/narrated/transcript.txt) · [Captions](docs/animations/narrated/captions.vtt) · [Editable source and reproduction](docs/animations/narrated/RENDERING.md)

Speech-led Manim animation with a synthetic stock voice. Offline linear toy · not benchmark evidence. This is an explanatory upgrade, not a new experimental result. Captions have approximate sentence timing. The [static-first viewer](docs/animations/narrated/index.html), poster and complete transcript provide a no-motion alternative. The original silent version and its historical QA remain unchanged below.

## Watch the concept in 24 seconds

[![A toy loss plot shows that the hard-now example has the greatest current loss, while a different threatened-next example has the greatest signed loss increase after a virtual update.](docs/animations/poster.png)](docs/animations/concept.mp4)

[24-second Manim MP4](docs/animations/concept.mp4) · [Optional GIF preview](docs/animations/concept.gif) · [Text transcript](docs/animations/transcript.txt) · [Captions](docs/animations/captions.vtt)

Rendered with Manim Community 0.22.0. Offline linear toy · not benchmark evidence. No audio. The poster and transcript provide a no-motion alternative. The downloaded [HTML viewer](docs/animations/index.html) is static by default and never autoplays.

