# Learning and Remembering

What helps a small neural network learn something new without losing what it already knows? And when we give a network separate computational roles, do activation interventions become easier to interpret?

This repository studies those questions through small, reproducible experiments. The replay studies separate **which examples stay in memory** from **which stored examples are rehearsed**. A separate synthetic study examines supplied boundaries between current evidence, retrieved evidence and context control.

## Start here

- [Learn the ideas with a tiny working example](docs/START_HERE.md). The guide explains forgetting, replay buffers and the difference between current error and predicted interference.
- [Read the research overview](docs/RESEARCH_OVERVIEW.md) for hypotheses, controls, related work and open questions.
- [Read the latest replay-selection report](studies/replay-selection-fashion-mnist/results/evaluation-report-public.md), then inspect its [protocol and evidence guide](studies/replay-selection-fashion-mnist/README.md).

## Watch the concept in 24 seconds

[![Illustrative linear learner: B-only learning changes A prediction from one to about one-half; replay retains A. High current loss and greatest predicted loss increase select different stored examples.](docs/animations/poster.png)](docs/animations/concept.mp4)

[MP4](docs/animations/concept.mp4) · [Optional GIF preview](docs/animations/concept.gif) · [Text transcript](docs/animations/transcript.txt) · [Captions](docs/animations/captions.vtt)

Offline linear toy · not benchmark evidence. No audio. The poster and transcript provide a no-motion alternative. The downloaded [HTML viewer](docs/animations/index.html) is static by default and never autoplays.

## What the experiments found

**Keeping historical coverage helped in the digit stream.** Across ten paired seeds, final macro accuracy was 88.44% for reservoir replay, 21.16% for equally sized FIFO memory and 19.70% for current-only repetition. Immediate acquisition was strong in every arm. The [digit report](docs/replay_results.md) links the selection lock, all development/evaluation runs and reproduction instructions.

**More elaborate selection did not clear the Fashion-MNIST study's support criterion.** Across 20 paired seeds and 100 runs, maximally interfered retrieval (MIR) exceeded high-loss replay by 0.6265 percentage points; the paired 95% bootstrap interval was −0.3595 to +1.5330 points. The primary result is inconclusive. Uniform replay had a higher descriptive mean, and the prespecified protection audit was adverse to MIR. [All results and compute controls](studies/replay-selection-fashion-mnist/results/evaluation-report-public.md) are retained.

**Engineered role boundaries affected both prediction and interventions.** The [six-architecture pilot](docs/results.md), [equal-parameter role controls](docs/role_controls.md) and [validation-matched competence study](docs/competence_controls.md) trace this dependence. Matching was incomplete, and one intervention diagnostic follows algebraically from the supplied masks. A competence-independent interpretability advantage remains unestablished.

These are task-specific computational findings. The [biological evidence map](research/replay-mechanism-map.md) explains the motivation and where the analogy ends.

## Reproduce

With Python and NumPy already available, from the repository root:

```sh
bash scripts/run_pilot.sh --output results/local-reproduction
PYTHONPATH=src python scripts/verify_run.py results/pilot \
  --allow-missing-checkpoints --compare results/local-reproduction
PYTHONPATH=src python -m unittest discover -s tests -v
```

The recorded runtime was Python 3.12.14 with NumPy 2.3.5; `requirements.txt` pins NumPy. These commands perform no downloads. Use a fresh output directory. The text-only pilot publication omits binary checkpoints; [availability and regeneration](docs/checkpoints.md) explain that boundary. The [verifier correction](docs/audit_coverage_fix.md) documents its exact-coverage checks. Each real-data study has separate setup and evidence-verification instructions.

## Next questions

Would replay help when classes recur naturally? Can a selection audit predict protection under the learner's actual joint update? Do role effects survive interventions that lack a designed-in alignment? The [delayed-memory proposal](docs/next_experiment.md) and [replay mechanism map](research/replay-mechanism-map.md) describe prospective work; their proposed experiments remain unrun.

[Design](docs/design.md), [claim matrix](docs/claim_matrix.md), [source](src/functional_circuits), [tests](tests) and [provenance](docs/provenance.md) provide the audit trail. Original code, generated data, trained weights and original documentation are [MIT licensed](LICENSE). External datasets, dependencies and papers retain their own licenses and attribution requirements.
