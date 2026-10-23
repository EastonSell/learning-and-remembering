# Read the evidence: which question did each experiment answer?

This is a companion to [Start here](START_HERE.md), not a new experiment. It connects the original replay/role studies to the later one-step diagnostics and separates reading saved results from running new computation. Source snapshot: [commit 1bff80d](https://github.com/EastonSell/learning-and-remembering/tree/1bff80d84f80209c926f779762f47a6a16af3cf3), 23 October 2026.

## 1. Choose the question, then the result

| Question | Experiment and measured endpoint | What the published result supports |
|---|---|---|
| Does memory need historical coverage? | Optdigits: final macro accuracy after a class-blocked stream; ten paired seeds. | Reservoir beat equally sized FIFO by 67.28 percentage points, interval [65.27, 69.04]. This supports historical coverage in this setup, not a universally optimal buffer. [Digit results][digits] |
| With coverage matched, does MIR select better replay? | Fashion-MNIST: final macro accuracy; twenty paired seeds and five arms. | MIR minus high-loss was +0.6265 points, interval [−0.3595, +1.5330]. Neither the ≥1-point mean requirement nor the positive lower-bound requirement was met. The primary result is inconclusive, not equivalence. [Locked report][fashion-results] |
| Did the selected examples protect old performance in the original audit? | Fashion-MNIST: selected-minus-within-class-shuffled old-class loss in separate sequential audit branches. | MIR's +0.078770 contrast was adverse. This is not the same comparator or update as either the primary accuracy outcome or the later diagnostic. [Audit definition and result][fashion-results] |
| Does update composition change the local selector comparison? | JOINT diagnostic: one-step old/new validation-panel loss at sixteen uniform-origin checkpoints, seeds 2001–2004. | MIR minus high-loss was −0.09549 for JOINT and +0.16334 for SEQUENTIAL. The local contrast changes sign, but old loss still rises by +0.71271 under JOINT MIR relative to the original state. [Signed diagnostic results][joint-results] |
| Can the exact earlier computation be reproduced? | Portable reproduction of the same sixteen states and fixed inputs. | Published saved checks find sixteen byte-identical state files and exact scientific-summary equality after the one disclosed resource-field exclusion. This is numerical reproduction, not sixteen new observations or independent live-execution attestation. [Reproduction boundary][portable] |
| Which specified update factors contribute? | 2×2 weighting-by-gradient-location diagnostic: sixteen additional historical states, seeds 2005–2008. | The old-panel MIR-minus-high-loss interaction is +0.517077; deployed JOINT still raises old loss by +0.905519 on average, with increases in all sixteen states. This is a local arithmetic decomposition, not robust retention efficacy. [Decomposition results and limits][factorial] |
| Do supplied functional boundaries aid intervention behavior? | Separate synthetic pilot, equal-parameter role controls and validation-matched competence study. | Effects depend on engineered alignment and predictive competence; 10/15 comparisons matched on validation and only 6/10 retained test tolerances. The jointly correct-subset 100% aligned IIA is algebraically privileged. [Pilot][pilot], [role controls][roles], [competence controls][competence] |

The synthetic role models do not learn memory retrieval, persistence or replay. Do not combine their intervention scores with the replay studies as if they measured the same mechanism. [Pilot interpretation][pilot]

## 2. Why the later findings do not overturn the earlier negative evidence

The apparent disagreement disappears when the comparison is written out:

1. **The locked final-accuracy question** compares completed MIR and high-loss training runs. Their weights diverge during training. Its inconclusive result remains unchanged.
2. **The original protection audit** compares a selected replay set with a within-class shuffled set after sequential fresh/replay steps. Its adverse result remains unchanged.
3. **The later JOINT diagnostic** starts all branches at the same historical uniform-origin checkpoint, fixes selection, and compares MIR with high-loss (or uniform). It changes the update recipe. It is not a rerun of the old selected-versus-shuffled estimand.
4. **The 2×2 study** uses a different historical seed cohort to split the recipe into specified factors. It does not upgrade previously used data and validation into unseen confirmation.

The sources explicitly preserve these distinctions: [original audit][fashion-results], [JOINT findings][joint-results], [factorial methods][factorial-methods]. A favorable relative selector difference can coexist with worse loss than doing nothing, and with a cost to new learning.

### Decode the sign before comparing numbers

- Accuracy differences are in **percentage points**: higher is better. A +0.6265-point accuracy difference is not a +0.6265% relative gain.
- The local diagnostics use **macro cross-entropy in nats/example**: lower is better. MIR minus high-loss below zero favors MIR on that panel.
- **Absolute damage** means post-update loss minus loss at the original checkpoint. Positive means the update made that panel worse, even if another selector was more harmful.
- **Old panel** and **new panel** refer to class-specific validation panels at a transition, not final test accuracy or future retention.
- “Fresh-only” is not one universal baseline in the factorial study: it reports both a unit-weight fresh step and the smaller deployed-weight fresh step. Against these, JOINT MIR's mean new-panel differences are +0.335974 and −0.057774 respectively. Do not exchange those controls. [Factorial methods][factorial-methods], [results][factorial]

## 3. Read the 2×2 as a conditional comparison

Each cell starts from the same saved parameters and selected IDs. The switches change (a) the fresh/replay weights and (b) where the replay gradient is evaluated. For old-panel MIR minus high-loss:

| Weights | Replay gradient at original state | Replay gradient after the weight-matched fresh step |
|---|---:|---:|
| Deployed: fresh 16/26, replay 10/26 | D00 = −0.025069 | D01 = −0.070031 |
| Unit: fresh 1, replay 1 | D10 = −0.131991 | D11 = +0.340123 |

The interaction is D11 − D10 − D01 + D00, reported as +0.517077 using full-precision values. It describes how the effect of one switch depends on the other. It is not itself a MIR benefit. Displayed cell means are rounded; use the full-precision CSV/JSON for exact arithmetic. [Methods][factorial-methods], [signed exports][factorial]

Two important limits survive the extra controls:

- The weighting switch changes both the fresh/replay ratio and total gradient scale. It does not isolate those two ingredients.
- JOINT's MIR-minus-high-loss interval [−0.051397, −0.004300] lies below zero but does not clear the frozen −0.005 engineering margin. A negative interval, a practically resolved margin and freedom from absolute damage are three different claims. MIR loses to high-loss in 7/16 states here. [Factorial results][factorial]

## 4. Count independent units, not output files

- Optdigits uncertainty resamples **ten paired seed differences** on one fixed dataset/split.
- The primary Fashion-MNIST comparison resamples **twenty paired seeds**. Its 100 runs, 500 checkpoints and 8,000 audit opportunities are not that many independent domains or subjects.
- Each one-step diagnostic has **four seed/stream clusters**, with four dependent transition states per cluster. Bootstrap resampling uses seed means, not sixteen independent states.
- Four uniform draws, tie branches, 350 reported summaries and exact reruns do not increase the number of independent research samples.

These intervals characterize the stated design and reused data, not uncertainty across new domains, people, architectures or learning rates. [Digits][digits], [Fashion-MNIST][fashion-results], [JOINT][joint-results], [factorial methods][factorial-methods]

## 5. Choose a reproduction level before running a command

Commands below are rooted at a full checkout unless a study-directory change is shown. They are different tasks, not interchangeable definitions of “reproduce.” The recorded numerical environment is Linux, Python 3.12.14 and NumPy 2.3.5; check each study's own instructions. Scripts do not install dependencies for you.

### A. Learn the idea, with no external data

```sh
python -B examples/learning/replay_demo.py
python -B -m unittest discover -s examples/learning -v
```

This standard-library-only toy is an illustration, not a benchmark result. [Learner guide][start]

### B. Inspect saved evidence without training a model

```sh
# Synthetic pilot: verify published hashes, acknowledging omitted checkpoints.
PYTHONPATH=src python -B scripts/verify_run.py results/pilot --allow-missing-checkpoints

# Earlier JOINT diagnostic: retained-file checks plus saved-scalar recount.
python -B studies/replay-joint-update-diagnostic/verify_release.py --recount

# Portable reproduction: stdlib-only public inventory and exact saved-result comparison.
python -B studies/replay-joint-update-reproduction/verify_release.py

# Later 2×2 diagnostic: saved JSON/CSV arithmetic and four-cluster bootstrap only.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -B studies/replay-update-decomposition/verify_saved.py
```

These checks have different scopes. The pilot's missing-checkpoint option does not make omitted reference weights present. The portable verifier hashes binary inputs without decoding their numeric payloads; it reports that execution is not independently attested. The factorial verifier does not import model/reference code or regenerate losses/gradients. A passing inventory is integrity relative to its catalog, not proof of every omitted historical record. [Pilot verifier][pilot-verifier], [JOINT release][joint], [portable release][portable], [factorial release][factorial]

For Fashion-MNIST, first reconstruct the published transport archive, then check retained predictions and their derived metrics:

```sh
cd studies/replay-selection-fashion-mnist
python -B unpack_evidence.py
python -B verify_public.py
```

Run the unpacker once in a fresh checkout; it refuses to replace already-extracted evidence. If extraction is already complete, proceed to verification instead. Reconstruction writes the packaged saved evidence locally; it does not recreate missing originals or train a learner. The base verifier reads saved prediction arrays and test labels, verifies data identities, and redoes summaries/audits. It does not recompute model predictions. [Public export instructions][fashion], [verifier source][fashion-verifier]

### C. Recompute predictions from saved weights, without training

From the same reconstructed Fashion-MNIST study directory:

```sh
python -B verify_public.py --reproduce-predictions
```

This adds model inference on retained test assets and exact comparison with all 500 saved prediction arrays. It is more than a scalar recount and less than retraining all 100 cases. [Verifier source][fashion-verifier]

### D. Intentionally execute a new scientific computation

- The root README's `scripts/run_pilot.sh` command **trains the synthetic pilot**. It is not a saved-only check or the command for every study.
- A new digit or Fashion-MNIST training run needs its study-specific data/provenance workflow and fresh output locations. The Fashion-MNIST preparation attestation must be truthful; do not tune on published test outcomes and then claim they were unseen. [Digit execution history][digits-execution], [Fashion-MNIST new-run guide][fashion]
- The portable JOINT package includes a separate TRAIN-only acquisition and exclusive new-run recipe for seeds **2001–2004**. Running it is a fixed historical one-step recomputation, not checkpoint retraining or fresh confirmation. [Portable instructions][portable]
- **The 2×2 cohort, seeds 2005–2008, is not packaged for full scientific rerunning.** Its cohort-specific checkpoints, images, streams, complete source/launch/input chain and original address-bearing provenance are not supplied. Do not substitute the earlier cohort or fabricate missing inputs. Saved-evidence verification and reference-code inspection remain available. [Explicit missing-input boundary][factorial]

Keep personal notes, redirected verification output and fresh runs outside closed study directories. Several study verifiers reject unexpected files, not only changed files. Do not edit historical protocols, freezes or completion markers to make a newer narrative fit.

## 6. Which document is current?

Use this guide and each study's **outer README** for navigation and current distribution scope. Preserve the older report/protocol as evidence of what it said at its own stage. For example, the portable study intentionally keeps a frozen inner README/VALIDATION that predates its rerun; the outer README explains the later result. The Fashion-MNIST public README qualifies the exact older report's descriptions of privately retained full bundles. [Portable][portable], [Fashion-MNIST][fashion]

The root `PUBLICATION_MANIFEST.json` identifies the earlier `neurofunctional-lab-text-publication-v7` package and destination name; it is not a current whole-repository inventory. Use the commit for this reader snapshot and the relevant study verifier for each closed release. Never silently regenerate historical manifests to describe later documentation. [Historical root manifest][legacy-manifest]

[start]: https://github.com/EastonSell/learning-and-remembering/blob/1bff80d84f80209c926f779762f47a6a16af3cf3/docs/START_HERE.md
[digits]: https://github.com/EastonSell/learning-and-remembering/blob/1bff80d84f80209c926f779762f47a6a16af3cf3/docs/replay_results.md
[digits-execution]: https://github.com/EastonSell/learning-and-remembering/blob/1bff80d84f80209c926f779762f47a6a16af3cf3/docs/replay_execution.md
[fashion]: https://github.com/EastonSell/learning-and-remembering/blob/1bff80d84f80209c926f779762f47a6a16af3cf3/studies/replay-selection-fashion-mnist/README.md
[fashion-results]: https://github.com/EastonSell/learning-and-remembering/blob/1bff80d84f80209c926f779762f47a6a16af3cf3/studies/replay-selection-fashion-mnist/results/evaluation-report-public.md
[fashion-verifier]: https://github.com/EastonSell/learning-and-remembering/blob/1bff80d84f80209c926f779762f47a6a16af3cf3/studies/replay-selection-fashion-mnist/verify_public.py
[joint]: https://github.com/EastonSell/learning-and-remembering/blob/1bff80d84f80209c926f779762f47a6a16af3cf3/studies/replay-joint-update-diagnostic/README.md
[joint-results]: https://github.com/EastonSell/learning-and-remembering/blob/1bff80d84f80209c926f779762f47a6a16af3cf3/studies/replay-joint-update-diagnostic/RESULTS.md
[portable]: https://github.com/EastonSell/learning-and-remembering/blob/1bff80d84f80209c926f779762f47a6a16af3cf3/studies/replay-joint-update-reproduction/README.md
[factorial]: https://github.com/EastonSell/learning-and-remembering/blob/1bff80d84f80209c926f779762f47a6a16af3cf3/studies/replay-update-decomposition/README.md
[factorial-methods]: https://github.com/EastonSell/learning-and-remembering/blob/1bff80d84f80209c926f779762f47a6a16af3cf3/studies/replay-update-decomposition/docs/METHODS.md
[pilot]: https://github.com/EastonSell/learning-and-remembering/blob/1bff80d84f80209c926f779762f47a6a16af3cf3/docs/results.md
[roles]: https://github.com/EastonSell/learning-and-remembering/blob/1bff80d84f80209c926f779762f47a6a16af3cf3/docs/role_controls.md
[competence]: https://github.com/EastonSell/learning-and-remembering/blob/1bff80d84f80209c926f779762f47a6a16af3cf3/docs/competence_controls.md
[pilot-verifier]: https://github.com/EastonSell/learning-and-remembering/blob/1bff80d84f80209c926f779762f47a6a16af3cf3/scripts/verify_run.py
[legacy-manifest]: https://github.com/EastonSell/learning-and-remembering/blob/1bff80d84f80209c926f779762f47a6a16af3cf3/PUBLICATION_MANIFEST.json
