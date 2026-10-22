# Neurofunctional Lab

A small, reproducible experiment asks whether separating **current evidence**, **retrieved evidence**, and **context control** makes neural computations easier to verify through activation interventions.

The included Functional Circuits pilot trains six NumPy architectures on the same generated task, then measures prediction, conflicting-evidence robustness, and counterfactual behavior across five seeds. It runs on CPU with no model or dataset downloads.

**Research status:** completed synthetic, handwritten-digit replay and Fashion-MNIST replay-selection studies. The separate [delayed-memory proposal](docs/next_experiment.md) remains unrun. These experiments do not establish biological fidelity or a new algorithm.

## Start with the completed studies

- [Fashion-MNIST replay selection](studies/replay-selection-fashion-mnist/README.md): 20 paired evaluation seeds and 100 runs. MIR minus high-loss final macro accuracy was +0.6265 percentage points (paired 95% bootstrap interval −0.3595 to +1.5330); it did not meet the prespecified support criterion. This is an inconclusive primary result. Read the [final report](studies/replay-selection-fashion-mnist/results/evaluation-report-public.md); the study README covers evidence reconstruction, verification and separate new-run setup.
- [Handwritten-digit replay](docs/replay_results.md): reservoir replay improved retention over FIFO under the fixed class-blocked stream. The report includes the locked selection, all development/evaluation runs and reproduction instructions.
- [Functional Circuits pilot](docs/results.md), [matched-capacity role controls](docs/role_controls.md) and [validation-matched competence controls](docs/competence_controls.md): synthetic tests of engineered role boundaries. The pilot reproduction commands are below.
- [Replay mechanism evidence map](research/replay-mechanism-map.md): a retrospective primary-source synthesis connecting the Fashion-MNIST result to biological replay evidence. Its next prediction is proposed, not executed.

## Measured pilot

Mean ± sample SD across seeds 11, 23, 37, 53 and 71. “Balanced” means the current and retrieved categories agree in half the examples. IIA is memory activation-interchange intervention accuracy against the known task counterfactual.

| Architecture | Parameters | Accuracy on 50%-agreement split | Memory IIA |
|---|---:|---:|---:|
| Functional roles | 132 | 97.66 ± 0.23% | 97.45 ± 0.44% |
| Monolithic | 136 | 91.37 ± 0.93% | 79.60 ± 2.16% |
| Random partition | 132 | 77.41 ± 3.86% | 68.17 ± 8.38% |
| Dense modular | 228 | 95.70 ± 0.60% | 95.81 ± 0.94% |
| No gate | 130 | 74.84 ± 0.89% | 75.18 ± 1.27% |
| Oracle gate | 130 | 98.28 ± 0.32% | 98.30 ± 0.39% |

Exact measurements, all distributions, paired differences, learning curves and checkpoint hashes: [results/pilot](results/pilot). Interpretation: [docs/results.md](docs/results.md).

## Matched-capacity role control

A fresh five-seed [role-mask control](docs/role_controls.md) keeps all conditions at 132 parameters and the same intervention/search budget. Mixing the supplied role boundaries lowers both prediction accuracy and intervention accuracy; merely exchanging branch names leaves predictions unchanged. This quantifies dependence on the engineered task alignment, not biological specificity. Full [results](results/role_controls) and limitations are retained separately from the pilot.

## Validation-matched competence study

A fresh [validation-only matching study](docs/competence_controls.md) finds eligible prediction-matched pairs in 10 of 15 comparisons, with only 6 retaining the matching tolerances on untouched test data. IIA differences remain in reported pairs, but incomplete matching and an explicit architectural identity prevent an accuracy-independent interpretability claim. All candidates, unmatched comparisons and held-out results are retained.

## Pilot evidence limits

- The task's true roles are supplied as input masks. The intervention target uses those same designed-in roles. This tests a helpful inductive bias, not autonomous discovery of interpretable structure.
- The functional model is also more accurate. The IIA difference cannot be cleanly separated from predictive competence without accuracy-matched comparisons.
- A generic dense modular model performs well, weakening any claim that the benefit specifically requires biological inspiration.
- Every model receives the same perfectly retrieved evidence. Learning storage, retrieval, delay robustness or consolidation is outside this pilot.
- Five seeds, one task family, unequal intervention granularity, and a greedy monolithic mapping search limit generalization. The architecture has no demonstrated anatomical fidelity or new-method claim.

## Reproduce

Tested with Python 3.12.14 and NumPy 2.3.5. NumPy is the sole runtime dependency; `requirements.txt` pins the tested version. The commands below do not install or download anything.

```sh
# From the repository root, with NumPy available:
bash scripts/run_pilot.sh --output results/local-reproduction

# Audit the committed run and compare exact numerical outputs/checkpoints:
PYTHONPATH=src python scripts/verify_run.py results/pilot \
  --allow-missing-checkpoints --compare results/local-reproduction

# Tests only:
PYTHONPATH=src python -m unittest discover -s tests -v
```

The full run took about 15 seconds in the measured CPU environment; runtime varies. Existing run directories containing a manifest are protected against overwrite. The text-only publication omits binary checkpoints; [regeneration and availability](docs/checkpoints.md) are documented. A separately rerun seed-11 experiment is retained in [results/repeat_seed11](results/repeat_seed11). Numerical equality excludes timings and platform metadata. The term “balanced” names the 50%-category-agreement test distribution; all reported prediction accuracies are ordinary micro accuracies, not macro-average recall.

The [post-run verifier correction](docs/audit_coverage_fix.md) closes an incomplete-comparison bug. Comparisons now enforce exact row/checkpoint coverage; partial seed comparisons require explicit scope. The 53-test suite passes, and saved experiment results remain unchanged.

## Neuroscience-grounded replay study

The [neural-motif evidence matrix](docs/neural_motifs.md) separates biological findings, computational analogies and existing ML results. A [frozen, now-executed protocol](docs/protocols/replay_digits_v1.md) tested historical replay on real handwritten digits with the official writer-held-out test split.

Across ten paired seeds, final macro accuracy was **88.44% reservoir**, **21.16% FIFO**, and **19.70% current-only**. The primary reservoir–FIFO gain was **67.28 percentage points**, with a descriptive paired-seed bootstrap interval of **[65.27, 69.04] points**, conditional on this dataset. Strong immediate acquisition in every arm distinguishes retention loss from failure to learn the current block. This is a known replay mechanism under an imposed class shift, not biological validation or a novelty claim.

[Full results, locked selection, reproducibility and limitations](docs/replay_results.md) preserve all 18 development comparisons and all 30 evaluation runs. The [data/core checkpoint](docs/replay_implementation.md) and [staged driver checkpoint](docs/replay_execution.md) remain as historical pre-run records.

## Explore

- [Design](docs/design.md) and [plan](docs/plan.md), written before implementation.
- [Research background](docs/research.md) and [citation/claim matrix](docs/claim_matrix.md).
- [Results and failures](docs/results.md).
- [Unrun delayed-memory proposal and resource gates](docs/next_experiment.md), separate from the completed replay-selection study above.
- [Data/model provenance](docs/provenance.md).
- [Implementation](src/functional_circuits), [tests](tests), [run manifest](results/pilot/manifest.json).

Original code, generated data, trained weights and original documentation: [MIT](LICENSE). Referenced papers and external dependencies retain their own licenses.
