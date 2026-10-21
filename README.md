# Neurofunctional Lab

A small, reproducible experiment asks whether separating **current evidence**, **retrieved evidence**, and **context control** makes neural computations easier to verify through activation interventions.

The included Functional Circuits pilot trains six NumPy architectures on the same generated task, then measures prediction, conflicting-evidence robustness, and counterfactual behavior across five seeds. It runs on CPU with no model or dataset downloads.

**Research status:** tested synthetic foundation. The next real-image protocol is specified but has not been run. Biological literature motivates functional questions; the measured result concerns an engineered causal prior.

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

## Evidence limits

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

## Explore

- [Design](docs/design.md) and [plan](docs/plan.md), written before implementation.
- [Research background](docs/research.md) and [citation/claim matrix](docs/claim_matrix.md).
- [Results and failures](docs/results.md).
- [Next real-data/model protocol and resource gates](docs/next_experiment.md).
- [Data/model provenance](docs/provenance.md).
- [Implementation](src/functional_circuits), [tests](tests), [run manifest](results/pilot/manifest.json).

Original code, generated data, trained weights and original documentation: [MIT](LICENSE). Referenced papers and external dependencies retain their own licenses.
