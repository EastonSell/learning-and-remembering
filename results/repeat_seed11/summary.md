# Measured pilot results

1 seeds; 120 epochs; all values below are means +/- sample SD.

| Model | Params | ID accuracy | Accuracy on 50%-agreement split | Conflict accuracy | Balanced memory IIA | Changed-case IIA |
|---|---:|---:|---:|---:|---:|---:|
| functional | 132 | 0.981 ± 0.000 | 0.977 ± 0.000 | 0.972 ± 0.000 | 0.975 ± 0.000 | 0.973 ± 0.000 |
| monolithic | 136 | 0.956 ± 0.000 | 0.923 ± 0.000 | 0.881 ± 0.000 | 0.824 ± 0.000 | 0.835 ± 0.000 |
| random_partition | 132 | 0.882 ± 0.000 | 0.741 ± 0.000 | 0.501 ± 0.000 | 0.589 ± 0.000 | 0.715 ± 0.000 |
| dense_modular | 228 | 0.974 ± 0.000 | 0.965 ± 0.000 | 0.952 ± 0.000 | 0.957 ± 0.000 | 0.954 ± 0.000 |
| no_gate | 130 | 0.896 ± 0.000 | 0.755 ± 0.000 | 0.507 ± 0.000 | 0.736 ± 0.000 | 0.713 ± 0.000 |
| oracle_gate | 130 | 0.979 ± 0.000 | 0.985 ± 0.000 | 0.985 ± 0.000 | 0.982 ± 0.000 | 0.981 ± 0.000 |

IIA is activation interchange intervention accuracy. Changed-case IIA excludes counterfactuals whose correct answer was unchanged.
Interpretation requires docs/results.md: architectural role assignment, different patch granularities, and perfect shared retrieval limit conclusions.
