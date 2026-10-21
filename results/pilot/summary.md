# Measured pilot results

5 seeds; 120 epochs; all values below are means +/- sample SD.

| Model | Params | ID accuracy | Accuracy on 50%-agreement split | Conflict accuracy | Balanced memory IIA | Changed-case IIA |
|---|---:|---:|---:|---:|---:|---:|
| functional | 132 | 0.982 ± 0.003 | 0.977 ± 0.002 | 0.965 ± 0.005 | 0.975 ± 0.004 | 0.975 ± 0.008 |
| monolithic | 136 | 0.945 ± 0.007 | 0.914 ± 0.009 | 0.862 ± 0.015 | 0.796 ± 0.022 | 0.691 ± 0.219 |
| random_partition | 132 | 0.885 ± 0.006 | 0.774 ± 0.039 | 0.582 ± 0.077 | 0.682 ± 0.084 | 0.661 ± 0.200 |
| dense_modular | 228 | 0.970 ± 0.005 | 0.957 ± 0.006 | 0.937 ± 0.012 | 0.958 ± 0.009 | 0.953 ± 0.012 |
| no_gate | 130 | 0.887 ± 0.010 | 0.748 ± 0.009 | 0.502 ± 0.006 | 0.752 ± 0.013 | 0.747 ± 0.025 |
| oracle_gate | 130 | 0.981 ± 0.004 | 0.983 ± 0.003 | 0.982 ± 0.004 | 0.983 ± 0.004 | 0.980 ± 0.008 |

IIA is activation interchange intervention accuracy. Changed-case IIA excludes counterfactuals whose correct answer was unchanged.
Interpretation requires docs/results.md: architectural role assignment, different patch granularities, and perfect shared retrieval limit conclusions.
