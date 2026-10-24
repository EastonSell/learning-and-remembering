# Evidence archive availability

The public Git package contains the full scalar metrics, frozen source/protocol and audit receipts. Binary checkpoints are not hidden in text blobs and are not included in Git.

The owner has separately retained these files in Library:

| Archive | Bytes | SHA256 |
| --- | ---: | --- |
| pendigits-pilot-full-checkpoint-evidence.zip | 84,837,498 | `4fcb793d591c684ca656bf5e21006eb94e5368c05e50e3fa8cbf1a3a1b069951` |
| pendigits-pilot-report-and-metrics.zip | 47,110 | `4b5bc4a41b194ef006925a38a1eb8334f4dd63ce069e5f07fb5341c631856690` |

The full archive contains all 300 checkpoint/prediction pairs, saved test inputs, split indices, the original frozen experiment, protocol and review, all results/audits, raw parent-level `pendigits.tra`, and a per-file checksum manifest. Its ZIP integrity and complete 300-checkpoint inventory were checked before retention. The compact report archive contains text evidence, not weights or data arrays.

Library retention does not create public access. No public archive URL is claimed here. A reader with only this repository can recompute the saved scalar results and perform a new labeled replication using the [official source-data instructions](README.md#c-intentionally-retrain-a-labeled-replication); they cannot independently infer from the original weights until they obtain the full archive. Any future redistribution must preserve the UCI CC BY 4.0 attribution and the repository MIT license for original code/weights. See README.md for the exact inference and historical timestamp limitations.
